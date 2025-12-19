# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from .aggregator import Aggregator
from .heads.camera_head import CameraHead
from .heads.dpt_head import DPTHead
from .heads.gs_head import DPTGSHead
from .heads.track_head import TrackHead
from .utils.pose_enc import pose_encoding_to_extri_intri

# Import storm components from local modules
from .decoder_layers import ModulatedLinearLayer
from .embedders import PluckerEmbedder

from contextlib import contextmanager

@contextmanager
def tf32_off():
    original = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original

def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            module.requires_grad = False


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, use_sky_token=True, use_scale_token=True, memory_efficient=True, sh_degree=0, use_gs_head=True, use_gs_head_velocity=False, use_gs_head_segment=False, use_gt_camera=False, velocity_head_small_init=False):
        super().__init__()
        self.sh_degree = sh_degree
        self.use_gs_head = use_gs_head
        self.use_gs_head_velocity = use_gs_head_velocity
        self.use_gs_head_segment = use_gs_head_segment
        self.use_gt_camera = use_gt_camera

        sh_dim = 3 * ((sh_degree + 1) ** 2)
        gaussian_output_dim = 3 + 3 + sh_dim + 4 + 1 + 1

        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            use_sky_token=use_sky_token,
            use_scale_token=use_scale_token,
            memory_efficient=memory_efficient,
            output_layers=[4, 11, 17, 23] if memory_efficient else None
        )
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")

        GaussianHeadClass = DPTGSHead if use_gs_head else DPTHead
        self.gaussian_head = GaussianHeadClass(dim_in=2 * embed_dim, output_dim=gaussian_output_dim, activation="linear", conf_activation="expp1")

        VelocityHeadClass = DPTGSHead if use_gs_head_velocity else DPTHead
        self.velocity_head = VelocityHeadClass(dim_in=2 * embed_dim, output_dim=4, activation="linear", conf_activation="expp1")

        if velocity_head_small_init:
            self._apply_small_init_to_velocity_head()

        SegmentHeadClass = DPTGSHead if use_gs_head_segment else DPTHead
        self.segment_head = SegmentHeadClass(dim_in=2 * embed_dim, output_dim=5, activation="linear", conf_activation="expp1")

        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)

        self.use_sky_token = use_sky_token
        if self.use_sky_token:
            self.sky_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.plucker_embedder = PluckerEmbedder(img_size=img_size)
            self.sky_head = ModulatedLinearLayer(
                3,
                hidden_channels=512,
                condition_channels=embed_dim*2,
                out_channels=3,
            )

        self.use_scale_token = use_scale_token
        if self.use_scale_token:
            self.scale_token = nn.Parameter(torch.zeros(embed_dim))
            torch.nn.init.trunc_normal_(self.scale_token, std=0.02)
            self.scale_head = nn.Sequential(
                nn.Linear(embed_dim * 2, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

    def _apply_small_init_to_velocity_head(self):
        """
        Apply small initialization to the last layer of velocity_head.
        Sets bias to 0 and weights to small values (std=0.001).
        """
        if hasattr(self.velocity_head, 'scratch') and hasattr(self.velocity_head.scratch, 'output_conv2'):
            output_conv2 = self.velocity_head.scratch.output_conv2
            if isinstance(output_conv2, nn.Sequential):
                last_layer = output_conv2[-1]
                if isinstance(last_layer, nn.Conv2d):
                    nn.init.normal_(last_layer.weight, mean=0.0, std=0.001)
                    if last_layer.bias is not None:
                        nn.init.zeros_(last_layer.bias)
                    print(f"[VGGT] Applied small initialization to velocity_head last layer: weight std=0.001, bias=0")

    def set_freeze(self, freeze):
        to_be_frozen = {
            "none": [],
            "noap": [
                self.aggregator,
                self.camera_head,
                self.point_head,
                self.velocity_head,
                self.scale_head if self.use_scale_token else None,
                self.scale_token if self.use_scale_token else None,
                self.track_head if hasattr(self, "track_head") else None,
            ],
        }
        freeze_all_params(to_be_frozen[freeze])

    def gradient_checkpointing_enable(self, enable=True):
        """Enable or disable gradient checkpointing for all relevant submodules."""
        def set_gradient_checkpointing(module):
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable(enable=enable)
            for child in module.children():
                set_gradient_checkpointing(child)

        modules = [self.aggregator, self.camera_head, self.point_head, self.depth_head]
        if hasattr(self, "track_head"):
            modules.append(self.track_head)
        if hasattr(self, "gaussian_head"):
            modules.append(self.gaussian_head)
        if hasattr(self, "sky_head"):
            modules.append(self.sky_head)

        for module in modules:
            set_gradient_checkpointing(module)

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None,
                gt_extrinsics: torch.Tensor = None, gt_intrinsics: torch.Tensor = None,
                frame_sample_ratio: float = 0.25):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None
            gt_extrinsics (torch.Tensor, optional): Ground truth extrinsics with shape [B, S, 4, 4].
                Default: None
            gt_intrinsics (torch.Tensor, optional): Ground truth intrinsics with shape [B, S, 3, 3].
                Default: None
            frame_sample_ratio (float, optional): Ratio of frames to sample for sky rendering.
                Default: 0.25

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - xyz_camera (torch.Tensor): 3D coordinates in camera frame with shape [B, S, H*W, 3]
                - images (torch.Tensor): Original input images, preserved for visualization
                - scale (torch.Tensor): Predicted scene scale with shape [B] (if use_scale_token=True)

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        kwargs = {}
        if self.use_sky_token:
            kwargs['sky_token'] = self.sky_token
        if self.use_scale_token:
            kwargs['scale_token'] = self.scale_token.unsqueeze(0).unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images, **kwargs)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

                extrinsics_to_use = None
                intrinsics_to_use = None

                if self.use_gt_camera and gt_extrinsics is not None and gt_intrinsics is not None:
                    extrinsics_to_use = gt_extrinsics
                    intrinsics_to_use = gt_intrinsics
                    if extrinsics_to_use.shape[-2] == 3:
                        B_cam, S_cam = extrinsics_to_use.shape[:2]
                        homo_row = torch.tensor([0, 0, 0, 1], device=extrinsics_to_use.device, dtype=extrinsics_to_use.dtype)
                        homo_row = homo_row.view(1, 1, 1, 4).expand(B_cam, S_cam, 1, 4)
                        extrinsics_to_use = torch.cat([extrinsics_to_use, homo_row], dim=-2)
                elif "pose_enc" in predictions:
                    from models.utils.pose_enc import pose_encoding_to_extri_intri
                    extrinsics_pred, intrinsics_pred = pose_encoding_to_extri_intri(
                        predictions["pose_enc"], images.shape[-2:]
                    )
                    B_cam, S_cam = extrinsics_pred.shape[:2]
                    homo_row = torch.tensor([0, 0, 0, 1], device=extrinsics_pred.device, dtype=extrinsics_pred.dtype)
                    homo_row = homo_row.view(1, 1, 1, 4).expand(B_cam, S_cam, 1, 4)
                    extrinsics_to_use = torch.cat([extrinsics_pred, homo_row], dim=-2)
                    intrinsics_to_use = intrinsics_pred

                if extrinsics_to_use is not None and intrinsics_to_use is not None:
                    from losses.loss import depth_to_world_points

                    B_d, S_d, H_d, W_d = depth.shape[0], depth.shape[1], depth.shape[2], depth.shape[3]

                    depth_reshaped = depth.reshape(B_d * S_d, H_d, W_d, 1)
                    intrinsics_reshaped = intrinsics_to_use.reshape(B_d * S_d, 3, 3)

                    world_points = depth_to_world_points(depth_reshaped, intrinsics_reshaped)
                    world_points = world_points.reshape(B_d, S_d, H_d * W_d, 3)

                    extrinsic_inv = torch.linalg.inv(extrinsics_to_use)

                    world_points_b0 = world_points[0]
                    extrinsic_inv_b0 = extrinsic_inv[0]

                    xyz_camera = torch.matmul(
                        extrinsic_inv_b0[:, :3, :3],
                        world_points_b0.transpose(-1, -2)
                    ).transpose(-1, -2) + extrinsic_inv_b0[:, :3, 3:4].transpose(-1, -2)

                    predictions["world_points"] = world_points.reshape(B_d, S_d, H_d, W_d, 3)
                    predictions["xyz_camera"] = xyz_camera.unsqueeze(0)
                    predictions["extrinsics"] = extrinsics_to_use
                    predictions["intrinsics"] = intrinsics_to_use

            if self.gaussian_head is not None:
                gaussian_params_raw, gaussian_conf = self.gaussian_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                B, S, H, W, _ = gaussian_params_raw.shape
                gaussian_params_activated = gaussian_params_raw.clone()

                sh_dim = 3 * ((self.sh_degree + 1) ** 2)
                scale_end = 6
                sh_start = 6
                sh_end = 6 + sh_dim
                rotation_start = sh_end
                rotation_end = rotation_start + 4
                opacity_idx = rotation_end

                gaussian_params_activated[..., :3] = predictions["xyz_camera"].reshape(B, S, H, W, 3)

                gaussian_params_activated[..., 3:scale_end] = (0.05 * torch.exp(gaussian_params_raw[..., 3:scale_end])).clamp_max(0.3)

                rotations = gaussian_params_raw[..., rotation_start:rotation_end]
                rotation_norms = torch.norm(rotations, dim=-1, keepdim=True).clamp(min=1e-8)
                gaussian_params_activated[..., rotation_start:rotation_end] = rotations / rotation_norms

                gaussian_params_activated[..., opacity_idx:opacity_idx+1] = gaussian_params_raw[..., opacity_idx:opacity_idx+1].sigmoid()

                predictions["gaussian_params"] = gaussian_params_activated
                predictions["gaussian_params_raw"] = gaussian_params_raw
                predictions["gaussian_conf"] = gaussian_conf
                predictions["sh_degree"] = self.sh_degree

            if self.velocity_head is not None:
                velocity_raw, velocity_conf = self.velocity_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                velocity = torch.sign(velocity_raw) * (torch.exp(torch.abs(velocity_raw)) - 1)
                predictions["velocity"] = velocity
                predictions["velocity_conf"] = velocity_conf

                if extrinsics_to_use is not None and intrinsics_to_use is not None:
                    extrinsic_cam2ref = torch.linalg.inv(extrinsics_to_use)

                    R_cam2ref = extrinsic_cam2ref[:, :, :3, :3]

                    B_v, S_v, H_v, W_v, _ = velocity.shape
                    velocity_reshaped = velocity.reshape(B_v, S_v, H_v * W_v, 3)

                    velocity_global = torch.matmul(R_cam2ref, velocity_reshaped.transpose(-1, -2)).transpose(-1, -2)
                    velocity_global = velocity_global.reshape(B_v, S_v, H_v, W_v, 3)

                    predictions["velocity_global"] = velocity_global

            if self.segment_head is not None:
                segment_logits, segment_conf = self.segment_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["segment_logits"] = segment_logits
                predictions["segment_conf"] = segment_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images

        if self.use_sky_token:
            all_sky_tokens = []
            for layer_idx in sorted(aggregated_tokens_list.keys()):
                layer_sky_token = aggregated_tokens_list[layer_idx][:, :, :1]
                all_sky_tokens.append(layer_sky_token)

            stacked_sky_tokens = torch.stack(all_sky_tokens, dim=0)
            sky_token = stacked_sky_tokens.mean(dim=0)
            predictions["sky_token"] = sky_token

            if gt_extrinsics is not None and gt_intrinsics is not None:
                import random
                B, S = sky_token.shape[:2]
                H, W = images.shape[-2:]

                num_frames_to_render = max(1, int(S * frame_sample_ratio))
                sampled_frame_indices = random.sample(range(S), num_frames_to_render)
                sampled_frame_indices = sorted(sampled_frame_indices)

                sampled_intrinsics = gt_intrinsics[0, sampled_frame_indices]
                sampled_extrinsics_w2c = gt_extrinsics[0, sampled_frame_indices]

                sampled_extrinsics = torch.inverse(sampled_extrinsics_w2c)

                ray_dict = self.plucker_embedder(
                    sampled_intrinsics,
                    sampled_extrinsics,
                    image_size=(H, W)
                )
                ray_dirs = ray_dict["dirs"]

                sampled_sky_token = sky_token[0, sampled_frame_indices, 0]
                sampled_sky_token = sampled_sky_token.mean(dim=0, keepdim=True)
                sky_colors_sampled = self.sky_head(
                    ray_dirs.reshape(num_frames_to_render, H * W, 3),
                    sampled_sky_token
                )
                sky_colors_sampled = sky_colors_sampled.view(num_frames_to_render, H, W, 3).permute(0, 3, 1, 2)

                predictions["sky_colors"] = sky_colors_sampled.unsqueeze(0)
                predictions["sampled_frame_indices"] = sampled_frame_indices

        if self.use_scale_token:
            last_layer_idx = max(aggregated_tokens_list.keys())
            token_idx = 1 if self.use_sky_token else 0
            scale_token_features = aggregated_tokens_list[last_layer_idx][:, :, token_idx:token_idx+1]
            scale_token_pooled = scale_token_features.mean(dim=1).squeeze(1)
            predicted_scale = self.scale_head(scale_token_pooled)
            predictions["scale"] = predicted_scale.squeeze(-1)

        B, S, C, H, W = images.shape
        predictions["batch_dims"] = {
            "B": B,
            "S": S,
            "H": H,
            "W": W
        }

        return predictions

    @torch.no_grad()
    def infer_target_frame_sky_colors(
        self,
        sky_token: torch.Tensor,
        target_frame_intrinsics: torch.Tensor,
        target_frame_extrinsics: torch.Tensor,
        image_size: tuple,
        chunk_size: int = 4
    ):
        """
        Infer sky colors for target frames (frames not used in context).

        This function is specifically for inference when render_target_frames=True.
        It generates sky colors for target frames using the sky_token from context frames.
        Processes frames in chunks to avoid OOM errors.

        Args:
            sky_token (torch.Tensor): Sky token from context frames, shape [B, S_context, 1, embed_dim]
            target_frame_intrinsics (torch.Tensor): Intrinsics for target frames, shape [N_target, 3, 3]
            target_frame_extrinsics (torch.Tensor): Extrinsics for target frames (W2C), shape [N_target, 4, 4]
            image_size (tuple): (H, W) image dimensions
            chunk_size (int): Number of views to process per chunk (default: 4)

        Returns:
            torch.Tensor: Sky colors for target frames, shape [N_target, 3, H, W]
        """
        if not self.use_sky_token:
            raise RuntimeError("Cannot infer sky colors: model was not trained with sky_token")

        N_target = target_frame_intrinsics.shape[0]
        H, W = image_size
        device = target_frame_intrinsics.device

        # Average sky token across context frames and batch
        # sky_token shape: [B, S_context, 1, embed_dim] -> [1, embed_dim]
        averaged_sky_token = sky_token.mean(dim=[0, 1, 2], keepdim=True)  # [1, embed_dim]

        # Process in chunks to avoid OOM
        all_sky_colors = []

        for chunk_start in range(0, N_target, chunk_size):
            chunk_end = min(chunk_start + chunk_size, N_target)
            chunk_intrinsics = target_frame_intrinsics[chunk_start:chunk_end]
            chunk_extrinsics = target_frame_extrinsics[chunk_start:chunk_end]

            # Convert W2C to C2W for this chunk
            chunk_c2w = torch.inverse(chunk_extrinsics)

            # Generate ray directions for this chunk
            ray_dict = self.plucker_embedder(
                chunk_intrinsics,
                chunk_c2w,
                image_size=image_size
            )
            ray_dirs = ray_dict["dirs"]  # [chunk_size, H, W, 3]

            # Predict sky colors for this chunk
            chunk_sky_colors = self.sky_head(
                ray_dirs.reshape(chunk_end - chunk_start, H * W, 3),
                averaged_sky_token
            )
            chunk_sky_colors = chunk_sky_colors.view(chunk_end - chunk_start, H, W, 3).permute(0, 3, 1, 2)  # [chunk_size, 3, H, W]

            all_sky_colors.append(chunk_sky_colors)

        # Concatenate all chunks
        target_sky_colors = torch.cat(all_sky_colors, dim=0)  # [N_target, 3, H, W]

        return target_sky_colors