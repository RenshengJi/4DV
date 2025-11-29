# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.gs_head import DPTGSHead
from vggt.heads.track_head import TrackHead
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# 直接引用storm中的组件
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../storm'))
from storm.models.decoder import ModulatedLinearLayer
from storm.models.embedders import PluckerEmbedder

from contextlib import contextmanager
# from einops import reduce
# from functools import cache
# from jaxtyping import Float
# from lpips import LPIPS
# from skimage.metrics import structural_similarity
# from torch import Tensor

@contextmanager
def tf32_off():
    original = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False  # disable tf32 temporarily
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original  # restore original setting

def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            module.requires_grad = False


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, use_sky_token=True, use_scale_token=True, memory_efficient=True, sh_degree=0, use_gs_head=True, use_gs_head_velocity=False, use_gt_camera=False, velocity_head_small_init=False):
        super().__init__()
        # Store sh_degree for use in forward and other methods
        self.sh_degree = sh_degree
        # Store use_gs_head for determining head type
        self.use_gs_head = use_gs_head
        # Store use_gs_head_velocity for determining velocity head type
        self.use_gs_head_velocity = use_gs_head_velocity
        # Store use_gt_camera for determining which camera parameters to use
        self.use_gt_camera = use_gt_camera

        # Calculate gaussian head output dimension based on sh_degree
        # Gaussian parameters: xyz_offset(3) + scale(3) + sh_coeffs + rotation(4) + opacity(1)
        # For sh_degree D, we need 3*(D+1)^2 parameters for RGB spherical harmonics
        # Note: +1 for confidence channel (will be separated by activate_head)
        sh_dim = 3 * ((sh_degree + 1) ** 2)
        gaussian_output_dim = 3 + 3 + sh_dim + 4 + 1 + 1  # = 12 + sh_dim (last +1 for confidence)

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

        # Choose head type based on use_gs_head flag for gaussian head
        GaussianHeadClass = DPTGSHead if use_gs_head else DPTHead
        self.gaussian_head = GaussianHeadClass(dim_in=2 * embed_dim, output_dim=gaussian_output_dim, activation="linear", conf_activation="expp1")
        # Choose head type based on use_gs_head_velocity flag for velocity head
        VelocityHeadClass = DPTGSHead if use_gs_head_velocity else DPTHead
        self.velocity_head = VelocityHeadClass(dim_in=2 * embed_dim, output_dim=4, activation="linear", conf_activation="expp1")

        # Apply small initialization to velocity_head if requested
        if velocity_head_small_init:
            self._apply_small_init_to_velocity_head()

        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)

        # Sky token support - now managed at VGGT level
        self.use_sky_token = use_sky_token
        if self.use_sky_token:
            # Sky token moved from aggregator to main model for better parameter control
            self.sky_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.plucker_embedder = PluckerEmbedder(img_size=img_size)
            self.sky_head = ModulatedLinearLayer(
                3,  # input channels (ray directions)
                hidden_channels=512,
                condition_channels=embed_dim*2,
                out_channels=3,  # RGB output
            )

        # Scale token support - similar to map-anything for scene scale prediction
        self.use_scale_token = use_scale_token
        if self.use_scale_token:
            # Scale token for global scene scale learning
            self.scale_token = nn.Parameter(torch.zeros(embed_dim))
            torch.nn.init.trunc_normal_(self.scale_token, std=0.02)
            # MLP head for scale prediction (outputs single scalar per batch)
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
        This helps velocity predictions start from near-zero values.
        """
        # For DPTHead: last layer is scratch.output_conv2[-1] (the final Conv2d)
        # For DPTGSHead: similar structure with output_conv2
        if hasattr(self.velocity_head, 'scratch') and hasattr(self.velocity_head.scratch, 'output_conv2'):
            output_conv2 = self.velocity_head.scratch.output_conv2
            # output_conv2 is a Sequential with the last layer being Conv2d
            if isinstance(output_conv2, nn.Sequential):
                last_layer = output_conv2[-1]
                if isinstance(last_layer, nn.Conv2d):
                    # Small weight initialization (std=0.001)
                    nn.init.normal_(last_layer.weight, mean=0.0, std=0.001)
                    # Zero bias initialization
                    if last_layer.bias is not None:
                        nn.init.zeros_(last_layer.bias)
                    print(f"[VGGT] Applied small initialization to velocity_head last layer: weight std=0.001, bias=0")

    def set_freeze(self, freeze):
        to_be_frozen = {
            "none": [],
            "all": [
                self.aggregator,
                self.camera_head,
                self.point_head,
                self.depth_head,
                self.gaussian_head,
                self.velocity_head,
                self.scale_head if self.use_scale_token else None,
                self.scale_token if self.use_scale_token else None,
                self.sky_token if self.use_sky_token else None,
                self.sky_head if self.use_sky_token else None,
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
        # scale_head is too small, no need for gradient checkpointing

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

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        # Pass sky_token and scale_token to aggregator if enabled
        kwargs = {}
        if self.use_sky_token:
            kwargs['sky_token'] = self.sky_token
        if self.use_scale_token:
            # Expand scale_token to [1, 1, embed_dim] to match sky_token format
            kwargs['scale_token'] = self.scale_token.unsqueeze(0).unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images, **kwargs)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

                # Compute world points from depth and camera parameters
                # Determine which camera parameters to use based on self.use_gt_camera flag
                extrinsics_to_use = None
                intrinsics_to_use = None

                if self.use_gt_camera and gt_extrinsics is not None and gt_intrinsics is not None:
                    # Use GT camera parameters
                    extrinsics_to_use = gt_extrinsics
                    intrinsics_to_use = gt_intrinsics
                    # Ensure extrinsics has homogeneous coordinates [B, S, 4, 4]
                    if extrinsics_to_use.shape[-2] == 3:
                        B_cam, S_cam = extrinsics_to_use.shape[:2]
                        homo_row = torch.tensor([0, 0, 0, 1], device=extrinsics_to_use.device, dtype=extrinsics_to_use.dtype)
                        homo_row = homo_row.view(1, 1, 1, 4).expand(B_cam, S_cam, 1, 4)
                        extrinsics_to_use = torch.cat([extrinsics_to_use, homo_row], dim=-2)
                elif "pose_enc" in predictions:
                    # Use predicted camera parameters
                    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
                    extrinsics_pred, intrinsics_pred = pose_encoding_to_extri_intri(
                        predictions["pose_enc"], images.shape[-2:]
                    )
                    # Add homogeneous coordinates to predicted extrinsics
                    B_cam, S_cam = extrinsics_pred.shape[:2]
                    homo_row = torch.tensor([0, 0, 0, 1], device=extrinsics_pred.device, dtype=extrinsics_pred.dtype)
                    homo_row = homo_row.view(1, 1, 1, 4).expand(B_cam, S_cam, 1, 4)
                    extrinsics_to_use = torch.cat([extrinsics_pred, homo_row], dim=-2)
                    intrinsics_to_use = intrinsics_pred

                # Compute world points if camera parameters are available
                if extrinsics_to_use is not None and intrinsics_to_use is not None:
                    from vggt.training.loss import depth_to_world_points

                    B_d, S_d, H_d, W_d = depth.shape[0], depth.shape[1], depth.shape[2], depth.shape[3]

                    # Reshape depth to [B*S, H, W, 1] for depth_to_world_points
                    depth_reshaped = depth.reshape(B_d * S_d, H_d, W_d, 1)
                    intrinsics_reshaped = intrinsics_to_use.reshape(B_d * S_d, 3, 3)

                    # Compute world points [B*S, H*W, 3]
                    world_points = depth_to_world_points(depth_reshaped, intrinsics_reshaped)
                    world_points = world_points.reshape(B_d, S_d, H_d * W_d, 3)

                    # Transform to camera coordinate frame (first frame as reference)
                    extrinsic_inv = torch.linalg.inv(extrinsics_to_use)  # [B, S, 4, 4]

                    # Use first batch [S, H*W, 3]
                    world_points_b0 = world_points[0]
                    extrinsic_inv_b0 = extrinsic_inv[0]  # [S, 4, 4]

                    # Apply transformation: xyz_cam = R_inv @ xyz_world + t_inv
                    xyz_camera = torch.matmul(
                        extrinsic_inv_b0[:, :3, :3],
                        world_points_b0.transpose(-1, -2)
                    ).transpose(-1, -2) + extrinsic_inv_b0[:, :3, 3:4].transpose(-1, -2)

                    # Store results in predictions
                    predictions["world_points"] = world_points.reshape(B_d, S_d, H_d, W_d, 3)
                    predictions["xyz_camera"] = xyz_camera.unsqueeze(0)  # [1, S, H*W, 3]
                    predictions["extrinsics"] = extrinsics_to_use
                    predictions["intrinsics"] = intrinsics_to_use

            # if self.point_head is not None:
            #     pts3d, pts3d_conf = self.point_head(
            #         aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            #     )
            #     predictions["world_points"] = pts3d
            #     predictions["world_points_conf"] = pts3d_conf

            if self.gaussian_head is not None:
                gaussian_params_raw, gaussian_conf = self.gaussian_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                # Apply activation to gaussian parameters (moved from loss functions)
                # gaussian_params_raw shape: [B, S, H, W, output_dim]
                # output_dim = 11 + 3*(sh_degree+1)^2
                # Channels: [xyz_offset(3), scale(3), sh_coefficients(3*(sh_degree+1)^2), rotation(4), opacity(1)]
                # Note: confidence is returned separately as gaussian_conf, not in gaussian_params
                B, S, H, W, _ = gaussian_params_raw.shape
                gaussian_params_activated = gaussian_params_raw.clone()

                # Calculate sh_dim and indices
                sh_dim = 3 * ((self.sh_degree + 1) ** 2)
                scale_end = 6
                sh_start = 6
                sh_end = 6 + sh_dim
                rotation_start = sh_end
                rotation_end = rotation_start + 4
                opacity_idx = rotation_end

                # Scale activation: 0.05 * exp(scale), clamped to max 0.3
                gaussian_params_activated[..., 3:scale_end] = (0.05 * torch.exp(gaussian_params_raw[..., 3:scale_end])).clamp_max(0.3)

                # Rotation normalization
                rotations = gaussian_params_raw[..., rotation_start:rotation_end]
                rotation_norms = torch.norm(rotations, dim=-1, keepdim=True).clamp(min=1e-8)
                gaussian_params_activated[..., rotation_start:rotation_end] = rotations / rotation_norms

                # Opacity activation: sigmoid
                gaussian_params_activated[..., opacity_idx:opacity_idx+1] = gaussian_params_raw[..., opacity_idx:opacity_idx+1].sigmoid()

                predictions["gaussian_params"] = gaussian_params_activated  # Activated params
                predictions["gaussian_params_raw"] = gaussian_params_raw  # Keep raw for backward compatibility
                predictions["gaussian_conf"] = gaussian_conf
                predictions["sh_degree"] = self.sh_degree  # Store sh_degree for use in loss functions

            if self.velocity_head is not None:
                velocity_raw, velocity_conf = self.velocity_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                # Apply velocity activation: sign(x) * (exp(|x|) - 1)
                velocity = torch.sign(velocity_raw) * (torch.exp(torch.abs(velocity_raw)) - 1)
                predictions["velocity"] = velocity  # cam coordinate
                predictions["velocity_conf"] = velocity_conf

                # Transform velocity to global coordinate frame if camera parameters are available
                if extrinsics_to_use is not None and intrinsics_to_use is not None:
                    # velocity is in camera coordinate frame, we need to transform it to world frame
                    # extrinsics_to_use is world2cam, so extrinsic_inv is cam2world
                    extrinsic_cam2world = torch.linalg.inv(extrinsics_to_use)  # [B, S, 4, 4]

                    # Extract rotation matrix from cam2world
                    R_cam2world = extrinsic_cam2world[:, :, :3, :3]  # [B, S, 3, 3]

                    # velocity shape: [B, S, H, W, 3]
                    # Transform: vel_world = R_cam2world @ vel_camera
                    B_v, S_v, H_v, W_v, _ = velocity.shape
                    velocity_reshaped = velocity.reshape(B_v, S_v, H_v * W_v, 3)  # [B, S, H*W, 3]

                    # Apply rotation: [B, S, 3, 3] @ [B, S, H*W, 3]^T -> [B, S, 3, H*W] -> transpose -> [B, S, H*W, 3]
                    velocity_global = torch.matmul(R_cam2world, velocity_reshaped.transpose(-1, -2)).transpose(-1, -2)
                    velocity_global = velocity_global.reshape(B_v, S_v, H_v, W_v, 3)

                    predictions["velocity_global"] = velocity_global  # World frame velocity (for downstream use)

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images

        # Add sky token processing if enabled
        if self.use_sky_token:
            # Extract sky token from all aggregated tokens and average them
            # Stack all layers and take the mean
            all_sky_tokens = []
            for layer_idx in sorted(aggregated_tokens_list.keys()):
                layer_sky_token = aggregated_tokens_list[layer_idx][:, :, :1]  # [B, S, 1, embed_dim]
                all_sky_tokens.append(layer_sky_token)

            # Stack and average across all layers
            stacked_sky_tokens = torch.stack(all_sky_tokens, dim=0)  # [num_layers, B, S, 1, embed_dim]
            sky_token = stacked_sky_tokens.mean(dim=0)  # [B, S, 1, embed_dim]
            predictions["sky_token"] = sky_token

            # Pre-compute sky colors if GT camera parameters are provided (for training)
            if gt_extrinsics is not None and gt_intrinsics is not None:
                import random
                B, S = sky_token.shape[:2]
                H, W = images.shape[-2:]  # Get height and width from last two dimensions

                # Randomly sample frames for rendering to save memory (default: 1/4 of frames)
                num_frames_to_render = max(1, int(S * frame_sample_ratio))
                sampled_frame_indices = random.sample(range(S), num_frames_to_render)
                sampled_frame_indices = sorted(sampled_frame_indices)  # Keep order for consistency

                # Generate ray directions using plucker embedder (only for sampled frames)
                sampled_intrinsics = gt_intrinsics[0, sampled_frame_indices]  # [num_frames, 3, 3]
                sampled_extrinsics_w2c = gt_extrinsics[0, sampled_frame_indices]  # [num_frames, 4, 4] world2camera

                # Convert world2camera to camera2world by inverting
                sampled_extrinsics = torch.inverse(sampled_extrinsics_w2c)  # [num_frames, 4, 4] camera2world

                ray_dict = self.plucker_embedder(
                    sampled_intrinsics,
                    sampled_extrinsics,
                    image_size=(H, W)
                )
                ray_dirs = ray_dict["dirs"]  # [num_frames, H, W, 3]

                # Generate sky colors using sky_head (only for sampled frames)
                sampled_sky_token = sky_token[0, sampled_frame_indices, 0]  # [num_frames, embed_dim]
                sky_colors_sampled = self.sky_head(
                    ray_dirs.reshape(num_frames_to_render, H * W, 3),
                    sampled_sky_token
                )  # [num_frames, H*W, 3]
                sky_colors_sampled = sky_colors_sampled.view(num_frames_to_render, H, W, 3).permute(0, 3, 1, 2)  # [num_frames, 3, H, W]

                # Store sampled sky colors and indices for use in loss function
                predictions["sky_colors"] = sky_colors_sampled.unsqueeze(0)  # [B, num_frames, 3, H, W]
                predictions["sampled_frame_indices"] = sampled_frame_indices

        # Add scale token processing if enabled
        if self.use_scale_token:
            # Extract scale token from aggregated tokens (last iteration)
            # Scale token is typically placed after sky_token if both are enabled
            # Get the last layer (max key) from the dict
            last_layer_idx = max(aggregated_tokens_list.keys())
            token_idx = 1 if self.use_sky_token else 0
            scale_token_features = aggregated_tokens_list[last_layer_idx][:, :, token_idx:token_idx+1]  # [B, S, 1, embed_dim]
            # Average across sequence dimension to get global scale
            scale_token_pooled = scale_token_features.mean(dim=1).squeeze(1)  # [B, embed_dim]
            # Pass through scale head to get predicted scale
            predicted_scale = self.scale_head(scale_token_pooled)  # [B, 1]
            predictions["scale"] = predicted_scale.squeeze(-1)  # [B]

        # Add batch dimension information to predictions for downstream processing
        B, S, C, H, W = images.shape
        predictions["batch_dims"] = {
            "B": B,
            "S": S,
            "H": H,
            "W": W
        }

        return predictions

    def generate_sky_color(self, ray_directions, sky_token):
        """
        Generate sky color using sky_head and sky_token.
        
        Args:
            ray_directions (torch.Tensor): Ray directions with shape [B, H, W, 3]
            sky_token (torch.Tensor): Sky token with shape [B, 1, embed_dim]
            
        Returns:
            torch.Tensor: Sky colors with shape [B, H, W, 3]
        """
        if not self.use_sky_token:
            return None
        
        # Reshape ray_directions for sky_head
        B, H, W, _ = ray_directions.shape
        ray_dirs_flat = ray_directions.view(B, H * W, 3)
        
        # Generate sky colors using sky_head
        sky_colors = self.sky_head(ray_dirs_flat, sky_token)  # [B, H*W, 3]
        sky_colors = sky_colors.view(B, H, W, 3)
        
        return sky_colors

    def generate_ray_directions(self, intrinsics, camtoworlds, image_size=None):
        """
        Generate ray directions using PluckerEmbedder.
        
        Args:
            intrinsics (torch.Tensor): Camera intrinsics with shape [B, S, 3, 3]
            camtoworlds (torch.Tensor): Camera poses with shape [B, S, 4, 4]
            image_size (tuple, optional): Image size (H, W)
            
        Returns:
            torch.Tensor: Ray directions with shape [B, S, H, W, 3]
        """
        if not self.use_sky_token:
            return None
        
        B, S = intrinsics.shape[:2]
        ray_directions = []
        
        for s in range(S):
            ray_dict = self.plucker_embedder(
                intrinsics[:, s],  # [B, 3, 3]
                camtoworlds[:, s],  # [B, 4, 4]
                image_size=image_size
            )
            ray_directions.append(ray_dict["dirs"])  # [B, H, W, 3]
        
        return torch.stack(ray_directions, dim=1)  # [B, S, H, W, 3]
