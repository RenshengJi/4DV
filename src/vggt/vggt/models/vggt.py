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
from einops import reduce
from functools import cache
from jaxtyping import Float
from lpips import LPIPS
from skimage.metrics import structural_similarity
from torch import Tensor

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
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, use_sky_token=True):
        super().__init__()
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, use_sky_token=use_sky_token)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.gaussian_head = DPTGSHead(dim_in=2 * embed_dim, output_dim=15, activation="linear", conf_activation="expp1")
        self.velocity_head = DPTGSHead(dim_in=2 * embed_dim, output_dim=4, activation="linear", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        
        # Sky token support
        self.use_sky_token = use_sky_token
        if self.use_sky_token:
            self.plucker_embedder = PluckerEmbedder(img_size=img_size)
            self.sky_head = ModulatedLinearLayer(
                3,  # input channels (ray directions)
                hidden_channels=512,
                condition_channels=embed_dim*2,
                out_channels=3,  # RGB output
            )
        
        self.set_freeze(freeze="vggt_wo_gaussian_and_velocity")

    def set_freeze(self, freeze):
        to_be_frozen = {
            "none": [],
            "encoder": [
                self.aggregator.camera_token,
                self.aggregator.register_token,
                self.aggregator.patch_embed,
            ],
            "encoder_frame": [
                self.aggregator.camera_token,
                self.aggregator.register_token,
                self.aggregator.patch_embed,
                self.aggregator.frame_blocks,
            ],
            "backbone": [
                self.aggregator,
            ],
            "backbone_camera": [
                self.aggregator,
                self.camera_head,
            ],
            "backbone_pts": [
                self.aggregator,
                self.point_head,
                self.depth_head,
            ],
            "vggt": [
                self.aggregator,
                self.camera_head,
                self.point_head,
                self.depth_head,
            ],
            "vggt_wo_velocity": [
                self.aggregator,
                self.camera_head,
                self.point_head,
                self.gaussian_head,
                self.depth_head,
            ],
            "vggt_wo_depth": [
                self.aggregator,
                self.camera_head,
                self.point_head,
                self.gaussian_head,
                # self.depth_head,
                self.velocity_head,
            ],
            "vggt_wo_gaussian_and_velocity": [
                # self.aggregator,
                self.camera_head,
                self.point_head,
                self.gaussian_head,
                self.depth_head,
                # self.velocity_head,
            ],
            "all": [
                self.aggregator,
                self.camera_head,
                self.point_head,
                self.gaussian_head,
                self.depth_head,
                self.velocity_head,
            ],
        }
        if hasattr(self, "track_head"):
            to_be_frozen["vggt"].append(self.track_head)
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

        for module in modules:
            set_gradient_checkpointing(module)

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None, compute_sky_color_loss=False, sky_masks=None, gt_images=None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

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

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

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

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.gaussian_head is not None:
                gaussian_params, gaussian_conf = self.gaussian_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["gaussian_params"] = gaussian_params
                predictions["gaussian_conf"] = gaussian_conf

            if self.velocity_head is not None:
                velocity, velocity_conf = self.velocity_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["velocity"] = velocity
                predictions["velocity_conf"] = velocity_conf

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
            # Extract sky token from the first aggregated tokens (last iteration)
            sky_token = aggregated_tokens_list[-1][:, :, :1]  # [B, S,1, embed_dim]
            predictions["sky_token"] = sky_token

        if compute_sky_color_loss and sky_masks is not None and gt_images is not None:
            pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(
                predictions["pose_enc"].detach(), (images.shape[-2], images.shape[-1])
            )
            pred_extrinsics = torch.cat([pred_extrinsics, torch.tensor([0, 0, 0, 1], device=pred_extrinsics.device)[None,None,None,:].repeat(1,pred_extrinsics.shape[1],1,1)], dim=-2)
            ray_directions = self.generate_ray_directions(
                pred_intrinsics, pred_extrinsics, image_size=(images.shape[-2], images.shape[-1])
            )
            pred_sky_colors = self.generate_sky_color(
                ray_directions.view(-1, ray_directions.shape[-3], ray_directions.shape[-2], 3),
                predictions["sky_token"].view(-1, 1, predictions["sky_token"].shape[-1])
            )
            B, S = images.shape[:2]
            H, W = pred_sky_colors.shape[1:3]
            pred_sky_colors = pred_sky_colors.view(B, S, H, W, 3)
            predictions["pred_sky_colors"] = pred_sky_colors

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
