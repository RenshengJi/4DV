# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from src.utils import tf32_off, compute_lpips
from gsplat.rendering import rasterization
from models.utils.pose_enc import pose_encoding_to_extri_intri
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import sys
import os



def check_and_fix_inf_nan(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """Check and fix inf/nan values in tensor."""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor


def create_dynamic_mask_from_pixel_indices(
    dynamic_objects: list,
    frame_idx: int,
    height: int,
    width: int
) -> torch.Tensor:
    """
    Create dynamic mask from 2D pixel indices of dynamic objects.
    Supports both single-camera and multi-camera modes.

    Args:
        dynamic_objects: List of dynamic objects containing frame_pixel_indices
        frame_idx: Current time frame index
        height, width: Image dimensions

    Returns:
        mask: [H, W] - Boolean mask, True indicates dynamic regions

    Note:
        Single-camera: frame_pixel_indices = {frame_idx: [pixel1, pixel2, ...]}
        Multi-camera: frame_pixel_indices = {frame_idx: {view_idx: [pixel1, pixel2, ...]}}
    """
    device = None
    for obj_data in dynamic_objects:
        if 'canonical_gaussians' in obj_data and obj_data['canonical_gaussians'] is not None:
            device = obj_data['canonical_gaussians'].device
            break

    if device is None:
        device = torch.device('cpu')

    mask = torch.zeros(height, width, device=device, dtype=torch.bool)

    for obj_data in dynamic_objects:
        frame_pixel_indices = obj_data.get('frame_pixel_indices', {})
        if frame_idx not in frame_pixel_indices:
            continue

        pixel_data = frame_pixel_indices[frame_idx]
        if not pixel_data:
            continue

        if isinstance(pixel_data, dict):
            all_pixel_indices = []
            for view_idx, view_pixels in pixel_data.items():
                if view_pixels:
                    all_pixel_indices.extend(view_pixels)

            if not all_pixel_indices:
                continue

            pixel_indices = all_pixel_indices

        elif isinstance(pixel_data, list):
            pixel_indices = pixel_data

        else:
            continue

        pixel_indices_tensor = torch.tensor(pixel_indices, dtype=torch.long, device=device)
        v_coords = pixel_indices_tensor // width
        u_coords = pixel_indices_tensor % width

        valid = (v_coords >= 0) & (v_coords < height) & (u_coords >= 0) & (u_coords < width)
        v_valid = v_coords[valid]
        u_valid = u_coords[valid]

        mask[v_valid, u_valid] = True

    return mask


def depth_to_world_points(depth, intrinsic):
    """
    Convert depth map to 3D points in camera coordinate system.

    Args:
        depth: [N, H, W, 1] - Depth map
        intrinsic: [1, N, 3, 3] - Camera intrinsic matrix

    Returns:
        camera_points: [N, H, W, 3] - 3D points in camera coordinates
    """
    with tf32_off():
        N, H, W, _ = depth.shape

        v, u = torch.meshgrid(torch.arange(H, device=depth.device),
                              torch.arange(W, device=depth.device),
                              indexing='ij')
        uv1 = torch.stack([u, v, torch.ones_like(u)],
                          dim=-1).float()
        uv1 = uv1.unsqueeze(0).expand(N, -1, -1, -1)

        depth = depth.squeeze(-1)
        intrinsic = intrinsic.squeeze(0)

        camera_points = torch.einsum(
            'nij,nhwj->nhwi', torch.inverse(intrinsic), uv1)
        camera_points = camera_points * depth.unsqueeze(-1)

    return camera_points


def prune_gaussians_by_voxel(means, scales, rotations, opacities, colors, voxel_size, depth_scale_factor=None):
    """
    Prune Gaussians by merging those in the same voxel using averaging.

    Args:
        means: [N, 3] - Gaussian centers
        scales: [N, 3] - Gaussian scales
        rotations: [N, 4] - Gaussian rotations (quaternions)
        opacities: [N] - Gaussian opacities
        colors: [N, K, 3] - Gaussian colors (K is number of SH bases or 1)
        voxel_size: Voxel size in metric scale
        depth_scale_factor: Depth scale factor to convert from normalized to metric scale

    Returns:
        Pruned Gaussian parameters: means, scales, rotations, opacities, colors
    """
    if means.shape[0] == 0:
        return means, scales, rotations, opacities, colors

    if depth_scale_factor is not None:
        effective_voxel_size = voxel_size * depth_scale_factor
    else:
        effective_voxel_size = voxel_size

    voxel_indices = (means / effective_voxel_size).floor().long()

    min_indices = voxel_indices.min(dim=0)[0]
    voxel_indices = voxel_indices - min_indices
    max_dims = voxel_indices.max(dim=0)[0] + 1

    flat_indices = (voxel_indices[:, 0] * max_dims[1] * max_dims[2] +
                   voxel_indices[:, 1] * max_dims[2] +
                   voxel_indices[:, 2])

    unique_voxels, inverse_indices = torch.unique(flat_indices, return_inverse=True)
    K = len(unique_voxels)

    device = means.device
    num_sh = colors.shape[1] if colors.ndim == 3 else 1

    merged_means = torch.zeros((K, 3), device=device, dtype=means.dtype)
    merged_scales = torch.zeros((K, 3), device=device, dtype=scales.dtype)
    merged_rotations = torch.zeros((K, 4), device=device, dtype=rotations.dtype)
    merged_opacities = torch.zeros(K, device=device, dtype=opacities.dtype)
    if colors.ndim == 3:
        merged_colors = torch.zeros((K, num_sh, 3), device=device, dtype=colors.dtype)
    else:
        merged_colors = torch.zeros((K, 3), device=device, dtype=colors.dtype)

    counts = torch.zeros(K, device=device, dtype=torch.float32)
    counts.scatter_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float32))
    counts = torch.clamp(counts, min=1.0)

    for d in range(3):
        merged_means[:, d].scatter_add_(0, inverse_indices, means[:, d])
    merged_means = merged_means / counts.unsqueeze(1)

    for d in range(3):
        merged_scales[:, d].scatter_add_(0, inverse_indices, scales[:, d])
    merged_scales = merged_scales / counts.unsqueeze(1)

    merged_opacities.scatter_add_(0, inverse_indices, opacities)
    merged_opacities = merged_opacities / counts

    if colors.ndim == 3:
        for sh_idx in range(num_sh):
            for d in range(3):
                merged_colors[:, sh_idx, d].scatter_add_(0, inverse_indices, colors[:, sh_idx, d])
        merged_colors = merged_colors / counts.unsqueeze(-1).unsqueeze(-1)
    else:
        for d in range(3):
            merged_colors[:, d].scatter_add_(0, inverse_indices, colors[:, d])
        merged_colors = merged_colors / counts.unsqueeze(1)

    for d in range(4):
        merged_rotations[:, d].scatter_add_(0, inverse_indices, rotations[:, d])
    quat_norms = torch.norm(merged_rotations, dim=1, keepdim=True)
    merged_rotations = merged_rotations / torch.clamp(quat_norms, min=1e-8)

    return merged_means, merged_scales, merged_rotations, merged_opacities, merged_colors


class Stage2RenderLoss(nn.Module):
    """
    Stage 2 rendering loss.

    Supports rendering and supervision on both context frames and target frames.
    Context frames: Sparse frames used for network inference
    Target frames: Dense frames for additional supervision (optional)
    """

    def __init__(
        self,
        rgb_weight: float = 1.0,
        depth_weight: float = 0.0,
        lpips_weight: float = 0.0,
        sh_degree: int = 0,
        enable_voxel_pruning: bool = True,
        voxel_size: float = 0.002,
        supervise_target_frames: bool = False
    ):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.depth_weight = depth_weight
        self.lpips_weight = lpips_weight
        self.sh_degree = sh_degree
        self.enable_voxel_pruning = enable_voxel_pruning
        self.voxel_size = voxel_size
        self.supervise_target_frames = supervise_target_frames

    def forward(
        self,
        refined_scene: Dict,
        gt_images: torch.Tensor,
        gt_depths: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        sky_masks: Optional[torch.Tensor] = None,
        sky_colors: Optional[torch.Tensor] = None,
        depth_scale_factor: Optional[torch.Tensor] = None,
        camera_indices: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
        is_context_frame: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute stage 2 rendering loss.

        Args:
            refined_scene: Refined scene representation
            gt_images: [B, S, 3, H, W] - Ground truth images
            gt_depths: [B, S, H, W] - Ground truth depths
            intrinsics: [B, S, 3, 3] - Camera intrinsics
            extrinsics: [B, S, 4, 4] - Camera extrinsics
            sky_masks: [B, S, H, W] - Sky region masks
            sky_colors: [B, S, 3, H, W] - Sky colors for compositing
            depth_scale_factor: Depth scale factor for voxel pruning
            camera_indices: [B, S] - Camera indices for multi-camera mode
            frame_indices: [B, S] - Frame indices for multi-camera mode (temporal frame indices)
            is_context_frame: [B, S] - Boolean mask indicating context frames (True) vs target frames (False)

        Returns:
            loss_dict: Loss dictionary
        """
        B, S, C, H, W = gt_images.shape
        device = gt_images.device

        # Determine which frames to supervise
        if is_context_frame is not None:
            if self.supervise_target_frames:
                # Supervise all frames (both context and target)
                supervise_mask = torch.ones(S, dtype=torch.bool, device=device)
            else:
                # Only supervise context frames
                supervise_mask = is_context_frame[0]  # [S]
        else:
            # If no context/target distinction, supervise all frames (backward compatibility)
            supervise_mask = torch.ones(S, dtype=torch.bool, device=device)

        rendered_images = []
        rendered_depths = []
        frame_indices_to_render = []

        num_camera_frames = intrinsics.shape[1]
        camera_idx = 0

        for frame_idx in range(S):
            if not supervise_mask[frame_idx]:
                continue
            if camera_idx >= num_camera_frames:
                break

            frame_intrinsic = intrinsics[0, camera_idx]
            frame_extrinsic = extrinsics[0, camera_idx]
            temporal_frame_idx = int(frame_indices[0, frame_idx].item())

            rendered_rgb, rendered_depth, rendered_alpha = self._render_frame(
                refined_scene, frame_intrinsic, frame_extrinsic, H, W, temporal_frame_idx,
                depth_scale_factor=depth_scale_factor
            )

            if sky_colors is not None:
                frame_sky_color = sky_colors[0, camera_idx]
                alpha_3ch = rendered_alpha.unsqueeze(0)
                rendered_rgb = alpha_3ch * rendered_rgb + (1 - alpha_3ch) * frame_sky_color

            camera_idx += 1
            rendered_images.append(rendered_rgb)
            rendered_depths.append(rendered_depth)
            frame_indices_to_render.append(frame_idx)

        if len(rendered_images) == 0:
            dummy_param = None
            if 'dynamic_objects_cars' in refined_scene and len(refined_scene['dynamic_objects_cars']) > 0:
                for obj_data in refined_scene['dynamic_objects_cars']:
                    if 'canonical_gaussians' in obj_data and obj_data['canonical_gaussians'] is not None:
                        dummy_param = obj_data['canonical_gaussians']
                        break

            if dummy_param is None and 'dynamic_objects_people' in refined_scene and len(refined_scene['dynamic_objects_people']) > 0:
                for obj_data in refined_scene['dynamic_objects_people']:
                    frame_gaussians = obj_data.get('frame_gaussians', {})
                    for frame_idx, gaussians in frame_gaussians.items():
                        if gaussians is not None and gaussians.requires_grad:
                            dummy_param = gaussians
                            break
                    if dummy_param is not None:
                        break

            if dummy_param is not None and dummy_param.requires_grad:
                zero_loss = dummy_param.sum() * 0.0
            else:
                zero_loss = torch.zeros(1, device=device, requires_grad=True).sum()

            return {
                'stage2_rgb_loss': zero_loss,
                'stage2_depth_loss': zero_loss,
                'stage2_lpips_loss': zero_loss,
                'stage2_total_loss': zero_loss
            }

        pred_rgb = torch.stack(rendered_images, dim=0)
        pred_depth = torch.stack(rendered_depths, dim=0)

        # Select GT frames that were supervised
        frame_indices_tensor = torch.tensor(frame_indices_to_render, device=device)
        gt_rgb = gt_images[0, frame_indices_tensor]
        gt_depth = gt_depths[0, frame_indices_tensor]

        rgb_loss = F.l1_loss(pred_rgb, gt_rgb)

        if self.lpips_weight > 0:
            lpips_loss = compute_lpips(pred_rgb, gt_rgb).mean()
        else:
            lpips_loss = rgb_loss * 0.0

        if self.depth_weight > 0:
            valid_depth_mask = gt_depth > 0
            if valid_depth_mask.sum() > 0:
                depth_loss = F.l1_loss(pred_depth[valid_depth_mask], gt_depth[valid_depth_mask])
            else:
                depth_loss = rgb_loss * 0.0
        else:
            depth_loss = rgb_loss * 0.0

        loss_dict = {
            'stage2_rgb_loss': self.rgb_weight * rgb_loss,
            'stage2_depth_loss': self.depth_weight * depth_loss,
            'stage2_lpips_loss': self.lpips_weight * lpips_loss,
        }
        loss_dict['stage2_total_loss'] = sum(loss_dict.values())

        if sky_masks is not None and len(rendered_images) > 0:
            # Select sky masks for supervised frames only
            sky_masks_supervised = sky_masks[0, frame_indices_tensor].bool()

            sky_mask_3ch = sky_masks_supervised.unsqueeze(1).expand(-1, 3, -1, -1)
            nonsky_mask_3ch = ~sky_mask_3ch

            if sky_mask_3ch.any():
                sky_pred = pred_rgb[sky_mask_3ch].detach()
                sky_gt = gt_rgb[sky_mask_3ch].detach()
                loss_dict['stage2_rgb_loss_sky'] = F.l1_loss(sky_pred, sky_gt)

            if nonsky_mask_3ch.any():
                nonsky_pred = pred_rgb[nonsky_mask_3ch].detach()
                nonsky_gt = gt_rgb[nonsky_mask_3ch].detach()
                loss_dict['stage2_rgb_loss_nonsky'] = F.l1_loss(nonsky_pred, nonsky_gt)

        for key, value in loss_dict.items():
            loss_dict[key] = check_and_fix_inf_nan(value, key)

        return loss_dict
    

    def _render_frame(
        self,
        refined_scene: Dict,
        intrinsic: torch.Tensor,
        extrinsic: torch.Tensor,
        height: int,
        width: int,
        frame_idx: int = 0,
        sky_colors: Optional[torch.Tensor] = None,
        depth_scale_factor: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Render a single frame.

        Args:
            refined_scene: Refined scene representation
            intrinsic: [3, 3] - Camera intrinsic matrix
            extrinsic: [4, 4] - Camera extrinsic matrix
            height, width: Image dimensions
            frame_idx: Current frame index
            sky_colors: [H, W, 3] - Sky colors (optional)
            depth_scale_factor: Depth scale factor for voxel pruning

        Returns:
            rendered_rgb: [3, H, W] - Rendered RGB image
            rendered_depth: [H, W] - Rendered depth map
            rendered_alpha: [H, W] - Rendered opacity
        """
        device = intrinsic.device

        all_means = []
        all_scales = []
        all_rotations = []
        all_opacities = []
        all_colors = []

        if refined_scene.get('static_gaussians') is not None:
            static_gaussians = refined_scene['static_gaussians']
            if static_gaussians.shape[0] > 0:
                all_means.append(static_gaussians[:, :3])
                all_scales.append(static_gaussians[:, 3:6])
                all_colors.append(static_gaussians[:, 6:9].unsqueeze(-2))
                all_rotations.append(static_gaussians[:, 9:13])
                all_opacities.append(static_gaussians[:, 13])

        dynamic_objects_cars = refined_scene.get('dynamic_objects_cars', [])
        for obj_data in dynamic_objects_cars:
            if 'frame_transforms' not in obj_data or frame_idx not in obj_data['frame_transforms']:
                continue

            canonical_gaussians = obj_data.get('canonical_gaussians')
            if canonical_gaussians is None or canonical_gaussians.shape[0] == 0:
                continue

            frame_transform = self._get_object_transform_to_frame(
                obj_data, frame_idx)
            if frame_transform is None:
                transformed_gaussians = canonical_gaussians
            else:
                transformed_gaussians = self._apply_transform_to_gaussians(
                    canonical_gaussians, frame_transform
                )

            if transformed_gaussians.shape[0] > 0:
                all_means.append(transformed_gaussians[:, :3])
                all_scales.append(transformed_gaussians[:, 3:6])
                all_colors.append(transformed_gaussians[:, 6:9].unsqueeze(-2))
                all_rotations.append(transformed_gaussians[:, 9:13])
                all_opacities.append(transformed_gaussians[:, 13])

        # Add dynamic people gaussians
        dynamic_objects_people = refined_scene.get('dynamic_objects_people', [])
        for obj_data in dynamic_objects_people:
            frame_gaussians = obj_data.get('frame_gaussians', {})

            # If target frame, try to extrapolate from nearest context frame
            if frame_idx not in frame_gaussians:
                # Find nearest context frame with gaussians
                available_frames = sorted(frame_gaussians.keys())
                if len(available_frames) == 0:
                    continue

                # Find frames before and after target
                frames_before = [f for f in available_frames if f < frame_idx]
                frames_after = [f for f in available_frames if f > frame_idx]

                # Determine which frames to use for interpolation
                if len(frames_before) > 0 and len(frames_after) > 0:
                    # Target is between two context frames - use velocity from frame before
                    frame_from = frames_before[-1]  # Closest frame before target
                    frame_to = frames_after[0]      # Next context frame (velocity points to this)

                    # Get gaussians and velocity from frame_from
                    gaussians_from = frame_gaussians[frame_from]
                    if gaussians_from is None or gaussians_from.shape[0] == 0:
                        continue

                    frame_velocities = obj_data.get('frame_velocities', {})
                    if frame_from in frame_velocities:
                        velocity = frame_velocities[frame_from]

                        # Velocity points from frame_from to frame_to
                        # Compute interpolation factor: alpha = (target - from) / (to - from)
                        alpha = (frame_idx - frame_from) / (frame_to - frame_from)

                        # Interpolate position: new_pos = old_pos + velocity * alpha
                        extrapolated_gaussians = gaussians_from.clone()
                        extrapolated_gaussians[:, :3] = gaussians_from[:, :3] + velocity * alpha

                        current_frame_gaussians = extrapolated_gaussians
                    else:
                        # No velocity, use gaussians from nearest frame
                        current_frame_gaussians = gaussians_from

                elif len(frames_before) > 0:
                    # Target is after all context frames - use last frame
                    nearest_frame = frames_before[-1]
                    current_frame_gaussians = frame_gaussians[nearest_frame]
                    if current_frame_gaussians is None or current_frame_gaussians.shape[0] == 0:
                        continue

                elif len(frames_after) > 0:
                    # Target is before all context frames - use first frame
                    nearest_frame = frames_after[0]
                    current_frame_gaussians = frame_gaussians[nearest_frame]
                    if current_frame_gaussians is None or current_frame_gaussians.shape[0] == 0:
                        continue
                else:
                    continue
            else:
                current_frame_gaussians = frame_gaussians[frame_idx]

            if current_frame_gaussians is None or current_frame_gaussians.shape[0] == 0:
                continue

            all_means.append(current_frame_gaussians[:, :3])
            all_scales.append(current_frame_gaussians[:, 3:6])
            all_colors.append(current_frame_gaussians[:, 6:9].unsqueeze(-2))
            all_rotations.append(current_frame_gaussians[:, 9:13])
            all_opacities.append(current_frame_gaussians[:, 13])

        if not all_means:
            return (
                torch.zeros(3, height, width, device=device),
                torch.zeros(height, width, device=device),
                torch.zeros(height, width, device=device)
            )

        means = torch.cat(all_means, dim=0)
        scales = torch.cat(all_scales, dim=0)
        colors = torch.cat(all_colors, dim=0)
        rotations = torch.cat(all_rotations, dim=0)
        opacities = torch.cat(all_opacities, dim=0)

        means = torch.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
        scales = torch.nan_to_num(scales, nan=0.01, posinf=1.0, neginf=0.01)
        colors = torch.nan_to_num(colors, nan=0.5, posinf=1.0, neginf=0.0)
        rotations = torch.nan_to_num(rotations, nan=0.0, posinf=1.0, neginf=-1.0)
        opacities = torch.nan_to_num(opacities, nan=0.5, posinf=1.0, neginf=0.0)

        if self.enable_voxel_pruning and means.shape[0] > 0:
            dsf = depth_scale_factor.item() if depth_scale_factor is not None and torch.is_tensor(depth_scale_factor) else depth_scale_factor

            colors_squeezed = colors.squeeze(1)
            means, scales, rotations, opacities, colors_pruned = prune_gaussians_by_voxel(
                means, scales, rotations, opacities, colors_squeezed,
                voxel_size=self.voxel_size,
                depth_scale_factor=dsf
            )
            colors = colors_pruned.unsqueeze(1)

        viewmat = extrinsic.unsqueeze(0)
        K = intrinsic.unsqueeze(0)

        render_result = rasterization(
            means, rotations, scales, opacities, colors,
            viewmat, K, width, height,
            sh_degree=self.sh_degree, render_mode="RGB+ED",
            radius_clip=0, near_plane=0.0001,
            far_plane=1000.0,
            eps2d=0.3,
        )

        rendered_image, rendered_alphas, _ = render_result
        rendered_rgb = rendered_image[0, :, :, :3].permute(2, 0, 1)
        rendered_depth = rendered_image[0, :, :, -1]
        rendered_alpha = rendered_alphas[0, :, :, 0] if rendered_alphas is not None else torch.zeros(height, width, device=device)

        return rendered_rgb, rendered_depth, rendered_alpha

    def _get_object_transform_to_frame(self, obj_data: Dict, frame_idx: int) -> Optional[torch.Tensor]:
        """Get transformation matrix from canonical space to specified frame."""
        reference_frame = obj_data.get('reference_frame', 0)
        if frame_idx == reference_frame:
            return None

        if 'frame_transforms' in obj_data:
            frame_transforms = obj_data['frame_transforms']
            if frame_idx in frame_transforms:
                frame_to_canonical = frame_transforms[frame_idx]
                try:
                    original_dtype = frame_to_canonical.dtype
                    canonical_to_frame = torch.linalg.inv(frame_to_canonical.float()).to(original_dtype)
                    return canonical_to_frame
                except Exception as e:
                    print(
                        f"Warning: Failed to invert transform for frame {frame_idx}: {e}")
                    return None

        return None


    def _apply_transform_to_gaussians(
        self,
        gaussians: torch.Tensor,
        transform: torch.Tensor
    ) -> torch.Tensor:
        """Apply transformation to Gaussian parameters."""
        transformed_gaussians = gaussians.clone()
        positions = gaussians[:, :3]
        positions_homo = torch.cat([positions, torch.ones(
            positions.shape[0], 1, device=positions.device)], dim=1)
        transformed_positions = torch.mm(
            transform, positions_homo.T).T[:, :3]
        transformed_gaussians[:, :3] = transformed_positions

        return transformed_gaussians


class Stage2CompleteLoss(nn.Module):
    """Stage 2 complete loss function."""

    def __init__(
        self,
        render_loss_config: Optional[Dict] = None
    ):
        super().__init__()

        if render_loss_config is None:
            render_loss_config = {
                'rgb_weight': 1.0,
                'depth_weight': 0.0,
            }

        self.render_loss = Stage2RenderLoss(**render_loss_config)

    def forward(
        self,
        refined_scene: Dict,
        gt_images: torch.Tensor,
        gt_depths: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        sky_masks: Optional[torch.Tensor] = None,
        sky_colors: Optional[torch.Tensor] = None,
        depth_scale_factor: Optional[torch.Tensor] = None,
        camera_indices: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
        is_context_frame: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute stage 2 complete loss.

        Args:
            refined_scene: Refined scene representation
            gt_images: Ground truth images
            gt_depths: Ground truth depths
            intrinsics: Camera intrinsics
            extrinsics: Camera extrinsics
            sky_masks: Sky region masks
            sky_colors: [B, S, 3, H, W] - Sky colors
            depth_scale_factor: Depth scale factor for voxel pruning
            camera_indices: [B, S_total] - Camera indices for multi-camera mode
            frame_indices: [B, S_total] - Frame indices for multi-camera mode
            is_context_frame: [B, S] - Boolean mask indicating context vs target frames

        Returns:
            complete_loss_dict: Complete loss dictionary
        """
        render_loss_dict = self.render_loss(
            refined_scene, gt_images, gt_depths,
            intrinsics, extrinsics, sky_masks,
            sky_colors, depth_scale_factor, camera_indices, frame_indices, is_context_frame
        )

        complete_loss_dict = render_loss_dict.copy()
        complete_loss_dict['stage2_final_total_loss'] = render_loss_dict['stage2_total_loss']

        return complete_loss_dict
