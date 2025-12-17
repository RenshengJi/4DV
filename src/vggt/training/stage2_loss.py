# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dust3r.utils.misc import tf32_off
from gsplat.rendering import rasterization
from dust3r.utils.metrics import compute_lpips
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import sys
import os

# 添加必要的路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dust3r/utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


def check_and_fix_inf_nan(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """检查并修复tensor中的inf和nan值"""
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
    从动态物体的2D pixel indices创建dynamic mask
    支持单相机和多相机模式

    Args:
        dynamic_objects: 动态物体列表，每个包含frame_pixel_indices
        frame_idx: 当前时间帧索引
        height, width: 图像尺寸

    Returns:
        mask: [H, W] 布尔mask，True表示动态区域

    Note:
        单相机: frame_pixel_indices = {frame_idx: [pixel1, pixel2, ...]}
        多相机: frame_pixel_indices = {frame_idx: {view_idx: [pixel1, pixel2, ...]}}
    """
    # 使用第一个物体的device（假设所有物体在同一device上）
    device = None
    for obj_data in dynamic_objects:
        if 'canonical_gaussians' in obj_data and obj_data['canonical_gaussians'] is not None:
            device = obj_data['canonical_gaussians'].device
            break

    if device is None:
        device = torch.device('cpu')

    mask = torch.zeros(height, width, device=device, dtype=torch.bool)

    for obj_data in dynamic_objects:
        # 获取该物体在当前帧的pixel indices
        frame_pixel_indices = obj_data.get('frame_pixel_indices', {})
        if frame_idx not in frame_pixel_indices:
            continue

        pixel_data = frame_pixel_indices[frame_idx]
        if not pixel_data:
            continue

        # ========== 检查数据格式：单相机(list)或多相机(dict) ==========
        if isinstance(pixel_data, dict):
            # ========== 多相机模式: {view_idx: [pixel_idx, ...]} ==========
            # 合并所有view的pixel indices
            all_pixel_indices = []
            for view_idx, view_pixels in pixel_data.items():
                if view_pixels:
                    all_pixel_indices.extend(view_pixels)

            if not all_pixel_indices:
                continue

            pixel_indices = all_pixel_indices

        elif isinstance(pixel_data, list):
            # ========== 单相机模式: [pixel_idx, ...] ==========
            pixel_indices = pixel_data

        else:
            # 未知格式，跳过
            continue

        # 将1D pixel indices (0 to H*W-1) 转换为2D坐标 (v, u)
        pixel_indices_tensor = torch.tensor(pixel_indices, dtype=torch.long, device=device)
        v_coords = pixel_indices_tensor // width  # 行坐标
        u_coords = pixel_indices_tensor % width   # 列坐标

        # 过滤在图像范围内的坐标
        valid = (v_coords >= 0) & (v_coords < height) & (u_coords >= 0) & (u_coords < width)
        v_valid = v_coords[valid]
        u_valid = u_coords[valid]

        # 在mask上标记这些像素
        mask[v_valid, u_valid] = True

    return mask


def depth_to_world_points(depth, intrinsic):
    """
    将深度图转换为世界坐标系下的3D点

    Args:
        depth: [N, H, W, 1] 深度图
        intrinsic: [1, N, 3, 3] 相机内参矩阵

    Returns:
        world_points: [N, H, W, 3] 世界坐标点
    """
    with tf32_off():
        N, H, W, _ = depth.shape

        # 生成像素坐标网格
        v, u = torch.meshgrid(torch.arange(H, device=depth.device),
                              torch.arange(W, device=depth.device),
                              indexing='ij')
        uv1 = torch.stack([u, v, torch.ones_like(u)],
                          dim=-1).float()  # [H, W, 3]
        uv1 = uv1.unsqueeze(0).expand(N, -1, -1, -1)  # [N, H, W, 3]

        # 转换为相机坐标
        depth = depth.squeeze(-1)  # [N, H, W]
        intrinsic = intrinsic.squeeze(0)  # [N, 3, 3]

        # 计算相机坐标
        camera_points = torch.einsum(
            'nij,nhwj->nhwi', torch.inverse(intrinsic), uv1)  # [N, H, W, 3]
        camera_points = camera_points * depth.unsqueeze(-1)  # [N, H, W, 3]

    return camera_points


def prune_gaussians_by_voxel(means, scales, rotations, opacities, colors, voxel_size, depth_scale_factor=None):
    """
    使用voxel方法对Gaussian进行剪枝，合并同一voxel内的Gaussian。

    参考自HunyuanWorld的prune_gs方法，但不使用权重加权，而是直接平均。

    Args:
        means: [N, 3] Gaussian中心位置
        scales: [N, 3] Gaussian缩放
        rotations: [N, 4] Gaussian旋转（四元数）
        opacities: [N] Gaussian不透明度
        colors: [N, K, 3] Gaussian颜色（K为球谐基数或1）
        voxel_size: float voxel大小（metric尺度）
        depth_scale_factor: float 深度缩放因子，用于将norm尺度转换为metric尺度

    Returns:
        pruned_means, pruned_scales, pruned_rotations, pruned_opacities, pruned_colors
    """
    if means.shape[0] == 0:
        return means, scales, rotations, opacities, colors

    # 如果提供了depth_scale_factor，需要将xyz从norm尺度转换到metric尺度
    if depth_scale_factor is not None:
        # depth_scale_factor = 1 / dist_avg，所以metric尺度 = norm尺度 / depth_scale_factor
        effective_voxel_size = voxel_size * depth_scale_factor
    else:
        effective_voxel_size = voxel_size

    # 计算voxel索引
    voxel_indices = (means / effective_voxel_size).floor().long()  # [N, 3]

    # 将3D voxel索引转换为1D索引以便分组
    min_indices = voxel_indices.min(dim=0)[0]
    voxel_indices = voxel_indices - min_indices  # 确保索引从0开始
    max_dims = voxel_indices.max(dim=0)[0] + 1

    # 将3D索引展平为1D
    flat_indices = (voxel_indices[:, 0] * max_dims[1] * max_dims[2] +
                   voxel_indices[:, 1] * max_dims[2] +
                   voxel_indices[:, 2])

    # 找到唯一的voxel
    unique_voxels, inverse_indices = torch.unique(flat_indices, return_inverse=True)
    K = len(unique_voxels)

    device = means.device
    num_sh = colors.shape[1] if colors.ndim == 3 else 1

    # 初始化合并后的Gaussian参数
    merged_means = torch.zeros((K, 3), device=device, dtype=means.dtype)
    merged_scales = torch.zeros((K, 3), device=device, dtype=scales.dtype)
    merged_rotations = torch.zeros((K, 4), device=device, dtype=rotations.dtype)
    merged_opacities = torch.zeros(K, device=device, dtype=opacities.dtype)
    if colors.ndim == 3:
        merged_colors = torch.zeros((K, num_sh, 3), device=device, dtype=colors.dtype)
    else:
        merged_colors = torch.zeros((K, 3), device=device, dtype=colors.dtype)

    # 计算每个voxel内的点数，用于平均
    counts = torch.zeros(K, device=device, dtype=torch.float32)
    counts.scatter_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float32))
    counts = torch.clamp(counts, min=1.0)  # 避免除0

    # 合并means（直接平均）
    for d in range(3):
        merged_means[:, d].scatter_add_(0, inverse_indices, means[:, d])
    merged_means = merged_means / counts.unsqueeze(1)

    # 合并scales（直接平均）
    for d in range(3):
        merged_scales[:, d].scatter_add_(0, inverse_indices, scales[:, d])
    merged_scales = merged_scales / counts.unsqueeze(1)

    # 合并opacities（直接平均）
    merged_opacities.scatter_add_(0, inverse_indices, opacities)
    merged_opacities = merged_opacities / counts

    # 合并colors（直接平均）
    if colors.ndim == 3:
        # [N, num_sh, 3] 格式
        for sh_idx in range(num_sh):
            for d in range(3):
                merged_colors[:, sh_idx, d].scatter_add_(0, inverse_indices, colors[:, sh_idx, d])
        merged_colors = merged_colors / counts.unsqueeze(-1).unsqueeze(-1)
    else:
        # [N, 3] 格式
        for d in range(3):
            merged_colors[:, d].scatter_add_(0, inverse_indices, colors[:, d])
        merged_colors = merged_colors / counts.unsqueeze(1)

    # 合并quaternions（平均后归一化）
    for d in range(4):
        merged_rotations[:, d].scatter_add_(0, inverse_indices, rotations[:, d])
    quat_norms = torch.norm(merged_rotations, dim=1, keepdim=True)
    merged_rotations = merged_rotations / torch.clamp(quat_norms, min=1e-8)

    return merged_means, merged_scales, merged_rotations, merged_opacities, merged_colors


class Stage2RenderLoss(nn.Module):
    """第二阶段渲染损失"""

    def __init__(
        self,
        rgb_weight: float = 1.0,
        depth_weight: float = 0.0,  # 禁用depth loss
        lpips_weight: float = 0.0,  # LPIPS loss权重
        render_only_dynamic: bool = False,  # 是否只渲染动态物体
        sh_degree: int = 0,  # 球谐函数阶数
        enable_voxel_pruning: bool = True,  # 是否启用voxel剪枝
        voxel_size: float = 0.002  # voxel大小（metric尺度，单位米）
    ):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.depth_weight = depth_weight
        self.lpips_weight = lpips_weight
        self.render_only_dynamic = render_only_dynamic
        self.sh_degree = sh_degree
        self.enable_voxel_pruning = enable_voxel_pruning
        self.voxel_size = voxel_size

    def forward(
        self,
        refined_scene: Dict,
        gt_images: torch.Tensor,
        gt_depths: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        sky_masks: Optional[torch.Tensor] = None,
        sky_colors: Optional[torch.Tensor] = None,
        sampled_frame_indices: Optional[torch.Tensor] = None,
        depth_scale_factor: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算第二阶段的渲染损失

        Args:
            refined_scene: 细化后的场景表示
            gt_images: [B, S, 3, H, W] 真实图像
            gt_depths: [B, S, H, W] 真实深度
            intrinsics: [B, S, 3, 3] 相机内参
            extrinsics: [B, S, 4, 4] 相机外参
            sky_masks: [B, S, H, W] 天空区域掩码
            sky_colors: [B, num_frames, 3, H, W] 天空颜色，用于与渲染结果合成
            sampled_frame_indices: [num_frames] 采样帧索引
            depth_scale_factor: 深度缩放因子，用于voxel pruning

        Returns:
            loss_dict: 损失字典
        """
        B, S, C, H, W = gt_images.shape
        device = gt_images.device

        # 收集所有帧的渲染结果
        rendered_images = []
        rendered_depths = []

        # 渲染每一帧
        actual_S = min(S, intrinsics.shape[1], extrinsics.shape[1])

        for frame_idx in range(actual_S):
            frame_intrinsic = intrinsics[0, frame_idx]  # [3, 3]
            frame_extrinsic = extrinsics[0, frame_idx]  # [4, 4]

            # 渲染当前帧
            rendered_rgb, rendered_depth, rendered_alpha = self._render_frame(
                refined_scene, frame_intrinsic, frame_extrinsic, H, W, frame_idx,
                render_only_dynamic=self.render_only_dynamic,
                depth_scale_factor=depth_scale_factor
            )

            # 如果提供了 sky_colors，使用 alpha 与 sky 合成最终图像
            if sky_colors is not None and sampled_frame_indices is not None:
                if not isinstance(sampled_frame_indices, torch.Tensor):
                    sampled_frame_indices = torch.tensor(sampled_frame_indices, device=device)

                matches = (sampled_frame_indices == frame_idx)
                if matches.any():
                    sky_idx = matches.nonzero(as_tuple=True)[0].item()
                    frame_sky_color = sky_colors[0, sky_idx]  # [3, H, W]
                    alpha_3ch = rendered_alpha.unsqueeze(0)  # [1, H, W]
                    rendered_rgb = alpha_3ch * rendered_rgb + (1 - alpha_3ch) * frame_sky_color

            rendered_images.append(rendered_rgb)
            rendered_depths.append(rendered_depth)

        # 如果没有渲染任何帧，返回零损失
        if len(rendered_images) == 0:
            dummy_param = None
            # 尝试从车辆物体获取dummy参数
            if 'dynamic_objects_cars' in refined_scene and len(refined_scene['dynamic_objects_cars']) > 0:
                for obj_data in refined_scene['dynamic_objects_cars']:
                    if 'canonical_gaussians' in obj_data and obj_data['canonical_gaussians'] is not None:
                        dummy_param = obj_data['canonical_gaussians']
                        break

            # 如果车辆没有，尝试从行人物体获取dummy参数
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

        # Stack所有渲染结果
        pred_rgb = torch.stack(rendered_images, dim=0)  # [actual_S, 3, H, W]
        pred_depth = torch.stack(rendered_depths, dim=0)  # [actual_S, H, W]
        gt_rgb = gt_images[0, :actual_S]  # [actual_S, 3, H, W]
        gt_depth = gt_depths[0, :actual_S]  # [actual_S, H, W]

        # 统一计算RGB loss
        rgb_loss = F.l1_loss(pred_rgb, gt_rgb)

        # 计算LPIPS loss
        if self.lpips_weight > 0:
            lpips_loss = compute_lpips(pred_rgb, gt_rgb).mean()
        else:
            lpips_loss = rgb_loss * 0.0

        # 计算Depth loss
        if self.depth_weight > 0:
            valid_depth_mask = gt_depth > 0
            if valid_depth_mask.sum() > 0:
                depth_loss = F.l1_loss(pred_depth[valid_depth_mask], gt_depth[valid_depth_mask])
            else:
                depth_loss = rgb_loss * 0.0
        else:
            depth_loss = rgb_loss * 0.0

        # 构建loss字典
        loss_dict = {
            'stage2_rgb_loss': self.rgb_weight * rgb_loss,
            'stage2_depth_loss': self.depth_weight * depth_loss,
            'stage2_lpips_loss': self.lpips_weight * lpips_loss,
        }
        loss_dict['stage2_total_loss'] = sum(loss_dict.values())

        # 【可视化指标】分别计算sky和non-sky区域的RGB loss（仅用于展示，不参与梯度回传）
        if sky_masks is not None and len(rendered_images) > 0:
            # 使用已stack的tensor，直接计算
            sky_masks_bool = sky_masks[0, :actual_S].bool()  # [actual_S, H, W]

            # 扩展mask到RGB维度
            sky_mask_3ch = sky_masks_bool.unsqueeze(1).expand(-1, 3, -1, -1)  # [actual_S, 3, H, W]
            nonsky_mask_3ch = ~sky_mask_3ch

            # Sky区域loss
            if sky_mask_3ch.any():
                sky_pred = pred_rgb[sky_mask_3ch].detach()
                sky_gt = gt_rgb[sky_mask_3ch].detach()
                loss_dict['stage2_rgb_loss_sky'] = F.l1_loss(sky_pred, sky_gt)

            # Non-sky区域loss
            if nonsky_mask_3ch.any():
                nonsky_pred = pred_rgb[nonsky_mask_3ch].detach()
                nonsky_gt = gt_rgb[nonsky_mask_3ch].detach()
                loss_dict['stage2_rgb_loss_nonsky'] = F.l1_loss(nonsky_pred, nonsky_gt)

        # 检查和修复异常值
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
        render_only_dynamic: bool = False,
        depth_scale_factor: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        渲染单帧图像

        Args:
            refined_scene: 细化后的场景
            intrinsic: [3, 3] 相机内参
            extrinsic: [4, 4] 相机外参
            height, width: 图像尺寸
            frame_idx: 当前帧索引
            sky_colors: [H, W, 3] 天空颜色 (可选)
            render_only_dynamic: 是否只渲染动态物体 (默认False)
            depth_scale_factor: 深度缩放因子，用于voxel pruning

        Returns:
            rendered_rgb: [3, H, W] 渲染的RGB图像
            rendered_depth: [H, W] 渲染的深度图
            rendered_alpha: [H, W] 渲染的不透明度
        """
        device = intrinsic.device

        # 收集所有Gaussian参数
        all_means = []
        all_scales = []
        all_rotations = []
        all_opacities = []
        all_colors = []

        # 静态Gaussian：只有在不是"仅渲染动态物体"模式时才渲染
        if not render_only_dynamic and refined_scene.get('static_gaussians') is not None:
            static_gaussians = refined_scene['static_gaussians']
            if static_gaussians.shape[0] > 0:
                all_means.append(static_gaussians[:, :3])
                all_scales.append(static_gaussians[:, 3:6])
                all_colors.append(static_gaussians[:, 6:9].unsqueeze(-2))
                all_rotations.append(static_gaussians[:, 9:13])
                all_opacities.append(static_gaussians[:, 13])

        # 动态Gaussian：处理车辆（使用canonical空间+变换）
        dynamic_objects_cars = refined_scene.get('dynamic_objects_cars', [])
        for obj_data in dynamic_objects_cars:
            # 检查物体是否在当前帧存在
            if 'frame_transforms' not in obj_data or frame_idx not in obj_data['frame_transforms']:
                continue

            # 获取物体在正规空间(canonical)的Gaussian参数
            canonical_gaussians = obj_data.get(
                'canonical_gaussians')  # [N, 14]
            if canonical_gaussians is None or canonical_gaussians.shape[0] == 0:
                continue

            # 获取从canonical空间到当前帧的变换
            frame_transform = self._get_object_transform_to_frame(
                obj_data, frame_idx)
            if frame_transform is None:
                # 如果没有变换信息，直接使用原始Gaussians（假设当前帧就是参考帧）
                transformed_gaussians = canonical_gaussians
            else:
                # 应用变换：将canonical空间的Gaussians变换到当前帧
                transformed_gaussians = self._apply_transform_to_gaussians(
                    canonical_gaussians, frame_transform
                )

            # 添加到渲染列表
            if transformed_gaussians.shape[0] > 0:
                all_means.append(transformed_gaussians[:, :3])
                all_scales.append(transformed_gaussians[:, 3:6])
                all_colors.append(transformed_gaussians[:, 6:9].unsqueeze(-2))
                all_rotations.append(transformed_gaussians[:, 9:13])
                all_opacities.append(transformed_gaussians[:, 13])

        # 动态Gaussian：处理行人（每帧单独的Gaussians，不使用变换）
        dynamic_objects_people = refined_scene.get('dynamic_objects_people', [])
        for obj_data in dynamic_objects_people:
            # 检查物体是否在当前帧存在
            frame_gaussians = obj_data.get('frame_gaussians', {})
            if frame_idx not in frame_gaussians:
                continue

            # 直接使用当前帧的Gaussians（不进行变换）
            current_frame_gaussians = frame_gaussians[frame_idx]  # [N, 14]
            if current_frame_gaussians is None or current_frame_gaussians.shape[0] == 0:
                continue

            # 添加到渲染列表
            all_means.append(current_frame_gaussians[:, :3])
            all_scales.append(current_frame_gaussians[:, 3:6])
            all_colors.append(current_frame_gaussians[:, 6:9].unsqueeze(-2))
            all_rotations.append(current_frame_gaussians[:, 9:13])
            all_opacities.append(current_frame_gaussians[:, 13])

        if not all_means:
            # 如果没有Gaussian，返回空图像
            return (
                torch.zeros(3, height, width, device=device),
                torch.zeros(height, width, device=device),
                torch.zeros(height, width, device=device)
            )

        # 合并所有Gaussian参数
        means = torch.cat(all_means, dim=0)  # [N, 3]
        scales = torch.cat(all_scales, dim=0)  # [N, 3]
        colors = torch.cat(all_colors, dim=0)  # [N, 1, 3]
        rotations = torch.cat(all_rotations, dim=0)  # [N, 4]
        opacities = torch.cat(all_opacities, dim=0)  # [N]

        # 安全性检查：检测并修复NaN/Inf值
        means = torch.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
        scales = torch.nan_to_num(scales, nan=0.01, posinf=1.0, neginf=0.01)
        colors = torch.nan_to_num(colors, nan=0.5, posinf=1.0, neginf=0.0)
        rotations = torch.nan_to_num(rotations, nan=0.0, posinf=1.0, neginf=-1.0)
        opacities = torch.nan_to_num(opacities, nan=0.5, posinf=1.0, neginf=0.0)

        # 应用voxel pruning（如果启用）
        if self.enable_voxel_pruning and means.shape[0] > 0:
            # 提取depth_scale_factor的数值
            dsf = depth_scale_factor.item() if depth_scale_factor is not None and torch.is_tensor(depth_scale_factor) else depth_scale_factor

            # colors需要保持[N, 1, 3]格式
            colors_squeezed = colors.squeeze(1)  # [N, 3]
            means, scales, rotations, opacities, colors_pruned = prune_gaussians_by_voxel(
                means, scales, rotations, opacities, colors_squeezed,
                voxel_size=self.voxel_size,
                depth_scale_factor=dsf
            )
            colors = colors_pruned.unsqueeze(1)  # [N, 3] -> [N, 1, 3]

        # 准备渲染参数
        viewmat = extrinsic.unsqueeze(0)  # [1, 4, 4]
        K = intrinsic.unsqueeze(0)  # [1, 3, 3]

        # 渲染
        render_result = rasterization(
            means, rotations, scales, opacities, colors,
            viewmat, K, width, height,
            sh_degree=self.sh_degree, render_mode="RGB+ED",
            radius_clip=0, near_plane=0.0001,
            far_plane=1000.0,
            eps2d=0.3,
        )

        # 提取渲染结果
        rendered_image, rendered_alphas, _ = render_result
        rendered_rgb = rendered_image[0, :, :, :3].permute(2, 0, 1)
        rendered_depth = rendered_image[0, :, :, -1]
        rendered_alpha = rendered_alphas[0, :, :, 0] if rendered_alphas is not None else torch.zeros(height, width, device=device)

        return rendered_rgb, rendered_depth, rendered_alpha

    def _get_object_transform_to_frame(self, obj_data: Dict, frame_idx: int) -> Optional[torch.Tensor]:
        """获取从canonical空间到指定帧的变换矩阵"""
        # 检查参考帧
        reference_frame = obj_data.get('reference_frame', 0)
        if frame_idx == reference_frame:
            # 如果要渲染的就是参考帧（canonical帧），不需要变换
            return None

        # 获取变换：frame_transforms存储的是从各帧到reference_frame的变换
        # 但我们需要从reference_frame到frame_idx的变换，所以需要求逆
        if 'frame_transforms' in obj_data:
            frame_transforms = obj_data['frame_transforms']
            if frame_idx in frame_transforms:
                # 存储的变换：frame_idx -> reference_frame
                # 我们需要的变换：reference_frame -> frame_idx（即逆变换）
                frame_to_canonical = frame_transforms[frame_idx]  # [4, 4] 变换矩阵
                try:
                    # 求逆得到从canonical到frame的变换
                    # 转换为float32以支持linalg.inv
                    original_dtype = frame_to_canonical.dtype
                    canonical_to_frame = torch.linalg.inv(frame_to_canonical.float()).to(original_dtype)
                    return canonical_to_frame
                except Exception as e:
                    print(
                        f"Warning: Failed to invert transform for frame {frame_idx}: {e}")
                    return None

        # 如果没有变换信息，返回None
        return None


    def _apply_transform_to_gaussians(
        self,
        gaussians: torch.Tensor,
        transform: torch.Tensor
    ) -> torch.Tensor:
        """将变换应用到Gaussian参数"""

        transformed_gaussians = gaussians.clone()
        positions = gaussians[:, :3]  # [N, 3]
        positions_homo = torch.cat([positions, torch.ones(
            positions.shape[0], 1, device=positions.device)], dim=1)  # [N, 4]
        transformed_positions = torch.mm(
            transform, positions_homo.T).T[:, :3]  # [N, 3]
        transformed_gaussians[:, :3] = transformed_positions

        return transformed_gaussians


class Stage2CompleteLoss(nn.Module):
    """第二阶段完整损失函数"""

    def __init__(
        self,
        render_loss_config: Optional[Dict] = None
    ):
        super().__init__()

        # 默认配置
        if render_loss_config is None:
            render_loss_config = {
                'rgb_weight': 1.0,
                'depth_weight': 0.0,  # 禁用depth loss
            }

        self.render_loss = Stage2RenderLoss(**render_loss_config)

    def forward(
        self,
        refinement_results: Dict,
        refined_scene: Dict,
        gt_images: torch.Tensor,
        gt_depths: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        sky_masks: Optional[torch.Tensor] = None,
        sky_colors: Optional[torch.Tensor] = None,
        sampled_frame_indices: Optional[torch.Tensor] = None,
        depth_scale_factor: Optional[torch.Tensor] = None,
        camera_indices: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算第二阶段的完整损失

        Args:
            refinement_results: Stage2Refiner的输出
            refined_scene: 细化后的场景
            gt_images: 真实图像
            gt_depths: 真实深度
            intrinsics: 相机内参
            extrinsics: 相机外参
            sky_masks: 天空区域掩码
            sky_colors: [B, num_frames, 3, H, W] 天空颜色
            sampled_frame_indices: [num_frames] 采样帧索引
            depth_scale_factor: 深度缩放因子，用于voxel pruning
            camera_indices: [B, S_total] 相机索引，用于多相机模式
            frame_indices: [B, S_total] 帧索引，用于多相机模式

        Returns:
            complete_loss_dict: 完整损失字典
        """
        # 渲染损失
        render_loss_dict = self.render_loss(
            refined_scene, gt_images, gt_depths,
            intrinsics, extrinsics, sky_masks,
            sky_colors, sampled_frame_indices,
            depth_scale_factor
        )

        # 直接使用渲染损失作为完整损失
        complete_loss_dict = render_loss_dict.copy()
        complete_loss_dict['stage2_final_total_loss'] = render_loss_dict['stage2_total_loss']

        return complete_loss_dict
