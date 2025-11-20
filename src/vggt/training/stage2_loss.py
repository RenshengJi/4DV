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

    Args:
        dynamic_objects: 动态物体列表，每个包含frame_pixel_indices
        frame_idx: 当前帧索引
        height, width: 图像尺寸

    Returns:
        mask: [H, W] 布尔mask，True表示动态区域
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

        pixel_indices = frame_pixel_indices[frame_idx]
        if not pixel_indices:
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


class Stage2RenderLoss(nn.Module):
    """第二阶段渲染损失"""

    def __init__(
        self,
        rgb_weight: float = 1.0,
        depth_weight: float = 0.0,  # 禁用depth loss
        lpips_weight: float = 0.0,  # LPIPS loss权重
        render_only_dynamic: bool = False,  # 是否只渲染动态物体
        supervise_only_dynamic: bool = False,  # 是否只监督动态区域
        supervise_middle_frame_only: bool = False,  # 是否只渲染监督中间帧
        sh_degree: int = 0  # 球谐函数阶数
    ):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.depth_weight = depth_weight
        self.lpips_weight = lpips_weight
        self.render_only_dynamic = render_only_dynamic
        self.supervise_only_dynamic = supervise_only_dynamic
        self.supervise_middle_frame_only = supervise_middle_frame_only
        self.sh_degree = sh_degree

    def forward(
        self,
        refined_scene: Dict,
        gt_images: torch.Tensor,
        gt_depths: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        sky_masks: Optional[torch.Tensor] = None,
        original_dynamic_objects: Optional[list] = None,
        sky_colors: Optional[torch.Tensor] = None,
        sampled_frame_indices: Optional[torch.Tensor] = None
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
            original_dynamic_objects: 原始的dynamic_objects（包含frame_pixel_indices）
            sky_colors: [B, num_frames, 3, H, W] 天空颜色，用于与渲染结果合成
            sampled_frame_indices: [num_frames] 采样帧索引

        Returns:
            loss_dict: 损失字典
        """
        B, S, C, H, W = gt_images.shape
        device = gt_images.device

        # 初始化loss为None，后续累加
        total_rgb_loss = None
        total_depth_loss = None
        total_lpips_loss = None
        num_valid_frames = 0

        rendered_images = []
        rendered_depths = []

        # 渲染每一帧 - 确保frame_idx不超过张量的维度
        actual_S = min(S, intrinsics.shape[1], extrinsics.shape[1])

        # 确定要渲染的帧范围
        if self.supervise_middle_frame_only:
            # 只渲染中间帧
            middle_frame_idx = actual_S // 2
            frame_indices = [middle_frame_idx]
        else:
            # 渲染所有帧
            frame_indices = list(range(actual_S))

        for frame_idx in frame_indices:
            frame_intrinsic = intrinsics[0, frame_idx]  # [3, 3]
            frame_extrinsic = extrinsics[0, frame_idx]  # [4, 4]
            frame_gt_image = gt_images[0, frame_idx]  # [3, H, W]
            frame_gt_depth = gt_depths[0, frame_idx]   # [H, W]

            # 渲染当前帧
            rendered_rgb, rendered_depth, rendered_alpha = self._render_frame(
                refined_scene, frame_intrinsic, frame_extrinsic, H, W, frame_idx,
                render_only_dynamic=self.render_only_dynamic  # 使用配置参数
            )

            # 如果提供了 sky_colors，使用 alpha 与 sky 合成最终图像
            if sky_colors is not None and sampled_frame_indices is not None:
                # 找到当前frame_idx对应的sky_colors索引
                # 将 sampled_frame_indices 转换为 tensor（如果还不是）
                if not isinstance(sampled_frame_indices, torch.Tensor):
                    sampled_frame_indices = torch.tensor(sampled_frame_indices, device=device)

                # 检查 frame_idx 是否在采样帧中
                matches = (sampled_frame_indices == frame_idx)
                if matches.any():
                    sky_idx = matches.nonzero(as_tuple=True)[0].item()
                    frame_sky_color = sky_colors[0, sky_idx]  # [3, H, W]

                    # 合成：final_image = alpha * rendered_rgb + (1 - alpha) * sky_colors
                    # rendered_alpha: [H, W], rendered_rgb: [3, H, W], sky_colors: [3, H, W]
                    alpha_3ch = rendered_alpha.unsqueeze(0)  # [1, H, W]
                    rendered_rgb = alpha_3ch * rendered_rgb + (1 - alpha_3ch) * frame_sky_color

            rendered_images.append(rendered_rgb)
            rendered_depths.append(rendered_depth)

            # 计算监督区域的mask - 不再使用sky_masks进行过滤
            if self.supervise_only_dynamic:
                # 创建dynamic mask：只监督动态物体区域
                # 使用原始的dynamic_objects（包含frame_pixel_indices），而不是refined_scene中的
                objects_for_mask = original_dynamic_objects if original_dynamic_objects is not None else refined_scene.get('dynamic_objects', [])
                dynamic_mask = create_dynamic_mask_from_pixel_indices(
                    objects_for_mask, frame_idx, H, W
                )
                mask = dynamic_mask
            else:
                # 监督所有区域（包括sky）
                mask = torch.ones_like(frame_gt_depth, dtype=torch.bool)

            # RGB损失 - 动态区域应该匹配GT
            if mask.sum() > 0:  # 确保有有效像素
                rgb_loss = F.l1_loss(
                    rendered_rgb[mask.unsqueeze(0).repeat(3, 1, 1)],
                    frame_gt_image[mask.unsqueeze(0).repeat(3, 1, 1)]
                )
                # 累加loss
                if total_rgb_loss is None:
                    total_rgb_loss = rgb_loss
                else:
                    total_rgb_loss = total_rgb_loss + rgb_loss
                num_valid_frames += 1

            # LPIPS损失 - 在完整图像上计算（包括sky）
            if self.lpips_weight > 0:
                # compute_lpips 需要 [B, C, H, W] 格式
                rendered_rgb_batch = rendered_rgb.unsqueeze(0)  # [1, 3, H, W]
                gt_rgb_batch = frame_gt_image.unsqueeze(0)  # [1, 3, H, W]

                lpips_loss = compute_lpips(rendered_rgb_batch, gt_rgb_batch).mean()

                # 累加loss
                if total_lpips_loss is None:
                    total_lpips_loss = lpips_loss
                else:
                    total_lpips_loss = total_lpips_loss + lpips_loss

            # 深度损失 - 仅在权重大于0时计算
            if self.depth_weight > 0:
                # 确保rendered_depth有正确的维度
                if rendered_depth.dim() == 1:
                    print(
                        f"Debug: rendered_depth has wrong shape {rendered_depth.shape}, expected 2D. Skipping depth loss.")
                    continue  # Skip this frame if depth has wrong dimensions

                depth_mask = mask & (frame_gt_depth > 0)  # 排除无效深度
                if depth_mask.sum() > 0:
                    depth_loss = F.l1_loss(
                        rendered_depth[depth_mask],
                        frame_gt_depth[depth_mask]
                    )
                    # 累加loss
                    if total_depth_loss is None:
                        total_depth_loss = depth_loss
                    else:
                        total_depth_loss = total_depth_loss + depth_loss

        # 平均损失并确保有grad_fn
        # 如果没有有效的dynamic区域，需要创建一个dummy loss以保持梯度流
        if total_rgb_loss is None:
            # 没有任何dynamic像素，创建一个与模型参数连接的零loss
            # 这样backward不会报错，但梯度为零
            # 从refined_scene中获取一个参数作为anchor
            dummy_param = None
            if 'dynamic_objects' in refined_scene and len(refined_scene['dynamic_objects']) > 0:
                for obj_data in refined_scene['dynamic_objects']:
                    if 'canonical_gaussians' in obj_data and obj_data['canonical_gaussians'] is not None:
                        dummy_param = obj_data['canonical_gaussians']
                        break

            if dummy_param is not None and dummy_param.requires_grad:
                # 创建一个连接到模型参数的零loss
                total_rgb_loss = (dummy_param.sum() * 0.0)
            else:
                # 如果找不到参数，使用requires_grad的tensor
                total_rgb_loss = torch.zeros(1, device=device, requires_grad=True).sum()

            num_valid_frames = max(S, 1)

        # 构建loss字典
        avg_rgb_loss = total_rgb_loss / num_valid_frames if num_valid_frames > 0 else total_rgb_loss

        loss_dict = {
            'stage2_rgb_loss': self.rgb_weight * avg_rgb_loss,
        }

        # Depth loss
        if self.depth_weight > 0 and total_depth_loss is not None:
            avg_depth_loss = total_depth_loss / num_valid_frames if num_valid_frames > 0 else total_depth_loss
            loss_dict['stage2_depth_loss'] = self.depth_weight * avg_depth_loss
        else:
            # 使用rgb_loss作为anchor创建零depth loss
            loss_dict['stage2_depth_loss'] = avg_rgb_loss * 0.0

        # LPIPS loss
        if self.lpips_weight > 0 and total_lpips_loss is not None:
            avg_lpips_loss = total_lpips_loss / num_valid_frames if num_valid_frames > 0 else total_lpips_loss
            loss_dict['stage2_lpips_loss'] = self.lpips_weight * avg_lpips_loss
        else:
            # 使用rgb_loss作为anchor创建零lpips loss
            loss_dict['stage2_lpips_loss'] = avg_rgb_loss * 0.0

        # 总损失
        total_loss = sum(loss_dict.values())
        loss_dict['stage2_total_loss'] = total_loss

        # 【可视化指标】分别计算sky和non-sky区域的RGB loss（仅用于展示，不参与梯度回传）
        if sky_masks is not None and len(rendered_images) > 0:
            vis_rgb_loss_sky = None
            vis_rgb_loss_nonsky = None
            vis_num_sky_frames = 0
            vis_num_nonsky_frames = 0

            # 确定要计算的帧范围（与forward中的frame_indices保持一致）
            if self.supervise_middle_frame_only:
                middle_frame_idx = actual_S // 2
                vis_frame_indices = [middle_frame_idx]
            else:
                vis_frame_indices = list(range(actual_S))

            for i, frame_idx in enumerate(vis_frame_indices):
                if i >= len(rendered_images):
                    break

                frame_gt_image = gt_images[0, frame_idx]  # [3, H, W]
                rendered_rgb = rendered_images[i]  # [3, H, W]
                frame_sky_mask = sky_masks[0, frame_idx].bool()  # [H, W], ensure boolean type

                # Sky区域
                if frame_sky_mask.sum() > 0:
                    # 使用masked_select提取sky区域的像素
                    sky_mask_3ch = frame_sky_mask.unsqueeze(0).expand(3, -1, -1)  # [3, H, W]
                    rendered_sky_pixels = rendered_rgb[sky_mask_3ch]  # [N_sky_pixels]
                    gt_sky_pixels = frame_gt_image[sky_mask_3ch]  # [N_sky_pixels]

                    rgb_loss_sky = F.l1_loss(
                        rendered_sky_pixels.detach(),
                        gt_sky_pixels.detach()
                    )
                    if vis_rgb_loss_sky is None:
                        vis_rgb_loss_sky = rgb_loss_sky
                    else:
                        vis_rgb_loss_sky = vis_rgb_loss_sky + rgb_loss_sky
                    vis_num_sky_frames += 1

                # Non-sky区域
                nonsky_mask = ~frame_sky_mask
                if nonsky_mask.sum() > 0:
                    # 使用masked_select提取non-sky区域的像素
                    nonsky_mask_3ch = nonsky_mask.unsqueeze(0).expand(3, -1, -1)  # [3, H, W]
                    rendered_nonsky_pixels = rendered_rgb[nonsky_mask_3ch]  # [N_nonsky_pixels]
                    gt_nonsky_pixels = frame_gt_image[nonsky_mask_3ch]  # [N_nonsky_pixels]

                    rgb_loss_nonsky = F.l1_loss(
                        rendered_nonsky_pixels.detach(),
                        gt_nonsky_pixels.detach()
                    )
                    if vis_rgb_loss_nonsky is None:
                        vis_rgb_loss_nonsky = rgb_loss_nonsky
                    else:
                        vis_rgb_loss_nonsky = vis_rgb_loss_nonsky + rgb_loss_nonsky
                    vis_num_nonsky_frames += 1

            # 添加平均后的可视化指标到loss_dict
            if vis_rgb_loss_sky is not None and vis_num_sky_frames > 0:
                loss_dict['stage2_rgb_loss_sky'] = (vis_rgb_loss_sky / vis_num_sky_frames).detach()
            if vis_rgb_loss_nonsky is not None and vis_num_nonsky_frames > 0:
                loss_dict['stage2_rgb_loss_nonsky'] = (vis_rgb_loss_nonsky / vis_num_nonsky_frames).detach()

        # 检查和修复异常值
        for key, value in loss_dict.items():
            loss_dict[key] = check_and_fix_inf_nan(value, key)

        return loss_dict
    
    def render_refined_scene(
        self,
        refined_scene: Dict,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_height: int,
        image_width: int,
        sky_colors: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        仅渲染细化后的场景，返回rendered_images和rendered_depths

        Args:
            refined_scene: 细化后的场景表示
            intrinsics: [B, S, 3, 3] 相机内参
            extrinsics: [B, S, 4, 4] 相机外参
            image_height: 图像高度
            image_width: 图像宽度
            sky_colors: [S, H, W, 3] 天空颜色，用于替换低opacity区域 (可选)

        Returns:
            rendered_images: List[torch.Tensor] 渲染的RGB图像列表
            rendered_depths: List[torch.Tensor] 渲染的深度图列表
            rendered_alphas: List[torch.Tensor] 渲染的不透明度列表
        """
        S = intrinsics.shape[1]
        rendered_images = []
        rendered_depths = []
        rendered_alphas = []

        # 渲染每一帧
        for frame_idx in range(S):
            frame_intrinsic = intrinsics[0, frame_idx]  # [3, 3]
            frame_extrinsic = extrinsics[0, frame_idx]  # [4, 4]

            # 获取当前帧的天空颜色
            frame_sky_colors = None
            if sky_colors is not None:
                frame_sky_colors = sky_colors[frame_idx]  # [H, W, 3]

            # 渲染当前帧
            rendered_rgb, rendered_depth, rendered_alpha = self._render_frame(
                refined_scene, frame_intrinsic, frame_extrinsic,
                image_height, image_width, frame_idx, frame_sky_colors
            )
            rendered_images.append(rendered_rgb)
            rendered_depths.append(rendered_depth)
            rendered_alphas.append(rendered_alpha)

        return rendered_images, rendered_depths, rendered_alphas

    def _render_frame(
        self,
        refined_scene: Dict,
        intrinsic: torch.Tensor,
        extrinsic: torch.Tensor,
        height: int,
        width: int,
        frame_idx: int = 0,
        sky_colors: Optional[torch.Tensor] = None,
        render_only_dynamic: bool = False
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

        # 动态Gaussian：只处理在当前帧存在的物体
        dynamic_objects_data = refined_scene.get('dynamic_objects', [])
        for obj_data in dynamic_objects_data:
            # 检查物体是否在当前帧存在
            if not self._object_exists_in_frame(obj_data, frame_idx):
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
        scales = torch.nan_to_num(scales, nan=0.01, posinf=1.0, neginf=0.01) #.clamp(min=0.001, max=10.0)
        colors = torch.nan_to_num(colors, nan=0.5, posinf=1.0, neginf=0.0) #.clamp(min=0.0, max=1.0)
        # rotations = F.normalize(torch.nan_to_num(rotations, nan=0.0, posinf=1.0, neginf=-1.0), p=2, dim=-1)
        rotations = torch.nan_to_num(rotations, nan=0.0, posinf=1.0, neginf=-1.0)
        opacities = torch.nan_to_num(opacities, nan=0.5, posinf=1.0, neginf=0.0) #.clamp(min=0.0, max=1.0)

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

        # Extract RGB: [1, H, W, C] -> [3, H, W]
        rendered_rgb = rendered_image[0, :, :, :3].permute(2, 0, 1)

        # Extract depth: [1, H, W, C] -> [H, W]
        rendered_depth = rendered_image[0, :, :, -1]

        # Extract alpha: [1, H, W, 1] -> [H, W]
        rendered_alpha = rendered_alphas[0, :, :, 0] if rendered_alphas is not None else torch.zeros(height, width, device=device)

        return rendered_rgb, rendered_depth, rendered_alpha

    def _apply_sky_replacement(
        self,
        rendered_rgb: torch.Tensor,
        sky_colors: torch.Tensor,
        rendered_alphas: torch.Tensor,
        opacity_threshold: float = 0.01
    ) -> torch.Tensor:
        """
        对渲染图像的低opacity区域应用天空颜色替换

        Args:
            rendered_rgb: [3, H, W] 渲染的RGB图像
            sky_colors: [H, W, 3] 天空颜色
            rendered_alphas: [..., C, H, W, 1] 渲染的alpha值
            opacity_threshold: opacity阈值，小于此值的区域将被替换

        Returns:
            torch.Tensor: [3, H, W] 替换天空后的RGB图像
        """
        try:
            # 提取alpha通道 [..., C, H, W, 1] -> [H, W]
            if len(rendered_alphas.shape) == 5:
                alpha = rendered_alphas[0, 0, :, :, 0]  # [H, W]
            elif len(rendered_alphas.shape) == 4:
                alpha = rendered_alphas[0, :, :, 0]  # [H, W]
            elif len(rendered_alphas.shape) == 3:
                alpha = rendered_alphas[:, :, 0]  # [H, W]
            else:
                alpha = rendered_alphas.squeeze()  # 尝试去掉多余维度

            # 确保alpha是[H, W]格式
            if alpha.dim() != 2:
                print(f"Warning: Unexpected alpha dimensions: {alpha.shape}")
                return rendered_rgb

            # 创建低opacity掩码
            low_opacity_mask = alpha < opacity_threshold  # [H, W]

            # 如果没有低opacity区域，直接返回原图
            if not low_opacity_mask.any():
                return rendered_rgb

            # 转换天空颜色格式 [H, W, 3] -> [3, H, W]
            sky_colors_chw = sky_colors.permute(2, 0, 1)  # [3, H, W]

            # 确保在同一设备上
            sky_colors_chw = sky_colors_chw.to(rendered_rgb.device)
            low_opacity_mask = low_opacity_mask.to(rendered_rgb.device)

            # 应用天空替换：在低opacity区域使用天空颜色
            result = rendered_rgb.clone()
            result[:, low_opacity_mask] = sky_colors_chw[:, low_opacity_mask]

            # 统计替换的像素数量
            replaced_pixels = low_opacity_mask.sum().item()
            total_pixels = low_opacity_mask.numel()
            replacement_ratio = replaced_pixels / total_pixels
            print(f"  天空替换: {replaced_pixels}/{total_pixels} ({replacement_ratio:.3f}) 像素被替换 (opacity < {opacity_threshold})")

            return result

        except Exception as e:
            print(f"Warning: Sky replacement failed: {e}")
            return rendered_rgb

    def _object_exists_in_frame(self, obj_data: Dict, frame_idx: int) -> bool:
        """检查动态物体是否在指定帧中存在"""
        if 'frame_transforms' in obj_data:
            frame_transforms = obj_data['frame_transforms']
            if frame_idx in frame_transforms:
                return True

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

        # 方法2：从pose deltas构建变换（如果有的话）
        if 'pose_deltas' in obj_data and obj_data['pose_deltas']:
            pose_deltas = obj_data['pose_deltas']
            if frame_idx < len(pose_deltas):
                # [6] - rotation + translation
                pose_delta = pose_deltas[frame_idx]
                # 这里假设pose_delta也是从frame到canonical的增量
                frame_to_canonical_delta = self._pose_delta_to_transform_matrix(
                    pose_delta)
                try:
                    # 转换为float32以支持linalg.inv
                    original_dtype = frame_to_canonical_delta.dtype
                    canonical_to_frame = torch.linalg.inv(
                        frame_to_canonical_delta.float()).to(original_dtype)
                    return canonical_to_frame
                except Exception:
                    return None

        # 如果没有变换信息，返回None
        return None

    def _pose_delta_to_transform_matrix(self, pose_delta: torch.Tensor) -> torch.Tensor:
        """
        将pose delta转换为4x4变换矩阵（使用6D旋转表示）

        Args:
            pose_delta: [9] - [6D rotation(6), translation(3)]

        Returns:
            transform: [4, 4] 变换矩阵
        """
        device = pose_delta.device

        # 提取6D旋转和平移
        rotation_6d = pose_delta[:6]  # [6]
        translation = pose_delta[6:9]  # [3]

        # 【安全性检查】如果pose_delta接近零（训练初期常见），返回单位矩阵
        if torch.abs(rotation_6d).max() < 1e-6:
            transform = torch.eye(4, device=device, dtype=pose_delta.dtype)
            transform[:3, 3] = translation  # 仍然应用平移
            return transform

        # 6D旋转 → 旋转矩阵 (Gram-Schmidt正交化)
        a1 = rotation_6d[:3]  # [3]
        a2 = rotation_6d[3:6]  # [3]

        # 【安全性检查】检查a1是否为零向量
        a1_norm = torch.norm(a1)
        if a1_norm < 1e-8:
            # 如果a1接近零，使用单位矩阵
            transform = torch.eye(4, device=device, dtype=pose_delta.dtype)
            transform[:3, 3] = translation
            return transform

        # Gram-Schmidt正交化
        import torch.nn.functional as F
        b1 = F.normalize(a1, dim=0, eps=1e-8)

        dot_product = (b1 * a2).sum()
        b2 = a2 - dot_product * b1

        # 【安全性检查】检查b2是否为零向量（a1和a2共线）
        b2_norm = torch.norm(b2)
        if b2_norm < 1e-8:
            # 如果a1和a2共线，使用单位矩阵
            transform = torch.eye(4, device=device, dtype=pose_delta.dtype)
            transform[:3, 3] = translation
            return transform

        b2 = F.normalize(b2, dim=0, eps=1e-8)

        b3 = torch.cross(b1, b2, dim=0)

        # 构建旋转矩阵
        R = torch.stack([b1, b2, b3], dim=1)  # [3, 3]

        # 构建4x4变换矩阵
        transform = torch.eye(4, device=device, dtype=pose_delta.dtype)
        transform[:3, :3] = R
        transform[:3, 3] = translation

        return transform

    def _apply_transform_to_gaussians(
        self,
        gaussians: torch.Tensor,
        transform: torch.Tensor
    ) -> torch.Tensor:
        """将变换应用到Gaussian参数"""
        # gaussians: [N, 14] - [xyz(3), scale(3), color(3), quat(4), opacity(1)]
        # transform: [4, 4] 变换矩阵

        # 检查变换矩阵是否异常
        if torch.allclose(transform, torch.zeros_like(transform), atol=1e-6):
            print(f"⚠️  检测到零变换矩阵！这会导致所有点聚集到原点形成大白球！")
            print(f"变换矩阵:\n{transform}")
            # 使用单位矩阵替代
            print(f"使用单位矩阵替代异常变换")
            transform = torch.eye(4, dtype=transform.dtype,
                                  device=transform.device)
        else:
            # 转换为float32以支持torch.det
            det_val = torch.det(transform[:3, :3].float()).abs()
            if det_val < 1e-8:
                print(f"⚠️  变换矩阵奇异(det={det_val:.2e})！")
                print(f"变换矩阵:\n{transform}")

        transformed_gaussians = gaussians.clone()

        # 变换位置
        positions = gaussians[:, :3]  # [N, 3]
        positions_homo = torch.cat([positions, torch.ones(
            positions.shape[0], 1, device=positions.device)], dim=1)  # [N, 4]
        transformed_positions = torch.mm(
            transform, positions_homo.T).T[:, :3]  # [N, 3]
        transformed_gaussians[:, :3] = transformed_positions

        return transformed_gaussians

    def _rotation_matrix_to_quaternion(self, R: torch.Tensor) -> torch.Tensor:
        """将旋转矩阵转换为四元数"""
        # R: [3, 3] 旋转矩阵
        # 返回: [4] 四元数 [w, x, y, z]

        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0:
            s = torch.sqrt(trace + 1.0) * 2  # s = 4 * w
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * x
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * y
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * z
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return torch.tensor([w, x, y, z], device=R.device)


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
        original_dynamic_objects: Optional[list] = None,
        sky_colors: Optional[torch.Tensor] = None,
        sampled_frame_indices: Optional[torch.Tensor] = None
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
            original_dynamic_objects: 原始的dynamic_objects（包含frame_pixel_indices）
            sky_colors: [B, num_frames, 3, H, W] 天空颜色
            sampled_frame_indices: [num_frames] 采样帧索引

        Returns:
            complete_loss_dict: 完整损失字典
        """
        # 渲染损失
        render_loss_dict = self.render_loss(
            refined_scene, gt_images, gt_depths,
            intrinsics, extrinsics, sky_masks,
            original_dynamic_objects,
            sky_colors, sampled_frame_indices
        )

        # 直接使用渲染损失作为完整损失
        complete_loss_dict = render_loss_dict.copy()
        complete_loss_dict['stage2_final_total_loss'] = render_loss_dict['stage2_total_loss']

        return complete_loss_dict
