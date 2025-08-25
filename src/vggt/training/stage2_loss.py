# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import sys
import os

# 添加必要的路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dust3r/utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from dust3r.utils.metrics import compute_lpips
from gsplat.rendering import rasterization
from dust3r.utils.misc import tf32_off


def check_and_fix_inf_nan(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """检查并修复tensor中的inf和nan值"""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"Warning: {name} contains inf or nan values, replacing with 0")
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor


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
        uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1).float()  # [H, W, 3]
        uv1 = uv1.unsqueeze(0).expand(N, -1, -1, -1)  # [N, H, W, 3]

        # 转换为相机坐标
        depth = depth.squeeze(-1)  # [N, H, W]
        intrinsic = intrinsic.squeeze(0)  # [N, 3, 3]

        # 计算相机坐标
        camera_points = torch.einsum('nij,nhwj->nhwi', torch.inverse(intrinsic), uv1)  # [N, H, W, 3]
        camera_points = camera_points * depth.unsqueeze(-1)  # [N, H, W, 3]

    return camera_points


class Stage2RenderLoss(nn.Module):
    """第二阶段渲染损失"""
    
    def __init__(
        self,
        rgb_weight: float = 1.0,
        depth_weight: float = 0.0,  # 禁用depth loss
        lpips_weight: float = 0.1,
        consistency_weight: float = 0.0  # 禁用consistency loss
    ):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.depth_weight = depth_weight
        self.lpips_weight = lpips_weight
        self.consistency_weight = consistency_weight
        
    def forward(
        self,
        refined_scene: Dict,
        gt_images: torch.Tensor,
        gt_depths: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        frame_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算第二阶段的渲染损失
        
        Args:
            refined_scene: 细化后的场景表示
            gt_images: [B, S, 3, H, W] 真实图像
            gt_depths: [B, S, H, W] 真实深度
            intrinsics: [B, S, 3, 3] 相机内参
            extrinsics: [B, S, 4, 4] 相机外参  
            frame_masks: [B, S, H, W] 有效区域掩码
            
        Returns:
            loss_dict: 损失字典
        """
        B, S, C, H, W = gt_images.shape
        device = gt_images.device
        
        total_rgb_loss = torch.tensor(0.0, device=device)
        total_depth_loss = torch.tensor(0.0, device=device)
        total_lpips_loss = torch.tensor(0.0, device=device)
        total_consistency_loss = torch.tensor(0.0, device=device)
        
        rendered_images = []
        rendered_depths = []
        
        # 渲染每一帧 - 确保frame_idx不超过张量的维度
        actual_S = min(S, intrinsics.shape[1], extrinsics.shape[1])
        for frame_idx in range(actual_S):
            frame_intrinsic = intrinsics[0, frame_idx]  # [3, 3]
            frame_extrinsic = extrinsics[0, frame_idx]  # [4, 4]
            frame_gt_image = gt_images[0, frame_idx]  # [3, H, W]
            frame_gt_depth = gt_depths[0, frame_idx]   # [H, W]
            
            # 渲染当前帧
            rendered_rgb, rendered_depth = self._render_frame(
                refined_scene, frame_intrinsic, frame_extrinsic, H, W, frame_idx
            )
            
            rendered_images.append(rendered_rgb)
            rendered_depths.append(rendered_depth)
            
            # 计算frame mask
            if frame_masks is not None:
                mask = frame_masks[0, frame_idx]  # [H, W]
            else:
                mask = torch.ones_like(frame_gt_depth, dtype=torch.bool)
            
            # RGB损失
            rgb_loss = F.l1_loss(
                rendered_rgb[mask.unsqueeze(0).repeat(3, 1, 1)], 
                frame_gt_image[mask.unsqueeze(0).repeat(3, 1, 1)]
            )
            total_rgb_loss += rgb_loss
            
            # 深度损失 - 仅在权重大于0时计算
            if self.depth_weight > 0:
                # 确保rendered_depth有正确的维度
                if rendered_depth.dim() == 1:
                    print(f"Debug: rendered_depth has wrong shape {rendered_depth.shape}, expected 2D. Skipping depth loss.")
                    continue  # Skip this frame if depth has wrong dimensions
                
                depth_mask = mask & (frame_gt_depth > 0)  # 排除无效深度
                if depth_mask.sum() > 0:
                    depth_loss = F.l1_loss(
                        rendered_depth[depth_mask], 
                        frame_gt_depth[depth_mask]
                    )
                    total_depth_loss += depth_loss
        
        # LPIPS损失（批量计算）
        if len(rendered_images) > 0:
            rendered_stack = torch.stack(rendered_images, dim=0)  # [S, 3, H, W]
            gt_stack = gt_images[0]  # [S, 3, H, W]
            
            lpips_loss = compute_lpips(rendered_stack, gt_stack).mean()
            total_lpips_loss = lpips_loss
        
        # 一致性损失 - 仅在权重大于0时计算
        if self.consistency_weight > 0:
            consistency_loss = self._compute_consistency_loss(rendered_images, rendered_depths)
            total_consistency_loss = consistency_loss
        
        # 平均损失
        num_frames = max(S, 1)
        
        loss_dict = {
            'stage2_rgb_loss': self.rgb_weight * (total_rgb_loss / num_frames),
            'stage2_depth_loss': torch.tensor(0.0, device=device) if self.depth_weight == 0 else self.depth_weight * (total_depth_loss / num_frames),
            'stage2_lpips_loss': self.lpips_weight * total_lpips_loss,
            'stage2_consistency_loss': torch.tensor(0.0, device=device) if self.consistency_weight == 0 else self.consistency_weight * total_consistency_loss,
        }
        
        # 总损失
        total_loss = sum(loss_dict.values())
        loss_dict['stage2_total_loss'] = total_loss
        
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
        frame_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        渲染单帧图像
        
        Args:
            refined_scene: 细化后的场景
            intrinsic: [3, 3] 相机内参
            extrinsic: [4, 4] 相机外参
            height, width: 图像尺寸
            frame_idx: 当前帧索引
            
        Returns:
            rendered_rgb: [3, H, W] 渲染的RGB图像
            rendered_depth: [H, W] 渲染的深度图
        """
        device = intrinsic.device
        
        # 收集所有Gaussian参数
        all_means = []
        all_scales = []  
        all_rotations = []
        all_opacities = []
        all_colors = []
        
        # # 静态Gaussian：所有帧都使用完整的静态背景
        # if refined_scene.get('static_gaussians') is not None:
        #     static_gaussians = refined_scene['static_gaussians']
        #     if static_gaussians.shape[0] > 0:
        #         all_means.append(static_gaussians[:, :3])
        #         all_scales.append(static_gaussians[:, 3:6])
        #         all_colors.append(static_gaussians[:, 6:9].unsqueeze(-2))
        #         all_rotations.append(static_gaussians[:, 9:13])
        #         all_opacities.append(static_gaussians[:, 13])
        
        # 动态Gaussian：只处理在当前帧存在的物体
        dynamic_objects_data = refined_scene.get('dynamic_objects', [])
        for obj_data in dynamic_objects_data:
            # 检查物体是否在当前帧存在
            if not self._object_exists_in_frame(obj_data, frame_idx):
                continue
            
            # 获取物体在正规空间(canonical)的Gaussian参数
            canonical_gaussians = obj_data.get('canonical_gaussians')  # [N, 14]
            if canonical_gaussians is None or canonical_gaussians.shape[0] == 0:
                continue
            
            # 获取从canonical空间到当前帧的变换
            frame_transform = self._get_object_transform_to_frame(obj_data, frame_idx)
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
                torch.zeros(height, width, device=device)
            )
        
        # 合并所有Gaussian参数
        means = torch.cat(all_means, dim=0)  # [N, 3]
        scales = torch.cat(all_scales, dim=0)  # [N, 3]  
        colors = torch.cat(all_colors, dim=0)  # [N, 1, 3]
        rotations = torch.cat(all_rotations, dim=0)  # [N, 4]
        opacities = torch.cat(all_opacities, dim=0)  # [N]
        
        
        # Safety check: 防止内存爆炸
        max_gaussians = 500000  # 最大Gaussian数量限制
        if means.shape[0] > max_gaussians:
            # print(f"Warning: Too many Gaussians ({means.shape[0]}), clamping to {max_gaussians}")
            means = means[:max_gaussians]
            scales = scales[:max_gaussians]
            colors = colors[:max_gaussians]
            rotations = rotations[:max_gaussians]
            opacities = opacities[:max_gaussians]
        
        # 注意：Gaussian参数已经在OnlineDynamicProcessor中经过激活处理，这里不需要再次激活
        # 如果再次激活会导致参数异常（如双重sigmoid导致不透明度过低）
        
        # # 处理参数 - 已注释，因为参数已经激活过
        # scales = (0.05 * torch.exp(scales)).clamp_max(0.3)
        
        # # 归一化旋转四元数 - 已注释，因为已经归一化过
        # rotation_norms = torch.norm(rotations, dim=-1, keepdim=True)
        # rotation_norms = torch.clamp(rotation_norms, min=1e-8)
        # rotations = rotations / rotation_norms
        
        # opacities = torch.sigmoid(opacities)  # 已注释，因为已经经过sigmoid
        
        # 准备渲染参数
        viewmat = extrinsic.unsqueeze(0)  # [1, 4, 4]
        K = intrinsic.unsqueeze(0)  # [1, 3, 3]
        
        try:
            # 渲染
            render_result = rasterization(
                means, rotations, scales, opacities, colors,
                viewmat, K, width, height,
                sh_degree=0, render_mode="RGB+ED",
                radius_clip=0, near_plane=0.0001,
                far_plane=1000.0,
                eps2d=0.3,
            )
            
            # Check if rendering returned valid results
            if render_result is None or len(render_result) < 3:
                print(f"Rendering returned invalid result: {render_result}")
                print(f"Debug render params: means.shape={means.shape}, rotations.shape={rotations.shape}, scales.shape={scales.shape}")
                print(f"Debug render params: opacities.shape={opacities.shape}, colors.shape={colors.shape}")
                print(f"Debug render params: viewmat.shape={viewmat.shape}, K.shape={K.shape}, width={width}, height={height}")
                # Check for NaN or invalid values
                print(f"Debug NaN check: means has NaN: {torch.isnan(means).any()}")
                print(f"Debug NaN check: rotations has NaN: {torch.isnan(rotations).any()}")
                print(f"Debug NaN check: scales has NaN: {torch.isnan(scales).any()}")
                print(f"Debug NaN check: opacities has NaN: {torch.isnan(opacities).any()}")
                print(f"Debug NaN check: colors has NaN: {torch.isnan(colors).any()}")
                # Check parameter ranges
                print(f"Debug ranges: means min/max: {means.min():.6f}/{means.max():.6f}")
                print(f"Debug ranges: scales min/max: {scales.min():.6f}/{scales.max():.6f}")
                print(f"Debug ranges: opacities min/max: {opacities.min():.6f}/{opacities.max():.6f}")
                rendered_rgb = torch.zeros(3, height, width, device=device)
                rendered_depth = torch.zeros(height, width, device=device)
            else:
                rendered_image, _, rendered_depth_raw = render_result
                
                if rendered_image is None or rendered_depth_raw is None:
                    print(f"Rendering returned None results: image={rendered_image is not None}, depth={rendered_depth_raw is not None}")
                    print(f"Debug render params: means.shape={means.shape}, rotations.shape={rotations.shape}, scales.shape={scales.shape}")
                    rendered_rgb = torch.zeros(3, height, width, device=device)
                    rendered_depth = torch.zeros(height, width, device=device)
                else:
                    # Debug: Check rendered_image and rendered_depth_raw
                    print(f"Debug: rendered_image type: {type(rendered_image)}, shape: {rendered_image.shape if hasattr(rendered_image, 'shape') else 'No shape'}")
                    print(f"Debug: rendered_depth_raw type: {type(rendered_depth_raw)}, shape: {rendered_depth_raw.shape if hasattr(rendered_depth_raw, 'shape') else 'No shape'}")
                    
                    try:
                        rendered_rgb = rendered_image[0, ..., :3].permute(2, 0, 1)  # [3, H, W]
                        print(f"Debug: Successfully extracted rendered_rgb, shape: {rendered_rgb.shape}")
                    except Exception as e:
                        print(f"Debug: Failed to extract rendered_rgb: {e}")
                        rendered_rgb = torch.zeros(3, height, width, device=device)
                    
                    try:
                        # Check if rendered_depth_raw is a dictionary (from gsplat.rasterization)
                        if isinstance(rendered_depth_raw, dict):
                            # Try common keys for depth information
                            if 'depths' in rendered_depth_raw:
                                rendered_depth = rendered_depth_raw['depths']
                                # Handle different depth tensor shapes
                                if rendered_depth.dim() == 3 and rendered_depth.shape[0] == 1:
                                    rendered_depth = rendered_depth[0]  # Remove batch dimension
                                elif rendered_depth.dim() == 1:
                                    # 1D depth tensor - need to reshape or create zero tensor
                                    print(f"Debug: 1D depth tensor with shape {rendered_depth.shape}, creating zero depth")
                                    rendered_depth = torch.zeros(height, width, device=device)
                                elif rendered_depth.dim() == 0:
                                    # Scalar depth - create zero tensor
                                    print(f"Debug: Scalar depth, creating zero depth")
                                    rendered_depth = torch.zeros(height, width, device=device)
                            elif 'depth' in rendered_depth_raw:
                                rendered_depth = rendered_depth_raw['depth']
                                if rendered_depth.dim() == 3 and rendered_depth.shape[0] == 1:
                                    rendered_depth = rendered_depth[0]  # Remove batch dimension
                                elif rendered_depth.dim() <= 1:
                                    rendered_depth = torch.zeros(height, width, device=device)
                            elif 'expected_depth' in rendered_depth_raw:
                                rendered_depth = rendered_depth_raw['expected_depth']
                                if rendered_depth.dim() == 3 and rendered_depth.shape[0] == 1:
                                    rendered_depth = rendered_depth[0]  # Remove batch dimension
                                elif rendered_depth.dim() <= 1:
                                    rendered_depth = torch.zeros(height, width, device=device)
                            else:
                                print(f"Debug: Unknown depth dictionary keys: {list(rendered_depth_raw.keys())}")
                                rendered_depth = torch.zeros(height, width, device=device)
                        else:
                            # Assume it's a tensor
                            if rendered_depth_raw.dim() >= 2:
                                rendered_depth = rendered_depth_raw[0]  # [H, W]
                            else:
                                rendered_depth = torch.zeros(height, width, device=device)
                        
                        # Final safety check
                        if rendered_depth.dim() != 2:
                            print(f"Debug: Final depth check failed, shape: {rendered_depth.shape}, creating zero depth")
                            rendered_depth = torch.zeros(height, width, device=device)
                        
                        print(f"Debug: Successfully extracted rendered_depth, shape: {rendered_depth.shape}")
                    except Exception as e:
                        print(f"Debug: Failed to extract rendered_depth: {e}")
                        rendered_depth = torch.zeros(height, width, device=device)
            
        except Exception as e:
            print(f"Rendering failed with exception: {e}")
            print(f"Exception type: {type(e).__name__}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            print(f"Debug render params: means.shape={means.shape}, rotations.shape={rotations.shape}, scales.shape={scales.shape}")
            print(f"Debug render params: opacities.shape={opacities.shape}, colors.shape={colors.shape}")
            print(f"Debug render params: viewmat.shape={viewmat.shape}, K.shape={K.shape}, width={width}, height={height}")
            # Check for NaN or invalid values
            print(f"Debug NaN check: means has NaN: {torch.isnan(means).any()}")
            print(f"Debug NaN check: rotations has NaN: {torch.isnan(rotations).any()}")
            print(f"Debug NaN check: scales has NaN: {torch.isnan(scales).any()}")
            print(f"Debug NaN check: opacities has NaN: {torch.isnan(opacities).any()}")
            print(f"Debug NaN check: colors has NaN: {torch.isnan(colors).any()}")
            rendered_rgb = torch.zeros(3, height, width, device=device)
            rendered_depth = torch.zeros(height, width, device=device)
        
        return rendered_rgb, rendered_depth
    
    def _object_exists_in_frame(self, obj_data: Dict, frame_idx: int) -> bool:
        """检查动态物体是否在指定帧中存在"""
        # 方法1：检查是否有帧存在信息
        if 'frame_existence' in obj_data:
            frame_existence = obj_data['frame_existence']  # [num_frames] bool tensor
            if frame_idx < len(frame_existence):
                return bool(frame_existence[frame_idx])
        
        # 方法2：检查是否有该帧的变换信息
        if 'frame_transforms' in obj_data:
            frame_transforms = obj_data['frame_transforms']
            if frame_idx in frame_transforms:
                return True
        
        # 方法3：检查是否有pose deltas信息
        if 'pose_deltas' in obj_data and obj_data['pose_deltas']:
            pose_deltas = obj_data['pose_deltas']
            if frame_idx < len(pose_deltas):
                return True
        
        # 默认：如果有正规Gaussians，假设在所有帧都存在
        return obj_data.get('canonical_gaussians') is not None
    
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
                    canonical_to_frame = torch.linalg.inv(frame_to_canonical)
                    return canonical_to_frame
                except Exception as e:
                    print(f"Warning: Failed to invert transform for frame {frame_idx}: {e}")
                    return None
        
        # 方法2：从pose deltas构建变换（如果有的话）
        if 'pose_deltas' in obj_data and obj_data['pose_deltas']:
            pose_deltas = obj_data['pose_deltas']
            if frame_idx < len(pose_deltas):
                pose_delta = pose_deltas[frame_idx]  # [6] - rotation + translation
                # 这里假设pose_delta也是从frame到canonical的增量
                frame_to_canonical_delta = self._pose_delta_to_transform_matrix(pose_delta)
                try:
                    canonical_to_frame = torch.linalg.inv(frame_to_canonical_delta)
                    return canonical_to_frame
                except Exception:
                    return None
        
        # 如果没有变换信息，返回None
        return None
    
    def _pose_delta_to_transform_matrix(self, pose_delta: torch.Tensor) -> torch.Tensor:
        """将pose delta转换为4x4变换矩阵"""
        # pose_delta: [6] - [rx, ry, rz, tx, ty, tz]
        device = pose_delta.device
        
        # 旋转部分 (轴角表示)
        rotation_vec = pose_delta[:3]  # [3]
        translation = pose_delta[3:6]  # [3]
        
        # 轴角转旋转矩阵
        angle = torch.norm(rotation_vec)
        if angle < 1e-8:
            # 接近零旋转
            R = torch.eye(3, device=device)
        else:
            axis = rotation_vec / angle
            # Rodrigues公式
            K = torch.tensor([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ], device=device)
            
            R = torch.eye(3, device=device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.mm(K, K)
        
        # 构建4x4变换矩阵
        transform = torch.eye(4, device=device)
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
            transform = torch.eye(4, dtype=transform.dtype, device=transform.device)
        elif torch.det(transform[:3, :3]).abs() < 1e-8:
            print(f"⚠️  变换矩阵奇异(det={torch.det(transform[:3, :3]):.2e})！")
            print(f"变换矩阵:\n{transform}")
        
        transformed_gaussians = gaussians.clone()
        
        # 变换位置
        positions = gaussians[:, :3]  # [N, 3]
        positions_homo = torch.cat([positions, torch.ones(positions.shape[0], 1, device=positions.device)], dim=1)  # [N, 4]
        transformed_positions = torch.mm(transform, positions_homo.T).T[:, :3]  # [N, 3]
        transformed_gaussians[:, :3] = transformed_positions
        
        # # 变换旋转（四元数）
        # if gaussians.shape[1] >= 13:  # 确保有四元数
        #     R = transform[:3, :3]  # [3, 3] 旋转矩阵
        #     quats = gaussians[:, 9:13]  # [N, 4] - [w, x, y, z]
            
        #     # 四元数转旋转矩阵
        #     transformed_quats = []
        #     for i in range(quats.shape[0]):
        #         quat = quats[i]  # [4]
        #         # 四元数转旋转矩阵
        #         w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        #         R_quat = torch.tensor([
        #             [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
        #             [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
        #             [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        #         ], device=gaussians.device)
                
        #         # 应用变换
        #         R_transformed = torch.mm(R, R_quat)
                
        #         # 旋转矩阵转四元数
        #         transformed_quat = self._rotation_matrix_to_quaternion(R_transformed)
        #         transformed_quats.append(transformed_quat)
            
        #     transformed_gaussians[:, 9:13] = torch.stack(transformed_quats, dim=0)
        
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
    
    def _compute_consistency_loss(
        self, 
        rendered_images: List[torch.Tensor], 
        rendered_depths: List[torch.Tensor]
    ) -> torch.Tensor:
        """计算跨帧一致性损失"""
        if len(rendered_images) < 2:
            return torch.tensor(0.0, device=rendered_images[0].device)
        
        consistency_loss = 0.0
        num_pairs = 0
        
        # 计算相邻帧的一致性
        for i in range(len(rendered_images) - 1):
            img1 = rendered_images[i]
            img2 = rendered_images[i + 1]
            
            # 图像梯度一致性
            grad1_x = torch.abs(img1[:, :, 1:] - img1[:, :, :-1])
            grad1_y = torch.abs(img1[:, 1:, :] - img1[:, :-1, :])
            grad2_x = torch.abs(img2[:, :, 1:] - img2[:, :, :-1])
            grad2_y = torch.abs(img2[:, 1:, :] - img2[:, :-1, :])
            
            grad_consistency = F.l1_loss(grad1_x, grad2_x) + F.l1_loss(grad1_y, grad2_y)
            consistency_loss += grad_consistency
            num_pairs += 1
        
        return consistency_loss / max(num_pairs, 1)


class Stage2GeometricLoss(nn.Module):
    """第二阶段几何损失"""
    
    def __init__(
        self,
        gaussian_regularization_weight: float = 0.0,  # 禁用gaussian regularization
        pose_regularization_weight: float = 0.0,  # 禁用pose regularization
        temporal_smoothness_weight: float = 0.0  # 禁用temporal smoothness
    ):
        super().__init__()
        self.gaussian_reg_weight = gaussian_regularization_weight
        self.pose_reg_weight = pose_regularization_weight
        self.temporal_smoothness_weight = temporal_smoothness_weight
    
    def forward(self, refinement_results: Dict) -> Dict[str, torch.Tensor]:
        """
        计算几何正则化损失
        
        Args:
            refinement_results: Stage2Refiner的输出结果
            
        Returns:
            loss_dict: 几何损失字典
        """
        loss_dict = {}
        # 安全地获取设备信息
        device = torch.device('cuda')
        if refinement_results['refined_dynamic_objects']:
            for obj_data in refinement_results['refined_dynamic_objects']:
                if obj_data.get('gaussian_deltas') is not None:
                    device = obj_data['gaussian_deltas'].device
                    break
                elif obj_data.get('refined_gaussians') is not None:
                    device = obj_data['refined_gaussians'].device
                    break
        
        total_gaussian_reg = torch.tensor(0.0, device=device)
        total_pose_reg = torch.tensor(0.0, device=device)
        total_temporal_smooth = torch.tensor(0.0, device=device)
        num_objects = 0
        
        for obj_data in refinement_results['refined_dynamic_objects']:
            num_objects += 1
            
            # Gaussian参数正则化 - 仅在权重大于0时计算
            if self.gaussian_reg_weight > 0 and obj_data['gaussian_deltas'] is not None:
                gaussian_deltas = obj_data['gaussian_deltas']
                
                # L1正则化：鼓励稀疏的变化
                gaussian_l1_reg = torch.mean(torch.abs(gaussian_deltas))
                total_gaussian_reg += gaussian_l1_reg
                
                # 尺度正则化：防止过大的尺度变化
                scale_deltas = gaussian_deltas[:, 3:6]
                scale_reg = torch.mean(torch.abs(scale_deltas))
                total_gaussian_reg += 0.5 * scale_reg
                
                # 透明度正则化：防止透明度剧变
                if gaussian_deltas.shape[1] > 10:
                    opacity_deltas = gaussian_deltas[:, 10:11]
                    opacity_reg = torch.mean(torch.abs(opacity_deltas))
                    total_gaussian_reg += 0.5 * opacity_reg
            
            # 位姿参数正则化 - 仅在权重大于0时计算
            if self.pose_reg_weight > 0 and obj_data['pose_deltas']:
                pose_deltas = torch.stack(obj_data['pose_deltas'])  # [num_frames, 6]
                
                # L1正则化：鼓励小的位姿变化
                pose_l1_reg = torch.mean(torch.abs(pose_deltas))
                total_pose_reg += pose_l1_reg
                
                # 旋转和平移分别正则化
                rot_deltas = pose_deltas[:, :3]
                trans_deltas = pose_deltas[:, 3:6]
                
                rot_reg = torch.mean(torch.abs(rot_deltas))
                trans_reg = torch.mean(torch.abs(trans_deltas))
                
                total_pose_reg += 0.3 * rot_reg + 0.7 * trans_reg
            
            # 时间平滑性：相邻帧的位姿变化应该平滑 - 仅在权重大于0时计算
            if self.temporal_smoothness_weight > 0 and obj_data['pose_deltas']:
                pose_deltas = torch.stack(obj_data['pose_deltas'])  # [num_frames, 6]
                if pose_deltas.shape[0] > 1:
                    temporal_diff = pose_deltas[1:] - pose_deltas[:-1]
                    temporal_smooth = torch.mean(torch.abs(temporal_diff))
                    total_temporal_smooth += temporal_smooth
        
        # 平均损失
        if num_objects > 0:
            total_gaussian_reg /= num_objects
            total_pose_reg /= num_objects
            total_temporal_smooth /= num_objects
        
        loss_dict = {
            'stage2_gaussian_reg': torch.tensor(0.0, device=device) if self.gaussian_reg_weight == 0 else self.gaussian_reg_weight * total_gaussian_reg,
            'stage2_pose_reg': torch.tensor(0.0, device=device) if self.pose_reg_weight == 0 else self.pose_reg_weight * total_pose_reg,
            'stage2_temporal_smooth': torch.tensor(0.0, device=device) if self.temporal_smoothness_weight == 0 else self.temporal_smoothness_weight * total_temporal_smooth,
        }
        
        # 检查和修复异常值
        for key, value in loss_dict.items():
            loss_dict[key] = check_and_fix_inf_nan(value, key)
        
        return loss_dict


class Stage2CompleteLoss(nn.Module):
    """第二阶段完整损失函数"""
    
    def __init__(
        self,
        render_loss_config: Optional[Dict] = None,
        geometric_loss_config: Optional[Dict] = None
    ):
        super().__init__()
        
        # 默认配置
        if render_loss_config is None:
            render_loss_config = {
                'rgb_weight': 1.0,
                'depth_weight': 0.0,  # 禁用depth loss
                'lpips_weight': 0.1,
                'consistency_weight': 0.0  # 禁用consistency loss
            }
            
        if geometric_loss_config is None:
            geometric_loss_config = {
                'gaussian_regularization_weight': 0.0,  # 禁用gaussian regularization
                'pose_regularization_weight': 0.0,  # 禁用pose regularization
                'temporal_smoothness_weight': 0.0  # 禁用temporal smoothness
            }
        
        self.render_loss = Stage2RenderLoss(**render_loss_config)
        self.geometric_loss = Stage2GeometricLoss(**geometric_loss_config)
    
    def forward(
        self,
        refinement_results: Dict,
        refined_scene: Dict,
        gt_images: torch.Tensor,
        gt_depths: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        frame_masks: Optional[torch.Tensor] = None
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
            frame_masks: 有效区域掩码
            
        Returns:
            complete_loss_dict: 完整损失字典
        """
        # 渲染损失
        render_loss_dict = self.render_loss(
            refined_scene, gt_images, gt_depths, 
            intrinsics, extrinsics, frame_masks
        )
        
        # 几何损失
        geometric_loss_dict = self.geometric_loss(refinement_results)
        
        # 合并损失
        complete_loss_dict = {**render_loss_dict, **geometric_loss_dict}
        
        # 计算总损失
        total_loss = render_loss_dict['stage2_total_loss']
        for key, value in geometric_loss_dict.items():
            total_loss = total_loss + value
        
        complete_loss_dict['stage2_final_total_loss'] = total_loss
        
        return complete_loss_dict