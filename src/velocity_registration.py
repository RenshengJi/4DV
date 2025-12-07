#!/usr/bin/env python3
"""
基于Velocity的点云配准系统
使用velocity场直接计算相邻两帧之间的3D变换
使用Procrustes算法进行精确的3D刚体变换估计
将同一物体的多帧点云聚合到中间帧上
"""

from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.dust3r.utils.misc import tf32_off
import sys
import os
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/vggt'))


class VelocityBasedRegistration:
    """基于Velocity的点云配准类"""

    def __init__(self,
                 device: str = "cuda",
                 min_inliers_ratio: float = 0.1,
                 velocity_transform_mode: str = "simple"):
        """
        初始化基于Velocity的点云配准器

        Args:
            device: 计算设备
            min_inliers_ratio: 最小内点比例
            velocity_transform_mode: velocity变换模式
                - "simple": 仅用velocity平均值估计平移T，旋转R为单位矩阵
                - "procrustes": 使用Procrustes算法估计完整R和T
        """
        self.device = device
        self.min_inliers_ratio = min_inliers_ratio
        self.velocity_transform_mode = velocity_transform_mode
        self.registration_results = {}

        print(f"Velocity-based 点云配准器初始化完成 - 变换模式: {velocity_transform_mode}")

    def estimate_transformation_direct(self,
                                       points_3d_src: torch.Tensor,
                                       points_3d_dst: torch.Tensor,
                                       weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        使用Procrustes算法计算3D-3D点对应的刚体变换

        Args:
            points_3d_src: 源帧3D点 [N, 3]
            points_3d_dst: 目标帧3D点 [N, 3]
            weights: 可选的权重 [N], 用于加权Procrustes

        Returns:
            R: 旋转矩阵 [3, 3]
            t: 平移向量 [3]
            inlier_ratio: 内点比例
        """
        if len(points_3d_src) < 3:
            return torch.eye(3, device=self.device, dtype=torch.float32), \
                   torch.zeros(3, device=self.device, dtype=torch.float32), 0.0

        if len(points_3d_src) != len(points_3d_dst):
            print(f"警告: 3D点对应数量不匹配 - src: {len(points_3d_src)}, dst: {len(points_3d_dst)}")
            return torch.eye(3, device=self.device, dtype=torch.float32), \
                   torch.zeros(3, device=self.device, dtype=torch.float32), 0.0

        if torch.any(torch.isnan(points_3d_src)) or torch.any(torch.isnan(points_3d_dst)):
            print("警告: 3D点包含NaN值")
            return torch.eye(3, device=self.device, dtype=torch.float32), \
                   torch.zeros(3, device=self.device, dtype=torch.float32), 0.0

        if torch.any(torch.isinf(points_3d_src)) or torch.any(torch.isinf(points_3d_dst)):
            print("警告: 3D点包含无限值")
            return torch.eye(3, device=self.device, dtype=torch.float32), \
                   torch.zeros(3, device=self.device, dtype=torch.float32), 0.0

        try:
            # 使用带权重的Procrustes或普通Procrustes
            if weights is not None:
                R, t = self._weighted_procrustes(points_3d_src, points_3d_dst, weights)
            else:
                R, t = self._procrustes_algorithm(points_3d_src, points_3d_dst)
            inlier_ratio = self._compute_inlier_ratio(points_3d_src, points_3d_dst, R, t)
            return R, t, inlier_ratio
        except (ValueError, RuntimeError) as e:
            print(f"变换估计失败: {str(e)}")
            return torch.eye(3, device=self.device, dtype=torch.float32), \
                   torch.zeros(3, device=self.device, dtype=torch.float32), 0.0

    def _weighted_procrustes(self, pts_src: torch.Tensor, pts_dst: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        带权重的Procrustes/Kabsch算法：计算3D点云之间的最优刚体变换

        Args:
            pts_src: 源点云 [N, 3]
            pts_dst: 目标点云 [N, 3]
            weights: 权重 [N], 要求非负

        Returns:
            R: 旋转矩阵 [3, 3]
            t: 平移向量 [3]
        """
        with tf32_off():
            if len(pts_src) != len(pts_dst):
                raise ValueError(f"点云大小不匹配: {len(pts_src)} vs {len(pts_dst)}")

            if len(pts_src) != len(weights):
                raise ValueError(f"权重数量不匹配: {len(weights)} vs {len(pts_src)}")

            if len(pts_src) < 3:
                raise ValueError(f"点数不足，至少需要3个点，实际有{len(pts_src)}个")

            # 归一化权重
            weights = weights / (weights.sum() + 1e-8)

            # 计算加权质心
            centroid_src = (pts_src * weights[:, None]).sum(dim=0)
            centroid_dst = (pts_dst * weights[:, None]).sum(dim=0)

            # 去中心化
            src_centered = pts_src - centroid_src
            dst_centered = pts_dst - centroid_dst

            # 检查点是否共线
            if torch.allclose(src_centered, torch.zeros_like(src_centered)) or \
               torch.allclose(dst_centered, torch.zeros_like(dst_centered)):
                return torch.eye(3, device=self.device, dtype=torch.float32), centroid_dst - centroid_src

            # 带权协方差矩阵
            H = src_centered.mul(weights[:, None]).T @ dst_centered

            if torch.allclose(H, torch.zeros_like(H)):
                return torch.eye(3, device=self.device, dtype=torch.float32), centroid_dst - centroid_src

            # SVD分解
            try:
                U, S, Vt = torch.linalg.svd(H)
            except RuntimeError as e:
                raise RuntimeError(f"SVD分解失败: {str(e)}")

            # 计算旋转矩阵
            R = Vt.T @ U.T

            # 处理反射情况
            if torch.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            if not self._is_valid_rotation_matrix(R):
                raise ValueError("计算得到的旋转矩阵无效")

            # 计算平移向量
            t = centroid_dst - R @ centroid_src

        return R, t

    def _procrustes_algorithm(self, points_src: torch.Tensor, points_dst: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Procrustes/Kabsch算法：计算3D点云之间的最优刚体变换

        Args:
            points_src: 源点云 [N, 3]
            points_dst: 目标点云 [N, 3]

        Returns:
            R: 旋转矩阵 [3, 3]
            t: 平移向量 [3]
        """
        with tf32_off():
            if len(points_src) != len(points_dst):
                raise ValueError(f"点云大小不匹配: {len(points_src)} vs {len(points_dst)}")

            if len(points_src) < 3:
                raise ValueError(f"点数不足，至少需要3个点，实际有{len(points_src)}个")

            # 计算质心
            centroid_src = torch.mean(points_src, dim=0)
            centroid_dst = torch.mean(points_dst, dim=0)

            # 去中心化
            points_src_centered = points_src - centroid_src
            points_dst_centered = points_dst - centroid_dst

            # 检查点是否共线
            if torch.allclose(points_src_centered, torch.zeros_like(points_src_centered)) or \
               torch.allclose(points_dst_centered, torch.zeros_like(points_dst_centered)):
                return torch.eye(3, device=self.device, dtype=torch.float32), centroid_dst - centroid_src

            # 计算协方差矩阵
            H = torch.matmul(points_src_centered.T, points_dst_centered)

            if torch.allclose(H, torch.zeros_like(H)):
                return torch.eye(3, device=self.device, dtype=torch.float32), centroid_dst - centroid_src

            # SVD分解
            try:
                U, S, Vt = torch.linalg.svd(H)
            except RuntimeError as e:
                raise RuntimeError(f"SVD分解失败: {str(e)}")

            # 计算旋转矩阵
            R = torch.matmul(Vt.T, U.T)

            # 处理反射情况
            if torch.linalg.det(R) < 0:
                Vt_corrected = Vt.clone()
                Vt_corrected[-1, :] *= -1
                R = torch.matmul(Vt_corrected.T, U.T)

            if not self._is_valid_rotation_matrix(R):
                raise ValueError("计算得到的旋转矩阵无效")

            # 计算平移向量
            t = centroid_dst - torch.matmul(R, centroid_src)

        return R, t

    def _is_valid_rotation_matrix(self, R: torch.Tensor, tolerance: float = 1e-6) -> bool:
        """验证是否为有效的旋转矩阵"""
        if R.shape != (3, 3):
            return False

        should_be_identity = torch.matmul(R.T, R)
        identity = torch.eye(3, device=R.device, dtype=R.dtype)
        if not torch.allclose(should_be_identity, identity, atol=tolerance):
            return False

        det = torch.linalg.det(R)
        if not torch.isclose(det, torch.tensor(1.0, device=R.device, dtype=R.dtype), atol=tolerance):
            return False

        return True

    def _compute_inlier_ratio(self, points_src: torch.Tensor, points_dst: torch.Tensor,
                             R: torch.Tensor, t: torch.Tensor, threshold: float = 0.1) -> float:
        """计算内点比例"""
        points_src_transformed = torch.matmul(points_src, R.T) + t
        distances = torch.norm(points_src_transformed - points_dst, dim=1)
        inliers = distances < threshold
        inlier_ratio = torch.sum(inliers).item() / len(points_src)
        return inlier_ratio

    def compute_optimized_chain_transformation(self,
                                               start_frame: int,
                                               end_frame: int,
                                               clustering_results: List[Dict],
                                               preds: Dict,
                                               vggt_batch: Dict,
                                               global_id: int,
                                               transformation_cache: Dict) -> Optional[torch.Tensor]:
        """
        优化版本的链式变换计算，使用缓存避免重复计算

        Args:
            start_frame: 起始帧
            end_frame: 目标帧
            clustering_results: 聚类结果
            preds: 模型预测结果
            vggt_batch: 输入数据批次
            global_id: 物体全局ID
            transformation_cache: 变换缓存字典

        Returns:
            累积变换矩阵或None
        """
        if start_frame == end_frame:
            return torch.eye(4, device=self.device, dtype=torch.float32, requires_grad=True)

        # 检查缓存
        cache_key = (start_frame, end_frame)
        if cache_key in transformation_cache:
            cached_transform = transformation_cache[cache_key]
            if isinstance(cached_transform, np.ndarray):
                cached_transform = torch.from_numpy(cached_transform).to(self.device).float()
                transformation_cache[cache_key] = cached_transform
            return cached_transform

        # 确定变换方向和路径
        if start_frame < end_frame:
            frame_sequence = list(range(start_frame, end_frame))
            direction = 1
        else:
            frame_sequence = list(range(start_frame, end_frame, -1))
            direction = -1

        # 尝试利用已有的变换进行优化
        for intermediate_frame in range(min(start_frame, end_frame) + 1, max(start_frame, end_frame)):
            key1 = (start_frame, intermediate_frame)
            key2 = (intermediate_frame, end_frame)

            if key1 in transformation_cache and key2 in transformation_cache:
                trans1 = transformation_cache[key1]
                trans2 = transformation_cache[key2]

                if isinstance(trans1, np.ndarray):
                    trans1 = torch.from_numpy(trans1).to(self.device).float()
                    transformation_cache[key1] = trans1
                if isinstance(trans2, np.ndarray):
                    trans2 = torch.from_numpy(trans2).to(self.device).float()
                    transformation_cache[key2] = trans2

                combined_transformation = torch.matmul(trans2, trans1)
                transformation_cache[cache_key] = combined_transformation
                return combined_transformation

        # 计算链式变换
        cumulative_transformation = torch.eye(4, device=self.device, dtype=torch.float32, requires_grad=True)

        for i, frame_idx in enumerate(frame_sequence):
            next_frame = frame_idx + direction

            # 检查单步变换的缓存
            step_key = (frame_idx, next_frame)
            if step_key in transformation_cache:
                step_transformation = transformation_cache[step_key]
                if isinstance(step_transformation, np.ndarray):
                    step_transformation = torch.from_numpy(step_transformation).to(self.device).float()
            else:
                # 获取当前帧和下一帧的聚类结果
                current_result = clustering_results[frame_idx]
                next_result = clustering_results[next_frame]

                current_global_ids = current_result.get('global_ids', [])
                next_global_ids = next_result.get('global_ids', [])

                if global_id not in current_global_ids or global_id not in next_global_ids:
                    return None

                # 获取数据
                depth_src = preds["depth"][0, frame_idx, :, :, 0]
                depth_dst = preds["depth"][0, next_frame, :, :, 0]
                velocity_src = preds['velocity_global'][0, frame_idx]
                velocity_dst = preds['velocity_global'][0, next_frame]
                depth_conf_src = preds.get('depth_conf', [None])[0, frame_idx] if 'depth_conf' in preds else None

                # 计算单步变换
                step_transformation = self.compute_single_step_transformation(
                    current_result, next_result, depth_src, depth_dst,
                    velocity_src, velocity_dst, global_id, direction, depth_conf_src
                )

                if step_transformation is None:
                    return None

                transformation_cache[step_key] = step_transformation

            if isinstance(step_transformation, np.ndarray):
                step_transformation = torch.from_numpy(step_transformation).to(self.device).float()

            cumulative_transformation = torch.matmul(step_transformation, cumulative_transformation)

        transformation_cache[cache_key] = cumulative_transformation
        return cumulative_transformation

    def compute_single_step_transformation(self,
                                           clustering_src: Dict,
                                           clustering_dst: Dict,
                                           depth_src: torch.Tensor,
                                           depth_dst: torch.Tensor,
                                           velocity_src: torch.Tensor,
                                           velocity_dst: Optional[torch.Tensor],
                                           global_id: int,
                                           direction: int = 1,
                                           depth_conf_src: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        计算单步变换（相邻两帧之间）

        Args:
            clustering_src: 源帧聚类结果
            clustering_dst: 目标帧聚类结果
            depth_src: 源帧深度图
            depth_dst: 目标帧深度图
            velocity_src: 源帧velocity场 [H, W, 3]
            velocity_dst: 目标帧velocity场
            global_id: 物体全局ID
            direction: 变换方向，1表示forward，-1表示backward
            depth_conf_src: 源帧深度置信度

        Returns:
            变换矩阵或None
        """
        H, W = depth_src.shape

        # 提取属于该物体的点的索引
        global_ids = clustering_src.get('global_ids', [])
        if global_id not in global_ids:
            return None

        cluster_idx = global_ids.index(global_id)
        cluster_indices = clustering_src.get('cluster_indices', [])
        if cluster_idx >= len(cluster_indices):
            return None

        object_indices = cluster_indices[cluster_idx]
        if len(object_indices) == 0:
            return None

        # 提取对应点的velocity
        if isinstance(velocity_src, torch.Tensor):
            if len(velocity_src.shape) == 3:
                velocity_flat = velocity_src.reshape(H * W, 3)
            elif len(velocity_src.shape) == 2:
                velocity_flat = velocity_src
            else:
                return None
            object_velocities = velocity_flat[object_indices]
        else:
            velocity_flat = velocity_src.reshape(H * W, 3) if len(velocity_src.shape) == 3 else velocity_src
            object_velocities = torch.from_numpy(velocity_flat[object_indices]).to(self.device).float()

        if object_velocities.device != torch.device(self.device):
            object_velocities = object_velocities.to(self.device)

        # 提取depth_conf作为权重（不过滤）
        object_weights = None
        if depth_conf_src is not None and isinstance(depth_conf_src, torch.Tensor):
            depth_conf_flat = depth_conf_src.reshape(H * W) if len(depth_conf_src.shape) == 2 else depth_conf_src
            if depth_conf_flat is not None and len(depth_conf_flat.shape) == 1:
                object_weights = depth_conf_flat[object_indices].detach()
                # 确保权重非负
                object_weights = torch.clamp(object_weights, min=0.0)

        # Simple模式
        if self.velocity_transform_mode == "simple":
            mean_velocity = object_velocities.mean(dim=0)
            adjusted_velocity = mean_velocity * direction
            transformation = torch.eye(4, device=self.device, dtype=torch.float32)
            transformation[:3, 3] = adjusted_velocity
            return transformation

        # Procrustes模式
        elif self.velocity_transform_mode == "procrustes":
            points_src = clustering_src.get('points', None)
            labels = clustering_src.get('labels', None)
            if points_src is None or labels is None:
                return None

            object_mask = labels == cluster_idx
            points_src_object = points_src[object_mask]

            if not isinstance(points_src_object, torch.Tensor):
                points_src_object = torch.from_numpy(points_src_object).to(self.device).float()
            elif points_src_object.device != torch.device(self.device):
                points_src_object = points_src_object.to(self.device)

            # 检查点数是否匹配
            if len(object_velocities) != len(points_src_object):
                min_len = min(len(object_velocities), len(points_src_object))
                if min_len < 3:
                    mean_velocity = object_velocities.mean(dim=0)
                    adjusted_velocity = mean_velocity * direction
                    transformation = torch.eye(4, device=self.device, dtype=torch.float32)
                    transformation[:3, 3] = adjusted_velocity
                    return transformation
                object_velocities = object_velocities[:min_len]
                points_src_object = points_src_object[:min_len]
                if object_weights is not None:
                    object_weights = object_weights[:min_len]

            points_dst_object = points_src_object + object_velocities * direction

            try:
                # 使用带权重的Procrustes（如果有depth_conf）
                R, t, inlier_ratio = self.estimate_transformation_direct(
                    points_src_object, points_dst_object, weights=object_weights
                )
                transformation = torch.eye(4, device=self.device, dtype=torch.float32)
                transformation[:3, :3] = R
                transformation[:3, 3] = t
                return transformation
            except Exception as e:
                print(f"    Procrustes估计失败: {e}，回退到simple模式")
                mean_velocity = object_velocities.mean(dim=0)
                adjusted_velocity = mean_velocity * direction
                transformation = torch.eye(4, device=self.device, dtype=torch.float32)
                transformation[:3, 3] = adjusted_velocity
                return transformation

        else:
            print(f"    警告: 未知的velocity_transform_mode: {self.velocity_transform_mode}")
            mean_velocity = object_velocities.mean(dim=0)
            adjusted_velocity = mean_velocity * direction
            transformation = torch.eye(4, device=self.device, dtype=torch.float32)
            transformation[:3, 3] = adjusted_velocity
            return transformation

    def aggregate_object_to_middle_frame(self,
                                         clustering_results: List[Dict],
                                         preds: Dict,
                                         vggt_batch: Dict,
                                         global_id: int) -> Optional[Dict]:
        """
        将同一物体的多帧点云聚合到中间帧

        Args:
            clustering_results: 所有帧的聚类结果
            preds: 模型预测结果
            vggt_batch: 输入数据批次
            global_id: 物体全局ID

        Returns:
            聚合结果字典或None
        """
        # 找到物体出现的帧
        object_frames = []
        for frame_idx, result in enumerate(clustering_results):
            global_ids = result.get('global_ids', [])
            if global_id in global_ids:
                object_frames.append(frame_idx)

        if len(object_frames) < 1:
            return None

        # 单帧物体处理
        if len(object_frames) == 1:
            frame_idx = object_frames[0]
            result = clustering_results[frame_idx]
            cluster_idx = result['global_ids'].index(global_id)
            object_mask = result['labels'] == cluster_idx
            object_points = result['points'][object_mask]

            cluster_indices = result.get('cluster_indices', [])
            single_frame_pixel_indices = cluster_indices[cluster_idx] if cluster_idx < len(cluster_indices) else []

            canonical_gaussians = None
            frame_gaussians = {}
            if preds and 'gaussian_params' in preds:
                single_frame_point_indices = [(frame_idx, pixel_idx) for pixel_idx in single_frame_pixel_indices[:len(object_points)]]
                canonical_gaussians = self._extract_all_frames_gaussian_params(
                    object_points, single_frame_point_indices, preds['gaussian_params'], vggt_batch
                )
                frame_gaussians[frame_idx] = self._extract_gaussian_params_for_object(
                    result, cluster_idx, frame_idx, preds['gaussian_params'], vggt_batch
                )

            return {
                'global_id': global_id,
                'aggregated_points': object_points,
                'point_indices': [(frame_idx, pixel_idx) for pixel_idx in single_frame_pixel_indices[:len(object_points)]],
                'middle_frame': frame_idx,
                'object_frames': [frame_idx],
                'transformations': {},
                'canonical_gaussians': canonical_gaussians,
                'frame_gaussians': frame_gaussians,
                'reference_frame': frame_idx,
                'num_frames': 1,
                'num_points': len(object_points)
            }

        # 多帧物体处理
        middle_frame_idx = object_frames[len(object_frames) // 2]

        middle_result = clustering_results[middle_frame_idx]
        middle_cluster_idx = middle_result['global_ids'].index(global_id)
        middle_points = middle_result['points']
        middle_labels = middle_result['labels']

        middle_object_mask = middle_labels == middle_cluster_idx
        middle_object_points = middle_points[middle_object_mask]

        cluster_indices = middle_result.get('cluster_indices', [])
        middle_pixel_indices = cluster_indices[middle_cluster_idx] if middle_cluster_idx < len(cluster_indices) else []

        transformations = {}
        transformation_cache = {}
        aggregated_points = [middle_object_points]
        all_point_indices = [(middle_frame_idx, pixel_idx) for pixel_idx in middle_pixel_indices[:len(middle_object_points)]]

        # 对其他帧进行链式变换
        for frame_idx in object_frames:
            if frame_idx == middle_frame_idx:
                continue

            chain_transformation = self.compute_optimized_chain_transformation(
                frame_idx, middle_frame_idx, clustering_results, preds, vggt_batch, global_id, transformation_cache
            )

            if chain_transformation is not None:
                transformations[frame_idx] = {
                    'transformation': chain_transformation,
                    'R': chain_transformation[:3, :3],
                    't': chain_transformation[:3, 3],
                    'inlier_ratio': 1.0,
                    'num_correspondences': 0
                }

                current_result = clustering_results[frame_idx]
                current_cluster_idx = current_result['global_ids'].index(global_id)
                current_object_mask = current_result['labels'] == current_cluster_idx
                current_object_points = current_result['points'][current_object_mask]

                current_cluster_indices = current_result.get('cluster_indices', [])
                current_pixel_indices = current_cluster_indices[current_cluster_idx] if current_cluster_idx < len(current_cluster_indices) else []

                transformed_points = self._apply_transformation(current_object_points, chain_transformation)
                aggregated_points.append(transformed_points)

                num_transformed_points = len(transformed_points)
                frame_point_indices = [(frame_idx, pixel_idx) for pixel_idx in current_pixel_indices[:num_transformed_points]]
                all_point_indices.extend(frame_point_indices)
            else:
                print(f"    ⚠️  链式变换失败: {frame_idx}")

        if len(aggregated_points) < 1:
            print(f"物体 {global_id}: 没有可用数据")
            return None

        all_points = torch.cat(aggregated_points, dim=0)

        # 提取Gaussian参数
        canonical_gaussians = None
        frame_gaussians = {}
        if preds and 'gaussian_params' in preds:
            canonical_gaussians = self._extract_all_frames_gaussian_params(
                all_points, all_point_indices, preds['gaussian_params'], vggt_batch
            )

            for frame_idx in object_frames:
                current_result = clustering_results[frame_idx]
                current_cluster_idx = current_result['global_ids'].index(global_id)
                frame_gaussians[frame_idx] = self._extract_gaussian_params_for_object(
                    current_result, current_cluster_idx, frame_idx, preds['gaussian_params'], vggt_batch
                )

        return {
            'global_id': global_id,
            'middle_frame': middle_frame_idx,
            'object_frames': object_frames,
            'aggregated_points': all_points,
            'point_indices': all_point_indices,
            'transformations': transformations,
            'canonical_gaussians': canonical_gaussians,
            'frame_gaussians': frame_gaussians,
            'num_frames': len(object_frames),
            'num_points': len(all_points)
        }

    def _extract_all_frames_gaussian_params(self,
                                            aggregated_points: torch.Tensor,
                                            point_indices: List[Tuple[int, int]],
                                            gaussian_params: torch.Tensor,
                                            vggt_batch: Dict) -> Optional[torch.Tensor]:
        """
        提取对应的Gaussian参数，并用聚合后的点坐标替换位置

        Args:
            aggregated_points: 聚合后的3D点坐标 [N, 3]
            point_indices: 每个点对应的(frame_idx, pixel_idx)
            gaussian_params: VGGT预测的Gaussian参数 [B, S, H, W, gaussian_output_dim]
            vggt_batch: 批次数据

        Returns:
            合并的Gaussian参数 [N, gaussian_output_dim]
        """
        try:
            if 'images' not in vggt_batch:
                print(f"    ⚠️ 提取Gaussian参数失败: vggt_batch中没有images")
                return None

            if len(point_indices) != len(aggregated_points):
                print(f"    ⚠️ 提取Gaussian参数失败: 点数量与索引数量不匹配: {len(aggregated_points)} vs {len(point_indices)}")
                return None

            B, S, C, H, W = vggt_batch['images'].shape
            gaussian_output_dim = gaussian_params.shape[-1]

            gaussian_params = gaussian_params.reshape(B, S * H * W, gaussian_output_dim)
            H_W = H * W

            frame_indices = [idx[0] for idx in point_indices]
            pixel_indices = [idx[1] for idx in point_indices]

            device = gaussian_params.device
            frame_indices_tensor = torch.tensor(frame_indices, dtype=torch.long, device=device)
            pixel_indices_tensor = torch.tensor(pixel_indices, dtype=torch.long, device=device)

            global_indices = frame_indices_tensor * H_W + pixel_indices_tensor
            valid_mask = (global_indices >= 0) & (global_indices < gaussian_params.shape[1])
            valid_global_indices = global_indices[valid_mask]

            if len(valid_global_indices) == 0:
                return None

            extracted_gaussians = gaussian_params[0, valid_global_indices].clone()

            if aggregated_points.device != extracted_gaussians.device:
                aggregated_points = aggregated_points.to(extracted_gaussians.device)

            valid_points = aggregated_points[valid_mask]
            extracted_gaussians[:, :3] = valid_points

            return extracted_gaussians

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

    def _extract_gaussian_params_for_object(self,
                                            clustering_result: Dict,
                                            cluster_idx: int,
                                            frame_idx: int,
                                            gaussian_params: torch.Tensor,
                                            vggt_batch: Dict) -> Optional[torch.Tensor]:
        """
        从聚类结果中提取对应物体的Gaussian参数

        Args:
            clustering_result: 单帧聚类结果
            cluster_idx: 聚类索引
            frame_idx: 帧索引
            gaussian_params: VGGT预测的Gaussian参数 [B, S, H, W, gaussian_output_dim]
            vggt_batch: 批次数据

        Returns:
            提取的Gaussian参数 [N, gaussian_output_dim]
        """
        try:
            cluster_indices = clustering_result.get('cluster_indices', [])
            if cluster_idx >= len(cluster_indices):
                print(f"    cluster_idx {cluster_idx} 超出范围，总聚类数: {len(cluster_indices)}")
                return None

            pixel_indices = cluster_indices[cluster_idx]
            if not pixel_indices:
                print(f"    物体在帧{frame_idx}中没有像素索引")
                return None

            if 'images' in vggt_batch:
                B, S, C, H, W = vggt_batch['images'].shape
            else:
                print(f"    无法从vggt_batch获取图像尺寸")
                return None

            gaussian_output_dim = gaussian_params.shape[-1]
            gaussian_params = gaussian_params.reshape(B, S * H * W, gaussian_output_dim)

            H_W = H * W
            pixel_indices_tensor = torch.tensor(pixel_indices, dtype=torch.long, device=gaussian_params.device)
            global_indices = frame_idx * H_W + pixel_indices_tensor

            selected_gaussians_tensor = gaussian_params[0, global_indices]

            if selected_gaussians_tensor.shape[0] == 0:
                print(f"    无法提取有效的Gaussian参数")
                return None

            return selected_gaussians_tensor

        except Exception as e:
            print(f"    提取Gaussian参数失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _apply_transformation(self, points: torch.Tensor, transformation: torch.Tensor) -> torch.Tensor:
        """应用变换矩阵到点云"""
        if isinstance(transformation, np.ndarray):
            transformation = torch.from_numpy(transformation).to(self.device).float()

        if points.device != transformation.device:
            points = points.to(transformation.device)

        ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
        points_homo = torch.cat([points, ones], dim=1)
        transformed_points_homo = torch.matmul(points_homo, transformation.T)

        return transformed_points_homo[:, :3]
