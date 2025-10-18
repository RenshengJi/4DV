# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
在线动态物体处理器

实时进行动态物体检测、分割、跟踪和聚合
使用demo_video_with_pointcloud_save.py和optical_flow_registration.py中的成熟方法
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
import time
from collections import defaultdict

# 导入现有的聚类和光流配准系统
import sys
import os
from cuml.cluster import DBSCAN
# from sklearn.cluster import DBSCAN as SklearnDBSCAN
import cupy as cp
from scipy.optimize import linear_sum_assignment


def _is_main_process():
    """检查是否为主进程（用于DDP训练）"""
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _print_main(*args, **kwargs):
    """只在主进程打印（用于DDP训练）"""
    if _is_main_process():
        print(*args, **kwargs)



def dynamic_object_clustering(xyz, velocity, velocity_threshold=0.01, eps=0.02, min_samples=10, area_threshold=750, conf_mask=None, gt_scale=None):
    """
    对每一帧进行动态物体聚类

    Args:
        xyz: [S, H*W, 3] 点云坐标
        velocity: [S, H*W, 3] 速度向量（非metric尺度）
        velocity_threshold: 速度阈值（metric尺度，m/s），用于过滤静态背景
        eps: DBSCAN的邻域半径
        min_samples: DBSCAN的最小样本数
        area_threshold: 面积阈值，过滤掉面积小于此值的聚类
        conf_mask: [S, H*W] confidence掩码，True表示高置信度像素
        gt_scale: float or tensor - GT scale factor，用于将velocity转换到metric尺度

    Returns:
        list: 每一帧的聚类结果，每个元素包含点云坐标和聚类标签
    """
    clustering_results = []
    device = xyz.device

    # 将gt_scale转换为tensor
    if gt_scale is not None:
        if not isinstance(gt_scale, torch.Tensor):
            gt_scale_tensor = torch.tensor(gt_scale, device=device, dtype=velocity.dtype)
        else:
            gt_scale_tensor = gt_scale
    else:
        gt_scale_tensor = torch.tensor(1.0, device=device, dtype=velocity.dtype)

    for frame_idx in range(xyz.shape[0]):
        # 获取当前帧的点云和速度
        frame_points = xyz[frame_idx]  # [H*W, 3]
        frame_velocity = velocity[frame_idx]  # [H*W, 3]

        # 将velocity转换到metric尺度: velocity_metric = velocity / gt_scale
        frame_velocity_metric = frame_velocity / gt_scale_tensor

        # 计算速度大小（metric尺度，m/s）
        velocity_magnitude = torch.norm(frame_velocity_metric, dim=-1)  # [H*W]

        # 过滤动态点（速度大于阈值的点，阈值现在是metric尺度）
        dynamic_mask = velocity_magnitude > velocity_threshold

        # 应用confidence掩码：只有高置信度且动态的点才参与聚类
        if conf_mask is not None:
            frame_conf_mask = conf_mask[frame_idx]  # [H*W]
            dynamic_mask = dynamic_mask & frame_conf_mask  # 同时满足动态和高置信度

        dynamic_points = frame_points[dynamic_mask]  # [N_dynamic, 3]
        dynamic_velocities = frame_velocity[dynamic_mask]  # [N_dynamic, 3]

        if len(dynamic_points) < min_samples:
            # 如果动态点太少，返回空聚类
            clustering_results.append({
                'points': frame_points,
                'labels': torch.full((len(frame_points),), -1, dtype=torch.long),
                'num_clusters': 0,
                'cluster_centers': [],
                'cluster_velocities': [],
                'cluster_sizes': [],
                'cluster_indices': []
            })
            continue

        # 尝试使用cuML GPU加速的DBSCAN聚类
        try:
            # 检查dynamic_points是否在GPU上
            if dynamic_points.is_cuda:
                # 如果已经在GPU上，直接转换为CuPy数组
                dynamic_points_cp = cp.asarray(dynamic_points.detach())
            else:
                # 如果在CPU上，先移到GPU再转换为CuPy数组
                dynamic_points_cp = cp.asarray(dynamic_points.detach().cpu().numpy())

            # 执行cuML DBSCAN聚类（GPU加速）
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, max_mbytes_per_batch=128)
            cluster_labels_cp = dbscan.fit_predict(dynamic_points_cp)

            # 转换回NumPy数组
            cluster_labels = cp.asnumpy(cluster_labels_cp)
            # print(f"cuML DBSCAN成功: 找到 {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} 个聚类")
        except Exception as e:
            # 回退到sklearn CPU版本
            # print(f"cuML DBSCAN失败，回退到sklearn: {e}")
            try:
                dynamic_points_np = dynamic_points.detach().cpu().numpy()
                dbscan_sklearn = SklearnDBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = dbscan_sklearn.fit_predict(dynamic_points_np)
                # print(f"sklearn DBSCAN成功: 找到 {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} 个聚类")
            except Exception as sklearn_e:
                # print(f"sklearn DBSCAN也失败: {sklearn_e}")
                # 简单回退：所有点标记为噪声
                cluster_labels = np.full(len(dynamic_points), -1)

        # 将聚类结果映射回原始点云
        full_labels = torch.full((len(frame_points),), -1, device=frame_points.device)
        full_labels[dynamic_mask] = torch.from_numpy(cluster_labels).to(frame_points.device).long()

        # 统计聚类数量（排除噪声点，标签为-1）
        all_unique_labels = set(cluster_labels)
        if -1 in all_unique_labels:
            all_unique_labels.remove(-1)
        initial_num_clusters = len(all_unique_labels)

        # 计算每个聚类的中心位置和平均速度
        cluster_centers = []
        cluster_velocities = []
        cluster_sizes = []
        valid_labels = []

        for label in sorted(all_unique_labels):
            cluster_mask = cluster_labels == label
            cluster_points = dynamic_points[cluster_mask]
            cluster_vel = dynamic_velocities[cluster_mask]

            # 计算聚类中心（平均位置）
            center = cluster_points.mean(dim=0)
            # 计算平均速度
            avg_velocity = cluster_vel.mean(dim=0)
            cluster_size = len(cluster_points)

            # 过滤掉面积太小的聚类
            if cluster_size >= area_threshold:
                cluster_centers.append(center)
                cluster_velocities.append(avg_velocity)
                cluster_sizes.append(cluster_size)
                valid_labels.append(label)
            else:
                # 将过滤掉的聚类重新标记为静态点（-1）
                cluster_indices = np.where(cluster_mask)[0]
                dynamic_indices = torch.where(dynamic_mask)[0]
                filtered_indices = dynamic_indices[cluster_indices]
                full_labels[filtered_indices] = -1

        # 更新聚类数量
        num_clusters = len(valid_labels)

        # 重新映射聚类标签，确保连续
        if num_clusters > 0:
            # 创建新的标签映射
            label_mapping = {old_label: new_label for new_label,
                             old_label in enumerate(valid_labels)}

            # 更新full_labels中的聚类标签
            for old_label, new_label in label_mapping.items():
                mask = full_labels == old_label
                full_labels[mask] = new_label

        # 计算每个聚类的点索引
        cluster_indices = []
        for label in range(num_clusters):
            # 找到属于当前聚类的点的索引
            cluster_mask = full_labels == label
            cluster_point_indices = torch.where(cluster_mask)[0].cpu().numpy().tolist()
            cluster_indices.append(cluster_point_indices)

        clustering_results.append({
            'points': frame_points,
            'labels': full_labels.cpu(),
            'num_clusters': num_clusters,
            'cluster_centers': cluster_centers,
            'cluster_velocities': cluster_velocities,
            'cluster_sizes': cluster_sizes,
            'cluster_indices': cluster_indices
        })

    return clustering_results


def match_objects_across_frames(clustering_results, position_threshold=0.5, velocity_threshold=0.2):
    """
    跨帧匹配动态物体（使用匈牙利算法）

    Args:
        clustering_results: 每一帧的聚类结果
        position_threshold: 位置匹配阈值
        velocity_threshold: 速度匹配阈值

    Returns:
        list: 每一帧的聚类结果，包含全局物体ID
    """
    if len(clustering_results) == 0:
        return clustering_results

    # 初始化全局物体ID
    next_global_id = 0
    global_object_tracks = {}  # {global_id: {frame_id, center, velocity, size}}

    # 为每一帧分配全局ID
    for frame_idx, frame_result in enumerate(clustering_results):
        if frame_result['num_clusters'] == 0:
            frame_result['global_ids'] = []
            # 确保 cluster_indices 字段存在
            if 'cluster_indices' not in frame_result:
                frame_result['cluster_indices'] = []
            continue

        frame_centers = frame_result['cluster_centers']
        frame_velocities = frame_result['cluster_velocities']
        frame_sizes = frame_result['cluster_sizes']

        # 初始化当前帧的全局ID数组，按照聚类标签的顺序
        frame_global_ids = [-1] * len(frame_centers)  # 初始化为-1表示未分配

        if frame_idx == 0:
            # 第一帧，为所有物体分配新的全局ID
            for cluster_idx in range(len(frame_centers)):
                global_id = next_global_id
                next_global_id += 1

                global_object_tracks[global_id] = {
                    'frame_id': frame_idx,
                    'center': frame_centers[cluster_idx],
                    'velocity': frame_velocities[cluster_idx],
                    'size': frame_sizes[cluster_idx]
                }
                frame_global_ids[cluster_idx] = global_id
        else:
            # 使用匈牙利算法进行匹配
            prev_result = clustering_results[frame_idx - 1]
            prev_global_ids = prev_result.get('global_ids', [])

            if len(prev_global_ids) == 0:
                # 前一帧没有物体，为当前帧所有物体分配新ID
                for cluster_idx in range(len(frame_centers)):
                    global_id = next_global_id
                    next_global_id += 1

                    global_object_tracks[global_id] = {
                        'frame_id': frame_idx,
                        'center': frame_centers[cluster_idx],
                        'velocity': frame_velocities[cluster_idx],
                        'size': frame_sizes[cluster_idx]
                    }
                    frame_global_ids[cluster_idx] = global_id
            else:
                # 构建成本矩阵
                num_prev = len(prev_global_ids)
                num_current = len(frame_centers)
                cost_matrix = np.full((num_prev, num_current), float('inf'))

                for i, prev_global_id in enumerate(prev_global_ids):
                    track_info = global_object_tracks[prev_global_id]
                    prev_center = track_info['center']
                    prev_velocity = track_info['velocity']

                    for j in range(num_current):
                        current_center = frame_centers[j]
                        current_velocity = frame_velocities[j]

                        # 使用T帧的位置和速度预测T+1帧的位置
                        predicted_center = prev_center + prev_velocity

                        # 计算预测位置与实际位置的距离
                        pos_distance = torch.norm(
                            current_center - predicted_center).item()

                        # 计算速度相似度
                        vel_distance = torch.norm(
                            current_velocity - prev_velocity).item()

                        # 综合评分（位置权重更高）
                        score = pos_distance  # + 0.3 * vel_distance

                        # 如果满足阈值条件，设置成本；否则保持无穷大
                        if pos_distance < position_threshold:  # and vel_distance < velocity_threshold:
                            cost_matrix[i, j] = score

                # 使用匈牙利算法求解最优匹配
                if num_prev > 0 and num_current > 0:
                    # 检查是否有有效的匹配（非无穷大成本）
                    has_valid_matches = np.any(cost_matrix < float('inf'))

                    if has_valid_matches:
                        try:
                            row_indices, col_indices = linear_sum_assignment(
                                cost_matrix)

                            # 处理匹配结果
                            matched_prev = set()
                            matched_current = set()

                            for i, j in zip(row_indices, col_indices):
                                if cost_matrix[i, j] < float('inf'):  # 有效匹配
                                    prev_global_id = prev_global_ids[i]
                                    matched_prev.add(i)
                                    matched_current.add(j)

                                    # 更新跟踪信息
                                    global_object_tracks[prev_global_id] = {
                                        'frame_id': frame_idx,
                                        'center': frame_centers[j],
                                        'velocity': frame_velocities[j],
                                        'size': frame_sizes[j]
                                    }
                                    # 按照聚类索引顺序存储
                                    frame_global_ids[j] = prev_global_id

                            # 为未匹配的当前帧物体分配新ID
                            for j in range(num_current):
                                if j not in matched_current:
                                    global_id = next_global_id
                                    next_global_id += 1

                                    global_object_tracks[global_id] = {
                                        'frame_id': frame_idx,
                                        'center': frame_centers[j],
                                        'velocity': frame_velocities[j],
                                        'size': frame_sizes[j]
                                    }
                                    # 按照聚类索引顺序存储
                                    frame_global_ids[j] = global_id

                            # 为未匹配的前一帧物体保持跟踪（可选：设置消失标记）
                            for i in range(num_prev):
                                if i not in matched_prev:
                                    prev_global_id = prev_global_ids[i]
                                    # 可以选择保持最后一帧的信息或标记为消失
                                    pass
                        except ValueError as e:
                            # 如果匈牙利算法失败，为所有物体分配新ID
                            for cluster_idx in range(len(frame_centers)):
                                global_id = next_global_id
                                next_global_id += 1

                                global_object_tracks[global_id] = {
                                    'frame_id': frame_idx,
                                    'center': frame_centers[cluster_idx],
                                    'velocity': frame_velocities[cluster_idx],
                                    'size': frame_sizes[cluster_idx]
                                }
                                frame_global_ids[cluster_idx] = global_id
                    else:
                        # 没有有效匹配，为所有物体分配新ID
                        for cluster_idx in range(len(frame_centers)):
                            global_id = next_global_id
                            next_global_id += 1

                            global_object_tracks[global_id] = {
                                'frame_id': frame_idx,
                                'center': frame_centers[cluster_idx],
                                'velocity': frame_velocities[cluster_idx],
                                'size': frame_sizes[cluster_idx]
                            }
                            frame_global_ids[cluster_idx] = global_id
                else:
                    # 没有前一帧物体，为当前帧所有物体分配新ID
                    for cluster_idx in range(len(frame_centers)):
                        global_id = next_global_id
                        next_global_id += 1

                        global_object_tracks[global_id] = {
                            'frame_id': frame_idx,
                            'center': frame_centers[cluster_idx],
                            'velocity': frame_velocities[cluster_idx],
                            'size': frame_sizes[cluster_idx]
                        }
                        frame_global_ids[cluster_idx] = global_id

        # 将全局ID添加到帧结果中
        frame_result['global_ids'] = frame_global_ids

    return clustering_results



def _import_optical_flow():
    """延迟导入光流配准"""
    try:
        import importlib.util
        import sys

        # 添加根目录到sys.path
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)

        optical_flow_spec = importlib.util.spec_from_file_location(
            "optical_flow_reg",
            os.path.join(root_dir, "optical_flow_registration.py")
        )
        optical_flow_module = importlib.util.module_from_spec(
            optical_flow_spec)
        optical_flow_spec.loader.exec_module(optical_flow_module)

        return optical_flow_module.OpticalFlowRegistration
    except Exception as e:
        _print_main(f"Failed to import optical flow: {e}")
        return None


class OnlineDynamicProcessor:
    """
    在线动态物体处理器

    实时执行：
    1. 动态物体检测和聚类（使用demo_video_with_pointcloud_save.py中的方法）
    2. 跨帧物体跟踪（使用成熟的匹配算法）
    3. 光流配准和聚合（使用optical_flow_registration.py中的方法）
    4. 第二阶段训练数据准备
    """

    def __init__(
        self,
        device: torch.device,
        memory_efficient: bool = True,
        min_object_size: int = 100,
        max_objects_per_frame: int = 10,
        velocity_threshold: float = 0.1,  # metric尺度, m/s
        clustering_eps: float = 0.02,  # DBSCAN邻域半径
        clustering_min_samples: int = 10,  # DBSCAN最小样本数
        tracking_position_threshold: float = 2.0,  # 跨帧跟踪位置阈值
        tracking_velocity_threshold: float = 0.2,  # 跨帧跟踪速度阈值
        use_optical_flow_aggregation: bool = True,
        use_velocity_based_transform: bool = False,  # 使用velocity计算变换（无需光流）
        enable_temporal_cache: bool = True,
        cache_size: int = 16
    ):
        """
        初始化在线动态物体处理器

        Args:
            device: 计算设备
            memory_efficient: 是否启用内存优化
            min_object_size: 最小物体尺寸（点数）
            max_objects_per_frame: 每帧最大物体数量
            velocity_threshold: 速度阈值（metric尺度，m/s）
            clustering_eps: DBSCAN邻域半径
            clustering_min_samples: DBSCAN最小样本数
            tracking_position_threshold: 跨帧跟踪位置阈值
            tracking_velocity_threshold: 跨帧跟踪速度阈值
            use_optical_flow_aggregation: 是否使用光流聚合
            enable_temporal_cache: 是否启用时序缓存
            cache_size: 缓存大小
        """
        self.device = device
        self.memory_efficient = memory_efficient
        self.min_object_size = min_object_size
        self.max_objects_per_frame = max_objects_per_frame
        self.velocity_threshold = velocity_threshold
        self.clustering_eps = clustering_eps
        self.clustering_min_samples = clustering_min_samples
        self.tracking_position_threshold = tracking_position_threshold
        self.tracking_velocity_threshold = tracking_velocity_threshold
        self.use_optical_flow_aggregation = use_optical_flow_aggregation

        # 初始化光流配准系统
        self.optical_flow_registration = None
        self._optical_flow_class = None
        if use_optical_flow_aggregation:
            self._optical_flow_class = _import_optical_flow()
            # 立即初始化光流配准系统
            if self._optical_flow_class is not None:
                try:
                    self.optical_flow_registration = self._optical_flow_class(
                        device=str(device),
                        min_inliers_ratio=0.1,  # 降低最小内点比例
                        ransac_threshold=5.0,   # 增加RANSAC阈值
                        max_flow_magnitude=200.0,  # 增加最大光流幅度
                        use_velocity_based_transform=use_velocity_based_transform  # 使用velocity计算变换
                    )
                except Exception as e:
                    self.optical_flow_registration = None

        # 聚类函数已直接定义在此文件中，不需要缓存

        # 时序缓存
        self.enable_temporal_cache = enable_temporal_cache
        self.cache_size = cache_size
        self.temporal_cache = defaultdict(
            list) if enable_temporal_cache else None

        # 统计信息
        self.processing_stats = {
            'total_sequences': 0,
            'total_objects_detected': 0,
            'total_processing_time': 0.0,
            'sam_time': 0.0,
            'optical_flow_time': 0.0,
            'aggregation_time': 0.0
        }

    def process_dynamic_objects(
        self,
        preds: Dict[str, Any],
        vggt_batch: Dict[str, Any],
        auxiliary_models: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理动态物体的主要接口

        Args:
            preds: VGGT模型预测结果
            vggt_batch: VGGT批次数据
            auxiliary_models: 辅助模型（如SAM2）

        Returns:
            包含动态物体和静态背景的字典
        """
        start_time = time.time()
        stage_times = {}

        try:
            # 获取基本信息
            images = vggt_batch['images']  # [B, S, 3, H, W]
            B, S, C, H, W = images.shape

            # 获取sky_masks（如果有的话）
            sky_masks = vggt_batch.get('sky_masks', None)  # [B, S, H, W] or None

            # ========== Stage 1: 数据预处理 ==========
            preprocessing_start = time.time()
            preds = self._preprocess_predictions(preds, images, B, S, H, W)
            velocity = preds.get('velocity')
            gaussian_params = preds.get('gaussian_params')
            stage_times['preprocessing'] = time.time() - preprocessing_start

            # ========== Stage 2: 聚类 + 背景分离 ==========
            clustering_start = time.time()
            clustering_results = self._perform_clustering_with_existing_method(preds, vggt_batch, velocity)
            static_gaussians = self._create_static_background_from_labels(preds, clustering_results, H, W, S, sky_masks)
            stage_times['clustering_background'] = time.time() - clustering_start

            # ========== Stage 3: 跨帧跟踪 ==========
            tracking_start = time.time()
            matched_clustering_results = match_objects_across_frames(
                clustering_results,
                position_threshold=self.tracking_position_threshold,
                velocity_threshold=self.tracking_velocity_threshold
            )
            stage_times['tracking'] = time.time() - tracking_start

            # ========== Stage 4: 动态物体聚合 ==========
            aggregation_start = time.time()
            dynamic_objects = self._aggregate_dynamic_objects(
                matched_clustering_results, preds, vggt_batch, gaussian_params, H, W
            )
            stage_times['aggregation'] = time.time() - aggregation_start

            # 计算总时间和统计
            total_time = time.time() - start_time
            self._update_stats(len(dynamic_objects), total_time)

            if self.memory_efficient:
                torch.cuda.empty_cache()

            return {
                'dynamic_objects': dynamic_objects,
                'static_gaussians': static_gaussians,
                'processing_time': total_time,
                'num_objects': len(dynamic_objects),
                'stage_times': stage_times,
                'matched_clustering_results': matched_clustering_results
            }

        except Exception as e:
            _print_main(f"❌ 动态物体处理失败: {e}")
            import traceback
            traceback.print_exc()
            return {'dynamic_objects': [], 'static_gaussians': None, 'stage_times': {}}

    def _preprocess_predictions(
        self,
        preds: Dict[str, Any],
        images: torch.Tensor,
        B: int, S: int, H: int, W: int
    ) -> Dict[str, Any]:
        """预处理VGGT预测结果：处理velocity和gaussian参数"""
        preds_updated = preds.copy()

        # 获取相机参数（一次性获取，避免重复计算）
        extrinsics, intrinsics = None, None
        if 'pose_enc' in preds:
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                preds["pose_enc"], images.shape[-2:]
            )
            # 添加齐次坐标行
            extrinsics = torch.cat([
                extrinsics,
                torch.tensor([0, 0, 0, 1], device=extrinsics.device)[
                    None, None, None, :].repeat(1, extrinsics.shape[1], 1, 1)
            ], dim=-2)

        # 处理velocity
        velocity = preds.get('velocity')
        if velocity is not None:
            velocity_processed = torch.sign(velocity) * (torch.exp(torch.abs(velocity)) - 1)

            # 转换到全局坐标系
            if extrinsics is not None:
                from vggt.training.loss import velocity_local_to_global
                extrinsic_inv = torch.linalg.inv(extrinsics)

                if len(velocity_processed.shape) == 5:  # [B, S, H, W, 3]
                    B_v, S_v, H_v, W_v, _ = velocity_processed.shape
                    velocity_flat = velocity_processed.reshape(B_v, S_v * H_v * W_v, 3)

                    velocity_transformed = []
                    for b in range(B_v):
                        vel_b = velocity_flat[b].reshape(-1, 3)
                        vel_transformed = velocity_local_to_global(vel_b, extrinsic_inv[b:b+1])
                        velocity_transformed.append(vel_transformed.reshape(S_v, H_v, W_v, 3))

                    velocity_processed = torch.stack(velocity_transformed, dim=0)

            preds_updated['velocity'] = velocity_processed

        # 处理gaussian参数
        gaussian_params = preds.get('gaussian_params')
        if gaussian_params is not None:
            # 重新整形为 [S, H*W, 14]
            if gaussian_params.dim() == 3 and gaussian_params.shape[1] == S * H * W:
                gaussian_params_reshaped = gaussian_params[0].reshape(S, H * W, 14)
            elif gaussian_params.dim() == 3 and gaussian_params.shape[0] == S:
                gaussian_params_reshaped = gaussian_params
            else:
                gaussian_params_reshaped = gaussian_params.reshape(S, H * W, 14)

            # 用depth计算的3D坐标替换前三维
            depth_data = preds.get('depth')
            if depth_data is not None and extrinsics is not None:
                try:
                    from vggt.training.loss import depth_to_world_points

                    depth_for_points = depth_data.reshape(B*S, H, W, 1)
                    world_points = depth_to_world_points(depth_for_points, intrinsics)
                    world_points = world_points.reshape(world_points.shape[0], -1, 3)

                    extrinsic_inv = torch.linalg.inv(extrinsics)
                    xyz_camera = torch.matmul(
                        extrinsic_inv[0, :, :3, :3],
                        world_points.transpose(-1, -2)
                    ).transpose(-1, -2) + extrinsic_inv[0, :, :3, 3:4].transpose(-1, -2)
                    xyz_camera = xyz_camera.reshape(S, H * W, 3)

                    gaussian_params_reshaped[:, :, :3] = xyz_camera
                except:
                    pass  # 静默失败

            preds_updated['gaussian_params'] = gaussian_params_reshaped.unsqueeze(0).reshape(B, S * H * W, 14)

        return preds_updated

    def _aggregate_dynamic_objects(
        self,
        matched_clustering_results: List[Dict],
        preds: Dict[str, Any],
        vggt_batch: Dict[str, Any],
        gaussian_params: torch.Tensor,
        H: int, W: int
    ) -> List[Dict]:
        """聚合动态物体（使用光流或简单方法）"""
        if not matched_clustering_results:
            return []

        # 尝试使用光流聚合
        if self.optical_flow_registration is not None:
            try:
                dynamic_objects, _ = self._aggregate_with_existing_optical_flow_method(
                    matched_clustering_results, preds, vggt_batch
                )
                return dynamic_objects
            except Exception as e:
                _print_main(f"  光流聚合失败，使用简单方法: {e}")

        # 回退到简单方法
        return self._create_objects_from_clustering_results(
            matched_clustering_results, gaussian_params, H, W, preds
        )

    def _update_stats(self, num_objects: int, total_time: float):
        """更新处理统计信息"""
        self.processing_stats['total_sequences'] += 1
        self.processing_stats['total_objects_detected'] += num_objects
        self.processing_stats['total_processing_time'] += total_time

    def _perform_clustering_with_existing_method(
        self,
        preds: Dict[str, Any],
        vggt_batch: Dict[str, Any],
        velocity: Optional[torch.Tensor]
    ) -> List[Dict]:
        """使用demo_video_with_pointcloud_save.py中的聚类方法"""
        try:
            # 从VGGT预测结果中提取点云坐标
            if 'depth' in preds and 'pose_enc' in preds:
                # 使用预测的相机参数（与demo一致）
                from vggt.training.loss import depth_to_world_points, velocity_local_to_global
                from vggt.utils.pose_enc import pose_encoding_to_extri_intri

                depths = preds['depth']  # 可能是 [B, S, H, W] 或 [S, H, W, 1]
                # 获取预测的相机参数
                pose_result = pose_encoding_to_extri_intri(
                    preds["pose_enc"], vggt_batch["images"].shape[-2:]
                )
                if len(pose_result) != 2:
                    raise ValueError(
                        f"pose_encoding_to_extri_intri returned {len(pose_result)} values, expected 2")
                extrinsics, intrinsics = pose_result
                # 添加齐次坐标行
                extrinsics = torch.cat([
                    extrinsics,
                    torch.tensor([0, 0, 0, 1], device=extrinsics.device)[
                        None, None, None, :].repeat(1, extrinsics.shape[1], 1, 1)
                ], dim=-2)

                # 处理不同的depth形状
                if len(depths.shape) == 5 and depths.shape[-1] == 1:
                    # 形状为 [B, S, H, W, 1]，转换为 [B, S, H, W]
                    B, S, H, W, _ = depths.shape
                    depths = depths.squeeze(-1)  # [B, S, H, W]
                elif len(depths.shape) == 4 and depths.shape[-1] == 1:
                    # 形状为 [S, H, W, 1]，转换为 [B, S, H, W]
                    S, H, W, _ = depths.shape
                    B = 1
                    depths = depths.squeeze(-1).unsqueeze(0)  # [B, S, H, W]
                elif len(depths.shape) == 4:
                    # 已经是 [B, S, H, W] 格式
                    B, S, H, W = depths.shape
                else:
                    raise ValueError(f"Unexpected depth shape: {depths.shape}")

                # 计算世界坐标点云
                # 先添加最后一个维度到depth以匹配函数期望的[N, H, W, 1]格式
                depth_with_dim = depths.reshape(B*S, H, W, 1)  # [B*S, H, W, 1]
                xyz_world = depth_to_world_points(
                    depth_with_dim,
                    intrinsics.reshape(B*S, 3, 3)
                )  # [B*S, H*W, 3]

                # 重新整形为 [B, S, H*W, 3]
                xyz_world = xyz_world.reshape(B, S, H*W, 3)

                # 转换到相机坐标系（与demo一致）
                extrinsic_inv = torch.linalg.inv(extrinsics)  # [B, S, 4, 4]
                xyz_world_flat = xyz_world[0]  # [S, H*W, 3]
                # 使用第一个batch的extrinsic_inv进行坐标变换
                extrinsic_inv_first = extrinsic_inv[0]  # [S, 4, 4]
                xyz = torch.matmul(extrinsic_inv_first[:, :3, :3], xyz_world_flat.transpose(-1, -2)).transpose(-1, -2) + \
                    extrinsic_inv_first[:, :3,
                                        3:4].transpose(-1, -2)  # [S, H*W, 3]

            else:
                B, S = velocity.shape[0], velocity.shape[1] if velocity is not None else 4
                H, W = 224, 224  # 默认尺寸
                xyz = torch.zeros(S, H*W, 3, device=self.device)

            # 处理速度信息（已经在函数开头处理过了，这里只需要reshape）
            if velocity is not None:
                if len(velocity.shape) == 5:
                    B, S, H, W, _ = velocity.shape
                    velocity_reshaped = velocity[0].reshape(
                        S, H*W, 3)  # 取第一个batch: [S, H*W, 3]
                elif len(velocity.shape) == 4:
                    # Handle case where velocity is [B, S, H*W, 3] format
                    B, S, HW, _ = velocity.shape
                    velocity_reshaped = velocity[0]  # [S, H*W, 3]
                else:
                    raise ValueError(
                        f"Unexpected velocity shape: {velocity.shape}")
                # 注意：速度后处理和坐标变换已经在process_dynamic_objects开头完成
            else:
                velocity_reshaped = torch.zeros(S, H*W, 3, device=self.device)

            # 获取confidence掩码
            conf_mask = None
            if 'depth_conf_mask' in preds and preds['depth_conf_mask'] is not None:
                conf_mask = preds['depth_conf_mask']  # [B, S, H, W]
                if len(conf_mask.shape) == 4:
                    conf_mask = conf_mask[0]  # [S, H, W]
                conf_mask = conf_mask.reshape(S, H*W)  # [S, H*W]

            # 获取gt_scale用于将velocity转换到metric尺度
            gt_scale = vggt_batch.get('depth_scale_factor', 1.0)
            if isinstance(gt_scale, torch.Tensor):
                gt_scale = gt_scale[0] if gt_scale.ndim > 0 else gt_scale

            # 使用直接定义的动态物体聚类函数
            # 分离张量梯度以便在聚类中使用numpy
            xyz_detached = xyz.detach()
            velocity_detached = velocity_reshaped.detach()

            clustering_results = dynamic_object_clustering(
                xyz_detached,  # [S, H*W, 3]
                velocity_detached,  # [S, H*W, 3] (非metric尺度)
                velocity_threshold=self.velocity_threshold,  # 速度阈值(metric尺度, m/s)
                eps=self.clustering_eps,  # DBSCAN的邻域半径
                min_samples=self.clustering_min_samples,  # DBSCAN的最小样本数
                area_threshold=self.min_object_size,  # 面积阈值
                conf_mask=conf_mask,  # confidence掩码
                gt_scale=gt_scale  # GT scale factor
            )

            return clustering_results

        except Exception as e:
            # 返回空的聚类结果
            return []


    def _aggregate_with_existing_optical_flow_method(
        self,
        clustering_results: List[Dict],
        preds: Dict[str, Any],
        vggt_batch: Dict[str, Any]
    ) -> tuple[List[Dict], Dict[str, float]]:
        """使用optical_flow_registration.py中的光流聚合方法，返回结果和详细时间统计"""
        import time
        method_start = time.time()
        detailed_times = {}

        try:
            if self.optical_flow_registration is None:
                # 获取图像尺寸
                H, W = 64, 64  # 默认值，实际应该从clustering结果或其他地方获取
                if clustering_results and len(clustering_results) > 0:
                    points = clustering_results[0].get('points')
                    if points is not None and len(points.shape) >= 2:
                        # 假设points是[H*W, 3]格式，我们可能需要从别处获取H, W
                        # 这里使用默认值，实际情况可能需要调整
                        pass
                fallback_start = time.time()
                result = self._create_objects_from_clustering_results(
                    clustering_results, None, H, W
                )
                detailed_times['回退到简单聚合'] = time.time() - fallback_start
                return result, detailed_times

            # 1. 预计算所有帧之间的光流
            flow_start = time.time()
            flows = self.optical_flow_registration.precompute_optical_flows(
                vggt_batch)
            flow_time = time.time() - flow_start
            detailed_times['1. 预计算光流'] = flow_time

            # 2. 获取所有全局物体ID
            ids_start = time.time()
            all_global_ids = set()
            for result in clustering_results:
                all_global_ids.update(result.get('global_ids', []))
            ids_time = time.time() - ids_start
            detailed_times['2. 获取全局ID'] = ids_time

            dynamic_objects = []

            # 3. 对每个全局物体进行光流聚合
            aggregation_start = time.time()
            individual_object_times = []

            for i, global_id in enumerate(all_global_ids):
                object_start = time.time()
                try:
                    aggregated_object = self.optical_flow_registration.aggregate_object_to_middle_frame(
                        clustering_results, preds, vggt_batch, global_id, flows
                    )
                    object_time = time.time() - object_start
                    individual_object_times.append(object_time)

                    if aggregated_object is not None:
                        # 使用aggregate_object_to_middle_frame已经提取的canonical_gaussians
                        aggregated_gaussians = aggregated_object.get(
                            'canonical_gaussians')

                        # 获取变换信息
                        reference_frame = aggregated_object.get(
                            'middle_frame', 0)  # 修正：使用middle_frame
                        transformations = aggregated_object.get(
                            'transformations', {})  # 各帧到reference_frame的变换
                        object_frames = aggregated_object.get(
                            'object_frames', [])

                        # 构建frame_transforms字典（Stage2Loss期望的格式）
                        frame_transforms = {}
                        for frame_idx in object_frames:
                            if frame_idx in transformations:
                                # 直接使用光流聚合器计算的变换矩阵
                                transform_data = transformations[frame_idx]
                                if isinstance(transform_data, dict) and 'transformation' in transform_data:
                                    transform = transform_data['transformation']
                                else:
                                    transform = transform_data

                                # 转换为torch tensor
                                if isinstance(transform, np.ndarray):
                                    transform = torch.from_numpy(
                                        transform).to(self.device).float()

                                # 关键修复：验证变换矩阵，防止大白球问题
                                if self._validate_and_fix_transform(transform, frame_idx, global_id):
                                    frame_transforms[frame_idx] = transform
                                else:
                                    _print_main(
                                        f"跳过对象{global_id}在帧{frame_idx}的异常变换矩阵")
                            elif frame_idx == reference_frame:
                                # reference_frame到自己的变换是恒等变换
                                frame_transforms[frame_idx] = torch.eye(
                                    4, device=self.device)

                        # 创建frame_existence标记
                        max_frame = max(
                            object_frames) if object_frames else reference_frame
                        frame_existence = []
                        for frame_idx in range(max_frame + 1):
                            frame_existence.append(frame_idx in object_frames)

                        # 转换为我们需要的格式 - 直接构建Stage2Loss期望的结构
                        dynamic_objects.append({
                            'object_id': global_id,
                            'canonical_gaussians': aggregated_gaussians,  # canonical空间的高斯参数
                            'reference_frame': reference_frame,  # 正规空间位于第几帧
                            'frame_transforms': frame_transforms,  # 其他帧和正规空间帧的transform
                            'frame_existence': torch.tensor(frame_existence, dtype=torch.bool, device=self.device),
                            'frame_gaussians': aggregated_object.get('frame_gaussians', {}),  # 新增：每帧的原始Gaussian参数
                            # 保留原始数据供调试使用
                            'aggregated_points': aggregated_object.get('aggregated_points'),
                            'aggregated_colors': aggregated_object.get('aggregated_colors'),
                            'transformations': transformations,  # 原始变换数据
                        })
                    else:
                        _print_main(f"物体 {global_id}: 聚合失败，aggregated_object为None")
                except Exception as e:
                    _print_main(f"物体 {global_id}: 聚合过程中出现异常: {e}")
                    import traceback
                    traceback.print_exc()

            aggregation_total_time = time.time() - aggregation_start
            detailed_times['3. 聚合所有物体'] = aggregation_total_time
            if individual_object_times:
                detailed_times['3.1 单物体平均耗时'] = sum(individual_object_times) / len(individual_object_times)
                detailed_times['3.2 单物体最大耗时'] = max(individual_object_times)
                detailed_times['3.3 物体数量'] = len(all_global_ids)

            method_total_time = time.time() - method_start
            detailed_times['总耗时'] = method_total_time

            return dynamic_objects, detailed_times

        except Exception as e:
            # 使用默认尺寸
            H, W = 64, 64
            detailed_times['异常回退'] = time.time() - method_start
            return self._create_objects_from_clustering_results(clustering_results, None, H, W), detailed_times

    def _create_objects_from_clustering_results(
        self,
        clustering_results: List[Dict],
        gaussian_params: Optional[torch.Tensor] = None,
        H: int = None,
        W: int = None,
        preds: Optional[Dict] = None
    ) -> List[Dict]:
        """从聚类结果创建动态物体（无光流聚合）"""
        dynamic_objects = []

        try:
            # 获取所有全局物体ID
            all_global_ids = set()
            for result in clustering_results:
                all_global_ids.update(result.get('global_ids', []))

            for global_id in all_global_ids:
                # 收集该物体在所有帧中的点云和索引
                object_points = []
                object_frames = []
                object_indices = []

                for frame_idx, result in enumerate(clustering_results):
                    global_ids = result.get('global_ids', [])
                    if global_id in global_ids:
                        cluster_idx = global_ids.index(global_id)
                        points = result['points']
                        labels = result['labels']
                        cluster_indices = result.get('cluster_indices', [])

                        # 提取物体点云
                        object_mask = labels == cluster_idx
                        frame_object_points = points[object_mask]

                        # 提取对应的点索引（从VGGT预测中提取gaussian参数需要）
                        frame_point_indices = []
                        if cluster_idx < len(cluster_indices):
                            frame_point_indices = cluster_indices[cluster_idx]

                        if len(frame_object_points) > 0:
                            object_points.append(frame_object_points)
                            object_frames.append(frame_idx)
                            object_indices.append(frame_point_indices)

                if object_points:
                    # 简单聚合：使用中间帧的点云
                    middle_idx = len(object_points) // 2
                    aggregated_points = object_points[middle_idx]

                    # 尝试从VGGT预测中提取gaussian参数
                    if preds is not None:
                        # 这里没有aggregated_colors，因为_create_objects_from_clustering_results
                        # 处理的是原始聚类结果，不是光流聚合结果
                        aggregated_gaussian = self._extract_gaussian_params_from_preds(
                            aggregated_points, preds, None
                        )
                        if aggregated_gaussian is None:
                            # 回退方案
                            aggregated_gaussian = self._points_to_gaussian_params_fallback(
                                aggregated_points, global_id)
                    elif gaussian_params is not None and H is not None and W is not None:
                        middle_frame_idx = object_frames[middle_idx]
                        middle_point_indices = object_indices[middle_idx]
                        aggregated_gaussian = self._extract_gaussian_params_from_vggt(
                            middle_point_indices, middle_frame_idx, gaussian_params, H, W
                        )
                        if aggregated_gaussian is None:
                            # 回退方案
                            aggregated_gaussian = self._points_to_gaussian_params_fallback(
                                aggregated_points, global_id)
                    else:
                        # 回退方案
                        aggregated_gaussian = self._points_to_gaussian_params_fallback(
                            aggregated_points, global_id)

                    # 为Stage2Refiner创建每帧的Gaussian参数和初始变换
                    frame_gaussians_dict = {}  # 改为字典格式
                    initial_transforms = []
                    for i, (frame_points, frame_idx, point_indices) in enumerate(zip(object_points, object_frames, object_indices)):
                        # 尝试从VGGT提取该帧的gaussian参数
                        if preds is not None:
                            frame_gaussian = self._extract_gaussian_params_from_preds(
                                frame_points, preds, None
                            )
                            if frame_gaussian is None:
                                frame_gaussian = self._points_to_gaussian_params_fallback(
                                    frame_points, global_id)
                        elif gaussian_params is not None and H is not None and W is not None:
                            frame_gaussian = self._extract_gaussian_params_from_vggt(
                                point_indices, frame_idx, gaussian_params, H, W
                            )
                            if frame_gaussian is None:
                                frame_gaussian = self._points_to_gaussian_params_fallback(
                                    frame_points, global_id)
                        else:
                            frame_gaussian = self._points_to_gaussian_params_fallback(
                                frame_points, global_id)

                        # 使用frame_idx作为key存储到字典中
                        frame_gaussians_dict[frame_idx] = (
                            frame_gaussian if frame_gaussian is not None else aggregated_gaussian)
                        # 创建单位变换矩阵作为初始变换
                        transform = torch.eye(4, device=self.device)
                        initial_transforms.append(transform)

                    dynamic_objects.append({
                        'object_id': global_id,
                        'aggregated_points': aggregated_points,
                        'aggregated_gaussians': aggregated_gaussian,  # Stage2Refiner需要的字段
                        'frame_gaussians': frame_gaussians_dict,  # Stage2Refiner需要的字段（字典格式）
                        'initial_transforms': initial_transforms,  # Stage2Refiner需要的字段
                        'reference_frame': object_frames[middle_idx],
                        'gaussian_params': aggregated_gaussian,  # 保留原字段以兼容
                        'num_frames': len(object_frames)
                    })

            return dynamic_objects

        except Exception as e:
            return []

    def _extract_gaussian_params_from_vggt(
        self,
        point_indices: List[int],
        frame_idx: int,
        vggt_gaussian_params: torch.Tensor,
        H: int, W: int
    ) -> Optional[torch.Tensor]:
        """从VGGT预测的gaussian_params中提取对应点的参数"""
        try:
            if not point_indices or len(point_indices) == 0:
                return None

            if vggt_gaussian_params is None:
                return None

            # vggt_gaussian_params应该是 [S, H*W, 14] 格式（已经经过activation）
            if len(vggt_gaussian_params.shape) != 3:
                return None

            S, HW, feature_dim = vggt_gaussian_params.shape
            if frame_idx >= S:
                return None

            # 提取该帧对应点的gaussian参数
            frame_gaussians = vggt_gaussian_params[frame_idx]  # [H*W, 14]

            # 根据点索引提取对应的gaussian参数
            selected_gaussians = []
            for idx in point_indices:
                if 0 <= idx < HW:
                    selected_gaussians.append(frame_gaussians[idx])

            if not selected_gaussians:
                return None

            # 堆叠成 [N, 14] 张量
            gaussian_params = torch.stack(selected_gaussians, dim=0)
            return gaussian_params

        except Exception as e:
            return None

    def _points_to_gaussian_params(self, aggregated_object, preds=None) -> Optional[torch.Tensor]:
        """将聚合物体转换为Gaussian参数，使用真实的VGGT预测参数"""
        if aggregated_object is None:
            return None

        # 优先从聚合物体中获取真实的Gaussian参数
        if isinstance(aggregated_object, dict) and 'aggregated_gaussians' in aggregated_object:
            aggregated_gaussians = aggregated_object['aggregated_gaussians']
            if aggregated_gaussians is not None:
                if isinstance(aggregated_gaussians, np.ndarray):
                    return torch.from_numpy(aggregated_gaussians).to(self.device).float()
                elif isinstance(aggregated_gaussians, torch.Tensor):
                    return aggregated_gaussians.to(self.device).float()

        # 如果有点云信息，尝试从VGGT预测中找到对应的Gaussian参数
        points = None
        if isinstance(aggregated_object, dict):
            points = aggregated_object.get('aggregated_points', [])
        else:
            points = aggregated_object

        if points is None or len(points) == 0:
            return None

        # 从VGGT预测中提取对应的Gaussian参数
        if preds is not None and 'gaussian_params' in preds:
            # 尝试从aggregated_object获取颜色信息
            aggregated_colors = None
            if isinstance(aggregated_object, dict):
                aggregated_colors = aggregated_object.get('aggregated_colors')
            return self._extract_gaussian_params_from_preds(points, preds, aggregated_colors)

        # 最后的回退方案
        # 尝试从aggregated_object获取object_id
        object_id = None
        if isinstance(aggregated_object, dict):
            object_id = aggregated_object.get(
                'object_id') or aggregated_object.get('global_id')
        return self._points_to_gaussian_params_fallback(points, object_id)

    def _points_to_gaussian_params_correct(self, aggregated_object, preds, clustering_results, global_id) -> Optional[torch.Tensor]:
        """正确的方法：通过像素索引直接对应Gaussian参数，而不是空间最近邻匹配"""
        try:
            if preds is None or 'gaussian_params' not in preds:
                return None

            gaussian_params = preds['gaussian_params']  # [B, S*H*W, 14]

            # 从aggregated_object中获取参考帧信息
            reference_frame = aggregated_object.get('middle_frame', 0)
            aggregated_points = aggregated_object.get('aggregated_points', [])

            if len(aggregated_points) == 0:
                return None

            # 从clustering_results中找到对应参考帧的聚类结果
            # 注意：clustering_results的索引可能与frame_idx不同
            reference_clustering = None


            # 方法1: 直接通过frame_idx匹配
            for result in clustering_results:
                frame_idx = result.get('frame_idx')
                if frame_idx == reference_frame:
                    reference_clustering = result
                    break

            # 方法2: 如果没找到，尝试通过索引匹配（reference_frame可能是相对索引）
            if reference_clustering is None and 0 <= reference_frame < len(clustering_results):
                reference_clustering = clustering_results[reference_frame]

            if reference_clustering is None:
                return self._points_to_gaussian_params_fallback(aggregated_points, global_id)

            # 获取该物体在参考帧中的像素索引
            global_ids = reference_clustering.get('global_ids', [])
            cluster_indices = reference_clustering.get('cluster_indices', [])

            # 找到属于该global_id的像素索引
            # cluster_indices是一个list of lists，每个元素是一个聚类的像素索引列表
            object_pixel_indices = []
            for i, gid in enumerate(global_ids):
                if gid == global_id:
                    # cluster_indices[i] 是该聚类的所有像素索引列表
                    if i < len(cluster_indices):
                        object_pixel_indices.extend(cluster_indices[i])

            if len(object_pixel_indices) == 0:
                return self._points_to_gaussian_params_fallback(aggregated_points, global_id)


            # 直接通过像素索引提取对应的Gaussian参数
            B, N_total, feature_dim = gaussian_params.shape

            # gaussian_params的形状是 [B, S*H*W, 14]，我们需要计算正确的全局索引
            # cluster_indices中的像素索引是相对于单帧的（0到H*W-1），需要转换为全局索引

            # 首先，我们需要推断H, W和S
            # 从clustering_results推断出H*W
            H_W = len(reference_clustering.get('points', []))
            if H_W == 0:
                return self._points_to_gaussian_params_fallback(aggregated_points, global_id)

            # 从N_total和H_W推断S
            S = N_total // H_W if H_W > 0 else 1

            selected_gaussians_list = []

            for pixel_idx in object_pixel_indices:
                # 计算在全局flatten结构中的索引
                # 全局索引 = reference_frame * H*W + pixel_idx
                global_idx = reference_frame * H_W + pixel_idx

                if 0 <= global_idx < N_total:
                    selected_gaussians_list.append(
                        gaussian_params[0, global_idx])  # 使用batch=0
                else:
                    pass  # 索引超出范围，跳过

            if len(selected_gaussians_list) == 0:
                return self._points_to_gaussian_params_fallback(aggregated_points, global_id)

            selected_gaussians = torch.stack(
                selected_gaussians_list, dim=0)  # [N, 14]

            # 激活Gaussian参数
            selected_gaussians = self._apply_gaussian_activation(
                selected_gaussians)

            # 使用聚合后的点云位置替换Gaussian的位置参数
            points_tensor = torch.from_numpy(
                aggregated_points).to(self.device).float()

            # 如果点数不匹配，取较小的数量
            min_count = min(len(selected_gaussians), len(points_tensor))
            selected_gaussians = selected_gaussians[:min_count]
            points_tensor = points_tensor[:min_count]

            selected_gaussians[:, :3] = points_tensor[:, :3]


            return selected_gaussians

        except Exception as e:
            return self._points_to_gaussian_params_fallback(aggregated_object.get('aggregated_points', []), global_id)

    def _extract_gaussian_params_from_preds(self, points, preds, aggregated_colors=None) -> Optional[torch.Tensor]:
        """从VGGT预测中提取对应点云的真实Gaussian参数，优先使用聚合颜色"""
        try:
            if 'gaussian_params' not in preds or points is None or len(points) == 0:
                return None

            gaussian_params = preds['gaussian_params']  # [B, S*H*W, 14]

            # 确保points是torch.Tensor
            if isinstance(points, np.ndarray):
                points = torch.from_numpy(points).to(self.device).float()
            elif isinstance(points, list):
                points = torch.tensor(
                    points, device=self.device, dtype=torch.float32)
            else:
                points = points.to(self.device).float()

            # gaussian_params的shape: [B, S*H*W, 14]
            # 我们需要找到与points最匹配的Gaussian参数
            B, N_total, feature_dim = gaussian_params.shape

            # 重塑gaussian_params为[B*S*H*W, 14]以便处理
            # [B*S*H*W, 14]
            gaussian_params_flat = gaussian_params.view(-1, feature_dim)

            # 获取Gaussian的位置信息（前3维）
            gaussian_positions = gaussian_params_flat[:, :3]  # [B*S*H*W, 3]

            # 使用KD-tree或最近邻搜索找到对应的Gaussian参数
            from sklearn.neighbors import NearestNeighbors

            # 将位置数据转换为numpy进行KD-tree搜索
            gaussian_pos_np = gaussian_positions.detach().cpu().numpy()
            points_np = points[:, :3].detach().cpu().numpy()

            # 修复大白球问题：避免选择相同的Gaussian参数
            N_points = len(points_np)
            N_gaussians = len(gaussian_pos_np)

            if N_gaussians < N_points:
                # 如果Gaussian数量少于点数，随机采样避免重复
                selected_indices = np.random.choice(
                    N_gaussians, N_points, replace=True)
                selected_gaussians = gaussian_params_flat[selected_indices]
            else:
                # 使用KD-tree但确保每个点都有独特的参数
                nbrs = NearestNeighbors(n_neighbors=min(
                    5, N_gaussians), algorithm='kd_tree').fit(gaussian_pos_np)
                distances, indices = nbrs.kneighbors(points_np)

                # 为每个点分配不同的Gaussian参数，避免重复
                selected_indices = []
                used_indices = set()

                for i in range(N_points):
                    # 对于每个点，从它的最近邻中选择一个未使用的
                    candidates = indices[i]  # k个最近邻的索引

                    # 优先选择未使用的索引
                    selected_idx = None
                    for candidate in candidates:
                        if candidate not in used_indices:
                            selected_idx = candidate
                            break

                    # 如果所有候选都被使用了，选择距离最近的
                    if selected_idx is None:
                        selected_idx = candidates[0]

                    selected_indices.append(selected_idx)
                    used_indices.add(selected_idx)

                selected_gaussians = gaussian_params_flat[selected_indices]

            # 关键修复：对从VGGT提取的原始参数进行激活处理
            # 因为VGGT预测的是原始未激活的参数，需要应用激活函数
            selected_gaussians = self._apply_gaussian_activation(
                selected_gaussians)

            # 使用聚合后的点云位置替换Gaussian的位置参数（激活后）
            selected_gaussians[:, :3] = points[:, :3]

            # 保持VGGT预测参数的原始性，不添加随机扰动
            # 大白球问题主要通过确保选择不同的Gaussian参数来解决

            # 保持VGGT预测的颜色参数，不用光流聚合的颜色替换
            # VGGT预测的颜色参数经过神经网络训练，适合3D Gaussian Splatting渲染
            # 光流聚合的颜色适合传统点云，但不适合Gaussian渲染


            return selected_gaussians

        except Exception as e:
            # 注意：不要再次激活，因为回退方案中已经会激活
            return self._points_to_gaussian_params_fallback(points, None)

    def _points_to_gaussian_params_fallback(self, points, object_id=None) -> Optional[torch.Tensor]:
        """回退方案：当无法从VGGT提取时，生成基本的Gaussian参数"""
        try:
            if points is None or len(points) == 0:
                return None

            # 确保points是torch.Tensor并在正确设备上
            if isinstance(points, np.ndarray):
                points = torch.from_numpy(points).to(self.device)
            elif isinstance(points, torch.Tensor):
                points = points.to(self.device)
            else:
                return None

            N = len(points)
            # 创建基本的Gaussian参数 [N, 14]: xyz(3) + scale(3) + color(3) + rotation(4) + opacity(1)
            gaussian_params = torch.zeros(
                N, 14, device=self.device, dtype=torch.float32)

            # 位置: xyz (positions 0:3)
            gaussian_params[:, :3] = points[:, :3]

            # 尺度: scale (positions 3:6) - raw values before activation
            gaussian_params[:, 3:6] = torch.log(torch.tensor(
                0.01 / 0.05))  # Will become 0.01 after activation

            # 颜色: color (positions 6:9) - 使用基于object_id的一致颜色
            if object_id is not None:
                # 基于object_id生成一致的颜色
                import math
                hue = (object_id * 137.5) % 360  # 黄金角度分割，确保颜色差异明显
                saturation = 0.8
                value = 0.9

                # HSV到RGB转换
                h = hue / 60.0
                c = value * saturation
                x = c * (1 - abs((h % 2) - 1))
                m = value - c

                if 0 <= h < 1:
                    r, g, b = c, x, 0
                elif 1 <= h < 2:
                    r, g, b = x, c, 0
                elif 2 <= h < 3:
                    r, g, b = 0, c, x
                elif 3 <= h < 4:
                    r, g, b = 0, x, c
                elif 4 <= h < 5:
                    r, g, b = x, 0, c
                else:
                    r, g, b = c, 0, x

                color = torch.tensor([r + m, g + m, b + m], device=self.device)
                gaussian_params[:, 6:9] = color.unsqueeze(0).repeat(N, 1)
                _print_main(
                    f"回退方案：为object_id={object_id}生成一致颜色 RGB=({r+m:.3f}, {g+m:.3f}, {b+m:.3f})")
            else:
                # 默认中性颜色
                gaussian_params[:, 6:9] = 0.5

            # 旋转: quaternion (positions 9:13) - normalized quaternion
            gaussian_params[:, 9:13] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=points.device)  # w, x, y, z

            # 不透明度: opacity (position 13) - raw value before sigmoid
            gaussian_params[:, 13] = torch.logit(torch.tensor(
                0.8, device=points.device))  # Will become 0.8 after sigmoid

            # Apply activation functions to get final parameters
            gaussian_params = self._apply_gaussian_activation(gaussian_params)

            return gaussian_params

        except Exception as e:
            return None

    def _apply_gaussian_activation(self, gaussian_params: torch.Tensor) -> torch.Tensor:
        """
        Apply activation functions to gaussian parameters
        Following the same post-processing as in src/vggt/training/loss.py self_render_and_loss

        Args:
            gaussian_params: [*, 14] tensor with raw gaussian parameters

        Returns:
            gaussian_params: [*, 14] tensor with activated parameters
        """
        if gaussian_params is None:
            return None

        # Clone to avoid modifying original
        processed_params = gaussian_params.clone()

        # Scale activation: (0.05 * exp(scale)).clamp_max(0.3) - applied to positions 3:6
        scale_raw = processed_params[..., 3:6]
        scale_activated = (0.05 * torch.exp(scale_raw)).clamp_max(0.3)
        processed_params[..., 3:6] = scale_activated

        # Color: positions 6:9 - no activation needed (kept as is)

        # Rotation quaternion normalization: positions 9:13
        rotations = processed_params[..., 9:13]
        rotation_norms = torch.norm(rotations, dim=-1, keepdim=True)
        rotation_norms = torch.clamp(rotation_norms, min=1e-8)
        processed_params[..., 9:13] = rotations / rotation_norms

        # Opacity activation: sigmoid for position 13
        opacity_raw = processed_params[..., 13:14]
        opacities = torch.sigmoid(opacity_raw)
        processed_params[..., 13:14] = opacities

        return processed_params

    def _validate_and_fix_transform(self, transform: torch.Tensor, frame_idx: int, global_id: int = None) -> bool:
        """验证和修复变换矩阵，防止大白球问题"""
        try:
            # 检查基本形状
            if transform.shape != (4, 4):
                _print_main(f"⚠️  变换矩阵形状异常: {transform.shape}, 期望 (4,4)")
                return False

            # 检查是否为零矩阵
            if torch.allclose(transform, torch.zeros_like(transform), atol=1e-8):
                _print_main(f"⚠️  对象{global_id}帧{frame_idx}: 检测到零变换矩阵！这会导致大白球问题")
                return False

            # 检查是否有NaN或Inf
            if torch.isnan(transform).any() or torch.isinf(transform).any():
                _print_main(f"⚠️  对象{global_id}帧{frame_idx}: 变换矩阵包含NaN或Inf值")
                return False

            # 检查旋转部分的行列式
            rotation_part = transform[:3, :3]
            det = torch.det(rotation_part)

            if det.abs() < 1e-6:
                _print_main(f"⚠️  对象{global_id}帧{frame_idx}: 变换矩阵奇异 (det={det:.2e})")
                return False

            # 检查是否过度缩放
            scales = torch.linalg.norm(rotation_part, dim=0)  # 各轴的缩放
            if scales.max() > 100 or scales.min() < 0.01:
                _print_main(
                    f"⚠️  对象{global_id}帧{frame_idx}: 异常缩放 {scales}, 可能导致渲染问题")
                return False

            # 检查平移是否过大
            translation = transform[:3, 3]
            if torch.norm(translation) > 1000:
                _print_main(
                    f"⚠️  对象{global_id}帧{frame_idx}: 平移过大 {translation}, 可能超出相机视野")
                # 这种情况仍然保留，但给出警告

            return True

        except Exception as e:
            _print_main(f"❌ 变换矩阵验证失败: {e}")
            return False

    def _create_static_background_from_labels(
        self,
        preds: Dict[str, Any],
        clustering_results: List[Dict],
        H: int, W: int, S: int,
        sky_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """直接使用动态聚类结果中的full_labels提取静态背景（标签为-1的gaussian），并过滤sky区域"""
        try:
            # 获取gaussian参数
            gaussian_params = preds.get('gaussian_params')
            if gaussian_params is None:
                return self._create_default_static_background(H, W)

            # 重新整形Gaussian参数为 [S, H*W, 14]
            if gaussian_params.dim() == 3 and gaussian_params.shape[1] == S * H * W:
                # [B, S*H*W, 14] -> [S, H*W, 14]
                gaussian_params = gaussian_params[0].reshape(S, H * W, 14)
            elif gaussian_params.dim() == 3 and gaussian_params.shape[0] == S:
                # [S, H*W, 14] -> 已经是正确形状
                pass
            else:
                # 其他情况，尝试重新整形
                gaussian_params = gaussian_params.reshape(S, H * W, 14)

            static_gaussians = []

            for s, clustering_result in enumerate(clustering_results):
                if s >= S:
                    break

                # 获取当前帧的full_labels，-1表示静态点
                full_labels = clustering_result['labels']  # [H*W]

                # 创建静态mask（标签为-1的点）
                static_mask = full_labels == -1  # [H*W]

                # 如果有sky_masks，过滤掉sky区域的点
                if sky_masks is not None:
                    # sky_masks: [B, S, H, W] -> [H*W]
                    sky_mask_frame = sky_masks[0, s].reshape(-1)  # [H*W]
                    # 确保sky_mask_frame是布尔类型并在正确的设备上
                    if sky_mask_frame.dtype != torch.bool:
                        sky_mask_frame = sky_mask_frame.bool()
                    sky_mask_frame = sky_mask_frame.to(static_mask.device)
                    # sky区域不应该被包含在静态背景中
                    static_mask = static_mask & (~sky_mask_frame)  # [H*W]

                # 获取当前帧的gaussian参数
                frame_gaussians = gaussian_params[s]  # [H*W, 14]

                # 提取静态gaussian
                static_frame_gaussians = frame_gaussians[static_mask]  # [N_static, 14]

                static_gaussians.append(static_frame_gaussians)

            # 合并所有帧的静态gaussian
            if static_gaussians:
                static_gaussians = torch.cat(static_gaussians, dim=0)
            else:
                static_gaussians = torch.empty(0, 14, device=self.device)

            return static_gaussians

        except Exception as e:
            # 回退到默认方法
            _print_main(f"⚠️  静态背景提取失败: {e}，回退到默认方法")
            import traceback
            traceback.print_exc()
            return self._create_default_static_background(H, W)

    def _create_default_static_background(self, H: int, W: int) -> torch.Tensor:
        """创建默认静态背景（回退方案）"""
        try:
            num_background_points = min(1000, H * W // 100)
            background_gaussians = torch.zeros(
                num_background_points, 14, device=self.device)

            # 随机分布在3D空间中
            background_gaussians[:, :3] = torch.randn(
                num_background_points, 3, device=self.device) * 2.0

            # 旋转（单位四元数）
            background_gaussians[:, 3:7] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=self.device)

            # 尺度
            background_gaussians[:, 7:10] = 0.1

            # 不透明度
            background_gaussians[:, 10] = 0.1

            # 颜色（灰色）
            background_gaussians[:, 11:14] = 0.3

            return background_gaussians

        except Exception as e:
            return torch.zeros(100, 14, device=self.device)

    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        total_time = self.processing_stats['total_processing_time']
        total_sequences = max(self.processing_stats['total_sequences'], 1)

        return {
            'total_sequences_processed': self.processing_stats['total_sequences'],
            'total_objects_detected': self.processing_stats['total_objects_detected'],
            'avg_objects_per_sequence': self.processing_stats['total_objects_detected'] / total_sequences,
            'avg_processing_time': total_time / total_sequences,
            'sam_time_ratio': self.processing_stats['sam_time'] / max(total_time, 1e-6),
            'optical_flow_time_ratio': self.processing_stats['optical_flow_time'] / max(total_time, 1e-6),
            'aggregation_time_ratio': self.processing_stats['aggregation_time'] / max(total_time, 1e-6)
        }

    def clear_cache(self):
        """清理缓存"""
        if self.temporal_cache:
            self.temporal_cache.clear()

        # 清理GPU内存
        if self.memory_efficient:
            torch.cuda.empty_cache()
