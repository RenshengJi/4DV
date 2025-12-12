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
import torch.distributed as dist
import numpy as np
from typing import Dict, List, Optional, Any
import time
from collections import defaultdict

# 聚类和匹配算法
import sys
import os
from cuml.cluster import DBSCAN
from sklearn.cluster import DBSCAN as SklearnDBSCAN
import cupy as cp
from scipy.optimize import linear_sum_assignment


def _is_main_process():
    """检查是否为主进程（用于DDP训练）"""
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _print_main(*args, **kwargs):
    """只在主进程打印（用于DDP训练）"""
    if _is_main_process():
        print(*args, **kwargs)



def dynamic_object_clustering(xyz, velocity, velocity_threshold=0.01, eps=0.02, min_samples=10, area_threshold=750, gt_scale=None):
    """
    对每一帧进行动态物体聚类

    Args:
        xyz: [S, H*W, 3] 点云坐标（非metric尺度，保留梯度）
        velocity: [S, H*W, 3] 速度向量（非metric尺度，保留梯度）
        velocity_threshold: 速度阈值（metric尺度，m/s），用于过滤静态背景
        eps: DBSCAN的邻域半径（metric尺度，米）
        min_samples: DBSCAN的最小样本数
        area_threshold: 面积阈值，过滤掉面积小于此值的聚类
        gt_scale: float or tensor - GT scale factor，用于将xyz和velocity转换到metric尺度

    Returns:
        list: 每一帧的聚类结果，每个元素包含点云坐标和聚类标签
              注意：返回的 'points' 是非metric尺度，但保留梯度，可以用于反向传播！
              聚类过程在metric空间进行，因此eps参数是真实的物理距离（米）
    """
    clustering_results = []
    device = xyz.device

    # 确保 gt_scale 是标量
    if gt_scale is None:
        gt_scale = 1.0
    if isinstance(gt_scale, torch.Tensor):
        gt_scale = gt_scale.item() if gt_scale.numel() == 1 else float(gt_scale)

    for frame_idx in range(xyz.shape[0]):
        # 获取当前帧的点云和速度（保留梯度！）
        frame_points = xyz[frame_idx]  # [H*W, 3] - 保留梯度
        frame_velocity = velocity[frame_idx]  # [H*W, 3] - 保留梯度

        # 将velocity转换到metric尺度: velocity_metric = velocity / gt_scale
        frame_velocity_metric = frame_velocity / gt_scale

        # 计算速度大小（metric尺度，m/s）
        velocity_magnitude = torch.norm(frame_velocity_metric, dim=-1)  # [H*W]

        # 过滤动态点（速度大于阈值的点，阈值现在是metric尺度）
        dynamic_mask = velocity_magnitude > velocity_threshold

        dynamic_points = frame_points[dynamic_mask]  # [N_dynamic, 3] - 非metric尺度
        dynamic_velocities = frame_velocity[dynamic_mask]  # [N_dynamic, 3] - 非metric尺度

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

        # 将动态点转换到metric尺度（用于DBSCAN聚类）
        # 这样eps参数就是真实的物理距离（米）
        dynamic_points_metric = dynamic_points / gt_scale  # [N_dynamic, 3] - metric尺度

        # 尝试使用cuML GPU加速的DBSCAN聚类
        try:
            # 转换为CuPy数组（使用metric尺度的坐标）
            dynamic_points_metric_cp = cp.asarray(dynamic_points_metric.detach())
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels_cp = dbscan.fit_predict(dynamic_points_metric_cp)
            # 直接转换为PyTorch tensor（零拷贝，使用DLPack协议）
            cluster_labels = torch.as_tensor(cluster_labels_cp, device='cuda')
        except Exception as e:
            # 回退到sklearn CPU版本
            try:
                dynamic_points_metric_np = dynamic_points_metric.detach().cpu().numpy()
                dbscan_sklearn = SklearnDBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels_np = dbscan_sklearn.fit_predict(dynamic_points_metric_np)
                cluster_labels = torch.from_numpy(cluster_labels_np).to(device)
            except Exception as sklearn_e:
                # 简单回退：所有点标记为噪声
                cluster_labels = torch.full((len(dynamic_points),), -1, dtype=torch.long, device=device)

        # 将聚类结果映射回原始点云
        full_labels = torch.full((len(frame_points),), -1, device=frame_points.device)
        full_labels[dynamic_mask] = cluster_labels.to(frame_points.device).long()

        # 统计聚类数量（排除噪声点，标签为-1）
        all_unique_labels = set(cluster_labels.cpu().numpy().tolist())
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
            # 这些只用于统计计算，不需要梯度
            cluster_points = dynamic_points[cluster_mask].detach()
            cluster_vel = dynamic_velocities[cluster_mask].detach()

            # 计算聚类中心（平均位置和平均速度）
            center = cluster_points.mean(dim=0)
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
                cluster_indices = torch.where(cluster_mask)[0]
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


def classify_dynamic_objects(
    matched_clustering_results: List[Dict],
    segment_logits: torch.Tensor,
    H: int,
    W: int
) -> Dict[int, str]:
    """
    对跨帧追踪后的每个动态物体进行分类（车辆 vs 行人）

    利用预测的语义分割信息，对每个物体所有帧中的所有像素的分割logits求和，
    然后统一softmax，实现分类。

    Waymo segmentation classes:
    - 0: background
    - 1: vehicle (car)
    - 2: sign
    - 3: pedestrian + cyclist (people)

    Args:
        matched_clustering_results: 跨帧追踪后的聚类结果
        segment_logits: [B, S, H, W, 4] 语义分割预测logits
        H, W: 图像高宽

    Returns:
        object_classes: {global_id: 'car' or 'people'} 每个物体的类别
    """
    device = segment_logits.device
    object_classes = {}

    # 收集所有全局物体ID
    all_global_ids = set()
    for result in matched_clustering_results:
        all_global_ids.update(result.get('global_ids', []))

    # 对每个全局物体进行分类
    for global_id in all_global_ids:
        # 收集该物体在所有帧中的像素索引
        all_pixel_logits = []

        for frame_idx, result in enumerate(matched_clustering_results):
            global_ids = result.get('global_ids', [])
            if global_id not in global_ids:
                continue

            # 找到该物体在当前帧的聚类索引
            cluster_idx = global_ids.index(global_id)

            # 获取该聚类的像素索引
            cluster_indices = result.get('cluster_indices', [])
            if cluster_idx >= len(cluster_indices):
                continue

            pixel_indices = cluster_indices[cluster_idx]
            if not pixel_indices:
                continue

            # 获取当前帧的分割logits: [H, W, 4]
            frame_logits = segment_logits[0, frame_idx]  # [H, W, 4]

            # 将pixel_indices转换为2D坐标
            pixel_indices_tensor = torch.tensor(pixel_indices, dtype=torch.long, device=device)
            v_coords = pixel_indices_tensor // W  # 行坐标
            u_coords = pixel_indices_tensor % W   # 列坐标

            # 过滤在图像范围内的坐标
            valid = (v_coords >= 0) & (v_coords < H) & (u_coords >= 0) & (u_coords < W)
            v_valid = v_coords[valid]
            u_valid = u_coords[valid]

            # 提取这些像素的logits
            pixel_logits = frame_logits[v_valid, u_valid]  # [N_pixels, 4]
            all_pixel_logits.append(pixel_logits.detach())  # detach以避免影响segmentation训练

        if not all_pixel_logits:
            # 没有像素数据，默认为car
            object_classes[global_id] = 'car'
            continue

        # 合并所有帧的logits并求和
        all_pixel_logits = torch.cat(all_pixel_logits, dim=0)  # [Total_pixels, 4]
        summed_logits = all_pixel_logits.sum(dim=0)  # [4]

        # Softmax得到概率分布
        probs = torch.softmax(summed_logits, dim=0)  # [4]

        # 分类决策：
        # - 如果vehicle (class 1) 概率最高 -> 'car'
        # - 如果pedestrian+cyclist (class 3) 概率最高 -> 'people'
        # - 其他情况默认为'car'
        pred_class = torch.argmax(probs).item()  # 转换为1-based class

        if pred_class == 1:  # vehicle
            object_classes[global_id] = 'car'
        elif pred_class == 3:  # pedestrian + cyclist
            object_classes[global_id] = 'people'
        else:
            # background或sign，默认为car
            object_classes[global_id] = 'car'

    return object_classes


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

                        # 综合评分（位置权重更高）
                        score = pos_distance 

                        # 如果满足阈值条件，设置成本；否则保持无穷大
                        if pos_distance < position_threshold: 
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



def _import_velocity_registration():
    """导入velocity配准模块"""
    try:
        import importlib.util

        # 获取路径
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src_dir = os.path.join(root_dir, 'src')
        registration_path = os.path.join(src_dir, "velocity_registration.py")

        # 检查文件是否存在
        if not os.path.exists(registration_path):
            _print_main(f"Error: velocity_registration.py not found at: {registration_path}")
            return None

        # 添加必要的路径到sys.path
        vggt_dir = os.path.join(src_dir, 'vggt')
        raft_core_dir = os.path.join(src_dir, 'SEA-RAFT', 'core')
        for path in [root_dir, src_dir, vggt_dir, raft_core_dir]:
            if path not in sys.path:
                sys.path.insert(0, path)

        # 动态加载模块
        spec = importlib.util.spec_from_file_location("velocity_registration", registration_path)
        if spec is None or spec.loader is None:
            _print_main(f"Error: Failed to create module spec")
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 返回VelocityBasedRegistration类
        if not hasattr(module, 'VelocityBasedRegistration'):
            _print_main(f"Error: VelocityBasedRegistration class not found")
            return None

        return module.VelocityBasedRegistration
    except Exception as e:
        _print_main(f"Failed to import velocity registration: {e}")
        import traceback
        traceback.print_exc()
        return None


class OnlineDynamicProcessor:
    """
    在线动态物体处理器

    实时执行：
    1. 动态物体检测和聚类
    2. 跨帧物体跟踪
    3. Velocity配准和聚合
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
        velocity_transform_mode: str = "simple",  # velocity变换模式: "simple"或"procrustes"
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
            use_optical_flow_aggregation: 是否使用velocity聚合
            velocity_transform_mode: velocity变换模式
                - "simple": 仅用velocity平均值估计平移T，旋转R为单位矩阵（快速）
                - "procrustes": 使用xyz+velocity，用Procrustes算法估计完整R和T（更准确）
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
        self.velocity_transform_mode = velocity_transform_mode

        # 初始化velocity配准系统（必须成功）
        self.optical_flow_registration = None
        if use_optical_flow_aggregation:
            # 导入VelocityBasedRegistration类
            VelocityRegistrationClass = _import_velocity_registration()
            if VelocityRegistrationClass is None:
                raise RuntimeError(
                    "Failed to import VelocityBasedRegistration class. "
                    "Please ensure velocity_registration.py exists and is accessible."
                )

            # 初始化velocity配准系统
            try:
                self.optical_flow_registration = VelocityRegistrationClass(
                    device=str(device),
                    min_inliers_ratio=0.1,
                    velocity_transform_mode=velocity_transform_mode
                )
                _print_main("✓ Velocity-based registration initialized successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize VelocityBasedRegistration: {e}") from e

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
            # 从preds获取批次维度信息（已在vggt.py的forward中保存）
            batch_dims = preds.get('batch_dims', {})
            B = batch_dims.get('B', 1)
            S = batch_dims.get('S', 4)
            H = batch_dims.get('H', 224)
            W = batch_dims.get('W', 224)

            # 获取images用于预处理
            images = vggt_batch.get('images')  # [B, S, 3, H, W]

            # 获取sky_masks（如果有的话）
            sky_masks = vggt_batch.get('sky_masks', None)  # [B, S, H, W] or None

            # ========== Stage 1: 数据预处理 ==========
            preprocessing_start = time.time()
            preds = self._preprocess_predictions(preds, vggt_batch, images)

            # 保存sh_degree和gaussian_output_dim到实例变量，供其他方法使用
            self.sh_degree = preds.get('sh_degree', 0)
            self.sh_dim = 3 * ((self.sh_degree + 1) ** 2)
            self.gaussian_output_dim = 11 + self.sh_dim

            stage_times['preprocessing'] = time.time() - preprocessing_start

            # ========== Stage 2: 聚类 + 背景分离 ==========
            clustering_start = time.time()
            clustering_results = self._perform_clustering_with_existing_method(preds, vggt_batch)
            static_gaussians = self._create_static_background_from_labels(preds, clustering_results, sky_masks)
            stage_times['clustering_background'] = time.time() - clustering_start

            # ========== Stage 3: 跨帧跟踪 ==========
            tracking_start = time.time()
            matched_clustering_results = match_objects_across_frames(
                clustering_results,
                position_threshold=self.tracking_position_threshold,
                velocity_threshold=self.tracking_velocity_threshold
            )
            stage_times['tracking'] = time.time() - tracking_start

            # ========== Stage 3.5: 物体分类（车辆 vs 行人）==========
            classification_start = time.time()
            segment_logits = preds.get('segment_logits')  # [B, S, H, W, 4]

            if segment_logits is not None:
                object_classes = classify_dynamic_objects(
                    matched_clustering_results, segment_logits, H, W
                )
            else:
                # 如果没有分割信息，所有物体默认为car
                all_global_ids = set()
                for result in matched_clustering_results:
                    all_global_ids.update(result.get('global_ids', []))
                object_classes = {gid: 'car' for gid in all_global_ids}

            stage_times['classification'] = time.time() - classification_start

            # ========== Stage 4: 动态物体聚合（仅对车辆）==========
            aggregation_start = time.time()

            # 分离车辆和行人的ID
            car_ids = [gid for gid, cls in object_classes.items() if cls == 'car']
            people_ids = [gid for gid, cls in object_classes.items() if cls == 'people']

            # 只对车辆进行聚合
            dynamic_objects_cars = self._aggregate_dynamic_objects(
                matched_clustering_results, preds, vggt_batch, object_ids_to_process=car_ids
            )

            # 对行人，提取每帧单独的Gaussians（不进行聚合）
            dynamic_objects_people = self._extract_people_objects(
                matched_clustering_results, preds, vggt_batch, object_ids_to_process=people_ids
            )

            stage_times['aggregation'] = time.time() - aggregation_start

            # 计算总时间和统计
            total_time = time.time() - start_time
            total_objects = len(dynamic_objects_cars) + len(dynamic_objects_people)
            self._update_stats(total_objects, total_time)

            if self.memory_efficient:
                torch.cuda.empty_cache()

            return {
                'dynamic_objects_cars': dynamic_objects_cars,
                'dynamic_objects_people': dynamic_objects_people,
                'static_gaussians': static_gaussians,
                'processing_time': total_time,
                'num_objects': total_objects,
                'num_cars': len(dynamic_objects_cars),
                'num_people': len(dynamic_objects_people),
                'stage_times': stage_times,
                'matched_clustering_results': matched_clustering_results,
                'object_classes': object_classes
            }

        except Exception as e:
            _print_main(f"❌ 动态物体处理失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'dynamic_objects_cars': [],
                'dynamic_objects_people': [],
                'static_gaussians': None,
                'stage_times': {}
            }

    def _preprocess_predictions(
        self,
        preds: Dict[str, Any],
        vggt_batch: Dict[str, Any],
        images: torch.Tensor
    ) -> Dict[str, Any]:
        """预处理VGGT预测结果：处理velocity和gaussian参数"""
        preds_updated = preds.copy()

        # 从preds获取批次维度信息
        batch_dims = preds['batch_dims']
        B = batch_dims['B']
        S = batch_dims['S']
        H = batch_dims['H']
        W = batch_dims['W']

        # 获取sh_degree并计算gaussian参数维度
        sh_degree = preds.get('sh_degree', 0)  # 默认为0
        sh_dim = 3 * ((sh_degree + 1) ** 2)
        gaussian_output_dim = 11 + sh_dim  # xyz(3) + scale(3) + sh(sh_dim) + rotation(4) + opacity(1)

        # 处理gaussian参数：VGGT输出是 [B, S, H, W, gaussian_output_dim]
        gaussian_params = preds['gaussian_params']  # [B, S, H, W, gaussian_output_dim]
        gaussian_params_reshaped = gaussian_params[0].reshape(S, H * W, gaussian_output_dim)  # [S, H*W, gaussian_output_dim]

        # 用vggt.py中计算的xyz_camera替换前三维（避免重复计算）
        xyz_camera = preds['xyz_camera'][0]  # [S, H*W, 3]
        gaussian_params_reshaped[:, :, :3] = xyz_camera

        preds_updated['gaussian_params'] = gaussian_params_reshaped.reshape(1, S, H, W, gaussian_output_dim)  # [B, S, H, W, gaussian_output_dim]

        return preds_updated

    def _aggregate_dynamic_objects(
        self,
        matched_clustering_results: List[Dict],
        preds: Dict[str, Any],
        vggt_batch: Dict[str, Any],
        object_ids_to_process: Optional[List[int]] = None
    ) -> List[Dict]:
        """聚合动态物体（必须使用velocity配准）

        Args:
            matched_clustering_results: 跨帧追踪后的聚类结果
            preds: VGGT模型预测结果
            vggt_batch: VGGT批次数据
            object_ids_to_process: 要处理的物体ID列表，如果为None则处理所有物体
        """
        if not matched_clustering_results:
            return []

        # 必须使用velocity配准
        if self.optical_flow_registration is None:
            raise RuntimeError("Velocity registration is not initialized! Must use velocity-based aggregation.")

        dynamic_objects, _ = self._aggregate_with_existing_optical_flow_method(
            matched_clustering_results, preds, vggt_batch, object_ids_to_process
        )
        return dynamic_objects

    def _update_stats(self, num_objects: int, total_time: float):
        """更新处理统计信息"""
        self.processing_stats['total_sequences'] += 1
        self.processing_stats['total_objects_detected'] += num_objects
        self.processing_stats['total_processing_time'] += total_time

    def _perform_clustering_with_existing_method(
        self,
        preds: Dict[str, Any],
        vggt_batch: Dict[str, Any]
    ) -> List[Dict]:
        """使用demo_video_with_pointcloud_save.py中的聚类方法"""
        try:
            # 从preds获取批次维度信息（已在vggt.py的forward中保存）
            batch_dims = preds['batch_dims']
            B = batch_dims['B']
            S = batch_dims['S']
            H = batch_dims['H']
            W = batch_dims['W']

            # 获取xyz_camera（已在vggt.py中计算）
            xyz_camera = preds['xyz_camera']  # [B, S, H*W, 3]
            xyz = xyz_camera[0]  # [S, H*W, 3]

            # 获取velocity_global（已在vggt.py中计算并转换到全局坐标系）
            velocity_global = preds['velocity_global']  # [B, S, H, W, 3] or [B, S, H*W, 3]
            velocity_reshaped = velocity_global[0]  # [S, H, W, 3] or [S, H*W, 3]

            # 统一为 [S, H*W, 3] 格式
            if velocity_reshaped.ndim == 4:  # [S, H, W, 3]
                velocity_reshaped = velocity_reshaped.reshape(S, H * W, 3)

            # 获取gt_scale用于将velocity转换到metric尺度
            gt_scale = vggt_batch.get('depth_scale_factor', 1.0)
            if isinstance(gt_scale, torch.Tensor):
                gt_scale = gt_scale[0] if gt_scale.ndim > 0 else gt_scale

            # 使用直接定义的动态物体聚类函数
            clustering_results = dynamic_object_clustering(
                xyz,  # [S, H*W, 3] - 保留梯度
                velocity_reshaped,  # [S, H*W, 3] - 保留梯度
                velocity_threshold=self.velocity_threshold,  # 速度阈值(metric尺度, m/s)
                eps=self.clustering_eps,  # DBSCAN的邻域半径
                min_samples=self.clustering_min_samples,  # DBSCAN的最小样本数
                area_threshold=self.min_object_size,  # 面积阈值
                gt_scale=gt_scale  # GT scale factor
            )

            return clustering_results

        except Exception as e:
            _print_main(f"⚠️  聚类失败: {e}")
            import traceback
            traceback.print_exc()
            return []


    def _aggregate_with_existing_optical_flow_method(
        self,
        clustering_results: List[Dict],
        preds: Dict[str, Any],
        vggt_batch: Dict[str, Any],
        object_ids_to_process: Optional[List[int]] = None
    ) -> tuple[List[Dict], Dict[str, float]]:
        """使用velocity_registration.py中的聚合方法，返回结果和详细时间统计

        Args:
            clustering_results: 聚类结果
            preds: 预测结果
            vggt_batch: 批次数据
            object_ids_to_process: 要处理的物体ID列表，如果为None则处理所有物体
        """
        import time
        method_start = time.time()
        detailed_times = {}

        try:
            # 获取所有全局物体ID
            ids_start = time.time()
            all_global_ids = set()
            for result in clustering_results:
                all_global_ids.update(result.get('global_ids', []))

            # 如果指定了要处理的ID，则过滤
            if object_ids_to_process is not None:
                all_global_ids = all_global_ids & set(object_ids_to_process)

            ids_time = time.time() - ids_start
            detailed_times['1. 获取全局ID'] = ids_time

            dynamic_objects = []

            # 对每个全局物体进行聚合
            aggregation_start = time.time()
            individual_object_times = []

            for i, global_id in enumerate(all_global_ids):   #TODO: parallel
                object_start = time.time()
                try:
                    aggregated_object = self.optical_flow_registration.aggregate_object_to_middle_frame(
                        clustering_results, preds, vggt_batch, global_id
                    )
                    object_time = time.time() - object_start
                    individual_object_times.append(object_time)

                    if aggregated_object is not None:
                        # 使用aggregate_object_to_middle_frame已经提取的canonical_gaussians
                        aggregated_gaussians = aggregated_object.get(
                            'canonical_gaussians')

                        if aggregated_gaussians is None:
                            _print_main(f"⚠️ Object {global_id}: aggregated_object中没有canonical_gaussians")

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
                                # 直接使用velocity配准器计算的变换矩阵
                                transform_data = transformations[frame_idx]
                                if isinstance(transform_data, dict) and 'transformation' in transform_data:
                                    transform = transform_data['transformation']
                                else:
                                    transform = transform_data

                                # 转换为torch tensor
                                if isinstance(transform, np.ndarray):
                                    transform = torch.from_numpy(
                                        transform).to(self.device).float()

                                frame_transforms[frame_idx] = transform
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

                        # 提取每帧的pixel indices用于创建dynamic mask
                        point_indices = aggregated_object.get('point_indices', [])
                        frame_pixel_indices = {}  # {frame_idx: [pixel_idx1, pixel_idx2, ...]}
                        for frame_idx, pixel_idx in point_indices:
                            if frame_idx not in frame_pixel_indices:
                                frame_pixel_indices[frame_idx] = []
                            frame_pixel_indices[frame_idx].append(pixel_idx)

                        # 转换为我们需要的格式 - 直接构建Stage2Loss期望的结构
                        dynamic_objects.append({
                            'object_id': global_id,
                            'canonical_gaussians': aggregated_gaussians,  # canonical空间的高斯参数
                            'reference_frame': reference_frame,  # 正规空间位于第几帧
                            'frame_transforms': frame_transforms,  # 其他帧和正规空间帧的transform
                            'frame_existence': torch.tensor(frame_existence, dtype=torch.bool, device=self.device),
                            'frame_gaussians': aggregated_object.get('frame_gaussians', {}),  # 新增：每帧的原始Gaussian参数
                            'frame_pixel_indices': frame_pixel_indices,  # 新增：每帧的2D pixel索引用于创建mask
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
            detailed_times['2. 聚合所有物体'] = aggregation_total_time
            if individual_object_times:
                detailed_times['2.1 单物体平均耗时'] = sum(individual_object_times) / len(individual_object_times)
                detailed_times['2.2 单物体最大耗时'] = max(individual_object_times)
                detailed_times['2.3 物体数量'] = len(all_global_ids)

            method_total_time = time.time() - method_start
            detailed_times['总耗时'] = method_total_time

            return dynamic_objects, detailed_times

        except Exception as e:
            # 异常时直接返回空列表
            detailed_times['异常回退'] = time.time() - method_start
            _print_main(f"❌ 聚合失败: {e}")
            import traceback
            traceback.print_exc()
            return [], detailed_times

    def _extract_people_objects(
        self,
        matched_clustering_results: List[Dict],
        preds: Dict[str, Any],
        vggt_batch: Dict[str, Any],
        object_ids_to_process: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        提取行人物体的每帧Gaussians（不进行聚合）

        对于行人，我们不使用刚体假设，因此不进行跨帧聚合。
        而是直接提取每帧中该物体的Gaussian参数。

        Args:
            matched_clustering_results: 跨帧追踪后的聚类结果
            preds: VGGT模型预测结果
            vggt_batch: VGGT批次数据
            object_ids_to_process: 要处理的行人物体ID列表

        Returns:
            people_objects: 行人物体列表，每个包含每帧的Gaussians
        """
        if not matched_clustering_results:
            return []

        # 获取批次维度信息
        batch_dims = preds['batch_dims']
        H = batch_dims['H']
        W = batch_dims['W']
        S = batch_dims['S']

        # 获取gaussian参数 [B, S, H, W, gaussian_output_dim]
        gaussian_params = preds['gaussian_params'][0]  # [S, H, W, gaussian_output_dim]

        # 获取所有全局物体ID
        all_global_ids = set()
        for result in matched_clustering_results:
            all_global_ids.update(result.get('global_ids', []))

        # 如果指定了要处理的ID，则过滤
        if object_ids_to_process is not None:
            all_global_ids = all_global_ids & set(object_ids_to_process)

        people_objects = []

        # 对每个行人物体
        for global_id in all_global_ids:
            # 存储每帧的Gaussians
            frame_gaussians = {}
            frame_pixel_indices = {}
            object_frames = []

            # 遍历所有帧
            for frame_idx, result in enumerate(matched_clustering_results):
                global_ids = result.get('global_ids', [])
                if global_id not in global_ids:
                    continue

                # 找到该物体在当前帧的聚类索引
                cluster_idx = global_ids.index(global_id)

                # 获取该聚类的像素索引
                cluster_indices = result.get('cluster_indices', [])
                if cluster_idx >= len(cluster_indices):
                    continue

                pixel_indices = cluster_indices[cluster_idx]
                if not pixel_indices:
                    continue

                # 记录该物体在当前帧存在
                object_frames.append(frame_idx)

                # 将pixel_indices转换为2D坐标
                pixel_indices_tensor = torch.tensor(pixel_indices, dtype=torch.long, device=self.device)
                v_coords = pixel_indices_tensor // W  # 行坐标
                u_coords = pixel_indices_tensor % W   # 列坐标

                # 过滤在图像范围内的坐标
                valid = (v_coords >= 0) & (v_coords < H) & (u_coords >= 0) & (u_coords < W)
                v_valid = v_coords[valid]
                u_valid = u_coords[valid]

                # 提取当前帧该物体的Gaussian参数
                frame_gaussian_params = gaussian_params[frame_idx]  # [H, W, gaussian_output_dim]
                object_gaussians = frame_gaussian_params[v_valid, u_valid]  # [N_points, gaussian_output_dim]

                # 存储该帧的Gaussians
                frame_gaussians[frame_idx] = object_gaussians
                frame_pixel_indices[frame_idx] = pixel_indices

            if not object_frames:
                continue

            # 创建frame_existence标记
            max_frame = max(object_frames)
            frame_existence = []
            for frame_idx in range(max_frame + 1):
                frame_existence.append(frame_idx in object_frames)

            # 构建行人物体数据结构
            people_objects.append({
                'object_id': global_id,
                'frame_gaussians': frame_gaussians,  # {frame_idx: gaussians} 每帧独立的Gaussians
                'frame_pixel_indices': frame_pixel_indices,  # {frame_idx: [pixel_idx1, ...]}
                'frame_existence': torch.tensor(frame_existence, dtype=torch.bool, device=self.device),
                'object_frames': object_frames,
                'is_people': True  # 标记为行人物体
            })

        return people_objects

    def _points_to_gaussian_params(self, aggregated_object, preds=None) -> Optional[torch.Tensor]:
        """将聚合物体转换为Gaussian参数，使用真实的VGGT预测参数"""
        if aggregated_object is None:
            return None

        # 从aggregated_object提取数据
        if not isinstance(aggregated_object, dict):
            return None

        # 优先使用已聚合的Gaussian参数
        aggregated_gaussians = aggregated_object.get('aggregated_gaussians')
        if aggregated_gaussians is not None:
            if isinstance(aggregated_gaussians, np.ndarray):
                return torch.from_numpy(aggregated_gaussians).to(self.device).float()
            return aggregated_gaussians.to(self.device).float()

        # 从VGGT预测中提取Gaussian参数
        points = aggregated_object.get('aggregated_points')
        if points is None or len(points) == 0 or preds is None:
            return None

        aggregated_colors = aggregated_object.get('aggregated_colors')
        return self._extract_gaussian_params_from_preds(points, preds, aggregated_colors)


    def _extract_gaussian_params_from_preds(self, points, preds, aggregated_colors=None) -> Optional[torch.Tensor]:
        """从VGGT预测中提取对应点云的真实Gaussian参数"""
        from sklearn.neighbors import NearestNeighbors

        try:
            # 转换points为torch.Tensor
            if isinstance(points, np.ndarray):
                points = torch.from_numpy(points).to(self.device).float()
            elif isinstance(points, list):
                points = torch.tensor(points, device=self.device, dtype=torch.float32)
            else:
                points = points.to(self.device).float()

            # 获取Gaussian参数并展平
            gaussian_params = preds['gaussian_params']  # [B, S*H*W, feature_dim]
            gaussian_params_flat = gaussian_params.view(-1, gaussian_params.shape[-1])

            # 使用最近邻搜索匹配Gaussian参数
            gaussian_positions = gaussian_params_flat[:, :3].detach().cpu().numpy()
            points_np = points[:, :3].detach().cpu().numpy()

            N_points = len(points_np)
            N_gaussians = len(gaussian_positions)

            if N_gaussians < N_points:
                # Gaussian数量不足，允许重复采样
                selected_indices = np.random.choice(N_gaussians, N_points, replace=True)
            else:
                # 使用KD-tree查找最近邻，每个点选择不同的Gaussian
                k_neighbors = min(5, N_gaussians)
                nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree').fit(gaussian_positions)
                _, indices = nbrs.kneighbors(points_np)

                selected_indices = []
                used_indices = set()
                for i in range(N_points):
                    # 优先选择未使用的索引
                    for candidate in indices[i]:
                        if candidate not in used_indices:
                            selected_indices.append(candidate)
                            used_indices.add(candidate)
                            break
                    else:
                        # 所有候选都已使用，选择最近的
                        selected_indices.append(indices[i][0])

            # 提取选中的Gaussian参数并更新位置
            selected_gaussians = gaussian_params_flat[selected_indices].clone()
            selected_gaussians[:, :3] = points[:, :3]

            return selected_gaussians

        except Exception as e:
            _print_main(f"⚠️ 提取Gaussian参数失败: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _create_static_background_from_labels(
        self,
        preds: Dict[str, Any],
        clustering_results: List[Dict],
        sky_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """直接使用动态聚类结果中的full_labels提取静态背景（标签为-1的gaussian），并过滤sky区域"""
        try:
            # 从preds获取批次维度信息
            batch_dims = preds['batch_dims']
            H = batch_dims['H']
            W = batch_dims['W']
            S = batch_dims['S']

            # 获取gaussian参数（必须存在）
            gaussian_params = preds['gaussian_params']  # [B, S, H, W, gaussian_output_dim]

            # 重新整形为 [S, H*W, gaussian_output_dim]
            gaussian_params = gaussian_params[0].reshape(S, H * W, self.gaussian_output_dim)

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
                frame_gaussians = gaussian_params[s]  # [H*W, gaussian_output_dim]

                # 提取静态gaussian
                static_frame_gaussians = frame_gaussians[static_mask]  # [N_static, gaussian_output_dim]

                static_gaussians.append(static_frame_gaussians)

            # 合并所有帧的静态gaussian
            if static_gaussians:
                static_gaussians = torch.cat(static_gaussians, dim=0)
            else:
                static_gaussians = torch.empty(0, self.gaussian_output_dim, device=self.device)

            return static_gaussians

        except Exception as e:
            _print_main(f"❌ 静态背景提取失败: {e}")
            import traceback
            traceback.print_exc()
            raise

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
