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
import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
import time
from collections import defaultdict

# 导入现有的聚类和光流配准系统
import sys
import os
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment

# 延迟导入以避免循环导入
def _import_clustering_functions():
    """延迟导入聚类函数"""
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # 导入单独的函数而不是整个模块
        import importlib.util
        
        # 导入聚类方法
        demo_spec = importlib.util.spec_from_file_location(
            "demo_clustering", 
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo_video_with_pointcloud_save.py")
        )
        demo_module = importlib.util.module_from_spec(demo_spec)
        demo_spec.loader.exec_module(demo_module)
        
        return demo_module.dynamic_object_clustering, demo_module.match_objects_across_frames
    except Exception as e:
        print(f"Failed to import clustering functions: {e}")
        return None, None

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
        optical_flow_module = importlib.util.module_from_spec(optical_flow_spec)
        optical_flow_spec.loader.exec_module(optical_flow_module)
        
        return optical_flow_module.OpticalFlowRegistration
    except Exception as e:
        print(f"Failed to import optical flow: {e}")
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
        velocity_threshold_percentile: float = 0.75,
        iou_threshold: float = 0.3,
        use_optical_flow_aggregation: bool = True,
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
            velocity_threshold_percentile: 速度阈值百分位数
            iou_threshold: IoU匹配阈值
            use_optical_flow_aggregation: 是否使用光流聚合
            enable_temporal_cache: 是否启用时序缓存
            cache_size: 缓存大小
        """
        self.device = device
        self.memory_efficient = memory_efficient
        self.min_object_size = min_object_size
        self.max_objects_per_frame = max_objects_per_frame
        self.velocity_threshold_percentile = velocity_threshold_percentile
        self.iou_threshold = iou_threshold
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
                        use_pnp=True,
                        min_inliers_ratio=0.1,  # 降低最小内点比例
                        ransac_threshold=5.0,   # 增加RANSAC阈值  
                        max_flow_magnitude=200.0  # 增加最大光流幅度
                    )
                except Exception as e:
                    self.optical_flow_registration = None
        
        # 缓存聚类函数
        self._dynamic_clustering_func = None
        self._match_objects_func = None
        
        # 时序缓存
        self.enable_temporal_cache = enable_temporal_cache
        self.cache_size = cache_size
        self.temporal_cache = defaultdict(list) if enable_temporal_cache else None
        
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
            images = vggt_batch.get('images')  # [B, S, 3, H, W]
            if images is None:
                return {'dynamic_objects': [], 'static_gaussians': None}
            
            B, S, C, H, W = images.shape
            velocity = preds.get('velocity')  # [B, S, H, W, 3]
            gaussian_params = preds.get('gaussian_params')  # [B, S*H*W, 14] or similar
            
            # ========== Stage 1: 数据后处理 ==========
            preprocessing_start = time.time()
            
            # 1. Velocity后处理
            if velocity is not None:
                # 应用速度后处理：velocity = sign(velocity) * (exp(|velocity|) - 1)
                velocity_processed = torch.sign(velocity) * (torch.exp(torch.abs(velocity)) - 1)
                
                # 如果有相机参数，将速度从局部坐标转换到全局坐标
                if 'pose_enc' in preds:
                    from vggt.training.loss import velocity_local_to_global
                    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
                    
                    # 获取预测的相机参数
                    extrinsics, intrinsics = pose_encoding_to_extri_intri(
                        preds["pose_enc"], vggt_batch["images"].shape[-2:]
                    )
                    # 添加齐次坐标行
                    extrinsics = torch.cat([
                        extrinsics, 
                        torch.tensor([0, 0, 0, 1], device=extrinsics.device)[None, None, None, :].repeat(1, extrinsics.shape[1], 1, 1)
                    ], dim=-2)
                    
                    # 转换速度坐标系
                    extrinsic_inv = torch.linalg.inv(extrinsics)  # [B, S, 4, 4]
                    
                    # Reshape velocity for coordinate transformation
                    if len(velocity_processed.shape) == 5:  # [B, S, H, W, 3]
                        B_v, S_v, H_v, W_v, _ = velocity_processed.shape
                        velocity_flat = velocity_processed.reshape(B_v, S_v * H_v * W_v, 3)  # [B, S*H*W, 3]
                        
                        # Transform each batch separately
                        velocity_transformed = []
                        for b in range(B_v):
                            vel_b = velocity_flat[b].reshape(-1, 3)  # [S*H*W, 3]
                            vel_transformed = velocity_local_to_global(vel_b, extrinsic_inv[b:b+1])
                            velocity_transformed.append(vel_transformed.reshape(S_v, H_v, W_v, 3))
                        
                        velocity_processed = torch.stack(velocity_transformed, dim=0)  # [B, S, H, W, 3]
                
            # 如果有数据需要后处理，复制preds避免修改原始数据
            preds_updated = preds.copy()
            
            if velocity is not None:
                # 更新处理后的velocity
                preds_updated['velocity'] = velocity_processed
                velocity = velocity_processed
                
            # 2. Gaussian参数后处理
            if gaussian_params is not None:
                # 重新整形gaussian参数为 [S, H*W, 14]
                if gaussian_params.dim() == 3 and gaussian_params.shape[1] == S * H * W:
                    # 情况1: [B, S*H*W, 14] -> [S, H*W, 14]
                    gaussian_params_reshaped = gaussian_params[0].reshape(S, H * W, 14)
                elif gaussian_params.dim() == 3 and gaussian_params.shape[0] == S:
                    # 情况2: [S, H*W, 14] -> 已经是正确形状
                    gaussian_params_reshaped = gaussian_params
                else:
                    # 其他情况，尝试重新整形
                    gaussian_params_reshaped = gaussian_params.reshape(S, H * W, 14)
                
                # 应用gaussian参数激活函数
                gaussian_params_processed = self._apply_gaussian_activation(gaussian_params_reshaped)
                
                # 用depth计算的3D坐标替换gaussian_params的前三维（参考flow_loss函数）
                depth_data = preds.get('depth')
                if depth_data is not None and 'pose_enc' in preds:
                    try:
                        from vggt.training.loss import depth_to_world_points
                        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
                        
                        # 获取相机参数
                        extrinsics, intrinsics = pose_encoding_to_extri_intri(
                            preds["pose_enc"], images.shape[-2:]
                        )
                        extrinsics = torch.cat([
                            extrinsics, 
                            torch.tensor([0, 0, 0, 1], device=extrinsics.device)[None, None, None, :].repeat(1, extrinsics.shape[1], 1, 1)
                        ], dim=-2)
                        
                        # 计算world points（与flow_loss中相同的逻辑）
                        depth_for_points = depth_data.reshape(B*S, H, W, 1)
                        world_points = depth_to_world_points(depth_for_points, intrinsics)
                        world_points = world_points.reshape(world_points.shape[0], world_points.shape[1]*world_points.shape[2], 3)
                        
                        # 转换到相机坐标系
                        extrinsic_inv = torch.linalg.inv(extrinsics)
                        xyz_camera = torch.matmul(extrinsic_inv[0, :, :3, :3], world_points.transpose(-1, -2)).transpose(-1, -2) + \
                                    extrinsic_inv[0, :, :3, 3:4].transpose(-1, -2)
                        xyz_camera = xyz_camera.reshape(S, H * W, 3)  # [S, H*W, 3]
                        
                        # 替换gaussian_params的前三维
                        gaussian_params_processed[:, :, :3] = xyz_camera
                        
                    except Exception as e:
                        pass  # 静默处理错误
                
                # 更新处理后的gaussian_params（保持原始格式）
                preds_updated['gaussian_params'] = gaussian_params_processed.unsqueeze(0).reshape(B, S * H * W, 14)
                gaussian_params = preds_updated['gaussian_params']
            
            # 使用更新后的preds
            preds = preds_updated
            stage_times['preprocessing'] = time.time() - preprocessing_start
            
            # ========== Stage 2: 动态物体聚类 ==========
            clustering_start = time.time()
            clustering_results = self._perform_clustering_with_existing_method(
                preds, vggt_batch, velocity
            )
            stage_times['Stage 2: 动态物体聚类'] = time.time() - clustering_start
            
            # ========== Stage 3: 跨帧物体跟踪 ==========
            tracking_start = time.time()
            if self._match_objects_func is None:
                _, self._match_objects_func = _import_clustering_functions()
            
            if self._match_objects_func is not None:
                matched_clustering_results = self._match_objects_func(
                    clustering_results, 
                    position_threshold=0.5, 
                    velocity_threshold=0.2
                )
            else:
                matched_clustering_results = clustering_results
            stage_times['Stage 3: 跨帧物体跟踪'] = time.time() - tracking_start
            
            # ========== Stage 4: 光流聚合 ==========
            aggregation_start = time.time()
            dynamic_objects = []
            if self.optical_flow_registration is not None and len(matched_clustering_results) > 0:
                try:
                    dynamic_objects = self._aggregate_with_existing_optical_flow_method(
                        matched_clustering_results, preds, vggt_batch
                    )
                except Exception as e:
                    print(f"光流聚合失败，回退到简单方法: {e}")
                    dynamic_objects = self._create_objects_from_clustering_results(
                        matched_clustering_results, gaussian_params, H, W, preds
                    )
            else:
                dynamic_objects = self._create_objects_from_clustering_results(
                    matched_clustering_results, gaussian_params, H, W, preds
                )
            stage_times['Stage 4: 光流聚合'] = time.time() - aggregation_start
            
            # ========== Stage 5: 背景分离 ==========
            background_start = time.time()
            static_gaussians = self._create_static_background(
                preds, velocity, matched_clustering_results, H, W, S
            )
            stage_times['Stage 5: 背景分离'] = time.time() - background_start
            
            # 显示各阶段耗时
            total_time = time.time() - start_time
            for stage_name, stage_time in stage_times.items():
                print(f"  {stage_name}: {stage_time:.4f}s")
            print(f"  总耗时: {total_time:.4f}s")
            
            # 更新统计信息
            self.processing_stats['total_sequences'] += 1
            self.processing_stats['total_objects_detected'] += len(dynamic_objects)
            self.processing_stats['total_processing_time'] += total_time
            
            # 内存清理
            if self.memory_efficient:
                torch.cuda.empty_cache()
            
            return {
                'dynamic_objects': dynamic_objects,
                'static_gaussians': static_gaussians,
                'processing_time': total_time,
                'num_objects': len(dynamic_objects),
                'stage_times': stage_times
            }
            
        except Exception as e:
            print(f"Error in dynamic object processing: {e}")
            return {'dynamic_objects': [], 'static_gaussians': None, 'stage_times': {}}
    
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
                    raise ValueError(f"pose_encoding_to_extri_intri returned {len(pose_result)} values, expected 2")
                extrinsics, intrinsics = pose_result
                # 添加齐次坐标行
                extrinsics = torch.cat([
                    extrinsics, 
                    torch.tensor([0, 0, 0, 1], device=extrinsics.device)[None, None, None, :].repeat(1, extrinsics.shape[1], 1, 1)
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
                      extrinsic_inv_first[:, :3, 3:4].transpose(-1, -2)  # [S, H*W, 3]
                
            else:
                B, S = velocity.shape[0], velocity.shape[1] if velocity is not None else 4
                H, W = 224, 224  # 默认尺寸
                xyz = torch.zeros(S, H*W, 3, device=self.device)
            
            # 处理速度信息（已经在函数开头处理过了，这里只需要reshape）
            if velocity is not None:
                if len(velocity.shape) == 5:
                    B, S, H, W, _ = velocity.shape
                    velocity_reshaped = velocity[0].reshape(S, H*W, 3)  # 取第一个batch: [S, H*W, 3]
                elif len(velocity.shape) == 4:
                    # Handle case where velocity is [B, S, H*W, 3] format  
                    B, S, HW, _ = velocity.shape
                    velocity_reshaped = velocity[0]  # [S, H*W, 3]
                else:
                    raise ValueError(f"Unexpected velocity shape: {velocity.shape}")
                # 注意：速度后处理和坐标变换已经在process_dynamic_objects开头完成
            else:
                velocity_reshaped = torch.zeros(S, H*W, 3, device=self.device)
            
            # 使用现有的动态物体聚类函数
            if self._dynamic_clustering_func is None:
                self._dynamic_clustering_func, _ = _import_clustering_functions()
            
            if self._dynamic_clustering_func is not None:
                # 分离张量梯度以便在聚类中使用numpy
                xyz_detached = xyz.detach()
                velocity_detached = velocity_reshaped.detach()
                
                clustering_results = self._dynamic_clustering_func(
                    xyz_detached,  # [S, H*W, 3]
                    velocity_detached,  # [S, H*W, 3]
                    velocity_threshold=0.01,  # 速度阈值
                    eps=0.02,  # DBSCAN的邻域半径
                    min_samples=10,  # DBSCAN的最小样本数
                    area_threshold=self.min_object_size  # 面积阈值
                )
            else:
                # 回退到简单实现
                clustering_results = self._simple_clustering(xyz, velocity_reshaped)
            
            return clustering_results
            
        except Exception as e:
            # 返回空的聚类结果
            return []
    
    def _simple_clustering(self, xyz: torch.Tensor, velocity: torch.Tensor) -> List[Dict]:
        """简单的聚类实现（作为回退方案）"""
        try:
            clustering_results = []
            S = xyz.shape[0]
            
            for frame_idx in range(S):
                frame_points = xyz[frame_idx]  # [H*W, 3]
                frame_velocity = velocity[frame_idx]  # [H*W, 3]
                
                # 计算速度大小
                velocity_magnitude = torch.norm(frame_velocity, dim=-1)  # [H*W]
                
                # 过滤动态点
                velocity_threshold = 0.01
                dynamic_mask = velocity_magnitude > velocity_threshold
                dynamic_points = frame_points[dynamic_mask]
                
                if len(dynamic_points) < 10:
                    clustering_results.append({
                        'frame_idx': frame_idx,  # 添加frame_idx字段
                        'points': frame_points,
                        'labels': torch.full((len(frame_points),), -1, dtype=torch.long),
                        'dynamic_mask': dynamic_mask,
                        'num_clusters': 0,
                        'cluster_centers': [],
                        'cluster_velocities': [],
                        'cluster_sizes': [],
                        'global_ids': [],
                        'cluster_indices': []  # 添加cluster_indices字段
                    })
                    continue
                
                # 简单的基于空间的聚类
                dynamic_points_np = dynamic_points.cpu().numpy()
                
                try:
                    # 使用DBSCAN聚类
                    dbscan = DBSCAN(eps=0.02, min_samples=10)
                    cluster_labels = dbscan.fit_predict(dynamic_points_np)
                    
                    # 映射回原始点云
                    full_labels = torch.full((len(frame_points),), -1, dtype=torch.long)
                    full_labels[dynamic_mask] = torch.from_numpy(cluster_labels)
                    
                    # 统计聚类信息
                    unique_labels = set(cluster_labels)
                    if -1 in unique_labels:
                        unique_labels.remove(-1)
                    
                    num_clusters = len(unique_labels)
                    cluster_centers = []
                    cluster_velocities = []
                    cluster_sizes = []
                    
                    for label in sorted(unique_labels):
                        cluster_mask = cluster_labels == label
                        cluster_points = dynamic_points[cluster_mask]
                        if len(cluster_points) >= self.min_object_size:
                            center = cluster_points.mean(dim=0)
                            cluster_centers.append(center)
                            cluster_velocities.append(frame_velocity[dynamic_mask][cluster_mask].mean(dim=0))
                            cluster_sizes.append(len(cluster_points))
                    
                    # 构建cluster_indices - 每个聚类中心对应的像素索引列表
                    cluster_indices = []
                    H_W = len(frame_points)
                    for label in range(len(cluster_centers)):
                        # 找到属于该聚类的所有点的索引
                        cluster_mask = (full_labels == label)
                        indices = torch.where(cluster_mask)[0].tolist()
                        cluster_indices.append(indices)
                    
                    clustering_results.append({
                        'frame_idx': frame_idx,  # 添加frame_idx字段
                        'points': frame_points,
                        'labels': full_labels,
                        'dynamic_mask': dynamic_mask,
                        'num_clusters': len(cluster_centers),
                        'cluster_centers': cluster_centers,
                        'cluster_velocities': cluster_velocities,
                        'cluster_sizes': cluster_sizes,
                        'global_ids': list(range(len(cluster_centers))),  # 简单分配ID
                        'cluster_indices': cluster_indices  # 添加每个聚类对应的像素索引列表
                    })
                    
                except Exception as e:
                    clustering_results.append({
                        'frame_idx': frame_idx,  # 添加frame_idx字段
                        'points': frame_points,
                        'labels': torch.full((len(frame_points),), -1, dtype=torch.long),
                        'dynamic_mask': dynamic_mask,
                        'num_clusters': 0,
                        'cluster_centers': [],
                        'cluster_velocities': [],
                        'cluster_sizes': [],
                        'global_ids': [],
                        'cluster_indices': []  # 添加cluster_indices字段
                    })
            
            return clustering_results
            
        except Exception as e:
            return []
    
    def _aggregate_with_existing_optical_flow_method(
        self,
        clustering_results: List[Dict],
        preds: Dict[str, Any],
        vggt_batch: Dict[str, Any]
    ) -> List[Dict]:
        """使用optical_flow_registration.py中的光流聚合方法"""
        import time
        method_start = time.time()
        
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
                return self._create_objects_from_clustering_results(
                    clustering_results, None, H, W
                )
            
            # 1. 预计算所有帧之间的光流
            flow_start = time.time()
            flows = self.optical_flow_registration.precompute_optical_flows(vggt_batch)
            flow_time = time.time() - flow_start
            print(f"    预计算光流耗时: {flow_time:.4f}s")
            
            # 2. 获取所有全局物体ID
            ids_start = time.time()
            all_global_ids = set()
            for result in clustering_results:
                all_global_ids.update(result.get('global_ids', []))
            ids_time = time.time() - ids_start
            print(f"    获取全局物体ID耗时: {ids_time:.4f}s ({len(all_global_ids)} 个物体)")
            
            dynamic_objects = []
            
            # 3. 对每个全局物体进行光流聚合
            aggregation_start = time.time()
            for i, global_id in enumerate(all_global_ids):
                object_start = time.time()
                try:
                    aggregated_object = self.optical_flow_registration.aggregate_object_to_middle_frame(
                        clustering_results, preds, vggt_batch, global_id, flows
                    )
                    object_time = time.time() - object_start
                    print(f"    物体 {global_id} ({i+1}/{len(all_global_ids)}) 聚合耗时: {object_time:.4f}s")
                    
                    if aggregated_object is not None:
                        print(f"物体 {global_id}: 聚合成功，包含 {len(aggregated_object.get('aggregated_points', []))} 个点")
                        
                        # 使用aggregate_object_to_middle_frame已经提取的canonical_gaussians
                        aggregated_gaussians = aggregated_object.get('canonical_gaussians')
                        
                        if aggregated_gaussians is not None:
                            print(f"物体 {global_id}: 成功获取到 {aggregated_gaussians.shape[0]} 个Gaussian参数")
                        else:
                            print(f"物体 {global_id}: ⚠️ 未能获取Gaussian参数")
                        
                        # 获取变换信息
                        reference_frame = aggregated_object.get('middle_frame', 0)  # 修正：使用middle_frame
                        transformations = aggregated_object.get('transformations', {})  # 各帧到reference_frame的变换
                        object_frames = aggregated_object.get('object_frames', [])
                        
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
                                    transform = torch.from_numpy(transform).to(self.device).float()
                                
                                # 关键修复：验证变换矩阵，防止大白球问题
                                if self._validate_and_fix_transform(transform, frame_idx, global_id):
                                    frame_transforms[frame_idx] = transform
                                else:
                                    print(f"跳过对象{global_id}在帧{frame_idx}的异常变换矩阵")
                            elif frame_idx == reference_frame:
                                # reference_frame到自己的变换是恒等变换
                                frame_transforms[frame_idx] = torch.eye(4, device=self.device)
                        
                        # 创建frame_existence标记
                        max_frame = max(object_frames) if object_frames else reference_frame
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
                            # 保留原始数据供调试使用
                            'aggregated_points': aggregated_object.get('aggregated_points'),
                            'aggregated_colors': aggregated_object.get('aggregated_colors'),
                            'transformations': transformations,  # 原始变换数据
                        })
                    else:
                        print(f"物体 {global_id}: 聚合失败，aggregated_object为None")
                except Exception as e:
                    print(f"物体 {global_id}: 聚合过程中出现异常: {e}")
                    import traceback
                    traceback.print_exc()
            
            aggregation_total_time = time.time() - aggregation_start
            method_total_time = time.time() - method_start
            
            print(f"    物体聚合总耗时: {aggregation_total_time:.4f}s")
            print(f"    光流聚合方法总耗时: {method_total_time:.4f}s")
            print(f"    性能分析：预计算光流({flow_time:.3f}s) + 获取ID({ids_time:.3f}s) + 物体聚合({aggregation_total_time:.3f}s)")
            print(f"    光流聚合完成: 处理了 {len(all_global_ids)} 个物体，成功聚合 {len(dynamic_objects)} 个物体")
            return dynamic_objects
            
        except Exception as e:
            # 使用默认尺寸
            H, W = 64, 64
            return self._create_objects_from_clustering_results(clustering_results, None, H, W)
    
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
                            aggregated_gaussian = self._points_to_gaussian_params_fallback(aggregated_points, global_id)
                    elif gaussian_params is not None and H is not None and W is not None:
                        middle_frame_idx = object_frames[middle_idx]
                        middle_point_indices = object_indices[middle_idx]
                        aggregated_gaussian = self._extract_gaussian_params_from_vggt(
                            middle_point_indices, middle_frame_idx, gaussian_params, H, W
                        )
                        if aggregated_gaussian is None:
                            # 回退方案
                            aggregated_gaussian = self._points_to_gaussian_params_fallback(aggregated_points, global_id)
                    else:
                        # 回退方案
                        aggregated_gaussian = self._points_to_gaussian_params_fallback(aggregated_points, global_id)
                    
                    # 为Stage2Refiner创建每帧的Gaussian参数和初始变换
                    frame_gaussians = []
                    initial_transforms = []
                    for i, (frame_points, frame_idx, point_indices) in enumerate(zip(object_points, object_frames, object_indices)):
                        # 尝试从VGGT提取该帧的gaussian参数
                        if preds is not None:
                            frame_gaussian = self._extract_gaussian_params_from_preds(
                                frame_points, preds, None
                            )
                            if frame_gaussian is None:
                                frame_gaussian = self._points_to_gaussian_params_fallback(frame_points, global_id)
                        elif gaussian_params is not None and H is not None and W is not None:
                            frame_gaussian = self._extract_gaussian_params_from_vggt(
                                point_indices, frame_idx, gaussian_params, H, W
                            )
                            if frame_gaussian is None:
                                frame_gaussian = self._points_to_gaussian_params_fallback(frame_points, global_id)
                        else:
                            frame_gaussian = self._points_to_gaussian_params_fallback(frame_points, global_id)
                        
                        frame_gaussians.append(frame_gaussian if frame_gaussian is not None else aggregated_gaussian)
                        # 创建单位变换矩阵作为初始变换
                        transform = torch.eye(4, device=self.device)
                        initial_transforms.append(transform)
                    
                    dynamic_objects.append({
                        'object_id': global_id,
                        'aggregated_points': aggregated_points,
                        'aggregated_gaussians': aggregated_gaussian,  # Stage2Refiner需要的字段
                        'frame_gaussians': frame_gaussians,  # Stage2Refiner需要的字段
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
            object_id = aggregated_object.get('object_id') or aggregated_object.get('global_id')
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
            
            print(f"🔍 查找参考帧{reference_frame}，clustering_results有{len(clustering_results)}个结果")
            
            # 方法1: 直接通过frame_idx匹配
            for result in clustering_results:
                frame_idx = result.get('frame_idx')
                print(f"  检查clustering_results中的frame_idx: {frame_idx}")
                if frame_idx == reference_frame:
                    reference_clustering = result
                    break
            
            # 方法2: 如果没找到，尝试通过索引匹配（reference_frame可能是相对索引）
            if reference_clustering is None and 0 <= reference_frame < len(clustering_results):
                print(f"  通过索引{reference_frame}直接访问clustering_results")
                reference_clustering = clustering_results[reference_frame]
            
            if reference_clustering is None:
                print(f"⚠️  未找到参考帧{reference_frame}的聚类结果")
                print(f"  可用的frame_idx: {[r.get('frame_idx') for r in clustering_results]}")
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
                print(f"⚠️  在参考帧{reference_frame}中未找到物体{global_id}的像素索引")
                return self._points_to_gaussian_params_fallback(aggregated_points, global_id)
            
            print(f"🔍 物体{global_id}: 在参考帧{reference_frame}找到{len(object_pixel_indices)}个像素索引")
            
            # 直接通过像素索引提取对应的Gaussian参数
            B, N_total, feature_dim = gaussian_params.shape
            print(f"🔍 Gaussian参数形状: B={B}, N_total={N_total}, feature_dim={feature_dim}")
            
            # gaussian_params的形状是 [B, S*H*W, 14]，我们需要计算正确的全局索引
            # cluster_indices中的像素索引是相对于单帧的（0到H*W-1），需要转换为全局索引
            
            # 首先，我们需要推断H, W和S
            # 从clustering_results推断出H*W
            H_W = len(reference_clustering.get('points', []))
            if H_W == 0:
                print(f"⚠️  无法推断图像尺寸")
                return self._points_to_gaussian_params_fallback(aggregated_points, global_id)
            
            # 从N_total和H_W推断S
            S = N_total // H_W if H_W > 0 else 1
            print(f"🔍 推断的参数: H*W={H_W}, S={S}, reference_frame={reference_frame}")
            
            selected_gaussians_list = []
            
            for pixel_idx in object_pixel_indices:
                # 计算在全局flatten结构中的索引
                # 全局索引 = reference_frame * H*W + pixel_idx
                global_idx = reference_frame * H_W + pixel_idx
                
                if 0 <= global_idx < N_total:
                    selected_gaussians_list.append(gaussian_params[0, global_idx])  # 使用batch=0
                    print(f"  提取像素{pixel_idx}->全局索引{global_idx}的Gaussian参数")
                else:
                    print(f"  ⚠️  全局索引{global_idx}(来自像素{pixel_idx})超出范围[0, {N_total-1}]")
            
            if len(selected_gaussians_list) == 0:
                print(f"⚠️  无法提取有效的Gaussian参数")
                return self._points_to_gaussian_params_fallback(aggregated_points, global_id)
            
            selected_gaussians = torch.stack(selected_gaussians_list, dim=0)  # [N, 14]
            
            # 激活Gaussian参数
            selected_gaussians = self._apply_gaussian_activation(selected_gaussians)
            
            # 使用聚合后的点云位置替换Gaussian的位置参数
            points_tensor = torch.from_numpy(aggregated_points).to(self.device).float()
            
            # 如果点数不匹配，取较小的数量
            min_count = min(len(selected_gaussians), len(points_tensor))
            selected_gaussians = selected_gaussians[:min_count]
            points_tensor = points_tensor[:min_count]
            
            selected_gaussians[:, :3] = points_tensor[:, :3]
            
            print(f"✅ 正确提取了{len(selected_gaussians)}个Gaussian参数（通过像素索引对应）")
            
            return selected_gaussians
            
        except Exception as e:
            print(f"❌ 像素索引对应方法失败: {e}")
            print(f"回退到传统方法")
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
                points = torch.tensor(points, device=self.device, dtype=torch.float32)
            else:
                points = points.to(self.device).float()
            
            # gaussian_params的shape: [B, S*H*W, 14]
            # 我们需要找到与points最匹配的Gaussian参数
            B, N_total, feature_dim = gaussian_params.shape
            
            # 重塑gaussian_params为[B*S*H*W, 14]以便处理
            gaussian_params_flat = gaussian_params.view(-1, feature_dim)  # [B*S*H*W, 14]
            
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
                selected_indices = np.random.choice(N_gaussians, N_points, replace=True)
                selected_gaussians = gaussian_params_flat[selected_indices]
                print(f"警告：Gaussian数量({N_gaussians}) < 点数({N_points})，使用随机采样")
            else:
                # 使用KD-tree但确保每个点都有独特的参数
                nbrs = NearestNeighbors(n_neighbors=min(5, N_gaussians), algorithm='kd_tree').fit(gaussian_pos_np)
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
            print("对提取的VGGT参数应用激活函数...")
            selected_gaussians = self._apply_gaussian_activation(selected_gaussians)
            
            # 使用聚合后的点云位置替换Gaussian的位置参数（激活后）
            selected_gaussians[:, :3] = points[:, :3]
            
            # 保持VGGT预测参数的原始性，不添加随机扰动
            # 大白球问题主要通过确保选择不同的Gaussian参数来解决
            
            # 保持VGGT预测的颜色参数，不用光流聚合的颜色替换
            # VGGT预测的颜色参数经过神经网络训练，适合3D Gaussian Splatting渲染
            # 光流聚合的颜色适合传统点云，但不适合Gaussian渲染
            
            print(f"从VGGT预测中提取并激活了 {selected_gaussians.shape[0]} 个Gaussian参数")
            
            return selected_gaussians
            
        except Exception as e:
            print(f"从VGGT预测提取Gaussian参数失败: {e}")
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
            gaussian_params = torch.zeros(N, 14, device=self.device, dtype=torch.float32)
            
            # 位置: xyz (positions 0:3)
            gaussian_params[:, :3] = points[:, :3]
            
            # 尺度: scale (positions 3:6) - raw values before activation
            gaussian_params[:, 3:6] = torch.log(torch.tensor(0.01 / 0.05))  # Will become 0.01 after activation
            
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
                print(f"回退方案：为object_id={object_id}生成一致颜色 RGB=({r+m:.3f}, {g+m:.3f}, {b+m:.3f})")
            else:
                # 默认中性颜色
                gaussian_params[:, 6:9] = 0.5
                print("回退方案：使用默认中性颜色")
            
            # 旋转: quaternion (positions 9:13) - normalized quaternion
            gaussian_params[:, 9:13] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=points.device)  # w, x, y, z
            
            # 不透明度: opacity (position 13) - raw value before sigmoid
            gaussian_params[:, 13] = torch.logit(torch.tensor(0.8, device=points.device))  # Will become 0.8 after sigmoid
            
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
        
        # 温和的不透明度增强：避免过度均质化
        mean_opacity = opacities.mean()
        min_opacity_threshold = 0.2  # 最小不透明度阈值
        
        # 只增强过低的不透明度值，保持高的值不变
        mask_low = opacities < min_opacity_threshold
        if mask_low.any():
            # 只提升过低的不透明度到阈值，不影响其他值
            opacities[mask_low] = min_opacity_threshold
            print(f"不透明度修正：提升了 {mask_low.sum().item()} 个过低的不透明度值到 {min_opacity_threshold}")
        
        # 如果整体平均值仍然过低，进行温和的整体提升
        if mean_opacity < 0.4:
            # 使用幂函数进行温和提升，保持相对差异
            enhanced_opacities = torch.pow(opacities, 0.7)  # 幂 < 1 会提升低值，保持高值
            print(f"整体不透明度温和提升：{mean_opacity.item():.3f} -> {enhanced_opacities.mean().item():.3f}")
            opacities = enhanced_opacities
        
        processed_params[..., 13:14] = opacities
        
        return processed_params
            
    
    def _validate_and_fix_transform(self, transform: torch.Tensor, frame_idx: int, global_id: int = None) -> bool:
        """验证和修复变换矩阵，防止大白球问题"""
        try:
            # 检查基本形状
            if transform.shape != (4, 4):
                print(f"⚠️  变换矩阵形状异常: {transform.shape}, 期望 (4,4)")
                return False
            
            # 检查是否为零矩阵
            if torch.allclose(transform, torch.zeros_like(transform), atol=1e-8):
                print(f"⚠️  对象{global_id}帧{frame_idx}: 检测到零变换矩阵！这会导致大白球问题")
                return False
            
            # 检查是否有NaN或Inf
            if torch.isnan(transform).any() or torch.isinf(transform).any():
                print(f"⚠️  对象{global_id}帧{frame_idx}: 变换矩阵包含NaN或Inf值")
                return False
            
            # 检查旋转部分的行列式
            rotation_part = transform[:3, :3]
            det = torch.det(rotation_part)
            
            if det.abs() < 1e-6:
                print(f"⚠️  对象{global_id}帧{frame_idx}: 变换矩阵奇异 (det={det:.2e})")
                return False
            
            # 检查是否过度缩放
            scales = torch.linalg.norm(rotation_part, dim=0)  # 各轴的缩放
            if scales.max() > 100 or scales.min() < 0.01:
                print(f"⚠️  对象{global_id}帧{frame_idx}: 异常缩放 {scales}, 可能导致渲染问题")
                return False
            
            # 检查平移是否过大
            translation = transform[:3, 3]
            if torch.norm(translation) > 1000:
                print(f"⚠️  对象{global_id}帧{frame_idx}: 平移过大 {translation}, 可能超出相机视野")
                # 这种情况仍然保留，但给出警告
            
            return True
            
        except Exception as e:
            print(f"❌ 变换矩阵验证失败: {e}")
            return False
    
    def _create_static_background(
        self, 
        preds: Dict[str, Any], 
        velocity: Optional[torch.Tensor],
        clustering_results: List[Dict],
        H: int, W: int, S: int
    ) -> torch.Tensor:
        """从VGGT预测的Gaussians中分离静态背景"""
        try:
            import time
            stage5_times = {}
            
            # Step 1: 获取和整形Gaussian参数
            step1_start = time.time()
            gaussian_params = preds.get('gaussian_params')  # [B, S*H*W, 14] or [B*S, H*W, 14]
            
            if gaussian_params is None:
                return self._create_default_static_background(H, W)
            
            # 重新整形Gaussian参数为 [S, H*W, 14]（激活函数已在函数开头应用）
            if gaussian_params.dim() == 3 and gaussian_params.shape[1] == S * H * W:
                # 情况1: [B, S*H*W, 14] -> [S, H*W, 14]
                gaussian_params = gaussian_params[0].reshape(S, H * W, 14)
            elif gaussian_params.dim() == 3 and gaussian_params.shape[0] == S:
                # 情况2: [S, H*W, 14] -> 已经是正确形状
                gaussian_params = gaussian_params
            else:
                # 其他情况，尝试重新整形
                gaussian_params = gaussian_params.reshape(S, H * W, 14)
            stage5_times['Step 1: 获取和整形Gaussian参数'] = time.time() - step1_start
            
            # Step 2: 处理速度信息
            step2_start = time.time()
            if velocity is not None:
                if len(velocity.shape) == 5:
                    B_v, S_v, H_v, W_v, _ = velocity.shape
                    velocity_reshaped = velocity[0].reshape(S, H * W, 3)  # [S, H*W, 3]
                elif len(velocity.shape) == 4:
                    B_v, S_v, HW_v, _ = velocity.shape
                    velocity_reshaped = velocity[0]  # [S, H*W, 3]
                else:
                    velocity_reshaped = velocity[0].reshape(S, H * W, 3) if velocity.numel() >= S * H * W * 3 else torch.zeros(S, H * W, 3, device=self.device)
                
                # 计算速度大小
                velocity_magnitude = torch.norm(velocity_reshaped, dim=-1)  # [S, H*W]
            else:
                velocity_magnitude = torch.zeros(S, H * W, device=self.device)
            stage5_times['Step 2: 处理速度信息'] = time.time() - step2_start
            
            # Step 3: 收集动态区域掩码
            step3_start = time.time()
            dynamic_mask_all = torch.zeros(S, H * W, dtype=torch.bool, device=self.device)
            
            for frame_idx, result in enumerate(clustering_results):
                if frame_idx >= S:
                    break
                dynamic_mask = result.get('dynamic_mask')
                if dynamic_mask is not None and len(dynamic_mask) == H * W:
                    dynamic_mask_all[frame_idx] = dynamic_mask
            stage5_times['Step 3: 收集动态区域掩码'] = time.time() - step3_start
            
            # Step 4: 计算静态区域掩码
            step4_start = time.time()
            velocity_threshold = 0.01  # 速度阈值
            static_velocity_mask = velocity_magnitude <= velocity_threshold  # 低速度区域
            static_object_mask = ~dynamic_mask_all  # 非动态物体区域
            
            # 静态区域 = 低速度 AND 非动态物体
            static_mask = static_velocity_mask & static_object_mask  # [S, H*W]
            stage5_times['Step 4: 计算静态区域掩码'] = time.time() - step4_start
            
            # Step 5: 收集所有静态Gaussians
            step5_start = time.time()
            all_static_gaussians = []
            for frame_idx in range(S):
                frame_static_mask = static_mask[frame_idx]
                if frame_static_mask.any():
                    frame_static_gaussians = gaussian_params[frame_idx][frame_static_mask]  # [N_static, 14]
                    all_static_gaussians.append(frame_static_gaussians)
            
            if not all_static_gaussians:
                return self._create_default_static_background(H, W)
            
            # 合并所有静态Gaussians
            all_static_gaussians = torch.cat(all_static_gaussians, dim=0)  # [Total_N, 14]
            stage5_times['Step 5: 收集所有静态Gaussians'] = time.time() - step5_start
            
            # Step 6: 下采样和去重处理
            step6_start = time.time()
            downsampled_static_gaussians = self._downsample_static_gaussians(
                all_static_gaussians, max_points=200000, spatial_threshold=0.01
            )
            stage5_times['Step 6: 下采样和去重处理'] = time.time() - step6_start
            
            # 显示Stage 5各步骤的详细耗时
            print("    Stage 5详细耗时:")
            for step_name, step_time in stage5_times.items():
                print(f"      {step_name}: {step_time:.4f}s")
            
            return downsampled_static_gaussians
            
        except Exception as e:
            return self._create_default_static_background(H, W)
    
    def _create_default_static_background(self, H: int, W: int) -> torch.Tensor:
        """创建默认静态背景（回退方案）"""
        try:
            num_background_points = min(1000, H * W // 100)
            background_gaussians = torch.zeros(num_background_points, 14, device=self.device)
            
            # 随机分布在3D空间中
            background_gaussians[:, :3] = torch.randn(num_background_points, 3, device=self.device) * 2.0
            
            # 旋转（单位四元数）
            background_gaussians[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            
            # 尺度
            background_gaussians[:, 7:10] = 0.1
            
            # 不透明度
            background_gaussians[:, 10] = 0.1
            
            # 颜色（灰色）
            background_gaussians[:, 11:14] = 0.3
            
            return background_gaussians
            
        except Exception as e:
            return torch.zeros(100, 14, device=self.device)
    
    def _downsample_static_gaussians(
        self, 
        static_gaussians: torch.Tensor, 
        max_points: int = 2000,
        spatial_threshold: float = 0.05
    ) -> torch.Tensor:
        """对静态Gaussians进行下采样和去重"""
        try:
            import time
            downsample_times = {}
            print(f"        开始下采样: 输入点数={len(static_gaussians)}, max_points={max_points}, spatial_threshold={spatial_threshold}")
            if len(static_gaussians) <= max_points:
                print(f"        跳过下采样: 点数已符合要求")
                return static_gaussians
            
            # Step 6.1: 基于空间距离的去重
            step61_start = time.time()
            positions = static_gaussians[:, :3]  # [N, 3]
            
            # 使用简单的网格下采样来去重
            # 将3D空间分为网格，每个网格只保留一个代表点
            grid_size = spatial_threshold
            
            # 量化位置到网格
            quantized_positions = torch.round(positions / grid_size) * grid_size
            downsample_times['Step 6.1: 位置量化'] = time.time() - step61_start
            
            # Step 6.2: 找到唯一的网格位置
            unique_positions, inverse_indices = torch.unique(
                quantized_positions, dim=0, return_inverse=True
            )
            downsample_times['Step 6.2: 唯一网格计算'] = time.time() - step62_start
            print(f"        去重前: {len(static_gaussians)} -> 去重后: {len(unique_positions)}")
            
            # Step 6.3: 选择代表Gaussian (向量化优化)
            step63_start = time.time()
            # 使用sort找到每个网格的第一个出现位置
            sorted_indices, sort_order = torch.sort(inverse_indices)
            # 找到相邻元素不同的位置（即每个唯一值第一次出现的位置）
            first_occurrence_mask = torch.cat([
                torch.tensor([True], device=self.device), 
                sorted_indices[1:] != sorted_indices[:-1]
            ])
            first_occurrence = sort_order[first_occurrence_mask]
            deduped_gaussians = static_gaussians[first_occurrence]
            downsample_times['Step 6.3: 选择代表Gaussian'] = time.time() - step63_start
            
            # Step 6.4: 随机下采样（如果仍然太多）
            step64_start = time.time()
            if len(deduped_gaussians) > max_points:
                indices = torch.randperm(len(deduped_gaussians), device=self.device)[:max_points]
                final_gaussians = deduped_gaussians[indices]
                print(f"        随机下采样: {len(deduped_gaussians)} -> {len(final_gaussians)}")
            else:
                final_gaussians = deduped_gaussians
                print(f"        最终点数: {len(final_gaussians)}")
            downsample_times['Step 6.4: 随机下采样'] = time.time() - step64_start
            
            # 显示下采样详细耗时
            print("        下采样详细耗时:")
            for step_name, step_time in downsample_times.items():
                print(f"          {step_name}: {step_time:.4f}s")
            
            return final_gaussians
            
        except Exception as e:
            # 回退到简单随机采样
            if len(static_gaussians) > max_points:
                indices = torch.randperm(len(static_gaussians), device=self.device)[:max_points]
                return static_gaussians[indices]
            else:
                return static_gaussians
    
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