# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import time
import logging

# 导入头网络
from vggt.heads.sparse_conv_refine_head import GaussianRefineHeadSparseConv, PoseRefineHeadSparseConv
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


class Stage2Refiner(nn.Module):
    """
    第二阶段细化网络
    
    用于细化动态物体的Gaussian参数和位姿，以提升几何一致性和新视角生成质量
    """
    
    def __init__(
        self,
        gaussian_refine_config: Optional[Dict] = None,
        pose_refine_config: Optional[Dict] = None,
        training_mode: str = "joint"  # "joint", "gaussian_only", "pose_only"
    ):
        super().__init__()
        
        # 默认配置 - 使用稀疏卷积优化速度和显存
        if gaussian_refine_config is None:
            gaussian_refine_config = {
                "input_gaussian_dim": 14,
                "output_gaussian_dim": 14,
                "feature_dim": 128,
                "num_conv_layers": 2,
                "voxel_size": 0.05,
                "max_num_points_per_voxel": 5
            }

        if pose_refine_config is None:
            pose_refine_config = {
                "input_dim": 3,
                "feature_dim": 128,
                "num_conv_layers": 2,
                "voxel_size": 0.1,
                "max_points": 4096
            }

        # 初始化网络组件 - 使用稀疏卷积版本
        self.gaussian_refine_head = GaussianRefineHeadSparseConv(**gaussian_refine_config)
        self.pose_refine_head = PoseRefineHeadSparseConv(**pose_refine_config)
        
        self.training_mode = training_mode
        
        # 设置梯度
        self._set_training_mode(training_mode)
        
    def _set_training_mode(self, mode: str):
        """设置训练模式，控制哪些组件参与训练"""
        self.training_mode = mode
        
        if mode == "gaussian_only":
            # 只训练Gaussian细化网络
            self.gaussian_refine_head.requires_grad_(True)
            self.pose_refine_head.requires_grad_(False)
        elif mode == "pose_only":
            # 只训练位姿细化网络
            self.gaussian_refine_head.requires_grad_(False)
            self.pose_refine_head.requires_grad_(True)
        else:  # joint
            # 联合训练
            self.gaussian_refine_head.requires_grad_(True)
            self.pose_refine_head.requires_grad_(True)
    
    def forward(
        self,
        dynamic_objects: List[Dict],
        static_gaussians: Optional[Dict] = None,
        frame_info: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            dynamic_objects: 动态物体列表，每个包含:
                - 'aggregated_gaussians': [N, 14] 聚合的Gaussian参数
                - 'frame_gaussians': List[Tensor] 每帧的原始Gaussian
                - 'initial_transforms': List[Tensor] 初始变换矩阵
                - 'object_id': int 物体ID
            static_gaussians: 静态场景的Gaussian参数 (可选)
            frame_info: 帧信息，包含相机参数等
            
        Returns:
            results: 包含细化结果的字典
        """
        results = {
            'refined_dynamic_objects': [],
            'pose_refinements': {},
            'gaussian_refinements': {}
        }
        
        for obj_idx, obj_data in enumerate(dynamic_objects):
            object_start_time = time.time()
            object_id = obj_data['object_id']
            
            # 支持两种数据格式：新格式和旧格式
            if 'canonical_gaussians' in obj_data:
                # 新格式：来自修正后的OnlineDynamicProcessor
                canonical_gaussians = obj_data['canonical_gaussians']  # [N, 14]
                frame_transforms_dict = obj_data.get('frame_transforms', {})
                
                # 转换frame_transforms字典为列表格式以兼容现有逻辑
                frame_gaussians = [canonical_gaussians]  # 默认使用canonical作为唯一帧
                initial_transforms = []
                
                # 从frame_transforms中提取变换矩阵
                if frame_transforms_dict:
                    sorted_frames = sorted(frame_transforms_dict.keys())
                    for frame_idx in sorted_frames:
                        initial_transforms.append(frame_transforms_dict[frame_idx])
                        # 每帧都使用相同的canonical gaussians，实际变换在渲染时处理
                        frame_gaussians.append(canonical_gaussians)
                
                # 如果没有变换信息，至少添加一个恒等变换
                if not initial_transforms:
                    initial_transforms = [torch.eye(4, device=canonical_gaussians.device)]
                    
                aggregated_gaussians = canonical_gaussians
                
            else:
                # 旧格式：传统的aggregated_gaussians格式
                aggregated_gaussians = obj_data['aggregated_gaussians']  # [N, 14]
                frame_gaussians = obj_data.get('frame_gaussians', [aggregated_gaussians])  # List[Tensor]
                initial_transforms = obj_data.get('initial_transforms', [torch.eye(4, device=aggregated_gaussians.device)])  # List[Tensor]
            
            refined_obj = {
                'object_id': object_id,
                'original_gaussians': aggregated_gaussians,
                'refined_gaussians': aggregated_gaussians,  # 默认值
                'refined_transforms': initial_transforms,   # 默认值
                'gaussian_deltas': None,
                'pose_deltas': [],
                # 保留新格式信息以便get_refined_scene使用
                'canonical_gaussians': aggregated_gaussians,
                'reference_frame': obj_data.get('reference_frame', 0),
                'frame_transforms': obj_data.get('frame_transforms', {}),
                'frame_existence': obj_data.get('frame_existence', torch.tensor([True], dtype=torch.bool))
            }
            
            # 1. Gaussian参数细化
            if self.training_mode in ["joint", "gaussian_only"]:
                
                # Safety check: 防止内存爆炸 - 限制最大Gaussian数量
                max_gaussians_stage2 = 1000000  # TODO:
                if aggregated_gaussians.shape[0] > max_gaussians_stage2:
                    # print(f"Warning: Object {object_id} has too many Gaussians ({aggregated_gaussians.shape[0]}), skipping Stage2 processing")
                    # 跳过这个物体的Stage2处理
                    refined_obj['refined_gaussians'] = aggregated_gaussians  # 保持原始参数
                    refined_obj['gaussian_deltas'] = None
                else:
                    gaussian_deltas = self.gaussian_refine_head(aggregated_gaussians)
                    refined_gaussians = self.gaussian_refine_head.apply_deltas(
                        aggregated_gaussians, gaussian_deltas
                    )
                    
                    refined_obj['refined_gaussians'] = refined_gaussians
                    refined_obj['gaussian_deltas'] = gaussian_deltas
                    results['gaussian_refinements'][object_id] = {
                        'deltas': gaussian_deltas,
                        'refined_params': refined_gaussians
                    }
            
            # 2. 位姿细化
            if self.training_mode in ["joint", "pose_only"]:
                refined_transforms = []
                pose_deltas = []
                
                # 获取细化后的Gaussian位置作为点云
                refined_positions = refined_obj['refined_gaussians'][:, :3]  # [N, 3]
                
                for frame_idx, (frame_gaussian, initial_T) in enumerate(
                    zip(frame_gaussians, initial_transforms)
                ):
                    # 获取当前帧的点云位置
                    frame_positions = frame_gaussian[:, :3]  # [M, 3]
                    
                    # 通过初始变换将参考点云变换到当前帧
                    ref_positions_homo = torch.cat([
                        refined_positions,
                        torch.ones(refined_positions.shape[0], 1, device=refined_positions.device)
                    ], dim=-1)  # [N, 4]
                    
                    transformed_ref = torch.matmul(initial_T, ref_positions_homo.T).T[:, :3]  # [N, 3]
                    
                    # 预测位姿细化
                    pose_delta = self.pose_refine_head(
                        source_points=transformed_ref,
                        target_points=frame_positions,
                        initial_transform=initial_T
                    )
                    
                    # 应用位姿细化
                    refined_transform = self.pose_refine_head.apply_pose_delta(
                        initial_T, pose_delta
                    )
                    
                    refined_transforms.append(refined_transform)
                    pose_deltas.append(pose_delta)
                
                refined_obj['refined_transforms'] = refined_transforms
                refined_obj['pose_deltas'] = pose_deltas
                results['pose_refinements'][object_id] = {
                    'deltas': pose_deltas,
                    'refined_transforms': refined_transforms
                }
            
            results['refined_dynamic_objects'].append(refined_obj)

        return results
    
    def set_training_mode(self, mode: str):
        """动态设置训练模式"""
        assert mode in ["joint", "gaussian_only", "pose_only"], f"Invalid training mode: {mode}"
        self._set_training_mode(mode)
    
    def get_refined_scene(
        self,
        refinement_results: Dict,
        static_gaussians: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        获取完整的细化后场景表示，格式兼容Stage2Loss的期望
        
        Args:
            refinement_results: forward方法的输出结果或直接的dynamic_objects列表
            static_gaussians: 静态场景Gaussian参数
            
        Returns:
            scene: 包含完整场景的字典，格式为Stage2Loss期望的结构
        """
        # 处理两种可能的输入格式
        if isinstance(refinement_results, dict) and 'refined_dynamic_objects' in refinement_results:
            # 来自Stage2Refiner.forward()的结果
            input_objects = refinement_results['refined_dynamic_objects']
            use_refined_params = True
        elif isinstance(refinement_results, list):
            # 直接传入的dynamic_objects列表（来自OnlineDynamicProcessor）
            input_objects = refinement_results
            use_refined_params = False
        else:
            # 回退处理
            input_objects = refinement_results.get('refined_dynamic_objects', refinement_results.get('dynamic_objects', []))
            use_refined_params = 'refined_dynamic_objects' in refinement_results
        
        dynamic_objects = []
        
        for obj_data in input_objects:
            # 如果输入已经是Stage2Loss期望的格式，直接使用
            if all(key in obj_data for key in ['canonical_gaussians', 'reference_frame', 'frame_transforms', 'frame_existence']):
                # 直接复制，但如果有细化后的参数，使用细化后的
                dynamic_obj = obj_data.copy()
                if use_refined_params and 'refined_gaussians' in obj_data:
                    dynamic_obj['canonical_gaussians'] = obj_data['refined_gaussians']
                dynamic_objects.append(dynamic_obj)
                continue
            
            # 需要转换格式的情况（传统的Stage2Refiner输出）
            object_id = obj_data['object_id']
            
            # 构建动态物体数据，使用Stage2Loss期望的结构
            reference_frame = obj_data.get('reference_frame', 0)
            if reference_frame is None:
                reference_frame = 0  # 默认为第0帧
            
            # 选择合适的gaussian参数
            if use_refined_params and 'refined_gaussians' in obj_data:
                canonical_gaussians = obj_data['refined_gaussians']  # 细化后的
            else:
                canonical_gaussians = obj_data.get('canonical_gaussians', obj_data.get('aggregated_gaussians'))
            
            dynamic_obj = {
                'object_id': object_id,
                'canonical_gaussians': canonical_gaussians,
                'reference_frame': reference_frame,
                'frame_transforms': obj_data.get('frame_transforms', {}),
                'frame_existence': obj_data.get('frame_existence', torch.tensor([True], dtype=torch.bool))
            }
            
            # 如果没有frame_transforms，尝试从其他字段构建
            if not dynamic_obj['frame_transforms']:
                # 处理refined_transforms或initial_transforms
                transforms = obj_data.get('refined_transforms', obj_data.get('initial_transforms', []))
                if transforms:
                    for frame_idx, transform in enumerate(transforms):
                        if transform is not None:
                            dynamic_obj['frame_transforms'][frame_idx] = transform
                
                # 处理transformations字典格式
                transformations_dict = obj_data.get('transformations', {})
                for frame_idx, transform_info in transformations_dict.items():
                    if isinstance(transform_info, dict) and 'transformation' in transform_info:
                        transform = transform_info['transformation']
                        if isinstance(transform, np.ndarray):
                            transform = torch.from_numpy(transform).float()
                        dynamic_obj['frame_transforms'][frame_idx] = transform
            
            # 如果没有frame_existence，从frame_transforms推导
            if dynamic_obj['frame_existence'] is None or len(dynamic_obj['frame_existence']) <= 1:
                if dynamic_obj['frame_transforms']:
                    max_frame = max(dynamic_obj['frame_transforms'].keys()) if dynamic_obj['frame_transforms'] else reference_frame
                    frame_existence = []
                    for frame_idx in range(max_frame + 1):
                        frame_existence.append(frame_idx in dynamic_obj['frame_transforms'])
                    dynamic_obj['frame_existence'] = torch.tensor(frame_existence, dtype=torch.bool)
            
            dynamic_objects.append(dynamic_obj)
        
        scene = {
            'static_gaussians': static_gaussians,
            'dynamic_objects': dynamic_objects  # 使用Stage2Loss期望的字段名
        }
        
        return scene
    
    def compute_consistency_loss(self, refinement_results: Dict) -> torch.Tensor:
        """
        计算几何一致性损失
        
        Args:
            refinement_results: forward方法的输出结果
            
        Returns:
            consistency_loss: 一致性损失
        """
        total_loss = 0.0
        num_objects = len(refinement_results['refined_dynamic_objects'])
        
        for obj_data in refinement_results['refined_dynamic_objects']:
            gaussian_deltas = obj_data['gaussian_deltas']
            pose_deltas = obj_data['pose_deltas']
            
            if gaussian_deltas is not None:
                # Gaussian变化量的正则化
                gaussian_reg_loss = torch.mean(torch.abs(gaussian_deltas))
                total_loss += 0.1 * gaussian_reg_loss
            
            if pose_deltas:
                # 位姿变化量的正则化
                pose_deltas_tensor = torch.stack(pose_deltas)  # [num_frames, 6]
                pose_reg_loss = torch.mean(torch.abs(pose_deltas_tensor))
                total_loss += 0.1 * pose_reg_loss
                
                # 位姿变化的时间平滑性
                if len(pose_deltas) > 1:
                    pose_diff = pose_deltas_tensor[1:] - pose_deltas_tensor[:-1]
                    temporal_smoothness_loss = torch.mean(torch.abs(pose_diff))
                    total_loss += 0.05 * temporal_smoothness_loss
        
        return total_loss / max(num_objects, 1)
    
    def gradient_checkpointing_enable(self, enable: bool = True):
        """启用或禁用梯度检查点（稀疏卷积版本不支持）"""
        # 稀疏卷积实现不支持梯度检查点，保留此方法仅用于兼容性
        pass