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
from collections import defaultdict

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
                "max_num_points_per_voxel": 5,
                "use_dilated_conv": False,  # 默认不使用dilated conv（向后兼容）
                "dilation_rates": None
            }

        if pose_refine_config is None:
            pose_refine_config = {
                "input_dim": 3,
                "feature_dim": 128,
                "num_conv_layers": 2,
                "voxel_size": 0.1,
                "max_points": 4096,
                "use_dilated_conv": False,  # 默认不使用dilated conv（向后兼容）
                "dilation_rates": None
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
        frame_info: Optional[Dict] = None,
        preds: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        前向传播

        Args:
            dynamic_objects: 动态物体列表，每个包含:
                - 'canonical_gaussians': [N, 14] canonical Gaussian参数
                - 'frame_transforms': Dict[int, Tensor] 每帧变换矩阵
                - 'object_id': int 物体ID
            static_gaussians: 静态场景的Gaussian参数 (可选)
            frame_info: 帧信息，包含相机参数等
            preds: Stage1预测结果，包含'scale'等信息 (可选)

        Returns:
            results: 包含细化结果的字典
        """
        results = {
            'refined_dynamic_objects': [],
            'pose_refinements': {},
            'gaussian_refinements': {},
            'timing_stats': {
                'gaussian_refine_time': 0.0,
                'pose_refine_time': 0.0,
                'num_gaussian_refines': 0,
                'num_pose_refines': 0
            }
        }

        # 从preds中获取pred_scale
        pred_scale = preds.get('scale') if preds is not None else None

        for obj_data in dynamic_objects:
            object_id = obj_data['object_id']
            canonical_gaussians = obj_data['canonical_gaussians']
            frame_transforms = obj_data.get('frame_transforms', {})

            # 转换为列表格式
            initial_transforms = [frame_transforms[f] for f in sorted(frame_transforms.keys())] if frame_transforms else [torch.eye(4, device=canonical_gaussians.device)]

            # 初始化结果
            # canonical_gaussians将在refinement后被更新为refined版本
            refined_obj = {
                'object_id': object_id,
                'original_canonical_gaussians': canonical_gaussians,  # 保留原始版本
                'canonical_gaussians': canonical_gaussians,  # 将被更新为refined版本
                'refined_gaussians': canonical_gaussians,
                'refined_transforms': initial_transforms,
                'gaussian_deltas': None,
                'pose_deltas': [],
                'reference_frame': obj_data.get('reference_frame', 0),
                'original_frame_transforms': frame_transforms,  # 保留原始版本
                'frame_transforms': frame_transforms,  # 将被更新为refined版本
                'frame_existence': obj_data.get('frame_existence', torch.tensor([True], dtype=torch.bool))
            }

            # Gaussian参数细化
            if self.training_mode in ["joint", "gaussian_only"]:
                start_time = time.time()
                gaussian_deltas = self.gaussian_refine_head(canonical_gaussians, pred_scale=pred_scale)
                refined_gaussians = self.gaussian_refine_head.apply_deltas(canonical_gaussians, gaussian_deltas)
                gaussian_time = time.time() - start_time

                refined_obj['refined_gaussians'] = refined_gaussians
                refined_obj['canonical_gaussians'] = refined_gaussians  # 更新canonical_gaussians为refined版本
                refined_obj['gaussian_deltas'] = gaussian_deltas
                results['gaussian_refinements'][object_id] = {
                    'deltas': gaussian_deltas,
                    'refined_params': refined_gaussians
                }

                # 累计时间统计
                results['timing_stats']['gaussian_refine_time'] += gaussian_time
                results['timing_stats']['num_gaussian_refines'] += 1

            # 位姿细化（仅对多帧物体进行）
            if self.training_mode in ["joint", "pose_only"] and len(initial_transforms) > 1:
                pose_start_time = time.time()
                refined_positions = refined_obj['refined_gaussians'][:, :3].detach()  # 使用refined后的Gaussian位置,不计算梯度
                refined_transforms = {}  # 改为字典以保持frame_idx映射
                pose_deltas = {}  # 改为字典以保持frame_idx映射

                # 获取每帧的原始Gaussian参数（字典格式：{frame_idx: gaussians}）
                frame_gaussians = obj_data.get('frame_gaussians', {})

                # 获取有序的frame_idx列表
                sorted_frame_indices = sorted(frame_transforms.keys()) if frame_transforms else list(range(len(initial_transforms)))

                for loop_idx, initial_T in enumerate(initial_transforms):
                    # 获取实际的frame_idx
                    actual_frame_idx = sorted_frame_indices[loop_idx] if loop_idx < len(sorted_frame_indices) else loop_idx

                    # source_points: refined canonical Gaussian经过initial_T变换后的位置
                    ones = torch.ones(refined_positions.shape[0], 1, device=refined_positions.device)
                    source_points = (initial_T @ torch.cat([refined_positions, ones], dim=-1).T).T[:, :3]

                    # target_points: 当前帧的原始Gaussian位置
                    if actual_frame_idx in frame_gaussians:
                        target_gaussians = frame_gaussians[actual_frame_idx]
                        if target_gaussians is not None:
                            target_points = target_gaussians[:, :3]  # 取前3维作为位置
                        else:
                            # 如果没有当前帧的Gaussian，使用source_points作为fallback
                            target_points = source_points
                            print(f"Warning: Object {object_id} frame {actual_frame_idx} has None gaussians, using source_points as fallback")
                    else:
                        # 如果没有当前帧的Gaussian，使用source_points作为fallback
                        target_points = source_points
                        print(f"Warning: Object {object_id} frame {actual_frame_idx} missing in frame_gaussians, using source_points as fallback")

                    # 预测并应用位姿细化（传入pred_scale）
                    pose_delta = self.pose_refine_head(
                        source_points=source_points,
                        target_points=target_points,
                        pred_scale=pred_scale
                    )
                    refined_transform = self.pose_refine_head.apply_pose_delta(initial_T, pose_delta)

                    # 使用actual_frame_idx作为key存储
                    refined_transforms[actual_frame_idx] = refined_transform
                    pose_deltas[actual_frame_idx] = pose_delta

                pose_time = time.time() - pose_start_time

                refined_obj['refined_transforms'] = refined_transforms
                refined_obj['frame_transforms'] = refined_transforms  # 更新frame_transforms为refined版本
                refined_obj['pose_deltas'] = pose_deltas
                results['pose_refinements'][object_id] = {
                    'deltas': pose_deltas,
                    'refined_transforms': refined_transforms
                }

                # 累计时间统计
                results['timing_stats']['pose_refine_time'] += pose_time
                results['timing_stats']['num_pose_refines'] += 1

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

        IMPORTANT: 此方法只接受经过forward()处理的refinement_results，
        必须包含'refined_dynamic_objects'字段，强制使用refined后的参数以确保梯度传播。

        Args:
            refinement_results: forward方法的输出结果（必须包含'refined_dynamic_objects'）
            static_gaussians: 静态场景Gaussian参数

        Returns:
            scene: 包含完整场景的字典，格式为Stage2Loss期望的结构

        Raises:
            ValueError: 如果refinement_results格式不正确或缺少refined参数
        """
        # 只接受经过refinement的结果
        if not isinstance(refinement_results, dict) or 'refined_dynamic_objects' not in refinement_results:
            raise ValueError(
                "get_refined_scene() requires refinement_results from forward() with 'refined_dynamic_objects'. "
                "If you want to render unrefined scenes, use get_initial_scene() instead."
            )

        input_objects = refinement_results['refined_dynamic_objects']
        dynamic_objects = []

        for obj_data in input_objects:
            object_id = obj_data['object_id']

            # 强制使用refined参数
            if 'refined_gaussians' not in obj_data:
                raise ValueError(
                    f"Object {object_id} missing 'refined_gaussians'. "
                    "All objects must be refined before calling get_refined_scene()."
                )

            # 构建动态物体数据，强制使用refined参数
            reference_frame = obj_data.get('reference_frame', 0)
            if reference_frame is None:
                reference_frame = 0

            # 使用refined后的Gaussian参数（确保梯度传播）
            canonical_gaussians = obj_data['refined_gaussians']

            # 使用refined后的transforms（如果有pose refinement）
            frame_transforms = obj_data.get('frame_transforms', {})
            if 'refined_transforms' in obj_data and obj_data['refined_transforms']:
                # 优先使用refined transforms（字典格式，直接使用）
                refined_transforms = obj_data['refined_transforms']
                if isinstance(refined_transforms, dict):
                    # 字典格式：直接使用
                    frame_transforms = refined_transforms
                    print(f"[get_refined_scene] Object {object_id}: Using refined_transforms (dict), keys = {list(frame_transforms.keys())}")
                else:
                    # 列表格式（兼容旧代码）：需要映射到frame_idx
                    frame_transforms = {}
                    original_frame_transforms = obj_data.get('frame_transforms', {})
                    sorted_frame_indices = sorted(original_frame_transforms.keys()) if original_frame_transforms else list(range(len(refined_transforms)))
                    for loop_idx, transform in enumerate(refined_transforms):
                        if transform is not None and loop_idx < len(sorted_frame_indices):
                            actual_frame_idx = sorted_frame_indices[loop_idx]
                            frame_transforms[actual_frame_idx] = transform
                    print(f"[get_refined_scene] Object {object_id}: Using refined_transforms (list), keys = {list(frame_transforms.keys())}")

            # 构建frame_existence
            frame_existence = obj_data.get('frame_existence')
            if frame_existence is None or len(frame_existence) <= 1:
                if frame_transforms:
                    max_frame = max(frame_transforms.keys()) if frame_transforms else reference_frame
                    frame_existence = torch.tensor(
                        [frame_idx in frame_transforms for frame_idx in range(max_frame + 1)],
                        dtype=torch.bool
                    )
                else:
                    frame_existence = torch.tensor([True], dtype=torch.bool)

            dynamic_obj = {
                'object_id': object_id,
                'canonical_gaussians': canonical_gaussians,  # 强制使用refined参数
                'reference_frame': reference_frame,
                'frame_transforms': frame_transforms,  # 使用refined transforms（如果有）
                'frame_existence': frame_existence
            }

            dynamic_objects.append(dynamic_obj)

        scene = {
            'static_gaussians': static_gaussians,
            'dynamic_objects': dynamic_objects
        }

        return scene

    def get_initial_scene(
        self,
        dynamic_objects: List[Dict],
        static_gaussians: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        获取未经refinement的初始场景表示（用于对比）

        此方法用于渲染未经Stage2细化的场景，不涉及梯度传播。

        Args:
            dynamic_objects: 原始动态物体列表（来自OnlineDynamicProcessor）
            static_gaussians: 静态场景Gaussian参数

        Returns:
            scene: 包含完整场景的字典，格式为Stage2Loss期望的结构
        """
        scene_dynamic_objects = []

        for obj_data in dynamic_objects:
            object_id = obj_data['object_id']

            # 构建动态物体数据
            reference_frame = obj_data.get('reference_frame', 0)
            if reference_frame is None:
                reference_frame = 0

            # 使用原始canonical_gaussians
            canonical_gaussians = obj_data.get('canonical_gaussians', obj_data.get('aggregated_gaussians'))
            if canonical_gaussians is None:
                print(f"Warning: Object {object_id} missing canonical_gaussians, skipping")
                continue

            # 使用原始transforms
            frame_transforms = obj_data.get('frame_transforms', {})
            if not frame_transforms and 'transformations' in obj_data:
                # 从transformations构建
                transformations_dict = obj_data['transformations']
                for frame_idx, transform_info in transformations_dict.items():
                    if isinstance(transform_info, dict) and 'transformation' in transform_info:
                        transform = transform_info['transformation']
                        if isinstance(transform, np.ndarray):
                            transform = torch.from_numpy(transform).float()
                        frame_transforms[frame_idx] = transform

            # 构建frame_existence
            frame_existence = obj_data.get('frame_existence')
            if frame_existence is None or len(frame_existence) <= 1:
                if frame_transforms:
                    max_frame = max(frame_transforms.keys()) if frame_transforms else reference_frame
                    frame_existence = torch.tensor(
                        [frame_idx in frame_transforms for frame_idx in range(max_frame + 1)],
                        dtype=torch.bool
                    )
                else:
                    frame_existence = torch.tensor([True], dtype=torch.bool)

            dynamic_obj = {
                'object_id': object_id,
                'canonical_gaussians': canonical_gaussians,
                'reference_frame': reference_frame,
                'frame_transforms': frame_transforms,
                'frame_existence': frame_existence
            }

            scene_dynamic_objects.append(dynamic_obj)

        scene = {
            'static_gaussians': static_gaussians,
            'dynamic_objects': scene_dynamic_objects
        }

        return scene
    
    
    def gradient_checkpointing_enable(self, enable: bool = True):
        """启用或禁用梯度检查点（稀疏卷积版本不支持）"""
        # 稀疏卷积实现不支持梯度检查点，保留此方法仅用于兼容性
        pass