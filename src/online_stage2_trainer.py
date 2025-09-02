# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
在线第二阶段训练器

直接从第一阶段训练流程中获取数据，实时进行动态物体细化训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import time
from collections import defaultdict

# 导入第二阶段组件
from vggt.models.stage2_refiner import Stage2Refiner
from vggt.training.stage2_loss import Stage2CompleteLoss

# 导入在线处理器
from online_dynamic_processor import OnlineDynamicProcessor


class OnlineStage2Trainer:
    """
    在线第二阶段训练器
    
    在第一阶段训练过程中实时进行第二阶段的细化训练
    """
    
    def __init__(
        self,
        stage2_config: Dict[str, Any],
        device: torch.device,
        enable_stage2: bool = True,
        stage2_start_epoch: int = 10,  # 从第10个epoch开始第二阶段训练
        stage2_frequency: int = 5,     # 每5个iteration进行一次第二阶段训练
        memory_efficient: bool = True
    ):
        """
        初始化在线第二阶段训练器
        
        Args:
            stage2_config: 第二阶段配置参数
            device: 计算设备
            enable_stage2: 是否启用第二阶段训练
            stage2_start_epoch: 开始第二阶段训练的epoch
            stage2_frequency: 第二阶段训练频率
            memory_efficient: 是否启用内存优化模式
        """
        self.device = device
        self.enable_stage2 = enable_stage2
        self.stage2_start_epoch = stage2_start_epoch
        self.stage2_frequency = stage2_frequency
        self.memory_efficient = memory_efficient
        
        if not enable_stage2:
            print("Stage2 training disabled")
            return
        
        # 初始化第二阶段模型
        self.stage2_model = self._create_stage2_model(stage2_config)
        self.stage2_criterion = self._create_stage2_criterion(stage2_config)
        
        # 初始化在线动态物体处理器
        self.dynamic_processor = OnlineDynamicProcessor(
            device=device,
            memory_efficient=memory_efficient,
            **stage2_config.get('dynamic_processor', {})
        )
        
        # 优化器（将在外部设置）
        self.stage2_optimizer = None
        
        # 统计信息
        self.stage2_iteration_count = 0
        self.last_stage2_loss = 0.0
        self.stage2_training_time = 0.0
        self.stage2_skip_count = 0  # 跳过的训练次数
        self.stage2_memory_usage = []  # 内存使用记录
        
        print(f"OnlineStage2Trainer initialized:")
        print(f"  - Start epoch: {stage2_start_epoch}")
        print(f"  - Training frequency: {stage2_frequency}")
        print(f"  - Memory efficient mode: {memory_efficient}")
    
    def _create_stage2_model(self, config: Dict[str, Any]) -> Stage2Refiner:
        """创建第二阶段模型"""
        gaussian_refine_config = {
            "input_gaussian_dim": config.get('input_gaussian_dim', 14),
            "output_gaussian_dim": config.get('output_gaussian_dim', 11),
            "feature_dim": config.get('gaussian_feature_dim', 128),  # 减少特征维度以节省内存
            "num_attention_layers": config.get('gaussian_num_layers', 2),  # 减少层数
            "num_heads": config.get('gaussian_num_heads', 4),
            "mlp_ratio": config.get('gaussian_mlp_ratio', 2.0)
        }
        
        pose_refine_config = {
            "input_dim": 3,
            "feature_dim": config.get('pose_feature_dim', 128),
            "num_heads": config.get('pose_num_heads', 4),
            "num_layers": config.get('pose_num_layers', 2),
            "max_points": config.get('max_points_per_object', 2048)  # 减少点数以节省内存
        }
        
        model = Stage2Refiner(
            gaussian_refine_config=gaussian_refine_config,
            pose_refine_config=pose_refine_config,
            training_mode=config.get('training_mode', 'joint')
        )
        
        model.to(self.device)
        model.gradient_checkpointing_enable(True)  # 启用梯度检查点节省内存
        
        return model
    
    def _create_stage2_criterion(self, config: Dict[str, Any]) -> Stage2CompleteLoss:
        """创建第二阶段损失函数"""
        render_loss_config = {
            'rgb_weight': config.get('rgb_loss_weight', 0.5),  # 降低权重避免过拟合
            'depth_weight': config.get('depth_loss_weight', 0.0),  # 使用配置中的实际值
            'lpips_weight': config.get('lpips_loss_weight', 0.05),
            'consistency_weight': config.get('consistency_loss_weight', 0.0)  # 使用配置中的实际值
        }
        
        geometric_loss_config = {
            'gaussian_regularization_weight': config.get('gaussian_reg_weight', 0.0),  # 使用配置中的实际值
            'pose_regularization_weight': config.get('pose_reg_weight', 0.0),  # 使用配置中的实际值
            'temporal_smoothness_weight': config.get('temporal_smooth_weight', 0.0)  # 使用配置中的实际值
        }
        
        criterion = Stage2CompleteLoss(
            render_loss_config=render_loss_config,
            geometric_loss_config=geometric_loss_config
        )
        criterion.to(self.device)
        
        return criterion
    
    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """设置第二阶段优化器"""
        self.stage2_optimizer = optimizer
    
    def should_run_stage2(self, epoch: int, iteration: int) -> bool:
        """判断是否应该运行第二阶段训练"""
        if not self.enable_stage2:
            return False
        
        if epoch < self.stage2_start_epoch:
            return False
        
        return (iteration % self.stage2_frequency) == 0
    
    def process_stage1_outputs(
        self,
        preds: Dict[str, torch.Tensor],
        vggt_batch: Dict[str, torch.Tensor],
        auxiliary_models: Dict[str, Any],
        epoch: int,
        iteration: int
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """
        处理第一阶段输出，进行第二阶段训练
        
        Args:
            preds: 第一阶段预测结果
            vggt_batch: VGGT批次数据
            auxiliary_models: 辅助模型
            epoch: 当前epoch
            iteration: 当前iteration
            
        Returns:
            stage2_loss: 第二阶段损失（如果执行了训练）
            loss_dict: 详细损失字典
        """
        if not self.should_run_stage2(epoch, iteration):
            self.stage2_skip_count += 1
            return None, {}
        
        start_time = time.time()
        
        # 记录初始GPU内存使用
        if self.memory_efficient and torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        try:
            # 实时处理动态物体
            dynamic_objects_data = self.dynamic_processor.process_dynamic_objects(
                preds, vggt_batch, auxiliary_models
            )
            
            if not dynamic_objects_data or len(dynamic_objects_data['dynamic_objects']) == 0:
                print(f"No dynamic objects found in iteration {iteration}")
                # 返回零损失而不是None，这样stage2损失会显示在日志中
                return torch.tensor(0.0, device=self.device, requires_grad=True), {
                    'stage2_rgb_loss': 0.0,
                    'stage2_depth_loss': 0.0,
                    'stage2_consistency_loss': 0.0,
                    'stage2_gaussian_reg': 0.0,
                    'stage2_pose_reg': 0.0,
                    'stage2_temporal_smooth': 0.0
                }
            
            # 执行第二阶段前向传播
            with torch.cuda.amp.autocast(enabled=True):  # 使用混合精度
                stage2_loss, loss_dict = self._run_stage2_forward(
                    dynamic_objects_data, vggt_batch, preds
                )
            
            # 不在这里执行反向传播，让主训练循环处理
            # Stage2优化器将在主训练循环中调用
            
            # 更新统计信息
            self.stage2_iteration_count += 1
            self.last_stage2_loss = float(stage2_loss) if stage2_loss is not None else 0.0
            self.stage2_training_time += (time.time() - start_time)
            
            # 返回原始loss，让主循环处理backward
            return_loss = stage2_loss
            return_dict = loss_dict
            
            # 内存清理
            if self.memory_efficient:
                # 记录最终GPU内存使用
                if torch.cuda.is_available():
                    final_memory = torch.cuda.memory_allocated() / 1024**2  # MB
                    memory_delta = final_memory - initial_memory
                    self.stage2_memory_usage.append(memory_delta)
                    # 只保留最近100次的记录
                    if len(self.stage2_memory_usage) > 100:
                        self.stage2_memory_usage = self.stage2_memory_usage[-100:]
                
                # 清理GPU显存
                torch.cuda.empty_cache()
                # 清理中间变量
                del dynamic_objects_data
                del stage2_loss
                del loss_dict
            
            return return_loss, return_dict
            
        except Exception as e:
            import traceback
            print(f"Error in stage2 processing at epoch {epoch}, iter {iteration}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return None, {}
    
    def _run_stage2_forward(
        self,
        dynamic_objects_data: Dict[str, Any],
        vggt_batch: Dict[str, torch.Tensor],
        preds: Dict[str, Any]
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """执行第二阶段前向传播"""
        
        dynamic_objects = dynamic_objects_data['dynamic_objects']
        static_gaussians = dynamic_objects_data.get('static_gaussians')
        
        # 第二阶段模型前向传播
        refinement_results = self.stage2_model(
            dynamic_objects=dynamic_objects,
            static_gaussians=static_gaussians
        )
        
        # 获取细化后的场景
        refined_scene = self.stage2_model.get_refined_scene(
            refinement_results, static_gaussians
        )
        
        # 计算损失
        B, S, C, H, W = vggt_batch['images'].shape
        gt_images = vggt_batch['images']
        gt_depths = vggt_batch.get('depths', torch.ones(B, S, H, W, device=self.device) * 5.0)
        
        # 使用预测的相机参数而不是GT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            preds["pose_enc"], vggt_batch["images"].shape[-2:]
        )
        # 添加齐次坐标行到外参矩阵
        extrinsics = torch.cat([
            extrinsics, 
            torch.tensor([0, 0, 0, 1], device=extrinsics.device)[None, None, None, :].repeat(1, extrinsics.shape[1], 1, 1)
        ], dim=-2)
        
        
        loss_dict = self.stage2_criterion(
            refinement_results=refinement_results,
            refined_scene=refined_scene,
            gt_images=gt_images,
            gt_depths=gt_depths,
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
        
        stage2_loss = loss_dict.get('stage2_final_total_loss')
        
        # 转换损失字典为float，并确保键名不重复添加stage2前缀
        float_loss_dict = {}
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                float_loss_dict[k] = float(v)
            else:
                float_loss_dict[k] = v
        
        # 内存优化：清理中间变量
        if self.memory_efficient:
            del refinement_results
            del refined_scene
            del gt_images
            del gt_depths
            del intrinsics
            del loss_dict
        
        return stage2_loss, float_loss_dict
    
    def get_stage2_parameters(self) -> List[torch.nn.Parameter]:
        """获取第二阶段模型参数"""
        if not self.enable_stage2:
            return []
        
        return list(self.stage2_model.parameters())
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取第二阶段训练统计信息"""
        if not self.enable_stage2:
            return {}
        
        avg_training_time = self.stage2_training_time / max(self.stage2_iteration_count, 1)
        avg_memory_usage = sum(self.stage2_memory_usage) / max(len(self.stage2_memory_usage), 1)
        
        return {
            'stage2_enabled': self.enable_stage2,
            'stage2_iteration_count': self.stage2_iteration_count,
            'stage2_skip_count': self.stage2_skip_count,
            'stage2_last_loss': self.last_stage2_loss,
            'stage2_total_training_time': self.stage2_training_time,
            'stage2_avg_training_time': avg_training_time,
            'stage2_avg_memory_usage_mb': avg_memory_usage,
            'stage2_memory_efficiency_ratio': 1.0 - (self.stage2_skip_count / max(self.stage2_iteration_count + self.stage2_skip_count, 1)),
            'dynamic_processor_stats': self.dynamic_processor.get_statistics()
        }
    
    def set_training_mode(self, mode: str):
        """设置第二阶段训练模式"""
        if self.enable_stage2:
            self.stage2_model.set_training_mode(mode)
    
    def save_state_dict(self) -> Dict[str, Any]:
        """保存第二阶段模型状态"""
        if not self.enable_stage2:
            return {}
        
        return {
            'stage2_model': self.stage2_model.state_dict(),
            'stage2_iteration_count': self.stage2_iteration_count,
            'stage2_training_time': self.stage2_training_time
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载第二阶段模型状态"""
        if not self.enable_stage2 or not state_dict:
            return
        
        if 'stage2_model' in state_dict:
            self.stage2_model.load_state_dict(state_dict['stage2_model'])
        
        self.stage2_iteration_count = state_dict.get('stage2_iteration_count', 0)
        self.stage2_training_time = state_dict.get('stage2_training_time', 0.0)
    
    def train(self):
        """设置为训练模式"""
        if self.enable_stage2:
            self.stage2_model.train()
    
    def eval(self):
        """设置为评估模式"""
        if self.enable_stage2:
            self.stage2_model.eval()