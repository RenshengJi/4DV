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
import torch.distributed as dist
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

        # 网络耗时统计
        self.gaussian_refine_times = []  # Gaussian refine网络耗时
        self.pose_refine_times = []  # Pose refine网络耗时
        
        print(f"OnlineStage2Trainer initialized:")
        print(f"  - Start epoch: {stage2_start_epoch}")
        print(f"  - Training frequency: {stage2_frequency}")
        print(f"  - Memory efficient mode: {memory_efficient}")
    
    def _create_stage2_model(self, config: Dict[str, Any]) -> Stage2Refiner:
        """创建第二阶段模型"""
        # 使用稀疏卷积配置
        gaussian_refine_config = {
            "input_gaussian_dim": config.get('input_gaussian_dim', 14),
            "output_gaussian_dim": config.get('output_gaussian_dim', 14),
            "feature_dim": config.get('gaussian_feature_dim', 128),
            "num_conv_layers": config.get('gaussian_num_conv_layers', 2),
            "voxel_size": config.get('gaussian_voxel_size', 0.05),
            "max_num_points_per_voxel": config.get('max_num_points_per_voxel', 5),
            "use_dilated_conv": config.get('use_dilated_conv', False),
            "dilation_rates": config.get('gaussian_dilation_rates', None)
        }

        pose_refine_config = {
            "input_dim": 3,
            "feature_dim": config.get('pose_feature_dim', 128),
            "num_conv_layers": config.get('pose_num_conv_layers', 2),
            "voxel_size": config.get('pose_voxel_size', 0.1),
            "max_points": config.get('max_points_per_object', 4096),
            "use_dilated_conv": config.get('use_dilated_conv', False),
            "dilation_rates": config.get('pose_dilation_rates', None)
        }

        training_mode = config.get('stage2_training_mode', config.get('training_mode', 'joint'))
        print(f"  - Stage2 training mode: {training_mode}")

        # 打印dilated conv配置
        if config.get('use_dilated_conv', False):
            print(f"  - Using dilated convolution:")
            print(f"    Gaussian dilation rates: {config.get('gaussian_dilation_rates', None)}")
            print(f"    Pose dilation rates: {config.get('pose_dilation_rates', None)}")

        model = Stage2Refiner(
            gaussian_refine_config=gaussian_refine_config,
            pose_refine_config=pose_refine_config,
            training_mode=training_mode
        )

        model.to(self.device)

        return model
    
    def _create_stage2_criterion(self, config: Dict[str, Any]) -> Stage2CompleteLoss:
        """创建第二阶段损失函数"""
        render_loss_config = {
            'rgb_weight': config.get('stage2_rgb_loss_weight', 0.5),  # 修正：使用stage2_前缀的键名
            'depth_weight': config.get('stage2_depth_loss_weight', 0.0),  # 修正：使用stage2_前缀的键名
            'render_only_dynamic': config.get('stage2_render_only_dynamic', False),  # 是否只渲染动态物体
            'supervise_only_dynamic': config.get('stage2_supervise_only_dynamic', False),  # 是否只监督动态区域
        }

        criterion = Stage2CompleteLoss(
            render_loss_config=render_loss_config
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
        处理第一阶段输出，进行第二阶段训练（支持DDP多卡训练）

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
        # 检查是否使用分布式训练
        is_distributed = dist.is_available() and dist.is_initialized()
        is_main_process = not is_distributed or dist.get_rank() == 0

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

            # 检查是否有动态物体
            num_objects = len(dynamic_objects_data['dynamic_objects']) if dynamic_objects_data else 0

            # 显示阶段时间统计（仅主进程）
            if is_main_process and dynamic_objects_data and 'stage_times' in dynamic_objects_data:
                stage_times = dynamic_objects_data['stage_times']
                print(f"[Stage Times] Preprocessing: {stage_times.get('preprocessing', 0):.3f}s | "
                      f"Clustering+Background: {stage_times.get('clustering_background', 0):.3f}s | "
                      f"Tracking: {stage_times.get('tracking', 0):.3f}s | "
                      f"Aggregation: {stage_times.get('aggregation', 0):.3f}s | "
                      f"Total: {dynamic_objects_data.get('processing_time', 0):.3f}s | "
                      f"Objects: {num_objects}")

            # 【修改】每个进程独立判断，不再强制同步
            # 如果当前进程没有物体，直接返回零损失（不影响其他进程）
            if num_objects == 0:
                # 创建一个带梯度的零损失（连接到计算图）
                dummy_param = next(self.stage2_model.parameters())
                zero_loss = (dummy_param * 0.0).sum()  # 这个tensor有grad_fn，可以backward
                return zero_loss, {
                    'stage2_rgb_loss': 0.0,
                    'stage2_depth_loss': 0.0,
                    'stage2_total_loss': 0.0,
                    'stage2_final_total_loss': 0.0,
                    'stage2_skipped': True  # 标记为跳过，用于过滤tensorboard日志
                }

            # 【修改】预先验证动态物体数据的合法性（避免spconv SIGABRT）
            # 每个进程独立判断，验证失败时只跳过当前进程
            if not self._validate_dynamic_objects(dynamic_objects_data):
                if is_main_process:
                    print(f"[WARNING] Stage2 validation failed at iteration {iteration}, skipping this GPU")
                # 创建带梯度的零损失
                dummy_param = next(self.stage2_model.parameters())
                zero_loss = (dummy_param * 0.0).sum()
                return zero_loss, {
                    'stage2_rgb_loss': 0.0,
                    'stage2_depth_loss': 0.0,
                    'stage2_total_loss': 0.0,
                    'stage2_final_total_loss': 0.0,
                    'stage2_skipped': True  # 标记为跳过
                }

            # 执行第二阶段前向传播 (包装在内部try-catch中以捕获CUDA/渲染错误)
            stage2_loss = None
            loss_dict = {}
            try:
                with torch.cuda.amp.autocast(enabled=True):  # 使用混合精度
                    stage2_loss, loss_dict = self._run_stage2_forward(
                        dynamic_objects_data, vggt_batch, preds
                    )
            except Exception as forward_error:
                # 【修改】捕获前向传播中的任何错误（CUDA OOM、渲染失败等）
                # 每个进程独立处理错误，不强制同步
                if is_main_process:
                    print(f"ERROR in _run_stage2_forward at iteration {iteration}: {forward_error}")
                    import traceback
                    print(f"Forward error traceback: {traceback.format_exc()}")

                # 返回零损失（带梯度），只影响当前GPU
                dummy_param = next(self.stage2_model.parameters())
                zero_loss = (dummy_param * 0.0).sum()
                return zero_loss, {
                    'stage2_rgb_loss': 0.0,
                    'stage2_depth_loss': 0.0,
                    'stage2_total_loss': 0.0,
                    'stage2_final_total_loss': 0.0,
                    'stage2_skipped': True  # 标记为跳过
                }

            # 【修改】移除barrier，允许不同GPU独立推进
            # DDP的梯度同步会在backward时自动处理，不需要显式barrier

            # 更新统计信息
            self.stage2_iteration_count += 1
            self.last_stage2_loss = float(stage2_loss) if stage2_loss is not None else 0.0
            self.stage2_training_time += (time.time() - start_time)

            # 实时显示网络耗时（仅主进程）
            if is_main_process and (len(self.gaussian_refine_times) > 0 or len(self.pose_refine_times) > 0):
                # 计算最新的耗时
                latest_gaussian = self.gaussian_refine_times[-1] * 1000 if self.gaussian_refine_times else 0.0
                latest_pose = self.pose_refine_times[-1] * 1000 if self.pose_refine_times else 0.0
                total_time = (time.time() - start_time) * 1000

                # 从loss_dict获取详细时间
                refine_time = loss_dict.get('refine_time_ms', 0.0)
                scene_time = loss_dict.get('scene_time_ms', 0.0)
                loss_time = loss_dict.get('loss_time_ms', 0.0)

                print(f"[Network Times] Gaussian: {latest_gaussian:.2f}ms | Pose: {latest_pose:.2f}ms")
                print(f"[Stage2 Times] Refine: {refine_time:.2f}ms | Scene: {scene_time:.2f}ms | Loss: {loss_time:.2f}ms | Total: {total_time:.2f}ms")

            # 每20次有效迭代打印平均统计（仅主进程）
            if is_main_process and self.stage2_iteration_count % 20 == 0 and len(self.gaussian_refine_times) > 0:
                stats = self.get_statistics()
                print(f"\n[Stage2 Avg Stats - Iter {self.stage2_iteration_count}]")
                print(f"  Gaussian Refine Avg: {stats['gaussian_refine_avg_time_ms']:.2f}ms (count: {stats['gaussian_refine_count']})")
                print(f"  Pose Refine Avg: {stats['pose_refine_avg_time_ms']:.2f}ms (count: {stats['pose_refine_count']})")
                print(f"  Stage2 Total Avg: {stats['stage2_avg_training_time']*1000:.2f}ms\n")

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
            # 【修改】外层异常捕获，每个进程独立处理，不强制同步
            import traceback
            if is_main_process:
                print(f"Error in stage2 processing at epoch {epoch}, iter {iteration}: {e}")
                print(f"Traceback: {traceback.format_exc()}")

            # 返回零损失（带梯度），只影响当前GPU
            dummy_param = next(self.stage2_model.parameters())
            zero_loss = (dummy_param * 0.0).sum()
            return zero_loss, {
                'stage2_rgb_loss': 0.0,
                'stage2_depth_loss': 0.0,
                'stage2_total_loss': 0.0,
                'stage2_final_total_loss': 0.0,
                'stage2_skipped': True  # 标记为跳过
            }
    
    def _validate_dynamic_objects(self, dynamic_objects_data: Dict[str, Any]) -> bool:
        """
        验证动态物体数据的合法性，避免spconv SIGABRT

        Args:
            dynamic_objects_data: 动态物体处理结果

        Returns:
            bool: True if valid, False otherwise
        """
        if not dynamic_objects_data:
            return False

        dynamic_objects = dynamic_objects_data.get('dynamic_objects', [])
        if not dynamic_objects:
            return False

        # 验证每个物体的canonical_gaussians
        for obj in dynamic_objects:
            canonical_gaussians = obj.get('canonical_gaussians')
            if canonical_gaussians is None:
                print(f"[WARNING] Object {obj.get('object_id', 'unknown')} has None canonical_gaussians")
                return False

            # 检查形状
            if canonical_gaussians.shape[0] == 0:
                print(f"[WARNING] Object {obj.get('object_id', 'unknown')} has empty canonical_gaussians")
                return False

            # 检查NaN/Inf
            if torch.isnan(canonical_gaussians).any() or torch.isinf(canonical_gaussians).any():
                print(f"[WARNING] Object {obj.get('object_id', 'unknown')} has NaN/Inf in canonical_gaussians")
                return False

            # 检查坐标范围（避免超大坐标导致spconv溢出）
            positions = canonical_gaussians[:, :3]
            pos_range = positions.max() - positions.min()
            if pos_range > 1000.0:  # 超过1000米认为异常
                print(f"[WARNING] Object {obj.get('object_id', 'unknown')} has excessive position range: {pos_range.item():.2f}m")
                return False

        return True

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
        t0 = time.time()
        try:
            refinement_results = self.stage2_model(
                dynamic_objects=dynamic_objects,
                static_gaussians=static_gaussians,
                preds=preds
            )
        except Exception as refine_error:
            print(f"ERROR in stage2_model forward: {refine_error}")
            print(f"  - num_dynamic_objects: {len(dynamic_objects)}")
            print(f"  - static_gaussians shape: {static_gaussians.shape if static_gaussians is not None else None}")
            raise  # Re-raise to be caught by outer try-catch
        refine_time = time.time() - t0

        # 收集网络耗时统计
        timing_stats = refinement_results.get('timing_stats', {})
        if timing_stats.get('num_gaussian_refines', 0) > 0:
            avg_gaussian_time = timing_stats['gaussian_refine_time'] / timing_stats['num_gaussian_refines']
            self.gaussian_refine_times.append(avg_gaussian_time)
        if timing_stats.get('num_pose_refines', 0) > 0:
            avg_pose_time = timing_stats['pose_refine_time'] / timing_stats['num_pose_refines']
            self.pose_refine_times.append(avg_pose_time)

        # 只保留最近100次的记录
        if len(self.gaussian_refine_times) > 100:
            self.gaussian_refine_times = self.gaussian_refine_times[-100:]
        if len(self.pose_refine_times) > 100:
            self.pose_refine_times = self.pose_refine_times[-100:]

        # 直接构建渲染需要的场景格式
        t1 = time.time()
        refined_scene = {
            'static_gaussians': static_gaussians,
            'dynamic_objects': refinement_results['refined_dynamic_objects']
        }
        scene_time = time.time() - t1

        # 计算损失
        t2 = time.time()
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


        # 获取sky_masks（如果有的话）
        sky_masks = vggt_batch.get('sky_masks', None)

        try:
            loss_dict = self.stage2_criterion(
                refinement_results=refinement_results,
                refined_scene=refined_scene,
                gt_images=gt_images,
                gt_depths=gt_depths,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                sky_masks=sky_masks
            )
        except Exception as loss_error:
            print(f"ERROR in stage2_criterion (loss computation): {loss_error}")
            print(f"  - gt_images shape: {gt_images.shape}")
            print(f"  - intrinsics shape: {intrinsics.shape}")
            print(f"  - extrinsics shape: {extrinsics.shape}")
            raise  # Re-raise to be caught by outer try-catch
        loss_time = time.time() - t2

        stage2_loss = loss_dict.get('stage2_final_total_loss')

        # 添加时间统计到loss_dict
        float_loss_dict = {
            'refine_time_ms': refine_time * 1000,
            'scene_time_ms': scene_time * 1000,
            'loss_time_ms': loss_time * 1000
        }

        # 转换损失字典为float，并确保键名不重复添加stage2前缀
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
        """获取第二阶段模型参数（只返回requires_grad=True的参数）"""
        if not self.enable_stage2:
            return []

        # 只返回需要梯度的参数
        params = [p for p in self.stage2_model.parameters() if p.requires_grad]
        return params
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取第二阶段训练统计信息"""
        if not self.enable_stage2:
            return {}

        avg_training_time = self.stage2_training_time / max(self.stage2_iteration_count, 1)
        avg_memory_usage = sum(self.stage2_memory_usage) / max(len(self.stage2_memory_usage), 1)

        # 计算网络耗时统计
        avg_gaussian_refine_time = sum(self.gaussian_refine_times) / max(len(self.gaussian_refine_times), 1) if self.gaussian_refine_times else 0.0
        avg_pose_refine_time = sum(self.pose_refine_times) / max(len(self.pose_refine_times), 1) if self.pose_refine_times else 0.0

        return {
            'stage2_enabled': self.enable_stage2,
            'stage2_iteration_count': self.stage2_iteration_count,
            'stage2_skip_count': self.stage2_skip_count,
            'stage2_last_loss': self.last_stage2_loss,
            'stage2_total_training_time': self.stage2_training_time,
            'stage2_avg_training_time': avg_training_time,
            'stage2_avg_memory_usage_mb': avg_memory_usage,
            'stage2_memory_efficiency_ratio': 1.0 - (self.stage2_skip_count / max(self.stage2_iteration_count + self.stage2_skip_count, 1)),
            'gaussian_refine_avg_time_ms': avg_gaussian_refine_time * 1000,  # 转换为毫秒
            'pose_refine_avg_time_ms': avg_pose_refine_time * 1000,  # 转换为毫秒
            'gaussian_refine_count': len(self.gaussian_refine_times),
            'pose_refine_count': len(self.pose_refine_times),
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