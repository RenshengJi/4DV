"""
ICP Supervision Training Script - 使用ICP GT训练Gaussian Refine网络
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入Stage2 refine head
from vggt.vggt.heads.sparse_conv_refine_head import GaussianRefineHeadSparseConv

# 导入ICP supervision模块
from icp_supervision.dataset import create_icp_dataloaders
from icp_supervision.icp_loss import ICPSupervisionLoss, ICPChamferLoss
from icp_supervision.utils import torch_to_numpy


class ICPTrainer:
    """
    ICP监督训练器
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device(config['device'])

        # 创建输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))

        # 创建数据加载器
        print("Creating data loaders...")
        self.train_loader, self.val_loader = create_icp_dataloaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            train_ratio=config['train_ratio'],
            seed=config['seed'],
            device=self.device,
            cache_in_memory=config.get('cache_in_memory', False)
        )

        print(f"  Train batches: {len(self.train_loader)}")

        # 检查是否有验证集
        self.has_val = config['train_ratio'] < 1.0 and len(self.val_loader) > 0
        if self.has_val:
            print(f"  Val batches: {len(self.val_loader)}")
        else:
            print(f"  Val batches: 0 (train_ratio=1.0, skipping validation)")

        # 创建模型
        print("\nCreating Gaussian Refine model...")
        self.model = self._create_model()
        self.model = self.model.to(self.device)

        # 统计参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # 创建损失函数
        self.criterion = ICPSupervisionLoss(
            position_weight=config.get('position_weight', 10.0),
            scale_weight=config.get('scale_weight', 1.0),
            rotation_weight=config.get('rotation_weight', 1.0),
            color_weight=config.get('color_weight', 1.0),
            opacity_weight=config.get('opacity_weight', 1.0),
            use_smooth_l1=config.get('use_smooth_l1', False),
            position_only=config.get('position_only', False),
        )

        # 可选的Chamfer loss
        if config.get('use_chamfer_loss', False):
            self.chamfer_criterion = ICPChamferLoss()
            self.chamfer_weight = config.get('chamfer_weight', 1.0)
        else:
            self.chamfer_criterion = None

        # 创建优化器
        self.optimizer = self._create_optimizer()

        # 学习率调度器
        self.scheduler = self._create_scheduler()

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # 加载checkpoint (如果指定)
        if config.get('resume_from', None):
            self._load_checkpoint(config['resume_from'])

    def _create_model(self) -> torch.nn.Module:
        """创建Gaussian Refine模型"""
        model = GaussianRefineHeadSparseConv(
            input_gaussian_dim=self.config.get('input_gaussian_dim', 14),
            output_gaussian_dim=self.config.get('output_gaussian_dim', 14),
            feature_dim=self.config.get('gaussian_feature_dim', 384),
            num_conv_layers=self.config.get('gaussian_num_conv_layers', 10),
            voxel_size=self.config.get('gaussian_voxel_size', 0.05),
        )
        return model

    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        optimizer_type = self.config.get('optimizer', 'adamw').lower()

        # 确保 lr 和 weight_decay 是数值类型（处理可能的字符串）
        lr = float(self.config['lr'])
        weight_decay = float(self.config.get('weight_decay', 1e-4))

        if optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        return optimizer

    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler_type = self.config.get('scheduler', 'cosine').lower()

        if scheduler_type == 'cosine':
            # 确保参数是正确的类型
            epochs = int(self.config['epochs'])
            min_lr = float(self.config.get('min_lr', 1e-6))

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=min_lr
            )
        elif scheduler_type == 'step':
            step_size = int(self.config.get('lr_decay_step', 10))
            gamma = float(self.config.get('lr_decay_gamma', 0.1))

            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            scheduler = None

        return scheduler

    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()

        epoch_loss = 0.0
        epoch_metrics = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['epochs']}")

        for batch_idx, batch in enumerate(pbar):
            if batch['batch_size'] == 0:
                continue

            batch_loss = 0.0
            batch_metrics = {}

            # 处理batch中的每个样本 (因为点数不同，所以是list)
            for i in range(batch['batch_size']):
                input_gaussians = batch['input_gaussians'][i].to(self.device)  # [N, 14]
                target_gaussians = batch['target_gaussians'][i].to(self.device)  # [N, 14]
                pred_scale = batch['pred_scale'][i].to(self.device)  # [1]

                # 前向传播
                pred_gaussians_delta = self.model(
                    input_gaussians,  # [N, 14] - 模型接收单个样本
                    pred_scale  # [1]
                )

                pred_gaussians = self.model.apply_deltas(
                    input_gaussians,
                    pred_gaussians_delta
                )

                # 计算损失
                loss, loss_dict = self.criterion(
                    pred_gaussians, target_gaussians, return_individual_losses=True
                )

                # 可选的Chamfer loss
                if self.chamfer_criterion is not None:
                    chamfer_loss = self.chamfer_criterion(
                        pred_gaussians[:, :3], target_gaussians[:, :3]
                    )
                    loss = loss + self.chamfer_weight * chamfer_loss
                    loss_dict['chamfer_loss'] = chamfer_loss

                batch_loss += loss

                # 累积metrics
                for key, value in loss_dict.items():
                    if key not in batch_metrics:
                        batch_metrics[key] = 0.0
                    batch_metrics[key] += value.item()

            # 平均loss
            batch_loss = batch_loss / batch['batch_size']

            # 反向传播
            self.optimizer.zero_grad()
            batch_loss.backward()

            # 梯度裁剪 (可选)
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config['grad_clip']
                )

            self.optimizer.step()

            # 更新统计
            epoch_loss += batch_loss.item()
            for key, value in batch_metrics.items():
                value_avg = value / batch['batch_size']
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value_avg

            # 更新进度条
            pbar.set_postfix({'loss': batch_loss.item()})

            # TensorBoard logging
            if self.global_step % self.config.get('log_freq', 10) == 0:
                for key, value in batch_metrics.items():
                    value_avg = value / batch['batch_size']
                    self.writer.add_scalar(f'train/{key}', value_avg, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1

        # Epoch平均
        epoch_loss /= len(self.train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= len(self.train_loader)

        return epoch_loss, epoch_metrics

    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()

        val_loss = 0.0
        val_metrics = {}

        for batch in tqdm(self.val_loader, desc="Validating"):
            if batch['batch_size'] == 0:
                continue

            batch_loss = 0.0
            batch_metrics = {}

            for i in range(batch['batch_size']):
                input_gaussians = batch['input_gaussians'][i].to(self.device)
                target_gaussians = batch['target_gaussians'][i].to(self.device)
                pred_scale = batch['pred_scale'][i].to(self.device)

                # 前向传播
                pred_gaussians_delta = self.model(
                    input_gaussians,  # [N, 14]
                    pred_scale
                )

                pred_gaussians = self.model.apply_deltas(
                    input_gaussians,
                    pred_gaussians_delta
                )

                # 计算损失
                loss, loss_dict = self.criterion(
                    pred_gaussians, target_gaussians, return_individual_losses=True
                )

                if self.chamfer_criterion is not None:
                    chamfer_loss = self.chamfer_criterion(
                        pred_gaussians[:, :3], target_gaussians[:, :3]
                    )
                    loss = loss + self.chamfer_weight * chamfer_loss
                    loss_dict['chamfer_loss'] = chamfer_loss

                batch_loss += loss

                for key, value in loss_dict.items():
                    if key not in batch_metrics:
                        batch_metrics[key] = 0.0
                    batch_metrics[key] += value.item()

            batch_loss = batch_loss / batch['batch_size']
            val_loss += batch_loss.item()

            for key, value in batch_metrics.items():
                value_avg = value / batch['batch_size']
                if key not in val_metrics:
                    val_metrics[key] = 0.0
                val_metrics[key] += value_avg

        # 平均
        val_loss /= len(self.val_loader)
        for key in val_metrics:
            val_metrics[key] /= len(self.val_loader)

        return val_loss, val_metrics

    def train(self):
        """完整训练循环"""
        print(f"\nStarting training for {self.config['epochs']} epochs...")

        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch

            # 训练
            train_loss, train_metrics = self.train_epoch(epoch)

            print(f"\nEpoch {epoch}/{self.config['epochs']} - Train Loss: {train_loss:.6f}")
            for key, value in train_metrics.items():
                print(f"  {key}: {value:.6f}")

            # 验证（仅当有验证集时）
            if self.has_val and (epoch + 1) % self.config.get('val_freq', 1) == 0:
                val_loss, val_metrics = self.validate()

                print(f"Validation Loss: {val_loss:.6f}")
                for key, value in val_metrics.items():
                    print(f"  {key}: {value:.6f}")

                # TensorBoard
                self.writer.add_scalar('val/total_loss', val_loss, epoch)
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{key}', value, epoch)

                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)
                    print(f"  ✓ Best model saved (val_loss: {val_loss:.6f})")
            elif not self.has_val:
                # 没有验证集时，基于训练损失保存最佳模型
                if train_loss < self.best_val_loss:
                    self.best_val_loss = train_loss
                    self._save_checkpoint(epoch, is_best=True)
                    print(f"  ✓ Best model saved (train_loss: {train_loss:.6f})")

            # 保存定期checkpoint
            if (epoch + 1) % self.config.get('save_freq', 10) == 0:
                self._save_checkpoint(epoch, is_best=False)

            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # 如果有验证集使用验证损失，否则使用训练损失
                    loss_for_scheduler = val_loss if (self.has_val and 'val_loss' in locals()) else train_loss
                    self.scheduler.step(loss_for_scheduler)
                else:
                    self.scheduler.step()

        print("\nTraining complete!")
        self.writer.close()

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # 保存最新checkpoint
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)

        # 保存最佳checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)

        # 保存epoch checkpoint
        epoch_path = self.checkpoint_dir / f'epoch_{epoch:04d}.pth'
        torch.save(checkpoint, epoch_path)

    def _load_checkpoint(self, checkpoint_path: str):
        """加载checkpoint"""
        print(f"Loading checkpoint from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"  Resumed from epoch {checkpoint['epoch']}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Train Gaussian Refine network with ICP supervision")

    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override data directory from config')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory from config')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 覆盖配置
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.resume:
        config['resume_from'] = args.resume

    # 创建训练器
    trainer = ICPTrainer(config)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
