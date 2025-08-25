# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple, List


class PointNetEncoder(nn.Module):
    """PointNet-style encoder for processing point clouds with features."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
            
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, input_dim] point features
        Returns:
            [N, output_dim] encoded features
        """
        return self.encoder(x)


class SelfAttentionBlock(nn.Module):
    """Self-attention block for point cloud feature refinement."""
    
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, dim] point features
        Returns:
            [N, dim] refined features
        """
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class GaussianRefineHead(nn.Module):
    """
    Gaussian参数细化网络头
    
    输入：聚合后的动态物体Gaussian参数
    输出：细化的Gaussian参数变化量（不包含velocity）
    """
    
    def __init__(
        self,
        input_gaussian_dim: int = 14,  # 原始Gaussian参数维度 
        output_gaussian_dim: int = 11,  # 输出参数维度(不包含velocity的3维)
        feature_dim: int = 256,
        num_attention_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.input_gaussian_dim = input_gaussian_dim
        self.output_gaussian_dim = output_gaussian_dim
        self.feature_dim = feature_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_gaussian_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Self-attention layers for feature refinement
        self.attention_layers = nn.ModuleList([
            SelfAttentionBlock(feature_dim, num_heads, mlp_ratio)
            for _ in range(num_attention_layers)
        ])
        
        # Position encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, 10000, feature_dim) * 0.02)
        
        # Output projection for Gaussian parameter deltas
        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, output_gaussian_dim)
        )
        
        # Initialize output layer to predict small changes
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)
        
    def forward(self, gaussian_params: torch.Tensor, point_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            gaussian_params: [N, input_gaussian_dim] 输入的Gaussian参数
            point_positions: [N, 3] 点的3D位置，用于位置编码（可选）
            
        Returns:
            gaussian_deltas: [N, output_gaussian_dim] Gaussian参数的变化量
        """
        N = gaussian_params.shape[0]
        
        # Input projection
        features = self.input_proj(gaussian_params)  # [N, feature_dim]
        
        # Add position encoding
        if N <= self.pos_encoding.shape[1]:
            pos_enc = self.pos_encoding[:, :N, :]  # [1, N, feature_dim]
            features = features + pos_enc.squeeze(0)  # [N, feature_dim]
        else:
            # Handle case where we have more points than pre-allocated position encodings
            pos_enc = F.interpolate(
                self.pos_encoding.permute(0, 2, 1),  # [1, feature_dim, 10000]
                size=N,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)  # [1, N, feature_dim]
            features = features + pos_enc.squeeze(0)
        
        # Apply self-attention layers
        for attention_layer in self.attention_layers:
            features = attention_layer(features)  # [N, feature_dim]
        
        # Output projection to get parameter deltas
        gaussian_deltas = self.output_proj(features)  # [N, output_gaussian_dim]
        
        return gaussian_deltas
    
    def apply_deltas(self, original_params: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """
        将预测的变化量应用到原始Gaussian参数上
        
        Args:
            original_params: [N, 14] 原始Gaussian参数
            deltas: [N, 11] 预测的变化量
            
        Returns:
            refined_params: [N, 14] 细化后的Gaussian参数
        """
        # 提取原始参数的各个组件
        # 假设Gaussian参数顺序为: [xyz(3), scale(3), color(3), rotation(4), opacity(1)]
        # 输出deltas顺序为: [xyz(3), scale(3), color(3), rotation(4), opacity(1)] (11维，不包含velocity)
        
        # Debug: 打印张量形状
        
        refined_params = original_params.clone()
        
        # 应用位置变化 (前3维) - 避免in-place操作
        position_refined = refined_params[:, :3] + deltas[:, :3]
        
        # 应用尺度变化 (4-6维) - 避免in-place操作
        scale_refined = refined_params[:, 3:6] + deltas[:, 3:6]
        
        # 应用颜色变化 (7-9维) - 避免in-place操作
        color_refined = refined_params[:, 6:9] + deltas[:, 6:9]
        
        # 应用旋转变化 
        # 对四元数使用特殊的更新方式
        original_quat = refined_params[:, 9:13]  # 原始参数中的四元数位置 [9:13]
        
        # 检查deltas的维度以适配输出维度
        if deltas.shape[1] >= 13:  # 如果deltas有足够的维度
            delta_quat = deltas[:, 9:13]
        elif deltas.shape[1] == 11:  # 如果deltas只有11维 (去掉velocity的3维)
            delta_quat = deltas[:, 7:11]  # 对应的四元数位置应该是 [7:11]
        else:
            # 如果维度不匹配，创建零四元数增量
            delta_quat = torch.zeros(deltas.shape[0], 4, device=deltas.device)
        
        # 归一化原始四元数
        original_quat_norm = F.normalize(original_quat, p=2, dim=-1)
        
        # 将delta视为轴角表示的小旋转，转换为四元数
        delta_quat_norm = F.normalize(delta_quat, p=2, dim=-1, eps=1e-8)
        
        # 四元数乘法来组合旋转 - 避免in-place操作
        refined_quat = self._quaternion_multiply(original_quat_norm, delta_quat_norm)
        rotation_refined = F.normalize(refined_quat, p=2, dim=-1)
        
        # 应用透明度变化 - 避免in-place操作
        if deltas.shape[1] >= 14:  # 如果deltas有足够的维度
            opacity_delta = deltas[:, 13:14]
        elif deltas.shape[1] == 11:  # 如果deltas只有11维
            opacity_delta = deltas[:, 10:11]  # 透明度在第10维
        else:
            opacity_delta = torch.zeros(deltas.shape[0], 1, device=deltas.device)
            
        opacity_refined = refined_params[:, 13:14] + opacity_delta
        
        # 构建完整的refined_params张量，避免所有in-place操作
        refined_params = torch.cat([
            position_refined,           # [:, :3]
            scale_refined,             # [:, 3:6] 
            color_refined,             # [:, 6:9]
            rotation_refined,          # [:, 9:13]
            opacity_refined            # [:, 13:14]
        ], dim=1)
        
        return refined_params
    
    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        四元数乘法: q1 * q2
        
        Args:
            q1, q2: [N, 4] 四元数 (w, x, y, z)
            
        Returns:
            [N, 4] 乘积四元数
        """
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.stack([w, x, y, z], dim=-1)