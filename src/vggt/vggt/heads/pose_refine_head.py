# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np
import time
import logging

try:
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class PointCloudEncoder(nn.Module):
    """点云编码器，用于提取点云特征"""
    
    def __init__(self, input_dim: int = 3, feature_dim: int = 256, hidden_dims: List[int] = None):
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
            
        # Final projection to feature space
        layers.append(nn.Linear(prev_dim, feature_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: [N, input_dim] 点云坐标
        Returns:
            features: [N, feature_dim] 点云特征
        """
        return self.encoder(points)


class CrossAttention(nn.Module):
    """跨点云注意力机制，用于对比两个点云"""
    
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)  
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query_features: torch.Tensor, key_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_features: [N1, feature_dim] 查询点云特征
            key_features: [N2, feature_dim] 键值点云特征
            
        Returns:
            attended_features: [N1, feature_dim] 注意力加权后的特征
        """
        N1 = query_features.shape[0]
        N2 = key_features.shape[0]
        
        # Project to Q, K, V
        Q = self.q_proj(query_features).contiguous().view(N1, self.num_heads, self.head_dim)  # [N1, H, D]
        K = self.k_proj(key_features).contiguous().view(N2, self.num_heads, self.head_dim)    # [N2, H, D]
        V = self.v_proj(key_features).contiguous().view(N2, self.num_heads, self.head_dim)    # [N2, H, D]
        
        # Compute attention scores
        attn_scores = torch.einsum('qhd,khd->qhk', Q, K) * self.scale  # [N1, H, N2]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [N1, H, N2]
        
        # Apply attention to values
        attended = torch.einsum('qhk,khd->qhd', attn_weights, V)  # [N1, H, D]
        attended = attended.contiguous().view(N1, self.feature_dim)  # [N1, feature_dim]
        
        # Output projection
        output = self.out_proj(attended)  # [N1, feature_dim]
        
        return output


class LocalCrossAttention(nn.Module):
    """局部跨注意力机制，使用k-NN图优化跨点云注意力计算"""
    
    def __init__(self, feature_dim: int, num_heads: int = 8, k_neighbors: int = 32):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.k_neighbors = k_neighbors
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        self.scale = self.head_dim ** -0.5
        
    def _build_cross_knn_kdtree(self, query_positions: torch.Tensor, key_positions: torch.Tensor, k: int) -> torch.Tensor:
        """使用KD-tree构建跨点云k-NN图"""
        if not HAS_SKLEARN:
            return self._build_cross_knn_batched(query_positions, key_positions, k)
            
        N_query = query_positions.shape[0]
        N_key = key_positions.shape[0]
        
        # 如果key点云太小，回退到全连接
        if N_key <= k:
            # 返回所有key点的索引
            return torch.arange(N_key, device=query_positions.device).unsqueeze(0).repeat(N_query, 1)
        
        knn_start_time = time.time()
        
        try:
            # 使用KD-tree在key点云中为每个query点找k个最近邻
            key_positions_np = key_positions.detach().cpu().numpy()
            query_positions_np = query_positions.detach().cpu().numpy()
            
            nbrs = NearestNeighbors(n_neighbors=min(k, N_key), algorithm='kd_tree')
            nbrs.fit(key_positions_np)
            _, indices = nbrs.kneighbors(query_positions_np)
            
            knn_time = time.time() - knn_start_time
            
            
            return torch.from_numpy(indices).to(query_positions.device)
            
        except Exception:
            return self._build_cross_knn_batched(query_positions, key_positions, k)
    
    def _build_cross_knn_batched(self, query_positions: torch.Tensor, key_positions: torch.Tensor, k: int) -> torch.Tensor:
        """分批构建跨点云k-NN图，避免大矩阵"""
        N_query = query_positions.shape[0]
        N_key = key_positions.shape[0]
        
        if N_key <= k:
            return torch.arange(N_key, device=query_positions.device).unsqueeze(0).repeat(N_query, 1)
        
        knn_start_time = time.time()
        batch_size = min(1000, N_query)  # 分批处理
        knn_indices = []
        
        for i in range(0, N_query, batch_size):
            end_i = min(i + batch_size, N_query)
            query_batch = query_positions[i:end_i]  # [batch_size, 3]
            
            # 计算这批query点到所有key点的距离
            distances = torch.cdist(query_batch, key_positions)  # [batch_size, N_key]
            
            # 找到k个最近邻
            _, batch_indices = torch.topk(distances, min(k, N_key), dim=-1, largest=False)
            knn_indices.append(batch_indices)
        
        knn_time = time.time() - knn_start_time
        
        
        return torch.cat(knn_indices, dim=0)  # [N_query, k]
    
    def forward(self, 
                query_features: torch.Tensor, 
                key_features: torch.Tensor,
                query_positions: torch.Tensor,
                key_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_features: [N1, feature_dim] 查询特征
            key_features: [N2, feature_dim] 键值特征
            query_positions: [N1, 3] 查询点3D位置
            key_positions: [N2, 3] 键值点3D位置
            
        Returns:
            attended_features: [N1, feature_dim] 注意力加权后的特征
        """
        infer_start_time = time.time()
        N1 = query_features.shape[0]
        N2 = key_features.shape[0]
        k = min(self.k_neighbors, N2)
        
        # 如果k太小，使用全局注意力
        if k < 4:
            return self._global_cross_attention(query_features, key_features)
        
        # 构建k-NN图：为每个query点找到k个最近的key点
        knn_indices = self._build_cross_knn_kdtree(query_positions, key_positions, k)  # [N1, k]
        
        # Project to Q, K, V
        Q = self.q_proj(query_features).contiguous().view(N1, self.num_heads, self.head_dim)  # [N1, H, D]
        K = self.k_proj(key_features).contiguous().view(N2, self.num_heads, self.head_dim)    # [N2, H, D]
        V = self.v_proj(key_features).contiguous().view(N2, self.num_heads, self.head_dim)    # [N2, H, D]
        
        # 使用局部k-NN进行注意力计算
        attended_features = []
        
        for head in range(self.num_heads):
            q_head = Q[:, head, :]  # [N1, D]
            k_head = K[:, head, :]  # [N2, D]
            v_head = V[:, head, :]  # [N2, D]
            
            # 为每个query点收集其k个邻居的K和V
            k_neighbors = torch.gather(k_head.unsqueeze(0).expand(N1, -1, -1),
                                     1, knn_indices.unsqueeze(-1).expand(-1, -1, self.head_dim))  # [N1, k, D]
            v_neighbors = torch.gather(v_head.unsqueeze(0).expand(N1, -1, -1), 
                                     1, knn_indices.unsqueeze(-1).expand(-1, -1, self.head_dim))  # [N1, k, D]
            
            # 计算局部注意力分数
            attn_scores = torch.sum(q_head.unsqueeze(1) * k_neighbors, dim=-1) * self.scale  # [N1, k]
            attn_weights = F.softmax(attn_scores, dim=-1)  # [N1, k]
            
            # 应用注意力权重
            head_output = torch.sum(attn_weights.unsqueeze(-1) * v_neighbors, dim=1)  # [N1, D]
            attended_features.append(head_output)
        
        # 合并多头结果
        attended = torch.cat(attended_features, dim=-1)  # [N1, feature_dim]
        
        # 输出投影
        output = self.out_proj(attended)
        
        
        return output
    
    def _global_cross_attention(self, query_features: torch.Tensor, key_features: torch.Tensor) -> torch.Tensor:
        """回退到全局跨注意力"""
        N1 = query_features.shape[0]
        N2 = key_features.shape[0]
        
        Q = self.q_proj(query_features).contiguous().view(N1, self.num_heads, self.head_dim)
        K = self.k_proj(key_features).contiguous().view(N2, self.num_heads, self.head_dim)
        V = self.v_proj(key_features).contiguous().view(N2, self.num_heads, self.head_dim)
        
        attn_scores = torch.einsum('qhd,khd->qhk', Q, K) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attended = torch.einsum('qhk,khd->qhd', attn_weights, V)
        attended = attended.contiguous().view(N1, self.feature_dim)
        
        return self.out_proj(attended)


class PoseRefineHead(nn.Module):
    """
    位姿细化网络头
    
    输入：原始帧点云 + 变换后的参考点云
    输出：6DOF位姿变化量
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        feature_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        max_points: int = 8192,
        k_neighbors: int = 32,
        use_local_attention: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.max_points = max_points
        self.k_neighbors = k_neighbors
        self.use_local_attention = use_local_attention
        
        # 源点云和目标点云的编码器
        self.source_encoder = PointCloudEncoder(input_dim, feature_dim)
        self.target_encoder = PointCloudEncoder(input_dim, feature_dim)
        
        # 跨注意力层 - 选择局部或全局注意力
        if use_local_attention:
            self.cross_attention_layers = nn.ModuleList([
                LocalCrossAttention(feature_dim, num_heads, k_neighbors) for _ in range(num_layers)
            ])
        else:
            self.cross_attention_layers = nn.ModuleList([
                CrossAttention(feature_dim, num_heads) for _ in range(num_layers)
            ])
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 全局特征聚合
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # 位姿预测头 (6DOF: 3旋转 + 3平移)
        self.pose_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 4, 6)  # [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
        )
        
        # Initialize all layers with Xavier uniform
        self._init_weights()
        
    def _init_weights(self):
        """Initialize all linear layers with Xavier uniform"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(
        self, 
        source_points: torch.Tensor, 
        target_points: torch.Tensor,
        initial_transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            source_points: [N1, 3] 源点云
            target_points: [N2, 3] 目标点云  
            initial_transform: [4, 4] 初始变换矩阵（可选）
            
        Returns:
            pose_delta: [6] 位姿变化量 [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
        """
        total_start_time = time.time()
        N1_orig, N2_orig = source_points.shape[0], target_points.shape[0]
        
        # 点云采样（如果点数太多）
        source_points = self._sample_points(source_points, self.max_points)
        target_points = self._sample_points(target_points, self.max_points)
        
        N1_final, N2_final = source_points.shape[0], target_points.shape[0]
        
        # 编码点云特征
        source_features = self.source_encoder(source_points)  # [N1, feature_dim]
        target_features = self.target_encoder(target_points)  # [N2, feature_dim]
        
        # 应用跨注意力机制
        attended_source = source_features
        attended_target = target_features
        
        for cross_attn in self.cross_attention_layers:
            if self.use_local_attention:
                # 局部跨注意力需要位置信息
                new_source = cross_attn(attended_source, attended_target, source_points, target_points)
                new_target = cross_attn(attended_target, attended_source, target_points, source_points)
            else:
                # 全局跨注意力
                new_source = cross_attn(attended_source, attended_target)
                new_target = cross_attn(attended_target, attended_source)
            
            # 残差连接
            attended_source = attended_source + new_source
            attended_target = attended_target + new_target
        
        # 特征融合
        # 为了融合，我们需要匹配点的数量，这里使用最近邻匹配
        if attended_source.shape[0] != attended_target.shape[0]:
            attended_target = self._match_points(attended_source, attended_target)
        
        fused_features = torch.cat([attended_source, attended_target], dim=-1)  # [N, 2*feature_dim]
        fused_features = self.feature_fusion(fused_features)  # [N, feature_dim]
        
        # 全局特征聚合
        global_features = fused_features.permute(1, 0).unsqueeze(0)  # [1, feature_dim, N]
        global_features = self.global_pool(global_features).squeeze()  # [feature_dim]
        
        # 预测位姿变化量
        pose_delta = self.pose_head(global_features)  # [6]
        
        
        return pose_delta
    
    def _sample_points(self, points: torch.Tensor, max_points: int) -> torch.Tensor:
        """随机采样点云"""
        if points.shape[0] <= max_points:
            return points
        
        indices = torch.randperm(points.shape[0], device=points.device)[:max_points]
        return points[indices]
    
    def _match_points(self, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """使用最近邻匹配将目标特征匹配到源特征的数量"""
        N_source = source_features.shape[0]
        N_target = target_features.shape[0]
        
        if N_source == N_target:
            return target_features
        
        if N_source < N_target:
            # 从目标特征中采样
            indices = torch.randperm(N_target, device=target_features.device)[:N_source]
            return target_features[indices]
        else:
            # 重复目标特征
            repeat_times = N_source // N_target
            remainder = N_source % N_target
            
            repeated_features = target_features.repeat(repeat_times, 1)
            if remainder > 0:
                additional_features = target_features[:remainder]
                repeated_features = torch.cat([repeated_features, additional_features], dim=0)
            
            return repeated_features
    
    def apply_pose_delta(self, initial_transform: torch.Tensor, pose_delta: torch.Tensor) -> torch.Tensor:
        """
        将预测的位姿变化量应用到初始变换矩阵上
        
        Args:
            initial_transform: [4, 4] 初始变换矩阵
            pose_delta: [6] 位姿变化量 [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
            
        Returns:
            refined_transform: [4, 4] 细化后的变换矩阵
        """
        device = initial_transform.device
        
        # 提取旋转和平移变化量
        rot_delta = pose_delta[:3]  # [3] 轴角表示的旋转
        trans_delta = pose_delta[3:6]  # [3] 平移
        
        # 将轴角转换为旋转矩阵
        angle = torch.norm(rot_delta)
        if angle < 1e-8:
            R_delta = torch.eye(3, device=device)
        else:
            axis = rot_delta / angle
            cos_angle = torch.cos(angle)
            sin_angle = torch.sin(angle)
            
            # 罗德里格斯公式
            K = torch.tensor([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]], 
                [-axis[1], axis[0], 0]
            ], device=device)
            
            R_delta = torch.eye(3, device=device) + sin_angle * K + (1 - cos_angle) * torch.matmul(K, K)
        
        # 构造变化量变换矩阵
        T_delta = torch.eye(4, device=device)
        T_delta[:3, :3] = R_delta
        T_delta[:3, 3] = trans_delta
        
        # 应用变化量到初始变换
        refined_transform = torch.matmul(T_delta, initial_transform)
        
        return refined_transform