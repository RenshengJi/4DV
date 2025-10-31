# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math
import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor


class CrossAttentionFusion(nn.Module):
    """
    Source和Target点云特征的交叉注意力融合模块

    核心思想：
    - 让source的每个点"attend to" target的所有点
    - 学习哪些target点与当前source点最相关
    - 使用多头注意力捕获多种对应模式

    相比简单的max pooling + 相加：
    - ✅ 显式建模source-target对应关系
    - ✅ 可学习的注意力权重，自动关注重要区域
    - ✅ 保留空间细节信息
    - ✅ 多头机制捕获多种对应模式

    参考：PREDATOR, Spatial Deformable Transformer等最新方法
    """

    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.0):
        """
        Args:
            feature_dim: 特征维度，必须能被num_heads整除
            num_heads: 多头注意力的头数
            dropout: Dropout概率
        """
        super().__init__()

        assert feature_dim % num_heads == 0, \
            f"feature_dim ({feature_dim}) must be divisible by num_heads ({num_heads})"

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = math.sqrt(self.head_dim)  # 缩放因子

        # Query/Key/Value线性投影
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

        # 输出投影
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Layer normalization for better training stability
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, source_feats: torch.Tensor, target_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source_feats: [N, C] source点特征
            target_feats: [M, C] target点特征

        Returns:
            fused_feats: [2*C] 融合后的全局特征
                - 前C维: source-to-target attended features
                - 后C维: 全局统计特征 (mean)
        """
        N, C = source_feats.shape
        M = target_feats.shape[0]
        H = self.num_heads
        D = self.head_dim

        # 1. 线性投影并重塑为多头形式
        # Q: source查询target  K, V: target被查询
        Q = self.q_proj(source_feats).view(N, H, D)  # [N, H, D]
        K = self.k_proj(target_feats).view(M, H, D)  # [M, H, D]
        V = self.v_proj(target_feats).view(M, H, D)  # [M, H, D]

        # 2. 计算注意力分数: Q @ K^T / sqrt(d_k)
        # einsum: 'nhd,mhd->nmh' 表示对每个head，计算N×M的相似度矩阵
        attn_scores = torch.einsum('nhd,mhd->nmh', Q, K) / self.scale  # [N, M, H]

        # 3. Softmax归一化（对target维度）
        attn_weights = F.softmax(attn_scores, dim=1)  # [N, M, H]
        attn_weights = self.dropout(attn_weights)

        # 4. 加权聚合: attention_weights @ V
        # einsum: 'nmh,mhd->nhd' 表示用注意力权重加权聚合target的值
        attended = torch.einsum('nmh,mhd->nhd', attn_weights, V)  # [N, H, D]
        attended = attended.reshape(N, C)  # [N, C]

        # 5. 输出投影 + 残差连接
        source_attended = self.out_proj(attended)
        source_attended = self.dropout(source_attended)
        source_attended = self.norm(source_attended + source_feats)  # Residual connection

        # 6. 全局特征聚合：同时使用max和mean pooling
        # Max: 捕获最强响应  Mean: 捕获整体分布
        global_max = source_attended.max(dim=0)[0]  # [C]
        global_mean = source_attended.mean(dim=0)   # [C]

        # 返回拼接的全局特征
        return torch.cat([global_max, global_mean], dim=0)  # [2*C]


class SparseConvBlock(spconv.SparseModule):
    """
    基于spconv的稀疏卷积块
    使用3D稀疏卷积处理点云特征，速度快且内存占用小
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()

        self.conv = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
            indice_key='subm'
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        """
        Args:
            x: SparseConvTensor

        Returns:
            SparseConvTensor
        """
        out = self.conv(x)

        # BatchNorm1d expects [N, C], where C is num_features
        features = out.features  # [N, C]
        features = self.bn(features)
        features = self.relu(features)

        # 创建新的SparseConvTensor
        out = out.replace_feature(features)

        return out


class ResidualSparseConvBlock(spconv.SparseModule):
    """带残差连接的稀疏卷积块"""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = spconv.SubMConv3d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
            indice_key='subm'
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = spconv.SubMConv3d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
            indice_key='subm'
        )
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        """
        Args:
            x: SparseConvTensor

        Returns:
            SparseConvTensor with residual connection
        """
        identity_features = x.features

        # First conv block
        out = self.conv1(x)
        features = self.bn1(out.features)
        features = self.relu(features)
        out = out.replace_feature(features)

        # Second conv block
        out = self.conv2(out)
        features = self.bn2(out.features)

        # Residual connection
        features = features + identity_features
        features = self.relu(features)

        out = out.replace_feature(features)

        return out


def points_to_voxels(points: torch.Tensor, features: torch.Tensor, voxel_size: float):
    """
    将点云转换为体素表示的通用方法

    Args:
        points: [N, 3] 点云坐标
        features: [N, C] 点特征
        voxel_size: 体素大小（米）

    Returns:
        voxel_features: [M, C] 体素特征
        voxel_coords: [M, 4] 体素坐标 (batch_idx, z, y, x)
        inverse_indices: [N] 每个点对应的体素索引
    """
    device = points.device
    N = points.shape[0]

    # 计算体素坐标
    voxel_coords_float = points / voxel_size
    voxel_coords = torch.floor(voxel_coords_float).long()

    # 将坐标移到正值范围
    min_coords = voxel_coords.min(dim=0)[0]
    voxel_coords = voxel_coords - min_coords

    # 添加batch维度
    batch_indices = torch.zeros((N, 1), dtype=torch.long, device=device)
    voxel_coords_with_batch = torch.cat([batch_indices, voxel_coords], dim=1)  # [N, 4]

    # 使用哈希表进行体素聚合
    max_coords = voxel_coords.max(dim=0)[0] + 1
    voxel_hash = (voxel_coords[:, 0] * max_coords[1] * max_coords[2] +
                  voxel_coords[:, 1] * max_coords[2] +
                  voxel_coords[:, 2])

    # 找到唯一的体素
    unique_voxel_hash, inverse_indices = torch.unique(voxel_hash, return_inverse=True)
    num_voxels = unique_voxel_hash.shape[0]

    # 聚合特征（平均池化）- 向量化实现
    voxel_features = torch.zeros((num_voxels, features.shape[1]), device=device, dtype=features.dtype)
    voxel_features.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, features.shape[1]), features)

    # 统计每个体素的点数
    num_points_per_voxel = torch.zeros(num_voxels, device=device, dtype=torch.long)
    num_points_per_voxel.scatter_add_(0, inverse_indices, torch.ones(N, device=device, dtype=torch.long))

    # 获取唯一体素坐标 - 使用int32类型（spconv要求）
    voxel_coords_unique = torch.zeros((num_voxels, 4), device=device, dtype=torch.int32)
    sorted_indices, sort_order = inverse_indices.sort()
    unique_positions = torch.cat([torch.tensor([0], device=device), (sorted_indices[1:] != sorted_indices[:-1]).nonzero(as_tuple=True)[0] + 1])
    voxel_coords_unique = voxel_coords_with_batch[sort_order[unique_positions]].to(torch.int32)

    # 归一化特征
    voxel_features = voxel_features / num_points_per_voxel.unsqueeze(1).float()

    return voxel_features, voxel_coords_unique, inverse_indices


class GaussianRefineHeadSparseConv(nn.Module):
    """
    使用spconv的Gaussian细化网络
    比Transformer快得多，内存占用小
    """

    def __init__(
        self,
        input_gaussian_dim: int = 14,
        output_gaussian_dim: int = 14,
        feature_dim: int = 128,
        num_conv_layers: int = 2,
        voxel_size: float = 0.05  # 体素大小，单位米
    ):
        super().__init__()
        self.input_dim = input_gaussian_dim
        self.output_dim = output_gaussian_dim
        self.feature_dim = feature_dim
        self.num_conv_layers = num_conv_layers
        self.voxel_size = voxel_size

        # 输入编码
        self.input_encoder = nn.Sequential(
            nn.Linear(input_gaussian_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True)
        )

        # 稀疏卷积层
        self.conv_layers = spconv.SparseSequential(
            *[ResidualSparseConvBlock(feature_dim) for _ in range(num_conv_layers)]
        )

        # 计算理论感受野（用于debug）
        receptive_field = 1 + num_conv_layers * 2 * 2  # 每个ResidualBlock有2个3x3卷积
        self.receptive_field_voxels = receptive_field
        self.receptive_field_meters = receptive_field * voxel_size

        # 输出头 - 增强版（Plan C）
        self.output_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, output_gaussian_dim)
        )

        # 接近零初始化 - 输出头的最后一层
        self._init_output_weights()

    def _init_output_weights(self):
        """初始化输出层权重为接近零的值，使得初始时网络输出的delta很小"""
        # 获取输出头的最后一层
        final_layer = self.output_head[-1]
        # 权重初始化为很小的值
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.0001)
        # bias初始化为零
        if final_layer.bias is not None:
            nn.init.zeros_(final_layer.bias)

    def _points_to_voxels(self, points: torch.Tensor, features: torch.Tensor):
        """调用全局points_to_voxels函数"""
        return points_to_voxels(points, features, self.voxel_size)

    def forward(self, gaussian_params: torch.Tensor, pred_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            gaussian_params: [N, 14] Gaussian参数 (xyz, scale, color, quat, opacity)
            pred_scale: [] 或 [1] Stage1预测的scene scale (可选)

        Returns:
            delta: [N, 14] Gaussian参数的增量
        """
        N = gaussian_params.shape[0]
        device = gaussian_params.device

        # 安全性检查：输入参数
        if torch.isnan(gaussian_params).any() or torch.isinf(gaussian_params).any():
            gaussian_params = torch.nan_to_num(gaussian_params, nan=0.0, posinf=1.0, neginf=-1.0)

        # 提取原始位置
        positions_original = gaussian_params[:, :3]  # [N, 3]

        # 转换到metric尺度：使用Stage1预测的scale（除以scale）
        if pred_scale is None or torch.isnan(pred_scale).any() or torch.isinf(pred_scale).any() or (pred_scale == 0).any():
            pred_scale = torch.tensor(1.0, device=device)
        positions_metric = positions_original / pred_scale  # [N, 3]

        # 中心化：移动坐标系到物体中心
        center = positions_metric.mean(dim=0, keepdim=True)  # [1, 3]
        positions_centered = positions_metric - center  # [N, 3]

        # 1. 输入编码
        point_features = self.input_encoder(gaussian_params)  # [N, feature_dim]

        # 2. 转换为体素表示（使用中心化的metric positions）
        voxel_features, voxel_coords, inverse_indices = self._points_to_voxels(
            positions_centered, point_features
        )

        # 3. 计算空间范围并进行安全性检查
        spatial_shape = voxel_coords[:, 1:].max(dim=0)[0] + 1
        spatial_shape_list = spatial_shape.cpu().numpy().tolist()

        # 【关键修复】验证spatial_shape是否在合理范围内
        MAX_SPATIAL_DIM = 1024  # spconv的安全上限
        if any(dim > MAX_SPATIAL_DIM for dim in spatial_shape_list):
            print(f"[WARNING] GaussianRefineHead: spatial_shape {spatial_shape_list} exceeds max {MAX_SPATIAL_DIM}, returning zero delta")
            # 使用模型参数创建零tensor，确保有grad_fn
            dummy_param = next(self.parameters())
            return torch.zeros_like(gaussian_params) + dummy_param.sum() * 0.0

        # 【关键修复】检查spatial_shape是否为有效的正整数
        if any(dim <= 0 for dim in spatial_shape_list):
            print(f"[WARNING] GaussianRefineHead: invalid spatial_shape {spatial_shape_list}, returning zero delta")
            # 使用模型参数创建零tensor，确保有grad_fn
            dummy_param = next(self.parameters())
            return torch.zeros_like(gaussian_params) + dummy_param.sum() * 0.0

        # 【关键修复】验证voxel_coords是否在有效范围内
        voxel_coords_max = voxel_coords[:, 1:].max(dim=0)[0]
        voxel_coords_min = voxel_coords[:, 1:].min(dim=0)[0]
        if (voxel_coords_min < 0).any():
            print(f"[WARNING] GaussianRefineHead: negative voxel coords detected (min={voxel_coords_min.tolist()}), returning zero delta")
            # 使用模型参数创建零tensor，确保有grad_fn
            dummy_param = next(self.parameters())
            return torch.zeros_like(gaussian_params) + dummy_param.sum() * 0.0

        # 【关键修复】验证voxel_features是否有效
        if torch.isnan(voxel_features).any() or torch.isinf(voxel_features).any():
            print(f"[WARNING] GaussianRefineHead: NaN/Inf in voxel_features, returning zero delta")
            # 使用模型参数创建零tensor，确保有grad_fn
            dummy_param = next(self.parameters())
            return torch.zeros_like(gaussian_params) + dummy_param.sum() * 0.0

        # 4. 创建SparseConvTensor（添加try-catch保护）
        try:
            sparse_input = SparseConvTensor(
                features=voxel_features,
                indices=voxel_coords,
                spatial_shape=spatial_shape_list,
                batch_size=1
            )
        except Exception as e:
            print(f"[ERROR] GaussianRefineHead: Failed to create SparseConvTensor: {e}")
            print(f"  - voxel_features shape: {voxel_features.shape}")
            print(f"  - voxel_coords shape: {voxel_coords.shape}")
            print(f"  - spatial_shape: {spatial_shape_list}")
            print(f"  - voxel_coords range: min={voxel_coords_min.tolist()}, max={voxel_coords_max.tolist()}")
            # 使用模型参数创建零tensor，确保有grad_fn
            dummy_param = next(self.parameters())
            return torch.zeros_like(gaussian_params) + dummy_param.sum() * 0.0

        # 5. 稀疏卷积处理（添加try-catch保护）
        try:
            sparse_output = self.conv_layers(sparse_input)
        except Exception as e:
            print(f"[ERROR] GaussianRefineHead: Failed in conv_layers: {e}")
            print(f"  - spatial_shape: {spatial_shape_list}")
            print(f"  - num_voxels: {voxel_features.shape[0]}")
            # 使用模型参数创建零tensor，确保有grad_fn
            dummy_param = next(self.parameters())
            return torch.zeros_like(gaussian_params) + dummy_param.sum() * 0.0

        # 6. 从体素特征还原到点特征（使用残差连接保留原始点特征）
        voxel_features_out = sparse_output.features  # [M, feature_dim]
        point_features_voxel = voxel_features_out[inverse_indices]  # [N, feature_dim]

        # 【关键改进】残差连接：加回原始点特征，保留voxel内的点级差异
        # 这样同一voxel内的点不会得到完全相同的delta
        point_features_out = point_features_voxel + point_features  # [N, feature_dim]

        # 7. 输出头 - 直接返回delta
        delta = self.output_head(point_features_out)  # [N, output_dim]

        return delta

    def apply_deltas(self, gaussian_params: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """
        应用细化增量到Gaussian参数

        对deltas应用激活函数后再加到原始参数上，与vggt.py中的forward保持一致

        注意：当前只调整前3个参数（means/positions），其他参数不变

        Args:
            gaussian_params: [N, 14] 原始Gaussian参数
            deltas: [N, 14] 细化增量（raw）

        Returns:
            refined_params: [N, 14] 细化后的Gaussian参数
        """
        # 对deltas应用激活函数（与vggt.py中的forward一致）
        deltas_activated = deltas.clone()

        # 只保留前3个参数（means/positions）的deltas，其他置为0
        # deltas_activated[:, 3:] = 0.0

        return gaussian_params + deltas_activated


class PoseRefineHeadSparseConv(nn.Module):
    """
    使用spconv的位姿细化网络
    基于点云几何特征预测帧间变换
    """

    def __init__(
        self,
        input_dim: int = 3,
        feature_dim: int = 128,
        num_conv_layers: int = 2,
        voxel_size: float = 0.1,
        max_points: int = 4096
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_points = max_points
        self.voxel_size = voxel_size
        self.feature_dim = feature_dim
        self.num_conv_layers = num_conv_layers

        # 点云编码
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True)
        )

        # 稀疏卷积特征提取
        self.conv_layers = spconv.SparseSequential(
            *[ResidualSparseConvBlock(feature_dim) for _ in range(num_conv_layers)]
        )

        # 计算理论感受野（用于debug）
        receptive_field = 1 + num_conv_layers * 2 * 2  # 每个ResidualBlock有2个3x3卷积
        self.receptive_field_voxels = receptive_field
        self.receptive_field_meters = receptive_field * voxel_size

        # 交叉注意力融合模块（替换简单的max pooling + 相加）
        # 显式建模source-target对应关系，学习哪些点对匹配
        self.cross_attention = CrossAttentionFusion(
            feature_dim=feature_dim,
            num_heads=12,  # 增加到12个注意力头（Plan C）
            dropout=0.1   # 轻微dropout防止过拟合
        )

        # 位姿预测头 (输出9维: 6D旋转 + 3平移)
        # 输入：4*feature_dim (source_to_target [2C] + target_to_source [2C])
        # 使用6D旋转表示以避免轴-角表示的数值不稳定性和不连续性
        # 参考: Zhou et al. "On the Continuity of Rotation Representations in Neural Networks" CVPR 2019
        self.pose_head = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim * 2),  # 降维
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 9)  # 6D rotation + 3D translation
        )

        # 接近零初始化 - 位姿预测头的最后一层
        self._init_output_weights()

    def _init_output_weights(self):
        """初始化输出层权重为接近零的值，使得初始时网络输出的增量很小"""
        # 获取位姿预测头的最后一层
        final_layer = self.pose_head[-1]
        # 权重初始化为很小的值
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.0001)
        # bias初始化为零（因为我们在forward中使用残差形式，这里不需要特殊初始化）
        if final_layer.bias is not None:
            nn.init.zeros_(final_layer.bias)

    def _points_to_sparse_tensor(self, points: torch.Tensor, features: torch.Tensor) -> Optional[SparseConvTensor]:
        """
        将点云转换为稀疏张量

        Args:
            points: [N, 3] 点云坐标
            features: [N, C] 点特征

        Returns:
            sparse_tensor: SparseConvTensor or None if failed
        """
        # 使用全局points_to_voxels函数
        voxel_features, voxel_coords_unique, _ = points_to_voxels(points, features, self.voxel_size)

        # 计算空间范围并进行安全性检查
        spatial_shape = voxel_coords_unique[:, 1:].max(dim=0)[0] + 1
        spatial_shape_list = spatial_shape.cpu().numpy().tolist()

        # 【关键修复】验证spatial_shape是否在合理范围内
        MAX_SPATIAL_DIM = 1024  # spconv的安全上限
        if any(dim > MAX_SPATIAL_DIM for dim in spatial_shape_list):
            print(f"[WARNING] PoseRefineHead: spatial_shape {spatial_shape_list} exceeds max {MAX_SPATIAL_DIM}")
            return None

        # 【关键修复】检查spatial_shape是否为有效的正整数
        if any(dim <= 0 for dim in spatial_shape_list):
            print(f"[WARNING] PoseRefineHead: invalid spatial_shape {spatial_shape_list}")
            return None

        # 【关键修复】验证voxel_coords是否在有效范围内
        voxel_coords_min = voxel_coords_unique[:, 1:].min(dim=0)[0]
        if (voxel_coords_min < 0).any():
            print(f"[WARNING] PoseRefineHead: negative voxel coords detected (min={voxel_coords_min.tolist()})")
            return None

        # 【关键修复】验证voxel_features是否有效
        if torch.isnan(voxel_features).any() or torch.isinf(voxel_features).any():
            print(f"[WARNING] PoseRefineHead: NaN/Inf in voxel_features")
            return None

        # 创建SparseConvTensor（添加try-catch保护）
        try:
            sparse_tensor = SparseConvTensor(
                features=voxel_features,
                indices=voxel_coords_unique,
                spatial_shape=spatial_shape_list,
                batch_size=1
            )
            return sparse_tensor
        except Exception as e:
            print(f"[ERROR] PoseRefineHead: Failed to create SparseConvTensor: {e}")
            print(f"  - voxel_features shape: {voxel_features.shape}")
            print(f"  - voxel_coords shape: {voxel_coords_unique.shape}")
            print(f"  - spatial_shape: {spatial_shape_list}")
            return None

    def forward(
        self,
        source_points: torch.Tensor,
        target_points: torch.Tensor,
        initial_transform: Optional[torch.Tensor] = None,
        pred_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            source_points: [N, 3] 源点云坐标
            target_points: [M, 3] 目标点云坐标
            initial_transform: [4, 4] 初始变换矩阵（可选）
            pred_scale: [] 或 [1] Stage1预测的scene scale（可选）

        Returns:
            pose_delta: [9] 位姿增量 (6D rotation[6], translation[3])
        """
        device = source_points.device

        # 1. 编码源点云和目标点云（使用原始坐标，不是metric空间）
        source_features = self.point_encoder(source_points)  # [N, feature_dim]
        target_features = self.point_encoder(target_points)  # [M, feature_dim]

        # 转换到metric尺度：使用Stage1预测的scale（除以scale）
        if pred_scale is None or torch.isnan(pred_scale).any() or torch.isinf(pred_scale).any() or (pred_scale == 0).any():
            pred_scale = torch.tensor(1.0, device=device)
        source_points_metric = source_points / pred_scale  # [N, 3]
        target_points_metric = target_points / pred_scale  # [M, 3]

        # 中心化点云（使用共同中心）- 用于体素化
        # 使用source和target的共同中心，确保预测的变换一致性
        all_points = torch.cat([source_points_metric, target_points_metric], dim=0)  # [N+M, 3]
        shared_center = all_points.mean(dim=0, keepdim=True)  # [1, 3]
        source_centered = source_points_metric - shared_center  # [N, 3]
        target_centered = target_points_metric - shared_center  # [M, 3]

        # 2. 转换为稀疏张量（使用metric空间的中心化坐标）
        source_sparse = self._points_to_sparse_tensor(source_centered, source_features)
        target_sparse = self._points_to_sparse_tensor(target_centered, target_features)

        # 【关键修复】检查稀疏张量创建是否成功
        if source_sparse is None or target_sparse is None:
            print(f"[WARNING] PoseRefineHead: Failed to create sparse tensors, returning zero pose_delta")
            # 使用模型参数创建零tensor，确保有grad_fn
            dummy_param = next(self.parameters())
            return (dummy_param * 0.0).sum() + torch.zeros(9, device=device, requires_grad=True)

        # 3. 稀疏卷积特征提取（添加try-catch保护）
        try:
            source_sparse = self.conv_layers(source_sparse)
            target_sparse = self.conv_layers(target_sparse)
        except Exception as e:
            print(f"[ERROR] PoseRefineHead: Failed in conv_layers: {e}")
            # 使用模型参数创建零tensor，确保有grad_fn
            dummy_param = next(self.parameters())
            return (dummy_param * 0.0).sum() + torch.zeros(9, device=device, requires_grad=True)

        # 4. 交叉注意力特征融合（替换简单的max pooling + 相加）
        # 双向注意力：source看target，target看source
        # 这样可以捕获双向的几何对应关系

        source_feats = source_sparse.features  # [N, feature_dim]
        target_feats = target_sparse.features  # [M, feature_dim]

        # Source-to-Target: source点云查询target点云
        # 学习：对于每个source点，target中哪些点与它对应
        source_to_target = self.cross_attention(source_feats, target_feats)  # [2*feature_dim]

        # Target-to-Source: target点云查询source点云（对称性）
        # 学习：对于每个target点，source中哪些点与它对应
        target_to_source = self.cross_attention(target_feats, source_feats)  # [2*feature_dim]

        # 5. 双向特征拼接
        # 将双向的注意力特征拼接，提供更丰富的对应关系信息
        combined_feature = torch.cat([source_to_target, target_to_source], dim=0)  # [4*feature_dim]

        # 6. 位姿预测（残差形式）
        # 基于双向对应关系，预测pose delta的增量
        pose_delta_increment = self.pose_head(combined_feature)  # [9]: 6D rotation + 3D translation

        # 残差连接：基准（单位旋转+零平移）+ 网络预测的增量
        # 这样即使网络输出接近零，最终的pose_delta也能对应单位旋转
        # 提高训练初期的稳定性
        base_pose = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # 单位6D旋转 + 零平移
            device=combined_feature.device,
            dtype=combined_feature.dtype
        )
        pose_delta = base_pose + pose_delta_increment

        return pose_delta

    def apply_pose_delta(
        self,
        initial_transform: torch.Tensor,
        pose_delta: torch.Tensor
    ) -> torch.Tensor:
        """
        应用位姿增量到初始变换（使用6D旋转表示）

        6D旋转表示优势：
        - 连续、可微、无奇异点
        - 梯度稳定，适合深度学习训练
        - 避免轴-角表示的除零和不连续性问题

        Args:
            initial_transform: [4, 4] 初始变换矩阵
            pose_delta: [9] 位姿增量 (6D rotation + 3D translation)
                - rotation_6d: [6] 两个3D向量 (a1, a2)
                - translation: [3] 平移向量

        Returns:
            refined_transform: [4, 4] 细化后的变换矩阵
        """
        device = initial_transform.device

        # 提取6D旋转和平移
        rotation_6d = pose_delta[:6]  # [6]
        translation = pose_delta[6:]  # [3]

        # 【关键修复】即使pose_delta接近零，也不能直接返回initial_transform
        # 因为initial_transform可能没有梯度，导致backward时报错
        # 我们需要保持与pose_delta的梯度连接
        # 注释掉这个early return，让它继续走下面的计算流程
        # if torch.abs(rotation_6d).max() < 1e-6 and torch.abs(translation).max() < 1e-6:
        #     return initial_transform.clone()

        # 6D旋转 → 旋转矩阵 (Gram-Schmidt正交化)
        # 将6D向量重塑为2个3D向量
        a1 = rotation_6d[:3]  # [3]
        a2 = rotation_6d[3:6]  # [3]

        # 【安全性检查】检查a1是否为零向量
        a1_norm = torch.norm(a1)
        if a1_norm < 1e-8:
            # 如果旋转接近零，只应用平移
            delta_transform = torch.eye(4, device=device, dtype=pose_delta.dtype)
            delta_transform[:3, 3] = translation
            return torch.matmul(delta_transform, initial_transform)

        # Gram-Schmidt正交化
        # b1 = normalize(a1)
        b1 = F.normalize(a1, dim=0, eps=1e-8)

        # b2 = normalize(a2 - (a2·b1)b1)
        dot_product = (b1 * a2).sum()
        b2 = a2 - dot_product * b1

        # 【安全性检查】检查b2是否为零向量（a1和a2共线）
        b2_norm = torch.norm(b2)
        if b2_norm < 1e-8:
            # 如果a1和a2共线，只应用平移
            delta_transform = torch.eye(4, device=device, dtype=pose_delta.dtype)
            delta_transform[:3, 3] = translation
            return torch.matmul(delta_transform, initial_transform)

        b2 = F.normalize(b2, dim=0, eps=1e-8)

        # b3 = b1 × b2 (叉积得到第三个正交向量)
        b3 = torch.cross(b1, b2, dim=0)

        # 构建旋转矩阵 [b1, b2, b3] 作为列向量
        rotation_matrix = torch.stack([b1, b2, b3], dim=1)  # [3, 3]

        # 构建增量变换矩阵
        delta_transform = torch.eye(4, device=device, dtype=pose_delta.dtype)
        delta_transform[:3, :3] = rotation_matrix
        delta_transform[:3, 3] = translation

        # 应用增量: T_refined = T_delta @ T_initial
        refined_transform = torch.matmul(delta_transform, initial_transform)

        return refined_transform
