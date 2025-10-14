# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor


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

        self.conv1 = spconv.SubMConv3d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
            indice_key='subm'
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = spconv.SubMConv3d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
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

    # 获取唯一体素坐标 - 使用scatter，确保int32类型
    voxel_coords_unique = torch.zeros((num_voxels, 4), device=device, dtype=torch.int32)
    # 为每个唯一体素找到第一个对应的点
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
        voxel_size: float = 0.05,  # 体素大小，单位米
        max_num_points_per_voxel: int = 5
    ):
        super().__init__()
        self.input_dim = input_gaussian_dim
        self.output_dim = output_gaussian_dim
        self.feature_dim = feature_dim
        self.voxel_size = voxel_size
        self.max_num_points_per_voxel = max_num_points_per_voxel

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

        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
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

        # 3. 计算空间范围
        spatial_shape = voxel_coords[:, 1:].max(dim=0)[0] + 1
        spatial_shape = spatial_shape.cpu().numpy().tolist()

        # 4. 创建SparseConvTensor
        sparse_input = SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords,
            spatial_shape=spatial_shape,
            batch_size=1
        )

        # 5. 稀疏卷积处理
        sparse_output = self.conv_layers(sparse_input)

        # 6. 从体素特征还原到点特征
        voxel_features_out = sparse_output.features  # [M, feature_dim]
        point_features_out = voxel_features_out[inverse_indices]  # [N, feature_dim]

        # 7. 输出头 - 直接返回delta
        delta = self.output_head(point_features_out)  # [N, output_dim]

        return delta

    def apply_deltas(self, gaussian_params: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """
        应用细化增量到Gaussian参数

        对deltas应用激活函数后再加到原始参数上，与vggt.py中的forward保持一致

        Args:
            gaussian_params: [N, 14] 原始Gaussian参数
            deltas: [N, 14] 细化增量（raw）

        Returns:
            refined_params: [N, 14] 细化后的Gaussian参数
        """
        # 对deltas应用激活函数（与vggt.py中的forward一致）
        deltas_activated = deltas.clone()

        # # Scale activation: 0.05 * exp(scale), clamped to max 0.3
        # deltas_activated[..., 3:6] = (0.05 * torch.exp(deltas[..., 3:6])).clamp_max(0.3)

        # # Rotation normalization
        # rotations = deltas[..., 9:13]
        # rotation_norms = torch.norm(rotations, dim=-1, keepdim=True).clamp(min=1e-8)
        # deltas_activated[..., 9:13] = rotations / rotation_norms

        # # Opacity activation: sigmoid
        # deltas_activated[..., 13:14] = deltas[..., 13:14].sigmoid()

        # 应用激活后的deltas
        return gaussian_params  + deltas_activated


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
        self.max_points = max_points
        self.voxel_size = voxel_size
        self.feature_dim = feature_dim

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

        # 位姿预测头 (输出6维: 3旋转 + 3平移)
        self.pose_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 6)
        )

        # 接近零初始化 - 位姿预测头的最后一层
        self._init_output_weights()

    def _init_output_weights(self):
        """初始化输出层权重为接近零的值，使得初始时网络输出的delta很小"""
        # 获取位姿预测头的最后一层
        final_layer = self.pose_head[-1]
        # 权重初始化为很小的值
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.0001)
        # bias初始化为零
        if final_layer.bias is not None:
            nn.init.zeros_(final_layer.bias)

    def _points_to_sparse_tensor(self, points: torch.Tensor, features: torch.Tensor) -> SparseConvTensor:
        """
        将点云转换为稀疏张量

        Args:
            points: [N, 3] 点云坐标
            features: [N, C] 点特征

        Returns:
            sparse_tensor: SparseConvTensor
        """
        # 使用全局points_to_voxels函数
        voxel_features, voxel_coords_unique, _ = points_to_voxels(points, features, self.voxel_size)

        # 计算空间范围
        spatial_shape = voxel_coords_unique[:, 1:].max(dim=0)[0] + 1
        spatial_shape = spatial_shape.cpu().numpy().tolist()

        # 创建SparseConvTensor
        sparse_tensor = SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords_unique,
            spatial_shape=spatial_shape,
            batch_size=1
        )

        return sparse_tensor

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
            pose_delta: [6] 位姿增量 (rotation_vector[3], translation[3])
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

        # 中心化点云（移动坐标系到各自中心）- 用于体素化
        source_center = source_points_metric.mean(dim=0, keepdim=True)  # [1, 3]
        target_center = target_points_metric.mean(dim=0, keepdim=True)  # [1, 3]
        source_centered = source_points_metric - source_center  # [N, 3]
        target_centered = target_points_metric - target_center  # [M, 3]

        # 2. 转换为稀疏张量（使用metric空间的中心化坐标）
        source_sparse = self._points_to_sparse_tensor(source_centered, source_features)
        target_sparse = self._points_to_sparse_tensor(target_centered, target_features)

        # 3. 稀疏卷积特征提取
        source_sparse = self.conv_layers(source_sparse)
        target_sparse = self.conv_layers(target_sparse)

        # 4. 全局特征聚合
        source_global = source_sparse.features.max(dim=0)[0]  # [feature_dim]
        target_global = target_sparse.features.max(dim=0)[0]  # [feature_dim]

        # 5. 特征融合
        combined_feature = source_global + target_global  # [feature_dim]

        # 6. 位姿预测
        pose_delta = self.pose_head(combined_feature)  # [6]

        return pose_delta

    def apply_pose_delta(
        self,
        initial_transform: torch.Tensor,
        pose_delta: torch.Tensor
    ) -> torch.Tensor:
        """
        应用位姿增量到初始变换

        Args:
            initial_transform: [4, 4] 初始变换矩阵
            pose_delta: [6] 位姿增量 (rotation_vector[3], translation[3])

        Returns:
            refined_transform: [4, 4] 细化后的变换矩阵
        """
        device = initial_transform.device

        # 提取旋转和平移
        rotation_vec = pose_delta[:3]  # [3]
        translation = pose_delta[3:]  # [3]

        # 旋转向量转旋转矩阵 (Rodrigues公式)
        angle = torch.norm(rotation_vec)
        # if angle < 1e-6:
        #     rotation_matrix = torch.eye(3, device=device)
        # else:
        axis = rotation_vec / angle
        K = torch.zeros((3, 3), device=device)
        K[0, 1] = -axis[2]
        K[0, 2] = axis[1]
        K[1, 0] = axis[2]
        K[1, 2] = -axis[0]
        K[2, 0] = -axis[1]
        K[2, 1] = axis[0]

        rotation_matrix = (
            torch.eye(3, device=device) +
            torch.sin(angle) * K +
            (1 - torch.cos(angle)) * torch.matmul(K, K)
        )

        # 构建增量变换矩阵
        delta_transform = torch.eye(4, device=device)
        delta_transform[:3, :3] = rotation_matrix
        delta_transform[:3, 3] = translation

        # 应用增量
        refined_transform = torch.matmul(delta_transform, initial_transform)

        return refined_transform
