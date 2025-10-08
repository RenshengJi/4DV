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

    def _points_to_voxels(self, points: torch.Tensor, features: torch.Tensor):
        """
        将点云转换为体素表示

        Args:
            points: [N, 3] 点云坐标
            features: [N, C] 点特征

        Returns:
            voxel_features: [M, C] 体素特征
            voxel_coords: [M, 4] 体素坐标 (batch_idx, z, y, x)
            num_points_per_voxel: [M] 每个体素的点数
        """
        device = points.device
        N = points.shape[0]

        # 计算体素坐标
        voxel_coords_float = points / self.voxel_size
        voxel_coords = torch.floor(voxel_coords_float).long()

        # 将坐标移到正值范围
        min_coords = voxel_coords.min(dim=0)[0]
        voxel_coords = voxel_coords - min_coords

        # 添加batch维度（全部设为0，因为是单个batch）
        batch_indices = torch.zeros((N, 1), dtype=torch.long, device=device)
        voxel_coords_with_batch = torch.cat([batch_indices, voxel_coords], dim=1)  # [N, 4]

        # 使用哈希表进行体素聚合
        # 创建唯一的体素标识符
        max_coords = voxel_coords.max(dim=0)[0] + 1
        voxel_hash = (voxel_coords[:, 0] * max_coords[1] * max_coords[2] +
                      voxel_coords[:, 1] * max_coords[2] +
                      voxel_coords[:, 2])

        # 找到唯一的体素
        unique_voxel_hash, inverse_indices = torch.unique(voxel_hash, return_inverse=True)
        num_voxels = unique_voxel_hash.shape[0]

        # 聚合特征（平均池化）
        voxel_features = torch.zeros((num_voxels, features.shape[1]), device=device, dtype=features.dtype)
        voxel_coords_unique = torch.zeros((num_voxels, 4), device=device, dtype=torch.int32)
        num_points_per_voxel = torch.zeros(num_voxels, device=device, dtype=torch.long)

        for i in range(N):
            voxel_idx = inverse_indices[i]
            voxel_features[voxel_idx] += features[i]
            voxel_coords_unique[voxel_idx] = voxel_coords_with_batch[i]
            num_points_per_voxel[voxel_idx] += 1

        # 归一化特征
        voxel_features = voxel_features / num_points_per_voxel.unsqueeze(1).float()

        return voxel_features, voxel_coords_unique, inverse_indices

    def forward(self, gaussian_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gaussian_params: [N, 14] Gaussian参数 (xyz, scale, color, quat, opacity)

        Returns:
            refined_params: [N, 14] 细化后的Gaussian参数
        """
        N = gaussian_params.shape[0]
        device = gaussian_params.device

        # 提取位置用于体素化
        positions = gaussian_params[:, :3]  # [N, 3]

        # 1. 输入编码
        point_features = self.input_encoder(gaussian_params)  # [N, feature_dim]

        # 2. 转换为体素表示
        voxel_features, voxel_coords, inverse_indices = self._points_to_voxels(
            positions, point_features
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

        # 7. 输出头
        delta = self.output_head(point_features_out)  # [N, output_dim]

        # 8. 残差连接
        refined_params = gaussian_params[:, :self.output_dim] + delta

        return refined_params

    def apply_deltas(self, gaussian_params: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """
        应用细化增量到Gaussian参数

        Args:
            gaussian_params: [N, 14] 原始Gaussian参数
            deltas: [N, 14] 细化增量

        Returns:
            refined_params: [N, 14] 细化后的Gaussian参数
        """
        return gaussian_params + deltas


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

    def _points_to_sparse_tensor(self, points: torch.Tensor, features: torch.Tensor) -> SparseConvTensor:
        """
        将点云转换为稀疏张量

        Args:
            points: [N, 3] 点云坐标
            features: [N, C] 点特征

        Returns:
            sparse_tensor: SparseConvTensor
        """
        device = points.device
        N = points.shape[0]

        # 计算体素坐标
        voxel_coords_float = points / self.voxel_size
        voxel_coords = torch.floor(voxel_coords_float).long()

        # 将坐标移到正值范围
        min_coords = voxel_coords.min(dim=0)[0]
        voxel_coords = voxel_coords - min_coords

        # 添加batch维度
        batch_indices = torch.zeros((N, 1), dtype=torch.long, device=device)
        voxel_coords_with_batch = torch.cat([batch_indices, voxel_coords], dim=1)

        # 体素聚合
        max_coords = voxel_coords.max(dim=0)[0] + 1
        voxel_hash = (voxel_coords[:, 0] * max_coords[1] * max_coords[2] +
                      voxel_coords[:, 1] * max_coords[2] +
                      voxel_coords[:, 2])

        unique_voxel_hash, inverse_indices = torch.unique(voxel_hash, return_inverse=True)
        num_voxels = unique_voxel_hash.shape[0]

        voxel_features = torch.zeros((num_voxels, features.shape[1]), device=device, dtype=features.dtype)
        voxel_coords_unique = torch.zeros((num_voxels, 4), device=device, dtype=torch.int32)
        num_points_per_voxel = torch.zeros(num_voxels, device=device, dtype=torch.long)

        for i in range(N):
            voxel_idx = inverse_indices[i]
            voxel_features[voxel_idx] += features[i]
            voxel_coords_unique[voxel_idx] = voxel_coords_with_batch[i]
            num_points_per_voxel[voxel_idx] += 1

        voxel_features = voxel_features / num_points_per_voxel.unsqueeze(1).float()

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
        initial_transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            source_points: [N, 3] 源点云坐标
            target_points: [M, 3] 目标点云坐标
            initial_transform: [4, 4] 初始变换矩阵（可选）

        Returns:
            pose_delta: [6] 位姿增量 (rotation_vector[3], translation[3])
        """
        # 下采样（如果点太多）
        if source_points.shape[0] > self.max_points:
            indices = torch.randperm(source_points.shape[0], device=source_points.device)[:self.max_points]
            source_points = source_points[indices]

        if target_points.shape[0] > self.max_points:
            indices = torch.randperm(target_points.shape[0], device=target_points.device)[:self.max_points]
            target_points = target_points[indices]

        # 1. 编码源点云和目标点云
        source_features = self.point_encoder(source_points)  # [N, feature_dim]
        target_features = self.point_encoder(target_points)  # [M, feature_dim]

        # 2. 转换为稀疏张量
        source_sparse = self._points_to_sparse_tensor(source_points, source_features)
        target_sparse = self._points_to_sparse_tensor(target_points, target_features)

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
        if angle < 1e-6:
            rotation_matrix = torch.eye(3, device=device)
        else:
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
