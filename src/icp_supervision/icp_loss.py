"""
ICP Supervision Loss - 用ICP GT进行强监督的训练损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ICPSupervisionLoss(nn.Module):
    """
    ICP监督损失

    核心思想:
    - 使用ICP配准结果作为ground truth
    - 直接监督Gaussian参数的精细化
    - 支持分项监督: positions, scales, rotations, colors, opacity

    损失组成:
    1. Position Loss: MSE loss on xyz positions (前3个参数)
    2. Scale Loss: MSE loss on scales (参数3-6)
    3. Rotation Loss: quaternion distance (参数9-13)
    4. Color Loss: MSE loss on RGB (参数6-9)
    5. Opacity Loss: MSE loss on opacity (参数13-14)
    """

    def __init__(
        self,
        position_weight: float = 10.0,
        scale_weight: float = 1.0,
        rotation_weight: float = 1.0,
        color_weight: float = 1.0,
        opacity_weight: float = 1.0,
        use_smooth_l1: bool = False,
        position_only: bool = False,
    ):
        """
        Args:
            position_weight: 位置损失权重
            scale_weight: 尺度损失权重
            rotation_weight: 旋转损失权重
            color_weight: 颜色损失权重
            opacity_weight: 透明度损失权重
            use_smooth_l1: 是否使用SmoothL1 loss (更鲁棒)
            position_only: 是否只监督位置 (其他参数不计算loss)
        """
        super().__init__()

        self.position_weight = position_weight
        self.scale_weight = scale_weight
        self.rotation_weight = rotation_weight
        self.color_weight = color_weight
        self.opacity_weight = opacity_weight
        self.use_smooth_l1 = use_smooth_l1
        self.position_only = position_only

        # 损失函数
        if use_smooth_l1:
            self.criterion = nn.SmoothL1Loss(reduction='mean')
        else:
            self.criterion = nn.MSELoss(reduction='mean')

    def forward(
        self,
        pred_gaussians: torch.Tensor,
        target_gaussians: torch.Tensor,
        return_individual_losses: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        计算ICP监督损失

        Args:
            pred_gaussians: [N, 14] 预测的refined Gaussian参数
            target_gaussians: [N, 14] ICP GT Gaussian参数
            return_individual_losses: 是否返回各项损失的详细信息

        Returns:
            (total_loss, loss_dict)
                - total_loss: 总损失
                - loss_dict: 各项损失的字典 (如果return_individual_losses=True)
        """
        loss_dict = {}

        # 1. Position Loss (xyz) - 最重要的损失
        position_pred = pred_gaussians[:, :3]
        position_target = target_gaussians[:, :3]
        position_loss = self.criterion(position_pred, position_target)
        loss_dict['position_loss'] = position_loss
        total_loss = self.position_weight * position_loss

        if not self.position_only:
            # 2. Scale Loss (3:6)
            scale_pred = pred_gaussians[:, 3:6]
            scale_target = target_gaussians[:, 3:6]
            scale_loss = self.criterion(scale_pred, scale_target)
            loss_dict['scale_loss'] = scale_loss
            total_loss += self.scale_weight * scale_loss

            # 3. Color Loss (6:9)
            color_pred = pred_gaussians[:, 6:9]
            color_target = target_gaussians[:, 6:9]
            color_loss = self.criterion(color_pred, color_target)
            loss_dict['color_loss'] = color_loss
            total_loss += self.color_weight * color_loss

            # 4. Rotation Loss (9:13) - 四元数距离
            quat_pred = pred_gaussians[:, 9:13]
            quat_target = target_gaussians[:, 9:13]
            rotation_loss = self.quaternion_distance(quat_pred, quat_target)
            loss_dict['rotation_loss'] = rotation_loss
            total_loss += self.rotation_weight * rotation_loss

            # 5. Opacity Loss (13:14)
            opacity_pred = pred_gaussians[:, 13:14]
            opacity_target = target_gaussians[:, 13:14]
            opacity_loss = self.criterion(opacity_pred, opacity_target)
            loss_dict['opacity_loss'] = opacity_loss
            total_loss += self.opacity_weight * opacity_loss

        # 记录总损失
        loss_dict['total_loss'] = total_loss

        if return_individual_losses:
            return total_loss, loss_dict
        else:
            return total_loss, None

    def quaternion_distance(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        计算两个四元数之间的距离

        使用内积距离: d = 1 - |<q1, q2>|
        范围: [0, 1], 0表示完全相同

        Args:
            q1, q2: [N, 4] 四元数 (w, x, y, z)

        Returns:
            distance: 标量距离
        """
        # 归一化四元数
        q1_norm = F.normalize(q1, p=2, dim=-1)
        q2_norm = F.normalize(q2, p=2, dim=-1)

        # 计算内积
        dot_product = torch.sum(q1_norm * q2_norm, dim=-1)

        # 取绝对值 (因为q和-q表示同一个旋转)
        dot_product = torch.abs(dot_product)

        # 距离: 1 - |<q1, q2>|
        distance = 1.0 - dot_product

        # 平均
        return distance.mean()


class ICPChamferLoss(nn.Module):
    """
    基于Chamfer距离的ICP损失 (备选方案)

    直接在点云空间计算Chamfer距离
    可能比参数空间的MSE更直观
    """

    def __init__(self, use_bidirectional: bool = True):
        """
        Args:
            use_bidirectional: 是否使用双向Chamfer距离
        """
        super().__init__()
        self.use_bidirectional = use_bidirectional

    def forward(
        self,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Chamfer距离

        Args:
            pred_positions: [N, 3] 预测的点位置
            target_positions: [M, 3] 目标点位置

        Returns:
            chamfer_loss: Chamfer距离损失
        """
        # pred -> target
        dist_pred_to_target = self._compute_nearest_neighbor_distance(
            pred_positions, target_positions
        )

        if self.use_bidirectional:
            # target -> pred
            dist_target_to_pred = self._compute_nearest_neighbor_distance(
                target_positions, pred_positions
            )
            chamfer_loss = (dist_pred_to_target + dist_target_to_pred) / 2.0
        else:
            chamfer_loss = dist_pred_to_target

        return chamfer_loss

    def _compute_nearest_neighbor_distance(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算source到target的平均最近邻距离

        Args:
            source: [N, 3]
            target: [M, 3]

        Returns:
            avg_distance: 平均距离
        """
        # 计算距离矩阵 [N, M]
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
        source_norm = (source ** 2).sum(dim=1, keepdim=True)  # [N, 1]
        target_norm = (target ** 2).sum(dim=1, keepdim=True)  # [M, 1]

        dist_matrix = source_norm + target_norm.T - 2.0 * torch.mm(source, target.T)  # [N, M]

        # 找到每个source点的最近邻距离
        min_distances, _ = torch.min(dist_matrix, dim=1)  # [N]

        # 平均
        avg_distance = min_distances.mean()

        return avg_distance


def test_icp_loss():
    """测试ICP损失函数"""
    print("Testing ICP Supervision Loss...")

    # 创建测试数据
    N = 100
    pred_gaussians = torch.randn(N, 14)
    target_gaussians = torch.randn(N, 14)

    # 归一化四元数
    pred_gaussians[:, 9:13] = F.normalize(pred_gaussians[:, 9:13], p=2, dim=-1)
    target_gaussians[:, 9:13] = F.normalize(target_gaussians[:, 9:13], p=2, dim=-1)

    # 颜色和opacity归一化到[0, 1]
    pred_gaussians[:, 6:9] = torch.sigmoid(pred_gaussians[:, 6:9])
    target_gaussians[:, 6:9] = torch.sigmoid(target_gaussians[:, 6:9])
    pred_gaussians[:, 13:14] = torch.sigmoid(pred_gaussians[:, 13:14])
    target_gaussians[:, 13:14] = torch.sigmoid(target_gaussians[:, 13:14])

    # 测试完整损失
    loss_fn = ICPSupervisionLoss()
    total_loss, loss_dict = loss_fn(pred_gaussians, target_gaussians, return_individual_losses=True)

    print("\nFull supervision:")
    print(f"  Total loss: {total_loss.item():.6f}")
    for key, value in loss_dict.items():
        if key != 'total_loss':
            print(f"  {key}: {value.item():.6f}")

    # 测试只监督位置
    loss_fn_pos_only = ICPSupervisionLoss(position_only=True)
    total_loss_pos, loss_dict_pos = loss_fn_pos_only(
        pred_gaussians, target_gaussians, return_individual_losses=True
    )

    print("\nPosition-only supervision:")
    print(f"  Total loss: {total_loss_pos.item():.6f}")
    for key, value in loss_dict_pos.items():
        print(f"  {key}: {value.item():.6f}")

    # 测试Chamfer损失
    chamfer_loss_fn = ICPChamferLoss()
    pred_positions = pred_gaussians[:, :3]
    target_positions = target_gaussians[:, :3]
    chamfer_loss = chamfer_loss_fn(pred_positions, target_positions)

    print("\nChamfer distance:")
    print(f"  Chamfer loss: {chamfer_loss.item():.6f}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_icp_loss()
