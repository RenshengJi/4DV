"""
Utility functions for ICP supervision module
"""

import numpy as np
import torch
import open3d as o3d
from typing import Tuple, Optional, Dict
import os


def gaussians_to_pointcloud(
    gaussians: np.ndarray,
    use_colors: bool = True,
    pred_scale: Optional[float] = None
) -> o3d.geometry.PointCloud:
    """
    将Gaussian参数转换为Open3D点云

    Gaussian参数格式 [N, 14]:
    - [0:3]: xyz positions (means) - 在归一化的非metric尺度下
    - [3:6]: scales
    - [6:9]: RGB colors
    - [9:13]: quaternion rotations (w, x, y, z)
    - [13:14]: opacity

    Args:
        gaussians: [N, 14] Gaussian参数
        use_colors: 是否使用颜色信息
        pred_scale: 预测的scale因子，用于将归一化坐标转换到metric尺度
                   如果提供，则执行: xyz_metric = xyz_normalized / pred_scale

    Returns:
        pcd: Open3D点云对象
    """
    pcd = o3d.geometry.PointCloud()

    # 提取位置 (means)
    positions = gaussians[:, :3].copy()

    # 如果提供了pred_scale，将归一化坐标转换到metric尺度
    if pred_scale is not None and pred_scale > 0:
        positions = positions / pred_scale
        print(f"  Converting to metric scale: positions / {pred_scale}")

    pcd.points = o3d.utility.Vector3dVector(positions)

    # 提取颜色 (如果需要)
    if use_colors and gaussians.shape[1] >= 9:
        colors = gaussians[:, 6:9]
        # gaussians -> colors transformation
        colors = 0.28209479177387814 * colors + 0.5
        # 确保颜色在[0, 1]范围内
        colors = np.clip(colors, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def pointcloud_to_gaussians(
    pcd: o3d.geometry.PointCloud,
    original_gaussians: np.ndarray,
    update_positions_only: bool = True,
    pred_scale: Optional[float] = None
) -> np.ndarray:
    """
    将ICP配准后的点云转换回Gaussian参数

    策略:
    - ICP只改变了位置(means)，其他参数保持不变
    - 如果update_positions_only=True，只更新前3个参数(xyz)
    - 如果update_positions_only=False，也会尝试更新颜色（需要应用逆变换）

    Args:
        pcd: ICP配准后的点云（在metric尺度下）
        original_gaussians: [N, 14] 原始Gaussian参数（在归一化尺度下）
        update_positions_only: 是否只更新位置
        pred_scale: 预测的scale因子，用于将metric尺度转换回归一化尺度
                   如果提供，则执行: xyz_normalized = xyz_metric * pred_scale

    Returns:
        updated_gaussians: [N, 14] 更新后的Gaussian参数（在归一化尺度下）
    """
    updated_gaussians = original_gaussians.copy()

    # 更新位置
    new_positions = np.asarray(pcd.points).copy()

    # 如果提供了pred_scale，将metric尺度转换回归一化尺度
    if pred_scale is not None and pred_scale > 0:
        new_positions = new_positions * pred_scale
        print(f"  Converting back to normalized scale: positions * {pred_scale}")

    updated_gaussians[:, :3] = new_positions

    # 如果需要，更新颜色
    if not update_positions_only and len(pcd.colors) > 0:
        new_colors = np.asarray(pcd.colors)
        # 应用逆变换：从[0,1]范围转换回Gaussian颜色格式
        # 正变换是: colors_pcd = 0.28209479177387814 * colors_gauss + 0.5
        # 逆变换是: colors_gauss = (colors_pcd - 0.5) / 0.28209479177387814
        new_colors = (new_colors - 0.5) / 0.28209479177387814
        # 确保颜色在合理范围内（通常在[-1.77, 1.77]左右）
        new_colors = np.clip(new_colors, -2.0, 2.0)
        updated_gaussians[:, 6:9] = new_colors

    return updated_gaussians


def save_pointcloud_visualization(
    pcd: o3d.geometry.PointCloud,
    save_path: str,
    format: str = 'ply'
) -> bool:
    """
    保存点云用于可视化验证

    Args:
        pcd: 点云对象
        save_path: 保存路径 (不包含扩展名)
        format: 保存格式 ('ply', 'pcd', 'xyz')

    Returns:
        success: 是否保存成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 添加扩展名
        full_path = f"{save_path}.{format}"

        # 保存点云
        success = o3d.io.write_point_cloud(full_path, pcd)

        if success:
            print(f"Point cloud saved to: {full_path}")
        else:
            print(f"Failed to save point cloud to: {full_path}")

        return success
    except Exception as e:
        print(f"Error saving point cloud: {e}")
        return False


def load_sample_pair(npz_path: str) -> Dict:
    """
    加载ICP样本对

    Args:
        npz_path: .npz文件路径

    Returns:
        data: 包含以下键的字典:
            - input_gaussians: [N, 14] 粗糙Gaussian参数
            - target_gaussians: [N, 14] ICP配准后的GT参数
            - pred_scale: float, 用于体素化的scale
            - object_id: int, 物体ID
            - (optional) input_pcd_path: 输入点云路径
            - (optional) target_pcd_path: GT点云路径
    """
    try:
        data = np.load(npz_path, allow_pickle=True)

        result = {
            'input_gaussians': data['input_gaussians'],
            'target_gaussians': data['target_gaussians'],
            'pred_scale': float(data['pred_scale']),
            'object_id': int(data['object_id']),
        }

        # 加载可选字段
        if 'input_pcd_path' in data:
            result['input_pcd_path'] = str(data['input_pcd_path'])
        if 'target_pcd_path' in data:
            result['target_pcd_path'] = str(data['target_pcd_path'])

        return result
    except Exception as e:
        print(f"Error loading sample pair from {npz_path}: {e}")
        return None


def compute_chamfer_distance(
    pcd1: o3d.geometry.PointCloud,
    pcd2: o3d.geometry.PointCloud
) -> float:
    """
    计算两个点云之间的Chamfer距离

    Args:
        pcd1, pcd2: 点云对象

    Returns:
        chamfer_dist: Chamfer距离
    """
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)

    # KD树加速最近邻搜索
    pcd1_tree = o3d.geometry.KDTreeFlann(pcd1)
    pcd2_tree = o3d.geometry.KDTreeFlann(pcd2)

    # pcd1 -> pcd2
    dist_1to2 = []
    for point in points1:
        [_, idx, dist] = pcd2_tree.search_knn_vector_3d(point, 1)
        dist_1to2.append(dist[0])

    # pcd2 -> pcd1
    dist_2to1 = []
    for point in points2:
        [_, idx, dist] = pcd1_tree.search_knn_vector_3d(point, 1)
        dist_2to1.append(dist[0])

    # Chamfer距离: 平均双向最近邻距离
    chamfer_dist = (np.mean(dist_1to2) + np.mean(dist_2to1)) / 2.0

    return chamfer_dist


def validate_gaussian_params(gaussians: np.ndarray) -> Tuple[bool, str]:
    """
    验证Gaussian参数的有效性

    Args:
        gaussians: [N, 14] Gaussian参数

    Returns:
        (is_valid, error_message)
    """
    if gaussians.shape[1] != 14:
        return False, f"Invalid shape: expected [N, 14], got {gaussians.shape}"

    # 检查NaN和Inf
    if np.isnan(gaussians).any():
        return False, "Contains NaN values"
    if np.isinf(gaussians).any():
        return False, "Contains Inf values"

    # 检查颜色范围 - Gaussian颜色不是直接在[0, 1]范围
    # 它们通过变换 colors_rgb = 0.28209479177387814 * colors + 0.5 转换到[0, 1]
    # 因此原始Gaussian颜色应该大约在[-1.77, 1.77]范围内
    colors = gaussians[:, 6:9]
    # 变换到RGB空间检查
    colors_rgb = 0.28209479177387814 * colors + 0.5
    if (colors_rgb < -0.1).any() or (colors_rgb > 1.1).any():
        # 允许小的浮点误差
        return False, f"Colors transform to RGB out of range [0, 1]: min={colors_rgb.min()}, max={colors_rgb.max()}"

    # 检查opacity范围 [0, 1]
    opacity = gaussians[:, 13:14]
    if (opacity < 0).any() or (opacity > 1).any():
        return False, f"Opacity out of range [0, 1]: min={opacity.min()}, max={opacity.max()}"

    # 检查quaternion norm
    quats = gaussians[:, 9:13]
    quat_norms = np.linalg.norm(quats, axis=1)
    if not np.allclose(quat_norms, 1.0, atol=1e-3):
        return False, f"Quaternions not normalized: norms range [{quat_norms.min()}, {quat_norms.max()}]"

    return True, "Valid"


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """将PyTorch tensor转换为numpy数组"""
    if tensor.is_cuda:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().numpy()


def numpy_to_torch(array: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """将numpy数组转换为PyTorch tensor"""
    return torch.from_numpy(array).to(device)
