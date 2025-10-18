#!/usr/bin/env python3
"""
基于光流的点云配准系统
使用光流模型计算相邻两帧之间的2D对应点，结合深度信息计算3D变换
使用Procrustes算法进行精确的3D刚体变换估计
将同一物体的多帧点云聚合到中间帧上
"""

from vggt.utils.auxiliary import RAFTCfg, calc_flow
from raft import RAFT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.training.loss import depth_to_world_points, velocity_local_to_global
from src.dust3r.utils.misc import tf32_off
import sys
import os
import numpy as np
import torch
import argparse
import pickle
import time
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import json
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/vggt'))


# 添加RAFT相关导入
sys.path.append(os.path.join(os.path.dirname(__file__), "src/SEA-RAFT/core"))

# 导入vggt中的RAFTCfg和calc_flow函数


class OpticalFlowRegistration:
    """基于光流的点云配准类"""

    def __init__(self,
                 flow_model_name: str = "raft",
                 device: str = "cuda",
                 min_inliers_ratio: float = 0.1,  # 降低最小内点比例
                 ransac_threshold: float = 5.0,  # 增加RANSAC阈值
                 max_flow_magnitude: float = 200.0,  # 增加最大光流幅度
                 use_simple_correspondence: bool = True,
                 use_direct_correspondence: bool = True,  # 使用直接索引匹配
                 use_velocity_based_transform: bool = False,  # 使用velocity直接计算变换
                 raft_model_path: str = None):
        """
        初始化光流配准器

        Args:
            flow_model_name: 光流模型名称 ("raft", "pwc", "flownet2")
            device: 计算设备
            min_inliers_ratio: 最小内点比例
            ransac_threshold: RANSAC阈值
            max_flow_magnitude: 最大光流幅度阈值
            use_simple_correspondence: 是否使用简单对应点查找方法（更快）
            use_direct_correspondence: 是否使用直接索引匹配（最快且最准确）
            use_velocity_based_transform: 是否使用velocity直接计算变换（无需光流，最快）
            raft_model_path: RAFT模型权重文件路径（可选）
        """
        self.device = device
        self.min_inliers_ratio = min_inliers_ratio
        self.ransac_threshold = ransac_threshold
        self.max_flow_magnitude = max_flow_magnitude
        self.use_simple_correspondence = use_simple_correspondence
        self.use_direct_correspondence = use_direct_correspondence
        self.use_velocity_based_transform = use_velocity_based_transform

        # 初始化光流模型
        self.flow_model = self._load_flow_model(
            flow_model_name, raft_model_path)

        # 存储配准结果
        self.registration_results = {}

        transform_method = "Velocity-based" if use_velocity_based_transform else "Flow-based"
        print(
            f"光流配准器初始化完成 - 模型: {flow_model_name}, 变换方法: {transform_method}, 对应点查找: {'简单' if use_simple_correspondence else '复杂'}")

    def _load_flow_model(self, model_name: str, raft_model_path: str = None):
        """加载光流模型"""
        try:
            if model_name == "raft":
                # 参照demo_viser.py中的RAFT模型加载方式
                try:
                    # RAFT模型配置 - 使用与demo_viser.py相同的配置
                    # 构建绝对路径
                    if raft_model_path:
                        model_path = raft_model_path
                    else:
                        current_dir = os.path.dirname(
                            os.path.abspath(__file__))
                        model_path = os.path.join(
                            current_dir, "src", "Tartan-C-T-TSKH-kitti432x960-M.pth")

                    raft_args = RAFTCfg(
                        name="kitti-M",
                        dataset="kitti",
                        path=model_path,
                        use_var=True,
                        var_min=0,
                        var_max=10,
                        pretrain="resnet34",
                        initial_dim=64,
                        block_dims=[64, 128, 256],
                        radius=4,
                        dim=128,
                        num_blocks=2,
                        iters=4,
                        image_size=[432, 960],
                        geo_thresh=2,
                        photo_thresh=-1
                    )

                    # 构建RAFT模型
                    raft_model = RAFT(raft_args)

                    # 加载预训练权重 - 参照demo_viser.py的方式
                    # 使用与RAFTCfg相同的路径，确保一致性
                    checkpoint_path = model_path
                    if os.path.exists(checkpoint_path):
                        state_dict = torch.load(
                            checkpoint_path, map_location=self.device)
                        raft_model.load_state_dict(state_dict)
                        pass  # 成功加载RAFT预训练权重
                    else:
                        print(f"警告: RAFT预训练权重文件不存在: {checkpoint_path}")
                        print("将使用随机初始化的权重")

                    raft_model.to(self.device)
                    raft_model.eval()
                    raft_model.requires_grad_(False)

                    pass  # 成功加载RAFT光流模型
                    return raft_model

                except Exception as e:
                    print(f"RAFT模型加载失败: {e}，使用OpenCV作为备选")
                    return "opencv"

            elif model_name == "pwc":
                # 尝试加载PWC-Net模型（需要额外安装）
                try:
                    # 这里需要根据实际的PWC-Net安装路径进行调整
                    print("警告: PWC-Net模型需要额外安装，使用OpenCV作为备选")
                    return "opencv"
                except ImportError:
                    print("PWC-Net模型未安装，使用OpenCV Farneback光流")
                    return "opencv"
            else:
                # 使用OpenCV的Farneback光流作为默认方法
                print(f"使用OpenCV Farneback光流")
                return "opencv"
        except Exception as e:
            print(f"加载光流模型失败: {e}，使用OpenCV Farneback光流")
            return "opencv"

    # 注意：现在使用calc_flow函数替代了compute_optical_flow方法

    def extract_object_points_2d_3d(self,
                                    clustering_result: Dict,
                                    depth: torch.Tensor,
                                    intrinsic: torch.Tensor,
                                    image_shape: Tuple[int, int],
                                    extrinsic: torch.Tensor = None,
                                    image_rgb: torch.Tensor = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        矢量化版本：批量提取物体在2D图像和3D空间中的点，性能提升100+倍

        Args:
            clustering_result: 聚类结果
            depth: 深度图 [H, W]
            intrinsic: 相机内参 [3, 3]
            image_shape: 图像尺寸 (H, W)
            extrinsic: 相机外参 [3, 4]，可选
            image_rgb: RGB图像 [3, H, W] 或 [H, W, 3]，可选

        Returns:
            points_2d: 2D点坐标 [N, 2]
            points_3d: 3D点坐标 [N, 3]
            point_indices: 点在原始图像中的索引 [N]
            colors: RGB颜色 [N, 3]
        """
        H, W = image_shape

        # 1. 预处理：获取所有cluster的像素索引
        cluster_indices = clustering_result.get('cluster_indices', [])

        # 如果没有cluster_indices，尝试从其他字段构建
        if not cluster_indices:
            if 'labels' in clustering_result and 'global_ids' in clustering_result:
                labels = clustering_result['labels']
                global_ids = clustering_result['global_ids']

                cluster_indices = []
                for i, gid in enumerate(global_ids):
                    mask = (labels == i)
                    indices = np.where(mask.flatten())[0] if hasattr(
                        mask, 'flatten') else np.where(mask)[0]
                    if len(indices) > 0:
                        cluster_indices.append(indices.tolist())

        if not cluster_indices:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # 2. 矢量化预处理：合并所有索引和对应的cluster_id
        all_indices = []
        cluster_ids = []

        for cluster_idx, point_indices in enumerate(cluster_indices):
            if point_indices:
                all_indices.extend(point_indices)
                cluster_ids.extend([cluster_idx] * len(point_indices))

        if not all_indices:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # 转换为numpy数组用于矢量化操作
        all_indices = np.array(all_indices, dtype=np.int32)
        cluster_ids = np.array(cluster_ids, dtype=np.int32)

        # 3. 矢量化边界检查
        valid_mask = (all_indices >= 0) & (all_indices < H * W)

        if not np.any(valid_mask):
            return np.array([]), np.array([]), np.array([]), np.array([])

        # 过滤有效索引
        valid_indices = all_indices[valid_mask]
        valid_cluster_ids = cluster_ids[valid_mask]

        # 4. 矢量化坐标转换：一维索引 -> 2D坐标
        y_coords = valid_indices // W
        x_coords = valid_indices % W
        coords_2d = np.column_stack([x_coords, y_coords])  # [N, 2]

        # 5. 矢量化深度提取
        depth_np = depth.detach().cpu().numpy()
        depths = depth_np[y_coords, x_coords]  # 使用fancy indexing批量提取

        # 6. 矢量化深度有效性检查
        depth_valid_mask = depths > 0

        if not np.any(depth_valid_mask):
            return np.array([]), np.array([]), np.array([]), np.array([])

        # 过滤有效深度的点
        final_coords_2d = coords_2d[depth_valid_mask]
        final_depths = depths[depth_valid_mask]
        final_indices = valid_indices[depth_valid_mask]
        final_cluster_ids = valid_cluster_ids[depth_valid_mask]

        # 7. 矢量化3D坐标计算
        points_3d = self._pixels_to_3d_vectorized(
            final_coords_2d, final_depths, intrinsic, extrinsic)

        # 8. 矢量化颜色提取
        if image_rgb is not None:
            # 处理RGB图像格式
            if isinstance(image_rgb, torch.Tensor):
                rgb_np = image_rgb.detach().cpu().numpy()
                if rgb_np.shape[0] == 3:  # [3, H, W] -> [H, W, 3]
                    rgb_np = rgb_np.transpose(1, 2, 0)
            else:
                rgb_np = image_rgb

            # 批量提取颜色
            colors = rgb_np[final_coords_2d[:, 1],
                            final_coords_2d[:, 0]]  # [N, 3]

            # 矢量化颜色格式转换
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            else:
                colors = colors.astype(np.uint8)
        else:
            # 矢量化默认颜色生成
            unique_clusters = np.unique(final_cluster_ids)
            color_map = {}

            for cluster_idx in unique_clusters:
                hue = (cluster_idx * 137.5) % 360
                color = (self._hsv_to_rgb(hue, 0.8, 0.9)
                         * 255).astype(np.uint8)
                color_map[cluster_idx] = color

            # 批量分配颜色
            colors = np.array([color_map[cluster_id]
                              for cluster_id in final_cluster_ids])

        # 9. 返回结果（注意2D坐标格式调整为[x, y]）
        points_2d = final_coords_2d  # 已经是[x, y]格式

        return points_2d, points_3d, final_indices, colors

    def _create_cluster_indices_for_global_id(self, clustering_result: Dict, global_id: int) -> Dict:
        """
        为特定global_id创建clustering结果，转换为extract_object_points_2d_3d期望的格式

        Args:
            clustering_result: 原始聚类结果，包含'labels', 'global_ids'等字段
            global_id: 目标物体的全局ID

        Returns:
            格式化的聚类结果字典，包含'cluster_indices'字段
        """
        if 'labels' not in clustering_result or 'global_ids' not in clustering_result:
            print(f"    调试: clustering_result缺少必要字段 - labels或global_ids")
            return {'cluster_indices': []}

        labels = clustering_result['labels']
        global_ids = clustering_result['global_ids']

        # 找到global_id对应的cluster索引
        if global_id not in global_ids:
            print(f"    调试: global_id {global_id} 不在global_ids中: {global_ids}")
            return {'cluster_indices': []}

        cluster_idx = global_ids.index(global_id)

        # 获取属于该cluster的点的mask
        object_mask = labels == cluster_idx

        # 如果是torch tensor，转换为numpy
        if isinstance(object_mask, torch.Tensor):
            object_mask = object_mask.detach().cpu().numpy()

        # 找到所有属于该物体的像素点索引
        object_indices = np.where(object_mask.flatten())[0]

        return {
            'cluster_indices': [object_indices.tolist()] if len(object_indices) > 0 else []
        }

    def _pixel_to_3d(self, x: float, y: float, depth: float, intrinsic: torch.Tensor, extrinsic: torch.Tensor = None) -> np.ndarray:
        """
        将像素坐标和深度转换为3D坐标

        Args:
            x: 像素x坐标
            y: 像素y坐标
            depth: 深度值
            intrinsic: 相机内参 [3, 3]
            extrinsic: 相机外参 [3, 4]，可选，如果提供则转换到世界坐标系

        Returns:
            3D点坐标 [3]
        """
        intrinsic_np = intrinsic.detach().cpu().numpy()

        # 检查内参矩阵的维度
        if intrinsic_np.ndim == 4:  # BxSx3x3
            intrinsic_np = intrinsic_np[0, 0]  # 取第一个batch和第一个序列
        elif intrinsic_np.ndim == 3:  # Sx3x3
            intrinsic_np = intrinsic_np[0]  # 取第一个序列

        fx, fy = intrinsic_np[0, 0], intrinsic_np[1, 1]
        cx, cy = intrinsic_np[0, 2], intrinsic_np[1, 2]

        # 相机坐标系下的3D点
        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth

        point_camera = np.array([X, Y, Z])

        # 如果提供了外参，转换到世界坐标系
        if extrinsic is not None:
            extrinsic_np = extrinsic.detach().cpu().numpy()

            # 检查外参矩阵的维度
            if extrinsic_np.ndim == 4:  # BxSx4x4 或 BxSx3x4
                extrinsic_np = extrinsic_np[0, 0]  # 取第一个batch和第一个序列
            elif extrinsic_np.ndim == 3:  # Sx4x4 或 Sx3x4
                extrinsic_np = extrinsic_np[0]  # 取第一个序列

            # 确保外参矩阵是4x4的齐次变换矩阵
            if extrinsic_np.shape == (3, 4):
                # 如果是3x4矩阵，扩展为4x4齐次矩阵
                bottom_row = np.array([[0, 0, 0, 1]])
                extrinsic_np = np.concatenate(
                    [extrinsic_np, bottom_row], axis=0)
            elif extrinsic_np.shape != (4, 4):
                # 如果不是预期的形状，记录错误并跳过变换
                return point_camera

            # 转换为齐次坐标
            point_homo = np.concatenate([point_camera, [1]])

            # 应用外参变换（相机到世界）
            point_world_homo = extrinsic_np @ point_homo

            # 返回3D坐标
            return point_world_homo[:3]

        return point_camera

    def _pixels_to_3d_vectorized(self, coords_2d: np.ndarray, depths: np.ndarray,
                                 intrinsic: torch.Tensor, extrinsic: torch.Tensor = None) -> np.ndarray:
        """
        批量将像素坐标和深度转换为3D坐标（矢量化版本）

        Args:
            coords_2d: 像素坐标 [N, 2] (x, y)
            depths: 深度值 [N]
            intrinsic: 相机内参 [3, 3]
            extrinsic: 相机外参 [3, 4] 或 [4, 4]，可选

        Returns:
            3D点坐标 [N, 3]
        """
        if len(coords_2d) == 0:
            return np.array([]).reshape(0, 3)

        # 处理内参矩阵
        intrinsic_np = intrinsic.detach().cpu().numpy()
        if intrinsic_np.ndim == 4:  # BxSx3x3
            intrinsic_np = intrinsic_np[0, 0]
        elif intrinsic_np.ndim == 3:  # Sx3x3
            intrinsic_np = intrinsic_np[0]

        fx, fy = intrinsic_np[0, 0], intrinsic_np[1, 1]
        cx, cy = intrinsic_np[0, 2], intrinsic_np[1, 2]

        # 矢量化计算相机坐标系下的3D点
        x, y = coords_2d[:, 0], coords_2d[:, 1]
        X = (x - cx) * depths / fx
        Y = (y - cy) * depths / fy
        Z = depths

        points_camera = np.column_stack([X, Y, Z])  # [N, 3]

        # 如果提供了外参，批量转换到世界坐标系
        if extrinsic is not None:
            extrinsic_np = extrinsic.detach().cpu().numpy()

            # 处理外参矩阵维度
            if extrinsic_np.ndim == 4:  # BxSx4x4 或 BxSx3x4
                extrinsic_np = extrinsic_np[0, 0]
            elif extrinsic_np.ndim == 3:  # Sx4x4 或 Sx3x4
                extrinsic_np = extrinsic_np[0]

            # 确保是4x4齐次变换矩阵
            if extrinsic_np.shape == (3, 4):
                bottom_row = np.array([[0, 0, 0, 1]])
                extrinsic_np = np.concatenate(
                    [extrinsic_np, bottom_row], axis=0)
            elif extrinsic_np.shape != (4, 4):
                return points_camera

            # 求外参的逆矩阵（从相机坐标系到世界坐标系）
            try:
                extrinsic_inv = np.linalg.inv(extrinsic_np)
            except np.linalg.LinAlgError:
                return points_camera

            # 转换为齐次坐标并批量变换
            points_homo = np.column_stack(
                [points_camera, np.ones(len(points_camera))])  # [N, 4]
            points_world_homo = (extrinsic_inv @ points_homo.T).T  # [N, 4]
            return points_world_homo[:, :3]  # [N, 3]

        return points_camera

    def _extract_colors_for_points(self,
                                   clustering_result: Dict,
                                   cluster_idx: int,
                                   frame_idx: int,
                                   preds: Dict,
                                   vggt_batch: Dict) -> Optional[np.ndarray]:
        """
        为特定聚类的点提取颜色信息

        Args:
            clustering_result: 聚类结果
            cluster_idx: 聚类索引
            frame_idx: 帧索引
            preds: 模型预测结果
            vggt_batch: 输入数据批次

        Returns:
            colors: RGB颜色数组 [N, 3] 或 None
        """
        try:
            # 获取图像数据
            if 'images' not in vggt_batch:
                return None

            # 获取RGB图像 [B, S, C, H, W]
            images = vggt_batch['images']
            if frame_idx >= images.shape[1]:
                return None

            image = images[0, frame_idx]  # [C, H, W]

            # 获取聚类点的索引
            cluster_indices = clustering_result.get('cluster_indices', [])
            if cluster_idx >= len(cluster_indices):
                return None

            point_indices = cluster_indices[cluster_idx]
            if not point_indices:
                return None

            # 提取颜色
            H, W = image.shape[1], image.shape[2]
            colors = []

            for idx in point_indices:
                y = idx // W
                x = idx % W

                if 0 <= y < H and 0 <= x < W:
                    # 提取RGB值 [C] -> [3]
                    color = image[:, y, x].detach().cpu().numpy()

                    # 确保颜色值在[0, 255]范围内
                    if color.max() <= 1.0:
                        color = (color * 255).astype(np.uint8)
                    else:
                        color = color.astype(np.uint8)

                    colors.append(color)

            return np.array(colors) if colors else None

        except Exception as e:
            print(f"提取颜色信息失败: {e}")
            return None


    def estimate_transformation_direct(self,
                                       points_3d_src: np.ndarray,
                                       points_3d_dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        使用Procrustes算法计算3D-3D点对应的刚体变换

        Args:
            points_3d_src: 源帧3D点 [N, 3]
            points_3d_dst: 目标帧3D点 [N, 3]

        Returns:
            R: 旋转矩阵 [3, 3]
            t: 平移向量 [3]
            inlier_ratio: 内点比例
        """
        if len(points_3d_src) < 3:
            return np.eye(3), np.zeros(3), 0.0

        # 检查输入有效性
        if len(points_3d_src) != len(points_3d_dst):
            print(f"警告: 3D点对应数量不匹配 - src: {len(points_3d_src)}, dst: {len(points_3d_dst)}")
            return np.eye(3), np.zeros(3), 0.0

        # 检查点是否包含无效值
        if np.any(np.isnan(points_3d_src)) or np.any(np.isnan(points_3d_dst)):
            print("警告: 3D点包含NaN值")
            return np.eye(3), np.zeros(3), 0.0

        if np.any(np.isinf(points_3d_src)) or np.any(np.isinf(points_3d_dst)):
            print("警告: 3D点包含无限值")
            return np.eye(3), np.zeros(3), 0.0

        try:
            # 使用RANSAC进行鲁棒估计
            if len(points_3d_src) > 100:
                return self._estimate_transformation_ransac(points_3d_src, points_3d_dst)
            else:
                # 点数较少时直接使用Procrustes算法
                R, t = self._procrustes_algorithm(points_3d_src, points_3d_dst)
                # 计算内点比例
                inlier_ratio = self._compute_inlier_ratio(points_3d_src, points_3d_dst, R, t)
                return R, t, inlier_ratio
            
            # 直接使用差值计算变换T，R不管了
            # best_R = np.eye(3)
            # best_t = points_3d_dst.mean(axis=0) - points_3d_src.mean(axis=0)
            # best_inlier_ratio = 0.99
            # return best_R, best_t, best_inlier_ratio

        except (ValueError, np.linalg.LinAlgError) as e:
            print(f"变换估计失败: {str(e)}")
            return np.eye(3), np.zeros(3), 0.0

    def _procrustes_algorithm(self, points_src: np.ndarray, points_dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Procrustes/Kabsch算法：计算3D点云之间的最优刚体变换

        Args:
            points_src: 源点云 [N, 3]
            points_dst: 目标点云 [N, 3]

        Returns:
            R: 旋转矩阵 [3, 3]
            t: 平移向量 [3]
        """

        with tf32_off():
            # 输入检查
            if len(points_src) != len(points_dst):
                raise ValueError(f"点云大小不匹配: {len(points_src)} vs {len(points_dst)}")

            if len(points_src) < 3:
                raise ValueError(f"点数不足，至少需要3个点，实际有{len(points_src)}个")

            # 1. 计算质心
            centroid_src = np.mean(points_src, axis=0)
            centroid_dst = np.mean(points_dst, axis=0)

            # 2. 去中心化
            points_src_centered = points_src - centroid_src
            points_dst_centered = points_dst - centroid_dst

            # 检查点是否共线（协方差矩阵是否退化）
            if np.allclose(points_src_centered, 0) or np.allclose(points_dst_centered, 0):
                # 所有点都是同一个点，只需要平移
                return np.eye(3), centroid_dst - centroid_src

            # 3. 计算协方差矩阵 H = P_src^T * P_dst
            H = points_src_centered.T @ points_dst_centered

            # 检查矩阵是否退化
            if np.allclose(H, 0):
                # 协方差矩阵为零矩阵，返回单位变换
                return np.eye(3), centroid_dst - centroid_src

            # 4. SVD分解
            try:
                U, S, Vt = np.linalg.svd(H)
            except np.linalg.LinAlgError as e:
                raise np.linalg.LinAlgError(f"SVD分解失败: {str(e)}")

            # 5. 计算旋转矩阵
            R = Vt.T @ U.T

            # 6. 处理反射情况（确保det(R) = 1）
            if np.linalg.det(R) < 0:
                # 修正最小奇异值对应的向量
                Vt_corrected = Vt.copy()
                Vt_corrected[-1, :] *= -1
                R = Vt_corrected.T @ U.T

            # 验证旋转矩阵
            if not self._is_valid_rotation_matrix(R):
                raise ValueError("计算得到的旋转矩阵无效")

            # 7. 计算平移向量
            t = centroid_dst - R @ centroid_src

        return R, t

    def _is_valid_rotation_matrix(self, R: np.ndarray, tolerance: float = 1e-6) -> bool:
        """
        验证是否为有效的旋转矩阵

        Args:
            R: 待验证的矩阵 [3, 3]
            tolerance: 数值精度容差

        Returns:
            是否为有效旋转矩阵
        """
        # 检查矩阵大小
        if R.shape != (3, 3):
            return False

        # 检查是否正交：R^T * R = I
        should_be_identity = R.T @ R
        identity = np.eye(3)
        if not np.allclose(should_be_identity, identity, atol=tolerance):
            return False

        # 检查行列式是否为1
        det = np.linalg.det(R)
        if not np.isclose(det, 1.0, atol=tolerance):
            return False

        return True

    def _compute_inlier_ratio(self, points_src: np.ndarray, points_dst: np.ndarray,
                             R: np.ndarray, t: np.ndarray, threshold: float = 0.1) -> float:
        """
        计算内点比例

        Args:
            points_src: 源点云 [N, 3]
            points_dst: 目标点云 [N, 3]
            R: 旋转矩阵 [3, 3]
            t: 平移向量 [3]
            threshold: 距离阈值

        Returns:
            内点比例
        """
        # 变换源点云
        points_src_transformed = (R @ points_src.T).T + t

        # 计算距离
        distances = np.linalg.norm(points_src_transformed - points_dst, axis=1)

        # 计算内点
        inliers = distances < threshold
        inlier_ratio = np.sum(inliers) / len(points_src)

        return inlier_ratio

    def _estimate_transformation_ransac(self, points_src: np.ndarray, points_dst: np.ndarray,
                                       max_iterations: int = 1000, threshold: float = 0.01) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        使用RANSAC + Procrustes算法进行鲁棒的刚体变换估计

        Args:
            points_src: 源点云 [N, 3]
            points_dst: 目标点云 [N, 3]
            max_iterations: RANSAC最大迭代次数
            threshold: 内点距离阈值

        Returns:
            R: 最优旋转矩阵 [3, 3]
            t: 最优平移向量 [3]
            inlier_ratio: 最佳内点比例
        """
        n_points = len(points_src)
        best_inlier_ratio = 0.0
        best_R = np.eye(3)
        best_t = np.zeros(3)

        # 至少需要3个点来估计刚体变换
        min_samples = 5

        for iteration in range(max_iterations):
            # 随机选择最小样本集
            sample_indices = np.random.choice(n_points, min_samples, replace=False)
            sample_src = points_src[sample_indices]
            sample_dst = points_dst[sample_indices]

            try:
                # 使用Procrustes算法估计变换
                R, t = self._procrustes_algorithm(sample_src, sample_dst)

                # 计算所有点的内点比例
                inlier_ratio = self._compute_inlier_ratio(points_src, points_dst, R, t, threshold)

                # 更新最佳模型
                if inlier_ratio > best_inlier_ratio:
                    best_inlier_ratio = inlier_ratio
                    best_R = R
                    best_t = t

                # 早停条件：如果内点比例足够高
                if inlier_ratio > 0.5:
                    break

            except np.linalg.LinAlgError:
                # SVD分解失败，跳过此次迭代
                continue

        # # 使用最佳模型的内点进行最终优化
        if best_inlier_ratio > 0.0:
            # 找到内点
            points_src_transformed = (best_R @ points_src.T).T + best_t
            distances = np.linalg.norm(points_src_transformed - points_dst, axis=1)
            inlier_mask = distances < threshold

            if np.sum(inlier_mask) >= min_samples:
                # 使用所有内点重新估计变换
                inlier_src = points_src[inlier_mask]
                inlier_dst = points_dst[inlier_mask]
                try:
                    best_R, best_t = self._procrustes_algorithm(inlier_src, inlier_dst)
                    best_inlier_ratio = self._compute_inlier_ratio(points_src, points_dst, best_R, best_t, threshold)
                except np.linalg.LinAlgError:
                    pass  # 保持原始最佳结果


        return best_R, best_t, best_inlier_ratio

    def _find_corresponding_points_direct(self,
                                          indices_src: np.ndarray,
                                          indices_dst: np.ndarray,
                                          flow: np.ndarray,
                                          max_flow_magnitude: float,
                                          H: int, W: int) -> np.ndarray:
        """
        直接基于光流变换建立对应点关系（完全矢量化版本）

        正确逻辑：
        1. 将indices_src通过光流变换到新位置
        2. 检查变换后的位置是否在indices_dst中存在
        3. 建立src位置到dst位置的对应关系

        Args:
            indices_src: 源帧像素索引 [N]
            indices_dst: 目标帧像素索引 [M] 
            flow: 光流场 [H, W, 2]
            max_flow_magnitude: 最大光流幅度阈值
            H, W: 图像尺寸

        Returns:
            corresponding_points: 对应点索引对 [K, 2] - [src_pos, dst_pos]
        """
        if len(indices_src) == 0 or len(indices_dst) == 0:
            return np.array([])

        # 1. 矢量化坐标转换：索引 → (y,x)坐标
        y_src = indices_src // W
        x_src = indices_src % W

        # 2. 矢量化边界检查（源点）
        src_boundary_mask = ((x_src >= 0) & (x_src < W) &
                             (y_src >= 0) & (y_src < H))
        if not np.any(src_boundary_mask):
            return np.array([])

        # 过滤边界内的源点
        valid_y_src = y_src[src_boundary_mask]
        valid_x_src = x_src[src_boundary_mask]
        valid_src_indices = np.where(src_boundary_mask)[0]

        # 3. 矢量化光流查询
        flow_vectors = flow[valid_y_src, valid_x_src]  # [N_valid, 2]

        # 4. 矢量化光流幅度验证
        flow_magnitudes = np.sqrt(np.sum(flow_vectors**2, axis=1))  # [N_valid]
        flow_valid_mask = flow_magnitudes <= max_flow_magnitude

        if not np.any(flow_valid_mask):
            return np.array([])

        # 过滤光流合理的点
        final_y_src = valid_y_src[flow_valid_mask]
        final_x_src = valid_x_src[flow_valid_mask]
        final_flow_vectors = flow_vectors[flow_valid_mask]
        final_src_indices = valid_src_indices[flow_valid_mask]

        # 5. 矢量化坐标变换：src_coords + flow_vectors
        predicted_x = final_x_src + final_flow_vectors[:, 0]
        predicted_y = final_y_src + final_flow_vectors[:, 1]

        # 转换为整数坐标（四舍五入）
        predicted_x = np.round(predicted_x).astype(int)
        predicted_y = np.round(predicted_y).astype(int)

        # 6. 矢量化边界检查（预测点）
        pred_boundary_mask = ((predicted_x >= 0) & (predicted_x < W) &
                              (predicted_y >= 0) & (predicted_y < H))

        if not np.any(pred_boundary_mask):
            return np.array([])

        # 过滤边界内的预测点
        final_pred_x = predicted_x[pred_boundary_mask]
        final_pred_y = predicted_y[pred_boundary_mask]
        final_final_src_indices = final_src_indices[pred_boundary_mask]

        # 7. 矢量化索引转换：(y,x) → 像素索引
        predicted_indices = final_pred_y * W + final_pred_x  # [N_final]

        # 8. 矢量化成员检查：预测索引是否在dst中
        membership_mask = np.isin(predicted_indices, indices_dst)

        if not np.any(membership_mask):
            return np.array([])

        # 9. 矢量化映射构建：找到dst中对应的位置
        matched_predicted_indices = predicted_indices[membership_mask]
        matched_src_indices = final_final_src_indices[membership_mask]

        # 使用searchsorted快速查找dst位置（前提：indices_dst已排序）
        indices_dst_sorted = np.sort(indices_dst)
        sort_indices = np.argsort(indices_dst)

        # 找到每个matched_predicted_index在sorted数组中的位置
        dst_positions_in_sorted = np.searchsorted(
            indices_dst_sorted, matched_predicted_indices)

        # 映射回原始dst数组中的位置
        dst_positions = sort_indices[dst_positions_in_sorted]

        # 10. 构建最终对应点数组
        corresponding_points = np.column_stack(
            [matched_src_indices, dst_positions])

        return corresponding_points

    def precompute_optical_flows(self, vggt_batch: Dict) -> Dict:
        """
        预先计算所有相邻帧之间的光流 - 使用calc_flow函数

        如果使用velocity-based方法，直接返回空字典（无需计算光流）

        Args:
            vggt_batch: 输入数据批次

        Returns:
            光流字典 {frame_pair: flow}，如果使用velocity-based则返回空字典
        """
        # 如果使用velocity-based变换方法，跳过光流计算
        if self.use_velocity_based_transform:
            return {}

        B, S, C, H, W = vggt_batch["images"].shape

        # 使用calc_flow函数计算所有相邻帧之间的光流
        # calc_flow返回 (forward_flow, backward_flow, forward_consist_mask, backward_consist_mask, forward_in_bound_mask, backward_in_bound_mask)
        print("  计算光流中...")
        forward_flow, backward_flow = calc_flow(
            vggt_batch["images"],
            self.flow_model,
            check_consistency=False,  # 暂时不检查一致性
            geo_thresh=self.flow_model.args.geo_thresh,
            photo_thresh=self.flow_model.args.photo_thresh,
            interval=1
        )

        # 构建光流字典
        flows = {}
        for frame_idx in range(S - 1):
            # forward_flow: [B, S, 2, H, W]
            flow = forward_flow[0, frame_idx].detach(
            ).cpu().numpy().transpose(1, 2, 0)  # [H, W, 2]
            flows[(frame_idx, frame_idx + 1)] = flow
        print(f"  光流计算完成，共 {len(flows)} 对")
        return flows

    def compute_chain_transformation(self,
                                     start_frame: int,
                                     end_frame: int,
                                     flows: Dict,
                                     clustering_results: List[Dict],
                                     preds: Dict,
                                     vggt_batch: Dict,
                                     global_id: int) -> Optional[np.ndarray]:
        """
        计算从起始帧到目标帧的链式变换

        Args:
            start_frame: 起始帧
            end_frame: 目标帧
            flows: 预计算的光流字典
            clustering_results: 聚类结果
            preds: 模型预测结果
            vggt_batch: 输入数据批次
            global_id: 物体全局ID

        Returns:
            累积变换矩阵或None
        """
        if start_frame == end_frame:
            return np.eye(4)

        # 获取相机参数
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            preds["pose_enc"], vggt_batch["images"].shape[-2:])

        # 确定变换方向
        if start_frame < end_frame:
            # 正向变换：start_frame -> end_frame
            frame_sequence = list(range(start_frame, end_frame))
            direction = 1
        else:
            # 反向变换：start_frame -> end_frame
            frame_sequence = list(range(start_frame, end_frame, -1))
            direction = -1

        # 累积变换矩阵
        cumulative_transformation = np.eye(4)

        for i, frame_idx in enumerate(frame_sequence):
            next_frame = frame_idx + direction

            # 获取当前帧和下一帧的聚类结果
            current_result = clustering_results[frame_idx]
            next_result = clustering_results[next_frame]

            # 检查物体是否在两帧中都存在
            current_global_ids = current_result.get('global_ids', [])
            next_global_ids = next_result.get('global_ids', [])

            if global_id not in current_global_ids or global_id not in next_global_ids:
                print(f"物体 {global_id} 在帧 {frame_idx} 或 {next_frame} 中不存在")
                return None

            # 获取光流（如果使用flow-based方法）
            flow = None
            if not self.use_velocity_based_transform:
                if direction == 1:
                    flow_key = (frame_idx, next_frame)
                else:
                    flow_key = (next_frame, frame_idx)

                if flow_key not in flows:
                    print(f"未找到帧 {frame_idx} -> {next_frame} 的光流")
                    return None

                flow = flows[flow_key]
                if direction == -1:
                    # 反向光流
                    flow = -flow

            # 获取深度数据
            depth_src = preds["depth"][0, frame_idx, :, :, 0]  # [H, W]
            depth_dst = preds["depth"][0, next_frame, :, :, 0]  # [H, W]

            # 获取velocity数据（如果启用velocity-based方法）
            velocity_src = None
            velocity_dst = None
            if self.use_velocity_based_transform and 'velocity' in preds:
                # velocity shape: [B, S, H, W, 3]
                velocity_src = preds['velocity'][0, frame_idx]  # [H, W, 3]
                velocity_dst = preds['velocity'][0, next_frame]  # [H, W, 3]

            # 获取对应帧的内参和外参
            intrinsic_src = intrinsic[0, frame_idx]  # [3, 3]
            intrinsic_dst = intrinsic[0, next_frame]  # [3, 3]
            extrinsic_src = extrinsic[0, frame_idx]  # [3, 4]
            extrinsic_dst = extrinsic[0, next_frame]  # [3, 4]

            # 计算单步变换
            step_transformation = self.compute_single_step_transformation(
                current_result, next_result,
                depth_src, depth_dst,
                flow, intrinsic_src, intrinsic_dst, extrinsic_src, extrinsic_dst, global_id,
                velocity_src=velocity_src, velocity_dst=velocity_dst,
                direction=direction
            )

            if step_transformation is None:
                print(f"帧 {frame_idx} -> {next_frame} 的变换计算失败")
                return None

            # 累积变换
            cumulative_transformation = step_transformation @ cumulative_transformation

        return cumulative_transformation

    def compute_optimized_chain_transformation(self,
                                               start_frame: int,
                                               end_frame: int,
                                               flows: Dict,
                                               clustering_results: List[Dict],
                                               preds: Dict,
                                               vggt_batch: Dict,
                                               global_id: int,
                                               transformation_cache: Dict) -> Optional[np.ndarray]:
        """
        优化版本的链式变换计算，使用缓存避免重复计算

        Args:
            start_frame: 起始帧
            end_frame: 目标帧  
            flows: 预计算的光流字典
            clustering_results: 聚类结果
            preds: 模型预测结果
            vggt_batch: 输入数据批次
            global_id: 物体全局ID
            transformation_cache: 变换缓存字典

        Returns:
            累积变换矩阵或None
        """
        if start_frame == end_frame:
            return np.eye(4)

        # 检查缓存中是否已有这个变换
        cache_key = (start_frame, end_frame)
        if cache_key in transformation_cache:
            return transformation_cache[cache_key]

        # 获取相机参数
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            preds["pose_enc"], vggt_batch["images"].shape[-2:])

        # 确定变换方向和路径
        if start_frame < end_frame:
            frame_sequence = list(range(start_frame, end_frame))
            direction = 1
        else:
            frame_sequence = list(range(start_frame, end_frame, -1))
            direction = -1

        # 尝试利用已有的变换进行优化
        # 例如：如果要计算1->4，但已经有了1->3和3->4，则直接组合
        for intermediate_frame in range(min(start_frame, end_frame) + 1, max(start_frame, end_frame)):
            key1 = (start_frame, intermediate_frame)
            key2 = (intermediate_frame, end_frame)

            if key1 in transformation_cache and key2 in transformation_cache:
                # 组合已有的变换
                combined_transformation = np.dot(
                    transformation_cache[key2], transformation_cache[key1])
                transformation_cache[cache_key] = combined_transformation
                return combined_transformation

        # 如果没有可用的缓存组合，按照原来的方法计算
        cumulative_transformation = np.eye(4)

        for i, frame_idx in enumerate(frame_sequence):
            next_frame = frame_idx + direction

            # 检查单步变换的缓存
            step_key = (frame_idx, next_frame)
            if step_key in transformation_cache:
                step_transformation = transformation_cache[step_key]
            else:
                # 获取当前帧和下一帧的聚类结果
                current_result = clustering_results[frame_idx]
                next_result = clustering_results[next_frame]

                # 检查物体是否在两帧中都存在
                current_global_ids = current_result.get('global_ids', [])
                next_global_ids = next_result.get('global_ids', [])

                if global_id not in current_global_ids or global_id not in next_global_ids:
                    return None

                # 获取光流（如果使用flow-based方法）
                flow = None
                if not self.use_velocity_based_transform:
                    if direction == 1:
                        flow_key = (frame_idx, next_frame)
                    else:
                        flow_key = (next_frame, frame_idx)

                    if flow_key not in flows:
                        return None

                    flow = flows[flow_key]
                    if direction == -1:
                        flow = -flow

                # 获取深度数据
                depth_src = preds["depth"][0, frame_idx, :, :, 0]
                depth_dst = preds["depth"][0, next_frame, :, :, 0]

                # 获取velocity数据（如果启用velocity-based方法）
                velocity_src = None
                velocity_dst = None
                if self.use_velocity_based_transform and 'velocity' in preds:
                    velocity_src = preds['velocity'][0, frame_idx]  # [H, W, 3]
                    velocity_dst = preds['velocity'][0, next_frame]  # [H, W, 3]

                # 计算单步变换
                step_transformation = self.compute_single_step_transformation(
                    current_result, next_result, depth_src, depth_dst, flow,
                    intrinsic[0, frame_idx], intrinsic[0, next_frame],
                    extrinsic[0, frame_idx], extrinsic[0, next_frame],
                    global_id,
                    velocity_src=velocity_src, velocity_dst=velocity_dst,
                    direction=direction
                )

                if step_transformation is None:
                    return None

                # 缓存单步变换
                transformation_cache[step_key] = step_transformation

            # 累积变换
            cumulative_transformation = np.dot(
                step_transformation, cumulative_transformation)

        # 缓存最终结果
        transformation_cache[cache_key] = cumulative_transformation

        return cumulative_transformation

    def _compute_velocity_based_transformation(self,
                                                clustering_src: Dict,
                                                clustering_dst: Dict,
                                                velocity_src: torch.Tensor,
                                                velocity_dst: Optional[torch.Tensor],
                                                global_id: int,
                                                H: int,
                                                W: int,
                                                direction: int = 1) -> Optional[np.ndarray]:
        """
        基于velocity直接计算变换矩阵（无需光流）

        策略：
        1. 提取源帧中属于该物体的所有点的velocity
        2. 对这些velocity取平均，得到平均运动向量
        3. 如果是backward (direction=-1)，对velocity取反
        4. 构建变换矩阵：R = I（单位阵），t = 平均velocity * direction

        Args:
            clustering_src: 源帧聚类结果
            clustering_dst: 目标帧聚类结果（暂未使用，保留用于未来扩展）
            velocity_src: 源帧velocity场 [H, W, 3]
            velocity_dst: 目标帧velocity场（可选，暂未使用）
            global_id: 物体全局ID
            H, W: 图像尺寸
            direction: 变换方向，1表示forward，-1表示backward

        Returns:
            4x4变换矩阵或None
        """
        # 1. 提取属于该物体的点的索引
        # 从clustering_src中找到global_id对应的聚类索引
        global_ids = clustering_src.get('global_ids', [])
        if global_id not in global_ids:
            return None

        cluster_idx = global_ids.index(global_id)
        cluster_indices = clustering_src.get('cluster_indices', [])
        if cluster_idx >= len(cluster_indices):
            return None

        object_indices = cluster_indices[cluster_idx]  # List of flattened indices
        if len(object_indices) == 0:
            return None

        # 2. 将velocity从[H, W, 3]转为[H*W, 3]并提取对应点的velocity
        if isinstance(velocity_src, torch.Tensor):
            if len(velocity_src.shape) == 3:  # [H, W, 3]
                velocity_flat = velocity_src.reshape(H * W, 3)
            elif len(velocity_src.shape) == 2:  # [H*W, 3]
                velocity_flat = velocity_src
            else:
                return None

            # 提取对应点的velocity
            object_velocities = velocity_flat[object_indices]  # [N, 3]

            # 3. 计算平均velocity
            mean_velocity = object_velocities.mean(dim=0)  # [3]

            # 转为numpy (先detach再转换)
            if isinstance(mean_velocity, torch.Tensor):
                mean_velocity = mean_velocity.detach().cpu().numpy()
        else:
            # velocity_src已经是numpy
            if len(velocity_src.shape) == 3:
                velocity_flat = velocity_src.reshape(H * W, 3)
            else:
                velocity_flat = velocity_src

            object_velocities = velocity_flat[object_indices]
            mean_velocity = np.mean(object_velocities, axis=0)

        # 4. 根据方向调整velocity
        # direction = 1: forward (src -> dst), 使用原velocity
        # direction = -1: backward (dst -> src), velocity取反
        adjusted_velocity = mean_velocity * direction

        # 5. 构建变换矩阵：R = I, t = adjusted_velocity
        transformation = np.eye(4)
        transformation[:3, :3] = np.eye(3)  # 单位旋转矩阵
        transformation[:3, 3] = adjusted_velocity  # 平移为调整后的velocity

        return transformation

    def compute_single_step_transformation(self,
                                           clustering_src: Dict,
                                           clustering_dst: Dict,
                                           depth_src: torch.Tensor,
                                           depth_dst: torch.Tensor,
                                           flow: np.ndarray,
                                           intrinsic_src: torch.Tensor,
                                           intrinsic_dst: torch.Tensor,
                                           extrinsic_src: torch.Tensor,
                                           extrinsic_dst: torch.Tensor,
                                           global_id: int,
                                           velocity_src: Optional[torch.Tensor] = None,
                                           velocity_dst: Optional[torch.Tensor] = None,
                                           direction: int = 1) -> Optional[np.ndarray]:
        """
        计算单步变换（相邻两帧之间）

        Args:
            clustering_src: 源帧聚类结果
            clustering_dst: 目标帧聚类结果
            depth_src: 源帧深度图
            depth_dst: 目标帧深度图
            flow: 光流场
            intrinsic_src: 源帧相机内参
            intrinsic_dst: 目标帧相机内参
            extrinsic_src: 源帧相机外参
            extrinsic_dst: 目标帧相机外参
            global_id: 物体全局ID
            velocity_src: 源帧velocity场 [H, W, 3]（可选，用于velocity-based方法）
            velocity_dst: 目标帧velocity场 [H, W, 3]（可选，用于velocity-based方法）
            direction: 变换方向，1表示forward (src->dst)，-1表示backward (dst->src)

        Returns:
            变换矩阵或None
        """
        import time
        method_start_time = time.time()

        H, W = depth_src.shape

        # ========== Velocity-based方法 ==========
        if self.use_velocity_based_transform and velocity_src is not None:
            # 使用velocity直接计算变换（无需光流，更快更简单）
            return self._compute_velocity_based_transformation(
                clustering_src, clustering_dst, velocity_src, velocity_dst, global_id, H, W, direction
            )

        # ========== Flow-based方法（原方法） ==========
        # 1. 聚类数据格式化
        clustering_format_start = time.time()
        # 提取物体的2D和3D点（暂时不需要颜色信息用于变换计算）
        # 为当前的clustering结果创建适合extract_object_points_2d_3d的格式
        clustering_src_formatted = self._create_cluster_indices_for_global_id(
            clustering_src, global_id)
        clustering_dst_formatted = self._create_cluster_indices_for_global_id(
            clustering_dst, global_id)
        clustering_format_time = time.time() - clustering_format_start

        # 2. 2D/3D点提取
        point_extraction_start = time.time()
        points_2d_src, points_3d_src, indices_src, _ = self.extract_object_points_2d_3d(
            clustering_src_formatted, depth_src, intrinsic_src, (H, W), extrinsic_src)
        points_2d_dst, points_3d_dst, indices_dst, _ = self.extract_object_points_2d_3d(
            clustering_dst_formatted, depth_dst, intrinsic_dst, (H, W), extrinsic_dst)
        point_extraction_time = time.time() - point_extraction_start

        if len(points_2d_src) == 0 or len(points_2d_dst) == 0:
            return None

        # 3. 对应点查找
        correspondence_start = time.time()
        # 为了减少计算量，只选取indices_src中的部分点进行匹配（随机0.1）
        # indices_src = np.random.choice(indices_src, int(len(indices_src) * 0.5), replace=False)

        # # For debug: 超级简单的方法------------------
        # src_centroid_3d = np.mean(points_3d_src, axis=0)
        # dst_centroid_3d = np.mean(points_3d_dst, axis=0)
        # translation_3d = dst_centroid_3d - src_centroid_3d
        # # 构建变换矩阵
        # transformation = np.eye(4)
        # transformation[:3, :3] = np.eye(3)  # 单位旋转矩阵
        # transformation[:3, 3] = translation_3d
        # return transformation
        # # --------------------------------------

        corresponding_points = self._find_corresponding_points_direct(
            indices_src, indices_dst, flow, self.max_flow_magnitude, H, W)
        correspondence_time = time.time() - correspondence_start

        if len(corresponding_points) < 3:  # Procrustes算法至少需要3个点
            method_name = "直接索引匹配" if self.use_direct_correspondence else "光流+最近邻匹配"

            # 回退方案：使用3D点质心偏移计算简单变换
            if len(points_3d_src) >= 1 and len(points_3d_dst) >= 1:
                # 计算3D质心偏移作为简单变换
                src_centroid_3d = np.mean(points_3d_src, axis=0)
                dst_centroid_3d = np.mean(points_3d_dst, axis=0)
                translation_3d = dst_centroid_3d - src_centroid_3d

                # 构建变换矩阵
                transformation = np.eye(4)
                transformation[:3, :3] = np.eye(3)  # 单位旋转矩阵
                transformation[:3, 3] = translation_3d
                return transformation
            else:
                # 返回单位变换矩阵
                return np.eye(4)

        # 4. 对应点数据准备
        correspondence_prep_start = time.time()
        # 提取对应的3D点
        src_indices = corresponding_points[:, 0].astype(int)
        dst_indices = corresponding_points[:, 1].astype(int)

        points_3d_src_corr = points_3d_src[src_indices]
        points_3d_dst_corr = points_3d_dst[dst_indices]
        correspondence_prep_time = time.time() - correspondence_prep_start


        # For debug: 超级简单的方法------------------
        src_centroid_3d = np.mean(points_3d_src_corr, axis=0)
        dst_centroid_3d = np.mean(points_3d_dst_corr, axis=0)
        translation_3d = dst_centroid_3d - src_centroid_3d
        # 构建变换矩阵
        transformation = np.eye(4)
        transformation[:3, :3] = np.eye(3)  # 单位旋转矩阵
        transformation[:3, 3] = translation_3d
        return transformation
        # --------------------------------------


        # 5. 变换估计
        transformation_estimation_start = time.time()
        # 使用Procrustes算法估计变换
        R, t, inlier_ratio = self.estimate_transformation_direct(
            points_3d_src_corr, points_3d_dst_corr)
        transformation_estimation_time = time.time() - transformation_estimation_start

        # 检查内点比例
        if inlier_ratio < self.min_inliers_ratio:
            method_total_time = time.time() - method_start_time
            pass  # 单步变换性能分析已移除
            print(
                f"          调试: 内点比例过低 - {inlier_ratio:.3f} < {self.min_inliers_ratio}")
            return None

        # 6. 变换矩阵构建
        matrix_construction_start = time.time()
        # 构建变换矩阵
        transformation = np.eye(4)
        transformation[:3, :3] = R
        transformation[:3, 3] = t
        matrix_construction_time = time.time() - matrix_construction_start

        return transformation

    def aggregate_object_to_middle_frame(self,
                                         clustering_results: List[Dict],
                                         preds: Dict,
                                         vggt_batch: Dict,
                                         global_id: int,
                                         flows: Dict) -> Optional[Dict]:
        """
        将同一物体的多帧点云聚合到中间帧（使用链式变换）

        Args:
            clustering_results: 所有帧的聚类结果
            preds: 模型预测结果
            vggt_batch: 输入数据批次
            global_id: 物体全局ID
            flows: 预计算的光流字典

        Returns:
            聚合结果字典或None
        """
        import time
        method_start_time = time.time()
        step_times = {}

        # 1. 找到物体出现的帧
        frame_discovery_start = time.time()
        object_frames = []
        for frame_idx, result in enumerate(clustering_results):
            global_ids = result.get('global_ids', [])
            if global_id in global_ids:
                object_frames.append(frame_idx)
        step_times['1. 发现物体帧'] = time.time() - frame_discovery_start

        if len(object_frames) < 1:
            print(f"物体 {global_id}: 未找到该物体")
            return None
        elif len(object_frames) == 1:
            # 对于单帧物体，直接使用该帧的数据，无需聚合
            single_frame_start = time.time()
            frame_idx = object_frames[0]
            result = clustering_results[frame_idx]
            cluster_idx = result['global_ids'].index(global_id)
            object_mask = result['labels'] == cluster_idx
            object_points = result['points'][object_mask]

            # 单帧颜色处理已删除（不再需要）

            # 获取单帧的像素索引
            cluster_indices = result.get('cluster_indices', [])
            single_frame_pixel_indices = cluster_indices[cluster_idx] if cluster_idx < len(cluster_indices) else []

            # 提取单帧的Gaussian参数，并用实际点坐标替换位置
            canonical_gaussians = None
            frame_gaussians = {}  # 新增：每帧的原始Gaussian参数
            if preds and 'gaussian_params' in preds:
                object_points_np = object_points.cpu().numpy() if isinstance(object_points, torch.Tensor) else object_points
                # 创建单帧的点索引对应关系
                single_frame_point_indices = [(frame_idx, pixel_idx) for pixel_idx in single_frame_pixel_indices[:len(object_points_np)]]

                canonical_gaussians = self._extract_all_frames_gaussian_params(
                    object_points_np,
                    single_frame_point_indices,
                    preds['gaussian_params'],
                    vggt_batch
                )

                # 提取单帧的原始Gaussian参数
                frame_gaussians[frame_idx] = self._extract_gaussian_params_for_object(
                    result,
                    cluster_idx,
                    frame_idx,
                    preds['gaussian_params'],
                    vggt_batch
                )

            step_times['2. 单帧处理'] = time.time() - single_frame_start
            step_times['总耗时'] = time.time() - method_start_time

            return {
                'global_id': global_id,
                'aggregated_points': object_points.cpu().numpy() if isinstance(object_points, torch.Tensor) else object_points,
                'point_indices': [(frame_idx, pixel_idx) for pixel_idx in single_frame_pixel_indices[:len(object_points)]],  # 添加点索引对应关系
                'middle_frame': frame_idx,  # 统一使用middle_frame
                'object_frames': [frame_idx],
                'transformations': {},
                'canonical_gaussians': canonical_gaussians,  # 添加Gaussian参数
                'frame_gaussians': frame_gaussians,  # 新增：每帧的原始Gaussian参数
                'reference_frame': frame_idx,  # 保留向后兼容
                'num_frames': 1,
                'num_points': len(object_points),
                'step_times': step_times  # 添加时间统计
            }

        # 2. 选择中间帧作为参考帧并提取其数据
        middle_frame_start = time.time()
        middle_frame_idx = object_frames[len(object_frames) // 2]
        pass  # 物体聚合信息

        # 获取中间帧的物体点云和颜色
        middle_result = clustering_results[middle_frame_idx]
        middle_cluster_idx = middle_result['global_ids'].index(global_id)
        middle_points = middle_result['points']
        middle_labels = middle_result['labels']

        # 提取中间帧物体的点
        middle_object_mask = middle_labels == middle_cluster_idx
        middle_object_points = middle_points[middle_object_mask]

        # 确保中间帧点云在CPU上
        if isinstance(middle_object_points, torch.Tensor):
            middle_object_points_cpu = middle_object_points.detach().cpu().numpy()
        else:
            middle_object_points_cpu = middle_object_points

        # 获取中间帧的像素索引（用于后续Gaussian参数提取）
        cluster_indices = middle_result.get('cluster_indices', [])
        middle_pixel_indices = cluster_indices[middle_cluster_idx] if middle_cluster_idx < len(cluster_indices) else []

        # 中间帧颜色处理已删除（不再需要）
        step_times['2. 中间帧数据提取'] = time.time() - middle_frame_start

        # 3. 存储所有帧的变换 - 优化版本
        transformations = {}
        transformation_cache = {}  # 缓存已计算的变换
        aggregated_points = [middle_object_points_cpu]
        # 同时记录每个点对应的(frame_idx, pixel_idx)
        all_point_indices = [(middle_frame_idx, pixel_idx) for pixel_idx in middle_pixel_indices[:len(middle_object_points_cpu)]]

        # 4. 对其他帧进行链式变换
        chain_transform_start = time.time()
        successful_transforms = 0
        failed_transforms = 0
        individual_transform_times = []

        # 详细统计每个子步骤
        transform_computation_times = []
        point_extraction_times = []
        point_transformation_times = []

        for frame_idx in object_frames:
            if frame_idx == middle_frame_idx:
                continue

            frame_transform_start = time.time()

            # 4.1 计算链式变换
            transform_compute_start = time.time()
            chain_transformation = self.compute_optimized_chain_transformation(
                frame_idx, middle_frame_idx, flows, clustering_results, preds, vggt_batch, global_id, transformation_cache)
            transform_compute_time = time.time() - transform_compute_start
            transform_computation_times.append(transform_compute_time)

            if chain_transformation is not None:
                successful_transforms += 1
                pass  # 变换成功
                transformations[frame_idx] = {
                    'transformation': chain_transformation,
                    'R': chain_transformation[:3, :3],
                    't': chain_transformation[:3, 3],
                    'inlier_ratio': 1.0,
                    'num_correspondences': 0
                }

                # 4.2 提取当前帧的物体点云
                point_extract_start = time.time()
                current_result = clustering_results[frame_idx]
                current_cluster_idx = current_result['global_ids'].index(global_id)
                current_object_mask = current_result['labels'] == current_cluster_idx
                current_object_points = current_result['points'][current_object_mask]

                # 获取当前帧的像素索引
                current_cluster_indices = current_result.get('cluster_indices', [])
                current_pixel_indices = current_cluster_indices[current_cluster_idx] if current_cluster_idx < len(current_cluster_indices) else []

                point_extract_time = time.time() - point_extract_start
                point_extraction_times.append(point_extract_time)

                # 4.3 应用变换
                transform_apply_start = time.time()
                transformed_points = self._apply_transformation(
                    current_object_points, chain_transformation)
                aggregated_points.append(transformed_points)

                # 记录当前帧点的索引信息（只记录有效变换点的数量）
                num_transformed_points = len(transformed_points)
                frame_point_indices = [(frame_idx, pixel_idx) for pixel_idx in current_pixel_indices[:num_transformed_points]]
                all_point_indices.extend(frame_point_indices)

                transform_apply_time = time.time() - transform_apply_start
                point_transformation_times.append(transform_apply_time)

                # 4.4 颜色处理已删除（不再需要）

                frame_total_time = time.time() - frame_transform_start
                individual_transform_times.append(frame_total_time)

            else:
                failed_transforms += 1
                print(f"    ⚠️  链式变换失败: {frame_idx}")
                # 即使失败也记录时间
                frame_total_time = time.time() - frame_transform_start
                individual_transform_times.append(frame_total_time)
                pass  # 链式变换失败

        # 如果没有成功的变换，但至少有中间帧数据，就返回中间帧
        if len(aggregated_points) < 2 and len(aggregated_points) >= 1:
            pass  # 链式变换失败，使用中间帧数据
        elif len(aggregated_points) < 1:
            print(f"物体 {global_id}: 没有可用数据")
            return None

        # 统计链式变换的详细时间
        step_times['3. 链式变换总耗时'] = time.time() - chain_transform_start
        if transform_computation_times:
            step_times['3.1 变换计算平均'] = sum(transform_computation_times) / len(transform_computation_times)
            step_times['3.2 变换计算最大'] = max(transform_computation_times)
        if point_extraction_times:
            step_times['3.3 点提取平均'] = sum(point_extraction_times) / len(point_extraction_times)
        if point_transformation_times:
            step_times['3.4 点变换平均'] = sum(point_transformation_times) / len(point_transformation_times)
        if individual_transform_times:
            step_times['3.6 单帧处理平均'] = sum(individual_transform_times) / len(individual_transform_times)
            step_times['3.7 单帧处理最大'] = max(individual_transform_times)
        step_times['3.8 成功变换数'] = successful_transforms
        step_times['3.9 失败变换数'] = failed_transforms

        # 4. 合并所有点云
        merge_start = time.time()
        all_points = np.concatenate(aggregated_points, axis=0)
        step_times['4. 点云合并'] = time.time() - merge_start

        # 5. 提取所有帧对应的Gaussian参数，并用实际点坐标替换位置
        gaussian_start = time.time()
        canonical_gaussians = None
        frame_gaussians = {}  # 新增：每帧的原始Gaussian参数
        if preds and 'gaussian_params' in preds:
            canonical_gaussians = self._extract_all_frames_gaussian_params(
                all_points,  # 聚合后的实际点坐标
                all_point_indices,  # 每个点对应的(frame_idx, pixel_idx)
                preds['gaussian_params'],
                vggt_batch
            )

            # 提取每帧单独的Gaussian参数（不进行坐标替换）
            for frame_idx in object_frames:
                current_result = clustering_results[frame_idx]
                current_cluster_idx = current_result['global_ids'].index(global_id)
                frame_gaussians[frame_idx] = self._extract_gaussian_params_for_object(
                    current_result,
                    current_cluster_idx,
                    frame_idx,
                    preds['gaussian_params'],
                    vggt_batch
                )
        step_times['5. Gaussian参数提取'] = time.time() - gaussian_start

        step_times['总耗时'] = time.time() - method_start_time

        return {
            'global_id': global_id,
            'middle_frame': middle_frame_idx,
            'object_frames': object_frames,
            'aggregated_points': all_points,
            'point_indices': all_point_indices,  # 添加点索引对应关系
            'transformations': transformations,
            'canonical_gaussians': canonical_gaussians,  # 添加Gaussian参数
            'frame_gaussians': frame_gaussians,  # 新增：每帧的原始Gaussian参数
            'num_frames': len(object_frames),
            'num_points': len(all_points),
            'step_times': step_times  # 添加详细时间统计
        }

    def _extract_all_frames_gaussian_params(self,
                                            aggregated_points: np.ndarray,
                                            point_indices: List[Tuple[int, int]],
                                            gaussian_params: torch.Tensor,
                                            vggt_batch: Dict) -> torch.Tensor:
        """
        使用矢量操作直接提取对应的Gaussian参数，并用聚合后的点坐标替换位置

        Args:
            aggregated_points: 聚合后的3D点坐标 [N, 3]
            point_indices: 每个点对应的(frame_idx, pixel_idx) [(frame, pixel), ...]
            gaussian_params: VGGT预测的Gaussian参数 [B, S*H*W, 14]
            vggt_batch: 批次数据

        Returns:
            合并的Gaussian参数 [N, 14] 或 None
        """
        try:
            if 'images' not in vggt_batch:
                print(f"    无法从vggt_batch获取图像尺寸")
                return None

            if len(point_indices) != len(aggregated_points):
                print(f"    点数量与索引数量不匹配: {len(aggregated_points)} vs {len(point_indices)}")
                return None

            B, S, C, H, W = vggt_batch['images'].shape
            H_W = H * W

            # 分离frame_indices和pixel_indices
            frame_indices = [idx[0] for idx in point_indices]
            pixel_indices = [idx[1] for idx in point_indices]

            # 转换为tensor进行矢量操作
            frame_indices_tensor = torch.tensor(frame_indices, dtype=torch.long)
            pixel_indices_tensor = torch.tensor(pixel_indices, dtype=torch.long)

            # 计算全局索引：frame_idx * H*W + pixel_idx
            global_indices = frame_indices_tensor * H_W + pixel_indices_tensor

            # 创建valid_mask来过滤有效索引
            valid_mask = (global_indices >= 0) & (global_indices < gaussian_params.shape[1])
            valid_global_indices = global_indices[valid_mask]

            if len(valid_global_indices) == 0:
                print(f"    没有有效的索引")
                return None

            # 使用矢量操作一次性提取所有对应的Gaussian参数
            extracted_gaussians = gaussian_params[0, valid_global_indices].clone()  # [N_valid, 14]

            # 用聚合后的点坐标替换Gaussian参数的前三维
            aggregated_points_tensor = torch.from_numpy(aggregated_points).float()

            # 只使用有效索引对应的点
            valid_points = aggregated_points_tensor[valid_mask]
            extracted_gaussians[:, :3] = valid_points

            return extracted_gaussians

        except Exception as e:
            print(f"    提取Gaussian参数失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_gaussian_params_for_object(self,
                                            clustering_result: Dict,
                                            cluster_idx: int,
                                            frame_idx: int,
                                            gaussian_params: torch.Tensor,
                                            vggt_batch: Dict) -> torch.Tensor:
        """
        从聚类结果中直接提取对应物体的Gaussian参数

        Args:
            clustering_result: 单帧聚类结果
            cluster_idx: 聚类索引  
            frame_idx: 帧索引
            gaussian_params: VGGT预测的Gaussian参数 [B, S*H*W, 14]
            vggt_batch: 批次数据

        Returns:
            提取的Gaussian参数 [N, 14] 或 None
        """
        try:
            # 获取该聚类的像素索引
            cluster_indices = clustering_result.get('cluster_indices', [])
            if cluster_idx >= len(cluster_indices):
                print(
                    f"    cluster_idx {cluster_idx} 超出范围，总聚类数: {len(cluster_indices)}")
                return None

            pixel_indices = cluster_indices[cluster_idx]  # 该物体的所有像素索引
            if not pixel_indices:
                print(f"    物体在帧{frame_idx}中没有像素索引")
                return None

            # 推断图像尺寸
            if 'images' in vggt_batch:
                B, S, C, H, W = vggt_batch['images'].shape
            else:
                print(f"    无法从vggt_batch获取图像尺寸")
                return None

            # 计算全局索引：frame_idx * H*W + pixel_idx
            H_W = H * W
            # 使用矢量操作计算全局索引
            pixel_indices_tensor = torch.tensor(pixel_indices, dtype=torch.long)
            global_indices = frame_idx * H_W + pixel_indices_tensor

            # 提取对应的Gaussian参数
            B_g, N_total, feature_dim = gaussian_params.shape

            # 使用矢量索引直接提取参数，避免for循环
            selected_gaussians_tensor = gaussian_params[0, global_indices]  # [N, 14]

            if selected_gaussians_tensor.shape[0] == 0:
                print(f"    无法提取有效的Gaussian参数")
                return None
            pass  # 成功提取Gaussian参数

            return selected_gaussians_tensor

        except Exception as e:
            print(f"    提取Gaussian参数失败: {e}")
            return None

    def _apply_transformation(self, points: torch.Tensor, transformation: np.ndarray) -> np.ndarray:
        """应用变换矩阵到点云"""
        points_np = points.detach().cpu().numpy()

        # 转换为齐次坐标
        points_homo = np.concatenate(
            [points_np, np.ones((len(points_np), 1))], axis=1)

        # 应用变换
        transformed_points_homo = (transformation @ points_homo.T).T

        # 返回3D坐标
        return transformed_points_homo[:, :3]

    def process_pointcloud_data(self, data_path: str, output_dir: str) -> Dict:
        """
        处理点云数据文件

        Args:
            data_path: 点云数据文件路径
            output_dir: 输出目录

        Returns:
            处理结果字典
        """
        print(f"处理点云数据: {data_path}")

        # 加载数据
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        clustering_results = data.get('clustering_results', [])
        preds = data.get('preds', {})
        vggt_batch = data.get('vggt_batch', {})

        if not clustering_results:
            print("未找到聚类结果")
            return {}

        # 预先计算所有相邻帧之间的光流
        flows = self.precompute_optical_flows(vggt_batch)

        # 找到所有唯一的全局ID
        all_global_ids = set()
        for result in clustering_results:
            global_ids = result.get('global_ids', [])
            all_global_ids.update([gid for gid in global_ids if gid != -1])

        print(f"找到 {len(all_global_ids)} 个唯一物体")

        # 为每个物体进行聚合
        aggregation_results = {}

        for global_id in sorted(all_global_ids):
            print(f"\n处理物体 {global_id}...")

            result = self.aggregate_object_to_middle_frame(
                clustering_results, preds, vggt_batch, global_id, flows)

            if result is not None:
                aggregation_results[global_id] = result

        # 保存结果
        output_path = os.path.join(
            output_dir, f"optical_flow_registration_results.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump({
                'aggregation_results': aggregation_results,
                'clustering_results': clustering_results,
                'preds': preds,
                'vggt_batch': vggt_batch,
                'flows': flows
            }, f)

        print(f"\n配准结果已保存到: {output_path}")
        pass  # 成功处理物体

        # 保存点云文件供本地查看器使用
        self.save_pointclouds_for_viewer(aggregation_results, output_dir)

        return aggregation_results

    def save_pointclouds_for_viewer(self, aggregation_results: Dict, output_dir: str):
        """
        保存点云文件供本地查看器使用

        Args:
            aggregation_results: 聚合结果字典
            output_dir: 输出目录
        """
        print("保存点云文件供本地查看器使用...")

        # 创建点云保存目录
        pointcloud_dir = os.path.join(output_dir, "pointclouds")
        os.makedirs(pointcloud_dir, exist_ok=True)

        # 为每个物体保存点云
        for global_id, result in aggregation_results.items():
            aggregated_points = result['aggregated_points']
            aggregated_colors = result.get('aggregated_colors', None)

            # 创建点云文件名
            pointcloud_filename = f"object_{global_id}_aggregated.ply"
            pointcloud_path = os.path.join(pointcloud_dir, pointcloud_filename)

            # 保存为PLY格式，使用真实颜色
            self._save_points_as_ply_with_colors(
                aggregated_points, pointcloud_path, global_id, aggregated_colors)

            print(
                f"  物体 {global_id}: 保存 {len(aggregated_points)} 个点到 {pointcloud_filename}")

        # 创建查看器配置文件
        viewer_config = {
            'pointclouds': [
                {
                    'path': f"pointclouds/object_{gid}_aggregated.ply",
                    'name': f"Object_{gid}",
                    'color': [1.0, 0.0, 0.0] if gid % 3 == 0 else [0.0, 1.0, 0.0] if gid % 3 == 1 else [0.0, 0.0, 1.0]
                }
                for gid in aggregation_results.keys()
            ],
            'camera': {
                'position': [0, 0, 5],
                'look_at': [0, 0, 0],
                'up': [0, 1, 0]
            }
        }

        config_path = os.path.join(output_dir, "viewer_config.json")
        with open(config_path, 'w') as f:
            json.dump(viewer_config, f, indent=2)

        print(f"查看器配置文件已保存到: {config_path}")
        print(f"可以使用 local_pointcloud_viewer.py 查看点云文件")

    def _save_points_as_ply(self, points: np.ndarray, filepath: str, object_id: int):
        """
        将点云保存为PLY格式

        Args:
            points: 点云数据 [N, 3]
            filepath: 保存路径
            object_id: 物体ID
        """
        # 为点云生成颜色（基于物体ID）
        colors = np.zeros((len(points), 3), dtype=np.uint8)

        # 使用物体ID生成不同的颜色
        hue = (object_id * 137.5) % 360  # 黄金角度
        rgb = self._hsv_to_rgb(hue, 0.8, 0.9)
        colors[:] = (rgb * 255).astype(np.uint8)

        # 写入PLY文件
        with open(filepath, 'w') as f:
            # PLY头部
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            # 写入点云数据
            for i in range(len(points)):
                x, y, z = points[i]
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

    def _save_points_as_ply_with_colors(self, points: np.ndarray, filepath: str, object_id: int, colors: np.ndarray = None):
        """
        将点云和颜色保存为PLY格式

        Args:
            points: 点云数据 [N, 3]
            filepath: 保存路径
            object_id: 物体ID
            colors: RGB颜色数据 [N, 3]，可选
        """
        # 如果没有提供颜色，使用基于物体ID的默认颜色
        if colors is None:
            colors = np.zeros((len(points), 3), dtype=np.uint8)
            hue = (object_id * 137.5) % 360  # 黄金角度
            rgb = self._hsv_to_rgb(hue, 0.8, 0.9)
            colors[:] = (rgb * 255).astype(np.uint8)
        else:
            # 确保颜色格式正确
            colors = colors.astype(np.uint8)

        # 写入PLY文件
        with open(filepath, 'w') as f:
            # PLY头部
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            # 写入点云数据
            for i in range(len(points)):
                x, y, z = points[i]
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> np.ndarray:
        """HSV转RGB"""
        h = h / 360.0
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        if i % 6 == 0:
            return np.array([v, t, p])
        elif i % 6 == 1:
            return np.array([q, v, p])
        elif i % 6 == 2:
            return np.array([p, v, t])
        elif i % 6 == 3:
            return np.array([p, q, v])
        elif i % 6 == 4:
            return np.array([t, p, v])
        else:
            return np.array([v, p, q])


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于光流的点云配准")
    parser.add_argument("--data_path", type=str, required=True,
                        help="点云数据文件路径")
    parser.add_argument("--output_dir", type=str, default="./optical_flow_results",
                        help="输出目录")
    parser.add_argument("--flow_model", type=str, default="raft",
                        choices=["raft", "pwc", "opencv"],
                        help="光流模型 (raft: RAFT深度学习模型, opencv: OpenCV Farneback)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备")
    parser.add_argument("--min_inliers_ratio", type=float, default=0.3,
                        help="最小内点比例")
    parser.add_argument("--ransac_threshold", type=float, default=3.0,
                        help="RANSAC阈值")
    parser.add_argument("--max_flow_magnitude", type=float, default=100.0,
                        help="最大光流幅度阈值")
    parser.add_argument("--use_simple_correspondence", action="store_true", default=True,
                        help="使用简单对应点查找方法（更快）")
    parser.add_argument("--use_complex_correspondence", action="store_true",
                        help="使用复杂对应点查找方法（更精确但更慢）")
    parser.add_argument("--raft_model_path", type=str,
                        default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/Tartan-C-T-TSKH-kitti432x960-M.pth",
                        help="RAFT模型权重文件路径")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 确定对应点查找方法
    use_simple_correspondence = args.use_simple_correspondence and not args.use_complex_correspondence

    # 初始化配准器
    registration = OpticalFlowRegistration(
        flow_model_name=args.flow_model,
        device=args.device,
        min_inliers_ratio=args.min_inliers_ratio,
        ransac_threshold=args.ransac_threshold,
        max_flow_magnitude=args.max_flow_magnitude,
        use_simple_correspondence=use_simple_correspondence,
        raft_model_path=args.raft_model_path
    )

    # 处理数据
    results = registration.process_pointcloud_data(
        args.data_path, args.output_dir)

    # 打印统计信息
    if results:
        total_points = sum(result['num_points'] for result in results.values())
        avg_points = total_points / len(results)
        print(f"\n统计信息:")
        print(f"  总物体数: {len(results)}")
        print(f"  总点数: {total_points}")
        print(f"  平均每物体点数: {avg_points:.1f}")

        # 保存统计信息
        stats = {
            'total_objects': len(results),
            'total_points': total_points,
            'avg_points_per_object': avg_points,
            'object_details': {gid: {
                'num_frames': result['num_frames'],
                'num_points': result['num_points'],
                'middle_frame': result['middle_frame']
            } for gid, result in results.items()}
        }

        stats_path = os.path.join(args.output_dir, "registration_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"统计信息已保存到: {stats_path}")


if __name__ == "__main__":
    main()
