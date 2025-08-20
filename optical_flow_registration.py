#!/usr/bin/env python3
"""
基于光流的点云配准系统
使用光流模型计算相邻两帧之间的2D对应点，结合深度信息计算3D变换
支持3DPnP（复杂版）和直接计算平移（简单版）两种方法
将同一物体的多帧点云聚合到中间帧上
"""

import os
import numpy as np
import torch
import cv2
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
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/vggt'))

from src.dust3r.utils.misc import tf32_off
from training.loss import depth_to_world_points, velocity_local_to_global
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# 添加RAFT相关导入
sys.path.append(os.path.join(os.path.dirname(__file__), "src/SEA-RAFT/core"))
from raft import RAFT

# 导入vggt中的RAFTCfg和calc_flow函数
from vggt.utils.auxiliary import RAFTCfg, calc_flow
        



class OpticalFlowRegistration:
    """基于光流的点云配准类"""
    
    def __init__(self, 
                 flow_model_name: str = "raft",
                 device: str = "cuda",
                 use_pnp: bool = True,
                 min_inliers_ratio: float = 0.3,
                 ransac_threshold: float = 3.0,
                 max_flow_magnitude: float = 100.0,
                 use_simple_correspondence: bool = True,
                 raft_model_path: str = None):
        """
        初始化光流配准器
        
        Args:
            flow_model_name: 光流模型名称 ("raft", "pwc", "flownet2")
            device: 计算设备
            use_pnp: 是否使用3DPnP方法（True为复杂版，False为简单版）
            min_inliers_ratio: 最小内点比例
            ransac_threshold: RANSAC阈值
            max_flow_magnitude: 最大光流幅度阈值
            use_simple_correspondence: 是否使用简单对应点查找方法（更快）
            raft_model_path: RAFT模型权重文件路径（可选）
        """
        self.device = device
        self.use_pnp = use_pnp
        self.min_inliers_ratio = min_inliers_ratio
        self.ransac_threshold = ransac_threshold
        self.max_flow_magnitude = max_flow_magnitude
        self.use_simple_correspondence = use_simple_correspondence
        
        # 初始化光流模型
        self.flow_model = self._load_flow_model(flow_model_name, raft_model_path)
        
        # 存储配准结果
        self.registration_results = {}
        
        print(f"光流配准器初始化完成 - 模型: {flow_model_name}, 方法: {'3DPnP' if use_pnp else '直接平移'}, 对应点查找: {'简单' if use_simple_correspondence else '复杂'}")
    
    def _load_flow_model(self, model_name: str, raft_model_path: str = None):
        """加载光流模型"""
        try:
            if model_name == "raft":
                # 参照demo_viser.py中的RAFT模型加载方式
                try:
                    # RAFT模型配置 - 使用与demo_viser.py相同的配置
                    raft_args = RAFTCfg(
                        name="kitti-M", 
                        dataset="kitti", 
                        path=raft_model_path if raft_model_path else "src/Tartan-C-T-TSKH-kitti432x960-M.pth",
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
                    checkpoint_path = raft_model_path if raft_model_path else "src/Tartan-C-T-TSKH-kitti432x960-M.pth"
                    if os.path.exists(checkpoint_path):
                        state_dict = torch.load(checkpoint_path, map_location=self.device)
                        raft_model.load_state_dict(state_dict)
                        print(f"成功加载RAFT预训练权重: {checkpoint_path}")
                    else:
                        print(f"警告: RAFT预训练权重文件不存在: {checkpoint_path}")
                        print("将使用随机初始化的权重")
                    
                    raft_model.to(self.device)
                    raft_model.eval()
                    raft_model.requires_grad_(False)
                    
                    print("成功加载RAFT光流模型")
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
        提取物体在2D图像和3D空间中的点，以及对应的颜色信息
        
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
        
        # 获取物体的点索引
        cluster_indices = clustering_result.get('cluster_indices', [])
        if not cluster_indices:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        all_points_2d = []
        all_points_3d = []
        all_indices = []
        all_colors = []
        
        # 处理RGB图像，确保格式正确
        rgb_image = None
        if image_rgb is not None:
            if isinstance(image_rgb, torch.Tensor):
                rgb_np = image_rgb.cpu().numpy()
                # 检查图像格式并转换为 [H, W, 3]
                if rgb_np.shape[0] == 3:  # [3, H, W] -> [H, W, 3]
                    rgb_np = rgb_np.transpose(1, 2, 0)
                rgb_image = rgb_np
            else:
                rgb_image = image_rgb
                
        for cluster_idx, point_indices in enumerate(cluster_indices):
            if not point_indices:
                continue
                
            # 将一维索引转换为2D坐标
            for idx in point_indices:
                y = idx // W
                x = idx % W
                
                if 0 <= y < H and 0 <= x < W:
                    # 获取深度值
                    depth_val = depth[y, x].item()
                    
                    if depth_val > 0:  # 有效深度
                        # 2D点坐标
                        point_2d = np.array([x, y])
                        
                        # 3D点坐标（相机坐标系）
                        point_3d = self._pixel_to_3d(x, y, depth_val, intrinsic, extrinsic)
                        
                        # 提取颜色信息
                        if rgb_image is not None:
                            # 确保坐标在有效范围内
                            color = rgb_image[y, x]
                            # 如果颜色值在[0,1]范围内，转换到[0,255]
                            if color.max() <= 1.0:
                                color = (color * 255).astype(np.uint8)
                            else:
                                color = color.astype(np.uint8)
                        else:
                            # 如果没有RGB图像，使用基于cluster_idx的默认颜色
                            hue = (cluster_idx * 137.5) % 360
                            color = (self._hsv_to_rgb(hue, 0.8, 0.9) * 255).astype(np.uint8)
                        
                        all_points_2d.append(point_2d)
                        all_points_3d.append(point_3d)
                        all_indices.append(idx)
                        all_colors.append(color)
        
        if not all_points_2d:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        return np.array(all_points_2d), np.array(all_points_3d), np.array(all_indices), np.array(all_colors)
    
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
        intrinsic_np = intrinsic.cpu().numpy()
        
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
            extrinsic_np = extrinsic.cpu().numpy()
            
            # 检查外参矩阵的维度
            if extrinsic_np.ndim == 4:  # BxSx3x4
                extrinsic_np = extrinsic_np[0, 0]  # 取第一个batch和第一个序列
            elif extrinsic_np.ndim == 3:  # Sx3x4
                extrinsic_np = extrinsic_np[0]  # 取第一个序列
            
            # 转换为齐次坐标
            point_homo = np.concatenate([point_camera, [1]])
            
            # 应用外参变换（相机到世界）
            point_world_homo = extrinsic_np @ point_homo
            
            # 返回3D坐标
            return point_world_homo[:3]
        
        return point_camera
    
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
                    color = image[:, y, x].cpu().numpy()
                    
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
    
    def estimate_transformation_3d_pnp(self, 
                                     points_3d_src: np.ndarray, 
                                     points_3d_dst: np.ndarray,
                                     points_2d_src: np.ndarray,
                                     points_2d_dst: np.ndarray,
                                     intrinsic: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        使用3DPnP方法估计变换矩阵（复杂版）
        
        Args:
            points_3d_src: 源帧3D点 [N, 3]
            points_3d_dst: 目标帧3D点 [N, 3]
            points_2d_src: 源帧2D点 [N, 2]
            points_2d_dst: 目标帧2D点 [N, 2]
            intrinsic: 相机内参 [3, 3]
            
        Returns:
            R: 旋转矩阵 [3, 3]
            t: 平移向量 [3]
            inlier_ratio: 内点比例
        """
        if len(points_3d_src) < 4:
            return np.eye(3), np.zeros(3), 0.0
        
        intrinsic_np = intrinsic.cpu().numpy()
        
        # 检查内参矩阵的维度
        if intrinsic_np.ndim == 4:  # BxSx3x3
            intrinsic_np = intrinsic_np[0, 0]  # 取第一个batch和第一个序列
        elif intrinsic_np.ndim == 3:  # Sx3x3
            intrinsic_np = intrinsic_np[0]  # 取第一个序列
        
        # 使用RANSAC进行PnP求解
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d_src.astype(np.float32),
            points_2d_dst.astype(np.float32),
            intrinsic_np,
            None,
            reprojectionError=self.ransac_threshold,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success or inliers is None:
            return np.eye(3), np.zeros(3), 0.0
        
        # 转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()
        
        # 计算内点比例
        inlier_ratio = len(inliers) / len(points_3d_src)
        
        return R, t, inlier_ratio
    
    def estimate_transformation_direct(self, 
                                     points_3d_src: np.ndarray, 
                                     points_3d_dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        直接计算平移向量（简单版）
        
        Args:
            points_3d_src: 源帧3D点 [N, 3]
            points_3d_dst: 目标帧3D点 [N, 3]
            
        Returns:
            R: 单位旋转矩阵 [3, 3]
            t: 平移向量 [3]
            inlier_ratio: 内点比例（这里设为1.0）
        """
        if len(points_3d_src) == 0:
            return np.eye(3), np.zeros(3), 0.0
        
        # 计算质心
        centroid_src = np.mean(points_3d_src, axis=0)
        centroid_dst = np.mean(points_3d_dst, axis=0)
        
        # 平移向量
        t = centroid_dst - centroid_src
        
        # 单位旋转矩阵（不进行旋转）
        R = np.eye(3)
        
        return R, t, 1.0
    
    
    def _find_corresponding_points_flow(self, 
                                      points_2d_src: np.ndarray,
                                      points_2d_dst: np.ndarray,
                                      flow: np.ndarray,
                                      max_flow_magnitude: float,
                                      distance_threshold: float = 10.0,
                                      use_simple_method: bool = True) -> np.ndarray:
        """
        使用光流找到对应点（优化版本）
        
        Args:
            points_2d_src: 源帧2D点 [N, 2]
            points_2d_dst: 目标帧2D点 [M, 2]
            flow: 光流场 [H, W, 2]
            max_flow_magnitude: 最大光流幅度阈值
            distance_threshold: 距离阈值，用于判断预测点是否在目标点附近
            use_simple_method: 是否使用简单方法（直接检查预测点是否在目标点中）
            
        Returns:
            corresponding_points: 对应点索引 [K, 2]
        """
        H, W = flow.shape[:2]
        corresponding_points = []
        
        if use_simple_method:
            # 简单方法：将目标点转换为字典，用于快速查找
            target_points_dict = {}
            for j, point_dst in enumerate(points_2d_dst):
                # 将浮点坐标转换为整数坐标，用于字典查找
                point_key = (int(point_dst[0]), int(point_dst[1]))
                target_points_dict[point_key] = j
        else:
            # 复杂方法：使用网格化的方法，将目标点按网格分组
            grid_size = max(1, int(distance_threshold))
            grid_dict = {}
            
            for j, point_dst in enumerate(points_2d_dst):
                grid_x = int(point_dst[0] // grid_size)
                grid_y = int(point_dst[1] // grid_size)
                grid_key = (grid_x, grid_y)
                
                if grid_key not in grid_dict:
                    grid_dict[grid_key] = []
                grid_dict[grid_key].append((j, point_dst))
        
        for i, point_src in enumerate(points_2d_src):
            x, y = int(point_src[0]), int(point_src[1])
            
            if 0 <= x < W and 0 <= y < H:
                # 获取光流
                flow_x, flow_y = flow[y, x]
                
                # 检查光流幅度
                flow_magnitude = np.sqrt(flow_x**2 + flow_y**2)
                if flow_magnitude > max_flow_magnitude:
                    continue
                
                # 预测目标帧中的位置
                predicted_x = x + flow_x
                predicted_y = y + flow_y
                
                # 检查预测点是否在图像范围内
                if not (0 <= predicted_x < W and 0 <= predicted_y < H):
                    continue
                
                if use_simple_method:
                    # 简单方法：直接检查预测点是否在目标点字典中
                    predicted_point_int = (int(predicted_x), int(predicted_y))
                    if predicted_point_int in target_points_dict:
                        # 直接获取对应的目标点索引
                        j = target_points_dict[predicted_point_int]
                        corresponding_points.append([i, j])
                else:
                    # 复杂方法：在预测点周围的网格中查找最近的目标点
                    pred_grid_x = int(predicted_x // grid_size)
                    pred_grid_y = int(predicted_y // grid_size)
                    
                    min_dist = float('inf')
                    best_match = -1
                    
                    # 检查预测点周围的9个网格
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            grid_key = (pred_grid_x + dx, pred_grid_y + dy)
                            if grid_key in grid_dict:
                                for j, point_dst in grid_dict[grid_key]:
                                    dist = np.sqrt((predicted_x - point_dst[0])**2 + (predicted_y - point_dst[1])**2)
                                    if dist < min_dist and dist < distance_threshold:
                                        min_dist = dist
                                        best_match = j
                    
                    if best_match != -1:
                        corresponding_points.append([i, best_match])
        
        return np.array(corresponding_points)
    
    def precompute_optical_flows(self, vggt_batch: Dict) -> Dict:
        """
        预先计算所有相邻帧之间的光流 - 使用calc_flow函数
        
        Args:
            vggt_batch: 输入数据批次
            
        Returns:
            光流字典 {frame_pair: flow}
        """
        print("预先计算所有相邻帧之间的光流...")
        
        B, S, C, H, W = vggt_batch["images"].shape
        
        # 使用calc_flow函数计算所有相邻帧之间的光流
        # calc_flow返回 (forward_flow, backward_flow, forward_consist_mask, backward_consist_mask, forward_in_bound_mask, backward_in_bound_mask)
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
            flow = forward_flow[0, frame_idx].cpu().numpy().transpose(1, 2, 0)  # [H, W, 2]
            flows[(frame_idx, frame_idx + 1)] = flow
            print(f"  计算帧 {frame_idx} -> {frame_idx + 1} 的光流")
        
        print(f"完成 {len(flows)} 个光流计算")
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
            
            # 获取光流
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
            
            # 获取对应帧的内参和外参
            intrinsic_src = intrinsic[0, frame_idx]  # [3, 3]
            intrinsic_dst = intrinsic[0, next_frame]  # [3, 3]
            extrinsic_src = extrinsic[0, frame_idx]  # [3, 4]
            extrinsic_dst = extrinsic[0, next_frame]  # [3, 4]
            
            # 计算单步变换
            step_transformation = self.compute_single_step_transformation(
                current_result, next_result,
                depth_src, depth_dst,
                flow, intrinsic_src, intrinsic_dst, extrinsic_src, extrinsic_dst, global_id
            )
            
            if step_transformation is None:
                print(f"帧 {frame_idx} -> {next_frame} 的变换计算失败")
                return None
            
            # 累积变换
            cumulative_transformation = step_transformation @ cumulative_transformation
        
        return cumulative_transformation
    
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
                                         global_id: int) -> Optional[np.ndarray]:
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
            
        Returns:
            变换矩阵或None
        """
        H, W = depth_src.shape
        
        # 提取物体的2D和3D点（暂时不需要颜色信息用于变换计算）
        points_2d_src, points_3d_src, indices_src, _ = self.extract_object_points_2d_3d(
            clustering_src, depth_src, intrinsic_src, (H, W), extrinsic_src)
        points_2d_dst, points_3d_dst, indices_dst, _ = self.extract_object_points_2d_3d(
            clustering_dst, depth_dst, intrinsic_dst, (H, W), extrinsic_dst)
        
        if len(points_2d_src) == 0 or len(points_2d_dst) == 0:
            return None
        
        # 使用光流找到对应点
        corresponding_points = self._find_corresponding_points_flow(
            points_2d_src, points_2d_dst, flow, self.max_flow_magnitude, use_simple_method=self.use_simple_correspondence)
        
        if len(corresponding_points) < 4:
            return None
        
        # 提取对应的3D点
        src_indices = corresponding_points[:, 0].astype(int)
        dst_indices = corresponding_points[:, 1].astype(int)
        
        points_3d_src_corr = points_3d_src[src_indices]
        points_3d_dst_corr = points_3d_dst[dst_indices]
        points_2d_src_corr = points_2d_src[src_indices]
        points_2d_dst_corr = points_2d_dst[dst_indices]
        
        # 估计变换
        if self.use_pnp:
            R, t, inlier_ratio = self.estimate_transformation_3d_pnp(
                points_3d_src_corr, points_3d_dst_corr,
                points_2d_src_corr, points_2d_dst_corr, intrinsic_dst)
        else:
            R, t, inlier_ratio = self.estimate_transformation_direct(
                points_3d_src_corr, points_3d_dst_corr)
        
        # 检查内点比例
        if inlier_ratio < self.min_inliers_ratio:
            return None
        
        # 构建变换矩阵
        transformation = np.eye(4)
        transformation[:3, :3] = R
        transformation[:3, 3] = t
        
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
        # 找到物体出现的帧
        object_frames = []
        for frame_idx, result in enumerate(clustering_results):
            global_ids = result.get('global_ids', [])
            if global_id in global_ids:
                object_frames.append(frame_idx)
        
        if len(object_frames) < 2:
            print(f"物体 {global_id}: 只出现在 {len(object_frames)} 帧中，无法聚合")
            return None
        
        # 选择中间帧作为参考帧
        middle_frame_idx = object_frames[len(object_frames) // 2]
        print(f"物体 {global_id}: 聚合到中间帧 {middle_frame_idx} (总帧数: {len(object_frames)})")
        
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
            middle_object_points_cpu = middle_object_points.cpu().numpy()
        else:
            middle_object_points_cpu = middle_object_points
            
        # 提取中间帧的颜色信息
        middle_colors = self._extract_colors_for_points(
            middle_result, middle_cluster_idx, middle_frame_idx, preds, vggt_batch)
        if middle_colors is None:
            # 如果无法提取颜色，使用默认颜色
            middle_colors = np.tile(
                (self._hsv_to_rgb((global_id * 137.5) % 360, 0.8, 0.9) * 255).astype(np.uint8),
                (len(middle_object_points_cpu), 1)
            )
        
        # 存储所有帧的变换
        transformations = {}
        aggregated_points = [middle_object_points_cpu]
        aggregated_colors = [middle_colors]
        
        # 对其他帧进行链式变换
        for frame_idx in object_frames:
            if frame_idx == middle_frame_idx:
                continue
            
            # 计算链式变换
            chain_transformation = self.compute_chain_transformation(
                frame_idx, middle_frame_idx, flows, clustering_results, preds, vggt_batch, global_id)
            
            if chain_transformation is not None:
                transformations[frame_idx] = {
                    'transformation': chain_transformation,
                    'R': chain_transformation[:3, :3],
                    't': chain_transformation[:3, 3],
                    'inlier_ratio': 1.0,  # 链式变换的内点比例需要重新计算
                    'num_correspondences': 0  # 链式变换的对应点数需要重新计算
                }
                
                # 变换当前帧的物体点云
                current_result = clustering_results[frame_idx]
                current_cluster_idx = current_result['global_ids'].index(global_id)
                current_object_mask = current_result['labels'] == current_cluster_idx
                current_object_points = current_result['points'][current_object_mask]
                
                # 应用变换
                transformed_points = self._apply_transformation(
                    current_object_points, chain_transformation)
                aggregated_points.append(transformed_points)
                
                # 提取当前帧的颜色信息
                current_colors = self._extract_colors_for_points(
                    current_result, current_cluster_idx, frame_idx, preds, vggt_batch)
                if current_colors is None:
                    # 如果无法提取颜色，使用默认颜色
                    current_colors = np.tile(
                        (self._hsv_to_rgb((global_id * 137.5) % 360, 0.8, 0.9) * 255).astype(np.uint8),
                        (len(current_object_points), 1)
                    )
                aggregated_colors.append(current_colors)
                
                print(f"  帧 {frame_idx} -> {middle_frame_idx}: 链式变换成功")
            else:
                print(f"  帧 {frame_idx} -> {middle_frame_idx}: 链式变换失败")
        
        if len(aggregated_points) < 2:
            print(f"物体 {global_id}: 没有成功的变换")
            return None
        
        # 合并所有点云和颜色
        all_points = np.concatenate(aggregated_points, axis=0)
        all_colors = np.concatenate(aggregated_colors, axis=0)
        
        return {
            'global_id': global_id,
            'middle_frame': middle_frame_idx,
            'object_frames': object_frames,
            'aggregated_points': all_points,
            'aggregated_colors': all_colors,
            'transformations': transformations,
            'num_frames': len(object_frames),
            'num_points': len(all_points)
        }
    
    def _apply_transformation(self, points: torch.Tensor, transformation: np.ndarray) -> np.ndarray:
        """应用变换矩阵到点云"""
        points_np = points.cpu().numpy()
        
        # 转换为齐次坐标
        points_homo = np.concatenate([points_np, np.ones((len(points_np), 1))], axis=1)
        
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
        output_path = os.path.join(output_dir, f"optical_flow_registration_results.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump({
                'aggregation_results': aggregation_results,
                'clustering_results': clustering_results,
                'preds': preds,
                'vggt_batch': vggt_batch,
                'flows': flows
            }, f)
        
        print(f"\n配准结果已保存到: {output_path}")
        print(f"成功处理 {len(aggregation_results)} 个物体")
        
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
            self._save_points_as_ply_with_colors(aggregated_points, pointcloud_path, global_id, aggregated_colors)
            
            print(f"  物体 {global_id}: 保存 {len(aggregated_points)} 个点到 {pointcloud_filename}")
        
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
    parser.add_argument("--use_pnp", action="store_true",
                       help="使用3DPnP方法（复杂版）")
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
        use_pnp=args.use_pnp,
        min_inliers_ratio=args.min_inliers_ratio,
        ransac_threshold=args.ransac_threshold,
        max_flow_magnitude=args.max_flow_magnitude,
        use_simple_correspondence=use_simple_correspondence,
        raft_model_path=args.raft_model_path
    )
    
    # 处理数据
    results = registration.process_pointcloud_data(args.data_path, args.output_dir)
    
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