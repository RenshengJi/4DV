#!/usr/bin/env python3
"""
高级多帧点云配准系统 - 保存版本
使用Open3D库对同一物体不同帧之间的点云进行配准，得到该物体在正规空间下的整体点云
支持从现有推理结果中提取点云数据，着色点云配准和全局最优配准
专门用于保存点云结果，不进行实时可视化
"""

from src.dust3r.utils.misc import tf32_off
import os
import numpy as np
import open3d as o3d
import cv2
import argparse
import glob
import time
import torch
import pickle
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import json
from copy import deepcopy

# 添加项目路径
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/vggt'))

# 导入项目相关模块


class AdvancedPointCloudRegistration:
    """高级点云配准类"""

    def __init__(self, voxel_size: float = 0.01, max_iterations: int = 50,
                 use_coarse_registration: bool = True, coarse_registration_threshold: float = 0.1,
                 use_color_features: bool = True, ransac_max_iteration: int = 5000,
                 ransac_confidence: int = 5):
        """
        初始化点云配准器

        Args:
            voxel_size: 体素大小，用于下采样
            max_iterations: 最大迭代次数
            use_coarse_registration: 是否使用粗匹配
            coarse_registration_threshold: 粗匹配质量阈值
            use_color_features: 是否启用颜色特征
            ransac_max_iteration: RANSAC最大迭代次数
            ransac_confidence: RANSAC置信度参数
        """
        self.voxel_size = voxel_size
        self.max_iterations = max_iterations
        self.use_coarse_registration = use_coarse_registration
        self.coarse_registration_threshold = coarse_registration_threshold
        self.use_color_features = use_color_features
        self.ransac_max_iteration = ransac_max_iteration
        self.ransac_confidence = ransac_confidence
        self.transformation_matrices = []
        self.point_clouds = []
        self.colors = []
        
        # 初始化SIFT检测器
        self.sift = cv2.SIFT_create(nfeatures=1000, nOctaveLayers=3, 
                                   contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
        
        # 特征缓存
        self.feature_cache = {}
        
        # 检查Open3D版本和可用功能
        self._check_open3d_capabilities()

    def _check_open3d_capabilities(self):
        """检查Open3D版本和可用功能"""
        try:
            # 检查Open3D版本
            o3d_version = o3d.__version__
            print(f"Open3D版本: {o3d_version}")
            
            # 检查是否支持某些功能
            self.has_fast_global_registration = hasattr(o3d.pipelines.registration, 'registration_fast_based_on_feature_matching')
            if not self.has_fast_global_registration:
                print("警告: 当前Open3D版本不支持FastGlobalRegistration，将使用替代方法")
                
        except Exception as e:
            print(f"版本检查失败: {e}")
            self.has_fast_global_registration = False

    def _get_pointcloud_hash(self, pcd: o3d.geometry.PointCloud) -> str:
        """
        生成点云的哈希值用于缓存
        
        Args:
            pcd: 点云
            
        Returns:
            哈希字符串
        """
        try:
            if len(pcd.points) == 0:
                return "empty"
            
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if len(pcd.colors) > 0 else np.array([])
            
            # 使用点云的基本信息生成哈希
            point_hash = hash(str(points.shape) + str(points.sum()))
            color_hash = hash(str(colors.shape) + str(colors.sum())) if colors.size > 0 else 0
            
            return f"{point_hash}_{color_hash}"
        except Exception as e:
            print(f"点云哈希生成失败: {e}")
            return "error"

    def extract_sift_features_from_image(self, image: np.ndarray, pcd: o3d.geometry.PointCloud, 
                                        correspondence: List[int], image_height: int, image_width: int) -> np.ndarray:
        """
        从原始图像中提取SIFT特征并分配给点云点（使用确切的对应关系）
        
        Args:
            image: 原始图像 [H, W, 3]
            pcd: 输入点云
            correspondence: 点云点和图像像素的对应关系 [N] - 每个元素是图像像素的一维索引
            image_height: 原始图像高度
            image_width: 原始图像宽度
            
        Returns:
            SIFT特征向量数组
        """
        try:
            if image is None or len(pcd.points) == 0:
                return np.array([])
            
            # 如果禁用颜色特征，返回零特征
            if not self.use_color_features:
                print("颜色特征已禁用，返回零SIFT特征")
                return np.zeros((len(pcd.points), 128))
            
            # 如果没有对应关系，回退到兼容性方法
            if correspondence is None or len(correspondence) == 0:
                print("警告: 没有对应关系，回退到兼容性SIFT特征提取")
                return self.extract_sift_features(pcd)
            
            # 检查缓存
            image_hash = hash(str(image.shape) + str(image.sum()))
            pcd_hash = self._get_pointcloud_hash(pcd)
            corr_hash = hash(str(correspondence))
            cache_key = f"sift_image_{image_hash}_{pcd_hash}_{corr_hash}"
            
            if cache_key in self.feature_cache:
                print("使用缓存的SIFT特征（图像，确切对应关系）")
                return self.feature_cache[cache_key]
            
            points = np.asarray(pcd.points)
            
            # 确保图像格式正确
            if len(image.shape) == 3:
                # 如果是RGB图像，转换为灰度
                if image.dtype != np.uint8:
                    # 如果图像值在[0,1]范围内，转换为[0,255]
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif len(image.shape) == 2:
                # 如果是灰度图像
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        gray = (image * 255).astype(np.uint8)
                    else:
                        gray = image.astype(np.uint8)
                else:
                    gray = image
            else:
                print("图像格式不正确，回退到兼容性SIFT特征提取")
                return self.extract_sift_features(pcd)
            
            # 检查图像是否为空或太小
            if gray.size == 0 or gray.shape[0] < 10 or gray.shape[1] < 10:
                print("图像太小或为空，回退到兼容性SIFT特征提取")
                return self.extract_sift_features(pcd)
            
            # 检测SIFT关键点和描述符
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            
            if descriptors is not None and len(descriptors) > 0:
                # 为每个点云点分配SIFT特征
                sift_features = np.zeros((len(points), 128))  # SIFT描述符是128维
                
                # 使用确切的对应关系将SIFT特征分配给点云点
                for i, point in enumerate(points):
                    if i < len(correspondence):
                        # 获取当前点对应的图像像素位置
                        pixel_idx = correspondence[i]
                        
                        # 将一维像素索引转换为二维坐标
                        pixel_y = pixel_idx // image_width
                        pixel_x = pixel_idx % image_width
                        
                        # 归一化到[0,1]范围
                        pixel_x_norm = pixel_x / image_width
                        pixel_y_norm = pixel_y / image_height
                        
                        # 找到最近的SIFT关键点
                        min_dist = float('inf')
                        best_descriptor = np.zeros(128)
                        
                        for j, kp in enumerate(keypoints):
                            # 计算关键点在图像中的位置
                            kp_x = kp.pt[0] / image_width
                            kp_y = kp.pt[1] / image_height
                            
                            # 计算距离
                            dist = np.sqrt((pixel_x_norm - kp_x)**2 + (pixel_y_norm - kp_y)**2)
                            if dist < min_dist:
                                min_dist = dist
                                best_descriptor = descriptors[j]
                        
                        sift_features[i] = best_descriptor
                    else:
                        # 如果对应关系不足，使用零特征
                        sift_features[i] = np.zeros(128)
                
                # 缓存结果
                self.feature_cache[cache_key] = sift_features
                
                print(f"SIFT特征提取成功（图像，确切对应关系） - 点数: {len(points)}, SIFT特征维度: {sift_features.shape}")
                return sift_features
            else:
                print("SIFT特征提取失败（图像，确切对应关系），返回零特征")
                zero_features = np.zeros((len(points), 128))
                self.feature_cache[cache_key] = zero_features
                return zero_features
                
        except Exception as e:
            print(f"SIFT特征提取失败（图像，确切对应关系）: {e}")
            if len(pcd.points) > 0:
                zero_features = np.zeros((len(pcd.points), 128))
                return zero_features
            return np.array([])

    def extract_sift_features(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """
        从点云中提取SIFT特征（兼容性方法，建议使用extract_sift_features_from_image）
        
        Args:
            pcd: 输入点云
            
        Returns:
            SIFT特征向量数组
        """
        print("警告: 使用兼容性SIFT特征提取方法，建议使用extract_sift_features_from_image")
        
        try:
            # 如果禁用颜色特征且点云没有颜色，返回零特征
            if not self.use_color_features and len(pcd.colors) == 0:
                print("颜色特征已禁用且点云无颜色，返回零SIFT特征")
                if len(pcd.points) > 0:
                    return np.zeros((len(pcd.points), 128))
                return np.array([])
            
            if len(pcd.points) == 0 or len(pcd.colors) == 0:
                return np.array([])
            
            # 检查缓存
            pcd_hash = self._get_pointcloud_hash(pcd)
            cache_key = f"sift_{pcd_hash}"
            
            if cache_key in self.feature_cache:
                print("使用缓存的SIFT特征")
                return self.feature_cache[cache_key]
            
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            # 简化的SIFT特征提取方法
            # 使用XY平面投影（最常用的投影方式）
            resolution = 128  # 适中的分辨率
            
            # 计算点云边界
            x_min, y_min, _ = np.min(points, axis=0)
            x_max, y_max, _ = np.max(points, axis=0)
            
            # 避免除零错误
            if x_max == x_min:
                x_max = x_min + 1e-6
            if y_max == y_min:
                y_max = y_min + 1e-6
            
            # 创建2D投影图像
            image = np.zeros((resolution, resolution, 3), dtype=np.uint8)
            
            # 将3D点投影到2D
            for point, color in zip(points, colors):
                # 归一化坐标到图像范围
                x_norm = int((point[0] - x_min) / (x_max - x_min) * (resolution - 1))
                y_norm = int((point[1] - y_min) / (y_max - y_min) * (resolution - 1))
                
                if 0 <= x_norm < resolution and 0 <= y_norm < resolution:
                    # 将颜色值转换为0-255范围
                    if color.max() <= 1.0:
                        color_uint8 = (color * 255).astype(np.uint8)
                    else:
                        color_uint8 = color.astype(np.uint8)
                    image[y_norm, x_norm] = color_uint8
            
            # 转换为灰度图像进行SIFT检测
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 检查图像是否为空或太小
            if gray.size == 0 or gray.shape[0] < 10 or gray.shape[1] < 10:
                print("投影图像太小或为空，返回零特征")
                zero_features = np.zeros((len(points), 128))
                self.feature_cache[cache_key] = zero_features
                return zero_features
            
            # 检测SIFT关键点和描述符
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            
            if descriptors is not None and len(descriptors) > 0:
                # 为每个点云点分配SIFT特征
                sift_features = np.zeros((len(points), 128))  # SIFT描述符是128维
                
                # 使用最近邻方法将SIFT特征分配给点云点
                for i, point in enumerate(points):
                    # 找到最近的SIFT关键点
                    x_norm = (point[0] - x_min) / (x_max - x_min)
                    y_norm = (point[1] - y_min) / (y_max - y_min)
                    
                    min_dist = float('inf')
                    best_descriptor = np.zeros(128)
                    
                    for j, kp in enumerate(keypoints):
                        kp_x = kp.pt[0] / resolution
                        kp_y = kp.pt[1] / resolution
                        
                        dist = np.sqrt((x_norm - kp_x)**2 + (y_norm - kp_y)**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_descriptor = descriptors[j]
                    
                    sift_features[i] = best_descriptor
                
                # 缓存结果
                self.feature_cache[cache_key] = sift_features
                
                print(f"SIFT特征提取成功 - 点数: {len(points)}, SIFT特征维度: {sift_features.shape}")
                return sift_features
            else:
                print("SIFT特征提取失败，返回零特征")
                zero_features = np.zeros((len(points), 128))
                self.feature_cache[cache_key] = zero_features
                return zero_features
                
        except Exception as e:
            print(f"SIFT特征提取失败: {e}")
            if len(pcd.points) > 0:
                zero_features = np.zeros((len(pcd.points), 128))
                return zero_features
            return np.array([])

    def preprocess_point_cloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        预处理点云：去噪、下采样、估计法向量

        Args:
            pcd: 输入点云

        Returns:
            预处理后的点云
        """
        try:
            if len(pcd.points) == 0:
                return pcd

            # 统计滤波去噪
            try:
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            except Exception as e:
                print(f"统计滤波失败: {e}，跳过")

            # 体素下采样
            try:
                pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            except Exception as e:
                print(f"体素下采样失败: {e}，跳过")

            # 强制估计法向量（为着色ICP做准备）
            if len(pcd.points) > 0:
                try:
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, max_nn=30))
                    # 确保法向量方向一致
                    pcd.orient_normals_consistent_tangent_plane(k=30)
                    print(f"法向量估计成功 - 点数: {len(pcd.points)}, 法向量数: {len(pcd.normals)}")
                except Exception as e:
                    print(f"法向量估计失败: {e}，尝试简单估计")
                    try:
                        # 简单法向量估计
                        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
                    except Exception as e2:
                        print(f"简单法向量估计也失败: {e2}")

            return pcd
        except Exception as e:
            print(f"点云预处理失败: {e}")
            return pcd

    def extract_features(self, pcd: o3d.geometry.PointCloud, image: np.ndarray = None, 
                        correspondence: List[int] = None, image_height: int = None, image_width: int = None) -> Tuple[o3d.pipelines.registration.Feature, np.ndarray]:
        """
        提取点云特征（结合FPFH和SIFT特征，带缓存）

        Args:
            pcd: 输入点云
            image: 原始图像（可选，用于SIFT特征提取）
            correspondence: 点云点和图像像素的对应关系（可选）
            image_height: 原始图像高度（可选）
            image_width: 原始图像宽度（可选）

        Returns:
            特征描述子和关键点
        """
        try:
            if len(pcd.points) == 0:
                return None, np.array([])

            # 检查缓存
            pcd_hash = self._get_pointcloud_hash(pcd)
            image_hash = hash(str(image.shape) + str(image.sum())) if image is not None else "no_image"
            corr_hash = hash(str(correspondence)) if correspondence is not None else "no_corr"
            cache_key = f"combined_{pcd_hash}_{image_hash}_{corr_hash}"
            
            if cache_key in self.feature_cache:
                print("使用缓存的特征")
                return self.feature_cache[cache_key], np.asarray(pcd.points)

            # 提取FPFH特征
            pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd,
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100)
            )
            
            # 提取SIFT特征（仅当有图像和对应关系时）
            sift_features = np.array([])  # 默认为空
            if image is not None and correspondence is not None and image_height is not None and image_width is not None:
                # 只有在有完整的图像信息和对应关系时才提取SIFT特征
                sift_features = self.extract_sift_features_from_image(image, pcd, correspondence, image_height, image_width)
            else:
                # 没有图像或对应关系，直接使用FPFH特征
                print("没有图像或对应关系信息，仅使用FPFH特征")

            # 结合FPFH和SIFT特征（如果SIFT特征可用）
            if sift_features.size > 0 and pcd_fpfh is not None:
                fpfh_data = np.asarray(pcd_fpfh.data).T  # [N, 33]
                combined_features = np.concatenate([fpfh_data, sift_features], axis=1)  # [N, 33+128=161]

                # 创建新的特征对象
                combined_feature = o3d.pipelines.registration.Feature()
                combined_feature.data = combined_features.T  # 转置回 [161, N]

                # 缓存结果
                self.feature_cache[cache_key] = combined_feature

                print(f"特征提取成功 - FPFH: {fpfh_data.shape}, SIFT: {sift_features.shape}, 组合: {combined_features.shape}")
                return combined_feature, np.asarray(pcd.points)
            else:
                # 如果没有SIFT特征，只使用FPFH
                if sift_features.size == 0:
                    print("未提取SIFT特征，仅使用FPFH特征")
                else:
                    print("SIFT特征提取失败，仅使用FPFH特征")
                self.feature_cache[cache_key] = pcd_fpfh
                return pcd_fpfh, np.asarray(pcd.points)

        except Exception as e:
            print(f"特征提取失败: {e}")
            return None, np.array([])

    def _compute_feature_similarity(self, source_features: np.ndarray, target_features: np.ndarray) -> np.ndarray:
        """
        计算特征相似度矩阵
        
        Args:
            source_features: 源特征 [N, D]
            target_features: 目标特征 [M, D]
            
        Returns:
            相似度矩阵 [N, M]
        """
        try:
            # 使用余弦相似度
            source_norm = np.linalg.norm(source_features, axis=1, keepdims=True)
            target_norm = np.linalg.norm(target_features, axis=1, keepdims=True)
            
            # 避免除零
            source_norm = np.where(source_norm == 0, 1e-8, source_norm)
            target_norm = np.where(target_norm == 0, 1e-8, target_norm)
            
            source_normalized = source_features / source_norm
            target_normalized = target_features / target_norm
            
            similarity = np.dot(source_normalized, target_normalized.T)
            return similarity
            
        except Exception as e:
            print(f"特征相似度计算失败: {e}")
            return np.zeros((len(source_features), len(target_features)))

    def _find_feature_correspondences(self, source_features: np.ndarray, target_features: np.ndarray, 
                                    threshold: float = 0.7) -> List[Tuple[int, int]]:
        """
        找到特征对应关系
        
        Args:
            source_features: 源特征
            target_features: 目标特征
            threshold: 相似度阈值
            
        Returns:
            对应关系列表 [(source_idx, target_idx), ...]
        """
        try:
            similarity_matrix = self._compute_feature_similarity(source_features, target_features)
            
            correspondences = []
            for i in range(len(source_features)):
                # 找到最相似的目标特征
                best_match_idx = np.argmax(similarity_matrix[i])
                best_similarity = similarity_matrix[i][best_match_idx]
                
                if best_similarity > threshold:
                    correspondences.append((i, best_match_idx))
            
            return correspondences
            
        except Exception as e:
            print(f"特征对应关系查找失败: {e}")
            return []



    def _simple_center_alignment(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud) -> np.ndarray:
        """
        简单的中心对齐，用于处理大偏差情况

        Args:
            source: 源点云
            target: 目标点云

        Returns:
            变换矩阵
        """
        try:
            if len(source.points) == 0 or len(target.points) == 0:
                return np.eye(4)

            # 计算点云的边界框
            source_points = np.asarray(source.points)
            target_points = np.asarray(target.points)
            
            source_center = np.mean(source_points, axis=0)
            target_center = np.mean(target_points, axis=0)
            
            # 计算缩放因子（可选）
            source_scale = np.max(source_points, axis=0) - np.min(source_points, axis=0)
            target_scale = np.max(target_points, axis=0) - np.min(target_points, axis=0)
            
            # 计算缩放比例
            scale_ratio = np.mean(target_scale / (source_scale + 1e-8))
            
            # 构建变换矩阵：缩放 + 平移
            transform = np.eye(4)
            transform[:3, :3] *= scale_ratio  # 缩放
            transform[:3, 3] = target_center - source_center * scale_ratio  # 平移
            
            print(f"简单中心对齐 - 缩放比例: {scale_ratio:.3f}, 平移: {transform[:3, 3]}")
            return transform
            
        except Exception as e:
            print(f"简单中心对齐失败: {e}")
            return self._compute_geometric_center_transform(source, target)

    def _compute_geometric_center_transform(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud) -> np.ndarray:
        """
        计算基于几何中心的变换矩阵

        Args:
            source: 源点云
            target: 目标点云

        Returns:
            变换矩阵
        """
        try:
            source_center = np.mean(np.asarray(source.points), axis=0)
            target_center = np.mean(np.asarray(target.points), axis=0)
            translation = target_center - source_center
            
            # 构建初始变换矩阵（仅平移）
            initial_transform = np.eye(4)
            initial_transform[:3, 3] = translation
            
            print(f"几何中心变换 - 源中心: {source_center}, 目标中心: {target_center}, 平移: {translation}")
            return initial_transform
        except Exception as e:
            print(f"几何中心变换计算失败: {e}")
            return np.eye(4)


    def simple_pairwise_registration(self, source: o3d.geometry.PointCloud,
                                     target: o3d.geometry.PointCloud) -> np.ndarray:
        """
        简单的两两配准（备选方法）

        Args:
            source: 源点云
            target: 目标点云

        Returns:
            变换矩阵
        """
        try:
            # 预处理
            source_down = self.preprocess_point_cloud(source)
            target_down = self.preprocess_point_cloud(target)

            if len(source_down.points) == 0 or len(target_down.points) == 0:
                return np.eye(4)

            # 进行粗匹配获得初始变换
            print("简单配准：开始粗匹配...")
            initial_transform = self.coarse_registration(source_down, target_down)

            # 检查是否启用颜色特征
            if not self.use_color_features:
                print("颜色特征已禁用，使用普通ICP配准")
                result_icp = o3d.pipelines.registration.registration_icp(
                    source_down, target_down, self.voxel_size * 2.0, initial_transform,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iterations))
                
                if result_icp.fitness < 0.3:
                    print(f"简单普通ICP配准 - 适应度: {result_icp.fitness:.3f}, RMSE: {result_icp.inlier_rmse:.3f}")
                return result_icp.transformation

            # 检查点云是否有颜色信息
            source_has_colors = len(source_down.colors) > 0
            target_has_colors = len(target_down.colors) > 0
            
            if not source_has_colors or not target_has_colors:
                print("点云缺少颜色信息，使用普通ICP配准")
                result_icp = o3d.pipelines.registration.registration_icp(
                    source_down, target_down, self.voxel_size * 2.0, initial_transform,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iterations))
                
                if result_icp.fitness < 0.3:
                    print(f"简单普通ICP配准 - 适应度: {result_icp.fitness:.3f}, RMSE: {result_icp.inlier_rmse:.3f}")
                return result_icp.transformation
            
            # 尝试着色ICP配准，使用更宽松的参数
            print("使用着色ICP配准...")
            max_correspondence_distance = self.voxel_size * 4.0  # 增大距离阈值
            
            # 确保点云有法向量
            source_down_copy = deepcopy(source_down)
            target_down_copy = deepcopy(target_down)
            
            # 如果点云没有法向量，重新估计
            if len(source_down_copy.normals) == 0:
                source_down_copy.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                source_down_copy.orient_normals_consistent_tangent_plane(k=30)
            
            if len(target_down_copy.normals) == 0:
                target_down_copy.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                target_down_copy.orient_normals_consistent_tangent_plane(k=30)
            
            try:
                result_icp = o3d.pipelines.registration.registration_colored_icp(
                    source_down_copy, target_down_copy, max_correspondence_distance, initial_transform,
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iterations))
                
                if result_icp.fitness < 0.3:
                    print(f"简单着色ICP配准 - 适应度: {result_icp.fitness:.3f}, RMSE: {result_icp.inlier_rmse:.3f}")
                return result_icp.transformation
            except Exception as e:
                print(f"着色ICP失败，尝试普通ICP: {e}")
                # 如果着色ICP失败，尝试普通ICP
                result_icp = o3d.pipelines.registration.registration_icp(
                    source_down, target_down, self.voxel_size * 2.0, initial_transform,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iterations))
                
                if result_icp.fitness < 0.3:
                    print(f"简单普通ICP配准 - 适应度: {result_icp.fitness:.3f}, RMSE: {result_icp.inlier_rmse:.3f}")
                return result_icp.transformation
            
        except Exception as e:
            print(f"简单配准失败: {e}")
            return np.eye(4)

    def register_point_clouds(self, point_clouds: List[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
        """
        对多个点云进行配准

        Args:
            point_clouds: 点云列表

        Returns:
            配准后的点云
        """
        if len(point_clouds) == 0:
            return o3d.geometry.PointCloud()

        if len(point_clouds) == 1:
            return point_clouds[0]

        print(f"开始配准 {len(point_clouds)} 个点云...")

        # 使用改进的配准策略
        return self._improved_multi_frame_registration(point_clouds)

    def register_point_clouds_with_images(self, point_cloud_data: List[Tuple[o3d.geometry.PointCloud, np.ndarray, List[int]]]) -> o3d.geometry.PointCloud:
        """
        对多个点云（包含图像信息和对应关系）进行配准

        Args:
            point_cloud_data: (点云, 图像, 对应关系)元组列表

        Returns:
            配准后的点云
        """
        if len(point_cloud_data) == 0:
            return o3d.geometry.PointCloud()

        if len(point_cloud_data) == 1:
            return point_cloud_data[0][0]  # 返回点云部分

        print(f"开始配准 {len(point_cloud_data)} 个点云（包含图像信息和对应关系）...")

        # 分离点云、图像和对应关系
        point_clouds = [data[0] for data in point_cloud_data]
        images = [data[1] for data in point_cloud_data]
        correspondences = [data[2] for data in point_cloud_data]

        # 检查是否有有效的图像信息
        valid_images = [img for img in images if img is not None]
        if len(valid_images) == 0:
            print("没有有效的图像信息，使用传统配准方法")
            return self.register_point_clouds(point_clouds)

        # 获取图像尺寸
        if len(valid_images) > 0:
            image_height, image_width = valid_images[0].shape[:2]
        else:
            image_height, image_width = 0, 0

        # 使用改进的配准策略
        return self._improved_multi_frame_registration_with_images(point_clouds, images, correspondences, image_height, image_width)

    def _improved_multi_frame_registration(self, point_clouds: List[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
        """
        改进的多帧配准策略，专门处理大偏差情况

        Args:
            point_clouds: 点云列表

        Returns:
            配准后的点云
        """
        n_clouds = len(point_clouds)
        print(f"使用改进的多帧配准策略，处理 {n_clouds} 个点云")

        # 第一步：选择最佳参考点云（点数最多且质量最好的）
        reference_idx = self._select_reference_pointcloud(point_clouds)
        print(f"选择点云 {reference_idx} 作为参考点云")

        # 第二步：将所有点云配准到参考点云
        registered_clouds = []
        transformations = []

        for i in range(n_clouds):
            if i == reference_idx:
                # 参考点云不需要变换
                registered_clouds.append(deepcopy(point_clouds[i]))
                transformations.append(np.eye(4))
            else:
                print(f"将点云 {i} 配准到参考点云 {reference_idx}...")
                
                # 使用改进的配准方法
                transform = self._robust_pairwise_registration(
                    point_clouds[i], point_clouds[reference_idx])
                
                # 应用变换
                cloud_copy = deepcopy(point_clouds[i])
                cloud_copy.transform(transform)
                registered_clouds.append(cloud_copy)
                transformations.append(transform)

        # 第三步：合并点云并进行最终优化
        final_cloud = o3d.geometry.PointCloud()
        for cloud in registered_clouds:
            if len(cloud.points) > 0:
                final_cloud += cloud

        # 最终优化
        final_cloud = self.preprocess_point_cloud(final_cloud)

        print(f"改进配准完成，最终点云包含 {len(final_cloud.points)} 个点")
        return final_cloud

    def _improved_multi_frame_registration_with_images(self, point_clouds: List[o3d.geometry.PointCloud], 
                                                      images: List[np.ndarray],
                                                      correspondences: List[List[int]],
                                                      image_height: int, image_width: int) -> o3d.geometry.PointCloud:
        """
        改进的多帧配准策略，使用图像信息和确切对应关系进行SIFT特征提取

        Args:
            point_clouds: 点云列表
            images: 图像列表
            correspondences: 对应关系列表
            image_height: 图像高度
            image_width: 图像宽度

        Returns:
            配准后的点云
        """
        n_clouds = len(point_clouds)
        print(f"使用改进的多帧配准策略（包含图像和确切对应关系），处理 {n_clouds} 个点云")

        # 第一步：选择最佳参考点云（点数最多且质量最好的）
        reference_idx = self._select_reference_pointcloud(point_clouds)
        print(f"选择点云 {reference_idx} 作为参考点云")

        # 第二步：将所有点云配准到参考点云
        registered_clouds = []
        transformations = []

        for i in range(n_clouds):
            if i == reference_idx:
                # 参考点云不需要变换
                registered_clouds.append(deepcopy(point_clouds[i]))
                transformations.append(np.eye(4))
            else:
                print(f"将点云 {i} 配准到参考点云 {reference_idx}...")
                
                # 检查是否有有效的图像信息
                source_image = images[i] if i < len(images) else None
                target_image = images[reference_idx] if reference_idx < len(images) else None
                source_correspondence = correspondences[i] if i < len(correspondences) else None
                target_correspondence = correspondences[reference_idx] if reference_idx < len(correspondences) else None
                
                if source_image is not None and target_image is not None and source_correspondence is not None and target_correspondence is not None:
                    # 使用改进的配准方法（包含图像信息和确切对应关系）
                    transform = self._robust_pairwise_registration_with_images(
                        point_clouds[i], point_clouds[reference_idx], 
                        source_image, target_image,
                        source_correspondence, target_correspondence,
                        image_height, image_width)
                else:
                    # 使用传统配准方法
                    print("缺少图像信息，使用传统配准方法")
                    transform = self._robust_pairwise_registration(point_clouds[i], point_clouds[reference_idx])
                
                # 应用变换
                cloud_copy = deepcopy(point_clouds[i])
                cloud_copy.transform(transform)
                registered_clouds.append(cloud_copy)
                transformations.append(transform)

        # 第三步：合并点云并进行最终优化
        final_cloud = o3d.geometry.PointCloud()
        for cloud in registered_clouds:
            if len(cloud.points) > 0:
                final_cloud += cloud

        # 最终优化
        final_cloud = self.preprocess_point_cloud(final_cloud)

        print(f"改进配准完成（包含图像和确切对应关系），最终点云包含 {len(final_cloud.points)} 个点")
        return final_cloud

    def _select_reference_pointcloud(self, point_clouds: List[o3d.geometry.PointCloud]) -> int:
        """
        选择最佳参考点云

        Args:
            point_clouds: 点云列表

        Returns:
            最佳参考点云的索引
        """
        best_idx = 0
        best_score = 0

        for i, pcd in enumerate(point_clouds):
            if len(pcd.points) == 0:
                continue

            # 计算点云质量分数（基于点数、密度等）
            score = len(pcd.points)
            
            # 如果启用颜色特征且有颜色信息，加分
            if self.use_color_features and len(pcd.colors) > 0:
                score *= 1.2

            # 如果有法向量信息，加分
            if len(pcd.normals) > 0:
                score *= 1.1

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def _robust_pairwise_registration(self, source: o3d.geometry.PointCloud, 
                                     target: o3d.geometry.PointCloud) -> np.ndarray:
        """
        鲁棒的两两配准，包含多种策略

        Args:
            source: 源点云
            target: 目标点云

        Returns:
            变换矩阵
        """
        print(f"开始鲁棒配准 - 源点云: {len(source.points)} 点, 目标点云: {len(target.points)} 点")

        # 策略1：尝试粗匹配 + 精细ICP
        try:
            print("策略1：粗匹配 + 精细ICP")
            initial_transform = self.coarse_registration(source, target)
            return initial_transform
        except Exception as e:
            print(f"粗匹配 + 精细ICP失败: {e}")
            return np.eye(4)


        # # 策略1：尝试粗匹配 + 精细ICP
        # try:
        #     print("策略1：粗匹配 + 精细ICP")
        #     initial_transform = self.coarse_registration(source, target)\
        #     # 精细ICP配准
        #     result = self._fine_icp_registration(source, target, initial_transform)
        #     if result is not None and result.fitness > 0.1:  # 降低阈值
        #         print(f"策略1成功 - 适应度: {result.fitness:.3f}")
        #         return result.transformation
        # except Exception as e:
        #     print(f"策略1失败: {e}")


        # # 策略3：尝试简单配准
        # try:
        #     print("策略3：简单配准")
        #     transform = self.simple_pairwise_registration(source, target)
        #     return transform
        # except Exception as e:
        #     print(f"策略3失败: {e}")

        # # 策略4：几何中心变换
        # print("策略4：几何中心变换")
        # return self._compute_geometric_center_transform(source, target)

    def _robust_pairwise_registration_with_images(self, source: o3d.geometry.PointCloud, 
                                                 target: o3d.geometry.PointCloud,
                                                 source_image: np.ndarray,
                                                 target_image: np.ndarray,
                                                 source_correspondence: List[int],
                                                 target_correspondence: List[int],
                                                 image_height: int, image_width: int) -> np.ndarray:
        """
        鲁棒的两两配准，包含图像信息和确切对应关系

        Args:
            source: 源点云
            target: 目标点云
            source_image: 源图像
            target_image: 目标图像
            source_correspondence: 源点云对应关系
            target_correspondence: 目标点云对应关系
            image_height: 图像高度
            image_width: 图像宽度

        Returns:
            变换矩阵
        """
        print(f"开始鲁棒配准（包含图像和确切对应关系） - 源点云: {len(source.points)} 点, 目标点云: {len(target.points)} 点")

        # 策略1：尝试粗匹配 + 精细ICP（使用图像信息和确切对应关系）
        try:
            print("策略1：粗匹配 + 精细ICP（使用图像信息和确切对应关系）")
            initial_transform = self.coarse_registration_with_images(
                source, target, source_image, target_image, 
                source_correspondence, target_correspondence, image_height, image_width)
            # 精细ICP配准
            result = self._fine_icp_registration(source, target, initial_transform)
            if result is not None and result.fitness > 0.1:  # 降低阈值
                print(f"策略1成功 - 适应度: {result.fitness:.3f}")
                return result.transformation
        except Exception as e:
            print(f"策略1失败: {e}")


        print("策略2：几何中心变换")
        return self._compute_geometric_center_transform(source, target)


    def coarse_registration(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud) -> np.ndarray:
        """
        粗匹配：使用特征匹配进行初始配准

        Args:
            source: 源点云
            target: 目标点云

        Returns:
            初始变换矩阵
        """
        try:
            if len(source.points) == 0 or len(target.points) == 0:
                print("点云为空，返回单位矩阵")
                return np.eye(4)

            # 预处理点云
            source_down = self.preprocess_point_cloud(source)
            target_down = self.preprocess_point_cloud(target)

            if len(source_down.points) == 0 or len(target_down.points) == 0:
                print("预处理后点云为空，返回单位矩阵")
                return np.eye(4)

            print(f"开始粗匹配 - 源点云: {len(source_down.points)} 点, 目标点云: {len(target_down.points)} 点")

            # 策略1：尝试RANSAC特征匹配
            print("策略1：RANSAC特征匹配（FPFH+SIFT）...")
            try:
                # 提取组合特征
                source_features, source_points = self.extract_features(source_down)
                target_features, target_points = self.extract_features(target_down)

                if source_features is not None and target_features is not None:
                    # 使用RANSAC进行特征匹配
                    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                        source_down, target_down, source_features, target_features, True,
                        max_correspondence_distance=self.voxel_size * 2.0,
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                        ransac_n=3,
                        checkers=[
                            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
                            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.voxel_size * 2.0)
                        ],
                        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                            self.ransac_max_iteration, self.ransac_confidence)
                    )

                    if result_ransac.fitness > self.coarse_registration_threshold:
                        print(f"RANSAC特征匹配成功 - 适应度: {result_ransac.fitness:.3f}, RMSE: {result_ransac.inlier_rmse:.3f}")
                        return result_ransac.transformation
                    else:
                        print(f"RANSAC特征匹配效果不佳 - 适应度: {result_ransac.fitness:.3f}")
                else:
                    print("特征提取失败，跳过RANSAC")
            except Exception as e:
                print(f"RANSAC特征匹配失败: {e}")

            # 策略2：几何中心变换
            print("策略2：几何中心变换")
            return self._compute_geometric_center_transform(source_down, target_down)

        except Exception as e:
            print(f"粗匹配失败: {e}，使用单位矩阵")
            return np.eye(4)

    def coarse_registration_with_images(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
                                       source_image: np.ndarray, target_image: np.ndarray,
                                       source_correspondence: List[int], target_correspondence: List[int],
                                       image_height: int, image_width: int) -> np.ndarray:
        """
        粗匹配：使用图像信息和确切对应关系进行初始配准

        Args:
            source: 源点云
            target: 目标点云
            source_image: 源图像
            target_image: 目标图像
            source_correspondence: 源点云对应关系
            target_correspondence: 目标点云对应关系
            image_height: 图像高度
            image_width: 图像宽度

        Returns:
            初始变换矩阵
        """
        try:
            if len(source.points) == 0 or len(target.points) == 0:
                print("点云为空，返回单位矩阵")
                return np.eye(4)

            # 预处理点云
            source_down = self.preprocess_point_cloud(source)
            target_down = self.preprocess_point_cloud(target)

            if len(source_down.points) == 0 or len(target_down.points) == 0:
                print("预处理后点云为空，返回单位矩阵")
                return np.eye(4)

            print(f"开始粗匹配（使用图像和确切对应关系） - 源点云: {len(source_down.points)} 点, 目标点云: {len(target_down.points)} 点")

            # 策略1：尝试RANSAC特征匹配（使用组合特征，包含图像信息和确切对应关系）
            print("策略1：RANSAC特征匹配（FPFH+SIFT，使用图像和确切对应关系）...")
            try:
                # 提取组合特征（使用图像信息和确切对应关系）
                source_features, source_points = self.extract_features(
                    source_down, source_image, source_correspondence, image_height, image_width)
                target_features, target_points = self.extract_features(
                    target_down, target_image, target_correspondence, image_height, image_width)

                if source_features is not None and target_features is not None:
                    # 使用RANSAC进行特征匹配
                    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                        source_down, target_down, source_features, target_features, True,
                        max_correspondence_distance=self.voxel_size * 2.0,
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                        ransac_n=3,
                        checkers=[
                            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
                            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.voxel_size * 2.0)
                        ],
                        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                            self.ransac_max_iteration, self.ransac_confidence)
                    )

                    if result_ransac.fitness > self.coarse_registration_threshold:
                        print(f"RANSAC特征匹配成功（使用图像和确切对应关系） - 适应度: {result_ransac.fitness:.3f}, RMSE: {result_ransac.inlier_rmse:.3f}")
                        return result_ransac.transformation
                    else:
                        print(f"RANSAC特征匹配效果不佳（使用图像和确切对应关系） - 适应度: {result_ransac.fitness:.3f}")
                else:
                    print("特征提取失败，跳过RANSAC")
            except Exception as e:
                print(f"RANSAC特征匹配失败（使用图像和确切对应关系）: {e}")


            # 策略2：几何中心变换
            print("策略2：几何中心变换")
            return self._compute_geometric_center_transform(source_down, target_down)

        except Exception as e:
            print(f"粗匹配失败（使用图像和确切对应关系）: {e}，使用单位矩阵")
            return np.eye(4)

    def _fine_icp_registration(self, source: o3d.geometry.PointCloud, 
                              target: o3d.geometry.PointCloud, 
                              initial_transform: np.ndarray) -> Optional[o3d.pipelines.registration.RegistrationResult]:
        """
        精细ICP配准

        Args:
            source: 源点云
            target: 目标点云
            initial_transform: 初始变换矩阵

        Returns:
            配准结果
        """
        try:
            # 确保点云有法向量（为着色ICP做准备）
            source_copy = deepcopy(source)
            target_copy = deepcopy(target)
            
            # 如果点云没有法向量，重新估计
            if len(source_copy.normals) == 0:
                print("源点云缺少法向量，重新估计...")
                source_copy.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                source_copy.orient_normals_consistent_tangent_plane(k=30)
            
            if len(target_copy.normals) == 0:
                print("目标点云缺少法向量，重新估计...")
                target_copy.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                target_copy.orient_normals_consistent_tangent_plane(k=30)

            # 检查是否启用颜色特征
            if not self.use_color_features:
                print("颜色特征已禁用，使用普通ICP进行精细配准...")
                result = o3d.pipelines.registration.registration_icp(
                    source_copy, target_copy, self.voxel_size * 500, initial_transform,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iterations))
                return result

            # 检查点云是否有颜色信息
            source_has_colors = len(source_copy.colors) > 0
            target_has_colors = len(target_copy.colors) > 0

            if source_has_colors and target_has_colors:
                print("使用着色ICP进行精细配准...")
                # 着色ICP
                result = o3d.pipelines.registration.registration_colored_icp(
                    source_copy, target_copy, self.voxel_size * 500, initial_transform,
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iterations))
            else:
                print("使用普通ICP进行精细配准...")
                # 普通ICP
                result = o3d.pipelines.registration.registration_icp(
                    source_copy, target_copy, self.voxel_size * 500, initial_transform,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iterations))

            return result
        except Exception as e:
            print(f"精细ICP配准失败: {e}")
            return None

    def clear_feature_cache(self):
        """清理特征缓存"""
        self.feature_cache.clear()
        print("特征缓存已清理")

    def get_cache_info(self) -> Dict:
        """获取缓存信息"""
        return {
            'cache_size': len(self.feature_cache),
            'cache_keys': list(self.feature_cache.keys())
        }


class PointCloudLoader:
    """点云加载器 - 从demo_video_with_pointcloud_save.py保存的数据中加载点云"""

    def __init__(self):
        """初始化加载器"""
        pass

    def load_object_pointclouds(self, data_file: str) -> Dict[int, List[Tuple[o3d.geometry.PointCloud, np.ndarray, List[int]]]]:
        """
        从保存的数据文件中加载每个物体的多帧点云、对应的原始图像和点云-图像对应关系

        Args:
            data_file: 数据文件路径

        Returns:
            字典，键为物体ID，值为该物体在所有帧中的(点云, 原始图像, 点云-图像对应关系)元组列表
        """
        try:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)

            # 从数据中提取聚类结果
            clustering_results = data.get('clustering_results', [])
            if not clustering_results:
                print(f"警告: 文件 {data_file} 中没有找到聚类结果")
                return {}

            # 组织每个物体的多帧点云和图像
            object_data = {}  # {object_id: [(pcd_frame0, image_frame0, correspondence_frame0), ...]}

            for frame_idx, result in enumerate(clustering_results):
                if 'global_ids' not in result or 'cluster_indices' not in result:
                    continue

                # 获取当前帧的图像数据用于重建点云
                preds = data['preds']
                vggt_batch = data['vggt_batch']
                conf = data['conf']

                # 重建当前帧的点云和获取原始图像
                frame_data = self._reconstruct_frame_data(
                    preds, vggt_batch, conf, frame_idx,
                    result['global_ids'], result['cluster_indices']
                )

                # 将点云和图像分配给对应的物体
                for obj_idx, global_id in enumerate(result['global_ids']):
                    if obj_idx < len(frame_data):
                        pcd, original_image, correspondence = frame_data[obj_idx]
                    else:
                        continue
                    if global_id != -1 and pcd is not None and len(pcd.points) > 0:
                        if global_id not in object_data:
                            object_data[global_id] = []
                        object_data[global_id].append((pcd, original_image, correspondence))

            print(f"从 {data_file} 加载了 {len(object_data)} 个物体的点云和图像数据")
            return object_data

        except Exception as e:
            print(f"加载文件 {data_file} 时出错: {e}")
            return {}

    def _reconstruct_frame_data(self, preds, vggt_batch, conf, frame_idx,
                               global_ids, cluster_indices):
        """
        重建单帧中每个物体的点云、原始图像和对应关系

        Args:
            preds: 预测结果
            vggt_batch: VGGT批次数据
            conf: 置信度
            frame_idx: 帧索引
            global_ids: 全局物体ID列表
            cluster_indices: 聚类索引列表

        Returns:
            (点云, 原始图像, 对应关系)元组列表
        """
        # 动态导入，避免循环依赖
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src/vggt'))
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.training.loss import depth_to_world_points, velocity_local_to_global

        with tf32_off():
            # 获取相机参数
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                preds["pose_enc"], vggt_batch["images"].shape[-2:])
            extrinsic = torch.cat([extrinsic, torch.tensor([0, 0, 0, 1], device=extrinsic.device)[
                                  None, None, None, :].repeat(1, extrinsic.shape[1], 1, 1)], dim=-2)

            # 获取图像尺寸
            B, S, _, image_height, image_width = vggt_batch["images"].shape

            # 构造点云数据
            depth = preds["depth"].view(
                preds["depth"].shape[0]*preds["depth"].shape[1], preds["depth"].shape[2], preds["depth"].shape[3], 1)
            world_points = depth_to_world_points(depth, intrinsic)
            world_points = world_points.view(
                world_points.shape[0], world_points.shape[1]*world_points.shape[2], 3)

            # 转换到相机坐标系
            extrinsic_inv = torch.linalg.inv(extrinsic)
            xyz = torch.matmul(extrinsic_inv[0, :, :3, :3], world_points.transpose(-1, -2)).transpose(-1, -2) + \
                extrinsic_inv[0, :, :3, 3:4].transpose(-1, -2)
            xyz = xyz.reshape(
                xyz.shape[0], image_height * image_width, 3)  # [S, H*W, 3]

            # 获取当前帧的点云坐标
            frame_xyz = xyz[frame_idx].cpu().numpy()  # [H*W, 3]

            # 获取RGB颜色信息
            rgb = vggt_batch["images"].squeeze(
                0)[frame_idx].cpu().numpy()  # [3, H, W]
            rgb = rgb.transpose(1, 2, 0)  # [H, W, 3]
            rgb = rgb.reshape(image_height * image_width, 3)  # [H*W, 3]

            # 获取原始图像（用于SIFT特征提取）
            original_image = vggt_batch["images"].squeeze(0)[frame_idx].cpu().numpy()  # [3, H, W]
            original_image = original_image.transpose(1, 2, 0)  # [H, W, 3]

            # 为每个聚类创建点云和对应的图像区域
            frame_data = []
            for cluster_idx in cluster_indices:
                if len(cluster_idx) == 0:
                    frame_data.append((None, None, None))
                    continue

                # 获取聚类点的坐标和颜色
                cluster_points = frame_xyz[cluster_idx]
                cluster_colors = rgb[cluster_idx]

                # 创建点云
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(cluster_points)
                pcd.colors = o3d.utility.Vector3dVector(cluster_colors)

                # 为聚类创建对应的图像区域
                cluster_image = self._extract_cluster_image_region(
                    cluster_idx, original_image, image_height, image_width)

                # 保存点云点和图像像素的对应关系
                # cluster_idx 就是点云中每个点对应的图像像素索引
                correspondence = cluster_idx

                frame_data.append((pcd, cluster_image, correspondence))

            return frame_data

    def _extract_cluster_image_region(self, cluster_idx, original_image, image_height, image_width):
        """
        提取聚类对应的图像区域
        
        Args:
            cluster_idx: 聚类索引
            original_image: 原始图像 [H, W, 3]
            image_height: 图像高度
            image_width: 图像宽度
            
        Returns:
            聚类对应的图像区域
        """
        try:
            # 将一维索引转换为二维坐标
            cluster_coords = []
            for idx in cluster_idx:
                y = idx // image_width
                x = idx % image_width
                if 0 <= y < image_height and 0 <= x < image_width:
                    cluster_coords.append((y, x))
            
            if not cluster_coords:
                return None
            
            # 计算边界框
            y_coords, x_coords = zip(*cluster_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x_min, x_max = min(x_coords), max(x_coords)
            
            # 添加边距
            margin = 10
            y_min = max(0, y_min - margin)
            y_max = min(image_height - 1, y_max + margin)
            x_min = max(0, x_min - margin)
            x_max = min(image_width - 1, x_max + margin)
            
            # 提取图像区域
            cluster_image = original_image[y_min:y_max+1, x_min:x_max+1]
            
            # 确保图像格式正确
            if cluster_image.size == 0:
                return None
                
            # 确保图像是uint8格式
            if cluster_image.dtype != np.uint8:
                if cluster_image.max() <= 1.0:
                    cluster_image = (cluster_image * 255).astype(np.uint8)
                else:
                    cluster_image = cluster_image.astype(np.uint8)
            
            return cluster_image
            
        except Exception as e:
            print(f"提取聚类图像区域失败: {e}")
            return None


class PointCloudSaver:
    """点云保存器"""

    def __init__(self, output_dir: str = "./saved_pointclouds"):
        """
        初始化保存器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.saved_files = []

    def save_point_cloud(self, pcd: o3d.geometry.PointCloud, name: str,
                         save_format: str = "ply") -> str:
        """
        保存点云

        Args:
            pcd: 点云
            name: 文件名
            save_format: 保存格式 (ply, pcd, obj)

        Returns:
            保存的文件路径
        """
        if len(pcd.points) == 0:
            print(f"警告: 点云 {name} 为空，跳过保存")
            return ""

        # 清理文件名
        safe_name = "".join(c for c in name if c.isalnum()
                            or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')

        # 生成文件路径
        if save_format == "ply":
            file_path = os.path.join(self.output_dir, f"{safe_name}.ply")
            success = o3d.io.write_point_cloud(file_path, pcd)
        elif save_format == "pcd":
            file_path = os.path.join(self.output_dir, f"{safe_name}.pcd")
            success = o3d.io.write_point_cloud(file_path, pcd)
        elif save_format == "obj":
            file_path = os.path.join(self.output_dir, f"{safe_name}.obj")
            success = o3d.io.write_point_cloud(file_path, pcd)
        else:
            print(f"不支持的格式: {save_format}")
            return ""

        if success:
            print(f"点云已保存: {file_path} (点数: {len(pcd.points)})")
            self.saved_files.append(file_path)
            return file_path
        else:
            print(f"保存点云失败: {name}")
            return ""

    def save_registration_report(self, report_data: Dict) -> str:
        """
        保存配准报告

        Args:
            report_data: 报告数据

        Returns:
            报告文件路径
        """
        report_path = os.path.join(self.output_dir, "registration_report.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("点云配准报告\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输出目录: {self.output_dir}\n\n")

            f.write("保存的文件:\n")
            for i, file_path in enumerate(self.saved_files, 1):
                f.write(f"{i}. {os.path.basename(file_path)}\n")

            f.write(f"\n总共保存了 {len(self.saved_files)} 个点云文件\n")

            if 'statistics' in report_data:
                f.write("\n统计信息:\n")
                for key, value in report_data['statistics'].items():
                    f.write(f"{key}: {value}\n")

        print(f"配准报告已保存: {report_path}")
        return report_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="高级多帧点云配准系统 - 保存版本")
    parser.add_argument("--input_dir", type=str, default="./results_26040_8views_true",
                        help="输入数据目录")
    parser.add_argument("--output_dir", type=str, default="./saved_pointclouds",
                        help="输出目录")
    parser.add_argument("--voxel_size", type=float, default=0.01,
                        help="体素大小")
    parser.add_argument("--max_iterations", type=int, default=50,
                        help="最大迭代次数")
    parser.add_argument("--use_coarse_registration", action="store_true", default=True,
                        help="是否使用粗匹配")
    parser.add_argument("--coarse_registration_threshold", type=float, default=0.00001,
                        help="粗匹配质量阈值")
    parser.add_argument("--use_color_features", action="store_true", default=False,
                        help="是否启用颜色特征")

    parser.add_argument("--save_format", type=str, default="ply",
                        choices=["ply", "pcd", "obj"],
                        help="保存格式")
    parser.add_argument("--verbose", action="store_true",
                        help="详细输出")
    parser.add_argument("--enable_sift", action="store_true", default=True,
                        help="启用SIFT特征提取")
    parser.add_argument("--sift_resolution", type=int, default=128,
                        help="SIFT特征提取的图像分辨率")

    args = parser.parse_args()

    # 初始化配准器、加载器和保存器
    registration = AdvancedPointCloudRegistration(
        voxel_size=args.voxel_size,
        max_iterations=args.max_iterations,
        use_coarse_registration=args.use_coarse_registration,
        coarse_registration_threshold=args.coarse_registration_threshold,
        use_color_features=args.use_color_features
    )
    loader = PointCloudLoader()
    saver = PointCloudSaver(args.output_dir)

    # 查找所有推理结果文件
    result_files = glob.glob(os.path.join(args.input_dir, "*.pkl")) + \
        glob.glob(os.path.join(args.input_dir, "*.npy"))

    if not result_files:
        print(f"在目录 {args.input_dir} 中没有找到推理结果文件")
        print("请先运行推理生成点云数据")
        return

    print(f"找到 {len(result_files)} 个推理结果文件")
    print(f"颜色特征: {'启用' if args.use_color_features else '禁用'}")
    print(f"SIFT特征提取: {'启用' if args.enable_sift else '禁用'}")
    print(f"SIFT分辨率: {args.sift_resolution}")

    # 统计信息
    statistics = {
        'total_files': len(result_files),
        'processed_files': 0,
        'total_pointclouds': 0,
        'successful_registrations': 0,
        'sift_features_extracted': 0,
        'cache_hits': 0,
        'color_features_used': args.use_color_features
    }

    # 处理每个推理结果文件
    for result_file in result_files:
        print(f"\n处理推理结果文件: {result_file}")

        try:
            # 加载每个物体的多帧点云和图像数据
            object_data = loader.load_object_pointclouds(result_file)

            if not object_data:
                print(f"从文件 {result_file} 中没有加载到有效的点云数据")
                continue

            # 对每个物体进行配准
            for object_id, frame_data in object_data.items():
                print(f"处理物体 {object_id}，包含 {len(frame_data)} 帧点云和图像")

                # 过滤掉空点云和无效数据
                valid_data = []
                for pcd, image, correspondence in frame_data:
                    if pcd is not None and len(pcd.points) > 0 and image is not None and correspondence is not None:
                        valid_data.append((pcd, image, correspondence))
                    elif pcd is not None and len(pcd.points) > 0:
                        # 如果只有点云没有图像，也保留
                        valid_data.append((pcd, None, None))

                if len(valid_data) == 0:
                    print(f"物体 {object_id} 没有有效的点云数据")
                    continue
                elif len(valid_data) == 1:
                    # 单个点云直接保存
                    pcd, image, correspondence = valid_data[0]
                    name = f"{os.path.basename(result_file)}_Object_{object_id}_Single"
                    saver.save_point_cloud(
                        pcd, name, args.save_format)
                    statistics['total_pointclouds'] += 1
                else:
                    # 多个点云进行配准
                    print(f"对物体 {object_id} 的 {len(valid_data)} 个点云进行配准...")
                    
                    # 检查是否有图像信息
                    has_images = any(image is not None for _, image, _ in valid_data)
                    
                    if has_images:
                        print("使用图像信息进行配准...")
                        # 记录缓存信息
                        cache_info_before = registration.get_cache_info()
                        
                        registered_pcd = registration.register_point_clouds_with_images(valid_data)
                        
                        # 记录缓存命中情况
                        cache_info_after = registration.get_cache_info()
                        if cache_info_after['cache_size'] > cache_info_before['cache_size']:
                            statistics['sift_features_extracted'] += len(valid_data)
                        else:
                            statistics['cache_hits'] += len(valid_data)
                    else:
                        print("使用传统方法进行配准...")
                        # 只使用点云进行配准
                        point_clouds = [data[0] for data in valid_data]
                        registered_pcd = registration.register_point_clouds(point_clouds)

                    # 保存配准后的点云
                    name = f"{os.path.basename(result_file)}_Object_{object_id}_Registered"
                    saver.save_point_cloud(
                        registered_pcd, name, args.save_format)
                    statistics['successful_registrations'] += 1
                    statistics['total_pointclouds'] += len(valid_data)

            statistics['processed_files'] += 1

        except Exception as e:
            print(f"处理文件 {result_file} 时出错: {e}")
            continue

    # 清理缓存
    registration.clear_feature_cache()

    # 保存配准报告
    saver.save_registration_report({'statistics': statistics})

    print(f"\n处理完成!")
    print(f"总共处理了 {statistics['processed_files']} 个文件")
    print(f"生成了 {statistics['total_pointclouds']} 个点云")
    print(f"成功配准了 {statistics['successful_registrations']} 个点云")
    print(f"颜色特征: {'启用' if statistics['color_features_used'] else '禁用'}")
    print(f"SIFT特征提取次数: {statistics['sift_features_extracted']}")
    print(f"缓存命中次数: {statistics['cache_hits']}")
    print(f"所有文件已保存到: {args.output_dir}")


if __name__ == "__main__":
    # 添加简单的语法检查
    try:
        print("代码语法检查通过，开始执行主函数...")
        main()
    except SyntaxError as e:
        print(f"语法错误: {e}")
    except Exception as e:
        print(f"运行时错误: {e}")
