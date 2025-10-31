#!/usr/bin/env python3
"""
Stage2推理代码 - 输出rendered_images和rendered_depths
基于demo_video.py的数据读取方式和输出格式
"""
import os
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
from copy import deepcopy
from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/vggt'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from vggt.models.vggt import VGGT
from dust3r.utils.misc import tf32_off
from src.dust3r.inference import inference
from src.train import cut3r_batch_to_vggt
from online_stage2_trainer import OnlineStage2Trainer
from vggt.models.stage2_refiner import Stage2Refiner
from vggt.training.stage2_loss import Stage2RenderLoss
from accelerate.logging import get_logger
import torch.multiprocessing
import re
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment



# Stage2相关导入


torch.multiprocessing.set_sharing_strategy("file_system")

printer = get_logger(__name__, log_level="DEBUG")

# Set random seed for reproducibility.
random.seed(42)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Stage2 inference and generate rendered images and depths."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/aggregator_resume_noflowgrad_nearestdynamic_resume_0point1_novoxel/checkpoint-epoch_0_45568.pth",
        help="Path to the Stage1 model checkpoint",
    )
    parser.add_argument(
        "--stage2_model_path",
        type=str,
        default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage2_online/stage2_test/stage2-checkpoint-epoch_0_8.pth",
        help="Path to the Stage2 model checkpoint",
    )
    parser.add_argument(
        "--seq_dir",
        type=str,
        default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train_with_flow/segment-15795616688853411272_1245_000_1265_000_with_camera_labels",
        # default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train",
        help="Path to the sequence directory or video file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./stage2_inference_outputs_test",
        help="Output directory for results",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=1400,
        help="Index of the sequence to process (for single inference)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=4,
        help="Number of views for inference",
    )

    # 批量推理参数
    parser.add_argument(
        "--batch_mode",
        action="store_true",
        help="Enable batch inference mode",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=100,
        help="Starting index for batch inference",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=10000,
        help="Ending index for batch inference",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=100,
        help="Step size for batch inference",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue batch processing even if some indices fail",
    )
    parser.add_argument(
        "--use_velocity_based_transform",
        action="store_true",
        help="Use velocity-based transformation instead of optical flow",
    )
    parser.add_argument(
        "--velocity_transform_mode",
        type=str,
        default="simple",
        choices=["simple", "procrustes"],
        help="Velocity transformation mode: 'simple' (translation only) or 'procrustes' (rotation + translation)",
    )

    # 动态物体聚类和跟踪参数
    parser.add_argument(
        "--velocity_threshold",
        type=float,
        default=0.1,
        help="Velocity threshold for dynamic object clustering",
    )
    parser.add_argument(
        "--clustering_eps",
        type=float,
        default=0.02,
        help="DBSCAN eps parameter for clustering",
    )
    parser.add_argument(
        "--clustering_min_samples",
        type=int,
        default=10,
        help="DBSCAN min_samples parameter for clustering",
    )
    parser.add_argument(
        "--tracking_position_threshold",
        type=float,
        default=2.0,
        help="Position threshold for object tracking across frames",
    )
    parser.add_argument(
        "--tracking_velocity_threshold",
        type=float,
        default=0.2,
        help="Velocity threshold for object tracking across frames",
    )

    return parser.parse_args()


def load_stage1_model(model_path, device):
    """加载Stage1模型"""
    print(f"Loading Stage1 model from {model_path}...")

    # 创建模型
    model = VGGT(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        use_sky_token=True
    )

    # 加载检查点（按照train.py中的正确方式）
    checkpoint = torch.load(model_path, map_location="cpu")
    ckpt = checkpoint['model']
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()

    print("Stage1 model loaded successfully")
    return model


def detect_checkpoint_config(stage2_model_path):
    """自动检测checkpoint的网络配置"""
    if not os.path.exists(stage2_model_path):
        return None

    print(f"Auto-detecting checkpoint architecture from {stage2_model_path}...")
    checkpoint = torch.load(stage2_model_path, map_location="cpu")

    if 'stage2_model' not in checkpoint:
        print("  Warning: Checkpoint format not recognized, using default config")
        return None

    ckpt_keys = checkpoint['stage2_model'].keys()

    # 检测层数
    max_layer_idx = -1
    for key in ckpt_keys:
        if 'conv_layers.' in key:
            # 提取层索引，例如 "gaussian_refine_head.conv_layers.2.conv1.weight"
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == 'conv_layers' and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        max_layer_idx = max(max_layer_idx, layer_idx)
                    except ValueError:
                        pass

    num_layers = max_layer_idx + 1 if max_layer_idx >= 0 else 2

    # 检测特征维度
    feature_dim = 128
    for key in ckpt_keys:
        if 'input_encoder.0.weight' in key:
            # shape: [feature_dim, input_dim]
            weight = checkpoint['stage2_model'][key]
            feature_dim = weight.shape[0]
            break

    # 检测是否使用dilated conv（通过检查indice_key）
    use_dilated_conv = any('subm_d' in key for key in ckpt_keys)

    detected_config = {
        'num_layers': num_layers,
        'feature_dim': feature_dim,
        'use_dilated_conv': use_dilated_conv
    }

    print(f"  Detected: num_layers={num_layers}, feature_dim={feature_dim}, use_dilated_conv={use_dilated_conv}")

    return detected_config


def load_stage2_components(stage2_model_path, device, use_velocity_based_transform=False,
                          velocity_transform_mode="simple",
                          velocity_threshold=0.1, clustering_eps=0.02, clustering_min_samples=10,
                          tracking_position_threshold=2.0, tracking_velocity_threshold=0.2):
    """加载Stage2组件"""
    print(f"Loading Stage2 components...")
    if use_velocity_based_transform:
        print(f"  Transformation method: Velocity-based ({velocity_transform_mode})")
    else:
        print(f"  Transformation method: Flow-based")
    print(f"  Clustering parameters: velocity_threshold={velocity_threshold}, eps={clustering_eps}, min_samples={clustering_min_samples}")
    print(f"  Tracking parameters: position_threshold={tracking_position_threshold}, velocity_threshold={tracking_velocity_threshold}")

    # 【智能配置】自动检测checkpoint的网络架构
    detected_config = detect_checkpoint_config(stage2_model_path)

    if detected_config:
        # 使用检测到的配置
        num_layers = detected_config['num_layers']
        feature_dim = detected_config['feature_dim']
        use_dilated = detected_config['use_dilated_conv']
        dilation_rates = [1, 1, 2, 2][:num_layers] if use_dilated else None
        print(f"Using detected checkpoint config: {num_layers} layers, {feature_dim} dim, dilated={use_dilated}")
    else:
        # 使用最新训练配置作为默认值
        num_layers = 4
        feature_dim = 128
        use_dilated = True
        dilation_rates = [1, 1, 2, 2]
        print(f"Using default config: {num_layers} layers, {feature_dim} dim, dilated={use_dilated}")

    # Stage2配置（使用稀疏卷积）
    stage2_config = {
        'training_mode': 'joint',  # 同时启用Gaussian和Pose refinement
        'input_gaussian_dim': 14,
        'output_gaussian_dim': 14,
        'gaussian_feature_dim': feature_dim,
        'gaussian_num_conv_layers': num_layers,
        'gaussian_voxel_size': 0.05,
        'use_dilated_conv': use_dilated,
        'gaussian_dilation_rates': dilation_rates,
        'max_num_points_per_voxel': 5,
        'pose_feature_dim': feature_dim,
        'pose_num_conv_layers': num_layers,
        'pose_voxel_size': 0.1,
        'pose_dilation_rates': dilation_rates,
        'max_points_per_object': 4096,
        'rgb_loss_weight': 1.0,
        'depth_loss_weight': 0.0,
        'dynamic_processor': {
            'use_velocity_based_transform': use_velocity_based_transform,
            'velocity_transform_mode': velocity_transform_mode,
            'velocity_threshold': velocity_threshold,
            'clustering_eps': clustering_eps,
            'clustering_min_samples': clustering_min_samples,
            'tracking_position_threshold': tracking_position_threshold,
            'tracking_velocity_threshold': tracking_velocity_threshold,
            'min_object_size': 50  # 降低最小物体尺寸阈值以检测更多动态物体
        }
    }

    # 创建Stage2训练器
    stage2_trainer = OnlineStage2Trainer(
        stage2_config=stage2_config,
        device=device,
        enable_stage2=True,
        stage2_start_epoch=0,
        stage2_frequency=1,
        memory_efficient=False
    )

    # 加载Stage2检查点
    if os.path.exists(stage2_model_path):
        print(f"Loading Stage2 checkpoint from {stage2_model_path}...")
        stage2_checkpoint = torch.load(stage2_model_path, map_location="cpu")

        if 'stage2_model' in stage2_checkpoint:
            # 由于我们已经自动检测了配置，现在应该可以完美匹配
            stage2_trainer.stage2_model.load_state_dict(stage2_checkpoint['stage2_model'], strict=True)
            print("✓ Stage2 model checkpoint loaded successfully (architecture matched)")
        else:
            # 旧格式的checkpoint
            stage2_trainer.load_state_dict(stage2_checkpoint)
            print("✓ Stage2 checkpoint loaded successfully")
    else:
        print(f"Warning: Stage2 checkpoint not found at {stage2_model_path}, using initialized model")

    # IMPORTANT: 设置为评估模式
    stage2_trainer.eval()
    print("Stage2 model set to eval mode")

    # 创建Stage2渲染损失（仅用于渲染）
    render_loss_config = {
        'rgb_weight': 1.0,
        'depth_weight': 0.0
    }

    stage2_render_loss = Stage2RenderLoss(**render_loss_config)
    stage2_render_loss.to(device)
    stage2_render_loss.eval()

    return stage2_trainer, stage2_render_loss


def dynamic_object_clustering(xyz, velocity, velocity_threshold=0.01, eps=0.02, min_samples=10, area_threshold=100):
    """
    对每一帧进行动态物体聚类

    Args:
        xyz: [S, H*W, 3] 点云坐标
        velocity: [S, H*W, 3] 速度向量
        velocity_threshold: 速度阈值，用于过滤静态背景
        eps: DBSCAN的邻域半径
        min_samples: DBSCAN的最小样本数
        area_threshold: 面积阈值，过滤掉面积小于此值的聚类

    Returns:
        list: 每一帧的聚类结果，每个元素包含点云坐标和聚类标签
    """
    clustering_results = []

    for frame_idx in range(xyz.shape[0]):
        # 获取当前帧的点云和速度
        frame_points = xyz[frame_idx]  # [H*W, 3]
        frame_velocity = velocity[frame_idx]  # [H*W, 3]

        # 计算速度大小
        velocity_magnitude = torch.norm(frame_velocity, dim=-1)  # [H*W]

        # 过滤动态点（速度大于阈值的点）
        dynamic_mask = velocity_magnitude > velocity_threshold
        dynamic_points = frame_points[dynamic_mask]  # [N_dynamic, 3]


        if len(dynamic_points) < min_samples:
            # 如果动态点太少，返回空聚类
            clustering_results.append({
                'points': frame_points,
                'labels': torch.full((len(frame_points),), -1, dtype=torch.long),
                'num_clusters': 0,
                'cluster_centers': [],
                'cluster_velocities': [],
                'cluster_sizes': [],
                'cluster_indices': []
            })
            continue

        # 使用DBSCAN进行聚类
        dynamic_points_np = dynamic_points.cpu().numpy()
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(dynamic_points_np)

        # 将聚类结果映射回原始点云
        full_labels = torch.full((len(frame_points),), -1, dtype=torch.long)
        full_labels[dynamic_mask] = torch.from_numpy(cluster_labels)

        # 统计聚类数量（排除噪声点，标签为-1）
        all_unique_labels = set(cluster_labels)
        if -1 in all_unique_labels:
            all_unique_labels.remove(-1)
        initial_num_clusters = len(all_unique_labels)

        # 计算每个聚类的中心位置和平均速度
        cluster_centers = []
        cluster_velocities = []
        cluster_sizes = []
        cluster_indices = []
        valid_labels = []

        for label in sorted(all_unique_labels):
            cluster_mask = cluster_labels == label
            cluster_points = dynamic_points[cluster_mask]
            cluster_vel = frame_velocity[dynamic_mask][cluster_mask]

            # 计算聚类中心（平均位置）
            center = cluster_points.mean(dim=0)
            # 计算平均速度
            avg_velocity = cluster_vel.mean(dim=0)
            cluster_size = len(cluster_points)

            # 过滤掉面积太小的聚类
            if cluster_size >= area_threshold:
                cluster_centers.append(center)
                cluster_velocities.append(avg_velocity)
                cluster_sizes.append(cluster_size)
                valid_labels.append(label)
                # 计算聚类的点索引
                cluster_point_indices = np.where(cluster_mask)[0]
                dynamic_indices = torch.where(dynamic_mask)[0]
                cluster_indices.append(dynamic_indices[cluster_point_indices].cpu().numpy().tolist())
            else:
                # 将过滤掉的聚类重新标记为静态点（-1）
                cluster_point_indices = np.where(cluster_mask)[0]
                dynamic_indices = torch.where(dynamic_mask)[0]
                filtered_indices = dynamic_indices[cluster_point_indices]
                full_labels[filtered_indices] = -1

        # 更新聚类数量
        num_clusters = len(valid_labels)

        # 重新映射聚类标签，确保连续
        if num_clusters > 0:
            # 创建新的标签映射
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_labels)}

            # 更新full_labels中的聚类标签
            for old_label, new_label in label_mapping.items():
                mask = full_labels == old_label
                full_labels[mask] = new_label

        clustering_results.append({
            'points': frame_points,
            'labels': full_labels,
            'num_clusters': num_clusters,
            'cluster_centers': cluster_centers,
            'cluster_velocities': cluster_velocities,
            'cluster_sizes': cluster_sizes,
            'cluster_indices': cluster_indices
        })

    return clustering_results


def match_objects_across_frames(clustering_results, position_threshold=1.0, velocity_threshold=0.5):
    """
    跨帧匹配动态物体（使用匈牙利算法）

    Args:
        clustering_results: 每一帧的聚类结果
        position_threshold: 位置匹配阈值
        velocity_threshold: 速度匹配阈值

    Returns:
        list: 每一帧的聚类结果，包含全局物体ID
    """
    if len(clustering_results) == 0:
        return clustering_results

    # 初始化全局物体ID
    next_global_id = 0
    global_object_tracks = {}  # {global_id: {frame_id, center, velocity, size}}

    # 为每一帧分配全局ID
    for frame_idx, frame_result in enumerate(clustering_results):
        if frame_result['num_clusters'] == 0:
            frame_result['global_ids'] = []
            continue

        frame_centers = frame_result['cluster_centers']
        frame_velocities = frame_result['cluster_velocities']
        frame_sizes = frame_result['cluster_sizes']

        # 初始化当前帧的全局ID数组，按照聚类标签的顺序
        frame_global_ids = [-1] * len(frame_centers)  # 初始化为-1表示未分配

        if frame_idx == 0:
            # 第一帧：为所有物体分配新的全局ID
            for cluster_idx in range(len(frame_centers)):
                global_id = next_global_id
                next_global_id += 1

                global_object_tracks[global_id] = {
                    'frame_id': frame_idx,
                    'center': frame_centers[cluster_idx],
                    'velocity': frame_velocities[cluster_idx],
                    'size': frame_sizes[cluster_idx]
                }

                frame_global_ids[cluster_idx] = global_id
        else:
            # 使用匈牙利算法进行匹配
            prev_result = clustering_results[frame_idx - 1]
            prev_global_ids = prev_result.get('global_ids', [])

            if len(prev_global_ids) == 0:
                # 前一帧没有物体，为当前帧所有物体分配新ID
                for cluster_idx in range(len(frame_centers)):
                    global_id = next_global_id
                    next_global_id += 1

                    global_object_tracks[global_id] = {
                        'frame_id': frame_idx,
                        'center': frame_centers[cluster_idx],
                        'velocity': frame_velocities[cluster_idx],
                        'size': frame_sizes[cluster_idx]
                    }

                    frame_global_ids[cluster_idx] = global_id
            else:
                # 构建代价矩阵
                prev_centers = prev_result['cluster_centers']
                prev_velocities = prev_result['cluster_velocities']

                cost_matrix = np.full((len(frame_centers), len(prev_centers)), float('inf'))

                for i, current_center in enumerate(frame_centers):
                    for j, prev_center in enumerate(prev_centers):
                        # 计算位置距离
                        pos_dist = torch.norm(current_center - prev_center).item()

                        # 计算速度差异
                        vel_dist = torch.norm(frame_velocities[i] - prev_velocities[j]).item()

                        # 组合代价
                        if pos_dist <= position_threshold and vel_dist <= velocity_threshold:
                            cost_matrix[i, j] = pos_dist + vel_dist

                # 使用匈牙利算法求解
                if cost_matrix.size > 0:
                    row_indices, col_indices = linear_sum_assignment(cost_matrix)

                    # 分配匹配的ID
                    for row_idx, col_idx in zip(row_indices, col_indices):
                        if cost_matrix[row_idx, col_idx] != float('inf'):
                            global_id = prev_global_ids[col_idx]
                            frame_global_ids[row_idx] = global_id

                            # 更新轨迹信息
                            global_object_tracks[global_id] = {
                                'frame_id': frame_idx,
                                'center': frame_centers[row_idx],
                                'velocity': frame_velocities[row_idx],
                                'size': frame_sizes[row_idx]
                            }

                # 为未匹配的物体分配新ID
                for cluster_idx in range(len(frame_centers)):
                    if frame_global_ids[cluster_idx] == -1:
                        global_id = next_global_id
                        next_global_id += 1

                        global_object_tracks[global_id] = {
                            'frame_id': frame_idx,
                            'center': frame_centers[cluster_idx],
                            'velocity': frame_velocities[cluster_idx],
                            'size': frame_sizes[cluster_idx]
                        }

                        frame_global_ids[cluster_idx] = global_id

        # 将全局ID添加到帧结果中
        frame_result['global_ids'] = frame_global_ids

    return clustering_results


def visualize_clustering_results(clustering_results, num_colors=20):
    """
    为聚类结果生成可视化颜色（支持跨帧一致性）

    Args:
        clustering_results: 聚类结果列表（包含全局ID）
        num_colors: 颜色数量

    Returns:
        list: 每一帧的彩色点云
    """
    # 生成颜色映射
    colors = plt.cm.tab20(np.linspace(0, 1, num_colors))  # 使用tab20颜色映射
    colors = (colors[:, :3] * 255).astype(np.uint8)  # 转换为0-255范围

    colored_results = []

    for frame_result in clustering_results:
        points = frame_result['points']  # [H*W, 3]
        labels = frame_result['labels']  # [H*W]
        global_ids = frame_result.get('global_ids', [])

        # 初始化颜色数组（默认黑色背景）
        point_colors = np.zeros((len(points), 3), dtype=np.uint8)

        # 为每个聚类分配颜色（基于全局ID）
        unique_labels = torch.unique(labels)
        colors_assigned = 0
        for label in unique_labels:
            if label == -1:
                continue  # 跳过噪声点

            label_val = label.item()
            if label_val < len(global_ids):
                global_id = global_ids[label_val]
                if global_id != -1:
                    color_idx = global_id % num_colors
                    color = colors[color_idx]

                    mask = labels == label
                    point_colors[mask] = color

        colored_results.append({
            'colors': point_colors,
            'num_clusters': frame_result['num_clusters'],
            'global_ids': global_ids
        })

    return colored_results


def create_clustering_visualization_from_matched_results(matched_clustering_results, vggt_batch, fusion_alpha=0.7):
    """
    从matched_clustering_results创建可视化图像

    Args:
        matched_clustering_results: 来自dynamic_processor的聚类结果
        vggt_batch: VGGT批次数据
        fusion_alpha: 融合透明度

    Returns:
        torch.Tensor: 可视化图像 [S, 3, H, W]
    """
    try:
        if not matched_clustering_results or len(matched_clustering_results) == 0:
            print("No matched clustering results available for visualization")
            # 返回源RGB图像
            B, S, C, H, W = vggt_batch["images"].shape
            return vggt_batch["images"][0]  # [S, 3, H, W]


        # 生成可视化颜色
        colored_results = visualize_clustering_results(matched_clustering_results, num_colors=20)

        # 调试信息：检查聚类结果
        for i, result in enumerate(colored_results):
            non_black_count = np.any(result['colors'] > 0, axis=1).sum()

        B, S, C, image_height, image_width = vggt_batch["images"].shape

        # 将聚类结果与源RGB图像融合
        clustering_images = []
        for frame_idx, colored_result in enumerate(colored_results):
            # 获取源RGB图像
            source_rgb = vggt_batch["images"][0, frame_idx].permute(1, 2, 0)  # [H, W, 3]
            source_rgb = (source_rgb * 255).cpu().numpy().astype(np.uint8)  # 转换为0-255范围

            # 检查是否有动态物体
            if colored_result['num_clusters'] > 0:
                # 将点云颜色重塑为图像格式
                point_colors = colored_result['colors']  # [H*W, 3]
                clustering_image = point_colors.reshape(image_height, image_width, 3)  # [H, W, 3]

                # 融合：将动态聚类结果叠加到源RGB图像上
                # 只有非黑色（非静态）的点才覆盖源图像
                mask = np.any(clustering_image > 0, axis=2)  # [H, W] 布尔掩码，True表示动态点
                mask = mask[:, :, np.newaxis]  # [H, W, 1] 扩展维度

                # 透明度混合：动态点使用聚类颜色与源RGB混合，静态点使用源RGB
                fused_image = np.where(mask,
                                     (fusion_alpha * clustering_image + (1 - fusion_alpha) * source_rgb).astype(np.uint8),
                                     source_rgb)
            else:
                # 没有动态物体时，显示源RGB图像并添加文本提示
                fused_image = source_rgb.copy()
                # 在图像上添加文本提示
                cv2.putText(fused_image, "No Dynamic Objects", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            clustering_images.append(fused_image)

        # 转换为tensor格式
        clustering_tensor = torch.stack([torch.from_numpy(img) for img in clustering_images], dim=0)  # [S, H, W, 3]
        clustering_tensor = clustering_tensor.permute(0, 3, 1, 2)  # [S, 3, H, W]

        return clustering_tensor

    except Exception as e:
        print(f"Error creating clustering visualization: {e}")
        import traceback
        traceback.print_exc()
        # 返回源RGB图像作为备用
        B, S, C, H, W = vggt_batch["images"].shape
        return vggt_batch["images"][0]  # [S, 3, H, W]


def parse_seq_path(p):
    """解析序列路径（复用demo_video.py的逻辑）"""
    if os.path.isdir(p):
        img_paths = sorted(glob.glob(f"{p}/*"))
        tmpdirname = None
    else:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        frame_interval = 1
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
    return img_paths, tmpdirname


def run_stage2_inference(dataset, stage1_model, stage2_trainer, stage2_render_loss, device, args):
    """执行Stage2推理 - 同时运行两种模式并返回拼接结果"""

    # 准备输入视图（复用demo_video.py的逻辑）
    idx = args.idx
    num_views = args.num_views
    views = dataset.__getitem__((idx, 2, num_views))

    # 运行Stage1推理
    start_time = time.time()
    with torch.no_grad():
        outputs, batch = inference(views, stage1_model, device)
    stage1_time = time.time() - start_time

    # 转换为VGGT格式的batch
    vggt_batch = cut3r_batch_to_vggt(views)

    # 运行Stage1 VGGT推理获得预测结果
    start_time = time.time()
    with torch.no_grad():
        stage1_preds = stage1_model(
            vggt_batch["images"]
        )
    stage1_pred_time = time.time() - start_time

    # Stage1的VGGT输出的velocity前三行容易出现异常，直接置零
    # if 'velocity' in stage1_preds and stage1_preds['velocity'] is not None:
    #     velocity = stage1_preds['velocity']  # [B, S, H, W, 3]
    #     if velocity.dim() == 5 and velocity.shape[2] >= 3:
    #         print(f"Applying velocity patch: zeroing top 3 rows (shape: {velocity.shape})")
    #         velocity[:, :, :5, :, :] = 0.0
    #         stage1_preds['velocity'] = velocity

    # ========== Scale Debugging/Override Section ==========
    # 控制是否使用GT scale替换pred scale
    USE_GT_SCALE = False  # 设为True以使用GT scale (来自vggt_batch['depth_scale_factor'])

    # 输出预测的scale信息
    if 'scale' in stage1_preds and stage1_preds['scale'] is not None:
        pred_scale = stage1_preds['scale']  # [B]
        print("\n========== Scale Head Output ==========")
        print(f"Predicted scale shape: {pred_scale.shape}")
        print(f"Predicted scale value: {pred_scale.cpu().numpy()}")

        # 如果vggt_batch中有GT scale数据，输出并可选择性替换
        if 'depth_scale_factor' in vggt_batch and vggt_batch['depth_scale_factor'] is not None:
            gt_scale = vggt_batch['depth_scale_factor']  # scalar or [B]
            print(f"\nGround truth scale (depth_scale_factor): {gt_scale if isinstance(gt_scale, (int, float)) else gt_scale.cpu().numpy()}")

            if USE_GT_SCALE:
                # 替换pred scale为GT scale
                if isinstance(gt_scale, (int, float)):
                    stage1_preds['scale'] = torch.tensor([gt_scale], device=pred_scale.device, dtype=pred_scale.dtype)
                else:
                    stage1_preds['scale'] = gt_scale
                print("\n[INFO] Replaced predicted scale with GT scale!")
        else:
            print("\n[INFO] No GT scale (depth_scale_factor) available in vggt_batch")
    else:
        print("\n[WARNING] No 'scale' output from scale_head in stage1_preds")
    print("=" * 40 + "\n")
    # ========== End Scale Debugging/Override Section ==========

    # 生成动态聚类可视化
    start_time = time.time()


    # 处理动态物体（如果有动态处理器）
    start_time = time.time()

    # 创建空的辅助模型字典（如果需要真实的SAM2等模型，需要额外加载）
    auxiliary_models = {}

    try:
        # 真正的动态物体处理 - 与stage2训练过程完全一致

        # 应用depth_conf过滤 (confidence < 2的像素不参与处理)
        if 'depth_conf' in stage1_preds:
            print(f"Applying depth confidence filter (threshold=2.0)")
            depth_conf = stage1_preds['depth_conf']  # [B, S, H, W]
            conf_mask = depth_conf >= 1.0  # 只保留confidence >= 2的像素
            stage1_preds['depth_conf_mask'] = conf_mask
            print(f"Confidence filter applied: {conf_mask.float().mean().item():.3f} of pixels kept")
        else:
            print("No depth_conf found in predictions, skipping confidence filter")
            stage1_preds['depth_conf_mask'] = None

        # 获取天空颜色用于低opacity区域的替换
        sky_colors = None
        if 'pred_sky_colors' in stage1_preds:
            sky_colors = stage1_preds['pred_sky_colors']  # [B, S, H, W, 3]
            if sky_colors is not None:
                sky_colors = sky_colors[0]  # [S, H, W, 3] - 取第一个batch
                print(f"Found sky colors: {sky_colors.shape}")
            else:
                print("pred_sky_colors is None")
        else:
            print("No pred_sky_colors found in predictions")

        dynamic_objects_data = stage2_trainer.dynamic_processor.process_dynamic_objects(
            stage1_preds, vggt_batch, auxiliary_models
        )
        dynamic_process_time = time.time() - start_time

        # 显示阶段时间统计
        if dynamic_objects_data and 'stage_times' in dynamic_objects_data:
            stage_times = dynamic_objects_data['stage_times']
            print(f"[Stage Times] Preprocessing: {stage_times.get('preprocessing', 0):.3f}s | "
                  f"Clustering+Background: {stage_times.get('clustering_background', 0):.3f}s | "
                  f"Tracking: {stage_times.get('tracking', 0):.3f}s | "
                  f"Aggregation: {stage_times.get('aggregation', 0):.3f}s | "
                  f"Total: {dynamic_objects_data.get('processing_time', 0):.3f}s")

        print(f"\n[Dynamic Objects Detection] Found {len(dynamic_objects_data.get('dynamic_objects', [])) if dynamic_objects_data else 0} dynamic objects")

        if not dynamic_objects_data or len(dynamic_objects_data['dynamic_objects']) == 0:
            print("[WARNING] No dynamic objects detected! Initial and Refined images will be identical.")
            # 如果没有动态物体，使用静态Gaussian进行渲染
            B, S, C, H, W = vggt_batch['images'].shape

            # 准备渲染参数
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                stage1_preds["pose_enc"], vggt_batch["images"].shape[-2:]
            )
            # 添加齐次坐标行到外参矩阵
            extrinsics = torch.cat([
                extrinsics,
                torch.tensor([0, 0, 0, 1], device=extrinsics.device)[
                    None, None, None, :].repeat(1, extrinsics.shape[1], 1, 1)
            ], dim=-2)

            # 使用静态Gaussian场景进行渲染（没有动态物体）
            static_scene = {
                'static_gaussians': dynamic_objects_data.get('static_gaussians'),
                'dynamic_objects': []
            }

            with torch.no_grad():
                rendered_images_tensor, _ = stage2_render_loss.render_refined_scene(
                    refined_scene=static_scene,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    image_height=H,
                    image_width=W,
                    sky_colors=sky_colors
                )

            # 返回结果 (没有动态物体的情况)
            return {
                'rendered_images': {
                    'initial': rendered_images_tensor,
                    'refined': rendered_images_tensor  # 没有动态物体时两者一样
                },
                'dynamic_clustering': create_clustering_visualization_from_matched_results(
                    dynamic_objects_data.get('matched_clustering_results', []), vggt_batch),
                'gt_images': vggt_batch['images'][0],  # [S, 3, H, W]
                'views': views
            }
        else:
            print(f"[INFO] Processing {len(dynamic_objects_data['dynamic_objects'])} dynamic objects with Stage2 refinement")

            # 准备渲染参数
            B, S, C, H, W = vggt_batch['images'].shape
            gt_images = vggt_batch['images']

            # 使用预测的相机参数
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                stage1_preds["pose_enc"], vggt_batch["images"].shape[-2:]
            )
            # 添加齐次坐标行到外参矩阵
            extrinsics = torch.cat([
                extrinsics,
                torch.tensor([0, 0, 0, 1], device=extrinsics.device)[
                    None, None, None, :].repeat(1, extrinsics.shape[1], 1, 1)
            ], dim=-2)

            # 模式1: 不使用Stage2细化，直接使用原始动态物体数据
            start_time = time.time()
            initial_scene = {
                'static_gaussians': dynamic_objects_data.get('static_gaussians'),
                'dynamic_objects': dynamic_objects_data['dynamic_objects']
            }
            with torch.no_grad():
                initial_rendered_images, _ = stage2_render_loss.render_refined_scene(
                    refined_scene=initial_scene,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    image_height=H,
                    image_width=W,
                    sky_colors=sky_colors
                )
            initial_render_time = time.time() - start_time

            # 模式2: 使用Stage2细化
            start_time = time.time()
            print(f"\n[Stage2 Refinement] Starting refinement with {len(dynamic_objects_data['dynamic_objects'])} dynamic objects")
            print(f"[Stage2 Refinement] Stage2 model training mode: {stage2_trainer.stage2_model.training}")

            with torch.no_grad():
                refinement_results = stage2_trainer.stage2_model(
                    dynamic_objects=dynamic_objects_data['dynamic_objects'],
                    static_gaussians=dynamic_objects_data.get('static_gaussians'),
                    preds=stage1_preds
                )
            stage2_refine_time = time.time() - start_time

            print(f"[Stage2 Refinement] Refinement completed in {stage2_refine_time:.3f}s")
            if isinstance(refinement_results, dict) and 'refined_dynamic_objects' in refinement_results:
                num_refined = len(refinement_results['refined_dynamic_objects'])
                print(f"[Stage2 Refinement] Successfully refined {num_refined} dynamic objects")

                # 比较refinement前后的Gaussian参数差异(检查第一个物体)
                if num_refined > 0 and len(dynamic_objects_data['dynamic_objects']) > 0:
                    refined_objects = refinement_results['refined_dynamic_objects']

                    # 计算所有物体的平均参数变化
                    total_means_diff = 0.0
                    total_scales_diff = 0.0
                    total_opacity_diff = 0.0
                    compared_count = 0

                    # 统计pose变换
                    total_translation_diff = 0.0
                    total_rotation_diff = 0.0
                    pose_compared_count = 0

                    # 调试信息
                    num_objects_with_pose_deltas = 0

                    for i, refined_obj in enumerate(refined_objects):
                        if i < len(dynamic_objects_data['dynamic_objects']):
                            original_obj = dynamic_objects_data['dynamic_objects'][i]

                            # 获取canonical_gaussians进行比较
                            if 'canonical_gaussians' in original_obj and 'refined_gaussians' in refined_obj:
                                orig_gauss = original_obj['canonical_gaussians']  # [N, 14]
                                refined_gauss = refined_obj['refined_gaussians']  # [N, 14]

                                # 计算参数差异
                                means_diff = torch.abs(orig_gauss[:, :3] - refined_gauss[:, :3]).mean()
                                scales_diff = torch.abs(orig_gauss[:, 7:10] - refined_gauss[:, 7:10]).mean()
                                opacity_diff = torch.abs(orig_gauss[:, 10] - refined_gauss[:, 10]).mean()

                                total_means_diff += means_diff.item()
                                total_scales_diff += scales_diff.item()
                                total_opacity_diff += opacity_diff.item()
                                compared_count += 1

                            # 统计pose deltas（pose_deltas现在是字典 {frame_idx: pose_delta}）
                            if 'pose_deltas' in refined_obj:
                                pose_deltas_dict = refined_obj['pose_deltas']
                                if pose_deltas_dict and len(pose_deltas_dict) > 0:
                                    num_objects_with_pose_deltas += 1
                                    # 遍历字典中的所有pose_delta
                                    for frame_idx, pose_delta in pose_deltas_dict.items():
                                        if pose_delta is not None and isinstance(pose_delta, torch.Tensor):
                                            # pose_delta: [6] 或 [7] - [rotation(3 or 4), translation(3)]
                                            if pose_delta.numel() >= 6:
                                                translation = pose_delta[-3:]  # 最后3个是translation
                                                rotation = pose_delta[:-3]  # 前面的是rotation

                                                translation_magnitude = torch.norm(translation).item()
                                                rotation_magnitude = torch.norm(rotation).item()

                                                total_translation_diff += translation_magnitude
                                                total_rotation_diff += rotation_magnitude
                                                pose_compared_count += 1

                    if compared_count > 0:
                        avg_means_diff = total_means_diff / compared_count
                        avg_scales_diff = total_scales_diff / compared_count
                        avg_opacity_diff = total_opacity_diff / compared_count

                        print(f"[Stage2 Refinement] Average Gaussian parameter changes across {compared_count} objects:")
                        print(f"  - Means (position) diff: {avg_means_diff:.6f}")
                        print(f"  - Scales diff: {avg_scales_diff:.6f}")
                        print(f"  - Opacity diff: {avg_opacity_diff:.6f}")

                    if pose_compared_count > 0:
                        avg_translation_diff = total_translation_diff / pose_compared_count
                        avg_rotation_diff = total_rotation_diff / pose_compared_count

                        print(f"[Stage2 Refinement] Average pose refinement across {pose_compared_count} frames ({num_objects_with_pose_deltas} objects):")
                        print(f"  - Translation magnitude: {avg_translation_diff:.6f}")
                        print(f"  - Rotation magnitude: {avg_rotation_diff:.6f} rad")
                    else:
                        print(f"[Stage2 Refinement] No pose deltas found (objects with pose_deltas: {num_objects_with_pose_deltas})")
            else:
                print(f"[Stage2 Refinement] Refined {len(refinement_results)} dynamic objects")

            # 直接构建渲染需要的场景格式（不使用get_refined_scene）
            refined_scene = {
                'static_gaussians': dynamic_objects_data.get('static_gaussians'),
                'dynamic_objects': refinement_results['refined_dynamic_objects']
            }

            start_time = time.time()
            with torch.no_grad():
                refined_rendered_images, _ = stage2_render_loss.render_refined_scene(
                    refined_scene=refined_scene,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    image_height=H,
                    image_width=W,
                    sky_colors=sky_colors
                )
            refined_render_time = time.time() - start_time

            # 返回两种模式的结果
            rendered_images = {
                'initial': initial_rendered_images,
                'refined': refined_rendered_images
            }

    except Exception as e:
        print(f"Error in Stage2 processing: {e}")
        print("Creating empty scene for rendering")
        # 创建空的渲染结果
        B, S, C, H, W = vggt_batch['images'].shape
        empty_images = [torch.zeros(3, H, W, device=device) for _ in range(S)]
        rendered_images = {
            'initial': empty_images,
            'refined': empty_images
        }


    return {
        'rendered_images': rendered_images,  # 包含'initial'和'refined'两种模式
        'dynamic_clustering': create_clustering_visualization_from_matched_results(
            dynamic_objects_data.get('matched_clustering_results', []), vggt_batch),
        'gt_images': vggt_batch['images'][0],  # [S, 3, H, W]
        'views': views
    }


def save_results_as_video(results, args):
    """保存结果为视频 - 拼接RGB和动态聚类结果"""

    # 准备输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    rendered_images = results['rendered_images']  # dict with 'initial' and 'refined'
    dynamic_clustering = results.get('dynamic_clustering')  # tensor [S, 3, H, W]
    gt_images = results['gt_images']
    views = results['views']

    # 转换为numpy格式
    def to_numpy(tensor_list):
        if isinstance(tensor_list, list):
            return [img.cpu().numpy() if isinstance(img, torch.Tensor) else img for img in tensor_list]
        else:
            return tensor_list.cpu().numpy() if isinstance(tensor_list, torch.Tensor) else tensor_list

    initial_images_np = to_numpy(rendered_images['initial'])
    refined_images_np = to_numpy(rendered_images['refined'])
    gt_images_np = gt_images.cpu().numpy()  # [S, 3, H, W]

    # 转换RGB图像到0-255范围
    def convert_rgb_to_uint8(rgb_list):
        result = []
        for rgb in rgb_list:
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.cpu().numpy()
            rgb = (rgb * 255).astype(np.uint8)
            result.append(rgb)
        return np.stack(result, axis=0)

    initial_images_uint8 = convert_rgb_to_uint8(initial_images_np)
    refined_images_uint8 = convert_rgb_to_uint8(refined_images_np)
    gt_images_uint8 = (gt_images_np * 255).astype(np.uint8)

    # 准备动态聚类图像
    if dynamic_clustering is not None:
        # 转换动态聚类为numpy格式 [S, 3, H, W] -> [S, H, W, 3]
        clustering_np = dynamic_clustering.permute(0, 2, 3, 1).cpu().numpy()
        clustering_uint8 = (clustering_np).astype(np.uint8)  # 已经是0-255范围
    else:
        # 创建空的动态聚类图像
        num_frames = len(initial_images_np)
        H, W = initial_images_uint8[0].shape[1:3]  # 从[3, H, W]中获取H, W
        clustering_uint8 = np.zeros((num_frames, H, W, 3), dtype=np.uint8)

    # 创建视频 - 四列比较：GT | Initial | Refined | Dynamic Clustering
    video_path = os.path.join(
        args.output_dir, f"stage2_inference_with_clustering_{args.idx}_{views[0]['label'].split('.')[0]}.mp4")

    with iio.get_writer(video_path, fps=10) as writer:
        num_frames = len(initial_images_np)

        for frame_idx in range(num_frames):
            # 获取当前帧的所有图像
            gt_img = gt_images_uint8[frame_idx]  # [3, H, W]
            initial_img = initial_images_uint8[frame_idx]  # [3, H, W]
            refined_img = refined_images_uint8[frame_idx]  # [3, H, W]
            clustering_img = clustering_uint8[frame_idx]  # [H, W, 3]

            # 转换为HWC格式
            gt_img = gt_img.transpose(1, 2, 0)  # [H, W, 3]
            initial_img = initial_img.transpose(1, 2, 0)  # [H, W, 3]
            refined_img = refined_img.transpose(1, 2, 0)  # [H, W, 3]

            # 水平拼接四个图像: GT | Initial | Refined | Dynamic Clustering
            combined_frame = np.concatenate(
                [gt_img, initial_img, refined_img, clustering_img], axis=1)  # [H, W*4, 3]

            writer.append_data(combined_frame)

    return video_path


def run_batch_inference(dataset, stage1_model, stage2_trainer, stage2_render_loss, device, args):
    """运行批量推理"""
    print("=" * 60)
    print("STARTING BATCH INFERENCE")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  Start IDX: {args.start_idx}")
    print(f"  End IDX: {args.end_idx}")
    print(f"  Step: {args.step}")
    print(f"  Mode: Both initial and refined comparison")
    print(f"  Continue on error: {args.continue_on_error}")
    print("")

    # 计算要处理的索引列表
    indices_to_process = list(range(args.start_idx, args.end_idx + 1, args.step))
    total_indices = len(indices_to_process)

    print(f"Will process {total_indices} indices: {indices_to_process}")
    print("")

    # 统计信息
    success_count = 0
    failed_count = 0
    failed_indices = []
    successful_videos = []

    # 批量处理
    for i, idx in enumerate(indices_to_process):
        print("=" * 40)
        print(f"Processing IDX {idx} ({i+1}/{total_indices})")
        print("=" * 40)

        try:
            # 临时修改args.idx为当前处理的索引
            original_idx = args.idx
            args.idx = idx

            # 运行单次推理
            with tf32_off():
                results = run_stage2_inference(
                    dataset, stage1_model, stage2_trainer, stage2_render_loss, device, args
                )

            # 保存结果
            video_path = save_results_as_video(results, args)
            successful_videos.append(video_path)

            success_count += 1

            # 恢复原始idx
            args.idx = original_idx

        except Exception as e:
            print(f"✗ IDX {idx} failed with error: {e}")
            failed_count += 1
            failed_indices.append(idx)

            # 恢复原始idx
            args.idx = original_idx

            if not args.continue_on_error:
                print("Stopping batch inference due to error (use --continue_on_error to continue)")
                break

        print("")

    # 输出最终统计
    print("=" * 60)
    print("BATCH INFERENCE COMPLETED")
    print("=" * 60)
    print(f"Summary:")
    print(f"  Total processed: {success_count + failed_count}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {failed_count}")

    if failed_indices:
        print(f"  Failed indices: {failed_indices}")

    if successful_videos:
        print(f"\nGenerated videos ({len(successful_videos)}):")
        for video in successful_videos:
            print(f"  - {os.path.basename(video)}")

    print(f"\nOutput directory: {args.output_dir}")

    return {
        'success_count': success_count,
        'failed_count': failed_count,
        'failed_indices': failed_indices,
        'successful_videos': successful_videos
    }


def main():
    args = parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 添加dust3r路径
    add_path_to_dust3r(args.model_path)

    # 加载数据集（一次性加载，所有推理共享）
    print(f"Loading dataset from: {args.seq_dir}")

    # 创建数据集对象
    from src.dust3r.datasets.waymo import Waymo_Multi

    # 提取序列名
    seq_name = os.path.basename(args.seq_dir)
    root_dir = os.path.dirname(args.seq_dir)

    print(f"ROOT: {root_dir}, Sequence: {seq_name}")

    dataset = Waymo_Multi(
        split=None,
        ROOT=root_dir,
        img_ray_mask_p=[1.0, 0.0, 0.0],
        valid_camera_id_list=["1", "2", "3"],
        resolution=[(518, 378), (518, 336), (518, 294), (518, 252), (518, 210),
                    (518, 140), (378, 518), (336, 518), (294, 518), (252, 518)],
        num_views=args.num_views,
        seed=42,
        n_corres=0,
        seq_aug_crop=True
    )

    # 加载模型（一次性加载，所有推理共享）
    print("Loading models (shared for all inferences)...")
    stage1_model = load_stage1_model(args.model_path, device)
    stage2_trainer, stage2_render_loss = load_stage2_components(
        args.stage2_model_path, device,
        use_velocity_based_transform=args.use_velocity_based_transform,
        velocity_transform_mode=args.velocity_transform_mode,
        velocity_threshold=args.velocity_threshold,
        clustering_eps=args.clustering_eps,
        clustering_min_samples=args.clustering_min_samples,
        tracking_position_threshold=args.tracking_position_threshold,
        tracking_velocity_threshold=args.tracking_velocity_threshold
    )
    print("Models loaded successfully!\n")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    if args.batch_mode:
        # 批量推理模式
        batch_results = run_batch_inference(
            dataset, stage1_model, stage2_trainer, stage2_render_loss, device, args
        )

        print(f"Batch inference completed!")
        if batch_results['successful_videos']:
            print(f"Generated {len(batch_results['successful_videos'])} videos successfully")

    else:
        # 单次推理模式
        print(f"Running single inference for IDX {args.idx}")

        with tf32_off():
            results = run_stage2_inference(
                dataset, stage1_model, stage2_trainer, stage2_render_loss, device, args
            )

        # 保存结果
        video_path = save_results_as_video(results, args)

        print(f"Single inference completed successfully!")
        print(f"Output video: {video_path}")


if __name__ == "__main__":
    main()
