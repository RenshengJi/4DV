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
        default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo/step2(true+fixmodel+lowlr!+nolpips+onlyflow+velocitylocal+fromscratch)/checkpoint-epoch_2_17880.pth",
        help="Path to the Stage1 model checkpoint",
    )
    parser.add_argument(
        "--stage2_model_path",
        type=str,
        default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo/step2(true+fixmodel+lowlr!+nolpips+onlyflow+velocitylocal+fromscratch)/stage2-checkpoint-final.pth",
        help="Path to the Stage2 model checkpoint",
    )
    parser.add_argument(
        "--seq_dir",
        type=str,
        default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/test/segment-11717495969710734380_2440_000_2460_000_with_camera_labels",
        # default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train",
        help="Path to the sequence directory or video file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./stage2_inference_outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=0,
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
        default=24,
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
        default=150,
        help="Starting index for batch inference",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=200,
        help="Ending index for batch inference",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="Step size for batch inference",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue batch processing even if some indices fail",
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
        use_sky_token=False
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


def load_stage2_components(stage2_model_path, device):
    """加载Stage2组件"""
    print(f"Loading Stage2 components...")

    # Stage2配置
    stage2_config = {
        'training_mode': 'gaussian_only',
        'input_gaussian_dim': 14,  # 添加输入Gaussian维度配置
        'output_gaussian_dim': 14,  # 添加输出Gaussian维度配置
        'gaussian_feature_dim': 128,
        'gaussian_num_layers': 4,
        'gaussian_num_heads': 8,
        'gaussian_mlp_ratio': 4.0,
        'k_neighbors': 20,
        'use_local_attention': True,
        'pose_feature_dim': 128,
        'pose_num_heads': 8,
        'pose_num_layers': 3,
        'max_points_per_object': 2048,
        'rgb_loss_weight': 1.0,
        'depth_loss_weight': 0.0,
        'lpips_loss_weight': 0.1,
        'consistency_loss_weight': 0.0,
        'gaussian_reg_weight': 0.0,
        'pose_reg_weight': 0.0,
        'temporal_smooth_weight': 0.0,
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
        stage2_trainer.load_state_dict(stage2_checkpoint)
        print("Stage2 checkpoint loaded successfully")
    else:
        print(f"Warning: Stage2 checkpoint not found at {stage2_model_path}, using initialized model")

    # 创建Stage2渲染损失（仅用于渲染）
    render_loss_config = {
        'rgb_weight': 1.0,
        'depth_weight': 0.0,
        'lpips_weight': 0.0,
        'consistency_weight': 0.0
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
                'dynamic_mask': dynamic_mask,
                'num_clusters': 0,
                'cluster_centers': [],
                'cluster_velocities': [],
                'cluster_sizes': []
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
            else:
                # 将过滤掉的聚类重新标记为静态点（-1）
                cluster_indices = np.where(cluster_mask)[0]
                dynamic_indices = torch.where(dynamic_mask)[0]
                filtered_indices = dynamic_indices[cluster_indices]
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
            'dynamic_mask': dynamic_mask,
            'num_clusters': num_clusters,
            'dynamic_points': dynamic_points,
            'cluster_labels': torch.from_numpy(cluster_labels),
            'cluster_centers': cluster_centers,
            'cluster_velocities': cluster_velocities,
            'cluster_sizes': cluster_sizes
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

        print(f"Creating visualization from {len(matched_clustering_results)} clustering frames")

        # 生成可视化颜色
        colored_results = visualize_clustering_results(matched_clustering_results, num_colors=20)

        # 调试信息：检查聚类结果
        print(f"总共生成了 {len(colored_results)} 帧的可视化结果")
        for i, result in enumerate(colored_results):
            non_black_count = np.any(result['colors'] > 0, axis=1).sum()
            print(f"帧 {i}: 聚类数量={result['num_clusters']}, 非黑色像素数量={non_black_count}/{len(result['colors'])}")

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
            print(f"帧 {frame_idx}: 检测到 {colored_result['num_clusters']} 个动态物体")

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


def extract_and_cluster_dynamic_objects(preds, vggt_batch, conf, interval, velocity_threshold=0.01, eps=0.1, min_samples=10, position_threshold=1.0, velocity_threshold_match=0.5, fusion_alpha=0.7, area_threshold=100):
    """
    提取点云和速度数据，并进行动态物体聚类

    Args:
        preds: 模型预测结果
        vggt_batch: VGGT批次数据
        conf: 置信度掩码
        interval: 帧间隔
        其他参数: 聚类和匹配参数

    Returns:
        dict: 包含动态聚类可视化的字典
    """
    try:
        B, S, C, image_height, image_width = vggt_batch["images"].shape
        print(f"Processing batch: B={B}, S={S}, H={image_height}, W={image_width}")

        # 提取点云和速度数据
        xyz = preds["pts3d"]  # [B, S, H, W, 3]
        print(f"Point cloud shape: {xyz.shape}")

        # 获取位姿编码用于速度变换
        extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], vggt_batch["images"].shape[-2:])
        extrinsic = torch.cat([extrinsic, torch.tensor([0, 0, 0, 1], device=extrinsic.device)[None,None,None,:].repeat(1,extrinsic.shape[1],1,1)], dim=-2)
        extrinsic_inv = torch.linalg.inv(extrinsic)

        # 检查是否有速度数据
        if "velocity" in preds:
            velocity = preds["velocity"]
            print(f"Velocity found in preds, shape: {velocity.shape}")

            # 应用与demo_video.py相同的速度变换
            velocity = torch.sign(velocity) * (torch.exp(torch.abs(velocity)) - 1)
        else:
            print("No velocity in preds, generating synthetic velocity from consecutive frames")
            # 从连续帧中计算类似速度的信息
            if S > 1:
                # 使用相邻帧之间的位置差异作为"速度"
                velocity = torch.zeros_like(xyz)
                for s in range(S-1):
                    velocity[0, s] = xyz[0, s+1] - xyz[0, s]
                # 最后一帧使用第二到最后一帧的差异
                if S > 1:
                    velocity[0, -1] = velocity[0, -2]
            else:
                velocity = torch.zeros_like(xyz)

        if B > 0 and S > 0:
            # 重塑为聚类所需的格式
            xyz = xyz[0].view(S, -1, 3)  # [S, H*W, 3]
            velocity = velocity[0].view(S, -1, 3)  # [S, H*W, 3]

            # 将速度从局部坐标系转换到全局坐标系（与demo_video.py一致）
            if "velocity" in preds:
                velocity = velocity_local_to_global(velocity.reshape(-1, 3), extrinsic_inv).reshape(S, image_height * image_width, 3)
            print(f"Reshaped - xyz: {xyz.shape}, velocity: {velocity.shape}")

            # 检查数据的有效性
            valid_xyz = torch.isfinite(xyz).all(dim=-1)  # [S, H*W]
            valid_velocity = torch.isfinite(velocity).all(dim=-1)  # [S, H*W]
            valid_points_per_frame = valid_xyz.sum(dim=-1)  # [S]
            print(f"Valid points per frame: {valid_points_per_frame.tolist()}")

            velocity_magnitudes = torch.norm(velocity, dim=-1)  # [S, H*W]
            non_zero_velocity_per_frame = (velocity_magnitudes > 1e-6).sum(dim=-1)  # [S]
            print(f"Non-zero velocity points per frame: {non_zero_velocity_per_frame.tolist()}")

            # 对每一帧进行动态物体聚类
            print(f"开始动态物体聚类... (velocity_threshold={velocity_threshold}, eps={eps}, min_samples={min_samples}, area_threshold={area_threshold})")
            clustering_results = dynamic_object_clustering(xyz, velocity, velocity_threshold=velocity_threshold, eps=eps, min_samples=min_samples, area_threshold=area_threshold)


            # 跨帧匹配动态物体
            print(f"开始跨帧物体匹配... (position_threshold={position_threshold}, velocity_threshold_match={velocity_threshold_match})")
            clustering_results = match_objects_across_frames(clustering_results, position_threshold=position_threshold, velocity_threshold=velocity_threshold_match)

            # 统计跟踪结果
            total_objects = 0
            for frame_idx, result in enumerate(clustering_results):
                global_ids = result.get('global_ids', [])
                # 过滤掉-1值，只统计有效的全局ID
                valid_global_ids = [gid for gid in global_ids if gid != -1]
                total_objects = max(total_objects, len(valid_global_ids))
                print(f"帧 {frame_idx}: 检测到 {result['num_clusters']} 个动态物体，全局ID: {valid_global_ids}")

            print(f"总共跟踪到 {total_objects} 个不同的动态物体")

            # 生成可视化颜色
            colored_results = visualize_clustering_results(clustering_results, num_colors=20)

            # 调试信息：检查聚类结果
            print(f"总共生成了 {len(colored_results)} 帧的可视化结果")
            for i, result in enumerate(colored_results):
                non_black_count = np.any(result['colors'] > 0, axis=1).sum()
                print(f"帧 {i}: 聚类数量={result['num_clusters']}, 非黑色像素数量={non_black_count}/{len(result['colors'])}")

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

                print(f"帧 {frame_idx}: 检测到 {colored_result['num_clusters']} 个动态物体")

            # 转换为tensor格式
            clustering_tensor = torch.stack([torch.from_numpy(img) for img in clustering_images], dim=0)  # [S, H, W, 3]
            clustering_tensor = clustering_tensor.permute(0, 3, 1, 2)  # [S, 3, H, W]

            return {
                "dynamic_clustering": clustering_tensor,
                "clustering_info": [{"num_clusters": result['num_clusters']} for result in colored_results]
            }
    except Exception as e:
        print(f"动态聚类处理出错: {e}")
        # 返回空的聚类结果
        S = vggt_batch["images"].shape[1]
        H, W = vggt_batch["images"].shape[-2:]
        empty_tensor = torch.zeros(S, 3, H, W)
        return {
            "dynamic_clustering": empty_tensor,
            "clustering_info": [{"num_clusters": 0} for _ in range(S)]
        }


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
    print("Preparing input views...")
    idx = args.idx
    num_views = args.num_views
    views = dataset.__getitem__((idx, 2, num_views))

    # 运行Stage1推理
    print("Running Stage1 inference...")
    start_time = time.time()
    with torch.no_grad():
        outputs, batch = inference(views, stage1_model, device)
    stage1_time = time.time() - start_time
    print(f"Stage1 inference completed in {stage1_time:.2f} seconds")

    # 转换为VGGT格式的batch
    print("Converting to VGGT batch format...")
    vggt_batch = cut3r_batch_to_vggt(views)

    # 运行Stage1 VGGT推理获得预测结果
    print("Running Stage1 VGGT predictions...")
    start_time = time.time()
    with torch.no_grad():
        stage1_preds = stage1_model(
            vggt_batch["images"],
            compute_sky_color_loss=False,
            sky_masks=vggt_batch.get("sky_masks"),
            gt_images=vggt_batch["images"],
        )
    stage1_pred_time = time.time() - start_time
    print(
        f"Stage1 VGGT predictions completed in {stage1_pred_time:.2f} seconds")

    # 生成动态聚类可视化
    print("Generating dynamic clustering visualization...")
    start_time = time.time()


    # 处理动态物体（如果有动态处理器）
    print("Processing dynamic objects...")
    start_time = time.time()

    # 创建空的辅助模型字典（如果需要真实的SAM2等模型，需要额外加载）
    auxiliary_models = {}

    try:
        # 真正的动态物体处理 - 与stage2训练过程完全一致
        print("Processing dynamic objects using dynamic_processor...")

        dynamic_objects_data = stage2_trainer.dynamic_processor.process_dynamic_objects(
            stage1_preds, vggt_batch, auxiliary_models
        )
        dynamic_process_time = time.time() - start_time
        print(
            f"Dynamic object processing completed in {dynamic_process_time:.2f} seconds")

        if not dynamic_objects_data or len(dynamic_objects_data['dynamic_objects']) == 0:
            print("No dynamic objects found, rendering static scene")
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

            # 使用静态Gaussian场景进行渲染
            static_scene = stage2_trainer.stage2_model.get_refined_scene(
                dynamic_objects=[], # 空的动态物体列表
                static_gaussians=dynamic_objects_data.get('static_gaussians')
            )

            with torch.no_grad():
                rendered_images_tensor, _ = stage2_render_loss.render_refined_scene(
                    refined_scene=static_scene,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    image_height=H,
                    image_width=W
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
            print(
                f"Found {len(dynamic_objects_data['dynamic_objects'])} dynamic objects")

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
            print("Mode 1: Rendering without Stage2 refinement (initial values)...")
            start_time = time.time()
            initial_scene = stage2_trainer.stage2_model.get_refined_scene(
                dynamic_objects_data['dynamic_objects'],
                dynamic_objects_data.get('static_gaussians')
            )
            with torch.no_grad():
                initial_rendered_images, _ = stage2_render_loss.render_refined_scene(
                    refined_scene=initial_scene,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    image_height=H,
                    image_width=W
                )
            initial_render_time = time.time() - start_time
            print(f"Initial rendering completed in {initial_render_time:.2f} seconds")

            # 模式2: 使用Stage2细化
            print("Mode 2: Running Stage2 refinement...")
            start_time = time.time()
            with torch.no_grad():
                refinement_results = stage2_trainer.stage2_model(
                    dynamic_objects=dynamic_objects_data['dynamic_objects'],
                    static_gaussians=dynamic_objects_data.get('static_gaussians')
                )
            stage2_refine_time = time.time() - start_time
            print(f"Stage2 refinement completed in {stage2_refine_time:.2f} seconds")

            # 获取细化后的场景
            refined_scene = stage2_trainer.stage2_model.get_refined_scene(
                refinement_results, dynamic_objects_data.get('static_gaussians')
            )

            print("Mode 2: Rendering with Stage2 refinement...")
            start_time = time.time()
            with torch.no_grad():
                refined_rendered_images, _ = stage2_render_loss.render_refined_scene(
                    refined_scene=refined_scene,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    image_height=H,
                    image_width=W
                )
            refined_render_time = time.time() - start_time
            print(f"Refined rendering completed in {refined_render_time:.2f} seconds")

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
    print("Saving concatenated results with dynamic clustering as video...")

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

    print(f"Enhanced comparison video with dynamic clustering saved to: {video_path}")
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

            print(f"✓ IDX {idx} completed successfully")
            print(f"  Output: {video_path}")
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

        # 简短休息避免GPU过热
        if i < total_indices - 1:  # 不是最后一个
            print("Waiting 1 second before next inference...")
            time.sleep(1)

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
        valid_camera_id_list=["1"],
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
        args.stage2_model_path, device
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
