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
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/vggt'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from vggt.models.vggt import VGGT
from dust3r.utils.misc import tf32_off
from training.loss import cross_render_and_loss
# Import model and inference functions after adding the ckpt path.
from src.dust3r.inference import inference
# Import cut3r_batch_to_vggt function
from src.train import cut3r_batch_to_vggt

import re
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from accelerate.logging import get_logger
printer = get_logger(__name__, log_level="DEBUG")

# Set random seed for reproducibility.
random.seed(42)

import pickle


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo/step2(true+fixmodel+lowlr!+nolpips+onlyflow+velocitylocal+fromscratch)/checkpoint-epoch_2_17880.pth",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--teacher_model_path",
        type=str,
        default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/model.pt",
        help="Path to the teacher model checkpoint. If provided, will use teacher predictions for depth and pose_enc.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="512",
        help="Shape that input images will be rescaled to; if using 224+linear model, choose 224 otherwise 512",
    )
    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=2,
        help="Visualization threshold for the point cloud viewer. Ranging from 1 to INF",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_26040_8views_true",
        help="value for tempfile.tempdir",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=1000,
        help="Index of the video frame to process. If using a video file, this is the frame index to start from.",
    )
    parser.add_argument(
        "--velocity_threshold",
        type=float,
        default=0.01,
        help="速度阈值，用于过滤静态背景点云。值越大，过滤的静态点越多。",
    )
    parser.add_argument(
        "--dbscan_eps",
        type=float,
        default=0.005,
        help="DBSCAN聚类的邻域半径参数。值越大，聚类越宽松。",
    )
    parser.add_argument(
        "--dbscan_min_samples",
        type=int,
        default=10,
        help="DBSCAN聚类的最小样本数。值越大，聚类越严格。",
    )
    parser.add_argument(
        "--position_threshold",
        type=float,
        default=0.5,
        help="帧间匹配的位置阈值。值越大，匹配越宽松。",
    )
    parser.add_argument(
        "--velocity_threshold_match",
        type=float,
        default=0.2,
        help="帧间匹配的速度阈值。值越大，匹配越宽松。",
    )
    parser.add_argument(
        "--fusion_alpha",
        type=float,
        default=0.7,
        help="动态物体与源图像的融合透明度。值越大，动态物体越突出。",
    )
    parser.add_argument(
        "--area_threshold",
        type=int,
        default=750,
        help="面积阈值，过滤掉面积小于此值的聚类。值越大，过滤越严格。",
    )
    parser.add_argument(
        "--save_pointcloud_data",
        action="store_true",
        help="是否保存点云数据供配准使用",
    )

    return parser.parse_args()


def prepare_output(preds, vggt_batch, args=None):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        preds (dict): Inference outputs.
        vggt_batch (dict): Input data batch.
        args: Command line arguments for clustering parameters.

    Returns:
        dict: Image dictionary for visualization.
    """
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    conf = preds["depth_conf"] > 0.0
    interval = 2

    # metric depth
    _, img_dict = cross_render_and_loss(conf, interval, None, None, preds["depth"].detach(
    ), preds["gaussian_params"], preds["velocity"], preds["pose_enc"], vggt_batch["extrinsics"], vggt_batch["intrinsics"], vggt_batch["images"], vggt_batch["depths"], vggt_batch["point_masks"])

    # 提取点云和速度数据进行动态物体聚类
    if args is not None:
        clustering_img_dict = extract_and_cluster_dynamic_objects(
            preds, vggt_batch, conf, interval,
            velocity_threshold=args.velocity_threshold,
            eps=args.dbscan_eps,
            min_samples=args.dbscan_min_samples,
            position_threshold=args.position_threshold,
            velocity_threshold_match=args.velocity_threshold_match,
            fusion_alpha=args.fusion_alpha,
            area_threshold=args.area_threshold
        )
    else:
        clustering_img_dict = extract_and_cluster_dynamic_objects(
            preds, vggt_batch, conf, interval)

    # 将聚类结果添加到img_dict中
    img_dict.update(clustering_img_dict)

    return img_dict


def extract_and_cluster_dynamic_objects(preds, vggt_batch, conf, interval, velocity_threshold=0.01, eps=0.02, min_samples=10, position_threshold=0.5, velocity_threshold_match=0.2, fusion_alpha=0.7, area_threshold=750):
    """
    提取点云和速度数据，并进行动态物体聚类

    Args:
        preds: 模型预测结果
        vggt_batch: 输入数据批次
        conf: 置信度掩码
        interval: 时间间隔
        velocity_threshold: 速度阈值，用于过滤静态背景
        eps: DBSCAN的邻域半径
        min_samples: DBSCAN的最小样本数
        position_threshold: 帧间匹配的位置阈值
        velocity_threshold_match: 帧间匹配的速度阈值
        fusion_alpha: 动态物体与源图像的融合透明度
        area_threshold: 面积阈值，过滤掉面积小于此值的聚类

    Returns:
        dict: 包含聚类可视化结果的字典
    """
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from training.loss import depth_to_world_points, velocity_local_to_global

    with tf32_off():
        # 获取相机参数
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            preds["pose_enc"], vggt_batch["images"].shape[-2:])
        extrinsic = torch.cat([extrinsic, torch.tensor([0, 0, 0, 1], device=extrinsic.device)[
                              None, None, None, :].repeat(1, extrinsic.shape[1], 1, 1)], dim=-2)

        # 获取图像尺寸
        B, S, _, image_height, image_width = vggt_batch["images"].shape

        # 构造点云数据
        depth = preds["depth"].view(preds["depth"].shape[0]*preds["depth"].shape[1],
                                    preds["depth"].shape[2], preds["depth"].shape[3], 1)
        world_points = depth_to_world_points(depth, intrinsic)
        world_points = world_points.view(
            world_points.shape[0], world_points.shape[1]*world_points.shape[2], 3)

        # 转换到相机坐标系
        extrinsic_inv = torch.linalg.inv(extrinsic)
        xyz = torch.matmul(extrinsic_inv[0, :, :3, :3], world_points.transpose(-1, -2)).transpose(-1, -2) + \
            extrinsic_inv[0, :, :3, 3:4].transpose(-1, -2)
        xyz = xyz.reshape(xyz.shape[0], image_height *
                          image_width, 3)  # [S, H*W, 3]

        # 处理速度数据
        velocity = preds["velocity"].squeeze(
            0).reshape(-1, image_height * image_width, 3)  # [S, H*W, 3]
        velocity = torch.sign(velocity) * (torch.exp(torch.abs(velocity)) - 1)
        velocity = velocity_local_to_global(
            velocity.reshape(-1, 3), extrinsic_inv).reshape(S, image_height * image_width, 3)

        # 应用置信度掩码
        conf_mask = conf.squeeze(0).reshape(
            S, image_height * image_width)  # [S, H*W]

        # 对每一帧进行动态物体聚类
        print(
            f"开始动态物体聚类... (velocity_threshold={velocity_threshold}, eps={eps}, min_samples={min_samples}, area_threshold={area_threshold})")
        clustering_results = dynamic_object_clustering(
            xyz, velocity, velocity_threshold=velocity_threshold, eps=eps, min_samples=min_samples, area_threshold=area_threshold)

        # 跨帧匹配动态物体
        print(
            f"开始跨帧物体匹配... (position_threshold={position_threshold}, velocity_threshold_match={velocity_threshold_match})")
        clustering_results = match_objects_across_frames(
            clustering_results, position_threshold=position_threshold, velocity_threshold=velocity_threshold_match)

        # 统计跟踪结果
        total_objects = 0
        for frame_idx, result in enumerate(clustering_results):
            global_ids = result.get('global_ids', [])
            # 过滤掉-1值，只统计有效的全局ID
            valid_global_ids = [gid for gid in global_ids if gid != -1]
            total_objects = max(total_objects, len(valid_global_ids))
            print(
                f"帧 {frame_idx}: 检测到 {result['num_clusters']} 个动态物体，全局ID: {valid_global_ids}")

            # 验证预测位置（如果有前一帧的数据）
            if frame_idx > 0 and len(valid_global_ids) > 0:
                prev_result = clustering_results[frame_idx-1]
                prev_global_ids = prev_result.get('global_ids', [])
                prev_valid_global_ids = [
                    gid for gid in prev_global_ids if gid != -1]

                for j, global_id in enumerate(global_ids):
                    if global_id != -1 and global_id in prev_valid_global_ids:
                        prev_idx = prev_global_ids.index(global_id)
                        prev_center = prev_result['cluster_centers'][prev_idx]
                        prev_velocity = prev_result['cluster_velocities'][prev_idx]
                        current_center = result['cluster_centers'][j]

                        # 计算预测位置
                        predicted_center = prev_center + prev_velocity
                        actual_distance = torch.norm(
                            current_center - predicted_center)

                        print(
                            f"  物体 {global_id}: 预测位置误差 = {actual_distance:.3f}")

        print(f"总共跟踪到 {total_objects} 个不同的动态物体")

        # 生成可视化颜色
        colored_results = visualize_clustering_results(
            clustering_results, num_colors=20)

        # 将聚类结果与源RGB图像融合
        clustering_images = []
        for frame_idx, colored_result in enumerate(colored_results):
            # 获取源RGB图像
            source_rgb = vggt_batch["images"][0,
                                              frame_idx].permute(1, 2, 0)  # [H, W, 3]
            # 转换为0-255范围
            source_rgb = (source_rgb * 255).cpu().numpy().astype(np.uint8)

            # 将点云颜色重塑为图像格式
            point_colors = colored_result['colors']  # [H*W, 3]
            clustering_image = point_colors.reshape(
                image_height, image_width, 3)  # [H, W, 3]

            # 融合：将动态聚类结果叠加到源RGB图像上
            # 只有非黑色（非静态）的点才覆盖源图像
            # [H, W] 布尔掩码，True表示动态点
            mask = np.any(clustering_image > 0, axis=2)
            mask = mask[:, :, np.newaxis]  # [H, W, 1] 扩展维度

            # 透明度混合：动态点使用聚类颜色与源RGB混合，静态点使用源RGB
            fused_image = np.where(mask,
                                   (fusion_alpha * clustering_image + (1 -
                                    fusion_alpha) * source_rgb).astype(np.uint8),
                                   source_rgb)

            clustering_images.append(fused_image)

            print(f"帧 {frame_idx}: 检测到 {colored_result['num_clusters']} 个动态物体")

        # 转换为tensor格式
        clustering_tensor = torch.stack(
            [torch.from_numpy(img) for img in clustering_images], dim=0)  # [S, H, W, 3]
        clustering_tensor = clustering_tensor.permute(
            0, 3, 1, 2)  # [S, 3, H, W]

        return {
            "dynamic_clustering": clustering_tensor,
            "clustering_info": [{"num_clusters": result['num_clusters']} for result in colored_results],
            "clustering_results": clustering_results  # 添加原始聚类结果
        }


def depth_to_world_points(depth, intrinsic):
    """
    将深度图转换为世界坐标系下的3D点

    参数:
    depth: [N, H, W, 1] 深度图(单位为米)
    intrinsic: [1, N, 3, 3] 相机内参矩阵

    返回:
    world_points: [N, H, W, 3] 世界坐标点(x,y,z)
    """
    with tf32_off():
        N, H, W, _ = depth.shape

        # 生成像素坐标网格 (u,v,1)
        v, u = torch.meshgrid(torch.arange(H, device=depth.device),
                              torch.arange(W, device=depth.device),
                              indexing='ij')
        uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1)  # [H, W, 3]
        uv1 = uv1.unsqueeze(0).expand(N, -1, -1, -1)  # [N, H, W, 3]
        # uv1 -> float32
        uv1 = uv1.float()

        # 转换为相机坐标 (X,Y,Z)
        depth = depth.squeeze(-1)  # [N, H, W]
        intrinsic = intrinsic.squeeze(0)  # [N, 3, 3]

        # 计算相机坐标: (u,v,1) * depth / fx,fy,1
        # 需要处理批量维度
        camera_points = torch.einsum(
            'nij,nhwj->nhwi', torch.inverse(intrinsic), uv1)  # [N, H, W, 3]
        camera_points = camera_points * depth.unsqueeze(-1)  # [N, H, W, 3]

    return camera_points


def dynamic_object_clustering(xyz, velocity, velocity_threshold=0.01, eps=0.02, min_samples=10, area_threshold=750):
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
        dynamic_velocities = frame_velocity[dynamic_mask]  # [N_dynamic, 3]

        if len(dynamic_points) < min_samples:
            # 如果动态点太少，返回空聚类
            clustering_results.append({
                'points': frame_points,
                'labels': torch.full((len(frame_points),), -1, dtype=torch.long),
                'dynamic_mask': dynamic_mask,
                'num_clusters': 0,
                'cluster_centers': [],
                'cluster_velocities': [],
                'cluster_sizes': [],
                'cluster_indices': []
            })
            continue

        # 转换为numpy进行DBSCAN聚类
        dynamic_points_np = dynamic_points.cpu().numpy()

        # 执行DBSCAN聚类
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
            cluster_vel = dynamic_velocities[cluster_mask]

            # 计算聚类中心（平均位置）
            center = cluster_points.mean(dim=0)
            # 计算平均速度（注意：速度是T到T+2的，需要除以2）
            avg_velocity = cluster_vel.mean(dim=0) / 2.0
            cluster_size = len(cluster_points)

            # 过滤掉面积太小的聚类
            if cluster_size >= area_threshold:
                cluster_centers.append(center)
                cluster_velocities.append(avg_velocity)
                cluster_sizes.append(cluster_size)
                valid_labels.append(label)
            else:
                print(
                    f"    过滤掉小聚类: 标签{label}, 大小{cluster_size} < 阈值{area_threshold}")
                # 将过滤掉的聚类重新标记为静态点（-1）
                cluster_indices = np.where(cluster_mask)[0]
                dynamic_indices = torch.where(dynamic_mask)[0]
                filtered_indices = dynamic_indices[cluster_indices]
                full_labels[filtered_indices] = -1

        # 更新聚类数量
        num_clusters = len(valid_labels)
        print(f"    初始聚类数量: {initial_num_clusters}, 过滤后聚类数量: {num_clusters}")

        # 重新映射聚类标签，确保连续
        if num_clusters > 0:
            # 创建新的标签映射
            label_mapping = {old_label: new_label for new_label,
                             old_label in enumerate(valid_labels)}

            # 更新full_labels中的聚类标签
            for old_label, new_label in label_mapping.items():
                mask = full_labels == old_label
                full_labels[mask] = new_label

        # 计算每个聚类的点索引
        cluster_indices = []
        for label in range(num_clusters):
            # 找到属于当前聚类的点的索引
            cluster_mask = full_labels == label
            cluster_point_indices = torch.where(cluster_mask)[0].cpu().numpy().tolist()
            cluster_indices.append(cluster_point_indices)

        clustering_results.append({
            'points': frame_points,
            'labels': full_labels,
            'dynamic_mask': dynamic_mask,
            'num_clusters': num_clusters,
            'dynamic_points': dynamic_points,
            'cluster_labels': torch.from_numpy(cluster_labels),
            'cluster_centers': cluster_centers,
            'cluster_velocities': cluster_velocities,
            'cluster_sizes': cluster_sizes,
            'cluster_indices': cluster_indices
        })

    return clustering_results


def match_objects_across_frames(clustering_results, position_threshold=0.5, velocity_threshold=0.2):
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
            # 确保cluster_indices字段存在
            if 'cluster_indices' not in frame_result:
                frame_result['cluster_indices'] = []
            continue

        frame_centers = frame_result['cluster_centers']
        frame_velocities = frame_result['cluster_velocities']
        frame_sizes = frame_result['cluster_sizes']

        # 初始化当前帧的全局ID数组，按照聚类标签的顺序
        frame_global_ids = [-1] * len(frame_centers)  # 初始化为-1表示未分配

        if frame_idx == 0:
            # 第一帧，为所有物体分配新的全局ID
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
                # 构建成本矩阵
                num_prev = len(prev_global_ids)
                num_current = len(frame_centers)
                cost_matrix = np.full((num_prev, num_current), float('inf'))

                for i, prev_global_id in enumerate(prev_global_ids):
                    track_info = global_object_tracks[prev_global_id]
                    prev_center = track_info['center']
                    prev_velocity = track_info['velocity']

                    for j in range(num_current):
                        current_center = frame_centers[j]
                        current_velocity = frame_velocities[j]

                        # 使用T帧的位置和速度预测T+1帧的位置
                        predicted_center = prev_center + prev_velocity

                        # 计算预测位置与实际位置的距离
                        pos_distance = torch.norm(
                            current_center - predicted_center).item()

                        # 计算速度相似度
                        vel_distance = torch.norm(
                            current_velocity - prev_velocity).item()

                        # 综合评分（位置权重更高）
                        score = pos_distance  # + 0.3 * vel_distance

                        # 如果满足阈值条件，设置成本；否则保持无穷大
                        if pos_distance < position_threshold:  # and vel_distance < velocity_threshold:
                            cost_matrix[i, j] = score

                # 使用匈牙利算法求解最优匹配
                if num_prev > 0 and num_current > 0:
                    # 检查是否有有效的匹配（非无穷大成本）
                    has_valid_matches = np.any(cost_matrix < float('inf'))

                    if has_valid_matches:
                        try:
                            row_indices, col_indices = linear_sum_assignment(
                                cost_matrix)

                            # 处理匹配结果
                            matched_prev = set()
                            matched_current = set()

                            for i, j in zip(row_indices, col_indices):
                                if cost_matrix[i, j] < float('inf'):  # 有效匹配
                                    prev_global_id = prev_global_ids[i]
                                    matched_prev.add(i)
                                    matched_current.add(j)

                                    # 更新跟踪信息
                                    global_object_tracks[prev_global_id] = {
                                        'frame_id': frame_idx,
                                        'center': frame_centers[j],
                                        'velocity': frame_velocities[j],
                                        'size': frame_sizes[j]
                                    }
                                    # 按照聚类索引顺序存储
                                    frame_global_ids[j] = prev_global_id

                            # 为未匹配的当前帧物体分配新ID
                            for j in range(num_current):
                                if j not in matched_current:
                                    global_id = next_global_id
                                    next_global_id += 1

                                    global_object_tracks[global_id] = {
                                        'frame_id': frame_idx,
                                        'center': frame_centers[j],
                                        'velocity': frame_velocities[j],
                                        'size': frame_sizes[j]
                                    }
                                    # 按照聚类索引顺序存储
                                    frame_global_ids[j] = global_id

                            # 为未匹配的前一帧物体保持跟踪（可选：设置消失标记）
                            for i in range(num_prev):
                                if i not in matched_prev:
                                    prev_global_id = prev_global_ids[i]
                                    # 可以选择保持最后一帧的信息或标记为消失
                                    pass
                        except ValueError as e:
                            print(f"    匈牙利算法失败: {e}，为所有物体分配新ID")
                            # 如果匈牙利算法失败，为所有物体分配新ID
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
                        print(f"    没有有效匹配，为所有物体分配新ID")
                        # 没有有效匹配，为所有物体分配新ID
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
                    # 没有前一帧物体，为当前帧所有物体分配新ID
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
        for label in unique_labels:
            if label == -1:  # 噪声点或静态点保持黑色
                continue

            # 获取该聚类对应的全局ID
            cluster_idx = label
            if cluster_idx < len(global_ids) and global_ids[cluster_idx] != -1:
                global_id = global_ids[cluster_idx]
                # 为全局ID分配颜色
                color_idx = global_id % num_colors
                mask = labels == label
                point_colors[mask] = colors[color_idx]

        colored_results.append({
            'points': points,
            'colors': point_colors,
            'labels': labels,
            'num_clusters': frame_result['num_clusters'],
            'global_ids': global_ids
        })

    return colored_results


def run_inference(dataset, model, device, args, teacher_model=None):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
        teacher_model: Optional teacher model for generating depth and pose_enc.
    """

    # Prepare input views.
    print("Preparing input views...")
    idx = args.idx
    num_views = 24
    views = dataset.__getitem__((idx, 2, num_views))

    # Run inference.
    print("Running inference...")
    start_time = time.time()
    outputs, batch = inference(views, model, device)
    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
    )

    # 如果提供了teacher模型，使用teacher的预测结果替换depth和pose_enc
    if teacher_model is not None:
        print("Using teacher model for depth and pose_enc predictions...")
        try:
            with torch.no_grad():
                # 将batch转换为vggt格式
                vggt_batch = cut3r_batch_to_vggt(views)

                # 使用teacher模型进行推理
                teacher_preds = teacher_model(
                    vggt_batch["images"],
                    compute_sky_color_loss=False,
                    sky_masks=vggt_batch.get("sky_masks"),
                    gt_images=vggt_batch["images"],
                )

                # 替换outputs中的depth和pose_enc为teacher的预测结果
                outputs["depth"] = teacher_preds["depth"]
                outputs["depth_conf"] = teacher_preds["depth_conf"]
                outputs["pose_enc"] = teacher_preds["pose_enc"]
                print(
                    "Successfully replaced depth and pose_enc with teacher predictions")
        except Exception as e:
            print(f"Error in teacher model inference: {e}")
            print("Falling back to student model predictions")
    else:
        print("No teacher model provided, using student model predictions")

    # Process outputs for visualization.
    print("Preparing output for visualization...")
    img_dict = prepare_output(outputs, batch, args)

    # 保存点云数据供配准使用
    if args.save_pointcloud_data:
        print("保存点云数据供配准使用...")
        conf = outputs["depth_conf"] > 0.0

        # 获取聚类结果
        clustering_dict = extract_and_cluster_dynamic_objects(
            outputs, batch, conf, 2,
            velocity_threshold=args.velocity_threshold,
            eps=args.dbscan_eps,
            min_samples=args.dbscan_min_samples,
            position_threshold=args.position_threshold,
            velocity_threshold_match=args.velocity_threshold_match,
            fusion_alpha=args.fusion_alpha,
            area_threshold=args.area_threshold
        )
        clustering_results = clustering_dict["clustering_results"]

        pointcloud_data = {
            'preds': outputs,
            'vggt_batch': batch,
            'conf': conf,
            'img_dict': img_dict,
            'clustering_results': clustering_results
        }

        # 保存到文件
        save_path = os.path.join(
            args.output_dir, f"pointcloud_data_{args.idx}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(pointcloud_data, f)
        print(f"点云数据已保存到: {save_path}")
        print(f"包含 {len(clustering_results)} 帧的聚类结果")

    # img_dict -> video
    img_dict = deepcopy(img_dict)
    # img的类型为tensor
    for key in img_dict:
        if isinstance(img_dict[key], torch.Tensor):
            img_dict[key] = img_dict[key].cpu().numpy()
        elif isinstance(img_dict[key], list):
            img_dict[key] = [img.cpu().numpy() if isinstance(
                img, torch.Tensor) else img for img in img_dict[key]]
        else:
            raise TypeError(
                f"Unsupported type {type(img_dict[key])} in img_dict[{key}]")

    def normalize_to_uint8(arr):
        arr = arr.astype(np.float32)
        # vmin, vmax = np.percentile(arr, 2), np.percentile(arr, 98)
        vmin = 0
        vmax = 2
        arr = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0, 1)
        return (arr * 255).astype(np.uint8)

    # 将两个深度图从灰度图转换为RGB图(红近, 蓝远)
    img_dict["target_depth_pred"] = np.stack([
        cv2.applyColorMap(
            normalize_to_uint8(
                img_dict["target_depth_pred"][i][0]), cv2.COLORMAP_JET
        ).transpose(2, 0, 1)
        for i in range(len(img_dict["target_depth_pred"]))
    ], axis=0)
    img_dict["target_depth_gt"] = np.stack([
        cv2.applyColorMap(
            normalize_to_uint8(
                img_dict["target_depth_gt"][i][0]), cv2.COLORMAP_JET
        ).transpose(2, 0, 1)
        for i in range(len(img_dict["target_depth_gt"]))
    ], axis=0)

    # 将其他的rgb图*255转换为int
    for key in ["source_rgb", "target_rgb_pred", "target_rgb_gt", "velocity"]:
        img_dict[key] = np.stack([
            (img_dict[key][i] * 255).astype(np.uint8) for i in range(len(img_dict[key]))
        ], axis=0)

    # 处理动态物体聚类结果
    if "dynamic_clustering" in img_dict:
        # 动态聚类结果已经是uint8格式，直接使用
        img_dict["dynamic_clustering"] = np.stack([
            img_dict["dynamic_clustering"][i] for i in range(len(img_dict["dynamic_clustering"]))
        ], axis=0)
        print("动态物体聚类结果（与源图像融合）已添加到视频中")
    else:
        print("警告：未找到动态物体聚类结果")

    # 将7种img_dict拼接为video（包括动态聚类）
    video_path = os.path.join(args.output_dir, str(
        args.idx) + "_" + views[0]['label'].split('.')[0] + ".mp4")
    with iio.get_writer(video_path, fps=10) as writer:
        # 定义要包含在视频中的键
        video_keys = ["source_rgb", "target_rgb_pred", "target_rgb_gt", "target_depth_pred",
                      "target_depth_gt", "velocity", "dynamic_clustering"]

        # 过滤掉不存在的键
        available_keys = [key for key in video_keys if key in img_dict]
        max_length = max(len(img_dict[key]) for key in available_keys)

        for i in range(max_length):
            frame = []
            for key in available_keys:
                if i < len(img_dict[key]):
                    frame.append(img_dict[key][i])
                else:
                    frame.append(np.zeros_like(img_dict[key][0]))
            combined_frame = np.concatenate(frame, axis=1)
            # Transpose to HWC format for video writer
            writer.append_data(combined_frame.transpose(1, 2, 0))
    print(f"输出视频已保存到 {video_path}")
    print(f"视频包含以下内容: {available_keys}")


def main():

    args = parse_args()

    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    # Add the checkpoint path (required for model imports in the dust3r package).
    add_path_to_dust3r(args.model_path)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    from src.dust3r.datasets.waymo import Waymo_Multi
    dataset = Waymo_Multi(allow_repeat=False, split=None, ROOT="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train", img_ray_mask_p=[1.0, 0.0, 0.0], aug_crop=16, resolution=[
                          (518, 378), (518, 336), (518, 294), (518, 252), (518, 210), (518, 140), (378, 518), (336, 518), (294, 518), (252, 518)], num_views=24, n_corres=0, seq_aug_crop=True)

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    model = VGGT(img_size=518, patch_size=14,
                 embed_dim=1024, use_sky_token=False)
    ckpt = torch.load(args.model_path, map_location=device)['model']
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    del ckpt

    model.eval()
    model = model.to(device)

    # Load teacher model if provided
    teacher_model = None
    if args.teacher_model_path is not None:
        print(f"Loading teacher model from {args.teacher_model_path}...")
        try:
            teacher_model = VGGT(img_size=518, patch_size=14,
                                 embed_dim=1024, use_sky_token=False)
            teacher_ckpt = torch.load(
                args.teacher_model_path, map_location=device)
            if "model" in teacher_ckpt:
                teacher_ckpt = teacher_ckpt["model"]
            teacher_ckpt = {k.replace("module.", ""): v for k, v in teacher_ckpt.items()}
            teacher_model.load_state_dict(teacher_ckpt, strict=False)
            del teacher_ckpt

            teacher_model.eval()
            teacher_model = teacher_model.to(device)
            teacher_model.requires_grad_(False)
            print("Teacher model loaded successfully")
        except Exception as e:
            print(f"Error loading teacher model: {e}")
            print("Continuing without teacher model")
            teacher_model = None

    idx = 600
    while True:
        print(f"\n========== Running inference for idx={idx} ==========")
        args.idx = idx
        run_inference(dataset, model, device, args, teacher_model)
        # try:
        #     run_inference(dataset, model, device, args)
        # except Exception as e:
        #     print(f"Error at idx={idx}: {e}")
        idx += 200


if __name__ == "__main__":
    main()
