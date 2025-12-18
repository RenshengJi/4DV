#!/usr/bin/env python3
"""
Multi-camera inference script for VGGT model.
Supports loading and visualizing results from multiple cameras simultaneously.
Each frame shows horizontally concatenated images from n cameras.

使用方式:
python inference_multi.py

可以通过命令行覆盖配置:
python inference_multi.py batch_mode=false single_idx=5
"""

import os
import sys
import numpy as np
import torch
import cv2
import imageio.v2 as iio
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import hydra
from omegaconf import OmegaConf

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/vggt'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from vggt.models.vggt import VGGT
from dust3r.utils.misc import tf32_off
from src.dust3r.inference import inference
from src.online_dynamic_processor import OnlineDynamicProcessor
from vggt.training.loss import self_render_and_loss
from vggt.training.stage2_loss import prune_gaussians_by_voxel


def load_model(model_path, device, cfg):
    """加载VGGT模型"""
    print(f"Loading model from: {model_path}")
    print(f"Model config: sh_degree={cfg.sh_degree}")

    # 使用eval()解析模型配置字符串
    model = eval(cfg.model)

    checkpoint = torch.load(model_path, map_location="cpu")
    ckpt = checkpoint.get('model', checkpoint)
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model


def load_dataset(dataset_cfg):
    """加载数据集"""
    from dataset import WaymoDataset, Waymo_Multi, ImgNorm

    print(f"Loading dataset from config...")

    # 使用eval()解析数据集配置字符串（类似train.py）
    dataset = eval(dataset_cfg)

    print(f"Dataset loaded: {len(dataset)} scenes")
    return dataset


def visualize_depth(depth):
    """
    将深度图可视化为彩色图像
    Args:
        depth: [S, H, W] or [S, H, W, 1] numpy array
    Returns:
        [S, H, W, 3] numpy array (uint8)
    """
    # Handle extra dimension if present
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)  # [S, H, W, 1] -> [S, H, W]

    if depth.ndim != 3:
        raise ValueError(f"Expected depth with shape [S, H, W], got shape {depth.shape}")

    S, H, W = depth.shape
    depth_vis = []

    for s in range(S):
        d = depth[s]
        valid_mask = d > 0

        if valid_mask.any():
            vmin, vmax = np.percentile(d[valid_mask], [2, 98])
            d_norm = np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1)
        else:
            d_norm = np.zeros_like(d)

        d_colored = (plt.cm.viridis(d_norm)[..., :3] * 255).astype(np.uint8)
        depth_vis.append(d_colored)

    return np.array(depth_vis)


def visualize_velocity(velocity, scale=0.2):
    """可视化velocity为RGB图像（参考inference.py）

    Args:
        velocity: [S, H, W, 3] tensor, velocity field
        scale: 速度缩放因子
    Returns:
        [S, 3, H, W] tensor (float32, [0, 1])
    """
    from dust3r.utils.image import scene_flow_to_rgb

    S, H, W, _ = velocity.shape
    velocity_rgb = scene_flow_to_rgb(velocity.detach(), scale).permute(0, 3, 1, 2)  # [S, 3, H, W]
    return velocity_rgb


def get_segmentation_colormap(num_classes=4):
    """生成分割颜色映射表（基于 Waymo 类别和 Cityscapes 风格）

    Waymo 类别定义（4类）:
        0: background/unlabeled
        1: vehicle
        2: sign
        3: pedestrian + cyclist
    """
    colormap = np.zeros((num_classes, 3), dtype=np.uint8)
    colormap[0] = [255, 255, 255]     # background - 白色
    colormap[1] = [0, 0, 142]         # vehicle - 深蓝色
    colormap[2] = [220, 220, 0]       # sign - 黄色
    colormap[3] = [220, 20, 60]       # pedestrian+cyclist - 红色
    return colormap


def visualize_segmentation(seg_labels, seg_mask=None, num_classes=4):
    """将分割标签可视化为RGB图像（返回[S, H, W, 3] uint8 numpy格式）"""
    colormap = get_segmentation_colormap(num_classes)

    # Convert labels to numpy if needed
    if isinstance(seg_labels, torch.Tensor):
        seg_labels = seg_labels.cpu().numpy()

    if seg_mask is not None and isinstance(seg_mask, torch.Tensor):
        seg_mask = seg_mask.cpu().numpy()

    # Handle shape
    if seg_labels.ndim == 2:
        seg_labels = seg_labels[np.newaxis, ...]
        if seg_mask is not None:
            seg_mask = seg_mask[np.newaxis, ...]

    S, H, W = seg_labels.shape
    seg_rgb = np.zeros((S, H, W, 3), dtype=np.uint8)

    # Map each class to its color
    for s in range(S):
        for class_id in range(num_classes):
            mask = (seg_labels[s] == class_id)
            seg_rgb[s][mask] = colormap[class_id]

        # Set invalid regions to gray if mask is provided
        if seg_mask is not None:
            invalid_mask = (seg_mask[s] == 0)
            seg_rgb[s][invalid_mask] = [128, 128, 128]  # Gray

    return seg_rgb


def visualize_clustering_results(clustering_results, num_colors=20):
    """为聚类结果生成颜色（参考inference.py）"""
    # 使用tab20颜色映射（与inference.py一致）
    colors = plt.cm.tab20(np.linspace(0, 1, num_colors))
    colors = (colors[:, :3] * 255).astype(np.uint8)

    colored_results = []

    for frame_result in clustering_results:
        points = frame_result['points']  # [H*W, 3]
        labels = frame_result['labels']  # [H*W] tensor
        global_ids = frame_result.get('global_ids', [])

        # 初始化颜色数组（默认黑色背景）
        point_colors = np.zeros((len(points), 3), dtype=np.uint8)

        # 为每个聚类分配颜色（基于全局ID）
        if isinstance(labels, torch.Tensor):
            unique_labels = torch.unique(labels)
        else:
            unique_labels = np.unique(labels)

        colors_assigned = 0
        for label in unique_labels:
            if isinstance(label, torch.Tensor):
                label_val = label.item()
            else:
                label_val = int(label)

            if label_val == -1:
                continue  # 跳过噪声点

            if label_val < len(global_ids):
                global_id = global_ids[label_val]
                if global_id != -1:
                    color_idx = global_id % num_colors
                    color = colors[color_idx]

                    if isinstance(labels, torch.Tensor):
                        mask = (labels == label).cpu().numpy()
                    else:
                        mask = (labels == label)

                    point_colors[mask] = color
                    colors_assigned += 1

        colored_results.append({
            'points': points,
            'colors': point_colors,
            'num_clusters': colors_assigned
        })

    return colored_results


def create_clustering_visualization(matched_clustering_results, vggt_batch, fusion_alpha=0.7):
    """从matched_clustering_results创建可视化图像（返回[S, H, W, 3] uint8 numpy格式）"""
    try:
        B, S, C, image_height, image_width = vggt_batch["images"].shape

        print(f"[DEBUG] create_clustering_visualization:")
        print(f"  vggt_batch['images'] shape: {vggt_batch['images'].shape}")
        print(f"  matched_clustering_results length: {len(matched_clustering_results) if matched_clustering_results else 0}")

        if not matched_clustering_results or len(matched_clustering_results) == 0:
            # 返回源RGB图像（转为uint8）
            images_np = (vggt_batch["images"][0].cpu().numpy() * 255).astype(np.uint8)  # [S, 3, H, W]
            images_np = images_np.transpose(0, 2, 3, 1)  # [S, H, W, 3]
            print(f"[DEBUG] No clustering results, returning RGB images: {images_np.shape}")
            return images_np

        # 生成可视化颜色
        colored_results = visualize_clustering_results(matched_clustering_results, num_colors=20)
        print(f"[DEBUG] colored_results length: {len(colored_results)}")

        # 检查是否是多相机模式
        camera_indices = vggt_batch.get('camera_indices', None)
        frame_indices = vggt_batch.get('frame_indices', None)
        is_multi_camera = (camera_indices is not None and frame_indices is not None)

        if is_multi_camera:
            # 多相机模式：colored_results按时间帧组织，每帧包含所有相机的点
            # 需要为每个view单独创建可视化
            camera_indices = camera_indices[0].cpu().numpy() if isinstance(camera_indices, torch.Tensor) else camera_indices
            frame_indices = frame_indices[0].cpu().numpy() if isinstance(frame_indices, torch.Tensor) else frame_indices

            num_cameras = len(np.unique(camera_indices))
            num_frames = len(colored_results)

            print(f"[DEBUG] Multi-camera mode: {num_cameras} cameras, {num_frames} frames")

            clustering_images = []
            for view_idx in range(S):
                cam_idx = camera_indices[view_idx]
                frame_idx = frame_indices[view_idx]

                # 获取源RGB图像
                source_rgb = vggt_batch["images"][0, view_idx].permute(1, 2, 0)  # [H, W, 3]
                source_rgb = (source_rgb * 255).cpu().numpy().astype(np.uint8)

                # 获取该帧的clustering结果
                colored_result = colored_results[frame_idx]
                point_colors = colored_result['colors']  # [num_cameras * H * W, 3]

                # 提取该相机的点云颜色
                # points按照view顺序组织，每个view有H*W个点
                start_idx = cam_idx * image_height * image_width
                end_idx = start_idx + image_height * image_width
                camera_point_colors = point_colors[start_idx:end_idx]  # [H*W, 3]

                if colored_result['num_clusters'] > 0 and np.any(camera_point_colors > 0):
                    # 重塑为图像
                    clustering_image = camera_point_colors.reshape(image_height, image_width, 3)

                    # 融合
                    mask = np.any(clustering_image > 0, axis=2)[:, :, np.newaxis]
                    fused_image = np.where(mask,
                                         (fusion_alpha * clustering_image + (1 - fusion_alpha) * source_rgb).astype(np.uint8),
                                         source_rgb)
                else:
                    fused_image = source_rgb.copy()

                clustering_images.append(fused_image)
        else:
            # 单相机模式：原有逻辑
            clustering_images = []
            for frame_idx, colored_result in enumerate(colored_results):
                print(f"[DEBUG] Processing frame {frame_idx}, num_clusters: {colored_result['num_clusters']}")

                # 获取源RGB图像
                source_rgb = vggt_batch["images"][0, frame_idx].permute(1, 2, 0)  # [H, W, 3]
                source_rgb = (source_rgb * 255).cpu().numpy().astype(np.uint8)

                # 检查是否有动态物体
                if colored_result['num_clusters'] > 0:
                    # 将点云颜色重塑为图像格式
                    point_colors = colored_result['colors']  # [H*W, 3]
                    print(f"[DEBUG] point_colors shape: {point_colors.shape}, non-zero: {np.sum(np.any(point_colors > 0, axis=1))}")
                    clustering_image = point_colors.reshape(image_height, image_width, 3)  # [H, W, 3]

                    # 融合
                    mask = np.any(clustering_image > 0, axis=2)  # [H, W]
                    print(f"[DEBUG] mask sum: {np.sum(mask)}, total pixels: {mask.size}")
                    mask = mask[:, :, np.newaxis]  # [H, W, 1]

                    fused_image = np.where(mask,
                                         (fusion_alpha * clustering_image + (1 - fusion_alpha) * source_rgb).astype(np.uint8),
                                         source_rgb)
                else:
                    # 没有动态物体
                    print(f"[DEBUG] No clusters for frame {frame_idx}")
                    fused_image = source_rgb.copy()
                    cv2.putText(fused_image, "No Dynamic Objects", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                clustering_images.append(fused_image)

        return np.array(clustering_images)  # [S, H, W, 3]

    except Exception as e:
        print(f"Error creating clustering visualization: {e}")
        import traceback
        traceback.print_exc()
        # 返回源RGB图像
        images_np = (vggt_batch["images"][0].cpu().numpy() * 255).astype(np.uint8)  # [S, 3, H, W]
        images_np = images_np.transpose(0, 2, 3, 1)  # [S, H, W, 3]
        return images_np


def render_gaussians_with_sky(scene, intrinsics, extrinsics, sky_colors, sampled_frame_indices, H, W, device,
                              enable_voxel_pruning=True, voxel_size=0.002, depth_scale_factor=None):
    """
    渲染gaussian场景（与train.py的Stage2RenderLoss相同逻辑）
    包括sky_color的alpha blending合成
    逐帧渲染以正确处理动态物体的变换

    Args:
        enable_voxel_pruning: bool 是否启用voxel剪枝
        voxel_size: float voxel大小（metric尺度，单位米）
        depth_scale_factor: float 深度缩放因子
    """
    from gsplat import rasterization

    S = intrinsics.shape[0]
    rendered_images = []
    rendered_depths = []

    # 逐帧渲染（参考stage2_loss.py的_render_frame逻辑）
    for frame_idx in range(S):
        # 每帧收集gaussians
        all_means = []
        all_scales = []
        all_colors = []
        all_rotations = []
        all_opacities = []

        # Static gaussians (参考stage2_loss.py line 493-499)
        if scene.get('static_gaussians') is not None:
            static_gaussians = scene['static_gaussians']  # [N, 14]
            if static_gaussians.shape[0] > 0:
                all_means.append(static_gaussians[:, :3])
                all_scales.append(static_gaussians[:, 3:6])
                all_colors.append(static_gaussians[:, 6:9].unsqueeze(-2))
                all_rotations.append(static_gaussians[:, 9:13])
                all_opacities.append(static_gaussians[:, 13])

        # Dynamic objects - Cars (使用canonical空间+变换，参考stage2_loss.py line 436-467)
        dynamic_objects_cars = scene.get('dynamic_objects_cars', [])
        for obj_data in dynamic_objects_cars:
            # 检查物体是否在当前帧存在
            if not _object_exists_in_frame(obj_data, frame_idx):
                continue

            # 获取物体在正规空间(canonical)的Gaussian参数
            canonical_gaussians = obj_data.get('canonical_gaussians')  # [N, 14]
            if canonical_gaussians is None or canonical_gaussians.shape[0] == 0:
                continue

            # 获取从canonical空间到当前帧的变换
            frame_transform = _get_object_transform_to_frame(obj_data, frame_idx)
            if frame_transform is None:
                # 如果没有变换信息，直接使用原始Gaussians（假设当前帧就是参考帧）
                transformed_gaussians = canonical_gaussians
            else:
                # 应用变换：将canonical空间的Gaussians变换到当前帧
                transformed_gaussians = _apply_transform_to_gaussians(
                    canonical_gaussians, frame_transform
                )

            # 添加到渲染列表
            if transformed_gaussians.shape[0] > 0:
                all_means.append(transformed_gaussians[:, :3])
                all_scales.append(transformed_gaussians[:, 3:6])
                all_colors.append(transformed_gaussians[:, 6:9].unsqueeze(-2))
                all_rotations.append(transformed_gaussians[:, 9:13])
                all_opacities.append(transformed_gaussians[:, 13])

        # Dynamic objects - People (每帧单独的Gaussians，不使用变换，参考stage2_loss.py line 469-487)
        dynamic_objects_people = scene.get('dynamic_objects_people', [])
        for obj_data in dynamic_objects_people:
            # 检查物体是否在当前帧存在
            frame_gaussians = obj_data.get('frame_gaussians', {})
            if frame_idx not in frame_gaussians:
                continue

            # 直接使用当前帧的Gaussians（不进行变换）
            current_frame_gaussians = frame_gaussians[frame_idx]  # [N, 14]
            if current_frame_gaussians is None or current_frame_gaussians.shape[0] == 0:
                continue

            # 添加到渲染列表
            all_means.append(current_frame_gaussians[:, :3])
            all_scales.append(current_frame_gaussians[:, 3:6])
            all_colors.append(current_frame_gaussians[:, 6:9].unsqueeze(-2))
            all_rotations.append(current_frame_gaussians[:, 9:13])
            all_opacities.append(current_frame_gaussians[:, 13])

        if len(all_means) == 0:
            # 如果没有Gaussian，返回空图像
            rendered_images.append(torch.zeros(3, H, W, device=device))
            rendered_depths.append(torch.zeros(H, W, device=device))
            continue

        # Concatenate
        means = torch.cat(all_means, dim=0)  # [N, 3]
        scales = torch.cat(all_scales, dim=0)  # [N, 3]
        colors = torch.cat(all_colors, dim=0)  # [N, 1, 3]
        rotations = torch.cat(all_rotations, dim=0)  # [N, 4]
        opacities = torch.cat(all_opacities, dim=0)  # [N]

        # Apply voxel pruning (如果启用)
        if enable_voxel_pruning and means.shape[0] > 0:
            means, scales, rotations, opacities, colors = prune_gaussians_by_voxel(
                means, scales, rotations, opacities, colors,
                voxel_size=voxel_size,
                depth_scale_factor=depth_scale_factor
            )

        # Fix NaN/Inf (参考stage2_loss.py line 550-555)
        means = torch.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
        scales = torch.nan_to_num(scales, nan=0.01, posinf=1.0, neginf=0.01)
        colors = torch.nan_to_num(colors, nan=0.5, posinf=1.0, neginf=0.0)
        rotations = torch.nan_to_num(rotations, nan=0.0, posinf=1.0, neginf=-1.0)
        opacities = torch.nan_to_num(opacities, nan=0.5, posinf=1.0, neginf=0.0)

        K = intrinsics[frame_idx]
        w2c = extrinsics[frame_idx]

        try:
            # 参考stage2_loss.py line 562-569
            render_result = rasterization(
                means, rotations, scales, opacities, colors,
                w2c.unsqueeze(0), K.unsqueeze(0), W, H,
                sh_degree=0, render_mode="RGB+ED",
                radius_clip=0, near_plane=0.0001,
                far_plane=1000.0,
                eps2d=0.3,
            )

            rendered_image = render_result[0][0, :, :, :3].permute(2, 0, 1)
            rendered_depth = render_result[0][0, :, :, -1]
            rendered_alpha = render_result[1][0, :, :, 0] if len(render_result) > 1 and render_result[1] is not None else torch.ones(H, W, device=device)

            rendered_image = torch.clamp(rendered_image, min=0, max=1)

            # Composite with sky
            if sky_colors is not None and sampled_frame_indices is not None:
                if not isinstance(sampled_frame_indices, torch.Tensor):
                    sampled_frame_indices = torch.tensor(sampled_frame_indices, device=device)

                matches = (sampled_frame_indices == frame_idx)
                if matches.any():
                    sky_idx = matches.nonzero(as_tuple=True)[0].item()
                    frame_sky_color = sky_colors[sky_idx]  # [3, H, W]
                    alpha_3ch = rendered_alpha.unsqueeze(0)  # [1, H, W]
                    rendered_image = alpha_3ch * rendered_image + (1 - alpha_3ch) * frame_sky_color
                    rendered_image = torch.clamp(rendered_image, min=0, max=1)

            rendered_images.append(rendered_image)
            rendered_depths.append(rendered_depth)

        except Exception as e:
            print(f"Error rendering frame {frame_idx}: {e}")
            rendered_images.append(torch.zeros(3, H, W, device=device))
            rendered_depths.append(torch.zeros(H, W, device=device))

    return torch.stack(rendered_images, dim=0), torch.stack(rendered_depths, dim=0)


def render_self(vggt_batch, preds, sh_degree=0, enable_voxel_pruning=True, voxel_size=0.002):
    """
    调用self_render_and_loss生成自渲染结果（用于可视化）

    Returns:
        rendered_rgb: [S, 3, H, W] 渲染的RGB图像
        rendered_depth: [S, H, W] 渲染的深度图像
    """
    # 调用self_render_and_loss
    _, img_dict = self_render_and_loss(
        vggt_batch, preds,
        sampled_frame_indices=None,  # 渲染所有帧
        sh_degree=sh_degree,
        enable_voxel_pruning=enable_voxel_pruning,
        voxel_size=voxel_size
    )

    # 提取渲染结果
    rendered_rgb = img_dict['self_rgb_pred']  # [S, 3, H, W]
    rendered_depth = img_dict['self_depth_pred'].squeeze(1)  # [S, H, W]

    return rendered_rgb, rendered_depth


def _object_exists_in_frame(obj_data, frame_idx):
    """检查动态物体是否在指定帧中存在（参考stage2_loss.py line 650-655）"""
    if 'frame_transforms' in obj_data:
        frame_transforms = obj_data['frame_transforms']
        if frame_idx in frame_transforms:
            return True
    return False


def _get_object_transform_to_frame(obj_data, frame_idx):
    """获取从canonical空间到指定帧的变换矩阵（参考stage2_loss.py line 657-680）"""
    # 检查参考帧
    reference_frame = obj_data.get('reference_frame', 0)
    if frame_idx == reference_frame:
        # 如果要渲染的就是参考帧（canonical帧），不需要变换
        return None

    # 获取变换：frame_transforms存储的是从各帧到reference_frame的变换
    # 但我们需要从reference_frame到frame_idx的变换，所以需要求逆
    if 'frame_transforms' in obj_data:
        frame_transforms = obj_data['frame_transforms']
        if frame_idx in frame_transforms:
            # 存储的变换：frame_idx -> reference_frame
            # 我们需要的变换：reference_frame -> frame_idx（即逆变换）
            frame_to_canonical = frame_transforms[frame_idx]  # [4, 4] 变换矩阵
            canonical_to_frame = torch.inverse(frame_to_canonical)
            return canonical_to_frame

    return None


def _apply_transform_to_gaussians(gaussians, transform):
    """将变换应用到Gaussian参数（参考stage2_loss.py line 768-802）"""
    # gaussians: [N, 14] - [xyz(3), scale(3), color(3), quat(4), opacity(1)]
    # transform: [4, 4] 变换矩阵

    # 检查变换矩阵是否异常
    if torch.allclose(transform, torch.zeros_like(transform), atol=1e-6):
        print(f"⚠️  检测到零变换矩阵！使用单位矩阵替代")
        transform = torch.eye(4, dtype=transform.dtype, device=transform.device)
    else:
        # 转换为float32以支持torch.det
        det_val = torch.det(transform[:3, :3].float()).abs()
        if det_val < 1e-8:
            print(f"⚠️  变换矩阵奇异(det={det_val:.2e})！")

    transformed_gaussians = gaussians.clone()

    # 变换位置（参考stage2_loss.py line 794-800）
    positions = gaussians[:, :3]  # [N, 3]
    positions_homo = torch.cat([positions, torch.ones(
        positions.shape[0], 1, device=positions.device)], dim=1)  # [N, 4]
    transformed_positions = torch.mm(
        transform, positions_homo.T).T[:, :3]  # [N, 3]
    transformed_gaussians[:, :3] = transformed_positions

    # 注意：stage2_loss.py 中只变换了位置，没有变换旋转和尺度
    # 这是简化处理，完整实现需要变换quaternion和scale

    return transformed_gaussians


@torch.no_grad()
def run_single_inference(model, dataset, idx, num_views, device, cfg):
    """
    运行单个场景的推理（支持多相机）

    Args:
        model: VGGT模型
        dataset: 数据集
        idx: 场景索引
        num_views: 每个相机的帧数
        device: 设备
        cfg: 配置

    Returns:
        包含可视化数据的字典
    """
    print(f"\n{'='*60}")
    print(f"Processing scene index: {idx}")
    print(f"{'='*60}\n")

    try:
        # 加载数据
        views_list = dataset[idx]

        # 使用collate function构建batch格式（参考inference.py）
        from dataset import vggt_collate_fn
        vggt_batch = vggt_collate_fn([views_list])  # [B=1, S, ...]

        # Move to device
        for key in vggt_batch:
            if isinstance(vggt_batch[key], torch.Tensor):
                vggt_batch[key] = vggt_batch[key].to(device)

        print(f"Loaded batch: images shape = {vggt_batch['images'].shape}")

        # 提取相机和帧索引（保留Python list用于后续处理）
        camera_indices = [v.get('camera_idx', 0) for v in views_list]
        frame_indices = [v.get('frame_idx', 0) for v in views_list]

        # 获取数据集元信息
        num_cameras = len(set(camera_indices))
        num_frames_per_camera = len(camera_indices) // num_cameras

        print(f"Loaded {len(views_list)} views: {num_cameras} cameras × {num_frames_per_camera} frames")

        # 获取图像尺寸和数据
        images = vggt_batch['images']
        B, S, C, H, W = images.shape
        intrinsics = vggt_batch['intrinsics']
        extrinsics = vggt_batch['extrinsics']
        depthmaps = vggt_batch['depths']

        # 运行模型推理
        print("Running model inference...")
        preds = model(
            images,
            gt_extrinsics=extrinsics,
            gt_intrinsics=intrinsics,
            frame_sample_ratio=1.0
        )

        # 提取预测结果
        pred_depth = preds.get('depth', None)
        if pred_depth is not None:
            pred_depth = pred_depth[0].cpu().numpy()  # Could be [S, H, W] or [S, H, W, 1]
            # Handle extra dimension if present
            if pred_depth.ndim == 4 and pred_depth.shape[-1] == 1:
                pred_depth = pred_depth.squeeze(-1)  # [S, H, W, 1] -> [S, H, W]

        # 创建动态处理器（使用配置参数）
        dynamic_processor = OnlineDynamicProcessor(
            device=device,
            velocity_transform_mode=cfg.velocity_transform_mode,
            velocity_threshold=cfg.velocity_threshold,
            clustering_eps=cfg.clustering_eps,
            clustering_min_samples=cfg.clustering_min_samples,
            min_object_size=cfg.min_object_size,
            tracking_position_threshold=cfg.tracking_position_threshold,
            tracking_velocity_threshold=cfg.tracking_velocity_threshold
        )

        # 处理use_gt_camera参数
        preds_for_dynamic = preds.copy() if isinstance(preds, dict) else preds
        if hasattr(cfg, 'use_gt_camera') and cfg.use_gt_camera and 'pose_enc' in preds_for_dynamic:
            # 使用GT相机参数替换预测的pose_enc
            from vggt.utils.pose_enc import extri_intri_to_pose_encoding
            image_size_hw = (H, W)
            gt_pose_enc = extri_intri_to_pose_encoding(
                extrinsics, intrinsics, image_size_hw, pose_encoding_type="absT_quaR_FoV"
            )
            preds_for_dynamic['pose_enc'] = gt_pose_enc
            print(f"[INFO] Using GT camera parameters for dynamic object processing")

        # 创建空的辅助模型字典
        auxiliary_models = {}

        # Process dynamic objects
        dynamic_objects_data = dynamic_processor.process_dynamic_objects(
            preds_for_dynamic, vggt_batch, auxiliary_models
        )

        # 提取GT velocity
        gt_velocity = vggt_batch.get('flowmap', None)
        if gt_velocity is not None:
            gt_velocity = gt_velocity[0, :, :, :, :3]  # [S, H, W, 3]
            # 坐标转换
            gt_velocity = gt_velocity[:, :, :, [2, 0, 1]]
            gt_velocity[:, :, :, 2] = -gt_velocity[:, :, :, 2]
        else:
            gt_velocity = torch.zeros(len(views_list), H, W, 3, device=device)

        # 提取pred velocity (from preds)
        pred_velocity_tensor = preds.get('velocity', torch.zeros(1, len(views_list), H, W, 3, device=device))[0]
        # 坐标转换
        pred_velocity_tensor = pred_velocity_tensor[:, :, :, [2, 0, 1]]
        pred_velocity_tensor[:, :, :, 2] = -pred_velocity_tensor[:, :, :, 2]

        # 提取GT和pred segmentation
        gt_seg_labels = vggt_batch.get('segment_label', None)
        gt_seg_mask = vggt_batch.get('segment_mask', None)
        pred_seg_logits = preds.get('segment_logits', None)

        # 创建动态聚类可视化
        matched_clustering_results = dynamic_objects_data.get('matched_clustering_results', []) if dynamic_objects_data is not None else []
        clustering_vis_np = create_clustering_visualization(matched_clustering_results, vggt_batch)  # [S, H, W, 3] uint8

        # 转换为numpy用于可视化
        gt_rgb = (images[0].cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)  # [S, H, W, 3]
        gt_depth = depthmaps[0].cpu().numpy()  # [S, H, W]

        # 可视化深度
        print("Creating visualizations...")
        gt_depth_vis = visualize_depth(gt_depth)  # [S, H, W, 3] uint8
        pred_depth_vis = visualize_depth(pred_depth) if pred_depth is not None else np.zeros_like(gt_depth_vis)

        # 可视化速度（输入tensor，输出[S, 3, H, W] tensor）
        gt_velocity_vis = visualize_velocity(gt_velocity, scale=0.1)  # [S, 3, H, W] tensor
        pred_velocity_vis = visualize_velocity(pred_velocity_tensor, scale=0.1)  # [S, 3, H, W] tensor

        # 转换为numpy [S, H, W, 3] uint8
        gt_velocity_vis = (gt_velocity_vis.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
        pred_velocity_vis = (pred_velocity_vis.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

        # 可视化分割
        if gt_seg_labels is not None and pred_seg_logits is not None:
            print(f"[DEBUG] GT seg_labels shape: {gt_seg_labels.shape}, dtype: {gt_seg_labels.dtype}")
            print(f"[DEBUG] Pred seg_logits shape: {pred_seg_logits.shape}, dtype: {pred_seg_logits.dtype}")

            gt_seg_vis = visualize_segmentation(gt_seg_labels[0], gt_seg_mask[0] if gt_seg_mask is not None else None, num_classes=4)  # [S, H, W, 3] uint8

            # Check pred_seg_logits shape
            print(f"[DEBUG] pred_seg_logits[0] shape: {pred_seg_logits[0].shape}")
            pred_seg_probs = torch.softmax(pred_seg_logits[0], dim=-1)  # Should be [S, H, W, 4]
            print(f"[DEBUG] pred_seg_probs shape: {pred_seg_probs.shape}")
            print(f"[DEBUG] pred_seg_probs min/max: {pred_seg_probs.min().item():.4f} / {pred_seg_probs.max().item():.4f}")

            pred_seg_labels = torch.argmax(pred_seg_probs, dim=-1)  # [S, H, W]
            print(f"[DEBUG] pred_seg_labels shape: {pred_seg_labels.shape}")
            print(f"[DEBUG] pred_seg_labels unique values: {torch.unique(pred_seg_labels)}")

            pred_seg_vis = visualize_segmentation(pred_seg_labels, num_classes=4)  # [S, H, W, 3] uint8
        else:
            gt_seg_vis = np.zeros((len(views_list), H, W, 3), dtype=np.uint8)
            pred_seg_vis = np.zeros((len(views_list), H, W, 3), dtype=np.uint8)
            if gt_seg_labels is None:
                print("Warning: No GT segmentation found")
            if pred_seg_logits is None:
                print("Warning: No predicted segmentation found")

        # 构建scene并渲染RGB（参考inference.py）
        print("Building scene for rendering...")
        dynamic_objects_cars = dynamic_objects_data.get('dynamic_objects_cars', []) if dynamic_objects_data is not None else []
        dynamic_objects_people = dynamic_objects_data.get('dynamic_objects_people', []) if dynamic_objects_data is not None else []
        static_gaussians = dynamic_objects_data.get('static_gaussians') if dynamic_objects_data is not None else None

        scene = {
            'static_gaussians': static_gaussians,
            'dynamic_objects_cars': dynamic_objects_cars,
            'dynamic_objects_people': dynamic_objects_people
        }

        print(f"[INFO] Scene built: {len(dynamic_objects_cars)} cars, {len(dynamic_objects_people)} people")

        # 获取sky colors和sampled_frame_indices
        sky_colors_full = preds.get('sky_colors', None)  # [B, num_sampled, 3, H, W]
        sampled_frame_indices = preds.get('sampled_frame_indices', None)

        # Extract sky colors (remove batch dimension)
        if sky_colors_full is not None:
            sky_colors = sky_colors_full[0]  # [num_sampled, 3, H, W]
        else:
            sky_colors = None

        # 获取depth_scale_factor用于voxel pruning
        depth_scale_factor = vggt_batch.get('depth_scale_factor', None)
        if depth_scale_factor is not None and torch.is_tensor(depth_scale_factor):
            depth_scale_factor = depth_scale_factor.item()

        # 渲染RGB和深度（使用render_gaussians_with_sky）
        print("Rendering gaussians with sky...")
        rendered_rgb, rendered_depth = render_gaussians_with_sky(
            scene, intrinsics[0], extrinsics[0], sky_colors, sampled_frame_indices, H, W, device,
            enable_voxel_pruning=False, voxel_size=0.05, depth_scale_factor=depth_scale_factor
        )

        # 转换rendered_rgb为numpy [S, H, W, 3] uint8
        pred_rgb_np = (rendered_rgb.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)  # [S, 3, H, W] -> [S, H, W, 3]

        return {
            'gt_rgb': gt_rgb,
            'gt_depth': gt_depth_vis,
            'pred_depth': pred_depth_vis,
            'pred_velocity': pred_velocity_vis,
            'pred_rgb': pred_rgb_np,
            'gt_velocity': gt_velocity_vis,
            'gt_segmentation': gt_seg_vis,
            'pred_segmentation': pred_seg_vis,
            'dynamic_clustering': clustering_vis_np,
            'num_cameras': num_cameras,
            'num_frames': num_frames_per_camera,
            'camera_indices': camera_indices,
            'frame_indices': frame_indices,
            'success': True
        }

    except Exception as e:
        print(f"Error processing idx {idx}: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def create_multi_camera_grid(
    gt_rgb,
    gt_depth,
    pred_depth,
    pred_velocity,
    num_cameras,
    num_frames,
    camera_indices,
    frame_indices,
    pred_rgb=None,
    gt_velocity=None,
    gt_segmentation=None,
    pred_segmentation=None,
    dynamic_clustering=None
):
    """
    创建多相机可视化网格
    每行显示: GT (左中右) | Pred (左中右)
    相机顺序: 中(front), 左(left), 右(right) 对应 camera_id [2, 1, 3]

    Layout:
    - Row 1: GT RGB (左中右) | Rendered RGB (左中右)
    - Row 2: GT Depth (左中右) | Rendered Depth (左中右)
    - Row 3: GT Velocity (左中右) | GT RGB + Pred Velocity 融合 (左中右)
    - Row 4: GT Segmentation (左中右) | Pred Segmentation (左中右)
    - Row 5: Dynamic Clustering (左中右, full width)

    Args:
        gt_rgb: [S, H, W, 3] GT RGB图像
        gt_depth: [S, H, W, 3] GT深度图（已可视化）
        pred_depth: [S, H, W, 3] 预测深度图（已可视化）
        pred_velocity: [S, H, W, 3] 预测速度图（已可视化）
        num_cameras: 相机数量 (应该是3)
        num_frames: 每个相机的帧数
        camera_indices: 相机索引列表
        frame_indices: 帧索引列表
        pred_rgb: [S, H, W, 3] 预测的RGB图像（带天空）
        gt_velocity: [S, H, W, 3] GT速度图（已可视化）
        gt_segmentation: [S, H, W, 3] GT分割图（已可视化）
        pred_segmentation: [S, H, W, 3] 预测分割图（已可视化）
        dynamic_clustering: [S, H, W, 3] 动态聚类图（已可视化）

    Returns:
        视频帧列表，每帧是网格布局
    """
    # Debug: print shapes
    print(f"[DEBUG] create_multi_camera_grid inputs:")
    print(f"  gt_rgb shape: {gt_rgb.shape}")
    print(f"  num_cameras: {num_cameras}, num_frames: {num_frames}")
    print(f"  camera_indices: {camera_indices}")
    print(f"  frame_indices: {frame_indices}")
    if dynamic_clustering is not None:
        print(f"  dynamic_clustering shape: {dynamic_clustering.shape}")

    grid_frames = []

    # Camera reordering map: [cam1, cam2, cam3] -> [cam2(center), cam1(left), cam3(right)]
    # 假设数据按照camera_id [1, 2, 3]的顺序组织
    # camera_id: 1=左, 2=前/中, 3=右
    # 我们希望显示顺序: 中, 左, 右 -> [2, 1, 3] -> [1, 0, 2] (在索引中)
    camera_order = [1, 0, 2]  # 重新排列: cam2(idx=1), cam1(idx=0), cam3(idx=2)

    for frame_idx in range(num_frames):
        # 使用camera_indices和frame_indices来找到对应的view
        # camera_indices和frame_indices告诉我们每个view的相机ID和帧ID
        frame_views_original = []
        for cam_idx in range(num_cameras):
            # 找到属于当前camera和frame的view索引
            for view_idx in range(len(camera_indices)):
                if camera_indices[view_idx] == cam_idx and frame_indices[view_idx] == frame_idx:
                    frame_views_original.append(view_idx)
                    break

        # 重新排列相机顺序: 中, 左, 右
        if len(frame_views_original) == num_cameras:
            frame_views = [frame_views_original[i] for i in camera_order]
        else:
            print(f"[WARNING] frame_idx={frame_idx}: found {len(frame_views_original)} views, expected {num_cameras}")
            frame_views = frame_views_original
        print(f"[DEBUG] frame_idx={frame_idx}, frame_views={frame_views}")

        # === Row 1: GT RGB (左中右) | Rendered RGB (左中右) ===
        gt_rgb_concat = np.concatenate([gt_rgb[v] for v in frame_views], axis=1)
        if pred_rgb is not None:
            pred_rgb_concat = np.concatenate([pred_rgb[v] for v in frame_views], axis=1)
        else:
            # 如果没有预测RGB，使用GT RGB作为占位符
            pred_rgb_concat = gt_rgb_concat.copy()
        row1 = np.concatenate([gt_rgb_concat, pred_rgb_concat], axis=1)

        # === Row 2: GT Depth (左中右) | Rendered Depth (左中右) ===
        gt_depth_concat = np.concatenate([gt_depth[v] for v in frame_views], axis=1)
        if pred_depth is not None:
            pred_depth_concat = np.concatenate([pred_depth[v] for v in frame_views], axis=1)
        else:
            pred_depth_concat = np.zeros_like(gt_depth_concat)
        row2 = np.concatenate([gt_depth_concat, pred_depth_concat], axis=1)

        # === Row 3: GT Velocity (左中右) | GT RGB + Pred Velocity 融合 (左中右) ===
        if gt_velocity is not None:
            gt_velocity_concat = np.concatenate([gt_velocity[v] for v in frame_views], axis=1)
        else:
            # 如果没有GT velocity，使用零图作为占位符
            gt_velocity_concat = np.zeros_like(gt_depth_concat)

        if pred_velocity is not None:
            # 融合GT RGB和预测速度 (加权叠加)
            # 注意: gt_rgb和pred_velocity都是uint8 [0, 255]，需要归一化后融合
            fused_velocity = []
            velocity_alpha = 1.0  # 速度可视化的权重
            for v in frame_views:
                # 归一化到[0, 1]
                gt_rgb_norm = gt_rgb[v].astype(np.float32) / 255.0
                pred_velocity_norm = pred_velocity[v].astype(np.float32) / 255.0
                # 加权融合
                fused = velocity_alpha * pred_velocity_norm + (1 - velocity_alpha) * gt_rgb_norm
                fused = np.clip(fused, 0, 1)
                # 转回uint8
                fused = (fused * 255).astype(np.uint8)
                fused_velocity.append(fused)
            pred_velocity_concat = np.concatenate(fused_velocity, axis=1)
        else:
            pred_velocity_concat = gt_rgb_concat.copy()
        row3 = np.concatenate([gt_velocity_concat, pred_velocity_concat], axis=1)

        # === Row 4: GT Segmentation (左中右) | Pred Segmentation (左中右) ===
        if gt_segmentation is not None:
            gt_seg_concat = np.concatenate([gt_segmentation[v] for v in frame_views], axis=1)
        else:
            gt_seg_concat = np.zeros_like(gt_depth_concat)

        if pred_segmentation is not None:
            pred_seg_concat = np.concatenate([pred_segmentation[v] for v in frame_views], axis=1)
        else:
            pred_seg_concat = np.zeros_like(gt_depth_concat)
        row4 = np.concatenate([gt_seg_concat, pred_seg_concat], axis=1)

        # === Row 5: Dynamic Clustering (左中右) | 空白 (左中右) ===
        if dynamic_clustering is not None:
            # dynamic_clustering是[num_views, H, W, 3]，按视图组织
            # 左侧显示3个相机的clustering结果
            clustering_concat = np.concatenate([dynamic_clustering[v] for v in frame_views], axis=1)
            # 右侧留白（白色背景）
            H = clustering_concat.shape[0]
            W = clustering_concat.shape[1]
            white_space = np.ones((H, W, 3), dtype=np.uint8) * 255
            row5 = np.concatenate([clustering_concat, white_space], axis=1)
        else:
            # 使用与row1相同宽度的零图作为占位符 (需要6个相机的宽度)
            H = gt_rgb_concat.shape[0]
            W = row1.shape[1]  # 使用row1的宽度（6个相机宽度）
            row5 = np.zeros((H, W, 3), dtype=np.uint8)

        # 纵向堆叠所有行
        grid_frame = np.concatenate([
            row1,
            row2,
            row3,
            row4,
            row5
        ], axis=0)

        grid_frames.append(grid_frame)

    return grid_frames


def save_video(grid_frames, output_path, fps=10):
    """保存视频"""
    print(f"Saving video to: {output_path}")

    with iio.get_writer(output_path, fps=fps, codec='libx264', quality=8) as writer:
        for frame in grid_frames:
            writer.append_data(frame)

    print(f"Video saved! Frames: {len(grid_frames)}")


def save_images(grid_frames, output_dir, prefix="frame"):
    """保存图像帧"""
    print(f"Saving images to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    for i, frame in enumerate(grid_frames):
        output_path = os.path.join(output_dir, f"{prefix}_{i:04d}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print(f"Images saved! Total: {len(grid_frames)}")


def run_batch_inference(model, dataset, cfg, device):
    """运行批量推理"""
    print(f"\n{'='*60}")
    print(f"Batch Inference Mode")
    print(f"Range: {cfg.start_idx} to {cfg.end_idx}, step {cfg.step}")
    print(f"{'='*60}\n")

    successful = []
    failed = []

    indices = range(cfg.start_idx, cfg.end_idx, cfg.step)

    for idx in tqdm(indices, desc="Batch processing"):
        result = run_single_inference(model, dataset, idx, cfg.num_views, device, cfg)

        if result['success']:
            # Create visualization grid
            grid_frames = create_multi_camera_grid(
                result['gt_rgb'],
                result['gt_depth'],
                result['pred_depth'],
                result['pred_velocity'],
                result['num_cameras'],
                result['num_frames'],
                result['camera_indices'],
                result['frame_indices'],
                pred_rgb=result.get('pred_rgb'),
                gt_velocity=result.get('gt_velocity'),
                gt_segmentation=result.get('gt_segmentation'),
                pred_segmentation=result.get('pred_segmentation'),
                dynamic_clustering=result.get('dynamic_clustering')
            )

            output_prefix = cfg.output_prefix if hasattr(cfg, 'output_prefix') else "multi_cam"
            output_path = os.path.join(cfg.output_dir, f"{output_prefix}_idx{idx}.mp4")

            # Save video
            if cfg.save_video:
                save_video(grid_frames, output_path, fps=cfg.fps)

            # Save images
            if cfg.save_images:
                images_dir = os.path.join(cfg.output_dir, f"{output_prefix}_idx{idx}_frames")
                save_images(grid_frames, images_dir, prefix="frame")

            successful.append(idx)

            if cfg.continue_on_error:
                print(f"✓ Scene {idx} completed. Cameras: {result['num_cameras']}, Frames: {result['num_frames']}")
        else:
            failed.append(idx)
            if cfg.continue_on_error:
                print(f"✗ Scene {idx} failed: {result['error']}")
            else:
                raise RuntimeError(f"Scene {idx} failed: {result['error']}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Batch Inference Summary")
    print(f"{'='*60}")
    print(f"Successful: {len(successful)}/{len(indices)}")
    print(f"Failed: {len(failed)}/{len(indices)}")
    if failed:
        print(f"Failed indices: {failed}")
    print(f"{'='*60}\n")


@hydra.main(
    version_base=None,
    config_path="config/waymo",
    config_name="infer_multi",
)
def main(cfg: OmegaConf):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # import debugpy
    # debugpy.listen(5698)
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()

    # Load model and dataset
    model = load_model(cfg.model_path, device, cfg)
    dataset = load_dataset(cfg.infer_dataset)

    # Create output dir
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Run inference
    with tf32_off():
        if cfg.batch_mode:
            run_batch_inference(model, dataset, cfg, device)
        else:
            result = run_single_inference(model, dataset, cfg.single_idx, cfg.num_views, device, cfg)

            if result['success']:
                grid_frames = create_multi_camera_grid(
                    result['gt_rgb'],
                    result['gt_depth'],
                    result['pred_depth'],
                    result['pred_velocity'],
                    result['num_cameras'],
                    result['num_frames'],
                    result['camera_indices'],
                    result['frame_indices'],
                    pred_rgb=result.get('pred_rgb'),
                    gt_velocity=result.get('gt_velocity'),
                    gt_segmentation=result.get('gt_segmentation'),
                    pred_segmentation=result.get('pred_segmentation'),
                    dynamic_clustering=result.get('dynamic_clustering')
                )

                output_prefix = cfg.output_prefix if hasattr(cfg, 'output_prefix') else "multi_cam"
                output_path = os.path.join(cfg.output_dir, f"{output_prefix}_idx{cfg.single_idx}.mp4")

                # Save video
                if cfg.save_video:
                    save_video(grid_frames, output_path, fps=cfg.fps)

                # Save images
                if cfg.save_images:
                    images_dir = os.path.join(cfg.output_dir, f"{output_prefix}_idx{cfg.single_idx}_frames")
                    save_images(grid_frames, images_dir, prefix="frame")

                print(f"\n{'='*60}")
                print(f"Success! Cameras: {result['num_cameras']}, Frames: {result['num_frames']}")
                print(f"Output: {output_path}")
                print(f"{'='*60}\n")
            else:
                print(f"Failed: {result['error']}")


if __name__ == "__main__":
    main()
