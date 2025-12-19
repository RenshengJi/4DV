#!/usr/bin/env python3
"""
Multi-camera inference script for VGGT model.
Supports loading and visualizing results from multiple cameras simultaneously.
Each frame shows horizontally concatenated images from n cameras.

Usage:
python inference_multi.py

Override config via command line:
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
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models import VGGT
from src.utils import tf32_off
from src.dynamic_processing import DynamicProcessor
from src.losses import self_render_and_loss, prune_gaussians_by_voxel


def load_model(model_path, device, cfg):
    """Load VGGT model"""
    print(f"Loading model from: {model_path}")
    print(f"Model config: sh_degree={cfg.sh_degree}")

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
    """Load dataset"""
    from dataset import WaymoDataset, Waymo_Multi, ImgNorm

    print(f"Loading dataset from config...")

    dataset = eval(dataset_cfg)

    print(f"Dataset loaded: {len(dataset)} scenes")
    return dataset


def visualize_depth(depth):
    """
    Visualize depth as color image
    Args:
        depth: [S, H, W] or [S, H, W, 1] numpy array
    Returns:
        [S, H, W, 3] numpy array (uint8)
    """
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
    """Visualize velocity as RGB image

    Args:
        velocity: [S, H, W, 3] tensor, velocity field
        scale: Velocity scale factor
    Returns:
        [S, 3, H, W] tensor (float32, [0, 1])
    """
    from src.utils import scene_flow_to_rgb

    S, H, W, _ = velocity.shape
    velocity_rgb = scene_flow_to_rgb(velocity.detach(), scale).permute(0, 3, 1, 2)  # [S, 3, H, W]
    return velocity_rgb


def get_segmentation_colormap(num_classes=4):
    """Generate segmentation colormap based on Waymo classes (Cityscapes style)

    Waymo class definitions (4 classes):
        0: background/unlabeled
        1: vehicle
        2: sign
        3: pedestrian + cyclist
    """
    colormap = np.zeros((num_classes, 3), dtype=np.uint8)
    colormap[0] = [255, 255, 255]     # background - white
    colormap[1] = [0, 0, 142]         # vehicle - dark blue
    colormap[2] = [220, 220, 0]       # sign - yellow
    colormap[3] = [220, 20, 60]       # pedestrian+cyclist - red
    return colormap


def visualize_segmentation(seg_labels, seg_mask=None, num_classes=4):
    """Visualize segmentation labels as RGB image (returns [S, H, W, 3] uint8 numpy format)"""
    colormap = get_segmentation_colormap(num_classes)
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

    for s in range(S):
        for class_id in range(num_classes):
            mask = (seg_labels[s] == class_id)
            seg_rgb[s][mask] = colormap[class_id]

        if seg_mask is not None:
            invalid_mask = (seg_mask[s] == 0)
            seg_rgb[s][invalid_mask] = [128, 128, 128]

    return seg_rgb


def visualize_clustering_results(clustering_results, num_colors=20):
    """Generate colors for clustering results"""
    colors = plt.cm.tab20(np.linspace(0, 1, num_colors))
    colors = (colors[:, :3] * 255).astype(np.uint8)

    colored_results = []

    for frame_result in clustering_results:
        points = frame_result['points']
        labels = frame_result['labels']
        global_ids = frame_result.get('global_ids', [])

        point_colors = np.zeros((len(points), 3), dtype=np.uint8)

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
                continue

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
    """Create visualization from matched_clustering_results (returns [S, H, W, 3] uint8 numpy format)"""
    try:
        B, S, C, image_height, image_width = vggt_batch["images"].shape

        print(f"[DEBUG] create_clustering_visualization:")
        print(f"  vggt_batch['images'] shape: {vggt_batch['images'].shape}")
        print(f"  matched_clustering_results length: {len(matched_clustering_results) if matched_clustering_results else 0}")

        if not matched_clustering_results or len(matched_clustering_results) == 0:
            images_np = (vggt_batch["images"][0].cpu().numpy() * 255).astype(np.uint8)
            images_np = images_np.transpose(0, 2, 3, 1)
            print(f"[DEBUG] No clustering results, returning RGB images: {images_np.shape}")
            return images_np

        colored_results = visualize_clustering_results(matched_clustering_results, num_colors=20)
        print(f"[DEBUG] colored_results length: {len(colored_results)}")

        camera_indices = vggt_batch.get('camera_indices', None)
        frame_indices = vggt_batch.get('frame_indices', None)
        is_multi_camera = (camera_indices is not None and frame_indices is not None)

        if is_multi_camera:
            camera_indices = camera_indices[0].cpu().numpy() if isinstance(camera_indices, torch.Tensor) else camera_indices
            frame_indices = frame_indices[0].cpu().numpy() if isinstance(frame_indices, torch.Tensor) else frame_indices

            num_cameras = len(np.unique(camera_indices))
            num_frames = len(colored_results)

            print(f"[DEBUG] Multi-camera mode: {num_cameras} cameras, {num_frames} frames")

            clustering_images = []
            for view_idx in range(S):
                cam_idx = camera_indices[view_idx]
                frame_idx = frame_indices[view_idx]

                source_rgb = vggt_batch["images"][0, view_idx].permute(1, 2, 0)
                source_rgb = (source_rgb * 255).cpu().numpy().astype(np.uint8)

                colored_result = colored_results[frame_idx]
                point_colors = colored_result['colors']

                start_idx = cam_idx * image_height * image_width
                end_idx = start_idx + image_height * image_width
                camera_point_colors = point_colors[start_idx:end_idx]

                if colored_result['num_clusters'] > 0 and np.any(camera_point_colors > 0):
                    clustering_image = camera_point_colors.reshape(image_height, image_width, 3)

                    mask = np.any(clustering_image > 0, axis=2)[:, :, np.newaxis]
                    fused_image = np.where(mask,
                                         (fusion_alpha * clustering_image + (1 - fusion_alpha) * source_rgb).astype(np.uint8),
                                         source_rgb)
                else:
                    fused_image = source_rgb.copy()

                clustering_images.append(fused_image)
        else:
            clustering_images = []
            for frame_idx, colored_result in enumerate(colored_results):
                print(f"[DEBUG] Processing frame {frame_idx}, num_clusters: {colored_result['num_clusters']}")

                source_rgb = vggt_batch["images"][0, frame_idx].permute(1, 2, 0)
                source_rgb = (source_rgb * 255).cpu().numpy().astype(np.uint8)

                if colored_result['num_clusters'] > 0:
                    point_colors = colored_result['colors']
                    print(f"[DEBUG] point_colors shape: {point_colors.shape}, non-zero: {np.sum(np.any(point_colors > 0, axis=1))}")
                    clustering_image = point_colors.reshape(image_height, image_width, 3)

                    mask = np.any(clustering_image > 0, axis=2)
                    print(f"[DEBUG] mask sum: {np.sum(mask)}, total pixels: {mask.size}")
                    mask = mask[:, :, np.newaxis]

                    fused_image = np.where(mask,
                                         (fusion_alpha * clustering_image + (1 - fusion_alpha) * source_rgb).astype(np.uint8),
                                         source_rgb)
                else:
                    print(f"[DEBUG] No clusters for frame {frame_idx}")
                    fused_image = source_rgb.copy()
                    cv2.putText(fused_image, "No Dynamic Objects", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                clustering_images.append(fused_image)

        return np.array(clustering_images)

    except Exception as e:
        print(f"Error creating clustering visualization: {e}")
        import traceback
        traceback.print_exc()
        images_np = (vggt_batch["images"][0].cpu().numpy() * 255).astype(np.uint8)
        images_np = images_np.transpose(0, 2, 3, 1)
        return images_np


def render_gaussians_with_sky(scene, intrinsics, extrinsics, sky_colors, sampled_frame_indices, H, W, device,
                              enable_voxel_pruning=True, voxel_size=0.002, depth_scale_factor=None):
    """
    Render Gaussian scene with sky color alpha blending
    Frame-by-frame rendering for correct dynamic object transforms

    Args:
        enable_voxel_pruning: bool, whether to enable voxel pruning
        voxel_size: float, voxel size in metric scale (meters)
        depth_scale_factor: float, depth scaling factor
    """
    from gsplat import rasterization

    S = intrinsics.shape[0]
    rendered_images = []
    rendered_depths = []

    for frame_idx in range(S):
        all_means = []
        all_scales = []
        all_colors = []
        all_rotations = []
        all_opacities = []

        if scene.get('static_gaussians') is not None:
            static_gaussians = scene['static_gaussians']
            if static_gaussians.shape[0] > 0:
                all_means.append(static_gaussians[:, :3])
                all_scales.append(static_gaussians[:, 3:6])
                all_colors.append(static_gaussians[:, 6:9].unsqueeze(-2))
                all_rotations.append(static_gaussians[:, 9:13])
                all_opacities.append(static_gaussians[:, 13])

        dynamic_objects_cars = scene.get('dynamic_objects_cars', [])
        for obj_data in dynamic_objects_cars:
            if not _object_exists_in_frame(obj_data, frame_idx):
                continue

            canonical_gaussians = obj_data.get('canonical_gaussians')
            if canonical_gaussians is None or canonical_gaussians.shape[0] == 0:
                continue

            frame_transform = _get_object_transform_to_frame(obj_data, frame_idx)
            if frame_transform is None:
                transformed_gaussians = canonical_gaussians
            else:
                transformed_gaussians = _apply_transform_to_gaussians(
                    canonical_gaussians, frame_transform
                )

            if transformed_gaussians.shape[0] > 0:
                all_means.append(transformed_gaussians[:, :3])
                all_scales.append(transformed_gaussians[:, 3:6])
                all_colors.append(transformed_gaussians[:, 6:9].unsqueeze(-2))
                all_rotations.append(transformed_gaussians[:, 9:13])
                all_opacities.append(transformed_gaussians[:, 13])

        dynamic_objects_people = scene.get('dynamic_objects_people', [])
        for obj_data in dynamic_objects_people:
            frame_gaussians = obj_data.get('frame_gaussians', {})
            if frame_idx not in frame_gaussians:
                continue

            current_frame_gaussians = frame_gaussians[frame_idx]
            if current_frame_gaussians is None or current_frame_gaussians.shape[0] == 0:
                continue

            all_means.append(current_frame_gaussians[:, :3])
            all_scales.append(current_frame_gaussians[:, 3:6])
            all_colors.append(current_frame_gaussians[:, 6:9].unsqueeze(-2))
            all_rotations.append(current_frame_gaussians[:, 9:13])
            all_opacities.append(current_frame_gaussians[:, 13])

        if len(all_means) == 0:
            rendered_images.append(torch.zeros(3, H, W, device=device))
            rendered_depths.append(torch.zeros(H, W, device=device))
            continue

        means = torch.cat(all_means, dim=0)
        scales = torch.cat(all_scales, dim=0)
        colors = torch.cat(all_colors, dim=0)
        rotations = torch.cat(all_rotations, dim=0)
        opacities = torch.cat(all_opacities, dim=0)

        if enable_voxel_pruning and means.shape[0] > 0:
            means, scales, rotations, opacities, colors = prune_gaussians_by_voxel(
                means, scales, rotations, opacities, colors,
                voxel_size=voxel_size,
                depth_scale_factor=depth_scale_factor
            )

        means = torch.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
        scales = torch.nan_to_num(scales, nan=0.01, posinf=1.0, neginf=0.01)
        colors = torch.nan_to_num(colors, nan=0.5, posinf=1.0, neginf=0.0)
        rotations = torch.nan_to_num(rotations, nan=0.0, posinf=1.0, neginf=-1.0)
        opacities = torch.nan_to_num(opacities, nan=0.5, posinf=1.0, neginf=0.0)

        K = intrinsics[frame_idx]
        w2c = extrinsics[frame_idx]

        try:
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

            if sky_colors is not None and sampled_frame_indices is not None:
                if not isinstance(sampled_frame_indices, torch.Tensor):
                    sampled_frame_indices = torch.tensor(sampled_frame_indices, device=device)

                matches = (sampled_frame_indices == frame_idx)
                if matches.any():
                    sky_idx = matches.nonzero(as_tuple=True)[0].item()
                    frame_sky_color = sky_colors[sky_idx]
                    alpha_3ch = rendered_alpha.unsqueeze(0)
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
    Call self_render_and_loss to generate self-rendering results for visualization

    Returns:
        rendered_rgb: [S, 3, H, W] rendered RGB image
        rendered_depth: [S, H, W] rendered depth image
    """
    _, img_dict = self_render_and_loss(
        vggt_batch, preds,
        sampled_frame_indices=None,
        sh_degree=sh_degree,
        enable_voxel_pruning=enable_voxel_pruning,
        voxel_size=voxel_size
    )

    rendered_rgb = img_dict['self_rgb_pred']
    rendered_depth = img_dict['self_depth_pred'].squeeze(1)

    return rendered_rgb, rendered_depth


def _object_exists_in_frame(obj_data, frame_idx):
    """Check if dynamic object exists in specified frame"""
    if 'frame_transforms' in obj_data:
        frame_transforms = obj_data['frame_transforms']
        if frame_idx in frame_transforms:
            return True
    return False


def _get_object_transform_to_frame(obj_data, frame_idx):
    """Get transform matrix from canonical space to specified frame"""
    reference_frame = obj_data.get('reference_frame', 0)
    if frame_idx == reference_frame:
        return None

    if 'frame_transforms' in obj_data:
        frame_transforms = obj_data['frame_transforms']
        if frame_idx in frame_transforms:
            frame_to_canonical = frame_transforms[frame_idx]
            canonical_to_frame = torch.inverse(frame_to_canonical)
            return canonical_to_frame

    return None


def _apply_transform_to_gaussians(gaussians, transform):
    """Apply transform to Gaussian parameters"""
    if torch.allclose(transform, torch.zeros_like(transform), atol=1e-6):
        print(f"Warning: Zero transform matrix detected! Using identity matrix instead")
        transform = torch.eye(4, dtype=transform.dtype, device=transform.device)
    else:
        det_val = torch.det(transform[:3, :3].float()).abs()
        if det_val < 1e-8:
            print(f"Warning: Singular transform matrix (det={det_val:.2e})!")

    transformed_gaussians = gaussians.clone()

    positions = gaussians[:, :3]
    positions_homo = torch.cat([positions, torch.ones(
        positions.shape[0], 1, device=positions.device)], dim=1)
    transformed_positions = torch.mm(
        transform, positions_homo.T).T[:, :3]
    transformed_gaussians[:, :3] = transformed_positions

    return transformed_gaussians


@torch.no_grad()
def run_single_inference(model, dataset, idx, num_views, device, cfg):
    """
    Run inference on single scene (supports multi-camera)

    Args:
        model: VGGT model
        dataset: Dataset
        idx: Scene index
        num_views: Number of frames per camera
        device: Device
        cfg: Configuration

    Returns:
        Dictionary containing visualization data
    """
    print(f"\n{'='*60}")
    print(f"Processing scene index: {idx}")
    print(f"{'='*60}\n")

    try:
        views_list = dataset[idx]

        from dataset import vggt_collate_fn
        vggt_batch = vggt_collate_fn([views_list])

        for key in vggt_batch:
            if isinstance(vggt_batch[key], torch.Tensor):
                vggt_batch[key] = vggt_batch[key].to(device)

        print(f"Loaded batch: images shape = {vggt_batch['images'].shape}")

        camera_indices = [v.get('camera_idx', 0) for v in views_list]
        frame_indices = [v.get('frame_idx', 0) for v in views_list]

        num_cameras = len(set(camera_indices))
        num_frames_per_camera = len(camera_indices) // num_cameras

        print(f"Loaded {len(views_list)} views: {num_cameras} cameras × {num_frames_per_camera} frames")

        images = vggt_batch['images']
        B, S, C, H, W = images.shape
        intrinsics = vggt_batch['intrinsics']
        extrinsics = vggt_batch['extrinsics']
        depthmaps = vggt_batch['depths']

        print("Running model inference...")
        preds = model(
            images,
            gt_extrinsics=extrinsics,
            gt_intrinsics=intrinsics,
            frame_sample_ratio=1.0
        )

        pred_depth = preds.get('depth', None)
        if pred_depth is not None:
            pred_depth = pred_depth[0].cpu().numpy()
            if pred_depth.ndim == 4 and pred_depth.shape[-1] == 1:
                pred_depth = pred_depth.squeeze(-1)

        dynamic_processor = DynamicProcessor(
            device=device,
            velocity_threshold=cfg.velocity_threshold,
            clustering_eps=cfg.clustering_eps,
            clustering_min_samples=cfg.clustering_min_samples,
            min_object_size=cfg.min_object_size,
            tracking_position_threshold=cfg.tracking_position_threshold,
            registration_mode=cfg.velocity_transform_mode,
            use_registration=True
        )

        preds_for_dynamic = preds.copy() if isinstance(preds, dict) else preds
        if hasattr(cfg, 'use_gt_camera') and cfg.use_gt_camera and 'pose_enc' in preds_for_dynamic:
            from models.utils.pose_enc import extri_intri_to_pose_encoding
            image_size_hw = (H, W)
            gt_pose_enc = extri_intri_to_pose_encoding(
                extrinsics, intrinsics, image_size_hw, pose_encoding_type="absT_quaR_FoV"
            )
            preds_for_dynamic['pose_enc'] = gt_pose_enc
            print(f"[INFO] Using GT camera parameters for dynamic object processing")

        auxiliary_models = {}

        result = dynamic_processor.process(preds_for_dynamic, vggt_batch)

        dynamic_objects_data = dynamic_processor.to_legacy_format(result)

        gt_velocity = vggt_batch.get('flowmap', None)
        if gt_velocity is not None:
            gt_velocity = gt_velocity[0, :, :, :, :3]
            gt_velocity = gt_velocity[:, :, :, [2, 0, 1]]
            gt_velocity[:, :, :, 2] = -gt_velocity[:, :, :, 2]
        else:
            gt_velocity = torch.zeros(len(views_list), H, W, 3, device=device)

        pred_velocity_tensor = preds.get('velocity', torch.zeros(1, len(views_list), H, W, 3, device=device))[0]
        pred_velocity_tensor = pred_velocity_tensor[:, :, :, [2, 0, 1]]
        pred_velocity_tensor[:, :, :, 2] = -pred_velocity_tensor[:, :, :, 2]

        gt_seg_labels = vggt_batch.get('segment_label', None)
        gt_seg_mask = vggt_batch.get('segment_mask', None)
        pred_seg_logits = preds.get('segment_logits', None)

        matched_clustering_results = dynamic_objects_data.get('matched_clustering_results', []) if dynamic_objects_data is not None else []
        clustering_vis_np = create_clustering_visualization(matched_clustering_results, vggt_batch)

        gt_rgb = (images[0].cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
        gt_depth = depthmaps[0].cpu().numpy()

        print("Creating visualizations...")
        gt_depth_vis = visualize_depth(gt_depth)
        pred_depth_vis = visualize_depth(pred_depth) if pred_depth is not None else np.zeros_like(gt_depth_vis)

        gt_velocity_vis = visualize_velocity(gt_velocity, scale=0.1)
        pred_velocity_vis = visualize_velocity(pred_velocity_tensor, scale=0.1)

        gt_velocity_vis = (gt_velocity_vis.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
        pred_velocity_vis = (pred_velocity_vis.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

        if gt_seg_labels is not None and pred_seg_logits is not None:
            print(f"[DEBUG] GT seg_labels shape: {gt_seg_labels.shape}, dtype: {gt_seg_labels.dtype}")
            print(f"[DEBUG] Pred seg_logits shape: {pred_seg_logits.shape}, dtype: {pred_seg_logits.dtype}")

            gt_seg_vis = visualize_segmentation(gt_seg_labels[0], gt_seg_mask[0] if gt_seg_mask is not None else None, num_classes=4)

            print(f"[DEBUG] pred_seg_logits[0] shape: {pred_seg_logits[0].shape}")
            pred_seg_probs = torch.softmax(pred_seg_logits[0], dim=-1)
            print(f"[DEBUG] pred_seg_probs shape: {pred_seg_probs.shape}")
            print(f"[DEBUG] pred_seg_probs min/max: {pred_seg_probs.min().item():.4f} / {pred_seg_probs.max().item():.4f}")

            pred_seg_labels = torch.argmax(pred_seg_probs, dim=-1)
            print(f"[DEBUG] pred_seg_labels shape: {pred_seg_labels.shape}")
            print(f"[DEBUG] pred_seg_labels unique values: {torch.unique(pred_seg_labels)}")

            pred_seg_vis = visualize_segmentation(pred_seg_labels, num_classes=4)
        else:
            gt_seg_vis = np.zeros((len(views_list), H, W, 3), dtype=np.uint8)
            pred_seg_vis = np.zeros((len(views_list), H, W, 3), dtype=np.uint8)
            if gt_seg_labels is None:
                print("Warning: No GT segmentation found")
            if pred_seg_logits is None:
                print("Warning: No predicted segmentation found")

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

        sky_colors_full = preds.get('sky_colors', None)
        sampled_frame_indices = preds.get('sampled_frame_indices', None)

        if sky_colors_full is not None:
            sky_colors = sky_colors_full[0]
        else:
            sky_colors = None

        depth_scale_factor = vggt_batch.get('depth_scale_factor', None)
        if depth_scale_factor is not None and torch.is_tensor(depth_scale_factor):
            depth_scale_factor = depth_scale_factor.item()

        print("Rendering gaussians with sky...")
        rendered_rgb, rendered_depth = render_gaussians_with_sky(
            scene, intrinsics[0], extrinsics[0], sky_colors, sampled_frame_indices, H, W, device,
            enable_voxel_pruning=False, voxel_size=0.05, depth_scale_factor=depth_scale_factor
        )

        pred_rgb_np = (rendered_rgb.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

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


def add_text_label(image, text, font_scale=1.0, thickness=2):
    """
    Add text label to the top-center of an image

    Args:
        image: numpy array [H, W, 3]
        text: text to add
        font_scale: font size scale
        thickness: text thickness

    Returns:
        Image with text label added at top-center
    """
    # Get text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Create label background
    label_height = text_height + baseline + 20  # 20 pixels padding
    label_bg = np.ones((label_height, image.shape[1], 3), dtype=np.uint8) * 255

    # Calculate text position (center-top)
    text_x = (image.shape[1] - text_width) // 2
    text_y = text_height + 10  # 10 pixels from top

    # Draw text on label background
    cv2.putText(label_bg, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Concatenate label with image
    return np.concatenate([label_bg, image], axis=0)


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
    Create multi-camera visualization grid
    Each row shows: GT (left-center-right) | Pred (left-center-right)
    Camera order: center(front), left, right corresponding to camera_id [2, 1, 3]

    Layout:
    - Row 1: GT RGB (left-center-right) | Rendered RGB (left-center-right)
    - Row 2: GT Depth (left-center-right) | Rendered Depth (left-center-right)
    - Row 3: GT Velocity (left-center-right) | GT RGB + Pred Velocity fusion (left-center-right)
    - Row 4: GT Segmentation (left-center-right) | Pred Segmentation (left-center-right)
    - Row 5: Dynamic Clustering (left-center-right, full width)

    Args:
        gt_rgb: [S, H, W, 3] GT RGB image
        gt_depth: [S, H, W, 3] GT depth map (visualized)
        pred_depth: [S, H, W, 3] Predicted depth map (visualized)
        pred_velocity: [S, H, W, 3] Predicted velocity map (visualized)
        num_cameras: Number of cameras (should be 3)
        num_frames: Number of frames per camera
        camera_indices: List of camera indices
        frame_indices: List of frame indices
        pred_rgb: [S, H, W, 3] Predicted RGB image (with sky)
        gt_velocity: [S, H, W, 3] GT velocity map (visualized)
        gt_segmentation: [S, H, W, 3] GT segmentation map (visualized)
        pred_segmentation: [S, H, W, 3] Predicted segmentation map (visualized)
        dynamic_clustering: [S, H, W, 3] Dynamic clustering map (visualized)

    Returns:
        List of video frames, each frame is a grid layout
    """
    print(f"[DEBUG] create_multi_camera_grid inputs:")
    print(f"  gt_rgb shape: {gt_rgb.shape}")
    print(f"  num_cameras: {num_cameras}, num_frames: {num_frames}")
    print(f"  camera_indices: {camera_indices}")
    print(f"  frame_indices: {frame_indices}")
    if dynamic_clustering is not None:
        print(f"  dynamic_clustering shape: {dynamic_clustering.shape}")

    grid_frames = []

    # Set camera order based on number of cameras
    if num_cameras == 3:
        camera_order = [1, 0, 2]  # Center, left, right for 3 cameras
    else:
        camera_order = list(range(num_cameras))  # Natural order for other cases

    for frame_idx in range(num_frames):
        frame_views_original = []
        for cam_idx in range(num_cameras):
            for view_idx in range(len(camera_indices)):
                if camera_indices[view_idx] == cam_idx and frame_indices[view_idx] == frame_idx:
                    frame_views_original.append(view_idx)
                    break

        if len(frame_views_original) == num_cameras:
            frame_views = [frame_views_original[i] for i in camera_order]
        else:
            print(f"[WARNING] frame_idx={frame_idx}: found {len(frame_views_original)} views, expected {num_cameras}")
            frame_views = frame_views_original
        print(f"[DEBUG] frame_idx={frame_idx}, frame_views={frame_views}")

        # Concatenate multi-camera images without gaps
        gt_rgb_concat = np.concatenate([gt_rgb[v] for v in frame_views], axis=1)
        if pred_rgb is not None:
            pred_rgb_concat = np.concatenate([pred_rgb[v] for v in frame_views], axis=1)
        else:
            pred_rgb_concat = gt_rgb_concat.copy()

        # Add labels to GT and Pred RGB, then concatenate with white gap
        camera_label = "" if num_cameras == 1 else ""  # Only label once for multi-camera
        gt_rgb_labeled = add_text_label(gt_rgb_concat, f"GT RGB {camera_label}(t={frame_idx})")
        pred_rgb_labeled = add_text_label(pred_rgb_concat, f"Pred RGB {camera_label}(t={frame_idx})")
        # Add white gap between GT and Pred
        gap_width = 20
        gap_rgb = np.ones((gt_rgb_labeled.shape[0], gap_width, 3), dtype=np.uint8) * 255
        row1 = np.concatenate([gt_rgb_labeled, gap_rgb, pred_rgb_labeled], axis=1)

        gt_depth_concat = np.concatenate([gt_depth[v] for v in frame_views], axis=1)
        if pred_depth is not None:
            pred_depth_concat = np.concatenate([pred_depth[v] for v in frame_views], axis=1)
        else:
            pred_depth_concat = np.zeros_like(gt_depth_concat)

        # Add labels and gap for depth
        gt_depth_labeled = add_text_label(gt_depth_concat, f"GT Depth (t={frame_idx})")
        pred_depth_labeled = add_text_label(pred_depth_concat, f"Pred Depth (t={frame_idx})")
        gap_depth = np.ones((gt_depth_labeled.shape[0], gap_width, 3), dtype=np.uint8) * 255
        row2 = np.concatenate([gt_depth_labeled, gap_depth, pred_depth_labeled], axis=1)

        if gt_velocity is not None:
            gt_velocity_concat = np.concatenate([gt_velocity[v] for v in frame_views], axis=1)
        else:
            gt_velocity_concat = np.zeros_like(gt_depth_concat)

        if pred_velocity is not None:
            fused_velocity = []
            velocity_alpha = 1.0
            for v in frame_views:
                gt_rgb_norm = gt_rgb[v].astype(np.float32) / 255.0
                pred_velocity_norm = pred_velocity[v].astype(np.float32) / 255.0
                fused = velocity_alpha * pred_velocity_norm + (1 - velocity_alpha) * gt_rgb_norm
                fused = np.clip(fused, 0, 1)
                fused = (fused * 255).astype(np.uint8)
                fused_velocity.append(fused)
            pred_velocity_concat = np.concatenate(fused_velocity, axis=1)
        else:
            pred_velocity_concat = gt_rgb_concat.copy()

        # Add labels and gap for velocity
        gt_velocity_labeled = add_text_label(gt_velocity_concat, f"GT Velocity (t={frame_idx})")
        pred_velocity_labeled = add_text_label(pred_velocity_concat, f"Pred Velocity (t={frame_idx})")
        gap_velocity = np.ones((gt_velocity_labeled.shape[0], gap_width, 3), dtype=np.uint8) * 255
        row3 = np.concatenate([gt_velocity_labeled, gap_velocity, pred_velocity_labeled], axis=1)

        if gt_segmentation is not None:
            gt_seg_concat = np.concatenate([gt_segmentation[v] for v in frame_views], axis=1)
        else:
            gt_seg_concat = np.zeros_like(gt_depth_concat)

        if pred_segmentation is not None:
            pred_seg_concat = np.concatenate([pred_segmentation[v] for v in frame_views], axis=1)
        else:
            pred_seg_concat = np.zeros_like(gt_depth_concat)

        # Add labels and gap for segmentation
        gt_seg_labeled = add_text_label(gt_seg_concat, f"GT Segmentation (t={frame_idx})")
        pred_seg_labeled = add_text_label(pred_seg_concat, f"Pred Segmentation (t={frame_idx})")
        gap_seg = np.ones((gt_seg_labeled.shape[0], gap_width, 3), dtype=np.uint8) * 255
        row4 = np.concatenate([gt_seg_labeled, gap_seg, pred_seg_labeled], axis=1)

        if dynamic_clustering is not None:
            clustering_concat = np.concatenate([dynamic_clustering[v] for v in frame_views], axis=1)
            # Add label for clustering (full width)
            clustering_labeled = add_text_label(clustering_concat, f"Dynamic Clustering (t={frame_idx})")
            # Make row5 full width by adding white space
            H = clustering_labeled.shape[0]
            target_W = row1.shape[1]
            current_W = clustering_labeled.shape[1]
            if current_W < target_W:
                white_space = np.ones((H, target_W - current_W, 3), dtype=np.uint8) * 255
                row5 = np.concatenate([clustering_labeled, white_space], axis=1)
            else:
                row5 = clustering_labeled
        else:
            H = row1.shape[0]
            W = row1.shape[1]
            row5 = np.zeros((H, W, 3), dtype=np.uint8)

        # Add white gaps between rows
        row_gap_height = 15
        row_gap = np.ones((row_gap_height, row1.shape[1], 3), dtype=np.uint8) * 255

        grid_frame = np.concatenate([
            row1,
            row_gap,
            row2,
            row_gap,
            row3,
            row_gap,
            row4,
            row_gap,
            row5
        ], axis=0)

        grid_frames.append(grid_frame)

    return grid_frames


def save_video(grid_frames, output_path, fps=10):
    """Save video"""
    print(f"Saving video to: {output_path}")

    with iio.get_writer(output_path, fps=fps, codec='libx264', quality=8) as writer:
        for frame in grid_frames:
            writer.append_data(frame)

    print(f"Video saved! Frames: {len(grid_frames)}")


def save_images(grid_frames, output_dir, prefix="frame"):
    """Save image frames"""
    print(f"Saving images to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    for i, frame in enumerate(grid_frames):
        output_path = os.path.join(output_dir, f"{prefix}_{i:04d}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print(f"Images saved! Total: {len(grid_frames)}")


def run_batch_inference(model, dataset, cfg, device):
    """Run batch inference"""
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

            if cfg.save_video:
                save_video(grid_frames, output_path, fps=cfg.fps)

            if cfg.save_images:
                images_dir = os.path.join(cfg.output_dir, f"{output_prefix}_idx{idx}_frames")
                save_images(grid_frames, images_dir, prefix="frame")

            successful.append(idx)

            if cfg.continue_on_error:
                print(f"Scene {idx} completed. Cameras: {result['num_cameras']}, Frames: {result['num_frames']}")
        else:
            failed.append(idx)
            if cfg.continue_on_error:
                print(f"Scene {idx} failed: {result['error']}")
            else:
                raise RuntimeError(f"Scene {idx} failed: {result['error']}")

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

    if cfg.get("debug", False):
        import debugpy
        debugpy.listen(5698)
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()

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
