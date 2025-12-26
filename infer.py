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

# Import visualization utilities
from src.visualization import (
    visualize_depth,
    visualize_velocity,
    visualize_segmentation,
    get_segmentation_colormap,
    create_clustering_visualization,
    visualize_clustering_results,
    create_multi_camera_grid,
    create_context_reference_row,
    add_text_label
)

# Import rendering utilities
from src.rendering import (
    render_gaussians_with_sky,
    prepare_target_frame_transforms,
    object_exists_in_frame,
    get_object_transform_to_frame,
    apply_transform_to_gaussians,
    interpolate_transforms
)

# Import frame mapping utilities
from src.utils.frame_mapping import remap_dynamic_objects_frame_indices


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



@torch.no_grad()
def run_single_inference(model, dataset, idx, num_context_frames, device, cfg, render_target_frames=False, start_frame=None):
    """
    Run inference on single scene (supports multi-camera)

    Args:
        model: VGGT model
        dataset: Dataset
        idx: Scene index
        num_context_frames: Number of context frames (sparse frames for inference)
        device: Device
        cfg: Configuration
        render_target_frames: Whether to render target frames in addition to context frames
        start_frame: Start frame position (None for random, int for fixed)

    Returns:
        Dictionary containing visualization data with context/target frame distinction
    """
    print(f"\n{'='*60}")
    print(f"Processing scene index: {idx}")
    print(f"Render target frames: {render_target_frames}")
    if start_frame is not None:
        print(f"Start frame: {start_frame}")
    print(f"{'='*60}\n")

    try:
        # ==================== Load and Prepare Data ====================
        if start_frame is not None and hasattr(dataset, 'get_views_with_start_frame'):
            views_list = dataset.get_views_with_start_frame(idx, start_frame=start_frame)
        else:
            views_list = dataset[idx]

        from dataset import vggt_collate_fn
        vggt_batch = vggt_collate_fn([views_list])

        for key in vggt_batch:
            if isinstance(vggt_batch[key], torch.Tensor):
                vggt_batch[key] = vggt_batch[key].to(device)

        # Extract basic metadata
        is_context_frame = vggt_batch['is_context_frame']
        camera_indices_tensor = vggt_batch['camera_indices']
        frame_indices_tensor = vggt_batch['frame_indices']

        # Convert to lists for compatibility with existing code
        camera_indices = camera_indices_tensor[0].cpu().tolist()
        frame_indices = frame_indices_tensor[0].cpu().tolist()

        # Get num_cameras and num_total_frames from batch metadata
        num_cameras = vggt_batch['num_cameras'][0].item() if vggt_batch['num_cameras'] is not None else len(set(camera_indices))
        num_total_frames = vggt_batch['num_total_frames'][0].item() if vggt_batch['num_total_frames'] is not None else (len(views_list) // num_cameras if num_cameras > 0 else len(views_list))

        # Separate context and target frames
        context_mask = is_context_frame[0]  # [S]
        context_indices = torch.where(context_mask)[0]
        target_indices = torch.where(~context_mask)[0]

        print(f"Loaded {len(views_list)} views: {num_cameras} cameras × {num_total_frames} frames ( context frames: {len(context_indices)}, target frames: {len(target_indices)} )")

        images = vggt_batch['images']
        B, S, C, H, W = images.shape
        intrinsics = vggt_batch['intrinsics']
        extrinsics = vggt_batch['extrinsics']
        depthmaps = vggt_batch['depths']

        # ==================== Model Inference on Context Frames ====================
        print(f"Running model inference on {images[:, context_indices].shape[1]} frames...")
        preds = model(
            images[:, context_indices],
            gt_extrinsics=extrinsics[:, context_indices],
            gt_intrinsics=intrinsics[:, context_indices],
            frame_sample_ratio=1.0
        )

        # ==================== Dynamic Scene Processing ====================
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

        # Create context-only batch for dynamic processor with remapped frame indices
        # Remap frame indices to continuous [0, 1, 2, ...] for processor
        original_frame_indices = vggt_batch['frame_indices'][:, context_indices]
        unique_frames = torch.unique(original_frame_indices, sorted=True)
        frame_mapping = {int(f.item()): i for i, f in enumerate(unique_frames)}

        context_frame_indices = torch.tensor(
            [[frame_mapping[int(idx.item())] for idx in original_frame_indices[0]]],
            device=device
        )

        vggt_batch_context = {
            'images': images[:, context_indices],
            'depths': vggt_batch['depths'][:, context_indices] if is_context_frame is not None else vggt_batch['depths'],
            'intrinsics': intrinsics[:, context_indices],
            'extrinsics': extrinsics[:, context_indices],
            'point_masks': vggt_batch['point_masks'][:, context_indices] if is_context_frame is not None and vggt_batch.get('point_masks') is not None else vggt_batch.get('point_masks'),
            'world_points': vggt_batch['world_points'][:, context_indices] if is_context_frame is not None and vggt_batch.get('world_points') is not None else vggt_batch.get('world_points'),
            'flowmap': vggt_batch['flowmap'][:, context_indices] if is_context_frame is not None and vggt_batch.get('flowmap') is not None else vggt_batch.get('flowmap'),
            'segment_label': vggt_batch['segment_label'][:, context_indices] if is_context_frame is not None and vggt_batch.get('segment_label') is not None else vggt_batch.get('segment_label'),
            'segment_mask': vggt_batch['segment_mask'][:, context_indices] if is_context_frame is not None and vggt_batch.get('segment_mask') is not None else vggt_batch.get('segment_mask'),
            'depth_scale_factor': vggt_batch.get('depth_scale_factor'),
            'sky_masks': vggt_batch['sky_masks'][:, context_indices] if is_context_frame is not None and vggt_batch.get('sky_masks') is not None else vggt_batch.get('sky_masks'),
            'camera_indices': vggt_batch['camera_indices'][:, context_indices] if is_context_frame is not None and vggt_batch.get('camera_indices') is not None else vggt_batch.get('camera_indices'),
            'frame_indices': context_frame_indices,
        }

        result = dynamic_processor.process(preds, vggt_batch_context)
        dynamic_objects_data = dynamic_processor.to_legacy_format(result)
        # Remap dynamic objects frame indices from context to global
        remap_dynamic_objects_frame_indices(dynamic_objects_data, frame_mapping, device)

        # ==================== Sky Color Inference (Context or Context+Target) ====================
        sky_token = preds.get('sky_token', None)
        sky_colors_full = preds.get('sky_colors', None)

        # Infer sky colors for all frames when rendering target frames
        if render_target_frames and is_context_frame is not None and sky_token is not None:
            print(f"[INFO] Inferring sky colors for all frames (context + target)...")

            # Get all intrinsics and extrinsics (context + target)
            all_intrinsics = intrinsics[0]
            all_extrinsics = extrinsics[0]

            # Infer sky colors for all frames at once
            sky_colors = model.infer_target_frame_sky_colors(
                sky_token=sky_token,
                target_frame_intrinsics=all_intrinsics,
                target_frame_extrinsics=all_extrinsics,
                image_size=(H, W)
            )
            sampled_frame_indices = torch.arange(len(views_list), device=device)
            print(f"[INFO] Generated sky colors for all {len(views_list)} frames")
        else:
            # Use context frame sky colors from model output
            sky_colors = sky_colors_full[0] if sky_colors_full is not None else None
            sampled_frame_indices = preds.get('sampled_frame_indices', None)

        # ==================== Gaussian Rendering (Context or Context+Target) ====================
        print("Building scene for rendering...")
        dynamic_objects_cars = dynamic_objects_data.get('dynamic_objects_cars', []) if dynamic_objects_data is not None else []
        dynamic_objects_people = dynamic_objects_data.get('dynamic_objects_people', []) if dynamic_objects_data is not None else []
        static_gaussians = dynamic_objects_data.get('static_gaussians') if dynamic_objects_data is not None else None

        # Pre-compute interpolated transforms for target frames
        if render_target_frames and is_context_frame is not None and len(target_indices) > 0:
            print(f"[INFO] Pre-computing interpolated transforms for target frames...")
            num_total_frames = len(torch.unique(vggt_batch['frame_indices']))
            prepare_target_frame_transforms(dynamic_objects_cars, num_total_frames, device)
            print(f"[INFO] Transforms prepared for all {num_total_frames} frames")

        scene = {
            'static_gaussians': static_gaussians,
            'dynamic_objects_cars': dynamic_objects_cars,
            'dynamic_objects_people': dynamic_objects_people
        }

        print(f"[INFO] Scene built: {len(dynamic_objects_cars)} cars, {len(dynamic_objects_people)} people")

        depth_scale_factor = vggt_batch.get('depth_scale_factor', None)
        if depth_scale_factor is not None and torch.is_tensor(depth_scale_factor):
            depth_scale_factor = depth_scale_factor.item()

        # Determine which frames to render based on render_target_frames setting
        if render_target_frames or is_context_frame is None:
            frames_to_render_indices = list(range(len(views_list)))
            print(f"[INFO] Rendering all {len(views_list)} frames")
        else:
            frames_to_render_indices = context_indices.cpu().tolist()
            print(f"[INFO] Rendering only {len(frames_to_render_indices)} context frames")

        frames_to_render_intrinsics = intrinsics[0, frames_to_render_indices]
        frames_to_render_extrinsics = extrinsics[0, frames_to_render_indices]
        frames_to_render_temporal_indices = [frame_indices[i] for i in frames_to_render_indices]


        print("Rendering gaussians with sky...")
        rendered_rgb, rendered_depth = render_gaussians_with_sky(
            scene, frames_to_render_intrinsics, frames_to_render_extrinsics, sky_colors, sampled_frame_indices, H, W, device,
            enable_voxel_pruning=cfg.enable_voxel_pruning, voxel_size=cfg.voxel_size, depth_scale_factor=depth_scale_factor,
            temporal_frame_indices=frames_to_render_temporal_indices
        )

        pred_rgb_np = (rendered_rgb.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
        rendered_depth_np = rendered_depth.cpu().numpy()

        # ==================== Organize Visualization Data ====================
        # --- Ground Truth Data ---
        gt_rgb = (images[0].cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
        gt_depth = depthmaps[0].cpu().numpy()
        gt_depth_vis = visualize_depth(gt_depth)

        gt_velocity = vggt_batch.get('flowmap', None)
        if gt_velocity is not None:
            gt_velocity = gt_velocity[0, :, :, :, :3]
            gt_velocity = gt_velocity[:, :, :, [2, 0, 1]]
            gt_velocity[:, :, :, 2] = -gt_velocity[:, :, :, 2]
        else:
            gt_velocity = torch.zeros(len(views_list), H, W, 3, device=device)
        gt_velocity_vis = visualize_velocity(gt_velocity, scale=0.1)
        gt_velocity_vis = (gt_velocity_vis.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

        gt_seg_labels = vggt_batch.get('segment_label', None)
        gt_seg_mask = vggt_batch.get('segment_mask', None)
        if gt_seg_labels is not None:
            gt_seg_vis = visualize_segmentation(gt_seg_labels[0], gt_seg_mask[0] if gt_seg_mask is not None else None, num_classes=4)
        else:
            gt_seg_vis = np.zeros((len(views_list), H, W, 3), dtype=np.uint8)
            print("Warning: No GT segmentation found")

        # --- Predicted Data ---
        print("Creating visualizations...")
        rendered_depth_vis = visualize_depth(rendered_depth_np)

        # pred_velocity only contains context frames
        num_context = len(context_indices) if is_context_frame is not None else len(views_list)
        pred_velocity_raw = preds.get('velocity', torch.zeros(1, num_context, H, W, 3, device=device))[0]
        pred_velocity_raw = pred_velocity_raw[:, :, :, [2, 0, 1]]
        pred_velocity_raw[:, :, :, 2] = -pred_velocity_raw[:, :, :, 2]

        # Expand pred_velocity to all frames (fill target frames with zeros for visualization)
        pred_velocity_tensor = torch.zeros(len(views_list), H, W, 3, device=device)
        if is_context_frame is not None:
            pred_velocity_tensor[context_indices] = pred_velocity_raw
        else:
            pred_velocity_tensor = pred_velocity_raw
        pred_velocity_vis = visualize_velocity(pred_velocity_tensor, scale=0.1)
        pred_velocity_vis = (pred_velocity_vis.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

        pred_seg_logits_context = preds.get('segment_logits', None)
        if pred_seg_logits_context is not None:
            # Expand pred_seg to all frames
            pred_seg_probs_context = torch.softmax(pred_seg_logits_context[0], dim=-1)
            pred_seg_labels_context = torch.argmax(pred_seg_probs_context, dim=-1)

            # Create full prediction array (fill target frames with zeros)
            pred_seg_labels_full = torch.zeros((len(views_list), H, W), dtype=pred_seg_labels_context.dtype, device=device)
            if is_context_frame is not None:
                pred_seg_labels_full[context_indices] = pred_seg_labels_context
            else:
                pred_seg_labels_full = pred_seg_labels_context

            pred_seg_vis = visualize_segmentation(pred_seg_labels_full, num_classes=4)
        else:
            pred_seg_vis = np.zeros((len(views_list), H, W, 3), dtype=np.uint8)
            print("Warning: No predicted segmentation found")

        # --- Clustering Visualization ---
        matched_clustering_results = dynamic_objects_data.get('matched_clustering_results', []) if dynamic_objects_data is not None else []
        clustering_vis_np = create_clustering_visualization(matched_clustering_results, vggt_batch)

        # --- Filter to Context Frames Only (if needed) ---
        # When render_target_frames=False, only return context frame data
        if not render_target_frames and is_context_frame is not None and len(context_indices) > 0:
            # Filter all visualization data to context frames only
            gt_rgb_filtered = gt_rgb[context_indices.cpu().numpy()]
            gt_depth_vis_filtered = gt_depth_vis[context_indices.cpu().numpy()]
            gt_velocity_vis_filtered = gt_velocity_vis[context_indices.cpu().numpy()]
            pred_velocity_vis_filtered = pred_velocity_vis[context_indices.cpu().numpy()]
            gt_seg_vis_filtered = gt_seg_vis[context_indices.cpu().numpy()]
            pred_seg_vis_filtered = pred_seg_vis[context_indices.cpu().numpy()]
            if clustering_vis_np is not None and len(clustering_vis_np) > 0:
                clustering_vis_np_filtered = clustering_vis_np[context_indices.cpu().numpy()]
            else:
                clustering_vis_np_filtered = clustering_vis_np

            # Update frame and camera indices to match context frames only
            camera_indices_filtered = [camera_indices[i] for i in context_indices.cpu().tolist()]
            frame_indices_filtered = [frame_indices[i] for i in context_indices.cpu().tolist()]

            # Remap frame_indices to be continuous [0, 1, 2, ...] for visualization
            unique_frame_indices = sorted(set(frame_indices_filtered))
            frame_idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_frame_indices)}
            frame_indices_remapped = [frame_idx_mapping[idx] for idx in frame_indices_filtered]
            num_frames_filtered = len(unique_frame_indices)

            print(f"[INFO] Filtered to {len(gt_rgb_filtered)} context frame views for visualization")

            return {
                'gt_rgb': gt_rgb_filtered,
                'gt_depth': gt_depth_vis_filtered,
                'pred_depth': rendered_depth_vis,
                'pred_velocity': pred_velocity_vis_filtered,
                'pred_rgb': pred_rgb_np,
                'gt_velocity': gt_velocity_vis_filtered,
                'gt_segmentation': gt_seg_vis_filtered,
                'pred_segmentation': pred_seg_vis_filtered,
                'dynamic_clustering': clustering_vis_np_filtered,
                'num_cameras': num_cameras,
                'num_frames': num_frames_filtered,
                'camera_indices': camera_indices_filtered,
                'frame_indices': frame_indices_remapped,  # Use remapped indices
                'original_frame_indices': frame_indices_filtered,  # Keep original for reference
                'context_indices': context_indices.cpu().tolist(),
                'target_indices': [],  # No target frames in traditional mode
                'is_context_frame': None,  # Not needed for visualization
                'success': True
            }
        else:
            # Return all frames (render_target_frames=True or no distinction)
            return {
                'gt_rgb': gt_rgb,
                'gt_depth': gt_depth_vis,
                'pred_depth': rendered_depth_vis,
                'pred_velocity': pred_velocity_vis,
                'pred_rgb': pred_rgb_np,
                'gt_velocity': gt_velocity_vis,
                'gt_segmentation': gt_seg_vis,
                'pred_segmentation': pred_seg_vis,
                'dynamic_clustering': clustering_vis_np,
                'num_cameras': num_cameras,
                'num_frames': num_total_frames,
                'camera_indices': camera_indices,
                'frame_indices': frame_indices,
                'context_indices': context_indices.cpu().tolist() if context_indices.numel() > 0 else [],
                'target_indices': target_indices.cpu().tolist() if target_indices.numel() > 0 else [],
                'is_context_frame': is_context_frame,
                'success': True
            }

    except Exception as e:
        print(f"Error processing idx {idx}: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


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

    render_target_frames = getattr(cfg, 'render_target_frames', False)
    start_frame = getattr(cfg, 'start_frame', None)

    for idx in tqdm(indices, desc="Batch processing"):
        result = run_single_inference(
            model, dataset, idx,
            cfg.num_context_frames if hasattr(cfg, 'num_context_frames') else cfg.num_views,
            device, cfg,
            render_target_frames=render_target_frames,
            start_frame=start_frame
        )

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
                dynamic_clustering=result.get('dynamic_clustering'),
                context_indices=result.get('context_indices'),
                target_indices=result.get('target_indices'),
                visualize_target_frames=render_target_frames
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
            render_target_frames = getattr(cfg, 'render_target_frames', False)
            start_frame = getattr(cfg, 'start_frame', None)

            result = run_single_inference(
                model, dataset, cfg.single_idx,
                cfg.num_context_frames if hasattr(cfg, 'num_context_frames') else cfg.num_views,
                device, cfg,
                render_target_frames=render_target_frames,
                start_frame=start_frame
            )

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
                    dynamic_clustering=result.get('dynamic_clustering'),
                    context_indices=result.get('context_indices'),
                    target_indices=result.get('target_indices'),
                    visualize_target_frames=render_target_frames
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
