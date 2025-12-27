#!/usr/bin/env python3
"""
Box-based inference script for VGGT model.
Uses GT 3D bounding boxes for dynamic/static separation.

Key differences from clustering-based approach (infer.py):
- No clustering: boxes provide object boundaries directly
- No tracking: boxes have track_id for cross-frame association
- Box pose transforms for car aggregation (instead of velocity registration)
"""

import os
import sys
import numpy as np
import torch
import imageio.v2 as iio
from typing import Dict, List
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models import VGGT
from src.utils import tf32_off

from src.visualization import (
    visualize_depth,
    visualize_velocity,
    visualize_segmentation,
    create_multi_camera_grid,
)

from src.rendering import (
    render_gaussians_with_sky,
    interpolate_transforms as interp_transform_pair
)

import cv2


def get_track_color(track_id: int) -> tuple:
    """Get consistent color for a track_id."""
    np.random.seed(track_id)
    color = tuple(int(c) for c in np.random.randint(50, 255, 3))
    return color


def project_box_to_2d(
    box: Dict,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    H: int,
    W: int
) -> np.ndarray:
    """
    Project 3D box corners to 2D image coordinates.

    Args:
        box: Box dict with 'corners' [8, 3]
        intrinsic: [3, 3] camera intrinsic matrix
        extrinsic: [4, 4] camera extrinsic (cam_to_world)
        H, W: Image dimensions

    Returns:
        corners_2d: [8, 2] projected corners or None if behind camera
    """
    corners_3d = np.array(box['corners'])  # [8, 3] in reference frame

    # Transform to camera frame: world_to_cam = inv(extrinsic)
    world_to_cam = np.linalg.inv(extrinsic)
    corners_cam = (world_to_cam[:3, :3] @ corners_3d.T).T + world_to_cam[:3, 3]

    # Check if any corner is behind camera
    if np.any(corners_cam[:, 2] <= 0):
        return None

    # Project to 2D
    corners_2d = (intrinsic @ corners_cam.T).T
    corners_2d = corners_2d[:, :2] / corners_2d[:, 2:3]

    # Check if corners are within image bounds (with margin)
    margin = 100
    if np.all(corners_2d[:, 0] < -margin) or np.all(corners_2d[:, 0] > W + margin):
        return None
    if np.all(corners_2d[:, 1] < -margin) or np.all(corners_2d[:, 1] > H + margin):
        return None

    return corners_2d


def draw_box_on_image(
    image: np.ndarray,
    corners_2d: np.ndarray,
    color: tuple,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw 3D box edges on image.

    Corner ordering (from get_box_corners_3d):
    0: (+l/2, +w/2, +h/2), 1: (+l/2, -w/2, +h/2), 2: (-l/2, -w/2, +h/2), 3: (-l/2, +w/2, +h/2)
    4: (+l/2, +w/2, -h/2), 5: (+l/2, -w/2, -h/2), 6: (-l/2, -w/2, -h/2), 7: (-l/2, +w/2, -h/2)
    """
    # Define edges: top face, bottom face, vertical edges
    edges = [
        # Top face (z = +h/2)
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Bottom face (z = -h/2)
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    corners = corners_2d.astype(np.int32)
    for i, j in edges:
        pt1 = tuple(corners[i])
        pt2 = tuple(corners[j])
        cv2.line(image, pt1, pt2, color, thickness)

    return image


def visualize_boxes_on_images(
    images: np.ndarray,
    boxes_by_view: Dict[int, List[Dict]],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    H: int,
    W: int
) -> np.ndarray:
    """
    Visualize 3D boxes projected onto images.

    Args:
        images: [N, H, W, 3] images
        boxes_by_view: {view_idx: [boxes]} boxes for each view
        intrinsics: [N, 3, 3] intrinsic matrices
        extrinsics: [N, 4, 4] extrinsic matrices (cam_to_world)
        H, W: Image dimensions

    Returns:
        images_with_boxes: [N, H, W, 3] images with boxes drawn
    """
    images_with_boxes = images.copy()

    for view_idx in range(len(images)):
        if view_idx not in boxes_by_view:
            continue

        intrinsic = intrinsics[view_idx]
        extrinsic = extrinsics[view_idx]

        for box in boxes_by_view[view_idx]:
            track_id = box['track_id']
            color = get_track_color(track_id)

            corners_2d = project_box_to_2d(box, intrinsic, extrinsic, H, W)
            if corners_2d is not None:
                images_with_boxes[view_idx] = draw_box_on_image(
                    images_with_boxes[view_idx], corners_2d, color, thickness=2
                )

    return images_with_boxes


def fill_missing_transforms(
    frame_transforms: Dict[int, torch.Tensor],
    all_frames: List[int],
    device: torch.device
) -> Dict[int, torch.Tensor]:
    """Fill missing frame transforms using interpolation."""
    if not frame_transforms:
        return {f: torch.eye(4, device=device) for f in all_frames}

    available = sorted(frame_transforms.keys())
    result = dict(frame_transforms)

    for frame_idx in all_frames:
        if frame_idx in result:
            continue

        before = [f for f in available if f < frame_idx]
        after = [f for f in available if f > frame_idx]

        if before and after:
            f1, f2 = before[-1], after[0]
            alpha = (frame_idx - f1) / (f2 - f1)
            result[frame_idx] = interp_transform_pair(
                frame_transforms[f1], frame_transforms[f2], alpha, device
            )
        elif before:
            result[frame_idx] = frame_transforms[before[-1]].clone()
        elif after:
            result[frame_idx] = frame_transforms[after[0]].clone()
        else:
            result[frame_idx] = torch.eye(4, device=device)

    return result


def transform_boxes_to_reference_frame(
    boxes_by_frame: Dict,
    world_to_ref: torch.Tensor,
    depth_scale_factor: float,
    device: torch.device
) -> Dict:
    """Transform boxes from world frame to reference camera frame."""
    transformed = {}

    R = world_to_ref[:3, :3].cpu().numpy()
    t = world_to_ref[:3, 3].cpu().numpy()

    for frame_idx, boxes in boxes_by_frame.items():
        transformed[frame_idx] = []
        for box in boxes:
            new_box = box.copy()

            center = np.array(box['center'])
            new_center = R @ center + t
            new_center *= depth_scale_factor
            new_box['center'] = new_center.tolist()

            corners = np.array(box['corners'])
            new_corners = (R @ corners.T).T + t
            new_corners *= depth_scale_factor
            new_box['corners'] = new_corners

            yaw_offset = np.arctan2(R[1, 0], R[0, 0])
            new_box['heading'] = box['heading'] + yaw_offset

            new_box['size'] = [s * depth_scale_factor for s in box['size']]

            transformed[frame_idx].append(new_box)

    return transformed


def points_in_box(points: torch.Tensor, box: Dict, device: torch.device, margin: float = 0.0) -> torch.Tensor:
    """Check if points are inside a 3D bounding box."""
    original_shape = points.shape[:-1]
    points_flat = points.reshape(-1, 3)

    corners = torch.tensor(box['corners'], dtype=torch.float32, device=device)

    center = corners.mean(dim=0)

    axis_x = (corners[0] + corners[1] + corners[4] + corners[5]) / 4 - \
             (corners[2] + corners[3] + corners[6] + corners[7]) / 4
    axis_y = (corners[0] + corners[3] + corners[4] + corners[7]) / 4 - \
             (corners[1] + corners[2] + corners[5] + corners[6]) / 4
    axis_z = (corners[0] + corners[1] + corners[2] + corners[3]) / 4 - \
             (corners[4] + corners[5] + corners[6] + corners[7]) / 4

    half_x = axis_x.norm() / 2
    half_y = axis_y.norm() / 2
    half_z = axis_z.norm() / 2

    axis_x = axis_x / (axis_x.norm() + 1e-8)
    axis_y = axis_y / (axis_y.norm() + 1e-8)
    axis_z = axis_z / (axis_z.norm() + 1e-8)

    relative = points_flat - center
    local_x = (relative * axis_x).sum(dim=-1)
    local_y = (relative * axis_y).sum(dim=-1)
    local_z = (relative * axis_z).sum(dim=-1)

    in_box = (local_x.abs() <= half_x + margin) & \
             (local_y.abs() <= half_y + margin) & \
             (local_z.abs() <= half_z + margin)

    return in_box.reshape(original_shape)


def compute_box_transform(
    box_src: Dict,
    box_dst: Dict,
    device: torch.device
) -> torch.Tensor:
    """Compute rigid transform from box_src pose to box_dst pose."""
    src_center = torch.tensor(box_src['center'], dtype=torch.float32, device=device)
    dst_center = torch.tensor(box_dst['center'], dtype=torch.float32, device=device)
    src_heading = box_src['heading']
    dst_heading = box_dst['heading']

    delta_heading = dst_heading - src_heading
    cos_h, sin_h = np.cos(delta_heading), np.sin(delta_heading)

    R = torch.tensor([
        [cos_h, -sin_h, 0],
        [sin_h, cos_h, 0],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)

    t = dst_center - R @ src_center

    T = torch.eye(4, dtype=torch.float32, device=device)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def process_with_boxes(
    preds: Dict,
    batch: Dict,
    boxes_by_frame: Dict,
    device: torch.device,
    box_margin: float = 0.0
) -> Dict:
    """
    Process gaussians using GT box information for dynamic/static separation.

    - Uses GT boxes instead of clustering for dynamic/static separation
    - Cars: multi-frame aggregation using box pose transforms
    - People: per-frame gaussians with velocity from predictions
    """
    gaussian_params = preds['gaussian_params']
    gaussians = gaussian_params[0].clone()
    S, H, W, C = gaussians.shape

    frame_indices = batch['frame_indices'][0].cpu().numpy()
    pred_points = gaussians[:, :, :, :3]
    velocity_global = preds.get('velocity_global')

    all_tracks = {}
    for frame_idx, boxes in boxes_by_frame.items():
        for box in boxes:
            track_id = box['track_id']
            if track_id not in all_tracks:
                all_tracks[track_id] = {
                    'class': box['class'],
                    'boxes': {},
                    'frame_data': {},
                }
            all_tracks[track_id]['boxes'][frame_idx] = box

    static_mask = torch.ones(S, H, W, dtype=torch.bool, device=device)

    for view_idx in range(S):
        frame_idx = int(frame_indices[view_idx])
        if frame_idx not in boxes_by_frame:
            continue

        points = pred_points[view_idx]

        for box in boxes_by_frame[frame_idx]:
            in_box = points_in_box(points, box, device, margin=box_margin)
            num_in = in_box.sum().item()

            if num_in == 0:
                continue

            static_mask[view_idx] = static_mask[view_idx] & (~in_box)

            track_id = box['track_id']
            pixel_indices = torch.where(in_box.reshape(-1))[0].cpu().tolist()

            if frame_idx not in all_tracks[track_id]['frame_data']:
                all_tracks[track_id]['frame_data'][frame_idx] = {
                    'view_pixel_indices': {}  # view_idx -> pixel_indices
                }
            all_tracks[track_id]['frame_data'][frame_idx]['view_pixel_indices'][view_idx] = pixel_indices

    static_gaussians = gaussians.reshape(S * H * W, C)[static_mask.reshape(-1)]

    unique_frames = sorted(set(int(f) for f in frame_indices))
    cars = []
    people = []

    for track_id, track in all_tracks.items():
        frame_data = track['frame_data']
        if not frame_data:
            continue

        boxes = track['boxes']
        object_frames = sorted(frame_data.keys())

        frame_gaussians = {}
        frame_pixel_indices = {}

        for frame_idx in object_frames:
            view_pixel_indices = frame_data[frame_idx]['view_pixel_indices']

            if not view_pixel_indices:
                continue

            gaussians_list = []
            for view_idx, pixel_indices in view_pixel_indices.items():
                if not pixel_indices:
                    continue

                pixel_tensor = torch.tensor(pixel_indices, dtype=torch.long, device=device)
                v_coords = pixel_tensor // W
                u_coords = pixel_tensor % W

                valid = (v_coords >= 0) & (v_coords < H) & (u_coords >= 0) & (u_coords < W)
                v_valid = v_coords[valid]
                u_valid = u_coords[valid]

                if len(v_valid) == 0:
                    continue

                view_gaussians = gaussians[view_idx, v_valid, u_valid]
                gaussians_list.append(view_gaussians)

            if gaussians_list:
                frame_gaussians[frame_idx] = torch.cat(gaussians_list, dim=0)
                first_view = next(iter(view_pixel_indices))
                frame_pixel_indices[frame_idx] = {first_view: view_pixel_indices[first_view]}

        if not frame_gaussians:
            continue

        max_frame = max(object_frames) if object_frames else 0
        frame_existence = torch.tensor(
            [f in object_frames for f in range(max_frame + 1)],
            dtype=torch.bool,
            device=device
        )

        if track['class'] in ['vehicle', 'cyclist']:
            ref_frame = object_frames[len(object_frames) // 2]
            ref_box = boxes[ref_frame]

            frame_transforms = {}
            for frame_idx in object_frames:
                if frame_idx == ref_frame:
                    frame_transforms[frame_idx] = torch.eye(4, device=device)
                else:
                    frame_transforms[frame_idx] = compute_box_transform(
                        boxes[frame_idx], ref_box, device
                    )

            aggregated_list = []
            for frame_idx in object_frames:
                if frame_idx not in frame_transforms:
                    continue

                transform = frame_transforms[frame_idx]
                g = frame_gaussians[frame_idx].clone()

                xyz = g[:, :3]
                ones = torch.ones((xyz.shape[0], 1), device=device, dtype=xyz.dtype)
                xyz_homo = torch.cat([xyz, ones], dim=1)
                xyz_transformed = torch.matmul(xyz_homo, transform.T)[:, :3]
                g[:, :3] = xyz_transformed

                aggregated_list.append(g)

            canonical_gaussians = torch.cat(aggregated_list, dim=0) if aggregated_list else frame_gaussians[ref_frame]

            frame_transforms = fill_missing_transforms(frame_transforms, unique_frames, device)

            cars.append({
                'track_id': track_id,
                'class': track['class'],
                'canonical_gaussians': canonical_gaussians,
                'reference_frame': ref_frame,
                'frame_transforms': frame_transforms,
                'frame_existence': frame_existence,
            })
        else:
            frame_velocities = {}
            if velocity_global is not None:
                for frame_idx, pixel_dict in frame_pixel_indices.items():
                    vel_list = []
                    for view_idx, pixels in pixel_dict.items():
                        if not pixels:
                            continue
                        pixel_tensor = torch.tensor(pixels, dtype=torch.long, device=device)
                        v_coords = pixel_tensor // W
                        u_coords = pixel_tensor % W
                        valid = (v_coords >= 0) & (v_coords < H) & (u_coords >= 0) & (u_coords < W)
                        if valid.any():
                            view_vel = velocity_global[0, view_idx, v_coords[valid], u_coords[valid]]
                            vel_list.append(view_vel)
                    if vel_list:
                        all_vels = torch.cat(vel_list, dim=0)
                        frame_velocities[frame_idx] = all_vels.mean(dim=0)

            people.append({
                'track_id': track_id,
                'class': track['class'],
                'frame_gaussians': frame_gaussians,
                'frame_velocities': frame_velocities,
                'frame_existence': frame_existence,
            })

    return {
        'static_gaussians': static_gaussians,
        'dynamic_objects_cars': cars,
        'dynamic_objects_people': people,
    }


def load_model(model_path, device, cfg):
    """Load VGGT model."""
    print(f"Loading model from: {model_path}")
    model = eval(cfg.model)

    checkpoint = torch.load(model_path, map_location="cpu")
    ckpt = checkpoint.get('model', checkpoint)
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    return model


def load_dataset(dataset_cfg):
    """Load dataset with box loading enabled."""
    from dataset import WaymoDataset, Waymo_Multi, ImgNorm
    dataset = eval(dataset_cfg)
    print(f"Dataset loaded: {len(dataset)} scenes")
    return dataset


@torch.no_grad()
def run_inference_with_boxes(model, dataset, idx, device, cfg, start_frame=None):
    """Run inference using box-based dynamic processing."""
    try:
        if start_frame is not None and hasattr(dataset, 'get_views_with_start_frame'):
            views_list = dataset.get_views_with_start_frame(idx, start_frame=start_frame)
        else:
            views_list = dataset[idx]

        from dataset import vggt_collate_fn
        batch = vggt_collate_fn([views_list])

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        is_context_frame = batch['is_context_frame']
        camera_indices = batch['camera_indices'][0].cpu().tolist()
        frame_indices = batch['frame_indices'][0].cpu().tolist()
        num_cameras = batch['num_cameras'][0].item()
        num_total_frames = batch['num_total_frames'][0].item()

        context_mask = is_context_frame[0]
        context_indices = torch.where(context_mask)[0]

        images = batch['images']
        B, S, C, H, W = images.shape
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']

        scene_name = dataset.scene_names[idx]
        seq2frames = dataset.scene_data[scene_name]
        camera_id_list = sorted(seq2frames.keys())  # e.g., ["1", "2", "3"]
        first_camera = camera_id_list[0]
        frame_id_list = seq2frames[first_camera]
        start_pos = start_frame if start_frame is not None else 0

        depth_scale_factor = views_list[0].get('depth_scale_factor', 1.0)
        if isinstance(depth_scale_factor, torch.Tensor):
            depth_scale_factor = depth_scale_factor.item()

        original_ref_cam_to_world = views_list[0].get('original_ref_cam_to_world')
        if original_ref_cam_to_world is None:
            raise ValueError("original_ref_cam_to_world not found in views_list")
        world_to_ref = torch.linalg.inv(original_ref_cam_to_world).to(device)

        # Get boxes per (frame_idx, camera_id) with visibility filtering
        boxes_by_frame_world = {}
        unique_frame_indices = sorted(set(frame_indices))

        for frame_idx in unique_frame_indices:
            actual_frame_id = int(frame_id_list[start_pos + frame_idx])
            # Collect boxes visible from any camera used in this frame
            frame_boxes = []
            seen_track_ids = set()

            # Find which cameras are used for this frame
            cameras_for_frame = set()
            for i, (f_idx, c_idx) in enumerate(zip(frame_indices, camera_indices)):
                if f_idx == frame_idx:
                    cameras_for_frame.add(c_idx)

            # Get boxes visible from each camera
            for cam_idx in cameras_for_frame:
                camera_id = camera_id_list[cam_idx]
                boxes = dataset.get_boxes_for_frame(scene_name, actual_frame_id, camera_id=camera_id)
                if boxes:
                    for box in boxes:
                        if box['track_id'] not in seen_track_ids:
                            frame_boxes.append(box)
                            seen_track_ids.add(box['track_id'])

            if frame_boxes:
                boxes_by_frame_world[frame_idx] = frame_boxes

        boxes_by_frame = transform_boxes_to_reference_frame(
            boxes_by_frame_world, world_to_ref, depth_scale_factor, device
        )

        preds = model(
            images[:, context_indices],
            gt_extrinsics=extrinsics[:, context_indices],
            gt_intrinsics=intrinsics[:, context_indices],
            frame_sample_ratio=1.0
        )

        original_frame_indices = batch['frame_indices'][:, context_indices]
        unique_frames = torch.unique(original_frame_indices, sorted=True)
        frame_mapping = {int(f.item()): i for i, f in enumerate(unique_frames)}

        context_frame_indices = torch.tensor(
            [[frame_mapping[int(idx.item())] for idx in original_frame_indices[0]]],
            device=device
        )

        batch_context = {
            'frame_indices': context_frame_indices,
            'extrinsics': extrinsics[:, context_indices],
        }

        boxes_context = {}
        for old_idx, boxes in boxes_by_frame.items():
            if old_idx in frame_mapping:
                boxes_context[frame_mapping[old_idx]] = boxes

        box_margin = getattr(cfg, 'box_margin_meters', 0.2) * depth_scale_factor
        scene = process_with_boxes(preds, batch_context, boxes_context, device, box_margin=box_margin)

        sky_colors = preds.get('sky_colors', None)
        if sky_colors is not None:
            sky_colors = sky_colors[0]
        sampled_frame_indices = preds.get('sampled_frame_indices', None)

        frames_to_render = context_indices.cpu().tolist()
        render_intrinsics = intrinsics[0, frames_to_render]
        render_extrinsics = extrinsics[0, frames_to_render]
        render_temporal = [frame_mapping[frame_indices[i]] for i in frames_to_render]

        rendered_rgb, rendered_depth = render_gaussians_with_sky(
            scene, render_intrinsics, render_extrinsics,
            sky_colors, sampled_frame_indices, H, W, device,
            enable_voxel_pruning=cfg.enable_voxel_pruning,
            voxel_size=cfg.voxel_size,
            depth_scale_factor=depth_scale_factor,
            temporal_frame_indices=render_temporal
        )

        # Render dynamic-only scene
        scene_dynamic_only = {
            'static_gaussians': torch.empty(0, scene['static_gaussians'].shape[1], device=device),
            'dynamic_objects_cars': scene['dynamic_objects_cars'],
            'dynamic_objects_people': scene['dynamic_objects_people'],
        }
        rendered_rgb_dynamic, _ = render_gaussians_with_sky(
            scene_dynamic_only, render_intrinsics, render_extrinsics,
            sky_colors, sampled_frame_indices, H, W, device,
            enable_voxel_pruning=cfg.enable_voxel_pruning,
            voxel_size=cfg.voxel_size,
            depth_scale_factor=depth_scale_factor,
            temporal_frame_indices=render_temporal
        )

        # Render static-only scene
        scene_static_only = {
            'static_gaussians': scene['static_gaussians'],
            'dynamic_objects_cars': [],
            'dynamic_objects_people': [],
        }
        rendered_rgb_static, _ = render_gaussians_with_sky(
            scene_static_only, render_intrinsics, render_extrinsics,
            sky_colors, sampled_frame_indices, H, W, device,
            enable_voxel_pruning=cfg.enable_voxel_pruning,
            voxel_size=cfg.voxel_size,
            depth_scale_factor=depth_scale_factor,
            temporal_frame_indices=render_temporal
        )

        pred_rgb = (rendered_rgb.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
        pred_rgb_dynamic = (rendered_rgb_dynamic.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
        pred_rgb_static = (rendered_rgb_static.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
        pred_depth_vis = visualize_depth(rendered_depth.cpu().numpy())

        gt_rgb = (images[0, context_indices].cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
        gt_depth_vis = visualize_depth(batch['depths'][0, context_indices].cpu().numpy())

        # Visualize 3D boxes on GT images
        # Use original world-coordinate boxes (before transform to reference frame)
        # Each view uses boxes from its corresponding frame in world coordinates
        boxes_by_view_world = {}
        for ctx_view_idx, orig_view_idx in enumerate(context_indices.cpu().tolist()):
            frame_idx = frame_indices[orig_view_idx]
            if frame_idx in boxes_by_frame_world:
                boxes_by_view_world[ctx_view_idx] = boxes_by_frame_world[frame_idx]

        # Get intrinsics and extrinsics for context views
        # Note: extrinsics are cam_to_world, boxes are in world coordinates
        ctx_intrinsics = intrinsics[0, context_indices].cpu().numpy()
        ctx_extrinsics = extrinsics[0, context_indices].cpu().numpy()

        # Scale box corners by depth_scale_factor to match scaled extrinsics
        boxes_by_view_scaled = {}
        for view_idx, boxes in boxes_by_view_world.items():
            scaled_boxes = []
            for box in boxes:
                scaled_box = box.copy()
                corners = np.array(box['corners'])
                scaled_box['corners'] = corners * depth_scale_factor
                scaled_boxes.append(scaled_box)
            boxes_by_view_scaled[view_idx] = scaled_boxes

        gt_rgb_with_boxes = visualize_boxes_on_images(
            gt_rgb, boxes_by_view_scaled, ctx_intrinsics, ctx_extrinsics, H, W
        )

        gt_vel = batch.get('flowmap')
        if gt_vel is not None:
            gt_vel = gt_vel[0, context_indices, :, :, :3]
            gt_vel = gt_vel[:, :, :, [2, 0, 1]]
            gt_vel[:, :, :, 2] = -gt_vel[:, :, :, 2]
        else:
            gt_vel = torch.zeros(len(context_indices), H, W, 3, device=device)
        gt_vel_vis = (visualize_velocity(gt_vel, scale=0.1).cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

        pred_vel = preds.get('velocity', torch.zeros(1, len(context_indices), H, W, 3, device=device))[0]
        pred_vel = pred_vel[:, :, :, [2, 0, 1]]
        pred_vel[:, :, :, 2] = -pred_vel[:, :, :, 2]
        pred_vel_vis = (visualize_velocity(pred_vel, scale=0.1).cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

        gt_seg = batch.get('segment_label')
        gt_seg_mask = batch.get('segment_mask')
        if gt_seg is not None:
            gt_seg_vis = visualize_segmentation(
                gt_seg[0, context_indices],
                gt_seg_mask[0, context_indices] if gt_seg_mask is not None else None,
                num_classes=4
            )
        else:
            gt_seg_vis = np.zeros((len(context_indices), H, W, 3), dtype=np.uint8)

        pred_seg = preds.get('segment_logits')
        if pred_seg is not None:
            pred_seg_labels = torch.argmax(torch.softmax(pred_seg[0], dim=-1), dim=-1)
            pred_seg_vis = visualize_segmentation(pred_seg_labels, num_classes=4)
        else:
            pred_seg_vis = np.zeros((len(context_indices), H, W, 3), dtype=np.uint8)

        cam_idx_ctx = [camera_indices[i] for i in context_indices.cpu().tolist()]
        frame_idx_ctx = [frame_mapping[frame_indices[i]] for i in context_indices.cpu().tolist()]

        return {
            'gt_rgb': gt_rgb,
            'gt_rgb_with_boxes': gt_rgb_with_boxes,
            'gt_depth': gt_depth_vis,
            'pred_depth': pred_depth_vis,
            'pred_velocity': pred_vel_vis,
            'pred_rgb': pred_rgb,
            'pred_rgb_dynamic': pred_rgb_dynamic,
            'pred_rgb_static': pred_rgb_static,
            'gt_velocity': gt_vel_vis,
            'gt_segmentation': gt_seg_vis,
            'pred_segmentation': pred_seg_vis,
            'num_cameras': num_cameras,
            'num_frames': len(unique_frames),
            'camera_indices': cam_idx_ctx,
            'frame_indices': frame_idx_ctx,
            'success': True
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def save_video(frames, path, fps=10):
    """Save video."""
    with iio.get_writer(path, fps=fps, codec='libx264', quality=8) as w:
        for f in frames:
            w.append_data(f)
    print(f"Saved: {path}")


def run_batch_inference(model, dataset, cfg, device):
    """Run batch inference on multiple scenes."""
    print(f"\n{'='*60}")
    print(f"Batch Inference Mode")
    print(f"Range: {cfg.start_idx} to {cfg.end_idx}, step {cfg.step}")
    print(f"{'='*60}\n")

    successful = []
    failed = []

    indices = range(cfg.start_idx, cfg.end_idx, cfg.step)
    start_frame = getattr(cfg, 'start_frame', None)

    for idx in tqdm(indices, desc="Processing scenes"):
        print(f"\n{'='*60}")
        print(f"Scene {idx}")
        print(f"{'='*60}")

        result = run_inference_with_boxes(
            model, dataset, idx, device, cfg,
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
            )

            if cfg.save_video:
                save_video(grid_frames, os.path.join(cfg.output_dir, f"box_idx{idx}.mp4"), fps=cfg.fps)

                # Save dynamic-only video
                if 'pred_rgb_dynamic' in result:
                    dynamic_frames = create_multi_camera_grid(
                        result['gt_rgb'],
                        result['gt_depth'],
                        result['pred_depth'],
                        result['pred_velocity'],
                        result['num_cameras'],
                        result['num_frames'],
                        result['camera_indices'],
                        result['frame_indices'],
                        pred_rgb=result['pred_rgb_dynamic'],
                        gt_velocity=result.get('gt_velocity'),
                        gt_segmentation=result.get('gt_segmentation'),
                        pred_segmentation=result.get('pred_segmentation'),
                    )
                    save_video(dynamic_frames, os.path.join(cfg.output_dir, f"box_idx{idx}_dynamic.mp4"), fps=cfg.fps)

                # Save static-only video
                if 'pred_rgb_static' in result:
                    static_frames = create_multi_camera_grid(
                        result['gt_rgb'],
                        result['gt_depth'],
                        result['pred_depth'],
                        result['pred_velocity'],
                        result['num_cameras'],
                        result['num_frames'],
                        result['camera_indices'],
                        result['frame_indices'],
                        pred_rgb=result['pred_rgb_static'],
                        gt_velocity=result.get('gt_velocity'),
                        gt_segmentation=result.get('gt_segmentation'),
                        pred_segmentation=result.get('pred_segmentation'),
                    )
                    save_video(static_frames, os.path.join(cfg.output_dir, f"box_idx{idx}_static.mp4"), fps=cfg.fps)

                # Save video with 3D box projections
                if 'gt_rgb_with_boxes' in result:
                    box_frames = create_multi_camera_grid(
                        result['gt_rgb_with_boxes'],
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
                    )
                    save_video(box_frames, os.path.join(cfg.output_dir, f"box_idx{idx}_boxes.mp4"), fps=cfg.fps)

            successful.append(idx)
            print(f"Scene {idx}: Success")
        else:
            failed.append(idx)
            print(f"Scene {idx}: Failed - {result['error']}")

            if not cfg.continue_on_error:
                raise RuntimeError(f"Scene {idx} failed: {result['error']}")

    print(f"\n{'='*60}")
    print(f"Batch Inference Summary")
    print(f"{'='*60}")
    print(f"Successful: {len(successful)}/{len(indices)}")
    print(f"Failed: {len(failed)}/{len(indices)}")
    if failed:
        print(f"Failed indices: {failed}")
    print(f"{'='*60}\n")


@hydra.main(version_base=None, config_path="config/waymo", config_name="infer_box")
def main(cfg: OmegaConf):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    if cfg.get("debug", False):
        import debugpy
        debugpy.listen(5699)
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()

    model = load_model(cfg.model_path, device, cfg)
    dataset = load_dataset(cfg.infer_dataset)

    os.makedirs(cfg.output_dir, exist_ok=True)

    with tf32_off():
        if cfg.batch_mode:
            run_batch_inference(model, dataset, cfg, device)
        else:
            result = run_inference_with_boxes(
                model, dataset, cfg.single_idx, device, cfg,
                start_frame=getattr(cfg, 'start_frame', None)
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
                )

                if cfg.save_video:
                    save_video(grid_frames, os.path.join(cfg.output_dir, f"box_idx{cfg.single_idx}.mp4"), fps=cfg.fps)

                    # Save dynamic-only video
                    if 'pred_rgb_dynamic' in result:
                        dynamic_frames = create_multi_camera_grid(
                            result['gt_rgb'],
                            result['gt_depth'],
                            result['pred_depth'],
                            result['pred_velocity'],
                            result['num_cameras'],
                            result['num_frames'],
                            result['camera_indices'],
                            result['frame_indices'],
                            pred_rgb=result['pred_rgb_dynamic'],
                            gt_velocity=result.get('gt_velocity'),
                            gt_segmentation=result.get('gt_segmentation'),
                            pred_segmentation=result.get('pred_segmentation'),
                        )
                        save_video(dynamic_frames, os.path.join(cfg.output_dir, f"box_idx{cfg.single_idx}_dynamic.mp4"), fps=cfg.fps)

                    # Save static-only video
                    if 'pred_rgb_static' in result:
                        static_frames = create_multi_camera_grid(
                            result['gt_rgb'],
                            result['gt_depth'],
                            result['pred_depth'],
                            result['pred_velocity'],
                            result['num_cameras'],
                            result['num_frames'],
                            result['camera_indices'],
                            result['frame_indices'],
                            pred_rgb=result['pred_rgb_static'],
                            gt_velocity=result.get('gt_velocity'),
                            gt_segmentation=result.get('gt_segmentation'),
                            pred_segmentation=result.get('pred_segmentation'),
                        )
                        save_video(static_frames, os.path.join(cfg.output_dir, f"box_idx{cfg.single_idx}_static.mp4"), fps=cfg.fps)

                    # Save video with 3D box projections
                    if 'gt_rgb_with_boxes' in result:
                        box_frames = create_multi_camera_grid(
                            result['gt_rgb_with_boxes'],
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
                        )
                        save_video(box_frames, os.path.join(cfg.output_dir, f"box_idx{cfg.single_idx}_boxes.mp4"), fps=cfg.fps)

                print(f"\nSuccess!")
            else:
                print(f"Failed: {result['error']}")


if __name__ == "__main__":
    main()
