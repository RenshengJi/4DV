"""
Gaussian rendering utilities
"""

import torch
from gsplat import rasterization
from src.losses import prune_gaussians_by_voxel
from .transform_utils import (
    object_exists_in_frame,
    get_object_transform_to_frame,
    apply_transform_to_gaussians,
    interpolate_transforms
)


def prepare_target_frame_transforms(dynamic_objects_cars, num_total_frames, device):
    """
    Pre-compute interpolated transforms for target frames for all cars.
    Updates frame_transforms and frame_existence in-place.

    Args:
        dynamic_objects_cars: List of car objects
        num_total_frames: Total number of frames (including target frames)
        device: torch device
    """
    for car in dynamic_objects_cars:
        if 'frame_transforms' not in car:
            continue

        frame_transforms = car['frame_transforms']
        available_frames = sorted(frame_transforms.keys())

        if len(available_frames) == 0:
            continue

        # Get current frame_existence
        frame_existence = car.get('frame_existence')
        if frame_existence is None:
            # Create frame_existence based on available transforms
            max_frame = max(available_frames)
            frame_existence = torch.zeros(max_frame + 1, dtype=torch.bool, device=device)
            for f in available_frames:
                frame_existence[f] = True

        # Extend frame_existence to cover all frames
        if len(frame_existence) < num_total_frames:
            extended_existence = torch.zeros(num_total_frames, dtype=torch.bool, device=device)
            extended_existence[:len(frame_existence)] = frame_existence
            frame_existence = extended_existence
            car['frame_existence'] = frame_existence

        # For each target frame, compute interpolated transform
        for frame_idx in range(num_total_frames):
            # Skip if this is already a context frame
            if frame_idx in frame_transforms:
                continue

            # Find nearest context frames
            frames_before = [f for f in available_frames if f < frame_idx]
            frames_after = [f for f in available_frames if f > frame_idx]

            interpolated_transform = None

            if len(frames_before) > 0 and len(frames_after) > 0:
                # Interpolate between closest before and after frames
                frame_before = frames_before[-1]
                frame_after = frames_after[0]

                transform_before = frame_transforms[frame_before]
                transform_after = frame_transforms[frame_after]

                # Calculate interpolation factor
                alpha = (frame_idx - frame_before) / (frame_after - frame_before)

                # Interpolate: frame_transforms stores frame_to_canonical, so interpolate those directly
                interpolated_transform = interpolate_transforms(
                    transform_before, transform_after, alpha, device
                )

            elif len(frames_before) > 0:
                # Only frames before: use the closest one
                frame_before = frames_before[-1]
                interpolated_transform = frame_transforms[frame_before].clone()

            elif len(frames_after) > 0:
                # Only frames after: use the closest one
                frame_after = frames_after[0]
                interpolated_transform = frame_transforms[frame_after].clone()

            # Add interpolated transform to frame_transforms
            if interpolated_transform is not None:
                frame_transforms[frame_idx] = interpolated_transform
                # Mark this frame as existing
                frame_existence[frame_idx] = True

        # Update frame_existence
        car['frame_existence'] = frame_existence


def render_gaussians_with_sky(scene, intrinsics, extrinsics, sky_colors, sampled_frame_indices, H, W, device,
                              enable_voxel_pruning=True, voxel_size=0.002, depth_scale_factor=None,
                              temporal_frame_indices=None):
    """
    Render Gaussian scene with sky color alpha blending
    Frame-by-frame rendering for correct dynamic object transforms

    Args:
        scene: Scene dictionary containing static and dynamic gaussians
        intrinsics: Camera intrinsics [S, 3, 3]
        extrinsics: Camera extrinsics [S, 4, 4]
        sky_colors: Sky colors [S, 3, H, W] or None
        sampled_frame_indices: Frame indices corresponding to sky_colors
        H, W: Image height and width
        device: torch device
        enable_voxel_pruning: bool, whether to enable voxel pruning
        voxel_size: float, voxel size in metric scale (meters)
        depth_scale_factor: float, depth scaling factor
        temporal_frame_indices: list or None, mapping from loop index to temporal frame index
            Used for dynamic object transform lookup (same for all cameras at same time)

    Returns:
        rendered_images: [S, 3, H, W] rendered RGB images
        rendered_depths: [S, H, W] rendered depth maps
    """
    S = intrinsics.shape[0]
    rendered_images = []
    rendered_depths = []

    for view_idx in range(S):
        # Determine temporal frame index for dynamic object lookup
        if temporal_frame_indices is not None:
            temporal_frame_idx = temporal_frame_indices[view_idx]
        else:
            temporal_frame_idx = view_idx

        all_means = []
        all_scales = []
        all_colors = []
        all_rotations = []
        all_opacities = []

        # Add static gaussians
        if scene.get('static_gaussians') is not None:
            static_gaussians = scene['static_gaussians']
            if static_gaussians.shape[0] > 0:
                all_means.append(static_gaussians[:, :3])
                all_scales.append(static_gaussians[:, 3:6])
                all_colors.append(static_gaussians[:, 6:9].unsqueeze(-2))
                all_rotations.append(static_gaussians[:, 9:13])
                all_opacities.append(static_gaussians[:, 13])

        # Add dynamic car gaussians
        dynamic_objects_cars = scene.get('dynamic_objects_cars', [])
        for obj_data in dynamic_objects_cars:
            if not object_exists_in_frame(obj_data, temporal_frame_idx):
                continue

            canonical_gaussians = obj_data.get('canonical_gaussians')
            if canonical_gaussians is None or canonical_gaussians.shape[0] == 0:
                continue

            frame_transform = get_object_transform_to_frame(obj_data, temporal_frame_idx)
            if frame_transform is None:
                transformed_gaussians = canonical_gaussians
            else:
                transformed_gaussians = apply_transform_to_gaussians(
                    canonical_gaussians, frame_transform
                )

            if transformed_gaussians.shape[0] > 0:
                all_means.append(transformed_gaussians[:, :3])
                all_scales.append(transformed_gaussians[:, 3:6])
                all_colors.append(transformed_gaussians[:, 6:9].unsqueeze(-2))
                all_rotations.append(transformed_gaussians[:, 9:13])
                all_opacities.append(transformed_gaussians[:, 13])

        # Add dynamic people gaussians
        dynamic_objects_people = scene.get('dynamic_objects_people', [])
        for obj_data in dynamic_objects_people:
            frame_gaussians = obj_data.get('frame_gaussians', {})

            # If target frame, try to extrapolate from nearest context frame
            if temporal_frame_idx not in frame_gaussians:
                # Find nearest context frame with gaussians
                available_frames = sorted(frame_gaussians.keys())
                if len(available_frames) == 0:
                    continue

                # Find frames before and after target
                frames_before = [f for f in available_frames if f < temporal_frame_idx]
                frames_after = [f for f in available_frames if f > temporal_frame_idx]

                # Determine which frames to use for interpolation
                if len(frames_before) > 0 and len(frames_after) > 0:
                    # Target is between two context frames - use velocity from frame before
                    frame_from = frames_before[-1]  # Closest frame before target
                    frame_to = frames_after[0]      # Next context frame (velocity points to this)

                    # Get gaussians and velocity from frame_from
                    gaussians_from = frame_gaussians[frame_from]
                    if gaussians_from is None or gaussians_from.shape[0] == 0:
                        continue

                    frame_velocities = obj_data.get('frame_velocities', {})
                    if frame_from in frame_velocities:
                        velocity = frame_velocities[frame_from]

                        # Velocity points from frame_from to frame_to
                        # Compute interpolation factor: alpha = (target - from) / (to - from)
                        alpha = (temporal_frame_idx - frame_from) / (frame_to - frame_from)

                        # Interpolate position: new_pos = old_pos + velocity * alpha
                        extrapolated_gaussians = gaussians_from.clone()
                        extrapolated_gaussians[:, :3] = gaussians_from[:, :3] + velocity * alpha

                        current_frame_gaussians = extrapolated_gaussians
                    else:
                        # No velocity, use gaussians from nearest frame
                        current_frame_gaussians = gaussians_from

                elif len(frames_before) > 0:
                    # Target is after all context frames - use last frame
                    nearest_frame = frames_before[-1]
                    current_frame_gaussians = frame_gaussians[nearest_frame]
                    if current_frame_gaussians is None or current_frame_gaussians.shape[0] == 0:
                        continue

                elif len(frames_after) > 0:
                    # Target is before all context frames - use first frame
                    nearest_frame = frames_after[0]
                    current_frame_gaussians = frame_gaussians[nearest_frame]
                    if current_frame_gaussians is None or current_frame_gaussians.shape[0] == 0:
                        continue
                else:
                    continue
            else:
                current_frame_gaussians = frame_gaussians[temporal_frame_idx]

            if current_frame_gaussians is None or current_frame_gaussians.shape[0] == 0:
                continue

            all_means.append(current_frame_gaussians[:, :3])
            all_scales.append(current_frame_gaussians[:, 3:6])
            all_colors.append(current_frame_gaussians[:, 6:9].unsqueeze(-2))
            all_rotations.append(current_frame_gaussians[:, 9:13])
            all_opacities.append(current_frame_gaussians[:, 13])

        # Handle empty scene
        if len(all_means) == 0:
            rendered_images.append(torch.zeros(3, H, W, device=device))
            rendered_depths.append(torch.zeros(H, W, device=device))
            continue

        # Concatenate all gaussians
        means = torch.cat(all_means, dim=0)
        scales = torch.cat(all_scales, dim=0)
        colors = torch.cat(all_colors, dim=0)
        rotations = torch.cat(all_rotations, dim=0)
        opacities = torch.cat(all_opacities, dim=0)

        # Apply voxel pruning if enabled
        if enable_voxel_pruning and means.shape[0] > 0:
            means, scales, rotations, opacities, colors = prune_gaussians_by_voxel(
                means, scales, rotations, opacities, colors,
                voxel_size=voxel_size,
                depth_scale_factor=depth_scale_factor
            )

        # Handle NaN/Inf values
        means = torch.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
        scales = torch.nan_to_num(scales, nan=0.01, posinf=1.0, neginf=0.01)
        colors = torch.nan_to_num(colors, nan=0.5, posinf=1.0, neginf=0.0)
        rotations = torch.nan_to_num(rotations, nan=0.0, posinf=1.0, neginf=-1.0)
        opacities = torch.nan_to_num(opacities, nan=0.5, posinf=1.0, neginf=0.0)

        K = intrinsics[view_idx]
        w2c = extrinsics[view_idx]

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

            # Apply sky color blending
            if sky_colors is not None and sampled_frame_indices is not None:
                if not isinstance(sampled_frame_indices, torch.Tensor):
                    sampled_frame_indices = torch.tensor(sampled_frame_indices, device=device)

                # Use view_idx to lookup sky color
                matches = (sampled_frame_indices == view_idx)
                if matches.any():
                    sky_idx = matches.nonzero(as_tuple=True)[0].item()
                    frame_sky_color = sky_colors[sky_idx]
                    alpha_3ch = rendered_alpha.unsqueeze(0)
                    rendered_image = alpha_3ch * rendered_image + (1 - alpha_3ch) * frame_sky_color
                    rendered_image = torch.clamp(rendered_image, min=0, max=1)

            rendered_images.append(rendered_image)
            rendered_depths.append(rendered_depth)

        except Exception as e:
            print(f"Error rendering view {view_idx} (temporal frame {temporal_frame_idx}): {e}")
            rendered_images.append(torch.zeros(3, H, W, device=device))
            rendered_depths.append(torch.zeros(H, W, device=device))

    return torch.stack(rendered_images, dim=0), torch.stack(rendered_depths, dim=0)
