"""Extract and manage dynamic object Gaussians."""

import torch
from typing import Dict, List, Optional, Tuple
from .types import DynamicObject, ViewMapping


def extract_object_gaussians(
    object_id: int,
    object_class: str,
    tracked_results: List[Dict],
    gaussian_params: torch.Tensor,
    view_mapping: ViewMapping,
    registration: Optional['VelocityRegistration'] = None,
    preds: Optional[Dict] = None
) -> DynamicObject:
    """
    Extract Gaussian parameters for a dynamic object.
    Unified for both cars and pedestrians, single and multi-camera.

    Args:
        object_id: Global object ID
        object_class: 'car' or 'pedestrian'
        tracked_results: Tracking results with global_ids
        gaussian_params: [B, V, H, W, D] Gaussian parameters from VGGT
        view_mapping: View/frame/camera mapping
        registration: Optional registration for car aggregation
        preds: Optional predictions dict for velocity data

    Returns:
        DynamicObject with extracted Gaussians
    """
    H, W = gaussian_params.shape[2:4]
    device = gaussian_params.device

    # Collect frames where object appears
    object_frames = []
    frame_data = {}  # {frame_idx: (cluster_idx, view_indices, pixel_indices)}

    for frame_idx, result in enumerate(tracked_results):
        global_ids = result.get('global_ids', [])
        if object_id not in global_ids:
            continue

        cluster_idx = global_ids.index(object_id)
        cluster_indices = result.get('cluster_indices', [])
        if cluster_idx >= len(cluster_indices):
            continue

        pixel_indices = cluster_indices[cluster_idx]
        if not pixel_indices:
            continue

        object_frames.append(frame_idx)
        view_indices = result.get('view_indices', [frame_idx])
        frame_data[frame_idx] = (cluster_idx, view_indices, pixel_indices)

    if not object_frames:
        return _create_empty_object(object_id, object_class, device)

    # Extract Gaussians and pixel indices for each frame
    frame_gaussians = {}
    frame_pixel_indices = {}

    for frame_idx in object_frames:
        cluster_idx, view_indices, pixel_indices = frame_data[frame_idx]

        # Convert pixel indices to view-specific format
        gaussians_list = []
        pixel_dict = {}

        for view_idx in view_indices:
            # Get pixels belonging to this view
            view_offset = view_indices.index(view_idx) * H * W if len(view_indices) > 1 else 0
            view_pixels = [p - view_offset for p in pixel_indices
                          if view_offset <= p < view_offset + H * W]

            if not view_pixels:
                continue

            # Extract Gaussians for this view
            view_gaussians = _extract_gaussians_for_view(
                view_idx, view_pixels, gaussian_params, H, W
            )

            if view_gaussians is not None:
                gaussians_list.append(view_gaussians)
                pixel_dict[view_idx] = view_pixels

        if gaussians_list:
            frame_gaussians[frame_idx] = torch.cat(gaussians_list, dim=0)
            frame_pixel_indices[frame_idx] = pixel_dict

    # Create frame existence mask
    max_frame = max(object_frames) if object_frames else 0
    frame_existence = torch.tensor(
        [f in object_frames for f in range(max_frame + 1)],
        dtype=torch.bool,
        device=device
    )

    # For cars: aggregate if registration available
    if object_class == 'car' and registration is not None and preds is not None:
        canonical_gaussians, frame_transforms, reference_frame = _aggregate_car_object(
            object_id, object_frames, frame_data, tracked_results,
            frame_gaussians, registration, preds, view_mapping
        )

        return DynamicObject(
            object_id=object_id,
            object_class=object_class,
            canonical_gaussians=canonical_gaussians,
            frame_gaussians=frame_gaussians,
            frame_pixel_indices=frame_pixel_indices,
            frame_transforms=frame_transforms,
            frame_existence=frame_existence,
            reference_frame=reference_frame
        )
    else:
        # For pedestrians or cars without registration: per-frame only
        return DynamicObject(
            object_id=object_id,
            object_class=object_class,
            frame_gaussians=frame_gaussians,
            frame_pixel_indices=frame_pixel_indices,
            frame_existence=frame_existence
        )


def _extract_gaussians_for_view(
    view_idx: int,
    pixel_indices: List[int],
    gaussian_params: torch.Tensor,
    H: int,
    W: int
) -> Optional[torch.Tensor]:
    """Extract Gaussians for specific pixels in a view."""
    try:
        pixel_tensor = torch.tensor(pixel_indices, dtype=torch.long, device=gaussian_params.device)
        v_coords = pixel_tensor // W
        u_coords = pixel_tensor % W

        # Filter valid coordinates
        valid = (v_coords >= 0) & (v_coords < H) & (u_coords >= 0) & (u_coords < W)
        v_valid = v_coords[valid]
        u_valid = u_coords[valid]

        if len(v_valid) == 0:
            return None

        # Extract Gaussians: [1, view_idx, v, u, D]
        view_gaussians = gaussian_params[0, view_idx, v_valid, u_valid]
        return view_gaussians

    except Exception:
        return None


def _aggregate_car_object(
    object_id: int,
    object_frames: List[int],
    frame_data: Dict,
    tracked_results: List[Dict],
    frame_gaussians: Dict[int, torch.Tensor],
    registration: 'VelocityRegistration',
    preds: Dict,
    view_mapping: ViewMapping
) -> Tuple[Optional[torch.Tensor], Optional[Dict[int, torch.Tensor]], Optional[int]]:
    """Aggregate car object across frames using registration."""
    if len(object_frames) == 1:
        # Single frame: no aggregation needed
        frame_idx = object_frames[0]
        return frame_gaussians[frame_idx], {}, frame_idx

    # Choose middle frame as reference
    reference_frame = object_frames[len(object_frames) // 2]

    # Prepare per-frame data for registration
    points_frames = {}
    velocity_frames = {}

    for frame_idx in object_frames:
        cluster_idx, _, _ = frame_data[frame_idx]
        result = tracked_results[frame_idx]

        # Get cluster points and velocities
        points = result['points']
        labels = result['labels']
        mask = labels == cluster_idx

        if isinstance(points, torch.Tensor):
            points_frames[frame_idx] = points[mask]
        else:
            points_frames[frame_idx] = torch.tensor(points[mask], device=registration.device)

        # Extract velocities from preds
        velocity_global = preds.get('velocity_global')
        if velocity_global is not None:
            # Get velocities for the views in this frame
            view_indices = result.get('view_indices', [frame_idx])
            velocities = []
            for view_idx in view_indices:
                vel = velocity_global[0, view_idx].reshape(-1, 3)
                velocities.append(vel)

            merged_velocity = torch.cat(velocities, dim=0)
            velocity_frames[frame_idx] = merged_velocity[mask]

    # Compute transforms to reference frame
    frame_transforms = {}
    transform_cache = {}

    for frame_idx in object_frames:
        if frame_idx == reference_frame:
            frame_transforms[frame_idx] = torch.eye(4, device=registration.device)
        else:
            transform = registration.compute_chain_transform(
                frame_idx, reference_frame, transform_cache,
                points_frames, velocity_frames
            )
            if transform is not None:
                frame_transforms[frame_idx] = transform

    # Aggregate Gaussians to reference frame
    aggregated_list = []
    for frame_idx in object_frames:
        if frame_idx not in frame_transforms:
            continue

        transform = frame_transforms[frame_idx]
        gaussians = frame_gaussians[frame_idx].clone()

        # Transform xyz coordinates (first 3 dims)
        xyz = gaussians[:, :3]
        ones = torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
        xyz_homo = torch.cat([xyz, ones], dim=1)
        xyz_transformed = torch.matmul(xyz_homo, transform.T)[:, :3]

        gaussians[:, :3] = xyz_transformed
        aggregated_list.append(gaussians)

    canonical_gaussians = torch.cat(aggregated_list, dim=0) if aggregated_list else None

    return canonical_gaussians, frame_transforms, reference_frame


def _create_empty_object(object_id: int, object_class: str, device) -> DynamicObject:
    """Create empty dynamic object."""
    return DynamicObject(
        object_id=object_id,
        object_class=object_class,
        frame_gaussians={},
        frame_pixel_indices={},
        frame_existence=torch.tensor([], dtype=torch.bool, device=device)
    )


def extract_static_gaussians(
    clustering_results: List['ClusteringResult'],
    gaussian_params: torch.Tensor,
    view_mapping: ViewMapping,
    sky_masks: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Extract static background Gaussians.
    Unified for single and multi-camera modes.

    Args:
        clustering_results: Clustering results
        gaussian_params: [B, V, H, W, D] Gaussian parameters
        view_mapping: View/frame/camera mapping
        sky_masks: Optional [B, V, H, W] sky masks

    Returns:
        [N, D] static Gaussian parameters
    """
    H, W = gaussian_params.shape[2:4]
    device = gaussian_params.device
    static_list = []

    for frame_idx, result in enumerate(clustering_results):
        view_indices = result.view_indices
        labels = result.labels  # Labels for merged point cloud

        if not view_indices:
            continue

        # Split labels back to individual views
        view_offset = 0
        for view_idx in view_indices:
            view_size = H * W
            view_labels = labels[view_offset:view_offset + view_size]
            view_offset += view_size

            # Static mask: label == -1
            static_mask = view_labels == -1

            # Filter sky if available
            if sky_masks is not None and view_idx < sky_masks.shape[1]:
                sky_mask = sky_masks[0, view_idx].reshape(-1).bool()
                static_mask = static_mask & (~sky_mask.to(static_mask.device))

            # Extract static Gaussians
            view_gaussians = gaussian_params[0, view_idx].reshape(H * W, -1)
            static_gaussians = view_gaussians[static_mask]
            static_list.append(static_gaussians)

    if static_list:
        return torch.cat(static_list, dim=0)
    else:
        return torch.empty(0, gaussian_params.shape[-1], device=device)
