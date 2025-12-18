"""Clustering and object detection utilities."""

import torch
import numpy as np
from typing import List, Optional, Tuple
from cuml.cluster import DBSCAN
from sklearn.cluster import DBSCAN as SklearnDBSCAN
import cupy as cp

from .types import ClusteringResult, ViewMapping


def cluster_dynamic_objects(
    xyz: torch.Tensor,
    velocity: torch.Tensor,
    gt_scale: float,
    view_mapping: ViewMapping,
    velocity_threshold: float = 0.01,
    eps: float = 0.02,
    min_samples: int = 10,
    area_threshold: int = 750
) -> List[ClusteringResult]:
    """
    Cluster dynamic objects unified for single and multi-camera modes.

    Args:
        xyz: [V, HW, 3] point cloud coordinates (non-metric scale)
        velocity: [V, HW, 3] velocity vectors (non-metric scale)
        gt_scale: GT scale factor for metric conversion
        view_mapping: View/frame/camera mapping info
        velocity_threshold: Velocity threshold in m/s (metric scale)
        eps: DBSCAN neighborhood radius in meters (metric scale)
        min_samples: DBSCAN minimum samples
        area_threshold: Minimum cluster size

    Returns:
        List of ClusteringResult, one per temporal frame
    """
    device = xyz.device
    results = []

    for frame_idx in range(view_mapping.num_frames):
        # Get all views for this temporal frame
        view_indices = view_mapping.get_views_for_frame(frame_idx)

        if not view_indices:
            results.append(_create_empty_clustering_result(device))
            continue

        # Merge all views for this frame
        merged_xyz = torch.cat([xyz[v] for v in view_indices], dim=0)
        merged_velocity = torch.cat([velocity[v] for v in view_indices], dim=0)

        # Cluster merged point cloud
        result = _cluster_single_frame(
            merged_xyz, merged_velocity, gt_scale,
            velocity_threshold, eps, min_samples, area_threshold
        )
        result.view_indices = view_indices
        results.append(result)

    return results


def _cluster_single_frame(
    points: torch.Tensor,
    velocity: torch.Tensor,
    gt_scale: float,
    velocity_threshold: float,
    eps: float,
    min_samples: int,
    area_threshold: int
) -> ClusteringResult:
    """Cluster points in a single frame."""
    device = points.device
    N = points.shape[0]

    # Convert velocity to metric scale
    velocity_metric = velocity / gt_scale
    velocity_magnitude = torch.norm(velocity_metric, dim=-1)

    # Filter dynamic points
    dynamic_mask = velocity_magnitude > velocity_threshold
    dynamic_points = points[dynamic_mask]
    dynamic_velocities = velocity[dynamic_mask]

    if len(dynamic_points) < min_samples:
        return _create_empty_clustering_result(device, points)

    # Convert to metric scale for clustering
    dynamic_points_metric = dynamic_points / gt_scale

    # Perform clustering (try GPU first)
    try:
        points_cp = cp.asarray(dynamic_points_metric.detach())
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels_cp = dbscan.fit_predict(points_cp)
        cluster_labels = torch.as_tensor(labels_cp, device='cuda')
    except Exception:
        # Fallback to CPU
        try:
            points_np = dynamic_points_metric.detach().cpu().numpy()
            dbscan = SklearnDBSCAN(eps=eps, min_samples=min_samples)
            labels_np = dbscan.fit_predict(points_np)
            cluster_labels = torch.from_numpy(labels_np).to(device)
        except Exception:
            cluster_labels = torch.full((len(dynamic_points),), -1, dtype=torch.long, device=device)

    # Map back to full point cloud
    full_labels = torch.full((N,), -1, device=device)
    full_labels[dynamic_mask] = cluster_labels.to(device).long()

    # Extract valid clusters
    unique_labels = set(cluster_labels.cpu().numpy().tolist())
    unique_labels.discard(-1)

    cluster_centers = []
    cluster_velocities = []
    cluster_sizes = []
    cluster_indices = []
    valid_label_mapping = {}

    for old_label in sorted(unique_labels):
        mask = cluster_labels == old_label
        cluster_pts = dynamic_points[mask].detach()
        cluster_vel = dynamic_velocities[mask].detach()

        if len(cluster_pts) < area_threshold:
            # Mark as static
            dynamic_indices = torch.where(dynamic_mask)[0]
            filtered_indices = dynamic_indices[torch.where(mask)[0]]
            full_labels[filtered_indices] = -1
            continue

        new_label = len(cluster_centers)
        valid_label_mapping[old_label] = new_label

        cluster_centers.append(cluster_pts.mean(dim=0))
        cluster_velocities.append(cluster_vel.mean(dim=0))
        cluster_sizes.append(len(cluster_pts))

        # Get pixel indices
        cluster_mask = full_labels == old_label
        pixel_indices = torch.where(cluster_mask)[0].cpu().numpy().tolist()
        cluster_indices.append(pixel_indices)

    # Remap labels to be continuous
    for old_label, new_label in valid_label_mapping.items():
        full_labels[full_labels == old_label] = new_label

    return ClusteringResult(
        points=points,
        labels=full_labels.cpu(),
        num_clusters=len(cluster_centers),
        cluster_centers=cluster_centers,
        cluster_velocities=cluster_velocities,
        cluster_sizes=cluster_sizes,
        cluster_indices=cluster_indices,
        view_indices=[]
    )


def _create_empty_clustering_result(device, points=None) -> ClusteringResult:
    """Create empty clustering result."""
    if points is None:
        points = torch.empty(0, 3, device=device)

    return ClusteringResult(
        points=points,
        labels=torch.full((len(points),), -1, dtype=torch.long),
        num_clusters=0,
        cluster_centers=[],
        cluster_velocities=[],
        cluster_sizes=[],
        cluster_indices=[],
        view_indices=[]
    )
