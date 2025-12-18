"""Type definitions for dynamic processing module."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch


@dataclass
class ViewIndex:
    """Unified view index for single and multi-camera modes."""

    view_idx: int  # Global view index (0, 1, 2, ...)
    frame_idx: int  # Temporal frame index (0, 1, 2, ...)
    camera_idx: Optional[int] = None  # Camera index if multi-camera (0, 1, 2, ...)

    @property
    def is_multi_camera(self) -> bool:
        return self.camera_idx is not None

    def __hash__(self):
        return hash((self.view_idx, self.frame_idx, self.camera_idx))

    def __eq__(self, other):
        return (self.view_idx == other.view_idx and
                self.frame_idx == other.frame_idx and
                self.camera_idx == other.camera_idx)


@dataclass
class ViewMapping:
    """Maps between view indices and frame/camera indices."""

    num_views: int
    num_frames: int
    num_cameras: int = 1
    camera_indices: Optional[torch.Tensor] = None  # [num_views]
    frame_indices: Optional[torch.Tensor] = None   # [num_views]

    @property
    def is_multi_camera(self) -> bool:
        return self.num_cameras > 1

    def get_view_index(self, view_idx: int) -> ViewIndex:
        """Convert view index to ViewIndex object."""
        if self.is_multi_camera and self.frame_indices is not None and self.camera_indices is not None:
            return ViewIndex(
                view_idx=view_idx,
                frame_idx=int(self.frame_indices[view_idx].item()),
                camera_idx=int(self.camera_indices[view_idx].item())
            )
        else:
            # Single camera: view_idx == frame_idx
            return ViewIndex(view_idx=view_idx, frame_idx=view_idx)

    def get_views_for_frame(self, frame_idx: int) -> List[int]:
        """Get all view indices for a given temporal frame."""
        if self.is_multi_camera and self.frame_indices is not None:
            mask = self.frame_indices == frame_idx
            return torch.where(mask)[0].tolist()
        else:
            # Single camera: one view per frame
            return [frame_idx] if frame_idx < self.num_views else []


@dataclass
class ClusteringResult:
    """Result of clustering for one temporal frame."""

    points: torch.Tensor  # [N, 3] point coordinates
    labels: torch.Tensor  # [N] cluster labels (-1 for static)
    num_clusters: int
    cluster_centers: List[torch.Tensor]
    cluster_velocities: List[torch.Tensor]
    cluster_sizes: List[int]
    cluster_indices: List[List[int]]  # Pixel indices for each cluster
    view_indices: List[int]  # Which views contribute to this frame


@dataclass
class DynamicObject:
    """Represents a tracked dynamic object."""

    object_id: int
    object_class: str  # 'car' or 'pedestrian'
    canonical_gaussians: Optional[torch.Tensor] = None  # [N, D] aggregated for cars
    frame_gaussians: Optional[Dict[int, torch.Tensor]] = None  # {frame_idx: [N, D]} for pedestrians
    frame_pixel_indices: Optional[Dict[int, Dict[int, List[int]]]] = None  # {frame_idx: {view_idx: [pixels]}}
    frame_transforms: Optional[Dict[int, torch.Tensor]] = None  # {frame_idx: [4,4]} for cars
    frame_existence: Optional[torch.Tensor] = None  # [num_frames] bool
    reference_frame: Optional[int] = None

    @property
    def is_aggregated(self) -> bool:
        """Whether object has aggregated canonical representation."""
        return self.canonical_gaussians is not None


@dataclass
class ProcessingResult:
    """Complete result of dynamic processing."""

    cars: List[DynamicObject]
    pedestrians: List[DynamicObject]
    static_gaussians: torch.Tensor
    clustering_results: List[ClusteringResult]
    tracked_results: List[Dict]  # Clustering results with global_ids (matched)
    processing_time: float
    num_objects: int

    @property
    def num_cars(self) -> int:
        return len(self.cars)

    @property
    def num_pedestrians(self) -> int:
        return len(self.pedestrians)
