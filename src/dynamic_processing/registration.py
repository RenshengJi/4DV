"""Velocity-based registration and transformation estimation."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

from src.utils import tf32_off


class VelocityRegistration:
    """Velocity-based point cloud registration."""

    def __init__(
        self,
        device: str = "cuda",
        mode: str = "simple",
        min_inliers_ratio: float = 0.1
    ):
        """
        Initialize registration.

        Args:
            device: Computation device
            mode: 'simple' (translation only) or 'procrustes' (full rigid transform)
            min_inliers_ratio: Minimum inlier ratio for valid transform
        """
        self.device = device
        self.mode = mode
        self.min_inliers_ratio = min_inliers_ratio

    def estimate_transform(
        self,
        points_src: torch.Tensor,
        velocity_src: torch.Tensor,
        direction: int = 1
    ) -> torch.Tensor:
        """
        Estimate transformation from velocity field.

        Args:
            points_src: [N, 3] source points
            velocity_src: [N, 3] velocity vectors
            direction: 1 for forward, -1 for backward

        Returns:
            [4, 4] transformation matrix
        """
        if self.mode == "simple":
            return self._estimate_simple(velocity_src, direction)
        else:
            return self._estimate_procrustes(points_src, velocity_src, direction)

    def _estimate_simple(self, velocity: torch.Tensor, direction: int) -> torch.Tensor:
        """Simple translation estimation from mean velocity."""
        mean_velocity = velocity.mean(dim=0) * direction
        transform = torch.eye(4, device=self.device, dtype=torch.float32)
        transform[:3, 3] = mean_velocity
        return transform

    def _estimate_procrustes(
        self,
        points_src: torch.Tensor,
        velocity: torch.Tensor,
        direction: int
    ) -> torch.Tensor:
        """Procrustes/Kabsch algorithm for rigid transform estimation."""
        if len(points_src) < 3:
            return self._estimate_simple(velocity, direction)

        points_dst = points_src + velocity * direction

        try:
            R, t = self._kabsch_algorithm(
                points_src.detach(),
                points_dst.detach(),
                trim_ratio=0.95
            )
            transform = torch.eye(4, device=self.device, dtype=torch.float32)
            transform[:3, :3] = R
            transform[:3, 3] = t
            return transform
        except Exception:
            return self._estimate_simple(velocity, direction)

    def _kabsch_algorithm(
        self,
        pts_src: torch.Tensor,
        pts_dst: torch.Tensor,
        trim_ratio: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Kabsch algorithm with trimming for robustness."""
        with tf32_off():
            if len(pts_src) < 3:
                raise ValueError("Need at least 3 points")

            # Initial estimate
            R, t = self._kabsch_step(pts_src, pts_dst)

            # Trim outliers
            if trim_ratio < 1.0:
                pts_transformed = torch.matmul(pts_src, R.T) + t
                residuals = torch.norm(pts_transformed - pts_dst, dim=1)
                num_keep = max(3, int(len(pts_src) * trim_ratio))
                _, inlier_indices = torch.topk(residuals, k=num_keep, largest=False)

                pts_src = pts_src[inlier_indices]
                pts_dst = pts_dst[inlier_indices]

                # Refine estimate
                R, t = self._kabsch_step(pts_src, pts_dst)

            return R, t

    def _kabsch_step(
        self,
        pts_src: torch.Tensor,
        pts_dst: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single Kabsch step."""
        # Center points
        centroid_src = pts_src.mean(dim=0)
        centroid_dst = pts_dst.mean(dim=0)
        pts_src_centered = pts_src - centroid_src
        pts_dst_centered = pts_dst - centroid_dst

        # Check for degenerate case
        if torch.allclose(pts_src_centered, torch.zeros_like(pts_src_centered)):
            return torch.eye(3, device=self.device), centroid_dst - centroid_src

        # Covariance matrix
        H = torch.matmul(pts_src_centered.T, pts_dst_centered)

        if torch.allclose(H, torch.zeros_like(H)):
            return torch.eye(3, device=self.device), centroid_dst - centroid_src

        # SVD
        U, _, Vt = torch.linalg.svd(H)
        R = torch.matmul(Vt.T, U.T)

        # Handle reflection
        if torch.linalg.det(R) < 0:
            Vt_corrected = Vt.clone()
            Vt_corrected[-1, :] *= -1
            R = torch.matmul(Vt_corrected.T, U.T)

        t = centroid_dst - torch.matmul(R, centroid_src)

        return R, t

    def compute_chain_transform(
        self,
        start_frame: int,
        end_frame: int,
        frame_transforms: Dict[Tuple[int, int], torch.Tensor],
        points_frames: Dict[int, torch.Tensor],
        velocity_frames: Dict[int, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Compute chained transformation from start to end frame.

        Args:
            start_frame: Starting frame index
            end_frame: Target frame index
            frame_transforms: Cache of computed transforms
            points_frames: Points for each frame
            velocity_frames: Velocities for each frame

        Returns:
            [4, 4] transformation matrix or None
        """
        if start_frame == end_frame:
            return torch.eye(4, device=self.device, dtype=torch.float32)

        # Check cache
        cache_key = (start_frame, end_frame)
        if cache_key in frame_transforms:
            return frame_transforms[cache_key]

        # Determine direction
        direction = 1 if start_frame < end_frame else -1
        frame_range = range(start_frame, end_frame, direction)

        cumulative = torch.eye(4, device=self.device, dtype=torch.float32)

        for frame_idx in frame_range:
            next_frame = frame_idx + direction
            step_key = (frame_idx, next_frame)

            if step_key in frame_transforms:
                step_transform = frame_transforms[step_key]
            else:
                # Compute transform
                if frame_idx not in points_frames or frame_idx not in velocity_frames:
                    return None

                step_transform = self.estimate_transform(
                    points_frames[frame_idx],
                    velocity_frames[frame_idx],
                    direction
                )
                frame_transforms[step_key] = step_transform

            cumulative = torch.matmul(step_transform, cumulative)

        frame_transforms[cache_key] = cumulative
        return cumulative
