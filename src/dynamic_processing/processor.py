"""Main dynamic object processor."""

import torch
import time
from typing import Dict, Any, Optional

from .types import ViewMapping, ProcessingResult, DynamicObject
from .clustering import cluster_dynamic_objects
from .tracking import track_objects_across_frames
from .classification import classify_objects
from .extraction import extract_object_gaussians, extract_static_gaussians
from .registration import VelocityRegistration


class DynamicProcessor:
    """Unified dynamic object processor for single and multi-camera modes."""

    def __init__(
        self,
        device: torch.device,
        velocity_threshold: float = 0.1,
        clustering_eps: float = 0.02,
        clustering_min_samples: int = 10,
        min_object_size: int = 100,
        tracking_position_threshold: float = 2.0,
        registration_mode: str = "simple",
        use_registration: bool = True
    ):
        """
        Initialize processor.

        Args:
            device: Computation device
            velocity_threshold: Velocity threshold in m/s
            clustering_eps: DBSCAN eps in meters
            clustering_min_samples: DBSCAN min samples
            min_object_size: Minimum cluster size
            tracking_position_threshold: Position threshold for tracking
            registration_mode: 'simple' or 'procrustes'
            use_registration: Whether to use registration for cars
        """
        self.device = device
        self.velocity_threshold = velocity_threshold
        self.clustering_eps = clustering_eps
        self.clustering_min_samples = clustering_min_samples
        self.min_object_size = min_object_size
        self.tracking_position_threshold = tracking_position_threshold

        self.registration = None
        if use_registration:
            self.registration = VelocityRegistration(
                device=str(device),
                mode=registration_mode
            )

    def process(
        self,
        preds: Dict[str, Any],
        vggt_batch: Dict[str, Any]
    ) -> ProcessingResult:
        """
        Process dynamic objects from VGGT predictions.

        Args:
            preds: VGGT predictions with xyz_camera, velocity_global, gaussian_params, etc.
            vggt_batch: Input batch with images, camera_indices, frame_indices, etc.

        Returns:
            ProcessingResult with cars, pedestrians, and static background
        """
        start_time = time.time()

        try:
            batch_dims = preds.get('batch_dims', {})
            B = batch_dims.get('B', 1)
            V = batch_dims.get('S', 4)
            H = batch_dims.get('H', 224)
            W = batch_dims.get('W', 224)

            view_mapping = self._create_view_mapping(vggt_batch, V)

            xyz = preds['xyz_camera'][0]  # [V, HW, 3]
            velocity_global = preds['velocity_global'][0]  # [V, H, W, 3]
            velocity = velocity_global.reshape(V, H * W, 3)  # [V, HW, 3]
            gaussian_params = preds['gaussian_params']  # [B, V, H, W, D]
            gt_scale = vggt_batch.get('depth_scale_factor', 1.0)

            if isinstance(gt_scale, torch.Tensor):
                gt_scale = gt_scale.item() if gt_scale.numel() == 1 else float(gt_scale)

            clustering_results = cluster_dynamic_objects(
                xyz, velocity, gt_scale, view_mapping,
                self.velocity_threshold, self.clustering_eps,
                self.clustering_min_samples, self.min_object_size
            )

            tracked_results = track_objects_across_frames(
                clustering_results,
                self.tracking_position_threshold
            )

            segment_logits = preds.get('segment_logits')
            if segment_logits is not None:
                object_classes = classify_objects(tracked_results, segment_logits, H, W)
            else:
                all_ids = set()
                for result in tracked_results:
                    all_ids.update(result.get('global_ids', []))
                object_classes = {gid: 'car' for gid in all_ids}

            cars = []
            pedestrians = []

            for object_id, object_class in object_classes.items():
                obj = extract_object_gaussians(
                    object_id, object_class, tracked_results,
                    gaussian_params, view_mapping,
                    registration=self.registration if object_class == 'car' else None,
                    preds=preds
                )

                if object_class == 'car':
                    cars.append(obj)
                else:
                    pedestrians.append(obj)

            sky_masks = vggt_batch.get('sky_masks', None)
            static_gaussians = extract_static_gaussians(
                clustering_results, gaussian_params, view_mapping, sky_masks
            )

            processing_time = time.time() - start_time

            return ProcessingResult(
                cars=cars,
                pedestrians=pedestrians,
                static_gaussians=static_gaussians,
                clustering_results=clustering_results,
                tracked_results=tracked_results,
                processing_time=processing_time,
                num_objects=len(cars) + len(pedestrians)
            )

        except Exception as e:
            print(f"Dynamic processing failed: {e}")
            import traceback
            traceback.print_exc()

            return ProcessingResult(
                cars=[],
                pedestrians=[],
                static_gaussians=torch.empty(0, 11, device=self.device),
                clustering_results=[],
                tracked_results=[],
                processing_time=time.time() - start_time,
                num_objects=0
            )

    def _create_view_mapping(self, vggt_batch: Dict, num_views: int) -> ViewMapping:
        """Create ViewMapping from batch data."""
        camera_indices = vggt_batch.get('camera_indices')
        frame_indices = vggt_batch.get('frame_indices')

        if camera_indices is not None and frame_indices is not None:
            camera_indices = camera_indices[0]
            frame_indices = frame_indices[0]
            num_cameras = int(camera_indices.max().item()) + 1
            # Use actual number of unique frames instead of max index
            num_frames = len(torch.unique(frame_indices))

            return ViewMapping(
                num_views=num_views,
                num_frames=num_frames,
                num_cameras=num_cameras,
                camera_indices=camera_indices,
                frame_indices=frame_indices
            )
        else:
            return ViewMapping(
                num_views=num_views,
                num_frames=num_views,
                num_cameras=1
            )

    def to_legacy_format(self, result: ProcessingResult) -> Dict[str, Any]:
        """
        Convert ProcessingResult to legacy format for compatibility.

        This allows gradual migration of train.py and inference_multi.py.
        """
        def obj_to_legacy(obj: DynamicObject) -> Dict:
            legacy = {
                'object_id': obj.object_id,
                'frame_existence': obj.frame_existence,
                'frame_pixel_indices': obj.frame_pixel_indices or {},
                'is_people': (obj.object_class == 'pedestrian')
            }

            if obj.canonical_gaussians is not None:
                legacy['canonical_gaussians'] = obj.canonical_gaussians
                legacy['reference_frame'] = obj.reference_frame
                legacy['frame_transforms'] = obj.frame_transforms or {}

            if obj.frame_gaussians is not None:
                legacy['frame_gaussians'] = obj.frame_gaussians

            if obj.frame_velocities is not None:
                legacy['frame_velocities'] = obj.frame_velocities

            return legacy

        return {
            'dynamic_objects_cars': [obj_to_legacy(obj) for obj in result.cars],
            'dynamic_objects_people': [obj_to_legacy(obj) for obj in result.pedestrians],
            'static_gaussians': result.static_gaussians,
            'matched_clustering_results': result.tracked_results,
            'processing_time': result.processing_time,
            'num_objects': result.num_objects,
            'num_cars': result.num_cars,
            'num_people': result.num_pedestrians
        }
