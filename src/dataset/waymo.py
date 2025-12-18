"""
Simplified Waymo dataset for our training pipeline.
Only keeps essential functionality needed for training.
"""
import os
import os.path as osp
import numpy as np
import torch
from dataset.base_dataset import BaseDataset
from dataset.utils import depthmap_to_absolute_camera_coordinates
from dust3r.utils.image import imread_cv2  # Keep this for now until we create our own image reader


class WaymoDataset(BaseDataset):
    """
    Waymo outdoor street scenes dataset.
    Loads multi-view images with camera parameters, depth, flow, and segmentation.
    """

    def __init__(
        self,
        ROOT,
        valid_camera_id_list=["1", "2", "3"],
        intervals=[1],  # List of possible frame intervals
        zero_ground_velocity=True,
        multi_camera_mode=False,  # New: whether to use multi-camera mode
        **kwargs
    ):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.valid_camera_id_list = valid_camera_id_list
        self.zero_ground_velocity = zero_ground_velocity
        self.multi_camera_mode = multi_camera_mode

        # Set intervals for sampling
        if not isinstance(intervals, list):
            intervals = [intervals]
        self._intervals = intervals

        super().__init__(**kwargs)
        self._load_data()

    def _load_data(self):
        """
        Load dataset metadata.
        Now only loads scene-level information, camera sequences are sampled dynamically.
        """
        scene_dirs = sorted([
            d for d in os.listdir(self.ROOT)
            if os.path.isdir(os.path.join(self.ROOT, d))
        ])

        # Store scene info: {scene_name: {camera_id: [sorted frame_ids]}}
        self.scene_data = {}

        for scene_name in scene_dirs:
            scene_dir = osp.join(self.ROOT, scene_name)
            seq2frames = {}

            # Collect frames for each camera sequence
            for f in os.listdir(scene_dir):
                if not f.endswith(".jpg"):
                    continue
                basename = f[:-4]
                frame_id = basename.split("_")[0]
                seq_id = basename.split("_")[1]

                if seq_id not in self.valid_camera_id_list:
                    continue

                if seq_id not in seq2frames:
                    seq2frames[seq_id] = []
                seq2frames[seq_id].append(frame_id)

            # Sort frames for each camera
            for seq_id in seq2frames:
                seq2frames[seq_id] = sorted(seq2frames[seq_id])

            # Only keep scenes that have valid camera sequences
            if seq2frames:
                self.scene_data[scene_name] = seq2frames

        self.scene_names = sorted(list(self.scene_data.keys()))
        print(f"Loaded {len(self.scene_names)} scenes with {len(self.valid_camera_id_list)} camera sequences")

    def __len__(self):
        """Return number of scenes"""
        return len(self.scene_names)

    def get_stats(self):
        return f"{len(self)} scenes"

    def _load_single_view(
        self,
        scene_dir,
        camera_id,
        frame_id,
        resolution,
        rng,
        scene_name,
        idx,
        camera_idx,
        frame_idx,
        interval
    ):
        """
        Load a single view with all associated data.

        Args:
            scene_dir: Path to scene directory
            camera_id: Camera ID string (e.g., "1", "2", "3")
            frame_id: Frame ID string (e.g., "0000", "0001")
            resolution: Target resolution tuple (H, W)
            rng: Random number generator
            scene_name: Name of the scene
            idx: Scene index
            camera_idx: Camera index in multi-camera mode (0, 1, 2, ...)
            frame_idx: Frame index within camera sequence (0, 1, 2, ...)
            interval: Frame interval for velocity scaling

        Returns:
            view_dict: Dictionary containing all view data
        """
        impath = f"{frame_id}_{camera_id}"

        # Load image and depth
        image = imread_cv2(osp.join(scene_dir, impath + ".jpg"))
        depthmap = imread_cv2(osp.join(scene_dir, impath + ".exr"))
        camera_params = np.load(osp.join(scene_dir, impath + ".npz"))

        # Ensure image is numpy array (imread_cv2 might return PIL Image in some cases)
        if hasattr(image, 'convert'):  # PIL Image
            image = np.array(image)

        # Load flow data (support both .npz and legacy .npy format)
        flow_path_npz = osp.join(scene_dir, impath + "_flow.npz")
        flow_path_npy = osp.join(scene_dir, impath + ".npy")
        flowmap = None
        if osp.exists(flow_path_npz):
            flow_data = np.load(flow_path_npz)
            flowmap = flow_data['data']
        elif osp.exists(flow_path_npy):
            flowmap = np.load(flow_path_npy)

        # Load semantic segmentation mask
        seg_path = osp.join(scene_dir, impath + "_seg.png")
        seg_mask = None
        if osp.exists(seg_path):
            seg_mask = imread_cv2(seg_path)

        intrinsics = np.float32(camera_params["intrinsics"])
        camera_pose = np.float32(camera_params["cam2world"])

        # Crop and resize
        if flowmap is not None or seg_mask is not None:
            image, depthmap, intrinsics, flowmap, seg_mask = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(scene_dir, impath),
                flowmap=flowmap, seg_mask=seg_mask
            )
        else:
            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(scene_dir, impath)
            )

        # Extract segmentation labels from flowmap
        segment_label = None
        segment_mask = None
        if flowmap is not None and flowmap.shape[-1] >= 4:
            raw_labels = flowmap[..., 3]  # [H, W]
            segment_mask = (raw_labels != 0).astype(np.float32)

            # Map raw labels to 4 classes: [bg, vehicle, sign, pedestrian+cyclist]
            segment_label = np.zeros_like(raw_labels, dtype=np.int64)
            segment_label[raw_labels == 1] = 0  # background/unlabeled
            segment_label[raw_labels == 2] = 1  # vehicle
            segment_label[raw_labels == 4] = 2  # sign
            segment_label[(raw_labels == 3) | (raw_labels == 5)] = 3  # pedestrian + cyclist

        # Extract sky mask from semantic segmentation
        sky_mask = None
        if seg_mask is not None:
            sky_color = np.array([70, 130, 180])
            sky_mask = np.all(seg_mask == sky_color, axis=-1).astype(np.float32)  # [H, W]

        # Apply semantic segmentation to zero out velocity on road and sidewalk
        if self.zero_ground_velocity and seg_mask is not None and flowmap is not None:
            road_color = np.array([128, 64, 128])
            sidewalk_color = np.array([244, 35, 232])

            road_mask = np.all(seg_mask == road_color, axis=-1)
            sidewalk_mask = np.all(seg_mask == sidewalk_color, axis=-1)
            static_ground_mask = road_mask | sidewalk_mask

            # Zero out velocity on static ground regions
            if static_ground_mask.any():
                flowmap[..., :3][static_ground_mask] = 0.0
                if segment_label is not None:
                    segment_label[static_ground_mask & (segment_label != 0)] = 0

        # Apply transform (ImgNorm: converts to tensor [-1, 1])
        if self.transform is not None:
            img_tensor = self.transform(image)
        else:
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Convert img from [-1, 1] to [0, 1] range for VGGT format
        img_tensor = img_tensor * 0.5 + 0.5

        # Convert to tensors
        depthmap_tensor = torch.from_numpy(depthmap).float()
        intrinsics_tensor = torch.from_numpy(intrinsics).float()
        camera_pose_tensor = torch.from_numpy(camera_pose).float()

        # Generate 3D points in current camera coordinates (not world coordinates yet)
        from dataset.utils import depthmap_to_camera_coordinates
        pts3d_cam, valid_mask = depthmap_to_camera_coordinates(
            depthmap=depthmap,
            camera_intrinsics=intrinsics
        )
        pts3d_tensor = torch.from_numpy(pts3d_cam).float()
        valid_mask_tensor = torch.from_numpy(valid_mask & np.isfinite(pts3d_cam).all(axis=-1)).bool()

        # Convert flowmap and segmentation data
        if flowmap is not None:
            flowmap_tensor = torch.from_numpy(flowmap).float()
            # Scale velocity component by (interval * 0.1) to convert to m/frame to m/0.1s
            flowmap_tensor[..., :3] *= (interval * 0.1)
        else:
            flowmap_tensor = None

        if segment_label is not None:
            segment_label_tensor = torch.from_numpy(segment_label).long()
        else:
            segment_label_tensor = None

        if segment_mask is not None:
            segment_mask_tensor = torch.from_numpy(segment_mask).float()
        else:
            segment_mask_tensor = None

        if sky_mask is not None:
            sky_mask_tensor = torch.from_numpy(sky_mask).float()
        else:
            sky_mask_tensor = None

        # Construct view dictionary in VGGT format
        view_dict = dict(
            idx=(idx, camera_idx, frame_idx),  # Changed: use real camera_idx instead of hardcoded 0
            camera_idx=camera_idx,  # New field
            frame_idx=frame_idx,    # New field
            img=img_tensor,
            depthmap=depthmap_tensor,
            camera_pose=camera_pose_tensor,
            camera_intrinsics=intrinsics_tensor,
            pts3d=pts3d_tensor,
            valid_mask=valid_mask_tensor,
            dataset="Waymo",
            label=scene_name,
            is_metric=self.is_metric,
            is_video=True,
            instance=osp.join(scene_dir, impath + ".jpg"),
            quantile=np.array(0.98, dtype=np.float32),
            rng=int.from_bytes(rng.bytes(4), "big"),
        )

        # Add optional fields
        if flowmap_tensor is not None:
            view_dict["flowmap"] = flowmap_tensor
        if segment_label_tensor is not None:
            view_dict["segment_label"] = segment_label_tensor
        if segment_mask_tensor is not None:
            view_dict["segment_mask"] = segment_mask_tensor
        if sky_mask_tensor is not None:
            view_dict["sky_mask"] = sky_mask_tensor

        return view_dict

    def _get_views(self, idx, rng):
        """
        Load views for a given scene index.
        Supports both single-camera and multi-camera modes.

        Single-camera mode: Randomly samples one camera and loads num_views frames from it.
        Multi-camera mode: Loads num_views frames from ALL cameras in valid_camera_id_list.

        Args:
            idx: Scene index (0 to len(scene_names)-1)
            rng: Random number generator

        Returns:
            List of view dictionaries in VGGT format
        """
        # Get scene name
        scene_name = self.scene_names[idx]
        scene_dir = osp.join(self.ROOT, scene_name)
        seq2frames = self.scene_data[scene_name]

        # Randomly select resolution from available resolutions
        ar_idx = rng.integers(0, len(self._resolutions))
        resolution = self._resolutions[ar_idx]

        # Randomly select num_views
        if self.num_views_range is not None:
            num_views = rng.integers(self.num_views_range[0], self.num_views_range[1] + 1)
        else:
            num_views = self.num_views

        views = []

        # ========== Multi-camera or single-camera mode ==========
        if self.multi_camera_mode:
            # Multi-camera mode: use all available cameras
            available_cameras = sorted(list(seq2frames.keys()))
            selected_cameras = available_cameras

            # In multi-camera mode, all cameras should use the SAME frame IDs (synchronized frames)
            # First, determine the interval and frame sampling strategy based on all cameras

            # Find the minimum number of frames across all cameras
            min_frames = min(len(seq2frames[cam_id]) for cam_id in selected_cameras)

            # Randomly select an interval from available intervals
            interval = rng.choice(self._intervals)

            # Calculate how many frames we need with this interval
            required_frames = 1 + (num_views - 1) * interval

            if min_frames < required_frames:
                # Not enough frames with this interval, adjust to max possible interval
                max_possible_interval = (min_frames - 1) // (num_views - 1) if num_views > 1 else 1
                if max_possible_interval < 1:
                    raise ValueError(
                        f"Scene {scene_name} has cameras with only {min_frames} frames, "
                        f"need at least {num_views} frames"
                    )
                interval = min(interval, max_possible_interval)
                required_frames = 1 + (num_views - 1) * interval

            # Randomly select starting frame position (same for all cameras)
            max_start_pos = min_frames - required_frames
            start_pos = rng.integers(0, max_start_pos + 1)

            # Sample frame positions (same for all cameras)
            sampled_positions = [start_pos + i * interval for i in range(num_views)]

            # For each camera, load views using the SAME frame positions
            for camera_idx, camera_id in enumerate(selected_cameras):
                frame_ids = seq2frames[camera_id]

                # Use the pre-determined frame positions
                sampled_frame_ids = [frame_ids[pos] for pos in sampled_positions]

                # Load views for this camera
                for frame_idx_in_camera, frame_id in enumerate(sampled_frame_ids):
                    view_dict = self._load_single_view(
                        scene_dir=scene_dir,
                        camera_id=camera_id,
                        frame_id=frame_id,
                        resolution=resolution,
                        rng=rng,
                        scene_name=scene_name,
                        idx=idx,
                        camera_idx=camera_idx,
                        frame_idx=frame_idx_in_camera,
                        interval=interval
                    )
                    views.append(view_dict)

        else:
            # Single-camera mode: randomly select one camera
            available_cameras = sorted(list(seq2frames.keys()))
            camera_id = rng.choice(available_cameras)
            frame_ids = seq2frames[camera_id]

            # Randomly select an interval from available intervals
            interval = rng.choice(self._intervals)

            # Calculate how many frames we need with this interval
            required_frames = 1 + (num_views - 1) * interval

            if len(frame_ids) < required_frames:
                # Not enough frames with this interval, try smaller interval or raise error
                max_possible_interval = (len(frame_ids) - 1) // (num_views - 1) if num_views > 1 else 1
                if max_possible_interval < 1:
                    raise ValueError(
                        f"Scene {scene_name} camera {camera_id} has only {len(frame_ids)} frames, "
                        f"need at least {num_views} frames"
                    )
                interval = min(interval, max_possible_interval)
                required_frames = 1 + (num_views - 1) * interval

            # Randomly select starting frame position
            max_start_pos = len(frame_ids) - required_frames
            start_pos = rng.integers(0, max_start_pos + 1)

            # Sample frames with the selected interval
            sampled_positions = [start_pos + i * interval for i in range(num_views)]
            sampled_frame_ids = [frame_ids[pos] for pos in sampled_positions]

            # Load views for single camera (camera_idx=0 for backward compatibility)
            for v, frame_id in enumerate(sampled_frame_ids):
                view_dict = self._load_single_view(
                    scene_dir=scene_dir,
                    camera_id=camera_id,
                    frame_id=frame_id,
                    resolution=resolution,
                    rng=rng,
                    scene_name=scene_name,
                    idx=idx,
                    camera_idx=0,
                    frame_idx=v,
                    interval=interval
                )
                views.append(view_dict)

        # ============ Apply coordinate transformations ============

        # Select reference view: always use the first camera's first frame (view_idx = 0)
        # Camera layout: [cam0_f0...cam0_f(n-1), cam1_f0...cam1_f(n-1), ...]
        # Both single-camera and multi-camera modes use cam0_f0 as reference
        reference_view_idx = 0

        # Get reference camera pose (ref_to_world)
        reference_cam_pose = views[reference_view_idx]['camera_pose']  # [4, 4]

        # Compute world_to_ref transformation
        world_to_ref = torch.linalg.inv(reference_cam_pose)  # [4, 4]

        # Transform pts3d from each camera's coordinate system to reference camera coordinate system
        # For each view: pts3d_ref = world_to_ref @ cam_to_world @ pts3d_cam
        for view in views:
            pts3d_cam = view['pts3d']  # [H, W, 3] - in current camera coordinates
            H, W, _ = pts3d_cam.shape

            cam_to_world = view['camera_pose']  # [4, 4]

            # Compute transformation: current_cam -> world -> reference_cam
            cam_to_ref = torch.matmul(world_to_ref, cam_to_world)  # [4, 4]

            # Apply transformation to points
            pts3d_flat = pts3d_cam.reshape(-1, 3)  # [H*W, 3]
            pts3d_ref = torch.matmul(cam_to_ref[:3, :3], pts3d_flat.T).T + cam_to_ref[:3, 3]
            view['pts3d'] = pts3d_ref.reshape(H, W, 3)

            # Update camera_pose to be ref_to_cam (inverse of cam_to_ref)
            view['camera_pose'] = torch.linalg.inv(cam_to_ref)

        # ============ Depth scale normalization (make non-metric) ============

        # Collect all valid points and compute average distance
        all_pts = []
        for view in views:
            pts = view['pts3d']  # [H, W, 3]
            mask = view['valid_mask'].bool()
            all_pts.append(pts[mask])

        all_pts = torch.cat(all_pts, dim=0)  # [N, 3]
        dist_avg = all_pts.norm(dim=-1).mean()
        depth_scale_factor = 1.0 / dist_avg

        # Apply normalization to depths, poses, and points
        for view in views:
            view['depthmap'] = view['depthmap'] * depth_scale_factor
            view['camera_pose'][:3, 3] = view['camera_pose'][:3, 3] * depth_scale_factor
            view['pts3d'] = view['pts3d'] * depth_scale_factor
            view['flowmap'][..., :3] = view['flowmap'][..., :3] * depth_scale_factor if 'flowmap' in view else None
            view['depth_scale_factor'] = depth_scale_factor
            
        return views
