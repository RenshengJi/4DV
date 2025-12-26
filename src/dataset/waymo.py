"""
Simplified Waymo dataset for our training pipeline.
Only keeps essential functionality needed for training.
"""
import os
import os.path as osp
import numpy as np
import torch
import json
from dataset.base_dataset import BaseDataset
from dataset.utils import depthmap_to_absolute_camera_coordinates
from src.utils import imread_cv2


class WaymoDataset(BaseDataset):
    """
    Waymo outdoor street scenes dataset.
    Loads multi-view images with camera parameters, depth, flow, and segmentation.

    Frame sampling modes:
    - Context frames: Sparse frames used for network inference (e.g., 0, 5, 10, 15 with interval=5)
    - Target frames: Dense frames between context frames (e.g., 1-4, 6-9, 11-14, 16-19)

    Total frames loaded: num_context_frames * interval
    """

    def __init__(
        self,
        ROOT,
        valid_camera_id_list=["1", "2", "3"],
        intervals=[1],
        zero_ground_velocity=True,
        multi_camera_mode=False,
        **kwargs
    ):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.valid_camera_id_list = valid_camera_id_list
        self.zero_ground_velocity = zero_ground_velocity
        self.multi_camera_mode = multi_camera_mode

        if not isinstance(intervals, list):
            intervals = [intervals]
        self._intervals = intervals

        super().__init__(**kwargs)
        self._load_data()

    def _get_cache_filename(self):
        """
        Generate cache filename based on valid camera IDs to support different configurations.
        """
        camera_ids_str = "_".join(sorted(self.valid_camera_id_list))
        return f"waymo_scene_cache_{camera_ids_str}.json"

    def _load_data(self):
        """
        Load dataset metadata with caching support.
        First checks for a cached metadata file in ROOT directory.
        If not found, performs full scan and saves cache for future use.
        """
        cache_path = osp.join(self.ROOT, self._get_cache_filename())

        # Try to load from cache first
        if osp.exists(cache_path):
            try:
                print(f"Loading scene metadata from cache: {cache_path}")
                with open(cache_path, 'r') as f:
                    self.scene_data = json.load(f)
                self.scene_names = sorted(list(self.scene_data.keys()))
                print(f"Loaded {len(self.scene_names)} scenes with {len(self.valid_camera_id_list)} camera sequences from cache")
                return
            except Exception as e:
                print(f"Warning: Failed to load cache file ({e}), performing full scan...")

        # Perform full scan if cache doesn't exist or loading failed
        print("Performing full dataset scan (this may take a while but only need to do once)...")
        scene_dirs = sorted([
            d for d in os.listdir(self.ROOT)
            if os.path.isdir(os.path.join(self.ROOT, d))
        ])

        self.scene_data = {}

        for scene_name in scene_dirs:
            scene_dir = osp.join(self.ROOT, scene_name)
            seq2frames = {}

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

            for seq_id in seq2frames:
                seq2frames[seq_id] = sorted(seq2frames[seq_id])

            if seq2frames:
                self.scene_data[scene_name] = seq2frames

        self.scene_names = sorted(list(self.scene_data.keys()))
        print(f"Loaded {len(self.scene_names)} scenes with {len(self.valid_camera_id_list)} camera sequences")

        # Save cache for future use
        try:
            print(f"Saving scene metadata cache to: {cache_path}")
            with open(cache_path, 'w') as f:
                json.dump(self.scene_data, f, indent=2)
            print("Cache saved successfully")
        except Exception as e:
            print(f"Warning: Failed to save cache file ({e}), cache will not be available for next run")

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
        interval,
        is_context_frame=True
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
            is_context_frame: Whether this is a context frame (vs target frame)

        Returns:
            view_dict: Dictionary containing all view data
        """
        impath = f"{frame_id}_{camera_id}"

        image = imread_cv2(osp.join(scene_dir, impath + ".jpg"))
        depthmap = imread_cv2(osp.join(scene_dir, impath + ".exr"))
        camera_params = np.load(osp.join(scene_dir, impath + ".npz"))

        if hasattr(image, 'convert'):
            image = np.array(image)

        flow_path_npz = osp.join(scene_dir, impath + "_flow.npz")
        flow_path_npy = osp.join(scene_dir, impath + ".npy")
        flowmap = None
        if osp.exists(flow_path_npz):
            flow_data = np.load(flow_path_npz)
            flowmap = flow_data['data']
        elif osp.exists(flow_path_npy):
            flowmap = np.load(flow_path_npy)

        seg_path = osp.join(scene_dir, impath + "_seg.png")
        seg_mask = None
        if osp.exists(seg_path):
            seg_mask = imread_cv2(seg_path)

        intrinsics = np.float32(camera_params["intrinsics"])
        camera_pose = np.float32(camera_params["cam2world"])

        if flowmap is not None or seg_mask is not None:
            image, depthmap, intrinsics, flowmap, seg_mask = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(scene_dir, impath),
                flowmap=flowmap, seg_mask=seg_mask
            )
        else:
            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(scene_dir, impath)
            )

        segment_label = None
        segment_mask = None
        if flowmap is not None and flowmap.shape[-1] >= 4:
            raw_labels = flowmap[..., 3]
            segment_mask = (raw_labels != 0).astype(np.float32)

            segment_label = np.zeros_like(raw_labels, dtype=np.int64)
            segment_label[raw_labels == 1] = 0
            segment_label[raw_labels == 2] = 1
            segment_label[raw_labels == 4] = 2
            segment_label[(raw_labels == 3) | (raw_labels == 5)] = 3

        sky_mask = None
        if seg_mask is not None:
            sky_color = np.array([70, 130, 180])
            sky_mask = np.all(seg_mask == sky_color, axis=-1).astype(np.float32)

        if self.zero_ground_velocity and seg_mask is not None and flowmap is not None:
            road_color = np.array([128, 64, 128])
            sidewalk_color = np.array([244, 35, 232])

            road_mask = np.all(seg_mask == road_color, axis=-1)
            sidewalk_mask = np.all(seg_mask == sidewalk_color, axis=-1)
            static_ground_mask = road_mask | sidewalk_mask

            if static_ground_mask.any():
                flowmap[..., :3][static_ground_mask] = 0.0
                if segment_label is not None:
                    segment_label[static_ground_mask & (segment_label != 0)] = 0

        if self.transform is not None:
            img_tensor = self.transform(image)
        else:
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        img_tensor = img_tensor * 0.5 + 0.5

        depthmap_tensor = torch.from_numpy(depthmap).float()
        intrinsics_tensor = torch.from_numpy(intrinsics).float()
        camera_pose_tensor = torch.from_numpy(camera_pose).float()

        from dataset.utils import depthmap_to_camera_coordinates
        pts3d_cam, valid_mask = depthmap_to_camera_coordinates(
            depthmap=depthmap,
            camera_intrinsics=intrinsics
        )
        pts3d_tensor = torch.from_numpy(pts3d_cam).float()
        valid_mask_tensor = torch.from_numpy(valid_mask & np.isfinite(pts3d_cam).all(axis=-1)).bool()

        if flowmap is not None:
            flowmap_tensor = torch.from_numpy(flowmap).float()
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

        view_dict = dict(
            idx=(idx, camera_idx, frame_idx),
            camera_idx=camera_idx,
            frame_idx=frame_idx,
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
            is_context_frame=is_context_frame,
        )

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

        Frame loading strategy:
        - num_views now represents num_context_frames
        - Total frames loaded: num_context_frames * interval
        - Context frames: Frames at positions [0, interval, 2*interval, ...]
        - Target frames: All other frames in between

        Example: num_context_frames=4, interval=5
        - Total frames: 20 (0-19)
        - Context frames: [0, 5, 10, 15]
        - Target frames: [1,2,3,4, 6,7,8,9, 11,12,13,14, 16,17,18,19]

        Single-camera mode: Randomly samples one camera and loads frames from it.
        Multi-camera mode: Loads frames from ALL cameras in valid_camera_id_list.

        Args:
            idx: Scene index (0 to len(scene_names)-1)
            rng: Random number generator

        Returns:
            List of view dictionaries in VGGT format
        """
        scene_name = self.scene_names[idx]
        scene_dir = osp.join(self.ROOT, scene_name)
        seq2frames = self.scene_data[scene_name]

        ar_idx = rng.integers(0, len(self._resolutions))
        resolution = self._resolutions[ar_idx]

        if self.num_views_range is not None:
            num_context_frames = rng.integers(self.num_views_range[0], self.num_views_range[1] + 1)
        else:
            num_context_frames = self.num_views

        views = []

        if self.multi_camera_mode:
            available_cameras = sorted(list(seq2frames.keys()))
            selected_cameras = available_cameras

            min_frames = min(len(seq2frames[cam_id]) for cam_id in selected_cameras)

            interval = rng.choice(self._intervals)

            # New frame calculation: total frames = num_context_frames * interval
            required_frames = num_context_frames * interval

            if min_frames < required_frames:
                max_possible_interval = min_frames // num_context_frames if num_context_frames > 0 else 1
                if max_possible_interval < 1:
                    raise ValueError(
                        f"Scene {scene_name} has cameras with only {min_frames} frames, "
                        f"need at least {required_frames} frames (num_context_frames={num_context_frames}, interval={interval})"
                    )
                interval = min(interval, max_possible_interval)
                required_frames = num_context_frames * interval

            max_start_pos = min_frames - required_frames
            start_pos = rng.integers(0, max_start_pos + 1)

            # Generate all frame positions (0 to required_frames-1)
            all_positions = [start_pos + i for i in range(required_frames)]
            # Context frame positions: every interval-th frame
            context_positions = [start_pos + i * interval for i in range(num_context_frames)]

            for camera_idx, camera_id in enumerate(selected_cameras):
                frame_ids = seq2frames[camera_id]

                # Load all frames (both context and target)
                for frame_idx_in_sequence, pos in enumerate(all_positions):
                    frame_id = frame_ids[pos]
                    is_context = pos in context_positions

                    view_dict = self._load_single_view(
                        scene_dir=scene_dir,
                        camera_id=camera_id,
                        frame_id=frame_id,
                        resolution=resolution,
                        rng=rng,
                        scene_name=scene_name,
                        idx=idx,
                        camera_idx=camera_idx,
                        frame_idx=frame_idx_in_sequence,
                        interval=interval,
                        is_context_frame=is_context
                    )
                    views.append(view_dict)

        else:
            available_cameras = sorted(list(seq2frames.keys()))
            camera_id = rng.choice(available_cameras)
            frame_ids = seq2frames[camera_id]

            interval = rng.choice(self._intervals)

            # New frame calculation: total frames = num_context_frames * interval
            required_frames = num_context_frames * interval

            if len(frame_ids) < required_frames:
                max_possible_interval = len(frame_ids) // num_context_frames if num_context_frames > 0 else 1
                if max_possible_interval < 1:
                    raise ValueError(
                        f"Scene {scene_name} camera {camera_id} has only {len(frame_ids)} frames, "
                        f"need at least {required_frames} frames (num_context_frames={num_context_frames}, interval={interval})"
                    )
                interval = min(interval, max_possible_interval)
                required_frames = num_context_frames * interval

            max_start_pos = len(frame_ids) - required_frames
            start_pos = rng.integers(0, max_start_pos + 1)

            # Generate all frame positions (0 to required_frames-1)
            all_positions = [start_pos + i for i in range(required_frames)]
            # Context frame positions: every interval-th frame
            context_positions = [start_pos + i * interval for i in range(num_context_frames)]

            for frame_idx_in_sequence, pos in enumerate(all_positions):
                frame_id = frame_ids[pos]
                is_context = pos in context_positions

                view_dict = self._load_single_view(
                    scene_dir=scene_dir,
                    camera_id=camera_id,
                    frame_id=frame_id,
                    resolution=resolution,
                    rng=rng,
                    scene_name=scene_name,
                    idx=idx,
                    camera_idx=0,
                    frame_idx=frame_idx_in_sequence,
                    interval=interval,
                    is_context_frame=is_context
                )
                views.append(view_dict)

        reference_view_idx = 0

        reference_cam_pose = views[reference_view_idx]['camera_pose']

        world_to_ref = torch.linalg.inv(reference_cam_pose)

        for view in views:
            pts3d_cam = view['pts3d']
            H, W, _ = pts3d_cam.shape

            cam_to_world = view['camera_pose']

            cam_to_ref = torch.matmul(world_to_ref, cam_to_world)

            pts3d_flat = pts3d_cam.reshape(-1, 3)
            pts3d_ref = torch.matmul(cam_to_ref[:3, :3], pts3d_flat.T).T + cam_to_ref[:3, 3]
            view['pts3d'] = pts3d_ref.reshape(H, W, 3)

            view['camera_pose'] = torch.linalg.inv(cam_to_ref)

        all_pts = []
        for view in views:
            pts = view['pts3d']
            mask = view['valid_mask'].bool()
            all_pts.append(pts[mask])

        all_pts = torch.cat(all_pts, dim=0)
        dist_avg = all_pts.norm(dim=-1).mean()
        depth_scale_factor = 1.0 / dist_avg

        for view in views:
            view['depthmap'] = view['depthmap'] * depth_scale_factor
            view['camera_pose'][:3, 3] = view['camera_pose'][:3, 3] * depth_scale_factor
            view['pts3d'] = view['pts3d'] * depth_scale_factor
            view['flowmap'][..., :3] = view['flowmap'][..., :3] * depth_scale_factor if 'flowmap' in view else None
            view['depth_scale_factor'] = depth_scale_factor

        # Add metadata for multi-camera scenes
        if self.multi_camera_mode:
            num_cameras = len(selected_cameras)
            num_total_frames = len(all_positions)
        else:
            num_cameras = 1
            num_total_frames = len(all_positions)

        for view in views:
            view['num_cameras'] = num_cameras
            view['num_total_frames'] = num_total_frames

        return views

    def get_views_with_start_frame(self, idx, start_frame=0, camera_id=None):
        """
        Load views for a given scene with specified start frame (for inference/testing).

        Args:
            idx: Scene index
            start_frame: Start frame position (default 0)
            camera_id: Specific camera ID to use (default None, uses first available)

        Returns:
            List of view dictionaries
        """
        rng = np.random.default_rng(seed=idx)

        scene_name = self.scene_names[idx]
        scene_dir = osp.join(self.ROOT, scene_name)
        seq2frames = self.scene_data[scene_name]

        resolution = self._resolutions[0]
        num_context_frames = self.num_views
        interval = self._intervals[0]

        views = []

        if self.multi_camera_mode:
            available_cameras = sorted(list(seq2frames.keys()))
            selected_cameras = available_cameras
            min_frames = min(len(seq2frames[cam_id]) for cam_id in selected_cameras)

            required_frames = num_context_frames * interval
            if start_frame + required_frames > min_frames:
                start_frame = max(0, min_frames - required_frames)

            all_positions = [start_frame + i for i in range(required_frames)]
            context_positions = [start_frame + i * interval for i in range(num_context_frames)]

            for camera_idx, cam_id in enumerate(selected_cameras):
                frame_ids = seq2frames[cam_id]
                for frame_idx_in_sequence, pos in enumerate(all_positions):
                    frame_id = frame_ids[pos]
                    is_context = pos in context_positions
                    view_dict = self._load_single_view(
                        scene_dir=scene_dir,
                        camera_id=cam_id,
                        frame_id=frame_id,
                        resolution=resolution,
                        rng=rng,
                        scene_name=scene_name,
                        idx=idx,
                        camera_idx=camera_idx,
                        frame_idx=frame_idx_in_sequence,
                        interval=interval,
                        is_context_frame=is_context
                    )
                    views.append(view_dict)

            num_cameras = len(selected_cameras)
            num_total_frames = len(all_positions)
        else:
            available_cameras = sorted(list(seq2frames.keys()))
            cam_id = camera_id if camera_id in available_cameras else available_cameras[0]
            frame_ids = seq2frames[cam_id]

            required_frames = num_context_frames * interval
            if start_frame + required_frames > len(frame_ids):
                start_frame = max(0, len(frame_ids) - required_frames)

            all_positions = [start_frame + i for i in range(required_frames)]
            context_positions = [start_frame + i * interval for i in range(num_context_frames)]

            for frame_idx_in_sequence, pos in enumerate(all_positions):
                frame_id = frame_ids[pos]
                is_context = pos in context_positions
                view_dict = self._load_single_view(
                    scene_dir=scene_dir,
                    camera_id=cam_id,
                    frame_id=frame_id,
                    resolution=resolution,
                    rng=rng,
                    scene_name=scene_name,
                    idx=idx,
                    camera_idx=0,
                    frame_idx=frame_idx_in_sequence,
                    interval=interval,
                    is_context_frame=is_context
                )
                views.append(view_dict)

            num_cameras = 1
            num_total_frames = len(all_positions)

        # Apply same transformations as _get_views
        reference_cam_pose = views[0]['camera_pose']
        world_to_ref = torch.linalg.inv(reference_cam_pose)

        for view in views:
            pts3d_cam = view['pts3d']
            H, W, _ = pts3d_cam.shape
            cam_to_world = view['camera_pose']
            cam_to_ref = torch.matmul(world_to_ref, cam_to_world)
            pts3d_flat = pts3d_cam.reshape(-1, 3)
            pts3d_ref = torch.matmul(cam_to_ref[:3, :3], pts3d_flat.T).T + cam_to_ref[:3, 3]
            view['pts3d'] = pts3d_ref.reshape(H, W, 3)
            view['camera_pose'] = torch.linalg.inv(cam_to_ref)

        all_pts = []
        for view in views:
            pts = view['pts3d']
            mask = view['valid_mask'].bool()
            all_pts.append(pts[mask])

        all_pts = torch.cat(all_pts, dim=0)
        dist_avg = all_pts.norm(dim=-1).mean()
        depth_scale_factor = 1.0 / dist_avg

        for view in views:
            view['depthmap'] = view['depthmap'] * depth_scale_factor
            view['camera_pose'][:3, 3] = view['camera_pose'][:3, 3] * depth_scale_factor
            view['pts3d'] = view['pts3d'] * depth_scale_factor
            if 'flowmap' in view and view['flowmap'] is not None:
                view['flowmap'][..., :3] = view['flowmap'][..., :3] * depth_scale_factor
            view['depth_scale_factor'] = depth_scale_factor
            view['num_cameras'] = num_cameras
            view['num_total_frames'] = num_total_frames

        return views
