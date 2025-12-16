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
        **kwargs
    ):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.valid_camera_id_list = valid_camera_id_list
        self.zero_ground_velocity = zero_ground_velocity

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

    def _get_views(self, idx, rng):
        """
        Load views for a given scene index.
        Randomly samples: resolution, num_views, camera_id, interval, and starting frame.
        Then processes and converts to VGGT format.

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

        # Randomly select a camera from available cameras in this scene
        available_cameras = list(seq2frames.keys())
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

        # Load and process views
        views = []
        first_cam_pose = None

        for v, frame_id in enumerate(sampled_frame_ids):
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

            # Store first camera pose for coordinate transformation
            if v == 0:
                first_cam_pose = camera_pose

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

            # Generate 3D points from depth
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(
                depthmap=depthmap,
                camera_intrinsics=intrinsics,
                camera_pose=camera_pose
            )
            pts3d_tensor = torch.from_numpy(pts3d).float()
            valid_mask_tensor = torch.from_numpy(valid_mask & np.isfinite(pts3d).all(axis=-1)).bool()

            # Convert flowmap and segmentation data
            if flowmap is not None:
                flowmap_tensor = torch.from_numpy(flowmap).float()
                # Scale velocity component by ( interval * 0.1 ) to convert to m/frame to m/0.1s
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
                idx=(idx, 0, v),
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

            views.append(view_dict)

        # ============ Apply coordinate transformations (cam to world → cam0 to cam) ============

        # Transform coordinate system to first camera frame
        first_cam_pose_tensor = views[0]['camera_pose']  # [4, 4]
        first_cam_pose_inv = torch.linalg.inv(first_cam_pose_tensor)

        # Transform world_points from world to cam0 coordinate system
        # Transform extrinsics from (cam to world) to (cam0 to cam)
        for view in views:
            pts3d = view['pts3d']  # [H, W, 3]
            H, W, _ = pts3d.shape
            pts3d_flat = pts3d.reshape(-1, 3)  # [H*W, 3]
            pts3d_cam0 = torch.matmul(first_cam_pose_inv[:3, :3], pts3d_flat.T).T + first_cam_pose_inv[:3, 3]
            view['pts3d'] = pts3d_cam0.reshape(H, W, 3)

            cam_pose = view['camera_pose']
            # Compute cam0 to cam: inv(cam_to_cam0) = inv(world_to_cam0 @ cam_to_world)
            cam_to_cam0 = torch.matmul(first_cam_pose_inv, cam_pose)
            view['camera_pose'] = torch.linalg.inv(cam_to_cam0)

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
            view['depth_scale_factor'] = depth_scale_factor

        return views
