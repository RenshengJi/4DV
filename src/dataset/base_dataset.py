"""
Simplified base dataset class for our training pipeline.
Only keeps essential functionality needed for training.
"""
import PIL
import numpy as np
import torch
import random
import itertools
from dataset.utils import depthmap_to_absolute_camera_coordinates
from dataset import cropping
from dataset.easy_dataset import EasyDataset


class BaseDataset(EasyDataset, torch.utils.data.Dataset):
    """
    Simplified base dataset class.
    Only keeps essential features:
    - Multi-view image loading
    - Camera parameters (intrinsics, extrinsics)
    - Depth maps
    - Flow maps
    - Segmentation masks
    """

    def __init__(
        self,
        num_views=None,
        resolution=None,
        transform=None,
        seed=None,
        num_views_range=None,  # Optional: range of num_views to sample from, e.g., [4, 8]
    ):
        assert num_views is not None, "num_views must be specified"
        self.num_views = num_views
        self._set_resolutions(resolution)
        self.transform = transform
        self.seed = seed

        # Initialize RNG once in __init__
        if seed is not None:
            self._rng = np.random.default_rng(seed=seed)
        else:
            random_seed = torch.randint(0, 2**32, (1,)).item()
            self._rng = np.random.default_rng(seed=random_seed)

        # Set num_views range for random sampling
        if num_views_range is not None:
            assert len(num_views_range) == 2, "num_views_range must be [min, max]"
            self.num_views_range = num_views_range
        else:
            self.num_views_range = None

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        All processing is delegated to _get_views which is implemented by subclasses.

        Args:
            idx: Scene index

        Returns:
            List of view dictionaries in VGGT format
        """
        while True:
            try:
                views = self._get_views(idx, self._rng)
                break
            except Exception as e:
                print(f"Error in getting sample {idx}: {e}")
                idx = random.randint(0, len(self) - 1)

        return views

    def _get_views(self, idx, rng):
        """
        To be implemented by subclasses.

        This method should handle ALL processing including:
        1. Random sampling (resolution, num_views, camera_id, interval, etc.)
        2. Loading raw data (images, depth, camera params, etc.)
        3. Cropping and resizing
        4. Applying transforms
        5. Generating 3D points and ray maps
        6. Converting to VGGT format (coordinate transforms, normalization, etc.)

        Args:
            idx: Scene index
            rng: Random number generator

        Returns:
            List of view dictionaries in VGGT format ready for batching
        """
        raise NotImplementedError()

    @staticmethod
    def _get_ray_map(c2w1, c2w2, intrinsics, h, w):
        """Generate ray map for rendering"""
        c2w = np.linalg.inv(c2w1) @ c2w2
        i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
        grid = np.stack([i, j, np.ones_like(i)], axis=-1)
        ro = c2w[:3, 3]
        rd = np.linalg.inv(intrinsics) @ grid.reshape(-1, 3).T
        rd = (c2w @ np.vstack([rd, np.ones_like(rd[0])])).T[:, :3].reshape(h, w, 3)
        rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)
        ro = np.broadcast_to(ro, (h, w, 3))
        ray_map = np.concatenate([ro, rd], axis=-1)
        return ray_map

    def get_seq_from_start_id(
        self,
        num_views,
        id_ref,
        ids_all,
        rng,
        min_interval=1,
        max_interval=25,
    ):
        """
        Sample a sequence of consecutive views starting from id_ref.
        Always samples video frames (consecutive) with fixed intervals.

        Args:
            num_views: Number of views to sample
            id_ref: Reference ID (starting point)
            ids_all: List of all available IDs
            rng: Random number generator
            min_interval: Minimum interval between frames
            max_interval: Maximum interval between frames

        Returns:
            pos: List of positions in ids_all
            is_video: Always True (consecutive frames)
        """
        assert min_interval > 0
        assert min_interval <= max_interval
        assert id_ref in ids_all

        pos_ref = ids_all.index(id_ref)
        remaining_sum = len(ids_all) - 1 - pos_ref

        # We have enough frames for the sequence
        if remaining_sum >= num_views - 1:
            if remaining_sum == num_views - 1:
                # Exact fit: use all remaining frames
                return [pos_ref + i for i in range(num_views)], True

            # Calculate max possible interval
            max_interval = min(max_interval, 2 * remaining_sum // (num_views - 1))

            # Always use fixed interval (video mode)
            fixed_interval = rng.choice(range(min_interval, min(remaining_sum // (num_views - 1) + 1, max_interval + 1)))
            intervals = [fixed_interval for _ in range(num_views - 1)]

            pos = list(itertools.accumulate([pos_ref] + intervals))
            pos = [p for p in pos if p < len(ids_all)]

            # If we don't have enough positions, fill with remaining candidates
            if len(pos) < num_views:
                all_possible_pos = np.arange(pos_ref, len(ids_all))
                pos_candidates = [p for p in all_possible_pos if p not in pos]
                pos = pos + rng.choice(pos_candidates, num_views - len(pos), replace=False).tolist()

            pos = sorted(pos)  # Always sorted for video
        else:
            # Not enough frames - this should not happen in our use case
            raise ValueError(f"Not enough frames: need {num_views}, but only {remaining_sum + 1} available from position {pos_ref}")

        assert len(pos) == num_views
        return pos, True

    def _set_resolutions(self, resolutions):
        """Set target resolutions for images"""
        assert resolutions is not None, "resolution must be specified"

        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int) and isinstance(height, int)
            self._resolutions.append((width, height))

    def _crop_resize_if_necessary(
        self, image, depthmap, intrinsics, resolution, rng=None, info=None, flowmap=None, seg_mask=None
    ):
        """
        Crop and resize image to target resolution.
        Handles flowmap and segmentation mask if provided.
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # Crop centered on principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        assert min_margin_x > W / 5, f"Bad principal point in view={info}"
        assert min_margin_y > H / 5, f"Bad principal point in view={info}"

        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)

        # Crop
        if flowmap is not None or seg_mask is not None:
            image, depthmap, flowmap, seg_mask, intrinsics = cropping.crop_image_depthmap_flowmap_segmask(
                image, depthmap, flowmap, seg_mask, intrinsics, crop_bbox
            )
        else:
            image, depthmap, intrinsics = cropping.crop_image_depthmap(
                image, depthmap, intrinsics, crop_bbox
            )

        # Resize to target resolution
        W, H = image.size
        target_resolution = np.array(resolution)

        if flowmap is not None or seg_mask is not None:
            image, depthmap, flowmap, seg_mask, intrinsics = cropping.rescale_image_depthmap_flowmap_segmask(
                image, depthmap, flowmap, seg_mask, intrinsics, target_resolution
            )
        else:
            image, depthmap, intrinsics = cropping.rescale_image_depthmap(
                image, depthmap, intrinsics, target_resolution
            )

        # Final crop to exact resolution
        intrinsics2 = cropping.camera_matrix_of_crop(
            intrinsics, image.size, resolution, offset_factor=0.5
        )
        crop_bbox = cropping.bbox_from_intrinsics_in_out(
            intrinsics, intrinsics2, resolution
        )

        if flowmap is not None or seg_mask is not None:
            image, depthmap, flowmap, seg_mask, intrinsics2 = cropping.crop_image_depthmap_flowmap_segmask(
                image, depthmap, flowmap, seg_mask, intrinsics, crop_bbox
            )
            return image, depthmap, intrinsics2, flowmap, seg_mask
        else:
            image, depthmap, intrinsics2 = cropping.crop_image_depthmap(
                image, depthmap, intrinsics, crop_bbox
            )
            return image, depthmap, intrinsics2
