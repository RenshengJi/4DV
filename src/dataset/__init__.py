"""
Simplified dataset module for our training pipeline.
"""
from .base_dataset import BaseDataset
from .waymo import WaymoDataset
from .easy_dataset import EasyDataset, MulDataset, ResizedDataset, CatDataset

# Import transforms and sampler (migrated from dust3r)
from .transforms import ImgNorm
from .batched_sampler import BatchedRandomSampler

from accelerate import Accelerator
import torch

# Create alias for compatibility with config files
Waymo_Multi = WaymoDataset


def vggt_collate_fn(batch):
    """
    Custom collate function for VGGT format data.
    Supports both single-camera and multi-camera modes.

    Args:
        batch: List of samples, where each sample is a list of view dicts

    Returns:
        Dict with batched tensors in VGGT format:
        - images: [B, S, 3, H, W] (or [B, C*S, 3, H, W] in multi-camera mode)
        - depths: [B, S, H, W]
        - intrinsics: [B, S, 3, 3]
        - extrinsics: [B, S, 4, 4]
        - point_masks: [B, S, H, W]
        - world_points: [B, S, H, W, 3]
        - flowmap: [B, S, H, W, C] (optional)
        - segment_label: [B, S, H, W] (optional)
        - segment_mask: [B, S, H, W] (optional)
        - sky_masks: [B, S, H, W] (optional)
        - depth_scale_factor: [B] (optional)
        - camera_indices: [B, S] (optional, multi-camera mode)
        - frame_indices: [B, S] (optional, multi-camera mode)
    """
    # batch is a list of samples
    # Each sample is a list of view dicts

    batch_size = len(batch)
    num_views = len(batch[0])

    # Initialize output dict
    output = {}

    # Get keys from first view of first sample
    sample_keys = batch[0][0].keys()

    # Keys that should be batched as tensors
    tensor_keys = {
        'img', 'depthmap', 'camera_intrinsics', 'camera_pose',
        'valid_mask', 'pts3d', 'flowmap', 'segment_label',
        'segment_mask', 'depth_scale_factor', 'sky_mask'
    }

    # Keys to skip (metadata)
    skip_keys = {'idx', 'dataset', 'label', 'instance', 'is_video',
                 'is_metric', 'quantile', 'rng', 'true_shape', 'ray_map',
                 'camera_idx', 'frame_idx'}  # Skip camera_idx and frame_idx from tensor_keys

    for key in sample_keys:
        if key in skip_keys:
            continue

        if key not in tensor_keys:
            continue

        # Check if all samples have this key with valid values
        has_key = all(
            key in view and view[key] is not None
            for sample in batch
            for view in sample
        )

        if not has_key:
            output[key] = None
            continue

        # Stack across batch and views
        try:
            if key == 'depth_scale_factor':
                # This is per-sample, not per-view
                tensors = [sample[0][key] for sample in batch]
                output[key] = torch.stack(tensors, dim=0)  # [B]
            else:
                # Stack across views first, then batch
                batch_tensors = []
                for sample in batch:
                    view_tensors = [view[key] for view in sample]
                    batch_tensors.append(torch.stack(view_tensors, dim=0))  # [S, ...]
                output[key] = torch.stack(batch_tensors, dim=0)  # [B, S, ...]
        except Exception as e:
            print(f"Error stacking key {key}: {e}")
            output[key] = None

    # ========== Extract camera_indices and frame_indices for multi-camera mode ==========
    # Check if the first view of the first sample has camera_idx and frame_idx fields
    if 'camera_idx' in batch[0][0] and 'frame_idx' in batch[0][0]:
        # Multi-camera mode: extract indices
        camera_indices_list = []
        frame_indices_list = []

        for sample in batch:
            sample_camera_indices = [view['camera_idx'] for view in sample]
            sample_frame_indices = [view['frame_idx'] for view in sample]

            camera_indices_list.append(torch.tensor(sample_camera_indices, dtype=torch.long))
            frame_indices_list.append(torch.tensor(sample_frame_indices, dtype=torch.long))

        output['camera_indices'] = torch.stack(camera_indices_list, dim=0)  # [B, S] or [B, C*S]
        output['frame_indices'] = torch.stack(frame_indices_list, dim=0)    # [B, S] or [B, C*S]
    else:
        # Single-camera mode: no camera indices
        output['camera_indices'] = None
        output['frame_indices'] = None

    # Rename keys to match VGGT format
    vggt_batch = {
        'images': output.get('img'),
        'depths': output.get('depthmap'),
        'intrinsics': output.get('camera_intrinsics'),
        'extrinsics': output.get('camera_pose'),
        'point_masks': output.get('valid_mask'),
        'world_points': output.get('pts3d'),
        'flowmap': output.get('flowmap'),
        'segment_label': output.get('segment_label'),
        'segment_mask': output.get('segment_mask'),
        'depth_scale_factor': output.get('depth_scale_factor'),
        'sky_masks': output.get('sky_mask'),
        'camera_indices': output.get('camera_indices'),  # New field
        'frame_indices': output.get('frame_indices'),    # New field
    }

    return vggt_batch


def get_data_loader(
    dataset,
    batch_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    accelerator: Accelerator = None,
    fixed_length=False,
):
    """
    Create a data loader for the given dataset.

    Args:
        dataset: Dataset instance or string to eval
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch
        pin_mem: Whether to pin memory
        accelerator: Accelerator instance for distributed training
        fixed_length: Whether to use fixed length sampling

    Returns:
        DataLoader instance
    """
    # Eval string if necessary
    if isinstance(dataset, str):
        dataset = eval(dataset)

    # Try to use custom sampler if available
    try:
        sampler = dataset.make_sampler(
            batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            world_size=accelerator.num_processes,
            fixed_length=fixed_length
        )
        shuffle = False

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_mem,
            collate_fn=vggt_collate_fn,
        )

    except (AttributeError, NotImplementedError):
        # Fallback to standard DataLoader
        sampler = None

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=drop_last,
            collate_fn=vggt_collate_fn,
        )

    return data_loader


__all__ = [
    'BaseDataset',
    'WaymoDataset',
    'Waymo_Multi',
    'EasyDataset',
    'MulDataset',
    'ResizedDataset',
    'CatDataset',
    'ImgNorm',
    'BatchedRandomSampler',
    'get_data_loader',
    'vggt_collate_fn',
]

