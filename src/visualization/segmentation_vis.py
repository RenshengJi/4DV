"""
Segmentation visualization utilities.
"""

import numpy as np
import torch


def get_segmentation_colormap(num_classes=4):
    """Generate segmentation colormap based on Waymo classes (Cityscapes style)

    Waymo class definitions (4 classes):
        0: background/unlabeled
        1: vehicle
        2: sign
        3: pedestrian + cyclist
    """
    colormap = np.zeros((num_classes, 3), dtype=np.uint8)
    colormap[0] = [255, 255, 255]     # background - white
    colormap[1] = [0, 0, 142]         # vehicle - dark blue
    colormap[2] = [220, 220, 0]       # sign - yellow
    colormap[3] = [220, 20, 60]       # pedestrian+cyclist - red
    return colormap


def visualize_segmentation(seg_labels, seg_mask=None, num_classes=4):
    """Visualize segmentation labels as RGB image (returns [S, H, W, 3] uint8 numpy format)"""
    colormap = get_segmentation_colormap(num_classes)
    if isinstance(seg_labels, torch.Tensor):
        seg_labels = seg_labels.cpu().numpy()

    if seg_mask is not None and isinstance(seg_mask, torch.Tensor):
        seg_mask = seg_mask.cpu().numpy()

    # Handle shape
    if seg_labels.ndim == 2:
        seg_labels = seg_labels[np.newaxis, ...]
        if seg_mask is not None:
            seg_mask = seg_mask[np.newaxis, ...]

    S, H, W = seg_labels.shape
    seg_rgb = np.zeros((S, H, W, 3), dtype=np.uint8)

    for s in range(S):
        for class_id in range(num_classes):
            mask = (seg_labels[s] == class_id)
            seg_rgb[s][mask] = colormap[class_id]

        if seg_mask is not None:
            invalid_mask = (seg_mask[s] == 0)
            seg_rgb[s][invalid_mask] = [128, 128, 128]

    return seg_rgb
