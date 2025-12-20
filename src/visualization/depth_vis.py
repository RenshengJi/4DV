"""
Depth visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_depth(depth):
    """
    Visualize depth as color image

    Args:
        depth: [S, H, W] or [S, H, W, 1] numpy array

    Returns:
        [S, H, W, 3] numpy array (uint8)
    """
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)  # [S, H, W, 1] -> [S, H, W]

    if depth.ndim != 3:
        raise ValueError(f"Expected depth with shape [S, H, W], got shape {depth.shape}")

    S, H, W = depth.shape
    depth_vis = []

    for s in range(S):
        d = depth[s]
        valid_mask = d > 0

        if valid_mask.any():
            vmin, vmax = np.percentile(d[valid_mask], [2, 98])
            d_norm = np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1)
        else:
            d_norm = np.zeros_like(d)

        d_colored = (plt.cm.viridis(d_norm)[..., :3] * 255).astype(np.uint8)
        depth_vis.append(d_colored)

    return np.array(depth_vis)
