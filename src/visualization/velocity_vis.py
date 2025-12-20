"""
Velocity visualization utilities
"""

import torch
from src.utils import scene_flow_to_rgb


def visualize_velocity(velocity, scale=0.2):
    """
    Visualize velocity as RGB image

    Args:
        velocity: [S, H, W, 3] tensor, velocity field
        scale: Velocity scale factor

    Returns:
        [S, 3, H, W] tensor (float32, [0, 1])
    """
    S, H, W, _ = velocity.shape
    velocity_rgb = scene_flow_to_rgb(velocity.detach(), scale).permute(0, 3, 1, 2)  # [S, 3, H, W]
    return velocity_rgb
