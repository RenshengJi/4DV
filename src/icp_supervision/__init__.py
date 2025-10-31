"""
ICP Supervision Module for Gaussian Refinement

This module provides strong supervision for training the Gaussian Refinement network
using ICP (Iterative Closest Point) registration as ground truth.

Main components:
- data_generator: Offline GT generation using ICP on multi-frame point clouds
- dataset: Custom Dataset for loading ICP sample pairs
- icp_loss: Training loss with ICP supervision
- train: Standalone training script
- visualize: Visualization tools for verifying ICP GT correctness
"""

__version__ = "1.0.0"
__author__ = "VGGT Team"

from .data_generator import ICPDataGenerator
from .dataset import ICPSupervisionDataset
from .icp_loss import ICPSupervisionLoss
from .utils import (
    gaussians_to_pointcloud,
    pointcloud_to_gaussians,
    save_pointcloud_visualization,
    load_sample_pair
)

__all__ = [
    'ICPDataGenerator',
    'ICPSupervisionDataset',
    'ICPSupervisionLoss',
    'gaussians_to_pointcloud',
    'pointcloud_to_gaussians',
    'save_pointcloud_visualization',
    'load_sample_pair',
]
