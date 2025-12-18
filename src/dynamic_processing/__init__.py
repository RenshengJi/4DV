"""
Dynamic object processing module for 4D scene reconstruction.

Unified architecture for single and multi-camera processing.
"""

from .processor import DynamicProcessor
from .types import (
    ProcessingResult,
    DynamicObject,
    ViewMapping,
    ViewIndex,
    ClusteringResult
)
from .registration import VelocityRegistration

__all__ = [
    'DynamicProcessor',
    'ProcessingResult',
    'DynamicObject',
    'ViewMapping',
    'ViewIndex',
    'ClusteringResult',
    'VelocityRegistration'
]
