"""
Utility functions migrated from dust3r.
"""
import torch
from contextlib import contextmanager


@contextmanager
def tf32_off():
    """Context manager to temporarily disable TF32 for CUDA matmul operations."""
    original = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False  # disable tf32 temporarily
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original  # restore original setting
