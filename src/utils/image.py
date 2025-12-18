"""
Image utility functions migrated from dust3r.
"""
import os
import numpy as np
import torch
from typing import Optional, Union
from collections import namedtuple
from itertools import accumulate

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """Open an image or a depthmap with opencv-python."""
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f"Could not load image={path} with {options=}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)


def _make_colorwheel(
    transitions: tuple = DEFAULT_TRANSITIONS, backend="torch"
) -> Union[np.ndarray, torch.Tensor]:
    """Creates a colorwheel (borrowed/modified from flowpy).
    A colorwheel defines the transitions between the six primary hues:
    Red(255, 0, 0), Yellow(255, 255, 0), Green(0, 255, 0), Cyan(0, 255, 255), Blue(0, 0, 255) and Magenta(255, 0, 255).
    Args:
        transitions: Contains the length of the six transitions, based on human color perception.
    Returns:
        colorwheel: The RGB values of the transitions in the color space.
    """
    colorwheel_length = sum(transitions)
    # The red hue is repeated to make the colorwheel cyclic
    base_hues = map(
        np.array,
        (
            [255, 0, 0],
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255],
            [0, 0, 255],
            [255, 0, 255],
            [255, 0, 0],
        ),
    )
    colorwheel = np.zeros((colorwheel_length, 3), dtype="uint8")
    hue_from = next(base_hues)
    start_index = 0
    for hue_to, end_index in zip(base_hues, accumulate(transitions)):
        transition_length = end_index - start_index
        colorwheel[start_index:end_index] = np.linspace(
            hue_from, hue_to, transition_length, endpoint=False
        )
        hue_from = hue_to
        start_index = end_index
    if backend == "torch":
        return torch.FloatTensor(colorwheel)
    else:
        return colorwheel


def scene_flow_to_rgb(
    flow: torch.Tensor,
    flow_max_radius: Optional[float] = None,
    background: Optional[str] = "bright",
) -> Union[torch.Tensor, np.ndarray]:
    """Creates a RGB representation of an optical flow (borrowed/modified from flowpy).
    Adapted from https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior/blob/main/visualize.py
    Args:
        flow: scene flow.
            flow[..., 0] should be the x-displacement
            flow[..., 1] should be the y-displacement
            flow[..., 2] should be the z-displacement
        flow_max_radius: Set the radius that gives the maximum color intensity, useful for comparing different flows.
            Default: The normalization is based on the input flow maximum radius.
        background: States if zero-valued flow should look 'bright' or 'dark'.
    Returns: An array of RGB colors.
    """
    valid_backgrounds = ("bright", "dark")
    if background not in valid_backgrounds:
        raise ValueError(
            f"background should be one the following: {valid_backgrounds}, not {background}."
        )
    if isinstance(flow, np.ndarray):
        backend = "np"
        op = np
    else:
        backend = "torch"
        op = torch

    # For scene flow, it's reasonable to assume displacements in x and y directions only for visualization pursposes.
    complex_flow = flow[..., 0] + 1j * flow[..., 1]
    radius, angle = op.abs(complex_flow), op.angle(complex_flow)
    if flow_max_radius is None:
        # flow_max_radius = torch.max(radius)
        flow_max_radius = op.quantile(radius, 0.99)
    if flow_max_radius > 0:
        radius /= flow_max_radius
    # Map the angles from (-pi, pi] to [0, 2pi) to [0, ncols - 1)
    wheel = _make_colorwheel(backend=backend)
    n_cols = len(wheel)
    wheel = op.vstack((wheel, wheel[0]))
    angle[angle < 0] += 2 * np.pi
    angle = angle * ((n_cols - 1) / (2 * np.pi))

    # Interpolate the hues
    angle_fractional, angle_floor, angle_ceil = (
        op.fmod(angle, 1),
        op.trunc(angle),
        op.ceil(angle),
    )
    angle_fractional = angle_fractional[..., None]
    if backend == "torch":
        _wheel = wheel.to(angle_floor.device)
        float_hue = (
            _wheel[angle_floor.long()] * (1 - angle_fractional)
            + _wheel[angle_ceil.long()] * angle_fractional
        )
    else:
        float_hue = (
            wheel[angle_floor.astype(op.int64)] * (1 - angle_fractional)
            + wheel[angle_ceil.astype(op.int64)] * angle_fractional
        )
    ColorizationArgs = namedtuple(
        "ColorizationArgs",
        ["move_hue_valid_radius", "move_hue_oversized_radius", "invalid_color"],
    )

    def move_hue_on_V_axis(hues, factors):
        return hues * factors[..., None]

    def move_hue_on_S_axis(hues, factors):
        return 255.0 - factors[..., None] * (255.0 - hues)

    if background == "dark":
        parameters = ColorizationArgs(
            move_hue_on_V_axis, move_hue_on_S_axis, op.tensor([255.0, 255.0, 255.0]) if backend == "torch" else op.array([255.0, 255.0, 255.0])
        )
    else:
        parameters = ColorizationArgs(move_hue_on_S_axis, move_hue_on_V_axis, op.zeros(3))
    colors = parameters.move_hue_valid_radius(float_hue, radius)
    oversized_radius_mask = radius > 1
    colors[oversized_radius_mask] = parameters.move_hue_oversized_radius(
        float_hue[oversized_radius_mask], 1 / radius[oversized_radius_mask]
    )
    colors = colors / 255.0
    return colors
