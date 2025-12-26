"""
Frame index mapping utilities for dynamic objects
"""

import torch
from typing import Dict, List, Optional


def remap_dynamic_objects_frame_indices(
    dynamic_objects_data: Dict,
    frame_mapping: Dict[int, int],
    device: torch.device
) -> None:
    """
    Remap frame indices in dynamic objects from context frame indices to global frame indices.
    Updates dynamic_objects_data in-place.

    Args:
        dynamic_objects_data: Dictionary containing 'dynamic_objects_cars' and 'dynamic_objects_people'
        frame_mapping: Mapping from global frame index to context frame index {global_idx: context_idx}
        device: torch device for tensor creation

    Note:
        This function modifies dynamic_objects_data in-place by updating:
        - frame_transforms keys
        - frame_pixel_indices keys
        - frame_existence tensor
        - reference_frame value
    """
    if dynamic_objects_data is None or frame_mapping is None:
        return

    # Create inverse mapping: context frame idx -> global frame idx
    context_to_global = {ctx_idx: global_idx for global_idx, ctx_idx in frame_mapping.items()}

    # Remap cars
    for car in dynamic_objects_data.get('dynamic_objects_cars', []):
        _remap_object_frame_indices(car, context_to_global, device, has_gaussians=True)

    # Remap pedestrians
    for person in dynamic_objects_data.get('dynamic_objects_people', []):
        _remap_object_frame_indices(person, context_to_global, device, has_gaussians=False)


def _remap_object_frame_indices(
    obj_data: Dict,
    context_to_global: Dict[int, int],
    device: torch.device,
    has_gaussians: bool = True
) -> None:
    """
    Remap frame indices for a single dynamic object.

    Args:
        obj_data: Single object data dictionary
        context_to_global: Mapping from context frame index to global frame index
        device: torch device
        has_gaussians: Whether object has canonical gaussians (True for cars, False for people)
    """
    # Remap frame_transforms (for cars with canonical representation)
    if has_gaussians and obj_data.get('frame_transforms') is not None:
        obj_data['frame_transforms'] = {
            context_to_global[ctx_frame]: transform
            for ctx_frame, transform in obj_data['frame_transforms'].items()
        }

    # Remap frame_gaussians (for people with per-frame gaussians)
    if not has_gaussians and obj_data.get('frame_gaussians') is not None:
        obj_data['frame_gaussians'] = {
            context_to_global[ctx_frame]: gaussians
            for ctx_frame, gaussians in obj_data['frame_gaussians'].items()
        }

    # Remap frame_velocities (for people)
    if obj_data.get('frame_velocities') is not None:
        obj_data['frame_velocities'] = {
            context_to_global[ctx_frame]: velocity
            for ctx_frame, velocity in obj_data['frame_velocities'].items()
        }

    # Remap frame_pixel_indices
    if obj_data.get('frame_pixel_indices') is not None:
        obj_data['frame_pixel_indices'] = {
            context_to_global[ctx_frame]: view_dict
            for ctx_frame, view_dict in obj_data['frame_pixel_indices'].items()
        }

    # Remap frame_existence tensor
    if obj_data.get('frame_existence') is not None:
        obj_data['frame_existence'] = _remap_frame_existence(
            obj_data['frame_existence'], context_to_global, device
        )

    # Remap reference_frame
    if obj_data.get('reference_frame') is not None:
        obj_data['reference_frame'] = context_to_global[obj_data['reference_frame']]


def _remap_frame_existence(
    context_existence: torch.Tensor,
    context_to_global: Dict[int, int],
    device: torch.device
) -> torch.Tensor:
    """
    Remap frame_existence tensor from context frame indices to global frame indices.

    Args:
        context_existence: [max_context_frame + 1] boolean tensor
        context_to_global: Mapping from context to global frame indices
        device: torch device

    Returns:
        global_existence: [max_global_frame + 1] boolean tensor
    """
    # Find which context frames the object exists in
    existing_context_frames = [
        i for i in range(len(context_existence)) if context_existence[i]
    ]

    if not existing_context_frames:
        return torch.tensor([], dtype=torch.bool, device=device)

    # Map to global frames
    existing_global_frames = [
        context_to_global[ctx_f] for ctx_f in existing_context_frames
    ]

    # Create new frame_existence tensor
    max_global_frame = max(existing_global_frames)
    global_existence = torch.zeros(max_global_frame + 1, dtype=torch.bool, device=device)

    for global_f in existing_global_frames:
        global_existence[global_f] = True

    return global_existence


def extract_velocity_from_predictions(
    dynamic_objects_people: List[Dict],
    preds: Dict,
    context_indices: torch.Tensor,
    frame_pixel_indices_mapping: Dict[int, Dict[int, int]]
) -> None:
    """
    Extract and store average velocity for each person from prediction velocity maps.
    Updates dynamic_objects_people in-place by adding 'frame_velocities'.

    Args:
        dynamic_objects_people: List of people object dictionaries
        preds: Model predictions containing 'velocity_global'
        context_indices: Tensor of context frame view indices
        frame_pixel_indices_mapping: Mapping to convert view indices
    """
    pred_velocity_global = preds.get('velocity_global', None)
    if pred_velocity_global is None:
        return

    for person in dynamic_objects_people:
        person['frame_velocities'] = {}
        frame_pixel_indices = person.get('frame_pixel_indices', {})

        for global_frame_idx, view_dict in frame_pixel_indices.items():
            frame_velocities = []

            for view_idx, pixel_list in view_dict.items():
                if len(pixel_list) == 0:
                    continue

                # Map view_idx to actual context view index
                if view_idx < len(context_indices):
                    actual_view_idx = context_indices[view_idx].item()
                    vel_map = pred_velocity_global[0, view_idx]  # [H, W, 3]
                    H_vel, W_vel = vel_map.shape[:2]

                    # Extract velocities for these pixels
                    for pixel_idx in pixel_list:
                        v = pixel_idx // W_vel
                        u = pixel_idx % W_vel
                        if v < H_vel and u < W_vel:
                            pixel_vel = vel_map[v, u]
                            frame_velocities.append(pixel_vel)

            if frame_velocities:
                # Average velocity for this person in this frame
                avg_velocity = torch.stack(frame_velocities).mean(dim=0)
                person['frame_velocities'][global_frame_idx] = avg_velocity
