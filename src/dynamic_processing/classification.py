"""Object classification (car vs pedestrian)."""

import torch
from typing import Dict, List


def classify_objects(
    tracked_results: List[Dict],
    segment_logits: torch.Tensor,
    H: int,
    W: int
) -> Dict[int, str]:
    """
    Classify dynamic objects as 'car' or 'pedestrian'.

    Uses semantic segmentation logits across all frames.
    Waymo classes: 0=background, 1=vehicle, 2=sign, 3=pedestrian+cyclist

    Args:
        tracked_results: Tracking results with global_ids
        segment_logits: [B, V, H, W, 4] segmentation logits
        H: Image height
        W: Image width

    Returns:
        {object_id: 'car' or 'pedestrian'}
    """
    device = segment_logits.device
    object_classes = {}

    all_ids = set()
    for result in tracked_results:
        all_ids.update(result.get('global_ids', []))

    for object_id in all_ids:
        logits_list = []

        for frame_idx, result in enumerate(tracked_results):
            global_ids = result.get('global_ids', [])
            if object_id not in global_ids:
                continue

            cluster_idx = global_ids.index(object_id)
            cluster_indices = result.get('cluster_indices', [])
            if cluster_idx >= len(cluster_indices):
                continue

            pixel_indices = cluster_indices[cluster_idx]
            if not pixel_indices:
                continue

            view_indices = result.get('view_indices', [frame_idx])

            for view_idx in view_indices:
                if view_idx >= segment_logits.shape[1]:
                    continue

                frame_logits = segment_logits[0, view_idx]  # [H, W, 4]

                view_offset = view_indices.index(view_idx) * H * W if len(view_indices) > 1 else 0
                view_pixels = [p - view_offset for p in pixel_indices
                              if view_offset <= p < view_offset + H * W]

                if not view_pixels:
                    continue

                pixel_tensor = torch.tensor(view_pixels, dtype=torch.long, device=device)
                v_coords = pixel_tensor // W
                u_coords = pixel_tensor % W

                valid = (v_coords >= 0) & (v_coords < H) & (u_coords >= 0) & (u_coords < W)
                v_valid = v_coords[valid]
                u_valid = u_coords[valid]

                if len(v_valid) > 0:
                    pixel_logits = frame_logits[v_valid, u_valid]  # [N, 4]
                    logits_list.append(pixel_logits.detach())

        if not logits_list:
            object_classes[object_id] = 'car'
            continue

        all_logits = torch.cat(logits_list, dim=0)  # [Total, 4]
        summed_logits = all_logits.sum(dim=0)  # [4]
        probs = torch.softmax(summed_logits, dim=0)

        pred_class = torch.argmax(probs).item()

        if pred_class == 1:
            object_classes[object_id] = 'car'
        elif pred_class == 3:
            object_classes[object_id] = 'pedestrian'
        else:
            object_classes[object_id] = 'car'

    return object_classes
