"""
Clustering visualization utilities.
"""

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt


def visualize_clustering_results(clustering_results, num_colors=20):
    """Generate colors for clustering results"""
    colors = plt.cm.tab20(np.linspace(0, 1, num_colors))
    colors = (colors[:, :3] * 255).astype(np.uint8)

    colored_results = []

    for frame_result in clustering_results:
        points = frame_result['points']
        labels = frame_result['labels']
        global_ids = frame_result.get('global_ids', [])

        point_colors = np.zeros((len(points), 3), dtype=np.uint8)

        if isinstance(labels, torch.Tensor):
            unique_labels = torch.unique(labels)
        else:
            unique_labels = np.unique(labels)

        colors_assigned = 0
        for label in unique_labels:
            if isinstance(label, torch.Tensor):
                label_val = label.item()
            else:
                label_val = int(label)

            if label_val == -1:
                continue

            if label_val < len(global_ids):
                global_id = global_ids[label_val]
                if global_id != -1:
                    color_idx = global_id % num_colors
                    color = colors[color_idx]

                    if isinstance(labels, torch.Tensor):
                        mask = (labels == label).cpu().numpy()
                    else:
                        mask = (labels == label)

                    point_colors[mask] = color
                    colors_assigned += 1

        colored_results.append({
            'points': points,
            'colors': point_colors,
            'num_clusters': colors_assigned
        })

    return colored_results


def create_clustering_visualization(matched_clustering_results, vggt_batch, fusion_alpha=0.7):
    """Create visualization from matched_clustering_results (returns [S, H, W, 3] uint8 numpy format)"""
    try:
        B, S, C, image_height, image_width = vggt_batch["images"].shape

        if not matched_clustering_results or len(matched_clustering_results) == 0:
            images_np = (vggt_batch["images"][0].cpu().numpy() * 255).astype(np.uint8)
            images_np = images_np.transpose(0, 2, 3, 1)
            return images_np

        colored_results = visualize_clustering_results(matched_clustering_results, num_colors=20)

        camera_indices = vggt_batch.get('camera_indices', None)
        frame_indices = vggt_batch.get('frame_indices', None)
        is_multi_camera = (camera_indices is not None and frame_indices is not None)

        if is_multi_camera:
            camera_indices = camera_indices[0].cpu().numpy() if isinstance(camera_indices, torch.Tensor) else camera_indices
            frame_indices = frame_indices[0].cpu().numpy() if isinstance(frame_indices, torch.Tensor) else frame_indices

            num_cameras = len(np.unique(camera_indices))
            num_frames = len(colored_results)

            clustering_images = []
            for view_idx in range(S):
                cam_idx = camera_indices[view_idx]
                frame_idx = frame_indices[view_idx]

                source_rgb = vggt_batch["images"][0, view_idx].permute(1, 2, 0)
                source_rgb = (source_rgb * 255).cpu().numpy().astype(np.uint8)

                # Check if frame_idx is within colored_results range
                if frame_idx >= len(colored_results):
                    fused_image = source_rgb.copy()
                    clustering_images.append(fused_image)
                    continue

                colored_result = colored_results[frame_idx]
                point_colors = colored_result['colors']

                start_idx = cam_idx * image_height * image_width
                end_idx = start_idx + image_height * image_width
                camera_point_colors = point_colors[start_idx:end_idx]

                if colored_result['num_clusters'] > 0 and np.any(camera_point_colors > 0):
                    clustering_image = camera_point_colors.reshape(image_height, image_width, 3)

                    mask = np.any(clustering_image > 0, axis=2)[:, :, np.newaxis]
                    fused_image = np.where(mask,
                                         (fusion_alpha * clustering_image + (1 - fusion_alpha) * source_rgb).astype(np.uint8),
                                         source_rgb)
                else:
                    fused_image = source_rgb.copy()

                clustering_images.append(fused_image)
        else:
            clustering_images = []
            for frame_idx, colored_result in enumerate(colored_results):
                source_rgb = vggt_batch["images"][0, frame_idx].permute(1, 2, 0)
                source_rgb = (source_rgb * 255).cpu().numpy().astype(np.uint8)

                if colored_result['num_clusters'] > 0:
                    point_colors = colored_result['colors']
                    clustering_image = point_colors.reshape(image_height, image_width, 3)

                    mask = np.any(clustering_image > 0, axis=2)
                    mask = mask[:, :, np.newaxis]

                    fused_image = np.where(mask,
                                         (fusion_alpha * clustering_image + (1 - fusion_alpha) * source_rgb).astype(np.uint8),
                                         source_rgb)
                else:
                    fused_image = source_rgb.copy()
                    cv2.putText(fused_image, "No Dynamic Objects", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                clustering_images.append(fused_image)

        return np.array(clustering_images)

    except Exception as e:
        print(f"Error creating clustering visualization: {e}")
        import traceback
        traceback.print_exc()
        images_np = (vggt_batch["images"][0].cpu().numpy() * 255).astype(np.uint8)
        images_np = images_np.transpose(0, 2, 3, 1)
        return images_np
