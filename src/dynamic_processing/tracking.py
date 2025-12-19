"""Cross-frame object tracking and matching."""

import numpy as np
import torch
from typing import List, Dict
from scipy.optimize import linear_sum_assignment

from .types import ClusteringResult


def track_objects_across_frames(
    clustering_results: List[ClusteringResult],
    position_threshold: float = 0.5,
    velocity_threshold: float = 0.2
) -> List[Dict]:
    """
    Track objects across frames using Hungarian algorithm.

    Args:
        clustering_results: Clustering results for each frame
        position_threshold: Position matching threshold
        velocity_threshold: Velocity matching threshold (currently unused, for future)

    Returns:
        Updated clustering results with global_ids added
    """
    if len(clustering_results) == 0:
        return []

    next_global_id = 0
    global_tracks = {}  # {global_id: {frame_id, center, velocity, size}}
    results_with_ids = []

    for frame_idx, result in enumerate(clustering_results):
        result_dict = {
            'points': result.points,
            'labels': result.labels,
            'num_clusters': result.num_clusters,
            'cluster_centers': result.cluster_centers,
            'cluster_velocities': result.cluster_velocities,
            'cluster_sizes': result.cluster_sizes,
            'cluster_indices': result.cluster_indices,
            'view_indices': result.view_indices
        }

        if result.num_clusters == 0:
            result_dict['global_ids'] = []
            results_with_ids.append(result_dict)
            continue

        if frame_idx == 0:
            global_ids = list(range(next_global_id, next_global_id + result.num_clusters))
            next_global_id += result.num_clusters

            for i, gid in enumerate(global_ids):
                global_tracks[gid] = {
                    'frame_id': frame_idx,
                    'center': result.cluster_centers[i],
                    'velocity': result.cluster_velocities[i],
                    'size': result.cluster_sizes[i]
                }
        else:
            prev_result = results_with_ids[frame_idx - 1]
            prev_global_ids = prev_result.get('global_ids', [])

            if len(prev_global_ids) == 0:
                global_ids = list(range(next_global_id, next_global_id + result.num_clusters))
                next_global_id += result.num_clusters

                for i, gid in enumerate(global_ids):
                    global_tracks[gid] = {
                        'frame_id': frame_idx,
                        'center': result.cluster_centers[i],
                        'velocity': result.cluster_velocities[i],
                        'size': result.cluster_sizes[i]
                    }
            else:
                num_prev = len(prev_global_ids)
                num_curr = result.num_clusters
                cost_matrix = np.full((num_prev, num_curr), float('inf'))

                for i, prev_gid in enumerate(prev_global_ids):
                    track = global_tracks[prev_gid]
                    prev_center = track['center']
                    prev_velocity = track['velocity']
                    predicted_center = prev_center + prev_velocity

                    for j in range(num_curr):
                        curr_center = result.cluster_centers[j]
                        pos_distance = torch.norm(curr_center - predicted_center).item()

                        if pos_distance < position_threshold:
                            cost_matrix[i, j] = pos_distance

                if np.any(cost_matrix < float('inf')):
                    row_indices, col_indices = linear_sum_assignment(cost_matrix)

                    global_ids = [-1] * num_curr
                    matched_curr = set()

                    for i, j in zip(row_indices, col_indices):
                        if cost_matrix[i, j] < float('inf'):
                            prev_gid = prev_global_ids[i]
                            global_ids[j] = prev_gid
                            matched_curr.add(j)

                            global_tracks[prev_gid] = {
                                'frame_id': frame_idx,
                                'center': result.cluster_centers[j],
                                'velocity': result.cluster_velocities[j],
                                'size': result.cluster_sizes[j]
                            }

                    for j in range(num_curr):
                        if j not in matched_curr:
                            gid = next_global_id
                            next_global_id += 1
                            global_ids[j] = gid

                            global_tracks[gid] = {
                                'frame_id': frame_idx,
                                'center': result.cluster_centers[j],
                                'velocity': result.cluster_velocities[j],
                                'size': result.cluster_sizes[j]
                            }
                else:
                    global_ids = list(range(next_global_id, next_global_id + result.num_clusters))
                    next_global_id += result.num_clusters

                    for i, gid in enumerate(global_ids):
                        global_tracks[gid] = {
                            'frame_id': frame_idx,
                            'center': result.cluster_centers[i],
                            'velocity': result.cluster_velocities[i],
                            'size': result.cluster_sizes[i]
                        }

        result_dict['global_ids'] = global_ids
        results_with_ids.append(result_dict)

    return results_with_ids
