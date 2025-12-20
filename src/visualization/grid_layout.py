"""
Grid layout utilities for creating multi-camera visualization grids.
"""

import numpy as np
import cv2


def add_text_label(image, text, font_scale=1.0, thickness=2):
    """
    Add text label to the top-center of an image

    Args:
        image: numpy array [H, W, 3]
        text: text to add
        font_scale: font size scale
        thickness: text thickness

    Returns:
        Image with text label added at top-center
    """
    # Get text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Create label background
    label_height = text_height + baseline + 20  # 20 pixels padding
    label_bg = np.ones((label_height, image.shape[1], 3), dtype=np.uint8) * 255

    # Calculate text position (center-top)
    text_x = (image.shape[1] - text_width) // 2
    text_y = text_height + 10  # 10 pixels from top

    # Draw text on label background
    cv2.putText(label_bg, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Concatenate label with image
    return np.concatenate([label_bg, image], axis=0)


def create_context_reference_row(
    gt_rgb,
    context_indices,
    num_cameras,
    camera_indices,
    frame_indices,
    max_views_per_row=2
):
    """
    Create context frames reference row (STORM-style)
    Shows context frames with proper layout: white gaps and multiple rows

    Args:
        gt_rgb: [S, H, W, 3] GT RGB image
        context_indices: List of context frame indices
        num_cameras: Number of cameras
        camera_indices: List of camera indices
        frame_indices: List of frame indices
        max_views_per_row: Maximum number of context views per row (default: 2)

    Returns:
        context_section: Vertically stacked rows of context frames
    """
    # Set camera order based on number of cameras
    if num_cameras == 3:
        camera_order = [1, 0, 2]  # Center, left, right for 3 cameras
    else:
        camera_order = list(range(num_cameras))

    # Group context indices by frame_idx (temporal position)
    context_by_time = {}
    for ctx_idx in context_indices:
        frame_idx = frame_indices[ctx_idx]
        if frame_idx not in context_by_time:
            context_by_time[frame_idx] = []
        context_by_time[frame_idx].append(ctx_idx)

    context_frames_labeled = []
    for frame_idx in sorted(context_by_time.keys()):
        view_indices = context_by_time[frame_idx]

        # Sort by camera order
        if num_cameras > 1:
            sorted_views = []
            for cam_order_idx in camera_order:
                for v_idx in view_indices:
                    if camera_indices[v_idx] == cam_order_idx:
                        sorted_views.append(v_idx)
                        break
            view_indices = sorted_views

        # Concatenate cameras horizontally
        frame_concat = np.concatenate([gt_rgb[v] for v in view_indices], axis=1)
        # Add label
        frame_labeled = add_text_label(frame_concat, f"Context RGB (t={frame_idx})", font_scale=0.8, thickness=2)
        context_frames_labeled.append(frame_labeled)

    # Arrange context frames in rows with white gaps
    context_rows = []
    gap_width = 15
    white_gap = np.ones((context_frames_labeled[0].shape[0], gap_width, 3), dtype=np.uint8) * 255

    for i in range(0, len(context_frames_labeled), max_views_per_row):
        row_frames = context_frames_labeled[i:i+max_views_per_row]

        # Add white gaps between frames in the same row
        row_with_gaps = []
        for j, frame in enumerate(row_frames):
            row_with_gaps.append(frame)
            if j < len(row_frames) - 1:  # Don't add gap after last frame
                row_with_gaps.append(white_gap)

        # Concatenate frames horizontally
        context_row = np.concatenate(row_with_gaps, axis=1)
        context_rows.append(context_row)

    # Stack rows vertically with gaps
    if len(context_rows) > 1:
        row_gap_height = 15
        final_rows = []
        for i, row in enumerate(context_rows):
            final_rows.append(row)
            if i < len(context_rows) - 1:  # Don't add gap after last row
                row_gap = np.ones((row_gap_height, row.shape[1], 3), dtype=np.uint8) * 255
                final_rows.append(row_gap)
        context_section = np.concatenate(final_rows, axis=0)
    else:
        context_section = context_rows[0]

    return context_section


def create_multi_camera_grid(
    gt_rgb,
    gt_depth,
    pred_depth,
    pred_velocity,
    num_cameras,
    num_frames,
    camera_indices,
    frame_indices,
    pred_rgb=None,
    gt_velocity=None,
    gt_segmentation=None,
    pred_segmentation=None,
    dynamic_clustering=None,
    context_indices=None,
    target_indices=None,
    visualize_target_frames=False
):
    """
    Create multi-camera visualization grid with STORM-style layout for context/target frames

    When context/target distinction exists and visualize_target_frames=True:
    - Top: Context frames reference row (all context frames concatenated)
    - Below: Target frames visualization (one frame per video frame)

    Traditional mode (no context/target or visualize_target_frames=False):
    - Each row shows: GT (left-center-right) | Pred (left-center-right)
    - Camera order: center(front), left, right corresponding to camera_id [2, 1, 3]

    Layout:
    - Row 1: GT RGB (left-center-right) | Rendered RGB (left-center-right)
    - Row 2: GT Depth (left-center-right) | Rendered Depth (left-center-right)
    - Row 3: GT Velocity (left-center-right) | GT RGB + Pred Velocity fusion (left-center-right)
    - Row 4: GT Segmentation (left-center-right) | Pred Segmentation (left-center-right)
    - Row 5: Dynamic Clustering (left-center-right, full width)

    Args:
        gt_rgb: [S, H, W, 3] GT RGB image
        gt_depth: [S, H, W, 3] GT depth map (visualized)
        pred_depth: [S, H, W, 3] Predicted depth map (visualized)
        pred_velocity: [S, H, W, 3] Predicted velocity map (visualized)
        num_cameras: Number of cameras (should be 3)
        num_frames: Number of frames per camera
        camera_indices: List of camera indices
        frame_indices: List of frame indices
        pred_rgb: [S, H, W, 3] Predicted RGB image (with sky)
        gt_velocity: [S, H, W, 3] GT velocity map (visualized)
        gt_segmentation: [S, H, W, 3] GT segmentation map (visualized)
        pred_segmentation: [S, H, W, 3] Predicted segmentation map (visualized)
        dynamic_clustering: [S, H, W, 3] Dynamic clustering map (visualized)
        context_indices: List of context frame indices
        target_indices: List of target frame indices
        visualize_target_frames: Whether to use STORM-style layout for target frames

    Returns:
        List of video frames, each frame is a grid layout
    """
    if dynamic_clustering is not None:
        print(f"  dynamic_clustering shape: {dynamic_clustering.shape}")
    print(f"  visualize_target_frames: {visualize_target_frames}")
    print(f"  context_indices: {context_indices}")
    print(f"  target_indices: {target_indices}")

    grid_frames = []

    # Check if we should use STORM-style layout
    use_storm_layout = (visualize_target_frames and
                        context_indices is not None and
                        target_indices is not None and
                        len(target_indices) > 0)

    if use_storm_layout:
        print("[INFO] Using STORM-style layout with context reference row")
        # Create context reference row at the top
        context_row = create_context_reference_row(
            gt_rgb, context_indices, num_cameras, camera_indices, frame_indices
        )

        # Generate frames for all frames (context + target) to avoid jumpy video
        frames_to_visualize = list(range(num_frames))
    else:
        print("[INFO] Using traditional layout")
        context_row = None
        # Visualize all frames in traditional mode
        frames_to_visualize = list(range(num_frames))

    # Set camera order based on number of cameras
    if num_cameras == 3:
        camera_order = [1, 0, 2]  # Center, left, right for 3 cameras
    else:
        camera_order = list(range(num_cameras))  # Natural order for other cases

    for frame_idx in frames_to_visualize:
        frame_views_original = []
        for cam_idx in range(num_cameras):
            for view_idx in range(len(camera_indices)):
                if camera_indices[view_idx] == cam_idx and frame_indices[view_idx] == frame_idx:
                    frame_views_original.append(view_idx)
                    break

        if len(frame_views_original) == num_cameras:
            frame_views = [frame_views_original[i] for i in camera_order]
        else:
            print(f"[WARNING] frame_idx={frame_idx}: found {len(frame_views_original)} views, expected {num_cameras}")
            frame_views = frame_views_original

        # Skip this frame if no views were found
        if len(frame_views) == 0:
            print(f"[WARNING] Skipping frame_idx={frame_idx} - no views found")
            continue

        # Concatenate multi-camera images without gaps
        gt_rgb_concat = np.concatenate([gt_rgb[v] for v in frame_views], axis=1)
        if pred_rgb is not None:
            pred_rgb_concat = np.concatenate([pred_rgb[v] for v in frame_views], axis=1)
        else:
            pred_rgb_concat = gt_rgb_concat.copy()

        # Add labels to GT and Pred RGB, then concatenate with white gap
        camera_label = "" if num_cameras == 1 else ""  # Only label once for multi-camera
        gt_rgb_labeled = add_text_label(gt_rgb_concat, f"GT RGB {camera_label}(t={frame_idx})")
        pred_rgb_labeled = add_text_label(pred_rgb_concat, f"Pred RGB {camera_label}(t={frame_idx})")
        # Add white gap between GT and Pred
        gap_width = 20
        gap_rgb = np.ones((gt_rgb_labeled.shape[0], gap_width, 3), dtype=np.uint8) * 255
        row1 = np.concatenate([gt_rgb_labeled, gap_rgb, pred_rgb_labeled], axis=1)

        gt_depth_concat = np.concatenate([gt_depth[v] for v in frame_views], axis=1)
        if pred_depth is not None:
            pred_depth_concat = np.concatenate([pred_depth[v] for v in frame_views], axis=1)
        else:
            pred_depth_concat = np.zeros_like(gt_depth_concat)

        # Add labels and gap for depth
        gt_depth_labeled = add_text_label(gt_depth_concat, f"GT Depth (t={frame_idx})")
        pred_depth_labeled = add_text_label(pred_depth_concat, f"Pred Depth (t={frame_idx})")
        gap_depth = np.ones((gt_depth_labeled.shape[0], gap_width, 3), dtype=np.uint8) * 255
        row2 = np.concatenate([gt_depth_labeled, gap_depth, pred_depth_labeled], axis=1)

        # In STORM-style layout, only show RGB and Depth (skip velocity, segmentation, clustering)
        if use_storm_layout:
            # Add white gaps between rows
            row_gap_height = 15
            row_gap = np.ones((row_gap_height, row1.shape[1], 3), dtype=np.uint8) * 255

            grid_frame = np.concatenate([
                row1,
                row_gap,
                row2
            ], axis=0)

            # Add context row at the top
            if context_row is not None:
                # Ensure context_row width matches grid_frame width
                if context_row.shape[1] != grid_frame.shape[1]:
                    # Pad or resize context_row to match
                    if context_row.shape[1] < grid_frame.shape[1]:
                        # Pad with white space
                        pad_width = grid_frame.shape[1] - context_row.shape[1]
                        white_pad = np.ones((context_row.shape[0], pad_width, 3), dtype=np.uint8) * 255
                        context_row = np.concatenate([context_row, white_pad], axis=1)
                    else:
                        # Crop context_row (should not happen in normal cases)
                        context_row = context_row[:, :grid_frame.shape[1], :]

                # Add big gap between context row and content
                big_gap_height = 30
                big_gap = np.ones((big_gap_height, grid_frame.shape[1], 3), dtype=np.uint8) * 255

                grid_frame = np.concatenate([
                    context_row,
                    big_gap,
                    grid_frame
                ], axis=0)

            grid_frames.append(grid_frame)
            continue  # Skip the rest for STORM layout

        # Traditional layout: continue with velocity, segmentation, clustering
        if gt_velocity is not None:
            gt_velocity_concat = np.concatenate([gt_velocity[v] for v in frame_views], axis=1)
        else:
            gt_velocity_concat = np.zeros_like(gt_depth_concat)

        if pred_velocity is not None:
            fused_velocity = []
            velocity_alpha = 1.0
            for v in frame_views:
                gt_rgb_norm = gt_rgb[v].astype(np.float32) / 255.0
                pred_velocity_norm = pred_velocity[v].astype(np.float32) / 255.0
                fused = velocity_alpha * pred_velocity_norm + (1 - velocity_alpha) * gt_rgb_norm
                fused = np.clip(fused, 0, 1)
                fused = (fused * 255).astype(np.uint8)
                fused_velocity.append(fused)
            pred_velocity_concat = np.concatenate(fused_velocity, axis=1)
        else:
            pred_velocity_concat = gt_rgb_concat.copy()

        # Add labels and gap for velocity
        gt_velocity_labeled = add_text_label(gt_velocity_concat, f"GT Velocity (t={frame_idx})")
        pred_velocity_labeled = add_text_label(pred_velocity_concat, f"Pred Velocity (t={frame_idx})")
        gap_velocity = np.ones((gt_velocity_labeled.shape[0], gap_width, 3), dtype=np.uint8) * 255
        row3 = np.concatenate([gt_velocity_labeled, gap_velocity, pred_velocity_labeled], axis=1)

        if gt_segmentation is not None:
            gt_seg_concat = np.concatenate([gt_segmentation[v] for v in frame_views], axis=1)
        else:
            gt_seg_concat = np.zeros_like(gt_depth_concat)

        if pred_segmentation is not None:
            pred_seg_concat = np.concatenate([pred_segmentation[v] for v in frame_views], axis=1)
        else:
            pred_seg_concat = np.zeros_like(gt_depth_concat)

        # Add labels and gap for segmentation
        gt_seg_labeled = add_text_label(gt_seg_concat, f"GT Segmentation (t={frame_idx})")
        pred_seg_labeled = add_text_label(pred_seg_concat, f"Pred Segmentation (t={frame_idx})")
        gap_seg = np.ones((gt_seg_labeled.shape[0], gap_width, 3), dtype=np.uint8) * 255
        row4 = np.concatenate([gt_seg_labeled, gap_seg, pred_seg_labeled], axis=1)

        if dynamic_clustering is not None:
            clustering_concat = np.concatenate([dynamic_clustering[v] for v in frame_views], axis=1)
            # Add label for clustering (full width)
            clustering_labeled = add_text_label(clustering_concat, f"Dynamic Clustering (t={frame_idx})")
            # Make row5 full width by adding white space
            H = clustering_labeled.shape[0]
            target_W = row1.shape[1]
            current_W = clustering_labeled.shape[1]
            if current_W < target_W:
                white_space = np.ones((H, target_W - current_W, 3), dtype=np.uint8) * 255
                row5 = np.concatenate([clustering_labeled, white_space], axis=1)
            else:
                row5 = clustering_labeled
        else:
            H = row1.shape[0]
            W = row1.shape[1]
            row5 = np.zeros((H, W, 3), dtype=np.uint8)

        # Add white gaps between rows
        row_gap_height = 15
        row_gap = np.ones((row_gap_height, row1.shape[1], 3), dtype=np.uint8) * 255

        # Traditional layout (not STORM): show all rows
        grid_frame = np.concatenate([
            row1,
            row_gap,
            row2,
            row_gap,
            row3,
            row_gap,
            row4,
            row_gap,
            row5
        ], axis=0)

        grid_frames.append(grid_frame)

    return grid_frames
