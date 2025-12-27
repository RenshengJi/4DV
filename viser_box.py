#!/usr/bin/env python3
"""
Viser-based 3D bounding box visualization for Waymo dataset.

Visualizes 3D bounding boxes extracted by preprocess_waymo_box.py.
Shows boxes in world coordinates with point cloud context.

Supports:
- Multi-camera point clouds (default: cameras 1, 2, 3)
- Camera visibility filtering for boxes
- Vehicle, pedestrian, cyclist display

Usage:
    python viser_box.py --data_dir data/waymo/train_full_test --scene_idx 0
    python viser_box.py --data_dir data/waymo/train_full_test --scene_name <sequence_name>
"""

import os
import sys
import time
import argparse
import math
from typing import List, Dict, Optional

import numpy as np
import viser
import viser.transforms as viser_tf
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


# Box colors by class
BOX_COLORS = {
    "vehicle": (0, 255, 0),      # Green
    "pedestrian": (255, 0, 0),   # Red
    "cyclist": (255, 165, 0),    # Orange
}

# Camera colors
CAMERA_COLORS = {
    "1": (255, 0, 0),    # Front: red
    "2": (0, 255, 0),    # Front-left: green
    "3": (0, 0, 255),    # Front-right: blue
    "4": (255, 255, 0),  # Side-left: yellow
    "5": (255, 0, 255),  # Side-right: magenta
}


def get_box_corners_3d(center, size, heading):
    """
    Compute 8 corners of a 3D bounding box in vehicle frame.

    Args:
        center: [x, y, z] box center
        size: [length, width, height]
        heading: yaw angle (rotation around z-axis)

    Returns:
        corners: [8, 3] array of corner coordinates
    """
    l, w, h = size
    x, y, z = center

    corners_local = np.array([
        [l / 2, w / 2, h / 2],
        [l / 2, -w / 2, h / 2],
        [-l / 2, -w / 2, h / 2],
        [-l / 2, w / 2, h / 2],
        [l / 2, w / 2, -h / 2],
        [l / 2, -w / 2, -h / 2],
        [-l / 2, -w / 2, -h / 2],
        [-l / 2, w / 2, -h / 2],
    ])

    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    rot = np.array([
        [cos_h, -sin_h, 0],
        [sin_h, cos_h, 0],
        [0, 0, 1]
    ])

    corners = (rot @ corners_local.T).T + np.array([x, y, z])
    return corners


def create_box_lines(corners: np.ndarray) -> np.ndarray:
    """
    Create line segments for a 3D bounding box from its 8 corners.

    Returns:
        lines: [12, 2, 3] array of line segments (12 edges, 2 endpoints each)
    """
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Bottom face
        (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
    ]

    lines = []
    for i, j in edges:
        lines.append([corners[i], corners[j]])

    return np.array(lines)


def load_track_data(data_dir: str, scene_name: str) -> Dict:
    """
    Load track data for a scene.

    Returns:
        Dictionary containing:
            - frame_boxes: {frame_id -> [boxes]}
            - camera_vis: {track_id -> {frame_id -> [cam_ids]}}
            - track_ids: {original_id -> track_id}
    """
    import json

    track_dir = os.path.join(data_dir, scene_name, "track")
    result = {
        'frame_boxes': {},
        'camera_vis': {},
        'track_ids': {},
    }

    track_info_path = os.path.join(track_dir, "track_info.txt")
    if not os.path.exists(track_info_path):
        return result

    # Parse track_info.txt
    frame_boxes = {}
    with open(track_info_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 13:
                continue
            frame_id = int(parts[0])
            box = {
                'track_id': int(parts[1]),
                'class': parts[2],
                'height': float(parts[4]),
                'width': float(parts[5]),
                'length': float(parts[6]),
                'center_x': float(parts[7]),
                'center_y': float(parts[8]),
                'center_z': float(parts[9]),
                'heading': float(parts[10]),
                'speed_x': float(parts[11]),
                'speed_y': float(parts[12]),
            }
            if frame_id not in frame_boxes:
                frame_boxes[frame_id] = []
            frame_boxes[frame_id].append(box)

    result['frame_boxes'] = frame_boxes

    # Load camera visibility
    vis_path = os.path.join(track_dir, "track_camera_vis.json")
    if os.path.exists(vis_path):
        with open(vis_path, 'r') as f:
            result['camera_vis'] = json.load(f)

    # Load track IDs
    ids_path = os.path.join(track_dir, "track_ids.json")
    if os.path.exists(ids_path):
        with open(ids_path, 'r') as f:
            result['track_ids'] = json.load(f)

    return result


def load_frame_data_multi_camera(
    data_dir: str,
    scene_name: str,
    frame_id: int,
    camera_ids: List[str] = ["1", "2", "3"]
) -> Dict:
    """
    Load point cloud data for a specific frame from multiple cameras.
    """
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2
    from PIL import Image

    scene_dir = os.path.join(data_dir, scene_name)

    all_points = []
    all_colors = []
    cameras = {}

    frame_id_str = f"{frame_id:05d}"

    for camera_id in camera_ids:
        impath = f"{frame_id_str}_{camera_id}"

        img_path = os.path.join(scene_dir, impath + ".jpg")
        if not os.path.exists(img_path):
            continue

        depth_path = os.path.join(scene_dir, impath + ".exr")
        if not os.path.exists(depth_path):
            continue

        cam_path = os.path.join(scene_dir, impath + ".npz")
        if not os.path.exists(cam_path):
            continue

        image = np.array(Image.open(img_path))
        depthmap = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        camera_params = np.load(cam_path)
        intrinsics = camera_params["intrinsics"]
        cam2world = camera_params["cam2world"]

        H, W = depthmap.shape[:2]
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        ones = np.ones_like(x)
        pixel_coords = np.stack([x, y, ones], axis=-1).astype(np.float32)

        K_inv = np.linalg.inv(intrinsics)
        cam_points = np.einsum('ij,hwj->hwi', K_inv, pixel_coords)
        cam_points = cam_points * depthmap[:, :, np.newaxis]

        cam_points_homo = np.concatenate([cam_points, np.ones((H, W, 1))], axis=-1)
        world_points = np.einsum('ij,hwj->hwi', cam2world, cam_points_homo)[:, :, :3]

        valid_mask = depthmap > 0.01
        points = world_points[valid_mask]
        colors = image[valid_mask]

        all_points.append(points)
        all_colors.append(colors)

        cameras[camera_id] = {
            'cam2world': cam2world,
            'intrinsics': intrinsics,
            'image': image,
        }

    if all_points:
        combined_points = np.concatenate(all_points, axis=0)
        combined_colors = np.concatenate(all_colors, axis=0)
    else:
        combined_points = np.zeros((0, 3), dtype=np.float32)
        combined_colors = np.zeros((0, 3), dtype=np.uint8)

    return {
        'points': combined_points,
        'colors': combined_colors,
        'cameras': cameras,
    }


def compute_scene_center(
    data_dir: str,
    scene_name: str,
    frame_ids: List[int],
    camera_ids: List[str] = ["1", "2", "3"],
    sample_frames: int = 10
) -> np.ndarray:
    """
    Compute a global scene center from sampled frames.
    """
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2

    scene_dir = os.path.join(data_dir, scene_name)

    if len(frame_ids) <= sample_frames:
        sampled_ids = frame_ids
    else:
        indices = np.linspace(0, len(frame_ids) - 1, sample_frames, dtype=int)
        sampled_ids = [frame_ids[i] for i in indices]

    all_points = []

    print(f"Computing scene center from {len(sampled_ids)} sampled frames...")
    for frame_id in tqdm(sampled_ids, desc="Sampling frames"):
        frame_id_str = f"{frame_id:05d}"
        for camera_id in camera_ids:
            impath = f"{frame_id_str}_{camera_id}"

            depth_path = os.path.join(scene_dir, impath + ".exr")
            if not os.path.exists(depth_path):
                continue

            cam_path = os.path.join(scene_dir, impath + ".npz")
            if not os.path.exists(cam_path):
                continue

            depthmap = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            camera_params = np.load(cam_path)
            intrinsics = camera_params["intrinsics"]
            cam2world = camera_params["cam2world"]

            H, W = depthmap.shape[:2]
            y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            ones = np.ones_like(x)
            pixel_coords = np.stack([x, y, ones], axis=-1).astype(np.float32)

            K_inv = np.linalg.inv(intrinsics)
            cam_points = np.einsum('ij,hwj->hwi', K_inv, pixel_coords)
            cam_points = cam_points * depthmap[:, :, np.newaxis]

            cam_points_homo = np.concatenate([cam_points, np.ones((H, W, 1))], axis=-1)
            world_points = np.einsum('ij,hwj->hwi', cam2world, cam_points_homo)[:, :, :3]

            valid_mask = depthmap > 0.01
            points = world_points[valid_mask]

            if len(points) > 1000:
                indices = np.random.choice(len(points), 1000, replace=False)
                points = points[indices]

            all_points.append(points)

    if all_points:
        all_points = np.concatenate(all_points, axis=0)
        scene_center = np.mean(all_points, axis=0)
    else:
        scene_center = np.zeros(3)

    print(f"Scene center: {scene_center}")
    return scene_center


def get_boxes_for_frame(
    track_data: Dict,
    frame_id: int,
    camera_ids: List[str] = None,
    filter_by_visibility: bool = True
) -> List[Dict]:
    """
    Get boxes for a frame, optionally filtered by camera visibility.

    Args:
        track_data: Track data dictionary
        frame_id: Frame ID (int)
        camera_ids: List of camera IDs to filter by (if filter_by_visibility=True)
        filter_by_visibility: Whether to filter boxes by camera visibility

    Returns:
        List of box dictionaries with computed corners
    """
    boxes = track_data['frame_boxes'].get(frame_id, [])
    camera_vis = track_data['camera_vis']

    result = []
    for box in boxes:
        # Filter by camera visibility
        if filter_by_visibility and camera_ids:
            track_id = str(box['track_id'])
            if track_id in camera_vis:
                frame_vis = camera_vis[track_id].get(str(frame_id), [])
                # Check if visible in any of the requested cameras
                if not any(cam_id in frame_vis for cam_id in camera_ids):
                    continue

        # Build box with corners
        center = [box['center_x'], box['center_y'], box['center_z']]
        size = [box['length'], box['width'], box['height']]
        corners = get_box_corners_3d(center, size, box['heading'])

        result.append({
            'track_id': box['track_id'],
            'class': box['class'],
            'center': center,
            'size': size,
            'heading': box['heading'],
            'speed': [box['speed_x'], box['speed_y']],
            'corners': corners,
        })

    return result


def viser_box_visualization(
    data_dir: str,
    scene_name: str,
    port: int = 8080,
    camera_ids: List[str] = ["1", "2", "3"],
    filter_by_visibility: bool = True,
):
    """
    Interactive 3D visualization of bounding boxes with multi-camera support.
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Load track data
    track_data = load_track_data(data_dir, scene_name)
    frame_ids = sorted(track_data['frame_boxes'].keys())

    if not frame_ids:
        # Fallback to scanning directory for frames
        scene_dir = os.path.join(data_dir, scene_name)
        frame_ids = sorted(set(
            int(f.split("_")[0]) for f in os.listdir(scene_dir)
            if f.endswith(".jpg")
        ))

    if not frame_ids:
        print(f"No frames found in {scene_name}")
        return

    print(f"Found {len(frame_ids)} frames, using cameras: {camera_ids}")
    print(f"Track data: {len(track_data['frame_boxes'])} frames with boxes")

    # Compute global scene center
    scene_center = compute_scene_center(data_dir, scene_name, frame_ids, camera_ids)

    # State variables
    current_frame_idx = [0]
    point_cloud_handle = [None]
    box_handles = []
    camera_handles = []
    is_playing = [False]
    play_thread = [None]

    # GUI controls
    gui_frame_slider = server.gui.add_slider(
        "Frame", min=0, max=len(frame_ids) - 1, step=1, initial_value=0
    )
    gui_play_button = server.gui.add_button("▶ Play")
    gui_play_fps = server.gui.add_slider(
        "Playback FPS", min=1, max=30, step=1, initial_value=10
    )
    gui_show_vehicles = server.gui.add_checkbox("Show Vehicles", initial_value=True)
    gui_show_pedestrians = server.gui.add_checkbox("Show Pedestrians", initial_value=True)
    gui_show_cyclists = server.gui.add_checkbox("Show Cyclists", initial_value=True)
    gui_filter_visibility = server.gui.add_checkbox("Filter by Camera Visibility", initial_value=filter_by_visibility)
    gui_show_points = server.gui.add_checkbox("Show Point Cloud", initial_value=True)
    gui_show_cameras = server.gui.add_checkbox("Show Cameras", initial_value=True)
    gui_point_size = server.gui.add_slider(
        "Point Size", min=0.001, max=0.05, step=0.001, initial_value=0.005
    )
    gui_box_line_width = server.gui.add_slider(
        "Box Line Width", min=1.0, max=10.0, step=0.5, initial_value=3.0
    )

    gui_info = server.gui.add_text("Frame Info", initial_value="")

    def clear_boxes():
        for handle in box_handles:
            handle.remove()
        box_handles.clear()

    def clear_cameras():
        for handle in camera_handles:
            handle.remove()
        camera_handles.clear()

    def update_visualization():
        frame_id = frame_ids[current_frame_idx[0]]

        try:
            data = load_frame_data_multi_camera(data_dir, scene_name, frame_id, camera_ids)
        except Exception as e:
            print(f"Error loading frame {frame_id}: {e}")
            gui_info.value = f"Error: {e}"
            return

        points = data['points']
        colors = data['colors']
        cameras = data['cameras']

        # Use global scene center
        points_centered = points - scene_center

        # Update point cloud
        if point_cloud_handle[0] is not None:
            point_cloud_handle[0].remove()
            point_cloud_handle[0] = None

        if gui_show_points.value and len(points) > 0:
            point_cloud_handle[0] = server.scene.add_point_cloud(
                name="point_cloud",
                points=points_centered.astype(np.float32),
                colors=colors.astype(np.uint8),
                point_size=gui_point_size.value,
                point_shape="circle",
            )

        # Update camera frustums
        clear_cameras()

        if gui_show_cameras.value:
            for cam_id, cam_data in cameras.items():
                cam2world = cam_data['cam2world']
                image = cam_data['image']

                cam_pos = cam2world[:3, 3] - scene_center
                c2w_centered = cam2world.copy()
                c2w_centered[:3, 3] = cam_pos
                T_world_camera = viser_tf.SE3.from_matrix(c2w_centered[:3, :])

                handle = server.scene.add_camera_frustum(
                    name=f"camera_{cam_id}",
                    fov=np.pi / 3,
                    aspect=image.shape[1] / image.shape[0],
                    scale=0.3,
                    wxyz=T_world_camera.rotation().wxyz,
                    position=T_world_camera.translation(),
                    image=image,
                    color=CAMERA_COLORS.get(cam_id, (128, 128, 128)),
                )
                camera_handles.append(handle)

        # Get boxes
        boxes = get_boxes_for_frame(
            track_data, frame_id, camera_ids,
            filter_by_visibility=gui_filter_visibility.value
        )

        # Update boxes
        clear_boxes()

        num_vehicles = 0
        num_pedestrians = 0
        num_cyclists = 0

        for i, box in enumerate(boxes):
            obj_class = box['class']

            # Filter by class
            if obj_class == "vehicle" and not gui_show_vehicles.value:
                continue
            if obj_class == "pedestrian" and not gui_show_pedestrians.value:
                continue
            if obj_class == "cyclist" and not gui_show_cyclists.value:
                continue

            if obj_class == "vehicle":
                num_vehicles += 1
            elif obj_class == "pedestrian":
                num_pedestrians += 1
            elif obj_class == "cyclist":
                num_cyclists += 1

            corners = box['corners'] - scene_center
            color = BOX_COLORS.get(obj_class, (128, 128, 128))

            lines = create_box_lines(corners).astype(np.float32)
            line_colors = np.tile(np.array(color, dtype=np.uint8), (12, 2, 1))

            handle = server.scene.add_line_segments(
                name=f"box_{i}",
                points=lines,
                colors=line_colors,
                line_width=gui_box_line_width.value,
            )
            box_handles.append(handle)

        gui_info.value = f"Frame: {frame_id} | V: {num_vehicles} P: {num_pedestrians} C: {num_cyclists} | Points: {len(points)} | Cams: {len(cameras)}"

    # Event handlers
    @gui_frame_slider.on_update
    def _(_):
        current_frame_idx[0] = int(gui_frame_slider.value)
        update_visualization()

    @gui_show_vehicles.on_update
    def _(_):
        update_visualization()

    @gui_show_pedestrians.on_update
    def _(_):
        update_visualization()

    @gui_show_cyclists.on_update
    def _(_):
        update_visualization()

    @gui_filter_visibility.on_update
    def _(_):
        update_visualization()

    @gui_show_points.on_update
    def _(_):
        update_visualization()

    @gui_show_cameras.on_update
    def _(_):
        update_visualization()

    @gui_point_size.on_update
    def _(_):
        if point_cloud_handle[0] is not None:
            point_cloud_handle[0].point_size = gui_point_size.value

    @gui_box_line_width.on_update
    def _(_):
        update_visualization()

    def playback_loop():
        while is_playing[0]:
            time.sleep(1.0 / gui_play_fps.value)
            if not is_playing[0]:
                break
            next_idx = (current_frame_idx[0] + 1) % len(frame_ids)
            current_frame_idx[0] = next_idx
            gui_frame_slider.value = next_idx
            update_visualization()

    @gui_play_button.on_click
    def _(_):
        import threading
        if is_playing[0]:
            is_playing[0] = False
            gui_play_button.name = "▶ Play"
        else:
            is_playing[0] = True
            gui_play_button.name = "⏸ Pause"
            play_thread[0] = threading.Thread(target=playback_loop, daemon=True)
            play_thread[0].start()

    # Initial visualization
    update_visualization()

    print("Viser server started!")
    print(f"Open http://localhost:{port} in your browser")
    print(f"Scene: {scene_name}")
    print(f"Frames: {len(frame_ids)}")
    print(f"Cameras: {camera_ids}")

    while True:
        time.sleep(0.01)


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D bounding boxes with viser")
    parser.add_argument("--data_dir", type=str, default="data/waymo/train_full_test",
                        help="Path to preprocessed Waymo dataset")
    parser.add_argument("--scene_idx", type=int, default=None,
                        help="Scene index to visualize")
    parser.add_argument("--scene_name", type=str, default=None,
                        help="Scene name to visualize (overrides scene_idx)")
    parser.add_argument("--camera_ids", type=str, default="1,2,3",
                        help="Comma-separated camera IDs to use (default: 1,2,3)")
    parser.add_argument("--no_filter", action="store_true",
                        help="Disable camera visibility filtering for boxes")
    parser.add_argument("--port", type=int, default=8080,
                        help="Viser server port")
    args = parser.parse_args()

    camera_ids = [c.strip() for c in args.camera_ids.split(",")]

    scenes = sorted([
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    ])

    if not scenes:
        print(f"No scenes found in {args.data_dir}")
        return

    if args.scene_name:
        if args.scene_name not in scenes:
            print(f"Scene {args.scene_name} not found. Available scenes:")
            for i, s in enumerate(scenes[:10]):
                print(f"  {i}: {s}")
            if len(scenes) > 10:
                print(f"  ... and {len(scenes) - 10} more")
            return
        scene_name = args.scene_name
    elif args.scene_idx is not None:
        if args.scene_idx >= len(scenes):
            print(f"Scene index {args.scene_idx} out of range. Available: 0-{len(scenes)-1}")
            return
        scene_name = scenes[args.scene_idx]
    else:
        scene_name = scenes[0]
        print(f"No scene specified, using first scene: {scene_name}")

    track_dir = os.path.join(args.data_dir, scene_name, "track")
    if not os.path.exists(os.path.join(track_dir, "track_info.txt")):
        print(f"Warning: track/track_info.txt not found for {scene_name}")
        print("Run preprocess_waymo_box.py first to extract bounding boxes.")

    viser_box_visualization(
        data_dir=args.data_dir,
        scene_name=scene_name,
        port=args.port,
        camera_ids=camera_ids,
        filter_by_visibility=not args.no_filter,
    )


if __name__ == "__main__":
    main()
