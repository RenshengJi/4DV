#!/usr/bin/env python3
"""
Patch script to extract 3D bounding boxes from Waymo dataset.
This is a patch-style preprocessing script to add 3D box data to an existing
preprocessed Waymo dataset.

Data structure follows street_gaussians format:
- track_info.txt: Per-frame box info (frame_id, track_id, class, dimensions, pose, speed)
- track_camera_vis.json: Which cameras can see each object at each frame
- track_ids.json: Mapping from original object ID to sequential track ID

Only extracts: vehicle, pedestrian, cyclist

Usage:
    python preprocess_waymo_box.py --waymo_dir /path/to/raw/tfrecords --output_dir /path/to/preprocessed
"""
import os
import os.path as osp
import json
import argparse
import math
from tqdm import tqdm
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()


# Waymo label types
LABEL_TYPE_VEHICLE = 1
LABEL_TYPE_PEDESTRIAN = 2
LABEL_TYPE_CYCLIST = 4

LABEL_TYPE_NAMES = {
    LABEL_TYPE_VEHICLE: "vehicle",
    LABEL_TYPE_PEDESTRIAN: "pedestrian",
    LABEL_TYPE_CYCLIST: "cyclist",
}

# Camera name mapping (waymo camera_name to our camera_id)
# Waymo: 1=FRONT, 2=FRONT_LEFT, 3=FRONT_RIGHT, 4=SIDE_LEFT, 5=SIDE_RIGHT
CAMERA_NAME_TO_ID = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}


def get_parser():
    parser = argparse.ArgumentParser(description="Extract 3D bounding boxes from Waymo dataset")
    parser.add_argument("--waymo_dir", default="/mnt/raw-datasets/waymo/raw/train",
                        help="Path to raw Waymo tfrecord files")
    parser.add_argument("--output_dir", default="data/waymo/train_full_test",
                        help="Path to existing preprocessed dataset")
    parser.add_argument("--workers", type=int, default=80, help="Number of workers")
    parser.add_argument("--start", type=int, default=0, help="Start index of sequences")
    parser.add_argument("--end", type=int, default=None, help="End index of sequences")
    return parser


def _list_sequences(db_root):
    """List all tfrecord files in the directory."""
    print(f">> Looking for sequences in {db_root}")
    res = sorted(f for f in os.listdir(db_root) if f.endswith(".tfrecord"))
    print(f"    found {len(res)} sequences")
    return res


def bbox_to_corner3d(bbox):
    """
    Convert bbox bounds to 8 corner points.
    bbox: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    Returns: [8, 3] corner coordinates
    """
    min_x, min_y, min_z = bbox[0]
    max_x, max_y, max_z = bbox[1]

    corner3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corner3d


def get_box_corners_in_vehicle_frame(length, width, height, tx, ty, tz, heading):
    """
    Get 8 corners of a 3D bounding box in vehicle frame.

    Args:
        length, width, height: Box dimensions
        tx, ty, tz: Box center position
        heading: Yaw angle (rotation around z-axis)

    Returns:
        corners: [8, 3] array of corner coordinates in vehicle frame
    """
    # Box bounds in local frame (centered at origin)
    bbox = np.array([[-length, -width, -height], [length, width, height]]) * 0.5
    corners_local = bbox_to_corner3d(bbox)

    # Rotation matrix for yaw
    c = math.cos(heading)
    s = math.sin(heading)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # Transform to vehicle frame
    corners_vehicle = (rot @ corners_local.T).T + np.array([tx, ty, tz])
    return corners_vehicle


def transform_box_to_world(tx, ty, tz, heading, vehicle_to_world):
    """
    Transform box center and heading from vehicle frame to world frame.

    Args:
        tx, ty, tz: Box center in vehicle frame
        heading: Yaw angle in vehicle frame
        vehicle_to_world: 4x4 transformation matrix

    Returns:
        world_tx, world_ty, world_tz, world_heading
    """
    # Transform center
    center_vehicle = np.array([tx, ty, tz, 1.0])
    center_world = vehicle_to_world @ center_vehicle
    world_tx, world_ty, world_tz = center_world[:3]

    # Transform heading: extract yaw from vehicle_to_world rotation
    # Vehicle forward is +x, so we look at where +x goes in world
    R = vehicle_to_world[:3, :3]
    # The heading in world = vehicle heading + vehicle yaw in world
    # Vehicle yaw in world: atan2(R[1,0], R[0,0])
    vehicle_yaw_world = math.atan2(R[1, 0], R[0, 0])
    world_heading = heading + vehicle_yaw_world

    return world_tx, world_ty, world_tz, world_heading


def project_box_to_camera(corners_vehicle, camera_calibration):
    """
    Project 3D box corners to camera image plane.

    Args:
        corners_vehicle: [8, 3] corners in vehicle frame
        camera_calibration: Waymo camera calibration

    Returns:
        corners_2d: [8, 2] projected 2D coordinates
        valid: [8] boolean mask for valid projections (in front of camera and in image)
    """
    # Waymo camera extrinsic: camera to vehicle transform
    # We need vehicle to camera (world2cam)
    cam_to_vehicle = np.array(camera_calibration.extrinsic.transform).reshape(4, 4)

    # Apply opencv2camera transform: Waymo uses [forward, left, up], OpenCV uses [right, down, forward]
    opencv2camera = np.array([
        [0., 0., 1., 0.],
        [-1., 0., 0., 0.],
        [0., -1., 0., 0.],
        [0., 0., 0., 1.]
    ])
    cam_to_vehicle = cam_to_vehicle @ opencv2camera

    # Invert to get vehicle to camera
    vehicle_to_cam = np.linalg.inv(cam_to_vehicle)

    # Camera intrinsic
    fx, fy, cx, cy = camera_calibration.intrinsic[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    width = camera_calibration.width
    height = camera_calibration.height

    # Transform corners to camera frame using the standard projection formula
    # xyz_cam = xyz @ R.T + t.T
    corners_cam = corners_vehicle @ vehicle_to_cam[:3, :3].T + vehicle_to_cam[:3, 3:].T

    # Check if points are in front of camera (z > 0)
    valid_depth = corners_cam[:, 2] > 0.1

    # Project to image plane: xyz_pixel = xyz_cam @ K.T
    corners_2d_homo = corners_cam @ K.T
    # Avoid division by zero
    z_safe = np.maximum(corners_2d_homo[:, 2:3], 0.1)
    corners_2d = corners_2d_homo[:, :2] / z_safe

    # Check if points are within image bounds
    valid_x = (corners_2d[:, 0] >= 0) & (corners_2d[:, 0] < width)
    valid_y = (corners_2d[:, 1] >= 0) & (corners_2d[:, 1] < height)
    valid_bounds = valid_x & valid_y

    valid = valid_depth & valid_bounds

    return corners_2d, valid


def process_one_sequence(db_root, output_dir, seq_name):
    """
    Process one sequence and extract 3D bounding boxes.

    Output files:
        - track/track_info.txt: Box info per frame
        - track/track_camera_vis.json: Camera visibility per object per frame
        - track/track_ids.json: Object ID mapping
    """
    from waymo_open_dataset import dataset_pb2 as open_dataset

    out_dir = osp.join(output_dir, seq_name)

    if not osp.isdir(out_dir):
        print(f"Warning: Output directory {out_dir} does not exist, skipping...")
        return

    track_dir = osp.join(out_dir, "track")

    # Check if already processed
    if osp.isfile(osp.join(track_dir, "track_info.txt")):
        print(f"Track data already extracted for {seq_name}, skipping...")
        return

    tfrecord_path = osp.join(db_root, seq_name)
    if not osp.isfile(tfrecord_path):
        print(f"Warning: TFRecord {tfrecord_path} not found, skipping...")
        return

    os.makedirs(track_dir, exist_ok=True)

    print(f">> Processing {seq_name}")
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="")

    # Output file
    track_info_path = osp.join(track_dir, "track_info.txt")
    track_info_file = open(track_info_path, 'w')

    # Header
    header = "frame_id track_id object_class alpha box_height box_width box_length "
    header += "box_center_x box_center_y box_center_z box_heading speed_x speed_y\n"
    track_info_file.write(header)

    # Track visibility and ID mapping
    bbox_visible_dict = {}  # track_id -> {frame_id -> [camera_ids]}
    object_ids = {}  # original_id -> sequential_track_id

    for frame_idx, data in enumerate(tqdm(dataset, leave=False, desc=seq_name)):
        frame = open_dataset.Frame()
        frame.ParseFromString(data.numpy())

        # Get vehicle to world transform from frame pose
        vehicle_to_world = np.array(frame.pose.transform).reshape(4, 4)

        # Get camera calibrations for visibility check
        camera_calibrations = {
            cam.name: cam for cam in frame.context.camera_calibrations
        }

        for label in frame.laser_labels:
            # Only process vehicle, pedestrian, cyclist
            if label.type not in LABEL_TYPE_NAMES:
                continue

            box = label.box
            meta = label.metadata

            # Box dimensions
            length = box.length
            width = box.width
            height = box.height

            # Box pose in vehicle frame
            tx_vehicle = box.center_x
            ty_vehicle = box.center_y
            tz_vehicle = box.center_z
            heading_vehicle = box.heading

            # Transform to world frame
            tx, ty, tz, heading = transform_box_to_world(
                tx_vehicle, ty_vehicle, tz_vehicle, heading_vehicle, vehicle_to_world
            )

            # Speed (keep in vehicle frame as it's relative velocity)
            speed_x = meta.speed_x
            speed_y = meta.speed_y

            # Assign track ID
            if label.id not in object_ids:
                object_ids[label.id] = len(object_ids)
            track_id = object_ids[label.id]

            # Initialize visibility dict for this track
            if track_id not in bbox_visible_dict:
                bbox_visible_dict[track_id] = {}
            bbox_visible_dict[track_id][frame_idx] = []

            # Get box corners in vehicle frame (for camera visibility check)
            corners_vehicle = get_box_corners_in_vehicle_frame(
                length, width, height, tx_vehicle, ty_vehicle, tz_vehicle, heading_vehicle
            )

            # Check visibility in each camera
            for cam_name, cam_calib in camera_calibrations.items():
                corners_2d, valid = project_box_to_camera(corners_vehicle, cam_calib)

                # If any corner is visible, consider the object visible
                if valid.any():
                    cam_id = CAMERA_NAME_TO_ID.get(cam_name)
                    if cam_id:
                        bbox_visible_dict[track_id][frame_idx].append(cam_id)

            bbox_visible_dict[track_id][frame_idx] = sorted(
                bbox_visible_dict[track_id][frame_idx]
            )

            # Object class name
            obj_class = LABEL_TYPE_NAMES[label.type]

            # Alpha (not used, set to -10 following street_gaussians)
            alpha = -10

            # Write track info line
            line = f"{frame_idx} {track_id} {obj_class} {alpha} "
            line += f"{height} {width} {length} "
            line += f"{tx} {ty} {tz} {heading} "
            line += f"{speed_x} {speed_y}\n"
            track_info_file.write(line)

    track_info_file.close()

    # Save camera visibility
    bbox_visible_path = osp.join(track_dir, "track_camera_vis.json")
    with open(bbox_visible_path, 'w') as f:
        json.dump(bbox_visible_dict, f, indent=1)

    # Save object ID mapping
    object_ids_path = osp.join(track_dir, "track_ids.json")
    with open(object_ids_path, 'w') as f:
        json.dump(object_ids, f, indent=2)

    print(f"Saved track data to {track_dir}")
    print(f"  - {len(object_ids)} unique objects")
    print(f"  - {frame_idx + 1} frames")


def main(waymo_dir, output_dir, workers=1, start=0, end=None):
    """Main entry point for box extraction."""
    sequences = _list_sequences(waymo_dir)

    if end is None:
        end = len(sequences)
    sequences = sequences[start:end]

    print(f">> Processing sequences {start} to {end} ({len(sequences)} sequences)")

    if workers == 1:
        for seq in tqdm(sequences, desc="Extracting boxes"):
            try:
                with tf.device("/CPU:0"):
                    process_one_sequence(waymo_dir, output_dir, seq)
            except Exception as e:
                print(f"Error processing {seq}: {e}")
                import traceback
                traceback.print_exc()
    else:
        import sys
        sys.path.append(osp.join(osp.dirname(__file__), '..'))
        from utils.parallel import parallel_processes as parallel_map

        args = [(waymo_dir, output_dir, seq) for seq in sequences]
        parallel_map(process_one_sequence, args, star_args=True, workers=workers)

    print("Done extracting 3D bounding boxes!")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.waymo_dir, args.output_dir, workers=args.workers, start=args.start, end=args.end)
