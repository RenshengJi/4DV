#!/usr/bin/env python3
"""
Debug script to check labels in Waymo tfrecord file
"""
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2
import numpy as np

tfrecord_path = "/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/test/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord"

print(f"Reading: {tfrecord_path}")
print("=" * 80)

dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="")

total_frames = 0
total_labels = 0
labels_per_frame = []

for data in dataset:
    frame = dataset_pb2.Frame()
    frame.ParseFromString(bytearray(data.numpy()))

    frame_labels = 0
    labels_with_speed = 0
    labels_with_box = 0
    labels_with_lidar = 0

    for label in frame.laser_labels:
        frame_labels += 1

        box = label.box
        meta = label.metadata

        if box.ByteSize():
            labels_with_box += 1

        if label.num_top_lidar_points_in_box:
            labels_with_lidar += 1

        speed = np.linalg.norm([meta.speed_x, meta.speed_y])
        if speed > 0:
            labels_with_speed += 1

        # Check if would pass our NEW filter
        if box.ByteSize() and (label.num_top_lidar_points_in_box or label.num_lidar_points_in_box):
            total_labels += 1

    labels_per_frame.append(frame_labels)

    if total_frames < 3:
        print(f"\nFrame {total_frames}:")
        print(f"  Total labels: {frame_labels}")
        print(f"  Labels with box: {labels_with_box}")
        print(f"  Labels with lidar points: {labels_with_lidar}")
        print(f"  Labels with speed > 0: {labels_with_speed}")
        print(f"  Labels passing filter: {total_labels - sum(labels_per_frame[:-1])}")

    total_frames += 1

    if total_frames >= 10:  # Check first 10 frames
        break

print("\n" + "=" * 80)
print(f"Summary (first {total_frames} frames):")
print(f"  Total frames: {total_frames}")
print(f"  Total labels (raw): {sum(labels_per_frame)}")
print(f"  Total labels (filtered): {total_labels}")
print(f"  Average labels per frame: {sum(labels_per_frame) / total_frames:.2f}")

if total_labels == 0:
    print("\n⚠️  WARNING: No labels found that pass the filter!")
    print("   This could mean:")
    print("   1. The dataset doesn't have laser labels")
    print("   2. All labels are missing boxes or lidar points")
    print("   3. The filter conditions are too strict")
