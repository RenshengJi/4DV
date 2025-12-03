#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Preprocessing code for the WayMo Open dataset
# dataset at https://github.com/waymo-research/waymo-open-dataset
# 1) Accept the license
# 2) download all training/*.tfrecord files from Perception Dataset, version 1.4.2
# 3) put all .tfrecord files in '/path/to/waymo_dir'
# 4) install the waymo_open_dataset package with
#    `python3 -m pip install gcsfs waymo-open-dataset-tf-2-12-0==1.6.4`
# 5) execute this script as `python preprocess_waymo.py --waymo_dir /path/to/waymo_dir`
# --------------------------------------------------------
import sys
import os
import os.path as osp
import shutil
import json
from tqdm import tqdm
import PIL.Image
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import sys
# 添加vggt路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))
import cv2

import tensorflow.compat.v1 as tf
import gc

tf.enable_eager_execution()

import path_to_root  # noqa
from src.dust3r.utils.geometry import geotrf, inv
from src.dust3r.utils.image import imread_cv2
from src.dust3r.utils.parallel import parallel_processes as parallel_map
from datasets_preprocess.utils import cropping

# 导入SAM预处理模块
from sam_preprocessing import SAMPreprocessor


def project_vehicle_to_image(vehicle_pose, calibration, points):
    """Projects from vehicle coordinate system to image with global shutter.

    Arguments:
      vehicle_pose: Vehicle pose transform from vehicle into world coordinate system.
      calibration: Camera calibration details (including intrinsics/extrinsics).
      points: Points to project of shape [N, 3] in vehicle coordinate system.

    Returns:
      Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
    """
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops

    # Transform points from vehicle to world coordinate system
    pose_matrix = np.array(vehicle_pose.transform).reshape(4, 4)
    world_points = np.zeros_like(points)
    for i, point in enumerate(points):
        cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
        world_points[i] = (cx, cy, cz)

    # Populate camera image metadata
    extrinsic = tf.reshape(
        tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32), [4, 4]
    )
    intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
    metadata = tf.constant(
        [calibration.width, calibration.height, dataset_pb2.CameraCalibration.GLOBAL_SHUTTER],
        dtype=tf.int32,
    )
    camera_image_metadata = list(vehicle_pose.transform) + [0.0] * 10

    # Perform projection and return projected image coordinates (u, v, ok)
    return py_camera_model_ops.world_to_image(
        extrinsic, intrinsic, metadata, camera_image_metadata, world_points
    ).numpy()


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--waymo_dir", default="/mnt/raw-datasets/waymo/raw/train")
    parser.add_argument("--precomputed_pairs")
    parser.add_argument("--output_dir", default="data/waymo/train_full_test")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--enable_sam", action="store_true", help="Enable SAM mask generation")
    parser.add_argument("--sam_model_type", default="sam2", choices=["sam2", "sam"], help="SAM model type")
    parser.add_argument("--sam_device", default="cuda", help="Device for SAM model")
    parser.add_argument("--sam_config_file", help="SAM2 config file")
    parser.add_argument("--sam_ckpt_path", help="SAM model checkpoint path")
    parser.add_argument("--start", type=int, default=0, help="Start index of sequences to process (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index of sequences to process (exclusive)")
    return parser


def main(waymo_root, pairs_path, output_dir, workers=1, enable_sam=False, sam_model_type="sam2",
         sam_device="cuda", sam_config_file=None, sam_ckpt_path=None, start=0, end=None):
    extract_frames(waymo_root, output_dir, workers=workers, start=start, end=end)
    make_crops(output_dir, workers=args.workers, enable_sam=enable_sam, sam_model_type=sam_model_type,
               sam_device=sam_device, sam_config_file=sam_config_file, sam_ckpt_path=sam_ckpt_path,
               start=start, end=end)

    # # make sure all pairs are there
    # with np.load(pairs_path) as data:
    #     scenes = data["scenes"]
    #     frames = data["frames"]
    #     pairs = data["pairs"]  # (array of (scene_id, img1_id, img2_id)

    # for scene_id, im1_id, im2_id in pairs:
    #     for im_id in (im1_id, im2_id):
    #         path = osp.join(output_dir, scenes[scene_id], frames[im_id] + ".jpg")
    #         assert osp.isfile(
    #             path
    #         ), f"Missing a file at {path=}\nDid you download all .tfrecord files?"

    shutil.rmtree(osp.join(output_dir, "tmp"))
    print("Done! all data generated at", output_dir)


def _list_sequences(db_root):
    print(">> Looking for sequences in", db_root)
    res = sorted(f for f in os.listdir(db_root) if f.endswith(".tfrecord"))
    print(f"    found {len(res)} sequences")
    return res


def extract_frames(db_root, output_dir, workers=8, start=0, end=None):
    sequences = _list_sequences(db_root)
    # Select sequences based on start and end indices
    if end is None:
        end = len(sequences)
    sequences = sequences[start:end]
    print(f">> Processing sequences {start} to {end} ({len(sequences)} sequences)")
    output_dir = osp.join(output_dir, "tmp")
    print(">> outputing result to", output_dir)
    args = [(db_root, output_dir, seq) for seq in sequences]
    parallel_map(process_one_seq, args, star_args=True, workers=workers)


def process_one_seq(db_root, output_dir, seq):
    out_dir = osp.join(output_dir, seq)
    os.makedirs(out_dir, exist_ok=True)
    calib_path = osp.join(out_dir, "calib.json")
    if osp.isfile(calib_path):
        return

    try:
        with tf.device("/CPU:0"):
            # Use generator to process frames one by one, avoiding memory accumulation
            calib = None
            for f, (frame_name, views) in enumerate(extract_frames_one_seq_generator(osp.join(db_root, seq))):
                if calib is None:
                    # First yield returns calibration
                    if isinstance(frame_name, list):
                        calib = frame_name
                        continue

                for cam_idx, view in views.items():
                    img = PIL.Image.fromarray(view.pop("img"))
                    img.save(osp.join(out_dir, f"{f:05d}_{cam_idx}.jpg"))

                    # Convert complex data to JSON strings for safe npz storage
                    if 'labels' in view:
                        view['labels_json'] = json.dumps(view.pop('labels'))
                    if 'calibration' in view:
                        view['calibration_json'] = json.dumps(view.pop('calibration'))

                    np.savez(osp.join(out_dir, f"{f:05d}_{cam_idx}.npz"), **view)

                # Clear memory after each frame
                del views
                gc.collect()

    except RuntimeError:
        print(f"/!\\ Error with sequence {seq} /!\\", file=sys.stderr)
        return  # nothing is saved

    with open(calib_path, "w") as f:
        json.dump(calib, f)


def extract_frames_one_seq(filename):
    """Original function kept for compatibility - uses generator internally"""
    calib = None
    frames = []
    for item in extract_frames_one_seq_generator(filename):
        if calib is None and isinstance(item[0], list):
            calib = item[0]
        else:
            frames.append(item)
    return calib, frames


def extract_frames_one_seq_generator(filename):
    """Generator version that yields frames one by one to avoid memory accumulation"""

    from waymo_open_dataset import dataset_pb2 as open_dataset
    from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils

    def parse_range_image_flow_and_camera_projection(frame):
        range_images = {}
        camera_projections = {}
        range_image_top_pose = None
        for laser in frame.lasers:
            if (
                len(laser.ri_return1.range_image_flow_compressed) > 0
            ):  # pylint: disable=g-explicit-length-test
                range_image_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_flow_compressed, "ZLIB"
                )
                ri = open_dataset.MatrixFloat()
                ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
                range_images[laser.name] = [ri]

                if laser.name == open_dataset.LaserName.TOP:
                    range_image_top_pose_str_tensor = tf.io.decode_compressed(
                        laser.ri_return1.range_image_pose_compressed, "ZLIB"
                    )
                    range_image_top_pose = open_dataset.MatrixFloat()
                    range_image_top_pose.ParseFromString(
                        bytearray(range_image_top_pose_str_tensor.numpy())
                    )

                camera_projection_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.camera_projection_compressed, "ZLIB"
                )
                cp = open_dataset.MatrixInt32()
                cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
                camera_projections[laser.name] = [cp]
            if (
                len(laser.ri_return2.range_image_flow_compressed) > 0
            ):  # pylint: disable=g-explicit-length-test
                range_image_str_tensor = tf.io.decode_compressed(
                    laser.ri_return2.range_image_flow_compressed, "ZLIB"
                )
                ri = open_dataset.MatrixFloat()
                ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
                range_images[laser.name].append(ri)

                camera_projection_str_tensor = tf.io.decode_compressed(
                    laser.ri_return2.camera_projection_compressed, "ZLIB"
                )
                cp = open_dataset.MatrixInt32()
                cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
                camera_projections[laser.name].append(cp)
        return range_images, camera_projections, range_image_top_pose


    def compute_range_image_cartesian(
        range_image_polar,
        extrinsic,
        pixel_pose=None,
        frame_pose=None,
        dtype=tf.float32,
        scope=None,
    ):
        """Computes range image cartesian coordinates from polar ones.

        Args:
        range_image_polar: [B, H, W, 3] float tensor. Lidar range image in polar
            coordinate in sensor frame.
        extrinsic: [B, 4, 4] float tensor. Lidar extrinsic.
        pixel_pose: [B, H, W, 4, 4] float tensor. If not None, it sets pose for each
            range image pixel.
        frame_pose: [B, 4, 4] float tensor. This must be set when pixel_pose is set.
            It decides the vehicle frame at which the cartesian points are computed.
        dtype: float type to use internally. This is needed as extrinsic and
            inclination sometimes have higher resolution than range_image.
        scope: the name scope.

        Returns:
        range_image_cartesian: [B, H, W, 3] cartesian coordinates.
        """
        range_image_polar_dtype = range_image_polar.dtype
        range_image_polar = tf.cast(range_image_polar, dtype=dtype)
        extrinsic = tf.cast(extrinsic, dtype=dtype)
        if pixel_pose is not None:
            pixel_pose = tf.cast(pixel_pose, dtype=dtype)
        if frame_pose is not None:
            frame_pose = tf.cast(frame_pose, dtype=dtype)

        with tf.compat.v1.name_scope(
            scope,
            "ComputeRangeImageCartesian",
            [range_image_polar, extrinsic, pixel_pose, frame_pose],
        ):
            azimuth, inclination, range_image_range = tf.unstack(range_image_polar, axis=-1)

            cos_azimuth = tf.cos(azimuth)
            sin_azimuth = tf.sin(azimuth)
            cos_incl = tf.cos(inclination)
            sin_incl = tf.sin(inclination)

            # [B, H, W].
            x = cos_azimuth * cos_incl * range_image_range
            y = sin_azimuth * cos_incl * range_image_range
            z = sin_incl * range_image_range

            # [B, H, W, 3]
            range_image_points = tf.stack([x, y, z], -1)
            range_image_origins = tf.zeros_like(range_image_points)
            # [B, 3, 3]
            rotation = extrinsic[..., 0:3, 0:3]
            # translation [B, 1, 3]
            translation = tf.expand_dims(tf.expand_dims(extrinsic[..., 0:3, 3], 1), 1)

            # To vehicle frame.
            # [B, H, W, 3]
            range_image_points = tf.einsum("bkr,bijr->bijk", rotation, range_image_points) + translation
            range_image_origins = (
                tf.einsum("bkr,bijr->bijk", rotation, range_image_origins) + translation
            )
            if pixel_pose is not None:
                # To global frame.
                # [B, H, W, 3, 3]
                pixel_pose_rotation = pixel_pose[..., 0:3, 0:3]
                # [B, H, W, 3]
                pixel_pose_translation = pixel_pose[..., 0:3, 3]
                # [B, H, W, 3]
                range_image_points = (
                    tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_points)
                    + pixel_pose_translation
                )
                range_image_origins = (
                    tf.einsum("bhwij,bhwj->bhwi", pixel_pose_rotation, range_image_origins)
                    + pixel_pose_translation
                )

                if frame_pose is None:
                    raise ValueError("frame_pose must be set when pixel_pose is set.")
                # To vehicle frame corresponding to the given frame_pose
                # [B, 4, 4]
                world_to_vehicle = tf.linalg.inv(frame_pose)
                world_to_vehicle_rotation = world_to_vehicle[:, 0:3, 0:3]
                world_to_vehicle_translation = world_to_vehicle[:, 0:3, 3]
                # [B, H, W, 3]
                range_image_points = (
                    tf.einsum("bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_points)
                    + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
                )
                range_image_origins = (
                    tf.einsum("bij,bhwj->bhwi", world_to_vehicle_rotation, range_image_origins)
                    + world_to_vehicle_translation[:, tf.newaxis, tf.newaxis, :]
                )

            range_image_points = tf.cast(range_image_points, dtype=range_image_polar_dtype)
            range_image_origins = tf.cast(range_image_origins, dtype=range_image_polar_dtype)
            return range_image_points, range_image_origins


    def extract_point_cloud_from_range_image(
        range_image,
        extrinsic,
        inclination,
        pixel_pose=None,
        frame_pose=None,
        dtype=tf.float32,
        scope=None,
    ):
        """Extracts point cloud from range image.

        Args:
        range_image: [B, H, W] tensor. Lidar range images.
        extrinsic: [B, 4, 4] tensor. Lidar extrinsic.
        inclination: [B, H] tensor. Inclination for each row of the range image.
            0-th entry corresponds to the 0-th row of the range image.
        pixel_pose: [B, H, W, 4, 4] tensor. If not None, it sets pose for each range
            image pixel.
        frame_pose: [B, 4, 4] tensor. This must be set when pixel_pose is set. It
            decides the vehicle frame at which the cartesian points are computed.
        dtype: float type to use internally. This is needed as extrinsic and
            inclination sometimes have higher resolution than range_image.
        scope: the name scope.

        Returns:
        range_image_points: [B, H, W, 3] with {x, y, z} as inner dims in vehicle frame.
        range_image_origins: [B, H, W, 3] with {x, y, z}, the origin of the range image
        """
        with tf.compat.v1.name_scope(
            scope,
            "ExtractPointCloudFromRangeImage",
            [range_image, extrinsic, inclination, pixel_pose, frame_pose],
        ):
            range_image_polar = range_image_utils.compute_range_image_polar(
                range_image, extrinsic, inclination, dtype=dtype
            )
            (
                range_image_points_cartesian,
                range_image_origins_cartesian,
            ) = compute_range_image_cartesian(
                range_image_polar,
                extrinsic,
                pixel_pose=pixel_pose,
                frame_pose=frame_pose,
                dtype=dtype,
            )
            return range_image_origins_cartesian, range_image_points_cartesian


    def convert_range_image_to_point_cloud_flow(
        frame,
        range_images,
        range_images_flow,
        camera_projections,
        range_image_top_pose,
        ri_index=0,
    ):
        """
        Modified from the codes of Waymo Open Dataset.
        Convert range images to point cloud.
        Convert range images flow to scene flow.
        Args:
            frame: open dataset frame
            range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
            range_imaages_flow: A dict similar to range_images.
            camera_projections: A dict of {laser_name,
                [camera_projection_from_first_return, camera_projection_from_second_return]}.
            range_image_top_pose: range image pixel pose for top lidar.
            ri_index: 0 for the first return, 1 for the second return.

        Returns:
            points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
            points_flow: {[N, 3]} list of scene flow vector of each point.
            cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
        """
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        origins, points, cp_points = [], [], []
        points_intensity = []
        points_elongation = []
        points_flow = []
        laser_ids = []

        frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
        )
        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
            range_image_top_pose_tensor[..., 0],
            range_image_top_pose_tensor[..., 1],
            range_image_top_pose_tensor[..., 2],
        )
        range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation, range_image_top_pose_tensor_translation
        )   
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            range_image_flow = range_images_flow[c.name][ri_index]
            if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0],
                )
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims
            )
            range_image_flow_tensor = tf.reshape(
                tf.convert_to_tensor(range_image_flow.data), range_image_flow.shape.dims
            )
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == open_dataset.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0
            range_image_intensity = range_image_tensor[..., 1]
            range_image_elongation = range_image_tensor[..., 2]

            flow_x = range_image_flow_tensor[..., 0]
            flow_y = range_image_flow_tensor[..., 1]
            flow_z = range_image_flow_tensor[..., 2]
            flow_class = range_image_flow_tensor[..., 3]

            mask_index = tf.where(range_image_mask)

            (origins_cartesian, points_cartesian,) = extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local,
            )
            origins_cartesian = tf.squeeze(origins_cartesian, axis=0)
            points_cartesian = tf.squeeze(points_cartesian, axis=0)

            origins_tensor = tf.gather_nd(origins_cartesian, mask_index)
            points_tensor = tf.gather_nd(points_cartesian, mask_index)

            points_intensity_tensor = tf.gather_nd(range_image_intensity, mask_index)
            points_elongation_tensor = tf.gather_nd(range_image_elongation, mask_index)

            points_flow_x_tensor = tf.expand_dims(tf.gather_nd(flow_x, mask_index), axis=1)
            points_flow_y_tensor = tf.expand_dims(tf.gather_nd(flow_y, mask_index), axis=1)
            points_flow_z_tensor = tf.expand_dims(tf.gather_nd(flow_z, mask_index), axis=1)
            points_flow_class_tensor = tf.expand_dims(tf.gather_nd(flow_class, mask_index), axis=1)

            origins.append(origins_tensor.numpy())
            points.append(points_tensor.numpy())
            points_intensity.append(points_intensity_tensor.numpy())
            points_elongation.append(points_elongation_tensor.numpy())
            laser_ids.append(np.full_like(points_intensity_tensor.numpy(), c.name - 1))

            points_flow.append(
                tf.concat(
                    [
                        points_flow_x_tensor,
                        points_flow_y_tensor,
                        points_flow_z_tensor,
                        points_flow_class_tensor,
                    ],
                    axis=-1,
                ).numpy()
            )

        return (
            origins,
            points,
            points_flow,
            cp_points,
            points_intensity,
            points_elongation,
            laser_ids,
        )

    print(">> Opening", filename)
    dataset = tf.data.TFRecordDataset(filename, compression_type="")

    calib = None

    for data in tqdm(dataset, leave=False):
        frame = open_dataset.Frame()
        frame.ParseFromString((data.numpy()))

        content = frame_utils.parse_range_image_and_camera_projection(frame)
        range_images, camera_projections, _, range_image_top_pose = content

        range_images_flow, _, _ = parse_range_image_flow_and_camera_projection(frame)

        views = {}

        # once in a sequence, read camera calibration info
        if calib is None:
            calib = []
            for cam in frame.context.camera_calibrations:
                calib.append(
                    (
                        cam.name,
                        dict(
                            width=cam.width,
                            height=cam.height,
                            intrinsics=list(cam.intrinsic),
                            extrinsics=list(cam.extrinsic.transform),
                        ),
                    )
                )
            # Yield calibration first
            yield (calib, {})

        # convert LIDAR to pointcloud
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose
        )

        (
            origins,
            points_new,
            flows,
            cp_points_new,
            intensity,
            elongation,
            laser_ids,
        ) = convert_range_image_to_point_cloud_flow(
            frame,
            range_images,
            range_images_flow,
            camera_projections,
            range_image_top_pose,
            ri_index=0,
        )

        # 3d points in vehicle frame.
        points_all = np.concatenate(points, axis=0)
        flows_all = np.concatenate(flows, axis=0)
        cp_points_all = np.concatenate(cp_points, axis=0)

        # The distance between lidar points and vehicle frame origin.
        cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

        for i, image in enumerate(frame.images):
            # select relevant 3D points for this view
            mask = tf.equal(cp_points_all_tensor[..., 0], image.name)
            cp_points_msk_tensor = tf.cast(
                tf.gather_nd(cp_points_all_tensor, tf.where(mask)), dtype=tf.float32
            )

            pose = np.asarray(image.pose.transform).reshape(4, 4)
            timestamp = image.pose_timestamp

            rgb = tf.image.decode_jpeg(image.image).numpy()

            pix = cp_points_msk_tensor[..., 1:3].numpy().round().astype(np.int16)
            pts3d = points_all[mask.numpy()]
            flows_view = flows_all[mask.numpy()]

            # Extract label data for dynamic mask generation
            labels_data = []
            for label in frame.laser_labels:
                box = label.box
                meta = label.metadata
                if not box.ByteSize():
                    continue
                # Use num_lidar_points_in_box if num_top_lidar_points_in_box is not available
                if not label.num_top_lidar_points_in_box and not label.num_lidar_points_in_box:
                    continue
                speed = np.linalg.norm([meta.speed_x, meta.speed_y])
                labels_data.append({
                    'box': [box.center_x, box.center_y, box.center_z,
                           box.length, box.width, box.height, box.heading],
                    'speed': float(speed)
                })

            # Get camera calibration for this view
            calibration = next(cc for cc in frame.context.camera_calibrations if cc.name == image.name)
            calibration_data = {
                'width': calibration.width,
                'height': calibration.height,
                'intrinsic': list(calibration.intrinsic),
                'extrinsic': list(calibration.extrinsic.transform)
            }

            views[image.name] = dict(
                img=rgb,
                pose=pose,
                pixels=pix,
                pts3d=pts3d,
                timestamp=timestamp,
                flows=flows_view,
                labels=labels_data,
                vehicle_pose=frame.pose.transform,
                calibration=calibration_data
            )

        # Yield frame data immediately instead of accumulating
        yield (frame.context.name, views)

        # Clear memory after each frame
        del frame, content, range_images, camera_projections, range_image_top_pose
        del range_images_flow, points, cp_points, origins, points_new, flows
        del cp_points_new, intensity, elongation, laser_ids
        del points_all, flows_all, cp_points_all, cp_points_all_tensor

        # Clear TensorFlow memory
        tf.keras.backend.clear_session()
        gc.collect()


def make_crops(output_dir, workers=16, enable_sam=False, sam_model_type="sam2",
               sam_device="cuda", sam_config_file=None, sam_ckpt_path=None, start=0, end=None, **kw):
    tmp_dir = osp.join(output_dir, "tmp")
    sequences = _list_sequences(tmp_dir)
    # Select sequences based on start and end indices
    if end is None:
        end = len(sequences)
    sequences = sequences[start:end]
    args = [(tmp_dir, output_dir, seq, enable_sam, sam_model_type, sam_device, sam_config_file, sam_ckpt_path) for seq in sequences]
    parallel_map(crop_one_seq, args, star_args=True, workers=workers, front_num=0)


def crop_one_seq(input_dir, output_dir, seq, enable_sam=False, sam_model_type="sam2", 
                 sam_device="cuda", sam_config_file=None, sam_ckpt_path=None, resolution=512):
    seq_dir = osp.join(input_dir, seq)
    out_dir = osp.join(output_dir, seq)
    if osp.isfile(osp.join(out_dir, "00100_1.jpg")):
        return
    os.makedirs(out_dir, exist_ok=True)
    
    # 初始化SAM预处理器
    sam_preprocessor = None
    if enable_sam:
        try:
            sam_preprocessor = SAMPreprocessor(
                model_type=sam_model_type,
                device=sam_device,
                config_file=sam_config_file,
                ckpt_path=sam_ckpt_path
            )
        except Exception as e:
            print(f"Failed to initialize SAM preprocessor for sequence {seq}: {e}")
            enable_sam = False

    # load calibration file
    try:
        with open(osp.join(seq_dir, "calib.json")) as f:
            calib = json.load(f)
    except IOError:
        print(f"/!\\ Error: Missing calib.json in sequence {seq} /!\\", file=sys.stderr)
        return

    axes_transformation = np.array(
        [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
    )

    cam_K = {}
    cam_distortion = {}
    cam_res = {}
    cam_to_car = {}
    for cam_idx, cam_info in calib:
        cam_idx = str(cam_idx)
        cam_res[cam_idx] = (W, H) = (cam_info["width"], cam_info["height"])
        f1, f2, cx, cy, k1, k2, p1, p2, k3 = cam_info["intrinsics"]
        cam_K[cam_idx] = np.asarray([(f1, 0, cx), (0, f2, cy), (0, 0, 1)])
        cam_distortion[cam_idx] = np.asarray([k1, k2, p1, p2, k3])
        cam_to_car[cam_idx] = np.asarray(cam_info["extrinsics"]).reshape(
            4, 4
        )  # cam-to-vehicle

    frames = sorted(f[:-3] for f in os.listdir(seq_dir) if f.endswith(".jpg"))

    # from dust3r.viz import SceneViz
    # viz = SceneViz()

    for frame in tqdm(frames, leave=False):
        cam_idx = frame[-2]  # cam index
        assert cam_idx in "12345", f"bad {cam_idx=} in {frame=}"
        data = np.load(osp.join(seq_dir, frame + "npz"))
        car_to_world = data["pose"]
        W, H = cam_res[cam_idx]

        # load depthmap
        pos2d = data["pixels"].round().astype(np.uint16)
        x, y = pos2d.T
        pts3d = data["pts3d"]  # already in the car frame
        flows = data["flows"]
        pts3d = geotrf(axes_transformation @ inv(cam_to_car[cam_idx]), pts3d)
        # Transform flows from car coordinate system to camera coordinate system
        # Only transform the first 3 dimensions, keep the 4th dimension (flow category) unchanged
        flows_xyz = flows[:, :3]  # Extract first 3 dimensions
        # For velocity vectors, only apply rotation transformation (no translation)
        rotation_matrix = (axes_transformation @ inv(cam_to_car[cam_idx]))[:3, :3]
        flows_xyz = flows_xyz @ rotation_matrix.T  # Apply rotation only
        flows[:, -1] += 1 # add 1 to the category(-1 -> 0)
        flows = np.concatenate([flows_xyz, flows[:, 3:]], axis=1)  # Combine transformed xyz with original category

        # X=LEFT_RIGHT y=ALTITUDE z=DEPTH

        # load image
        image = imread_cv2(osp.join(seq_dir, frame + "jpg"))

        # downscale image
        output_resolution = (resolution, 1) if W > H else (1, resolution)
        image, _, intrinsics2 = cropping.rescale_image_depthmap(
            image, None, cam_K[cam_idx], output_resolution
        )
        image.save(osp.join(out_dir, frame + "jpg"))

        # save as an EXR file? yes it's smaller (and easier to load)
        W, H = image.size
        depthmap = np.zeros((H, W), dtype=np.float32)
        pos2d = (
            geotrf(intrinsics2 @ inv(cam_K[cam_idx]), pos2d).round().astype(np.int16)
        )
        x, y = pos2d.T
        depthmap[y.clip(min=0, max=H - 1), x.clip(min=0, max=W - 1)] = pts3d[:, 2]
        cv2.imwrite(osp.join(out_dir, frame + "exr"), depthmap)

        # save flow
        flowmap = np.zeros((H, W, 4), dtype=np.float32)  # [x_flow, y_flow, z_flow, category]
        valid_mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        flowmap[y[valid_mask], x[valid_mask]] = flows[valid_mask]
        np.save(osp.join(out_dir, frame + "npy"), flowmap)

        # Generate dynamic mask
        if 'labels_json' in data.files and 'vehicle_pose' in data.files and 'calibration_json' in data.files:
            try:
                from waymo_open_dataset.utils import box_utils
                from waymo_open_dataset import dataset_pb2

                # Load saved data from JSON strings
                labels_list = json.loads(str(data['labels_json']))
                vehicle_pose_transform = data['vehicle_pose']
                calib_data = json.loads(str(data['calibration_json']))

                # Create a mock vehicle pose object
                class MockPose:
                    def __init__(self, transform):
                        self.transform = transform

                class MockCalibration:
                    def __init__(self, calib_dict):
                        self.width = int(calib_dict['width'])
                        self.height = int(calib_dict['height'])
                        self.intrinsic = calib_dict['intrinsic']

                        class MockExtrinsic:
                            def __init__(self, transform):
                                self.transform = transform

                        self.extrinsic = MockExtrinsic(calib_dict['extrinsic'])

                vehicle_pose = MockPose(vehicle_pose_transform)
                calibration = MockCalibration(calib_data)

                # Initialize dynamic mask
                dynamic_mask = np.zeros((H, W), dtype=np.float32)

                # Process each label
                for label_info in labels_list:
                    speed = label_info['speed']
                    box_coords = np.array([label_info['box']])

                    # Get 3D box corners
                    corners = box_utils.get_upright_3d_box_corners(box_coords)[0].numpy()

                    # Project to image
                    projected_corners = project_vehicle_to_image(vehicle_pose, calibration, corners)
                    u, v, ok = projected_corners.transpose()
                    ok = ok.astype(bool)

                    # Skip if projection failed
                    if not all(ok):
                        continue

                    u = u[ok]
                    v = v[ok]

                    # Clip to original image bounds (before downscaling)
                    u = np.clip(u, 0, calib_data['width'])
                    v = np.clip(v, 0, calib_data['height'])

                    # Scale to downsampled image size
                    scale_x = W / calib_data['width']
                    scale_y = H / calib_data['height']
                    u = u * scale_x
                    v = v * scale_y

                    if u.max() - u.min() == 0 or v.max() - v.min() == 0:
                        continue

                    # Get 2D bounding box
                    xy = (u.min(), v.min())
                    width = u.max() - u.min()
                    height = v.max() - v.min()

                    # Fill mask with speed (max pooling for overlaps)
                    x1, y1 = int(xy[0]), int(xy[1])
                    x2, y2 = int(xy[0] + width), int(xy[1] + height)
                    x1, x2 = max(0, x1), min(W, x2)
                    y1, y2 = max(0, y1), min(H, y2)

                    dynamic_mask[y1:y2, x1:x2] = np.maximum(
                        dynamic_mask[y1:y2, x1:x2],
                        speed
                    )

                # Threshold: objects moving > 1.0 m/s are dynamic
                dynamic_mask = np.clip((dynamic_mask > 1.0) * 255, 0, 255).astype(np.uint8)
                dynamic_output_path = osp.join(out_dir, frame + "dynamic.png")
                PIL.Image.fromarray(dynamic_mask, 'L').save(dynamic_output_path)
            except Exception as e:
                print(f"Error generating dynamic mask for {seq}/{frame}: {e}")

        # save camera parametes
        cam2world = car_to_world @ cam_to_car[cam_idx] @ inv(axes_transformation)
        np.savez(
            osp.join(out_dir, frame + "npz"),
            intrinsics=intrinsics2,
            cam2world=cam2world,
            distortion=cam_distortion[cam_idx],
        )
        
        # 生成SAM掩码
        if enable_sam and sam_preprocessor is not None:
            try:
                # 创建SAM掩码输出目录
                sam_output_dir = osp.join(out_dir, "sam_masks")
                os.makedirs(sam_output_dir, exist_ok=True)
                
                # 生成SAM掩码文件路径
                sam_output_path = osp.join(sam_output_dir, frame + "json")
                
                if not osp.exists(sam_output_path):  # 避免重复处理
                    masks = sam_preprocessor.generate_masks(image)
                    sam_preprocessor.save_masks(masks, sam_output_path)
            except Exception as e:
                print(f"Error generating SAM masks for {seq}/{frame}: {e}")

        # viz.add_rgbd(np.asarray(image), depthmap, intrinsics2, cam2world)
    # viz.show()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    import debugpy
    debugpy.listen(5697)
    print("Waiting for debugger to attach...")
    debugpy.wait_for_client()

    main(args.waymo_dir, args.precomputed_pairs, args.output_dir, workers=args.workers,
         enable_sam=args.enable_sam, sam_model_type=args.sam_model_type, sam_device=args.sam_device,
         sam_config_file=args.sam_config_file, sam_ckpt_path=args.sam_ckpt_path,
         start=args.start, end=args.end)
