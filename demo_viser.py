# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import time
import threading
import argparse
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2
import imageio
from safetensors.torch import load_model


try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

import sys
# 添加vggt路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/vggt'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map, unproject_depth_map_to_point_map_batch, homo_matrix_inverse
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.training.loss import velocity_local_to_global
sys.path.append(os.path.join(os.path.dirname(__file__), "src/SEA-RAFT/core"))
from raft import RAFT
from vggt.utils.auxiliary import RAFTCfg, calc_flow


def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,  # represents percentage (e.g., 50 means filter lowest 50%)
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: str = None,
):
    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        background_mode (bool): Whether to run the server in background thread.
        mask_sky (bool): Whether to apply sky segmentation to filter out sky points.
        image_folder (str): Path to the folder containing input images.
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)

    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)

    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
    conf = depth_conf

    world_points_in_next = pred_dict["pts_in_next_frame"]
    world_points_in_next = np.concatenate(
        [world_points_in_next, np.zeros_like(world_points_in_next[:1])], axis=0
    )
    world_points_in_next_mask = pred_dict["pts_in_next_frame_mask"]
    world_points_in_next_mask = np.concatenate(
        [world_points_in_next_mask, np.zeros_like(world_points_in_next_mask[:1])], axis=0
    )

    # Apply sky segmentation if enabled
    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder)

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)
    points_in_next = world_points_in_next.reshape(-1, 3)
    points_in_next_mask = world_points_in_next_mask.reshape(-1)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) typically
    # For convenience, we store only (3,4) portion
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center
    points_in_next = points_in_next - scene_center

    # Store frame indices so we can filter by frame
    frame_indices = np.repeat(np.arange(S), H * W)

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox(
        "Show Cameras",
        initial_value=True,
    )

    # Now the slider represents percentage of points to filter out
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent",
        min=0,
        max=100,
        step=0.1,
        initial_value=init_conf_threshold,
    )

    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames",
        options=["All"] + [str(i) for i in range(S)],
        initial_value="All",
    )

    gui_show_next_frame = server.gui.add_button("Next Frame")
    gui_show_prev_frame = server.gui.add_button("Previous Frame")
    gui_anchor = server.gui.add_checkbox(
        "Anchor Points",
        initial_value=True,
    )

    gui_point_size = server.gui.add_slider(
        "Point Size",
        min=0.0001,
        max=0.1,
        step=0.0001,
        initial_value=0.002,
    )

    # Create the main point cloud handle
    # Compute the threshold value as the given percentile
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.002,
        point_shape="circle",
    )

    # We will store references to frames & frustums so we can toggle visibility
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        """
        Add camera frames and frustums to the scene.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """
        # Clear any existing frames or frustums
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        # Optionally attach a callback that sets the viewpoint to the chosen camera
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        img_ids = range(S)
        for img_id in tqdm(img_ids):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # Add a small frame axis
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            # Convert the image for the frustum
            img = images_[img_id]  # shape (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # If you want correct FOV from intrinsics, do something like:
            # fx = intrinsics_cam[img_id, 0, 0]
            # fov = 2 * np.arctan2(h/2, fx)
            # For demonstration, we pick a simple approximate FOV:
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Add the frustum
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum",
                fov=fov,
                aspect=w / h,
                scale=0.05,
                image=img,
                line_width=1.0,
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        """Update the point cloud based on current GUI selections."""
        # Here we compute the threshold value based on the current percentage
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)

        print(f"Threshold absolute value: {threshold_val}, percentage: {current_percentage}%")

        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask
        # point_cloud.points = points_in_next[combined_mask]
        if gui_anchor.value == False:
            point_cloud.points = points_in_next[combined_mask]
        else:
            point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_next_frame.on_click
    def _(_) -> None:
        """Toggle visibility of next frame points."""
        if gui_frame_selector.value == "All":
            return
        selected_idx = int(gui_frame_selector.value)
        if gui_anchor.value:
            gui_anchor.value = False
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx
            conf_mask = (conf_flat >= np.percentile(conf_flat, gui_points_conf.value)) & (conf_flat > 0.1)
            combined_mask = frame_mask & points_in_next_mask & conf_mask
            point_cloud.points = points_in_next[combined_mask]
            point_cloud.colors = colors_flat[combined_mask]
        else:
            gui_anchor.value = True
            gui_frame_selector.value = str(selected_idx + 1 if selected_idx + 1 < S else 0)

    @gui_show_prev_frame.on_click
    def _(_) -> None:
        """Toggle visibility of previous frame points."""
        if gui_frame_selector.value == "All":
            return
        selected_idx = int(gui_frame_selector.value)
        if gui_anchor.value:
            gui_anchor.value = False
            gui_frame_selector.value = str(selected_idx - 1 if selected_idx > 0 else S - 1)
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx
            conf_mask = (conf_flat >= np.percentile(conf_flat, gui_points_conf.value)) & (conf_flat > 0.1)
            combined_mask = frame_mask & points_in_next_mask & conf_mask
            point_cloud.points = points_in_next[combined_mask]
            point_cloud.colors = colors_flat[combined_mask]
        else:
            gui_anchor.value = True
            update_point_cloud()


    @gui_point_size.on_update
    def _(_) -> None:
        """Update the point size of the point cloud."""
        point_cloud.point_size = gui_point_size.value

    @gui_show_frames.on_update
    def _(_) -> None:
        """Toggle visibility of camera frames and frustums."""
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    # Add the camera frames to the scene
    visualize_frames(cam_to_world, images)

    print("Starting viser server...")
    # If background_mode is True, spawn a daemon thread so the main thread can continue.
    if background_mode:

        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server


# Helper functions for sky segmentation


def apply_sky_segmentation(conf: np.ndarray, image_folder: str) -> np.ndarray:
    """
    Apply sky segmentation to confidence scores.

    Args:
        conf (np.ndarray): Confidence scores with shape (S, H, W)
        image_folder (str): Path to the folder containing input images

    Returns:
        np.ndarray: Updated confidence scores with sky regions masked out
    """
    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    # Download skyseg.onnx if it doesn't exist
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    sky_mask_list = []

    print("Generating sky masks...")
    for i, image_path in enumerate(tqdm(image_files[:S])):  # Limit to the number of images in the batch
        image_name = os.path.basename(image_path)
        mask_filepath = os.path.join(sky_masks_dir, image_name)

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

        # Resize mask to match H×W if needed
        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))

        sky_mask_list.append(sky_mask)

    # Convert list to numpy array with shape S×H×W
    sky_mask_array = np.array(sky_mask_list)
    # Apply sky mask to confidence scores
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
    conf = conf * sky_mask_binary

    print("Sky segmentation applied successfully")
    return conf


def get_next_pts(preds, pred_pts=False):
    images = preds["images"]
    depthmaps = preds["depth"]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
    pad_row = torch.tensor([0, 0, 0, 1], device=images.device)[None, None, None].expand(*extrinsic.shape[:-2], -1, -1)
    extrinsic = torch.cat([extrinsic, pad_row], dim=-2)
    B, S, H, W, _ = depthmaps.shape
    world_points, _, world_points_mask = unproject_depth_map_to_point_map_batch(
        depthmaps.reshape(B * S, H, W, -1),
        extrinsic.reshape(B * S, 4, 4),
        intrinsic.reshape(B * S, 3, 3),
    )
    world_points = world_points.reshape(B, S, H, W, 3)
    world_points_mask = world_points_mask.reshape(B, S, H, W)

    velocity = preds.pop("velocity")
    velocity = torch.sign(velocity) * (torch.exp(torch.abs(velocity)) - 1)
    # velocity = torch.zeros_like(velocity) #TODO: remove this

    # 使用velocity_local_to_global函数将velocity从局部坐标系转换到全局坐标系
    # velocity shape: (B, S, H, W, 3) -> 需要重塑为 (N, 3) 其中 N = B*S*H*W
    velocity_reshaped = velocity.reshape(-1, 3)  # (B*S*H*W, 3)
    
    # 使用velocity_local_to_global函数进行坐标系转换
    extrinsic_inv = torch.linalg.inv(extrinsic)
    velocity_global = velocity_local_to_global(velocity_reshaped, extrinsic_inv)  # (B*S*H*W, 3)
    
    # 重塑回原始形状
    velocity_global = velocity_global.reshape(B, S, H, W, 3)  # (B, S, H, W, 3)

    pts_in_next = world_points + velocity_global
    pts_in_next = pts_in_next[:, :-1]  # remove the last frame
    pts_in_next_mask = torch.ones_like(pts_in_next[..., 0], dtype=torch.bool)
    preds["pts_in_next_frame"] = pts_in_next
    preds["pts_in_next_frame_mask"] = pts_in_next_mask
    return preds




parser = argparse.ArgumentParser(description="VGGT demo with viser for 3D visualization")
# parser.add_argument(
#     "--image_folder", type=str, default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/val/segment-1505698981571943321_1186_773_1206_773_with_camera_labels.tfrecord", help="Path to folder containing images"
# )
# parser.add_argument(
#     "--image_folder", type=str, default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train/segment-15795616688853411272_1245_000_1265_000_with_camera_labels.tfrecord", help="Path to folder containing images"
# )
parser.add_argument(
    "--image_folder", type=str, default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train/segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord", help="Path to folder containing images"
)
# parser.add_argument(
#     "--image_folder", type=str, default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train/segment-14830022845193837364_3488_060_3508_060_with_camera_labels.tfrecord", help="Path to folder containing images"
# )
# parser.add_argument(
#     "--image_folder", type=str, default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord", help="Path to folder containing images"
# )
parser.add_argument(
    "--image_interval", type=int, default=1, help="Interval for selecting images from the folder"
)
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
parser.add_argument(
    "--conf_threshold", type=float, default=50.0, help="Initial percentage of low-confidence points to filter out"
)
parser.add_argument("--pred_pts", action="store_true", help="Use next frame points in local coordinate")
parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
parser.add_argument("--debug", action="store_true", help="Enable debug mode for additional logging")

def main():
    """
    Main function for the VGGT demo with viser for 3D visualization.

    This function:
    1. Loads the VGGT model
    2. Processes input images from the specified folder
    3. Runs inference to generate 3D points and camera poses
    4. Optionally applies sky segmentation to filter out sky points
    5. Visualizes the results using viser

    Command-line arguments:
    --image_folder: Path to folder containing input images
    --use_point_map: Use point map instead of depth-based points
    --background_mode: Run the viser server in background mode
    --port: Port number for the viser server
    --conf_threshold: Initial percentage of low-confidence points to filter out
    --mask_sky: Apply sky segmentation to filter out sky points
    """
    args = parser.parse_args()
    if args.debug:
        import debugpy
        debugpy.listen(("localhost", 5678))
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")

    model = VGGT()
    # ckpt = torch.load("/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo/step2(true+fixmodel+lowlr!+nolpips+onlyflow+velocitylocal+fromscratch)/checkpoint-epoch_2_17880.pth", map_location=device)['model']
    ckpt = torch.load("/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/stage1_gtflow+depthgrad(true)+depth+flowgradconf+aggregatorenderloss+fixopacity+no1opacityloss+fixdirection+fromvresume+lpips+noquantize/checkpoint-epoch_0_19530.pth", map_location=device)['model']
    # ckpt = torch.load("/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/stage1_gtflow/checkpoint-epoch_4_24208.pth", map_location=device)['model']
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    # model = VGGT(use_sky_token=False, use_scale_token=False)    
    # model.load_state_dict(torch.load("src/model.pt"), strict=False)

    model.eval()
    model = model.to(device)

    auxiliary_models = dict()
    flow = eval("""RAFT(RAFTCfg(name="kitti-M", dataset="kitti", path="src/Tartan-C-T-TSKH-kitti432x960-M.pth",
        use_var=True, var_min=0, var_max=10, pretrain="resnet34", initial_dim=64, block_dims=[64, 128, 256],
        radius=4, dim=128, num_blocks=2, iters=4, image_size=[432, 960],
        geo_thresh=2, photo_thresh=-1))""")
    flow.load_state_dict(torch.load("src/Tartan-C-T-TSKH-kitti432x960-M.pth"))
    flow.eval()
    flow = flow.to(device)
    auxiliary_models["flow"] = flow

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = glob.glob(os.path.join(args.image_folder, "*.jpg")) + glob.glob(os.path.join(args.image_folder, "*.png"))
    # 只提取_后面是1.jpg或.png的图片
    image_names = [name for name in image_names if name.split("/")[-1].split("_")[-1] in ["1.jpg", "1.png"]]
    # image_names = sorted(image_names)[::5]
    image_names = sorted(image_names)[160:168]
    # 第一个不变，其他逆序
    # image_names = [image_names[0]] + image_names[1:][::-1]
    print(f"Found {len(image_names)} images")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")


    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    
    predictions = get_next_pts(predictions, pred_pts=args.pred_pts)


    print("Processing model outputs...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            # Convert BFloat16 to float32 before converting to numpy
            tensor = predictions[key]
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            predictions[key] = tensor.cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

    if args.use_point_map:
        print("Visualizing 3D points from point map")
    else:
        print("Visualizing 3D points by unprojecting depth map by cameras")

    if args.mask_sky:
        print("Sky segmentation enabled - will filter out sky points")

    print("Starting viser visualization...")

    viser_server = viser_wrapper(
        predictions,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
    )
    print("Visualization complete")


if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(("localhost", 5678))
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()
    main()


