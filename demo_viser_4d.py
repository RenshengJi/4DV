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
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map, unproject_depth_map_to_point_map_batch, homo_matrix_inverse
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
sys.path.append(os.path.join(os.path.dirname(__file__), "src/SEA-RAFT/core"))
from raft import RAFT
from vggt.utils.auxiliary import RAFTCfg, calc_flow


def viser_wrapper_4d(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,  # represents percentage (e.g., 50 means filter lowest 50%)
    init_velocity_threshold: float = 0.1,  # 初始速度阈值
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: str = None,
):
    """
    Visualize predicted 3D points and camera poses with viser, with 4D dynamic/static separation.

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
                "velocity": (S-1, H, W, 3),  # 速度场
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        init_velocity_threshold (float): Initial velocity threshold for static/dynamic separation.
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
    velocity = pred_dict["velocity"]  # (S, H, W, 3)

    world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
    conf = depth_conf

    # Apply sky segmentation if enabled
    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder)

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # 计算速度幅值用于动静态分离
    velocity_magnitude = np.linalg.norm(velocity, axis=-1)  # (S, H, W)
    
    

    # Flatten all data
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)
    velocity_magnitude_flat = velocity_magnitude.reshape(-1)
    frame_indices = np.repeat(np.arange(S), H * W)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox(
        "Show Cameras",
        initial_value=True,
    )

    gui_points_conf = server.gui.add_slider(
        "Confidence Percent",
        min=0,
        max=100,
        step=0.1,
        initial_value=init_conf_threshold,
    )

    gui_velocity_threshold = server.gui.add_slider(
        "Velocity Threshold",
        min=0.0,
        max=2.0,
        step=0.01,
        initial_value=init_velocity_threshold,
    )

    gui_frame_slider = server.gui.add_slider(
        "Dynamic Frame",
        min=0,
        max=S-1,
        step=1,
        initial_value=0,
    )

    gui_auto_play = server.gui.add_checkbox(
        "Auto Play",
        initial_value=False,
    )

    gui_playback_speed = server.gui.add_slider(
        "Playback Speed (FPS)",
        min=0.1,
        max=10.0,
        step=0.1,
        initial_value=2.0,
    )

    gui_show_static = server.gui.add_checkbox(
        "Show Static Points",
        initial_value=True,
    )

    gui_show_dynamic = server.gui.add_checkbox(
        "Show Dynamic Points",
        initial_value=True,
    )

    gui_point_size = server.gui.add_slider(
        "Point Size",
        min=0.0001,
        max=0.1,
        step=0.0001,
        initial_value=0.002,
    )

    # 创建静态和动态点云
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    init_velocity_mask = velocity_magnitude_flat <= init_velocity_threshold
    
    # 静态点云
    static_mask = init_conf_mask & init_velocity_mask
    static_point_cloud = server.scene.add_point_cloud(
        name="static_pcd",
        points=points_centered[static_mask],
        colors=colors_flat[static_mask],
        point_size=0.002,
        point_shape="circle",
    )

    # 动态点云（初始显示第0帧）
    dynamic_mask = init_conf_mask & (velocity_magnitude_flat > init_velocity_threshold) & (frame_indices == 0)
    dynamic_point_cloud = server.scene.add_point_cloud(
        name="dynamic_pcd",
        points=points_centered[dynamic_mask],
        colors=colors_flat[dynamic_mask],
        point_size=0.002,
        point_shape="circle",
    )

    # We will store references to frames & frustums so we can toggle visibility
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []
    
    # 自动播放相关变量
    last_play_time = time.time()
    current_frame_index = 0

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

    def update_point_clouds() -> None:
        """Update both static and dynamic point clouds based on current GUI selections."""
        # 计算置信度阈值
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)
        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)
        
        # 计算速度阈值
        velocity_threshold = gui_velocity_threshold.value
        velocity_mask = velocity_magnitude_flat <= velocity_threshold
        
        # 更新静态点云
        static_mask = conf_mask & velocity_mask
        static_point_cloud.points = points_centered[static_mask]
        static_point_cloud.colors = colors_flat[static_mask]
        
        # 更新动态点云
        current_frame = int(gui_frame_slider.value)
        dynamic_mask = conf_mask & (velocity_magnitude_flat > velocity_threshold) & (frame_indices == current_frame)
        dynamic_point_cloud.points = points_centered[dynamic_mask]
        dynamic_point_cloud.colors = colors_flat[dynamic_mask]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_clouds()

    @gui_velocity_threshold.on_update
    def _(_) -> None:
        update_point_clouds()

    @gui_frame_slider.on_update
    def _(_) -> None:
        update_point_clouds()

    @gui_auto_play.on_update
    def _(_) -> None:
        """Toggle auto play mode."""
        nonlocal current_frame_index
        if not gui_auto_play.value:
            # 停止自动播放时，重置到当前滑块位置
            current_frame_index = int(gui_frame_slider.value)

    @gui_playback_speed.on_update
    def _(_) -> None:
        """Update playback speed."""
        pass  # 速度更新在自动播放循环中处理

    @gui_show_static.on_update
    def _(_) -> None:
        """Toggle visibility of static point cloud."""
        static_point_cloud.visible = gui_show_static.value

    @gui_show_dynamic.on_update
    def _(_) -> None:
        """Toggle visibility of dynamic point cloud."""
        dynamic_point_cloud.visible = gui_show_dynamic.value

    @gui_point_size.on_update
    def _(_) -> None:
        """Update the point size of both point clouds."""
        static_point_cloud.point_size = gui_point_size.value
        dynamic_point_cloud.point_size = gui_point_size.value

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
            # 自动播放逻辑
            if gui_auto_play.value:
                current_time = time.time()
                frame_interval = 1.0 / gui_playback_speed.value
                
                if current_time - last_play_time >= frame_interval:
                    # 更新帧索引
                    current_frame_index = (current_frame_index + 1) % S
                    gui_frame_slider.value = current_frame_index
                    last_play_time = current_time
                    
                    # 更新点云
                    update_point_clouds()
            
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



parser = argparse.ArgumentParser(description="VGGT demo with viser for 4D visualization")
# parser.add_argument(
#     "--image_folder", type=str, default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/val/segment-1505698981571943321_1186_773_1206_773_with_camera_labels.tfrecord", help="Path to folder containing images"
# )
# parser.add_argument(
#     "--image_folder", type=str, default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train/segment-15795616688853411272_1245_000_1265_000_with_camera_labels.tfrecord", help="Path to folder containing images"
# )
# parser.add_argument(
#     "--image_folder", type=str, default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train/segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord", help="Path to folder containing images"
# )
parser.add_argument(
    "--image_folder", type=str, default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train/segment-10226164909075980558_180_000_200_000_with_camera_labels.tfrecord", help="Path to folder containing images"
)
parser.add_argument(
    "--image_interval", type=int, default=1, help="Interval for selecting images from the folder"
)
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
parser.add_argument(
    "--conf_threshold", type=float, default=50.0, help="Initial percentage of low-confidence points to filter out"
)
parser.add_argument(
    "--velocity_threshold", type=float, default=0.1, help="Initial velocity threshold for static/dynamic separation"
)
parser.add_argument("--pred_pts", action="store_true", help="Use next frame points in local coordinate")
parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
parser.add_argument("--debug", action="store_true", help="Enable debug mode for additional logging")

def main():
    """
    Main function for the VGGT demo with viser for 4D visualization.

    This function:
    1. Loads the VGGT model
    2. Processes input images from the specified folder
    3. Runs inference to generate 3D points and camera poses
    4. Separates static and dynamic points based on velocity threshold
    5. Visualizes the results using viser with 4D capabilities

    Command-line arguments:
    --image_folder: Path to folder containing input images
    --use_point_map: Use point map instead of depth-based points
    --background_mode: Run the viser server in background mode
    --port: Port number for the viser server
    --conf_threshold: Initial percentage of low-confidence points to filter out
    --velocity_threshold: Initial velocity threshold for static/dynamic separation
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
    model = VGGT()
    ckpt = torch.load("/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo/step2(true+fixmodel+lowlr!+nolpips+onlyflow)/checkpoint-epoch_0_52448.pth", map_location=device)['model']
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)

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
    image_names = sorted(image_names)[::args.image_interval][100:124]
    print(f"Found {len(image_names)} images")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # 将images保存为video
    video_path = os.path.join("/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo", "video.mp4")
    
    print(f"图像形状: {images.shape}")
    print(f"图像数据类型: {images.dtype}")
    print(f"图像值范围: [{images.min():.3f}, {images.max():.3f}]")
    
    try:
        # 准备视频帧数据
        video_frames = []
        for i, image in enumerate(images):
            # 确保图像格式正确：从 (C, H, W) 转换为 (H, W, C)
            img_np = image.cpu().numpy().transpose(1, 2, 0)
            
            # 确保像素值在0-255范围内
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            
            # 检查图像数据是否有效
            if img_np.shape[0] == 0 or img_np.shape[1] == 0:
                print(f"跳过无效图像帧 {i}")
                continue
            
            video_frames.append(img_np)
        
        # 使用imageio保存视频，确保浏览器兼容性
        if video_frames:
            imageio.mimsave(video_path, video_frames, fps=5, codec='libx264')
            print(f"成功保存视频到 {video_path}")
            
            # 验证文件是否成功创建
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path)
                print(f"视频文件大小: {file_size} 字节")
                if file_size == 0:
                    print("警告: 视频文件大小为0，可能损坏")
            else:
                print("错误: 视频文件未创建")
        else:
            print("没有有效的视频帧可保存")
            
    except Exception as e:
        print(f"保存视频时出错: {e}")
        import traceback
        traceback.print_exc()

    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    
    print("Processing model outputs...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

    if args.use_point_map:
        print("Visualizing 3D points from point map")
    else:
        print("Visualizing 3D points by unprojecting depth map by cameras")

    if args.mask_sky:
        print("Sky segmentation enabled - will filter out sky points")

    print("Starting viser 4D visualization...")

    viser_server = viser_wrapper_4d(
        predictions,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        init_velocity_threshold=args.velocity_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
    )
    print("4D Visualization complete")


if __name__ == "__main__":
    main() 