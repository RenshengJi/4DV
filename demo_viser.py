#!/usr/bin/env python3
"""
Viser-based 3D visualization demo for 4DVideo project.

Interactive visualization of:
- 3D point clouds from depth maps
- Per-pixel velocity (flow) as 3D vectors
- Camera poses and frustums
- Frame-by-frame navigation

Usage:
    python demo_viser.py --data_idx <scene_idx>
    python demo_viser.py  # Use defaults from config
"""

import os
import sys
import time
import threading
import argparse
from typing import List, Dict

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import hydra
from omegaconf import OmegaConf

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models import VGGT
from src.utils import tf32_off


def closed_form_inverse_se3(extrinsics: np.ndarray) -> np.ndarray:
    """Compute inverse of SE3 matrices (world-to-cam -> cam-to-world)."""
    S = extrinsics.shape[0]
    cam_to_world = np.zeros((S, 4, 4), dtype=extrinsics.dtype)
    cam_to_world[:, 3, 3] = 1.0

    R = extrinsics[:, :3, :3]
    t = extrinsics[:, :3, 3]

    R_inv = np.transpose(R, (0, 2, 1))
    t_inv = -np.einsum('bij,bj->bi', R_inv, t)

    cam_to_world[:, :3, :3] = R_inv
    cam_to_world[:, :3, 3] = t_inv

    return cam_to_world


def unproject_depth_to_points(
    depth: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray
) -> np.ndarray:
    """
    Unproject depth map to world points.
    Args:
        depth: (S, H, W)
        extrinsics: (S, 4, 4) world-to-camera
        intrinsics: (S, 3, 3)
    Returns:
        world_points: (S, H, W, 3)
    """
    S, H, W = depth.shape
    world_points = np.zeros((S, H, W, 3), dtype=np.float32)

    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    ones = np.ones_like(x)
    pixel_coords = np.stack([x, y, ones], axis=-1).astype(np.float32)

    for i in range(S):
        K = intrinsics[i]
        w2c = extrinsics[i]
        c2w = np.linalg.inv(w2c)

        K_inv = np.linalg.inv(K)
        cam_points = np.einsum('ij,hwj->hwi', K_inv, pixel_coords)
        cam_points = cam_points * depth[i, :, :, np.newaxis]

        cam_points_homo = np.concatenate([cam_points, np.ones((H, W, 1))], axis=-1)
        world_pts = np.einsum('ij,hwj->hwi', c2w, cam_points_homo)[:, :, :3]
        world_points[i] = world_pts

    return world_points


def load_model(model_path: str, device: torch.device, cfg) -> torch.nn.Module:
    """Load VGGT model from checkpoint."""
    print(f"Loading model from: {model_path}")
    model = eval(cfg.model)

    checkpoint = torch.load(model_path, map_location="cpu")
    ckpt = checkpoint.get('model', checkpoint)
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model


def load_dataset(dataset_cfg: str):
    """Load dataset from config string."""
    from dataset import WaymoDataset, Waymo_Multi, ImgNorm
    print(f"Loading dataset...")
    dataset = eval(dataset_cfg)
    print(f"Dataset loaded: {len(dataset)} scenes")
    return dataset


def prepare_scene_data(model, dataset, idx: int, device: torch.device) -> Dict:
    """
    Run inference and prepare scene data for visualization.

    Returns dict with:
        - images: (S, 3, H, W)
        - depth: (S, H, W) - predicted depth
        - depth_conf: (S, H, W)
        - gt_depth: (S, H, W) - ground truth depth
        - extrinsic: (S, 4, 4)
        - intrinsic: (S, 3, 3)
        - velocity: (S, H, W, 3) - per-pixel velocity in world space
        - gt_velocity: (S, H, W, 3) - ground truth velocity
        - H, W: image dimensions
    """
    print(f"\nProcessing scene index: {idx}")

    from dataset import vggt_collate_fn
    views_list = dataset[idx]
    vggt_batch = vggt_collate_fn([views_list])

    for key in vggt_batch:
        if isinstance(vggt_batch[key], torch.Tensor):
            vggt_batch[key] = vggt_batch[key].to(device)

    is_context_frame = vggt_batch.get('is_context_frame', None)
    if is_context_frame is not None:
        context_mask = is_context_frame[0]
        context_indices = torch.where(context_mask)[0]
    else:
        context_indices = torch.arange(vggt_batch["images"].shape[1], device=device)

    images = vggt_batch['images']
    B, S, C, H, W = images.shape
    intrinsics = vggt_batch.get('intrinsics')
    extrinsics = vggt_batch.get('extrinsics')
    gt_depths = vggt_batch.get('depths', None)
    gt_flowmap = vggt_batch.get('flowmap', None)

    print(f"Running model inference on {len(context_indices)} frames...")
    with torch.no_grad():
        preds = model(
            images[:, context_indices],
            gt_extrinsics=extrinsics[:, context_indices] if extrinsics is not None else None,
            gt_intrinsics=intrinsics[:, context_indices] if intrinsics is not None else None,
            frame_sample_ratio=1.0
        )

    pred_depth = preds.get('depth', None)
    pred_depth_conf = preds.get('depth_conf', None)
    pred_velocity = preds.get('velocity', None)
    pred_extrinsics = preds.get('extrinsics', extrinsics[:, context_indices])
    pred_intrinsics = preds.get('intrinsics', intrinsics[:, context_indices])

    images_np = images[:, context_indices][0].cpu().numpy()
    depth_np = pred_depth[0, :, :, :, 0].cpu().numpy() if pred_depth is not None else None
    depth_conf_np = pred_depth_conf[0].cpu().numpy() if pred_depth_conf is not None else None
    extrinsics_np = pred_extrinsics[0].cpu().numpy()
    intrinsics_np = pred_intrinsics[0].cpu().numpy()
    velocity_np = pred_velocity[0].cpu().numpy() if pred_velocity is not None else None

    # GT depth
    if gt_depths is not None:
        gt_depth_np = gt_depths[:, context_indices][0].cpu().numpy()  # (S, H, W)
        if gt_depth_np.ndim == 4:
            gt_depth_np = gt_depth_np[:, :, :, 0]
    else:
        gt_depth_np = None

    # GT velocity/flowmap
    if gt_flowmap is not None:
        gt_velocity_np = gt_flowmap[:, context_indices][0].cpu().numpy()  # (S, H, W, C)
        # flowmap format: first 3 channels are velocity
        if gt_velocity_np.shape[-1] >= 3:
            gt_velocity_np = gt_velocity_np[:, :, :, :3]
        else:
            gt_velocity_np = None
    else:
        gt_velocity_np = None

    return {
        'images': images_np,
        'depth': depth_np,
        'depth_conf': depth_conf_np,
        'gt_depth': gt_depth_np,
        'extrinsic': extrinsics_np,
        'intrinsic': intrinsics_np,
        'velocity': velocity_np,
        'gt_velocity': gt_velocity_np,
        'H': H,
        'W': W,
    }


def viser_wrapper(
    scene_data: Dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,
    background_mode: bool = False,
):
    """
    Visualize 3D scene with viser.

    Args:
        scene_data: Dictionary from prepare_scene_data()
        port: Port for viser server
        init_conf_threshold: Initial confidence percentile threshold
        background_mode: Run in background thread
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    images = scene_data['images']
    depth = scene_data['depth']
    depth_conf = scene_data['depth_conf']
    gt_depth = scene_data.get('gt_depth', None)
    extrinsics = scene_data['extrinsic']
    intrinsics = scene_data['intrinsic']
    velocity = scene_data['velocity']
    gt_velocity = scene_data.get('gt_velocity', None)
    H, W = scene_data['H'], scene_data['W']

    S = images.shape[0]

    # Unproject predicted depth to world points
    world_points = unproject_depth_to_points(depth, extrinsics, intrinsics)
    conf = depth_conf if depth_conf is not None else np.ones_like(depth)

    # Unproject GT depth to world points
    if gt_depth is not None:
        gt_world_points = unproject_depth_to_points(gt_depth, extrinsics, intrinsics)
        # Create GT confidence mask (valid where depth > 0)
        gt_conf = (gt_depth > 0.01).astype(np.float32)
    else:
        gt_world_points = None
        gt_conf = None

    # Decode velocity: sign(v) * (exp(|v|) - 1)
    if velocity is not None:
        velocity_decoded = np.sign(velocity) * (np.exp(np.abs(velocity)) - 1)
        pts_in_next = world_points + velocity_decoded
        pts_in_next = np.concatenate([pts_in_next[:-1], np.zeros_like(pts_in_next[:1])], axis=0)
        pts_in_next_mask = np.ones(pts_in_next.shape[:3], dtype=bool)
        pts_in_next_mask[-1] = False
    else:
        pts_in_next = None
        pts_in_next_mask = None
        velocity_decoded = None

    # GT velocity for next frame points
    if gt_velocity is not None and gt_world_points is not None:
        gt_pts_in_next = gt_world_points + gt_velocity
        gt_pts_in_next = np.concatenate([gt_pts_in_next[:-1], np.zeros_like(gt_pts_in_next[:1])], axis=0)
        gt_pts_in_next_mask = np.ones(gt_pts_in_next.shape[:3], dtype=bool)
        gt_pts_in_next_mask[-1] = False
    else:
        gt_pts_in_next = None
        gt_pts_in_next_mask = None

    # Camera-to-world
    cam_to_world = closed_form_inverse_se3(extrinsics)

    # Colors from images
    colors = images.transpose(0, 2, 3, 1)  # (S, H, W, 3)

    # Flatten predicted points
    points_flat = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)
    frame_indices_flat = np.repeat(np.arange(S), H * W)

    # Flatten GT points
    if gt_world_points is not None:
        gt_points_flat = gt_world_points.reshape(-1, 3)
        gt_conf_flat = gt_conf.reshape(-1)
    else:
        gt_points_flat = None
        gt_conf_flat = None

    if pts_in_next is not None:
        pts_in_next_flat = pts_in_next.reshape(-1, 3)
        pts_in_next_mask_flat = pts_in_next_mask.reshape(-1)
    else:
        pts_in_next_flat = None
        pts_in_next_mask_flat = None

    if gt_pts_in_next is not None:
        gt_pts_in_next_flat = gt_pts_in_next.reshape(-1, 3)
        gt_pts_in_next_mask_flat = gt_pts_in_next_mask.reshape(-1)
    else:
        gt_pts_in_next_flat = None
        gt_pts_in_next_mask_flat = None

    if velocity_decoded is not None:
        velocity_flat = velocity_decoded.reshape(-1, 3)
    else:
        velocity_flat = None

    if gt_velocity is not None:
        gt_velocity_flat = gt_velocity.reshape(-1, 3)
    else:
        gt_velocity_flat = None

    # Scene center (use predicted points)
    scene_center = np.mean(points_flat, axis=0)
    points_centered = points_flat - scene_center
    cam_to_world[..., :3, 3] -= scene_center

    if gt_points_flat is not None:
        gt_points_centered = gt_points_flat - scene_center
    else:
        gt_points_centered = None

    if pts_in_next_flat is not None:
        pts_in_next_centered = pts_in_next_flat - scene_center
    else:
        pts_in_next_centered = None

    if gt_pts_in_next_flat is not None:
        gt_pts_in_next_centered = gt_pts_in_next_flat - scene_center
    else:
        gt_pts_in_next_centered = None

    # GUI controls
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)

    # Data source selection
    gui_data_source = server.gui.add_dropdown(
        "Data Source",
        options=["Predicted", "GT"] if gt_depth is not None else ["Predicted"],
        initial_value="Predicted",
    )

    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )
    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames",
        options=["All"] + [str(i) for i in range(S)],
        initial_value="All",
    )
    gui_show_next_frame = server.gui.add_button("Next Frame")
    gui_show_prev_frame = server.gui.add_button("Previous Frame")
    gui_anchor = server.gui.add_checkbox("Anchor Points", initial_value=True)
    gui_point_size = server.gui.add_slider(
        "Point Size", min=0.0001, max=0.1, step=0.0001, initial_value=0.002
    )
    gui_show_velocity = server.gui.add_checkbox("Show Velocity Arrows", initial_value=False)
    gui_velocity_scale = server.gui.add_slider(
        "Velocity Scale", min=0.1, max=10.0, step=0.1, initial_value=1.0
    )
    gui_velocity_subsample = server.gui.add_slider(
        "Velocity Subsample", min=1, max=50, step=1, initial_value=10
    )

    # Initial point cloud
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)

    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.002,
        point_shape="circle",
    )

    # Velocity arrows (initially empty)
    velocity_lines = None

    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(cam2world: np.ndarray, images_: np.ndarray) -> None:
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        for img_id in tqdm(range(S), desc="Adding camera frames"):
            c2w_3x4 = cam2world[img_id, :3, :]
            T_world_camera = viser_tf.SE3.from_matrix(c2w_3x4)

            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            img = images_[img_id]
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            fy = intrinsics[img_id, 1, 1]
            fov = 2 * np.arctan2(h / 2, fy)

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
        use_gt = gui_data_source.value == "GT" and gt_points_centered is not None

        if use_gt:
            current_points = gt_points_centered
            current_conf = gt_conf_flat
            current_pts_in_next = gt_pts_in_next_centered
            current_pts_mask = gt_pts_in_next_mask_flat
        else:
            current_points = points_centered
            current_conf = conf_flat
            current_pts_in_next = pts_in_next_centered
            current_pts_mask = pts_in_next_mask_flat

        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(current_conf, current_percentage)
        conf_mask = (current_conf >= threshold_val) & (current_conf > 1e-5)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices_flat == selected_idx

        combined_mask = conf_mask & frame_mask

        if gui_anchor.value or current_pts_in_next is None:
            point_cloud.points = current_points[combined_mask]
        else:
            point_cloud.points = current_pts_in_next[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    def update_velocity_arrows() -> None:
        nonlocal velocity_lines

        if velocity_lines is not None:
            velocity_lines.remove()
            velocity_lines = None

        if not gui_show_velocity.value:
            return

        use_gt = gui_data_source.value == "GT" and gt_velocity_flat is not None

        if use_gt:
            current_velocity = gt_velocity_flat
            current_points = gt_points_centered
            current_conf = gt_conf_flat
        else:
            current_velocity = velocity_flat
            current_points = points_centered
            current_conf = conf_flat

        if current_velocity is None or current_points is None:
            return

        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(current_conf, current_percentage)
        conf_mask = (current_conf >= threshold_val) & (current_conf > 1e-5)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices_flat == selected_idx

        combined_mask = conf_mask & frame_mask

        # Subsample for performance
        subsample = int(gui_velocity_subsample.value)
        indices = np.where(combined_mask)[0][::subsample]

        if len(indices) == 0:
            return

        starts = current_points[indices]
        velocities = current_velocity[indices] * gui_velocity_scale.value
        ends = starts + velocities

        # Filter out zero/tiny velocities
        velocity_mag = np.linalg.norm(velocities, axis=1)
        valid = velocity_mag > 0.001
        if not valid.any():
            return

        starts = starts[valid]
        ends = ends[valid]
        velocity_mag = velocity_mag[valid]

        # Color by velocity magnitude
        max_mag = np.percentile(velocity_mag, 95) + 1e-6
        normalized_mag = np.clip(velocity_mag / max_mag, 0, 1)

        # Colormap: blue (slow) -> red (fast)
        arrow_colors = np.zeros((len(starts), 3), dtype=np.uint8)
        arrow_colors[:, 0] = (normalized_mag * 255).astype(np.uint8)  # R
        arrow_colors[:, 2] = ((1 - normalized_mag) * 255).astype(np.uint8)  # B

        # Create line segments
        line_points = np.zeros((len(starts) * 2, 3))
        line_points[0::2] = starts
        line_points[1::2] = ends

        line_colors = np.zeros((len(starts) * 2, 3), dtype=np.uint8)
        line_colors[0::2] = arrow_colors
        line_colors[1::2] = arrow_colors

        velocity_lines = server.scene.add_line_segments(
            name="velocity_arrows",
            points=line_points,
            colors=line_colors,
            line_width=1.0,
        )

    @gui_data_source.on_update
    def _(_) -> None:
        update_point_cloud()
        if gui_show_velocity.value:
            update_velocity_arrows()

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()
        if gui_show_velocity.value:
            update_velocity_arrows()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()
        if gui_show_velocity.value:
            update_velocity_arrows()

    @gui_show_next_frame.on_click
    def _(_) -> None:
        if gui_frame_selector.value == "All":
            return
        selected_idx = int(gui_frame_selector.value)
        if gui_anchor.value:
            gui_anchor.value = False
            use_gt = gui_data_source.value == "GT" and gt_points_centered is not None
            if use_gt:
                current_pts_in_next = gt_pts_in_next_centered
                current_pts_mask = gt_pts_in_next_mask_flat
                current_conf = gt_conf_flat
            else:
                current_pts_in_next = pts_in_next_centered
                current_pts_mask = pts_in_next_mask_flat
                current_conf = conf_flat

            frame_mask = frame_indices_flat == selected_idx
            conf_mask = (current_conf >= np.percentile(current_conf, gui_points_conf.value)) & (current_conf > 0.1)
            if current_pts_mask is not None and current_pts_in_next is not None:
                combined_mask = frame_mask & current_pts_mask & conf_mask
                point_cloud.points = current_pts_in_next[combined_mask]
                point_cloud.colors = colors_flat[combined_mask]
        else:
            gui_anchor.value = True
            gui_frame_selector.value = str((selected_idx + 1) % S)

    @gui_show_prev_frame.on_click
    def _(_) -> None:
        if gui_frame_selector.value == "All":
            return
        selected_idx = int(gui_frame_selector.value)
        if gui_anchor.value:
            gui_anchor.value = False
            gui_frame_selector.value = str((selected_idx - 1) % S)
            selected_idx = int(gui_frame_selector.value)

            use_gt = gui_data_source.value == "GT" and gt_points_centered is not None
            if use_gt:
                current_pts_in_next = gt_pts_in_next_centered
                current_pts_mask = gt_pts_in_next_mask_flat
                current_conf = gt_conf_flat
            else:
                current_pts_in_next = pts_in_next_centered
                current_pts_mask = pts_in_next_mask_flat
                current_conf = conf_flat

            frame_mask = frame_indices_flat == selected_idx
            conf_mask = (current_conf >= np.percentile(current_conf, gui_points_conf.value)) & (current_conf > 0.1)
            if current_pts_mask is not None and current_pts_in_next is not None:
                combined_mask = frame_mask & current_pts_mask & conf_mask
                point_cloud.points = current_pts_in_next[combined_mask]
                point_cloud.colors = colors_flat[combined_mask]
        else:
            gui_anchor.value = True
            update_point_cloud()

    @gui_point_size.on_update
    def _(_) -> None:
        point_cloud.point_size = gui_point_size.value

    @gui_show_frames.on_update
    def _(_) -> None:
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    @gui_show_velocity.on_update
    def _(_) -> None:
        update_velocity_arrows()

    @gui_velocity_scale.on_update
    def _(_) -> None:
        if gui_show_velocity.value:
            update_velocity_arrows()

    @gui_velocity_subsample.on_update
    def _(_) -> None:
        if gui_show_velocity.value:
            update_velocity_arrows()

    # Add camera frames
    visualize_frames(cam_to_world, images)

    print("Viser server started!")
    print(f"Open http://localhost:{port} in your browser")

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


@hydra.main(
    version_base=None,
    config_path="config/waymo",
    config_name="infer_multi",
)
def main(cfg: OmegaConf):
    parser = argparse.ArgumentParser(description="Viser 3D Visualization Demo")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--data_idx", type=int, default=None, help="Scene index to visualize")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--conf_threshold", type=float, default=50.0, help="Confidence percentile threshold")
    parser.add_argument("--background_mode", action="store_true", help="Run in background mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args, _ = parser.parse_known_args()

    if args.debug or cfg.get("debug", False):
        import debugpy
        debugpy.listen(5698)
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model_path = args.checkpoint if args.checkpoint else cfg.model_path
    model = load_model(model_path, device, cfg)

    dataset = load_dataset(cfg.infer_dataset)

    idx = args.data_idx if args.data_idx is not None else cfg.single_idx

    with tf32_off():
        with torch.no_grad():
            scene_data = prepare_scene_data(model, dataset, idx, device)

    viser_wrapper(
        scene_data,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        background_mode=args.background_mode,
    )


if __name__ == "__main__":
    main()
