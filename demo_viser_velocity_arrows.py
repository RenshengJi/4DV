#!/usr/bin/env python3
"""
demo_viser_velocity_arrows.py - 使用箭头（直线）可视化velocity

与demo_viser_flow_gt.py不同，本文件使用箭头来表示每个点的velocity：
- 箭头起点：当前3D点位置
- 箭头方向：velocity的方向
- 箭头长度：velocity的大小
"""

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
# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/vggt'))

from vggt.vggt.models.vggt import VGGT
from vggt.vggt.utils.load_fn import load_and_preprocess_images
from vggt.vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map, unproject_depth_map_to_point_map_batch, homo_matrix_inverse
from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.training.loss import warp_pts3d_for_gt_velocity, depth_to_world_points
from dust3r.utils.image import scene_flow_to_rgb
from dust3r.utils.misc import tf32_off
from dust3r.inference import inference
from train import cut3r_batch_to_vggt
from accelerate.logging import get_logger


def load_model_fn(model_path, device):
    """加载VGGT模型用于预测"""
    print(f"Loading model from: {model_path}")

    model = VGGT(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        use_sky_token=True,
        sh_degree=0,
        use_gs_head=True,
        use_gs_head_velocity=True,
        use_gt_camera=True
    )

    checkpoint = torch.load(model_path, map_location="cpu")
    ckpt = checkpoint.get('model', checkpoint)
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)

    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model


def predict_with_model(model, vggt_batch, device):
    """使用模型预测depth和velocity"""
    print("Running model prediction...")

    with torch.no_grad(), tf32_off():
        preds = model(
            vggt_batch["images"],
            gt_extrinsics=vggt_batch.get("extrinsics"),
            gt_intrinsics=vggt_batch.get("intrinsics"),
        )

    # 提取预测的depth和velocity
    pred_depth = preds.get("depth")  # [B, S, H, W, 1] or [B, S, H, W]
    pred_velocity = preds.get("velocity")  # [B, S, H, W, 3]

    # 如果depth有额外的channel维度，squeeze掉
    if pred_depth is not None and pred_depth.dim() == 5:
        pred_depth = pred_depth.squeeze(-1)  # [B, S, H, W, 1] -> [B, S, H, W]

    print(f"Predicted depth shape: {pred_depth.shape if pred_depth is not None else None}")
    print(f"Predicted velocity shape: {pred_velocity.shape if pred_velocity is not None else None}")

    # 提取xyz_camera（相机坐标系下的3D点）
    xyz_camera = preds.get("xyz_camera")  # [B, S, H*W, 3]

    # 如果xyz_camera存在，需要reshape回 [B, S, H, W, 3]
    if xyz_camera is not None and pred_depth is not None:
        B, S, H, W = pred_depth.shape
        xyz_camera = xyz_camera.reshape(B, S, H, W, 3)
        print(f"Predicted xyz_camera shape: {xyz_camera.shape}")

    return {
        'depth': pred_depth,
        'velocity': pred_velocity,
        'xyz_camera': xyz_camera,  # 相机坐标系下的3D点
        'all_preds': preds
    }


def convert_views_to_tensors(views, device='cpu'):
    """
    将从dataset.__getitem__获取的views转换为tensor格式
    模拟DataLoader的collate功能，并添加batch维度
    """
    converted_views = []
    for view in views:
        converted_view = {}
        for key, value in view.items():
            if isinstance(value, np.ndarray):
                # 将numpy数组转换为tensor
                tensor_value = torch.from_numpy(value).to(device)
                # 为特定字段添加batch维度
                if key in ['img', 'depthmap', 'camera_intrinsics', 'camera_pose', 'valid_mask', 'pts3d', 'flowmap']:
                    # img: [3, H, W] -> [1, 3, H, W]
                    # depthmap: [H, W] -> [1, H, W]
                    # pts3d: [H, W, 3] -> [1, H, W, 3]
                    # camera_*: [N, N] -> [1, N, N]
                    # valid_mask: [H, W] -> [1, H, W]
                    # flowmap: [H, W, C] -> [1, H, W, C]
                    tensor_value = tensor_value.unsqueeze(0)
                converted_view[key] = tensor_value
            elif isinstance(value, torch.Tensor):
                tensor_value = value.to(device)
                # 为特定字段添加batch维度（如果还没有）
                if key in ['img', 'depthmap', 'camera_intrinsics', 'camera_pose', 'valid_mask', 'pts3d', 'flowmap']:
                    if key == 'img' and tensor_value.dim() == 3:  # [3, H, W]
                        tensor_value = tensor_value.unsqueeze(0)  # -> [1, 3, H, W]
                    elif key in ['depthmap', 'valid_mask'] and tensor_value.dim() == 2:  # [H, W]
                        tensor_value = tensor_value.unsqueeze(0)  # -> [1, H, W]
                    elif key in ['pts3d', 'flowmap'] and tensor_value.dim() == 3:  # [H, W, C]
                        tensor_value = tensor_value.unsqueeze(0)  # -> [1, H, W, C]
                    elif key in ['camera_intrinsics', 'camera_pose'] and tensor_value.dim() == 2:  # [N, N]
                        tensor_value = tensor_value.unsqueeze(0)  # -> [1, N, N]
                converted_view[key] = tensor_value
            else:
                # 保持其他类型不变
                converted_view[key] = value
        converted_views.append(converted_view)
    return converted_views


def compute_velocity_arrows_data(vggt_batch, device, pred_data=None, use_pred_depth=False, use_pred_velocity=False):
    """
    从GT flowmap或预测数据中提取velocity信息，构建箭头可视化所需的数据

    Args:
        vggt_batch: VGGT格式的batch数据
        device: 设备
        pred_data: 预测数据字典，包含'depth'和'velocity'
        use_pred_depth: 是否使用预测的depth（否则使用GT depth）
        use_pred_velocity: 是否使用预测的velocity（否则使用GT velocity）

    Returns:
        dict: {
            'frames_data': list of dict, 每个dict包含一帧的数据:
                {
                    'arrow_starts': numpy array [N_frame, 3] - 箭头起点（3D点位置）
                    'arrow_ends': numpy array [N_frame, 3] - 箭头终点（起点 + velocity）
                    'arrow_colors': numpy array [N_frame, 3] - 箭头颜色（从图像RGB提取）
                    'velocity_magnitude': numpy array [N_frame] - velocity的大小
                    'frame_idx': int - 帧索引
                }
            'scene_center': numpy array [3] - 场景中心点
            'max_magnitude': float - 最大velocity magnitude
            'total_arrows': int - 总箭头数量
        }
    """
    try:
        mode_str = f"{'pred' if use_pred_velocity else 'GT'} velocity + {'pred' if use_pred_depth else 'GT'} depth"
        print(f"Computing velocity arrows using {mode_str}...")

        # 获取GT数据
        gt_flowmaps = vggt_batch.get('flowmap')  # [B, S, H, W, 4]
        gt_depths = vggt_batch.get('depths')     # [B, S, H, W]
        intrinsics = vggt_batch.get('intrinsics')  # [B, S, 3, 3]
        extrinsics = vggt_batch.get('extrinsics')  # [B, S, 4, 4]
        world_points = vggt_batch.get('world_points')  # [B, S, H, W, 3]
        images = vggt_batch.get('images')  # [B, S, 3, H, W]
        point_masks = vggt_batch.get('point_masks')  # [B, S, H, W]

        # 选择使用的depth和3D points
        if use_pred_depth and pred_data is not None and pred_data.get('depth') is not None:
            depths = pred_data['depth']  # [B, S, H, W]
            print(f"Using predicted depth with shape: {depths.shape}")

            # 使用预测的xyz_camera并转换到世界坐标系
            pred_xyz_camera = pred_data.get('xyz_camera')  # [B, S, H, W, 3]
            if pred_xyz_camera is not None and extrinsics is not None:
                print(f"Using predicted xyz_camera with shape: {pred_xyz_camera.shape}")
                B, S, H, W, _ = pred_xyz_camera.shape
                xyz_camera_flat = pred_xyz_camera.reshape(B, S, H * W, 3)

                # 添加齐次坐标
                ones = torch.ones(B, S, H * W, 1, device=xyz_camera_flat.device)
                xyz_camera_homo = torch.cat([xyz_camera_flat, ones], dim=-1)  # [B, S, H*W, 4]

                # 对每一帧应用extrinsic变换
                pred_world_points = []
                for s in range(S):
                    world_pts = torch.matmul(extrinsics[0, s], xyz_camera_homo[0, s].T).T  # [H*W, 4]
                    pred_world_points.append(world_pts[:, :3])  # 取前3维 [H*W, 3]

                pred_world_points = torch.stack(pred_world_points, dim=0)  # [S, H*W, 3]
                pred_world_points = pred_world_points.reshape(S, H, W, 3).unsqueeze(0)  # [1, S, H, W, 3]
                world_points = pred_world_points
                print(f"Converted predicted world_points shape: {world_points.shape}")
            else:
                print("Warning: pred_xyz_camera not available, will compute from depth")
                world_points = None
        else:
            depths = gt_depths
            print(f"Using GT depth")

        # 选择使用的velocity
        if use_pred_velocity and pred_data is not None and pred_data.get('velocity') is not None:
            velocities = pred_data['velocity']  # [B, S, H, W, 3]
            print(f"Using predicted velocity with shape: {velocities.shape}")
        else:
            # 从GT flowmap提取velocity (前3维)
            velocities = gt_flowmaps[..., :3] if gt_flowmaps is not None else None  # [B, S, H, W, 3]
            print(f"Using GT velocity from flowmap")

        if velocities is None:
            print("Error: No velocity data available")
            return None

        if depths is None or intrinsics is None or extrinsics is None:
            print("Warning: Missing data for position computation")
            return None

        # 获取数据维度
        B, S, H, W = depths.shape
        print(f"Processing data with shape: B={B}, S={S}, H={H}, W={W}")

        frames_data = []
        all_arrow_starts = []
        all_arrow_ends = []
        global_max_magnitude = 0.0

        # 逐帧处理
        for frame_idx in range(S):
            print(f"Processing frame {frame_idx}...")

            # 获取当前帧的velocity和depth
            frame_velocity = velocities[0, frame_idx]  # [H, W, 3]
            frame_depth = depths[0, frame_idx]  # [H, W]
            frame_image = images[0, frame_idx] if images is not None else None  # [3, H, W]

            # 决定使用mask策略
            if use_pred_depth and use_pred_velocity:
                # 密集输出：使用所有预测点
                print(f"Frame {frame_idx}: Using dense prediction (all points)")
                combined_mask = torch.ones(H, W, dtype=torch.bool, device=device)
            else:
                # 只展示激光雷达点（GT depth存在的点）
                if gt_depths is not None:
                    gt_depth_mask = gt_depths[0, frame_idx] > 0  # [H, W]
                elif point_masks is not None:
                    gt_depth_mask = point_masks[0, frame_idx]  # [H, W]
                else:
                    print("Warning: No GT depth mask available, using all points")
                    gt_depth_mask = torch.ones(H, W, dtype=torch.bool, device=device)

                # 如果使用GT velocity，还需要检查flowmap的第4维
                if not use_pred_velocity and gt_flowmaps is not None:
                    gt_velocity_mask = gt_flowmaps[0, frame_idx, :, :, 3] != 0  # [H, W]
                    combined_mask = gt_depth_mask & gt_velocity_mask
                else:
                    combined_mask = gt_depth_mask

            # 获取有效位置的索引
            valid_indices = torch.nonzero(combined_mask, as_tuple=True)  # (h_idx, w_idx)
            valid_h, valid_w = valid_indices

            if len(valid_h) == 0:
                print(f"Warning: No valid points found in frame {frame_idx}")
                continue

            print(f"Frame {frame_idx}: Found {len(valid_h)} valid points")

            # 提取有效位置的3D velocity
            velocity_3d = frame_velocity[valid_h, valid_w]  # [N, 3]

            # 提取有效位置的RGB颜色
            if frame_image is not None:
                # frame_image是[3, H, W]格式，需要转置为[H, W, 3]
                frame_image_hwc = frame_image.permute(1, 2, 0)  # [H, W, 3]
                point_colors = frame_image_hwc[valid_h, valid_w]  # [N, 3]
            else:
                # 如果没有图像数据，使用白色作为默认颜色
                point_colors = torch.ones(len(valid_h), 3, device=device)

            # 获取对应位置的3D点坐标（箭头起点）
            if world_points is not None:
                arrow_starts = world_points[0, frame_idx, valid_h, valid_w]  # [N, 3]
            else:
                # 如果没有world_points，从depth重新计算
                print("Warning: No world_points available, computing from depth...")
                frame_intrinsics = intrinsics[0, frame_idx]  # [3, 3]

                # 创建像素坐标网格
                pixel_coords = torch.stack([
                    valid_w.float(),  # x coordinates
                    valid_h.float(),  # y coordinates
                    torch.ones_like(valid_w.float())  # homogeneous coordinates
                ], dim=1)  # [N, 3]

                # 反投影到相机坐标系
                valid_depths = frame_depth[valid_h, valid_w].unsqueeze(1)  # [N, 1]
                camera_coords = torch.matmul(torch.linalg.inv(frame_intrinsics), pixel_coords.T).T * valid_depths  # [N, 3]

                arrow_starts = camera_coords

            # 计算箭头终点：起点 + velocity
            arrow_ends = arrow_starts + velocity_3d

            # 计算velocity magnitude
            velocity_magnitude = torch.norm(velocity_3d, dim=1)
            frame_max_magnitude = velocity_magnitude.max() if len(velocity_magnitude) > 0 and velocity_magnitude.max() > 0 else 1.0
            global_max_magnitude = max(global_max_magnitude, frame_max_magnitude)

            # 保存帧数据
            frames_data.append({
                'arrow_starts': arrow_starts,
                'arrow_ends': arrow_ends,
                'arrow_colors': point_colors,
                'velocity_magnitude': velocity_magnitude,
                'frame_idx': frame_idx
            })

            # 收集所有点用于计算场景中心
            all_arrow_starts.append(arrow_starts)
            all_arrow_ends.append(arrow_ends)

        if len(frames_data) == 0:
            print("Warning: No valid frames found")
            return None

        # 计算场景中心
        all_starts_cat = torch.cat(all_arrow_starts, dim=0)
        all_ends_cat = torch.cat(all_arrow_ends, dim=0)
        all_points = torch.cat([all_starts_cat, all_ends_cat], dim=0)
        scene_center = all_points.mean(dim=0)

        # 最终化数据
        final_frames_data = []
        total_arrows = 0

        for frame_data in frames_data:
            arrow_colors = frame_data['arrow_colors']
            N_valid = len(arrow_colors)
            total_arrows += N_valid

            if N_valid == 0:
                continue

            # 确保颜色值在[0,1]范围内
            if arrow_colors.max() > 1.0:
                arrow_colors = torch.clamp(arrow_colors / 255.0, 0, 1)
            else:
                arrow_colors = torch.clamp(arrow_colors, 0, 1)

            # 将点重新定位到场景中心
            centered_starts = frame_data['arrow_starts'] - scene_center
            centered_ends = frame_data['arrow_ends'] - scene_center

            # 保存最终的帧数据
            final_frames_data.append({
                'arrow_starts': centered_starts.cpu().numpy(),
                'arrow_ends': centered_ends.cpu().numpy(),
                'arrow_colors': arrow_colors.cpu().numpy(),
                'velocity_magnitude': frame_data['velocity_magnitude'].cpu().numpy(),
                'frame_idx': frame_data['frame_idx']
            })

        # 构造最终结果
        result = {
            'frames_data': final_frames_data,
            'scene_center': scene_center.cpu().numpy(),
            'max_magnitude': global_max_magnitude.cpu().item() if isinstance(global_max_magnitude, torch.Tensor) else global_max_magnitude,
            'total_arrows': total_arrows
        }

        print(f"Computed velocity arrows for {len(final_frames_data)} frames")
        print(f"Total arrows: {total_arrows}")
        print(f"Global max velocity magnitude: {result['max_magnitude']:.6f}")

        return result

    except Exception as e:
        print(f"Error in compute_velocity_arrows_data: {e}")
        import traceback
        traceback.print_exc()
        return None


def viser_wrapper_velocity_arrows(
    arrow_data: dict,
    port: int = 8080,
    init_display_ratio: float = 50.0,
    background_mode: bool = False,
):
    """
    使用viser可视化velocity箭头

    Args:
        arrow_data (dict): 包含frames_data等的字典
        port (int): 端口号
        init_display_ratio (float): 初始显示比例（百分比）
        background_mode (bool): 是否后台运行
    """
    print(f"Starting viser velocity arrows server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # 解包数据
    frames_data = arrow_data['frames_data']
    max_magnitude = arrow_data['max_magnitude']
    total_arrows = arrow_data['total_arrows']

    num_frames = len(frames_data)
    print(f"Visualizing {num_frames} frames with {total_arrows} total velocity arrows")

    # 构建GUI控件
    gui_frame_selector = server.gui.add_dropdown(
        "Select Frame",
        options=["All"] + [f"Frame {frame_data['frame_idx']}" for frame_data in frames_data],
        initial_value="All",
    )

    gui_display_ratio = server.gui.add_slider(
        "Display Ratio (%)",
        min=1,
        max=100,
        step=1,
        initial_value=int(init_display_ratio),
    )

    gui_arrow_width = server.gui.add_slider(
        "Arrow Width",
        min=0.001,
        max=0.05,
        step=0.001,
        initial_value=0.005,
    )

    gui_arrow_scale = server.gui.add_slider(
        "Arrow Length Scale",
        min=0.1,
        max=5.0,
        step=0.1,
        initial_value=1.0,
    )

    gui_show_points = server.gui.add_checkbox(
        "Show Start Points",
        initial_value=False,
    )

    gui_point_size = server.gui.add_slider(
        "Point Size",
        min=0.001,
        max=0.02,
        step=0.001,
        initial_value=0.005,
    )

    gui_reset_view = server.gui.add_button("Reset View")

    # 添加信息显示
    gui_info = server.gui.add_text(
        "Velocity Info",
        initial_value=f"Total Frames: {num_frames}\nTotal Arrows: {total_arrows}\nMax Velocity: {max_magnitude:.6f}m/s"
    )

    # 存储箭头和点云的句柄
    arrow_handles = []
    point_cloud_handle = None

    def get_current_frame_data():
        """获取当前选择的帧数据"""
        if gui_frame_selector.value == "All":
            # 合并所有帧的数据
            all_starts = []
            all_ends = []
            all_colors = []
            all_magnitudes = []

            for frame_data in frames_data:
                all_starts.append(frame_data['arrow_starts'])
                all_ends.append(frame_data['arrow_ends'])
                all_colors.append(frame_data['arrow_colors'])
                all_magnitudes.append(frame_data['velocity_magnitude'])

            return {
                'arrow_starts': np.concatenate(all_starts, axis=0),
                'arrow_ends': np.concatenate(all_ends, axis=0),
                'arrow_colors': np.concatenate(all_colors, axis=0),
                'velocity_magnitude': np.concatenate(all_magnitudes, axis=0),
            }
        else:
            # 选择特定帧
            frame_idx = int(gui_frame_selector.value.split()[-1])
            for frame_data in frames_data:
                if frame_data['frame_idx'] == frame_idx:
                    return frame_data
            return frames_data[0]  # fallback

    def update_visualization():
        """更新箭头和点云显示"""
        nonlocal arrow_handles, point_cloud_handle

        # 移除旧的箭头
        for handle in arrow_handles:
            handle.remove()
        arrow_handles = []

        # 移除旧的点云
        if point_cloud_handle is not None:
            point_cloud_handle.remove()
            point_cloud_handle = None

        # 获取当前帧数据
        current_data = get_current_frame_data()
        arrow_starts = current_data['arrow_starts']
        arrow_ends = current_data['arrow_ends']
        arrow_colors = current_data['arrow_colors']
        velocity_magnitude = current_data['velocity_magnitude']

        N = len(arrow_starts)
        if N == 0:
            return

        # 计算要显示的箭头数量
        display_ratio = gui_display_ratio.value / 100.0
        num_display = max(1, int(N * display_ratio))

        # 根据velocity magnitude选择要显示的箭头（显示magnitude较大的）
        if num_display < N:
            sorted_indices = np.argsort(velocity_magnitude)[::-1]  # 降序排列
            display_indices = sorted_indices[:num_display]
        else:
            display_indices = np.arange(N)

        display_starts = arrow_starts[display_indices]
        display_ends = arrow_ends[display_indices]
        display_colors = arrow_colors[display_indices]

        # 应用箭头长度缩放
        arrow_scale = gui_arrow_scale.value
        if arrow_scale != 1.0:
            # 计算scaled的终点：start + scale * (end - start)
            display_ends = display_starts + arrow_scale * (display_ends - display_starts)

        # 将颜色转换为0-255范围
        display_colors_uint8 = (display_colors * 255).astype(np.uint8)

        # 添加箭头
        arrow_width = gui_arrow_width.value
        for i in range(len(display_starts)):
            start = display_starts[i]
            end = display_ends[i]
            color_rgb = tuple(display_colors_uint8[i])

            # 使用viser添加箭头（使用线段来模拟箭头）
            arrow_handle = server.scene.add_spline_catmull_rom(
                f"/arrows/arrow_{i}",
                positions=np.array([start, end]),
                color=color_rgb,
                line_width=arrow_width,
                segments=1,
            )
            arrow_handles.append(arrow_handle)

        # 如果启用，显示起点
        if gui_show_points.value:
            point_cloud_handle = server.scene.add_point_cloud(
                name="/points/start_points",
                points=display_starts,
                colors=display_colors_uint8,
                point_size=gui_point_size.value,
                point_shape="circle",
            )

        # 更新信息显示
        frame_info = f"Frame: {gui_frame_selector.value}"
        if gui_frame_selector.value != "All":
            frame_idx = int(gui_frame_selector.value.split()[-1])
            frame_info += f" ({N} arrows)"

        gui_info.value = f"{frame_info}\nDisplaying: {len(display_indices)}/{N} arrows\nMax Velocity: {max_magnitude:.6f}m/s\nArrow Scale: {arrow_scale:.1f}x"

    # GUI事件处理
    @gui_frame_selector.on_update
    def _(_) -> None:
        update_visualization()

    @gui_display_ratio.on_update
    def _(_) -> None:
        update_visualization()

    @gui_arrow_width.on_update
    def _(_) -> None:
        update_visualization()

    @gui_arrow_scale.on_update
    def _(_) -> None:
        update_visualization()

    @gui_show_points.on_update
    def _(_) -> None:
        update_visualization()

    @gui_point_size.on_update
    def _(_) -> None:
        if gui_show_points.value:
            update_visualization()

    @gui_reset_view.on_click
    def _(_) -> None:
        """重置视图"""
        gui_frame_selector.value = "All"
        gui_display_ratio.value = 50
        gui_arrow_scale.value = 1.0
        update_visualization()

    # 初始化显示
    update_visualization()

    print("Starting viser velocity arrows server...")
    # 如果background_mode是True，在守护线程中运行
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VGGT velocity visualization with arrows")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train_full",
        help="Path to dataset root directory"
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint (optional, for prediction)")
    parser.add_argument("--use_pred_depth", action="store_true", help="Use predicted depth instead of GT depth")
    parser.add_argument("--use_pred_velocity", action="store_true", help="Use predicted velocity instead of GT velocity")
    parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
    parser.add_argument("--port", type=int, default=8082, help="Port number for the viser server")
    parser.add_argument(
        "--display_ratio", type=float, default=50.0, help="Initial percentage of arrows to display"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for additional logging")
    parser.add_argument("--num_views", type=int, default=8, help="Number of views for inference")
    parser.add_argument("--idx", type=int, default=9600, help="Index of the sequence to process")

    args = parser.parse_args()

    # 验证参数
    if (args.use_pred_depth or args.use_pred_velocity) and args.model_path is None:
        print("Error: --model_path is required when using --use_pred_depth or --use_pred_velocity")
        return

    # 显示当前模式
    mode_str = f"{'Pred' if args.use_pred_velocity else 'GT'} velocity + {'Pred' if args.use_pred_depth else 'GT'} depth"
    print(f"\n{'='*60}")
    print(f"Running in mode: {mode_str}")

    # 根据模式说明点的显示策略
    if args.use_pred_depth and args.use_pred_velocity:
        print(f"Dense output: Displaying all predicted points")
    else:
        print(f"Sparse output: Only displaying points where GT depth exists (LiDAR points)")
    print(f"{'='*60}\n")

    if args.debug:
        import debugpy
        debugpy.listen(("localhost", 5697))
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载数据集
    print(f"Loading dataset from: {args.dataset_root}")

    # 创建数据集对象
    from dust3r.datasets.waymo import Waymo_Multi

    dataset = Waymo_Multi(
        split=None,
        ROOT=args.dataset_root,
        img_ray_mask_p=[1.0, 0.0, 0.0],
        valid_camera_id_list=["1","2","3"],
        resolution=[(518, 378), (518, 336), (518, 294), (518, 252), (518, 210),
                    (518, 140), (378, 518), (336, 518), (294, 518), (252, 518)],
        num_views=args.num_views,
        seed=42,
        n_corres=0,
        seq_aug_crop=True
    )

    # 准备输入视图
    print("Preparing input views...")
    idx = args.idx
    num_views = args.num_views

    # 检查数据集大小
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size}")

    if idx >= dataset_size:
        print(f"Warning: idx {idx} >= dataset_size {dataset_size}, using idx 0")
        idx = 0

    try:
        views = dataset.__getitem__((idx, 2, num_views))
        print(f"Successfully loaded {len(views)} views for idx {idx}")

        # 检查flowmap是否存在
        has_flowmap = any('flowmap' in view and view['flowmap'] is not None for view in views)
        print(f"GT flowmap data available: {has_flowmap}")

        if not has_flowmap:
            print("Error: No GT flowmap data found in views. Please check if flowmap data is available.")
            return

    except Exception as e:
        print(f"Error loading views for idx {idx}: {e}")
        # 尝试使用idx 0
        if idx != 0:
            print("Trying idx 0...")
            idx = 0
            views = dataset.__getitem__((idx, 2, num_views))
        else:
            raise e

    # 将views转换为tensor格式
    print("Converting views to tensors...")
    views = convert_views_to_tensors(views, device)

    # 转换为VGGT格式的batch
    print("Converting to VGGT batch format...")
    vggt_batch = cut3r_batch_to_vggt(views)

    # 确保batch在正确的设备上
    for key, value in vggt_batch.items():
        if isinstance(value, torch.Tensor):
            vggt_batch[key] = value.to(device)

    # 如果需要预测，加载模型并运行预测
    pred_data = None
    if args.model_path is not None:
        model = load_model_fn(args.model_path, device)
        pred_data = predict_with_model(model, vggt_batch, device)

    # 计算velocity箭头数据
    print("Computing velocity arrows data...")
    arrow_data = compute_velocity_arrows_data(
        vggt_batch,
        device,
        pred_data=pred_data,
        use_pred_depth=args.use_pred_depth,
        use_pred_velocity=args.use_pred_velocity
    )

    if arrow_data is None:
        print("Failed to compute velocity arrows data. Exiting.")
        return

    print("Starting viser velocity arrows visualization...")
    viser_server = viser_wrapper_velocity_arrows(
        arrow_data,
        port=args.port,
        init_display_ratio=args.display_ratio,
        background_mode=args.background_mode,
    )
    print("Velocity arrows visualization complete")


if __name__ == "__main__":
    main()
