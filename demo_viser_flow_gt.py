#!/usr/bin/env python3
"""
demo_viser_flow_gt.py - 使用Ground Truth Flowmap数据进行3D velocity可视化

基于demo_viser_flow.py的实现，展示从GT flowmap数据中提取的3D velocity flow，
flowmap的最后一个维度为非0表示有GT velocity，且有GT velocity的pixel一定有depth。
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


def load_model(model_path, device):
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


def compute_gt_flow_and_velocity_points(vggt_batch, device, pred_data=None, use_pred_depth=False, use_pred_velocity=False):
    """
    从GT flowmap数据中提取3D velocity points，返回按帧组织的valid_current_pts和valid_warped_pts数据
    与demo_viser_flow.py保持一致的数据结构

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
                    'valid_current_pts': numpy array [N_frame, 3] - 当前帧的有效3D点
                    'valid_warped_pts': numpy array [N_frame, 3] - 通过velocity计算的warped 3D点
                    'velocity_colors': numpy array [N_frame, 3] - 点的原始RGB颜色（从图像中提取）
                    'velocity_magnitude': numpy array [N_frame] - velocity的magnitude
                    'frame_idx': int - 帧索引
                }
            'scene_center': numpy array [3] - 场景中心点用于重新定位
            'max_magnitude': float - 所有帧中velocity的最大magnitude
            'total_points': int - 所有帧的总点数
        }
    """
    try:
        mode_str = f"{'pred' if use_pred_velocity else 'GT'} velocity + {'pred' if use_pred_depth else 'GT'} depth"
        print(f"Computing flow and velocity points using {mode_str}...")

        # 获取GT数据
        gt_flowmaps = vggt_batch.get('flowmap')  # [B, S, H, W, 4]
        gt_depths = vggt_batch.get('depths')     # [B, S, H, W]
        intrinsics = vggt_batch.get('intrinsics')  # [B, S, 3, 3]
        extrinsics = vggt_batch.get('extrinsics')  # [B, S, 4, 4]
        world_points = vggt_batch.get('world_points')  # [B, S, H, W, 3]
        images = vggt_batch.get('images')  # [B, S, 3, H, W] - 用于获取原始RGB颜色
        point_masks = vggt_batch.get('point_masks')  # [B, S, H, W]

        # 选择使用的depth和3D points
        if use_pred_depth and pred_data is not None and pred_data.get('depth') is not None:
            depths = pred_data['depth']  # [B, S, H, W]
            print(f"Using predicted depth with shape: {depths.shape}")

            # 使用预测的xyz_camera（相机坐标系下的3D点）并转换到世界坐标系
            pred_xyz_camera = pred_data.get('xyz_camera')  # [B, S, H, W, 3]
            if pred_xyz_camera is not None and extrinsics is not None:
                print(f"Using predicted xyz_camera with shape: {pred_xyz_camera.shape}")
                # 将相机坐标系下的点转换到世界坐标系
                # world_pt = extrinsic @ [camera_pt; 1]
                B, S, H, W, _ = pred_xyz_camera.shape
                # Reshape to [B, S, H*W, 3]
                xyz_camera_flat = pred_xyz_camera.reshape(B, S, H * W, 3)

                # 添加齐次坐标
                ones = torch.ones(B, S, H * W, 1, device=xyz_camera_flat.device)
                xyz_camera_homo = torch.cat([xyz_camera_flat, ones], dim=-1)  # [B, S, H*W, 4]

                # 对每一帧应用extrinsic变换
                pred_world_points = []
                for s in range(S):
                    # extrinsics[0, s] 是 [4, 4]
                    # xyz_camera_homo[0, s] 是 [H*W, 4]
                    world_pts = torch.matmul(extrinsics[0, s], xyz_camera_homo[0, s].T).T  # [H*W, 4]
                    pred_world_points.append(world_pts[:, :3])  # 取前3维 [H*W, 3]

                # Stack并reshape回 [B, S, H, W, 3]
                pred_world_points = torch.stack(pred_world_points, dim=0)  # [S, H*W, 3]
                pred_world_points = pred_world_points.reshape(S, H, W, 3).unsqueeze(0)  # [1, S, H, W, 3]
                world_points = pred_world_points
                print(f"Converted predicted world_points shape: {world_points.shape}")
            else:
                print("Warning: pred_xyz_camera not available, will compute from depth")
                world_points = None  # 将在后面从depth重新计算
        else:
            depths = gt_depths
            print(f"Using GT depth")

        # 选择使用的velocity（从flowmap或pred）
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
        all_current_pts = []
        all_warped_pts = []
        global_max_magnitude = 0.0

        # 逐帧处理
        for frame_idx in range(S):
            print(f"Processing frame {frame_idx}...")

            # 获取当前帧的velocity
            frame_velocity = velocities[0, frame_idx]  # [H, W, 3]
            frame_depth = depths[0, frame_idx]  # [H, W]
            frame_image = images[0, frame_idx] if images is not None else None  # [3, H, W]

            # 决定使用mask策略：
            # - 如果同时使用pred depth和pred velocity，则输出密集结果（不使用mask）
            # - 否则只展示激光雷达点（使用GT depth mask）
            if use_pred_depth and use_pred_velocity:
                # 密集输出：使用所有预测点
                print(f"Frame {frame_idx}: Using dense prediction (all points)")
                combined_mask = torch.ones(H, W, dtype=torch.bool, device=device)
            else:
                # 只展示激光雷达点（GT depth存在的点）
                if gt_depths is not None:
                    gt_depth_mask = gt_depths[0, frame_idx] > 0  # [H, W] - GT depth存在的点
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

            # 获取对应位置的3D点坐标
            if world_points is not None:
                # 使用已经转换好的world_points
                valid_current_pts = world_points[0, frame_idx, valid_h, valid_w]  # [N, 3]
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

                # 转换到世界坐标系（如果需要）
                valid_current_pts = camera_coords

            # 计算warped points：current_pts + velocity
            valid_warped_pts = valid_current_pts + velocity_3d

            # 计算velocity magnitude
            velocity_magnitude = torch.norm(velocity_3d, dim=1)
            frame_max_magnitude = velocity_magnitude.max() if len(velocity_magnitude) > 0 and velocity_magnitude.max() > 0 else 1.0
            global_max_magnitude = max(global_max_magnitude, frame_max_magnitude)

            # 保存帧数据
            frames_data.append({
                'valid_current_pts': valid_current_pts,
                'valid_warped_pts': valid_warped_pts,
                'gt_velocity': velocity_3d,
                'velocity_magnitude': velocity_magnitude,
                'point_colors': point_colors,  # 添加RGB颜色
                'frame_idx': frame_idx
            })

            # 收集所有点用于计算场景中心
            all_current_pts.append(valid_current_pts)
            all_warped_pts.append(valid_warped_pts)

        if len(frames_data) == 0:
            print("Warning: No valid GT frames found")
            return None

        # 计算场景中心
        all_current_pts_cat = torch.cat(all_current_pts, dim=0)
        all_warped_pts_cat = torch.cat(all_warped_pts, dim=0)
        all_points = torch.cat([all_current_pts_cat, all_warped_pts_cat], dim=0)
        scene_center = all_points.mean(dim=0)

        # 为每帧生成颜色并最终化数据
        final_frames_data = []
        total_points = 0

        for frame_data in frames_data:
            gt_velocity = frame_data['gt_velocity']
            velocity_magnitude = frame_data['velocity_magnitude']
            point_colors = frame_data['point_colors']  # 获取RGB颜色
            N_valid = len(gt_velocity)
            total_points += N_valid

            if N_valid == 0:
                continue

            # 使用原始的RGB颜色，确保颜色值在[0,1]范围内
            if point_colors is not None:
                # 确保颜色值在[0,1]范围内（图像可能已经在[0,1]范围内，或在[0,255]范围内）
                if point_colors.max() > 1.0:
                    velocity_colors = torch.clamp(point_colors / 255.0, 0, 1)
                else:
                    velocity_colors = torch.clamp(point_colors, 0, 1)
            else:
                # 如果没有颜色数据，使用基于velocity的颜色作为备选
                velocity_colors = torch.zeros(N_valid, 3, device=device)
                velocity_colors[:, 0] = torch.clamp((gt_velocity[:, 0] / global_max_magnitude * 0.5 + 0.5), 0, 1)  # R
                velocity_colors[:, 1] = torch.clamp((gt_velocity[:, 1] / global_max_magnitude * 0.5 + 0.5), 0, 1)  # G
                velocity_colors[:, 2] = torch.clamp((gt_velocity[:, 2] / global_max_magnitude * 0.5 + 0.5), 0, 1)  # B

            # 将点重新定位到场景中心
            centered_current_pts = frame_data['valid_current_pts'] - scene_center
            centered_warped_pts = frame_data['valid_warped_pts'] - scene_center

            # 保存最终的帧数据
            final_frames_data.append({
                'valid_current_pts': centered_current_pts.cpu().numpy(),
                'valid_warped_pts': centered_warped_pts.cpu().numpy(),
                'velocity_colors': velocity_colors.cpu().numpy(),
                'velocity_magnitude': velocity_magnitude.cpu().numpy(),
                'frame_idx': frame_data['frame_idx']
            })

        # 构造最终结果
        result = {
            'frames_data': final_frames_data,
            'scene_center': scene_center.cpu().numpy(),
            'max_magnitude': global_max_magnitude.cpu().item() if isinstance(global_max_magnitude, torch.Tensor) else global_max_magnitude,
            'total_points': total_points
        }

        print(f"Computed GT flow points for {len(final_frames_data)} frames")
        print(f"Total GT points: {total_points}")
        print(f"Global max GT velocity magnitude: {result['max_magnitude']:.6f}")

        return result

    except Exception as e:
        print(f"Error in compute_gt_flow_and_velocity_points: {e}")
        import traceback
        traceback.print_exc()
        return None


def viser_wrapper_gt_flow(
    flow_data: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,
    background_mode: bool = False,
):
    """
    使用viser可视化GT flow数据，与demo_viser_flow.py保持一致的展示逻辑

    Args:
        flow_data (dict): 包含frames_data等的字典
        port (int): 端口号
        init_conf_threshold (float): 初始置信度阈值（这里用作点的显示比例）
        background_mode (bool): 是否后台运行
    """
    print(f"Starting viser GT flow server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # 解包流数据
    frames_data = flow_data['frames_data']
    max_magnitude = flow_data['max_magnitude']
    total_points = flow_data['total_points']

    num_frames = len(frames_data)
    print(f"Visualizing {num_frames} GT frames with {total_points} total valid GT flow points")

    # 构建GUI控件 - 与demo_viser_flow.py保持一致
    gui_frame_selector = server.gui.add_dropdown(
        "Select Frame",
        options=["All"] + [f"Frame {frame_data['frame_idx']}" for frame_data in frames_data],
        initial_value="All",
    )

    gui_show_current = server.gui.add_checkbox(
        "Show Current Points",
        initial_value=True,
    )

    gui_show_warped = server.gui.add_checkbox(
        "Show Warped Points",
        initial_value=False,
    )

    gui_points_threshold = server.gui.add_slider(
        "Points Display Ratio (%)",
        min=1,
        max=100,
        step=1,
        initial_value=int(init_conf_threshold),
    )

    gui_point_size = server.gui.add_slider(
        "Point Size",
        min=0.0001,
        max=0.05,
        step=0.0001,
        initial_value=0.005,
    )

    gui_flow_animation = server.gui.add_button("Animate GT Flow")
    gui_reset_view = server.gui.add_button("Reset to Current")

    # 添加信息显示
    gui_info = server.gui.add_text(
        "GT Flow Info",
        initial_value=f"Total GT Frames: {num_frames}\nTotal GT Points: {total_points}\nMax GT Velocity: {max_magnitude:.6f}m"
    )

    # 创建点云句柄
    current_point_cloud = None
    warped_point_cloud = None

    def get_current_frame_data():
        """获取当前选择的帧数据"""
        if gui_frame_selector.value == "All":
            # 合并所有帧的数据
            all_current_pts = []
            all_warped_pts = []
            all_colors = []
            all_magnitudes = []

            for frame_data in frames_data:
                all_current_pts.append(frame_data['valid_current_pts'])
                all_warped_pts.append(frame_data['valid_warped_pts'])
                all_colors.append(frame_data['velocity_colors'])
                all_magnitudes.append(frame_data['velocity_magnitude'])

            return {
                'valid_current_pts': np.concatenate(all_current_pts, axis=0),
                'valid_warped_pts': np.concatenate(all_warped_pts, axis=0),
                'velocity_colors': np.concatenate(all_colors, axis=0),
                'velocity_magnitude': np.concatenate(all_magnitudes, axis=0),
            }
        else:
            # 选择特定帧
            frame_idx = int(gui_frame_selector.value.split()[-1])
            for frame_data in frames_data:
                if frame_data['frame_idx'] == frame_idx:
                    return frame_data
            return frames_data[0]  # fallback

    def update_point_clouds():
        """更新点云显示"""
        nonlocal current_point_cloud, warped_point_cloud

        # 获取当前帧数据
        current_data = get_current_frame_data()
        valid_current_pts = current_data['valid_current_pts']
        valid_warped_pts = current_data['valid_warped_pts']
        velocity_colors = current_data['velocity_colors']
        velocity_magnitude = current_data['velocity_magnitude']

        N = len(valid_current_pts)
        if N == 0:
            return

        # 将颜色转换为0-255范围
        velocity_colors_uint8 = (velocity_colors * 255).astype(np.uint8)

        # 计算要显示的点的数量
        display_ratio = gui_points_threshold.value / 100.0
        num_display = max(1, int(N * display_ratio))

        # 根据velocity magnitude选择要显示的点（显示magnitude较大的点）
        if num_display < N:
            sorted_indices = np.argsort(velocity_magnitude)[::-1]  # 降序排列
            display_indices = sorted_indices[:num_display]
        else:
            display_indices = np.arange(N)

        display_current_pts = valid_current_pts[display_indices]
        display_warped_pts = valid_warped_pts[display_indices]
        display_colors = velocity_colors_uint8[display_indices]

        # 移除旧的点云
        if current_point_cloud is not None:
            current_point_cloud.remove()
            current_point_cloud = None
        if warped_point_cloud is not None:
            warped_point_cloud.remove()
            warped_point_cloud = None

        # 添加当前点云
        if gui_show_current.value:
            current_point_cloud = server.scene.add_point_cloud(
                name="current_points",
                points=display_current_pts,
                colors=display_colors,
                point_size=gui_point_size.value,
                point_shape="circle",
            )

        # 添加warped点云
        if gui_show_warped.value:
            warped_point_cloud = server.scene.add_point_cloud(
                name="warped_points",
                points=display_warped_pts,
                colors=display_colors,
                point_size=gui_point_size.value,
                point_shape="circle",
            )

        # 更新信息显示
        frame_info = f"Frame: {gui_frame_selector.value}"
        if gui_frame_selector.value != "All":
            frame_idx = int(gui_frame_selector.value.split()[-1])
            frame_info += f" ({N} GT points)"

        gui_info.value = f"{frame_info}\nDisplaying: {len(display_indices)}/{N} GT points\nMax GT Velocity: {max_magnitude:.6f}m"

    def animate_flow():
        """动画显示从current到warped的GT流动"""
        print("Starting GT flow animation...")

        # 获取当前帧数据
        current_data = get_current_frame_data()
        valid_current_pts = current_data['valid_current_pts']
        valid_warped_pts = current_data['valid_warped_pts']
        velocity_colors = current_data['velocity_colors']
        velocity_magnitude = current_data['velocity_magnitude']

        N = len(valid_current_pts)
        if N == 0:
            print("No GT points to animate")
            return

        velocity_colors_uint8 = (velocity_colors * 255).astype(np.uint8)

        # 计算要显示的点
        display_ratio = gui_points_threshold.value / 100.0
        num_display = max(1, int(N * display_ratio))

        if num_display < N:
            sorted_indices = np.argsort(velocity_magnitude)[::-1]
            display_indices = sorted_indices[:num_display]
        else:
            display_indices = np.arange(N)

        display_current_pts = valid_current_pts[display_indices]
        display_warped_pts = valid_warped_pts[display_indices]
        display_colors = velocity_colors_uint8[display_indices]

        # 移除现有点云
        if current_point_cloud is not None:
            current_point_cloud.remove()
        if warped_point_cloud is not None:
            warped_point_cloud.remove()

        # 创建动画
        animation_steps = 50
        for step in range(animation_steps + 1):
            alpha = step / animation_steps
            # 线性插值
            interpolated_pts = display_current_pts * (1 - alpha) + display_warped_pts * alpha

            # 移除之前的动画点云
            try:
                server.scene.remove("gt_animation_points")
            except:
                pass

            # 添加新的动画点云
            server.scene.add_point_cloud(
                name="gt_animation_points",
                points=interpolated_pts,
                colors=display_colors,
                point_size=gui_point_size.value,
                point_shape="circle",
            )

            time.sleep(0.05)  # 50ms间隔

        # 动画结束后恢复正常显示
        try:
            server.scene.remove("gt_animation_points")
        except:
            pass
        update_point_clouds()
        print("GT flow animation completed")

    # GUI事件处理 - 与demo_viser_flow.py保持一致
    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_clouds()

    @gui_show_current.on_update
    def _(_) -> None:
        update_point_clouds()

    @gui_show_warped.on_update
    def _(_) -> None:
        update_point_clouds()

    @gui_points_threshold.on_update
    def _(_) -> None:
        update_point_clouds()

    @gui_point_size.on_update
    def _(_) -> None:
        update_point_clouds()

    @gui_flow_animation.on_click
    def _(_) -> None:
        """启动GT流动画动画"""
        # 在单独的线程中运行动画，避免阻塞GUI
        def run_animation():
            animate_flow()

        import threading
        animation_thread = threading.Thread(target=run_animation, daemon=True)
        animation_thread.start()

    @gui_reset_view.on_click
    def _(_) -> None:
        """重置为显示当前点"""
        gui_show_current.value = True
        gui_show_warped.value = False
        update_point_clouds()

    # 初始化显示
    update_point_clouds()

    print("Starting viser GT flow server...")
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
    parser = argparse.ArgumentParser(description="VGGT GT flow demo with viser for 3D GT velocity visualization")
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
    parser.add_argument("--port", type=int, default=8081, help="Port number for the viser server")
    parser.add_argument(
        "--conf_threshold", type=float, default=100.0, help="Initial percentage of points to display"
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

    # 将views转换为tensor格式（模拟DataLoader的collate功能）
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
        model = load_model(args.model_path, device)
        pred_data = predict_with_model(model, vggt_batch, device)

    # 计算flow和velocity点（支持4种模式）
    print("Computing flow and velocity points...")
    flow_data = compute_gt_flow_and_velocity_points(
        vggt_batch,
        device,
        pred_data=pred_data,
        use_pred_depth=args.use_pred_depth,
        use_pred_velocity=args.use_pred_velocity
    )

    if flow_data is None:
        print("Failed to compute GT flow data. Exiting.")
        return

    print("Starting viser GT flow visualization...")
    viser_server = viser_wrapper_gt_flow(
        flow_data,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        background_mode=args.background_mode,
    )
    print("GT flow visualization complete")


if __name__ == "__main__":
    main()