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
# 添加vggt路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/vggt'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map, unproject_depth_map_to_point_map_batch, homo_matrix_inverse
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.training.loss import velocity_local_to_global, warp_pts3d_for_gt_velocity, depth_to_world_points
from dust3r.utils.image import scene_flow_to_rgb
from dust3r.utils.misc import tf32_off
from src.dust3r.inference import inference
from src.train import cut3r_batch_to_vggt
from accelerate.logging import get_logger


def ensure_tensor(data, device):
    """确保数据是tensor格式"""
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


def fix_views_data_types(views, device):
    """修复views中的数据类型，确保所有数组都是tensor"""
    fixed_views = []

    for i, view in enumerate(views):
        fixed_view = {}
        for key, value in view.items():
            if key in ['img', 'depthmap', 'camera_intrinsics', 'camera_pose', 'valid_mask', 'pts3d', 'flowmap']:
                # 这些字段需要是tensor
                if value is not None:
                    tensor_value = ensure_tensor(value, device)

                    # 特殊处理img字段，确保维度正确
                    if key == 'img':
                        if tensor_value.dim() == 3:  # [3, H, W]
                            tensor_value = tensor_value.unsqueeze(0)  # [1, 3, H, W]
                        elif tensor_value.dim() == 4 and tensor_value.shape[0] != 1:
                            print(f"Warning: img tensor has unexpected batch size {tensor_value.shape[0]}, taking first batch")
                            tensor_value = tensor_value[:1]  # [1, 3, H, W]

                        print(f"View {i}, img shape: {tensor_value.shape}")

                    # 特殊处理flowmap字段
                    if key == 'flowmap' and tensor_value is not None:
                        print(f"View {i}, flowmap shape: {tensor_value.shape}")

                    fixed_view[key] = tensor_value
                else:
                    fixed_view[key] = value
            else:
                # 其他字段保持原样
                fixed_view[key] = value
        fixed_views.append(fixed_view)

    return fixed_views


def safe_cut3r_batch_to_vggt(views, device):
    """参考src/train.py中cut3r_batch_to_vggt的正确实现"""
    try:
        print(f"Processing {len(views)} views")

        # 按照train.py的方式构建：先构建[S, B, ...]格式
        imgs = [v['img'] for v in views]  # List of [B,3,H,W]
        imgs = torch.stack(imgs, dim=0)  # [S,B,3,H,W]

        vggt_batch = {
            'images': imgs * 0.5 + 0.5,  # [S,B,3,H,W], 归一化到[0,1]
            'depths': torch.stack([v['depthmap'] for v in views], dim=0) if 'depthmap' in views[0] else None,
            'intrinsics': torch.stack([v['camera_intrinsics'] for v in views], dim=0) if 'camera_intrinsics' in views[0] else None,
            'extrinsics': torch.stack([v['camera_pose'] for v in views], dim=0) if 'camera_pose' in views[0] else None,
            'point_masks': torch.stack([v['valid_mask'] for v in views], dim=0) if 'valid_mask' in views[0] else None,
            'world_points': torch.stack([v['pts3d'] for v in views], dim=0) if 'pts3d' in views[0] else None,
            'flowmap': torch.stack([torch.from_numpy(v['flowmap']).float() if isinstance(v['flowmap'], np.ndarray) else v['flowmap'].float() for v in views], dim=0) if 'flowmap' in views[0] and views[0]['flowmap'] is not None else None,
        }

        print(f"Initial shapes - images: {vggt_batch['images'].shape}")
        if vggt_batch['depths'] is not None:
            print(f"depths: {vggt_batch['depths'].shape}")
        if vggt_batch['world_points'] is not None:
            print(f"world_points: {vggt_batch['world_points'].shape}")
        if vggt_batch['flowmap'] is not None:
            print(f"flowmap: {vggt_batch['flowmap'].shape}")

        # 执行坐标转换和非metric化（如果有必要的数据）
        with tf32_off(), torch.amp.autocast("cuda", enabled=False):
            # 转换world points的坐标系到第一帧相机坐标系
            if vggt_batch['world_points'] is not None:
                # 检查维度并添加batch维度（如果需要）
                if vggt_batch['world_points'].dim() == 4:  # [S, H, W, 3]
                    vggt_batch['world_points'] = vggt_batch['world_points'].unsqueeze(1)  # [S, 1, H, W, 3]
                    vggt_batch['depths'] = vggt_batch['depths'].unsqueeze(1) if vggt_batch['depths'] is not None else None
                    vggt_batch['intrinsics'] = vggt_batch['intrinsics'].unsqueeze(1) if vggt_batch['intrinsics'] is not None else None
                    vggt_batch['extrinsics'] = vggt_batch['extrinsics'].unsqueeze(1) if vggt_batch['extrinsics'] is not None else None
                    vggt_batch['point_masks'] = vggt_batch['point_masks'].unsqueeze(1) if vggt_batch['point_masks'] is not None else None
                    if vggt_batch['flowmap'] is not None:
                        vggt_batch['flowmap'] = vggt_batch['flowmap'].unsqueeze(1)  # [S, 1, H, W, 4]
                    print(f"Added batch dimension - world_points: {vggt_batch['world_points'].shape}")

                B, S, H, W, _ = vggt_batch['world_points'].shape
                world_points = vggt_batch['world_points'].reshape(B, S, H*W, 3)
                world_points = torch.matmul(torch.linalg.inv(vggt_batch['extrinsics'][0])[:, :3, :3], world_points.transpose(-1, -2)).transpose(-1, -2) + \
                                           torch.linalg.inv(vggt_batch['extrinsics'][0])[:, :3, 3:4].transpose(-1, -2)
                vggt_batch['world_points'] = world_points.reshape(B, S, H, W, 3)

                # 转换extrinsics的坐标系到第一帧相机坐标系
                vggt_batch['extrinsics'] = torch.matmul(
                        torch.linalg.inv(vggt_batch['extrinsics']),
                        vggt_batch['extrinsics'][0]
                    )
                    
                # 处理flowmap
                if vggt_batch['flowmap'] is not None:
                    vggt_batch['flowmap'][..., :3] *=  0.1

                # 将extrinsics(中的T)以及world_points、depth进行非metric化
                world_points_flatten = vggt_batch['world_points'].reshape(-1, 3)
                world_points_mask_flatten = vggt_batch['point_masks'].reshape(-1) if vggt_batch['point_masks'] is not None else torch.ones_like(world_points_flatten[:, 0], dtype=torch.bool)
                dist_avg = world_points_flatten[world_points_mask_flatten].norm(dim=-1).mean()
                depth_scale_factor = 1 / dist_avg
                pose_scale_factor = depth_scale_factor

                print(f"Applying non-metric normalization with scale factor: {depth_scale_factor:.6f}")

                # 应用非metric化
                vggt_batch['depths'] = vggt_batch['depths'] * depth_scale_factor
                vggt_batch['extrinsics'][:, :, :3, 3] = vggt_batch['extrinsics'][:, :, :3, 3] * pose_scale_factor
                vggt_batch['world_points'] = vggt_batch['world_points'] * depth_scale_factor

                # 对flowmap应用非metric化：只对velocity magnitude进行缩放
                if vggt_batch['flowmap'] is not None:
                    vggt_batch['flowmap'][..., :3] = vggt_batch['flowmap'][..., :3] * depth_scale_factor

        # 转置到[B, S, ...]格式
        vggt_batch['images'] = vggt_batch['images'].permute(1, 0, 2, 3, 4).contiguous()
        vggt_batch['depths'] = vggt_batch['depths'].permute(1, 0, 2, 3).contiguous() if vggt_batch['depths'] is not None else None
        vggt_batch['intrinsics'] = vggt_batch['intrinsics'].permute(1, 0, 2, 3).contiguous() if vggt_batch['intrinsics'] is not None else None
        vggt_batch['extrinsics'] = vggt_batch['extrinsics'].permute(1, 0, 2, 3).contiguous() if vggt_batch['extrinsics'] is not None else None
        vggt_batch['point_masks'] = vggt_batch['point_masks'].permute(1, 0, 2, 3).contiguous() if vggt_batch['point_masks'] is not None else None
        vggt_batch['world_points'] = vggt_batch['world_points'].permute(1, 0, 2, 3, 4).contiguous() if vggt_batch['world_points'] is not None else None
        vggt_batch['flowmap'] = vggt_batch['flowmap'].permute(1, 0, 2, 3, 4).contiguous() if vggt_batch['flowmap'] is not None else None

        print(f"Final shapes - images: {vggt_batch['images'].shape}")
        if vggt_batch['depths'] is not None:
            print(f"depths: {vggt_batch['depths'].shape}")
        if vggt_batch['world_points'] is not None:
            print(f"world_points: {vggt_batch['world_points'].shape}")
        if vggt_batch['flowmap'] is not None:
            print(f"flowmap: {vggt_batch['flowmap'].shape}")

        print("Safe VGGT batch conversion completed successfully")
        return vggt_batch

    except Exception as e:
        print(f"Error in safe_cut3r_batch_to_vggt: {e}")
        import traceback
        traceback.print_exc()
        raise e


def compute_gt_flow_and_velocity_points(vggt_batch, device):
    """
    从GT flowmap数据中提取3D velocity points，返回按帧组织的valid_current_pts和valid_warped_pts数据
    与demo_viser_flow.py保持一致的数据结构

    Returns:
        dict: {
            'frames_data': list of dict, 每个dict包含一帧的数据:
                {
                    'valid_current_pts': numpy array [N_frame, 3] - 当前帧的有效3D点
                    'valid_warped_pts': numpy array [N_frame, 3] - 通过GT velocity计算的warped 3D点
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
        print("Computing GT flow and velocity points from flowmap...")

        # 获取GT数据
        flowmaps = vggt_batch.get('flowmap')  # [B, S, H, W, 4]
        depths = vggt_batch.get('depths')     # [B, S, H, W]
        intrinsics = vggt_batch.get('intrinsics')  # [B, S, 3, 3]
        extrinsics = vggt_batch.get('extrinsics')  # [B, S, 4, 4]
        world_points = vggt_batch.get('world_points')  # [B, S, H, W, 3]
        images = vggt_batch.get('images')  # [B, S, 3, H, W] - 用于获取原始RGB颜色

        if flowmaps is None:
            print("Error: No flowmap data found")
            return None

        if depths is None or intrinsics is None or extrinsics is None:
            print("Warning: Missing GT data for position computation")
            return None

        B, S, H, W, flow_dim = flowmaps.shape
        print(f"Processing GT flowmap data: {flowmaps.shape}")

        frames_data = []
        all_current_pts = []
        all_warped_pts = []
        global_max_magnitude = 0.0

        # 逐帧处理GT flowmap
        for frame_idx in range(S):
            print(f"Processing GT frame {frame_idx}...")

            # 获取当前帧的flowmap
            frame_flowmap = flowmaps[0, frame_idx]  # [H, W, 4]
            frame_depth = depths[0, frame_idx]      # [H, W]
            frame_image = images[0, frame_idx] if images is not None else None  # [3, H, W]

            # 检查第4维是否为非0来确定有GT velocity的像素
            gt_velocity_mask = frame_flowmap[..., 3] != 0  # [H, W]

            # 同时确保这些像素有有效的depth
            valid_depth_mask = frame_depth > 0  # [H, W]

            # 合并mask：有GT velocity且有有效depth的像素
            combined_mask = gt_velocity_mask & valid_depth_mask  # [H, W]

            # 获取有效位置的索引
            valid_indices = torch.nonzero(combined_mask, as_tuple=True)  # (h_idx, w_idx)
            valid_h, valid_w = valid_indices

            if len(valid_h) == 0:
                print(f"Warning: No valid GT flow points found in frame {frame_idx}")
                continue

            print(f"Frame {frame_idx}: Found {len(valid_h)} valid GT flow points")

            # 提取有效位置的3D velocity (前3维)
            gt_velocity_3d = frame_flowmap[valid_h, valid_w, :3]  # [N, 3]

            # 提取有效位置的RGB颜色
            if frame_image is not None:
                # frame_image是[3, H, W]格式，需要转置为[H, W, 3]
                frame_image_hwc = frame_image.permute(1, 2, 0)  # [H, W, 3]
                point_colors = frame_image_hwc[valid_h, valid_w]  # [N, 3]
            else:
                # 如果没有图像数据，使用白色作为默认颜色
                point_colors = torch.ones(len(valid_h), 3, device=device)

            # 使用velocity_local_to_global将GT velocity从当前帧坐标系转换到第一帧坐标系
            if gt_velocity_3d.shape[0] > 0:  # 确保有有效的velocity数据
                # 首先检查extrinsics的形状并确保格式正确
                print(f"extrinsics shape: {extrinsics.shape}")

                # 确保extrinsics是[B, S, 4, 4]格式
                if extrinsics.dim() == 3:  # [S, 4, 4]
                    extrinsics_batch = extrinsics.unsqueeze(0)  # [1, S, 4, 4]
                elif extrinsics.dim() == 4:  # [B, S, 4, 4]
                    extrinsics_batch = extrinsics
                else:
                    raise ValueError(f"Unexpected extrinsics shape: {extrinsics.shape}")

                # 为velocity_local_to_global创建合适的输入格式
                # 需要将当前帧的velocity扩展为完整的batch格式用于转换
                temp_velocity = torch.zeros(S, H, W, 3, device=gt_velocity_3d.device)
                temp_velocity[frame_idx, valid_h, valid_w] = gt_velocity_3d
                temp_velocity_flat = temp_velocity.reshape(-1, 3)  # [S*H*W, 3]

                # 调用velocity_local_to_global进行坐标转换
                # 注意：velocity_local_to_global期望cam2world格式，而我们的extrinsics已经转换为第一帧到各帧的变换
                # 所以需要传入逆矩阵
                temp_velocity_global = velocity_local_to_global(temp_velocity_flat, torch.linalg.inv(extrinsics_batch))

                # 提取转换后的当前帧velocity
                temp_velocity_global = temp_velocity_global.reshape(S, H, W, 3)
                gt_velocity_3d = temp_velocity_global[frame_idx, valid_h, valid_w]  # [N, 3]

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
            valid_warped_pts = valid_current_pts + gt_velocity_3d

            # 计算velocity magnitude
            velocity_magnitude = torch.norm(gt_velocity_3d, dim=1)
            frame_max_magnitude = velocity_magnitude.max() if len(velocity_magnitude) > 0 and velocity_magnitude.max() > 0 else 1.0
            global_max_magnitude = max(global_max_magnitude, frame_max_magnitude)

            # 保存帧数据
            frames_data.append({
                'valid_current_pts': valid_current_pts,
                'valid_warped_pts': valid_warped_pts,
                'gt_velocity': gt_velocity_3d,
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
        "--image_folder",
        type=str,
        default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train_with_flow/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord",
        help="Path to folder containing images"
    )
    parser.add_argument(
        "--image_interval", type=int, default=1, help="Interval for selecting images from the folder"
    )
    parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
    parser.add_argument("--port", type=int, default=8081, help="Port number for the viser server")
    parser.add_argument(
        "--conf_threshold", type=float, default=50.0, help="Initial percentage of points to display"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for additional logging")
    parser.add_argument("--num_views", type=int, default=8, help="Number of views for inference")
    parser.add_argument("--idx", type=int, default=9600, help="Index of the sequence to process")

    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(("localhost", 5679))
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载数据集
    print(f"Loading dataset from: {args.image_folder}")

    # 创建数据集对象
    from src.dust3r.datasets.waymo import Waymo_Multi

    # 提取序列名
    seq_name = os.path.basename(args.image_folder)
    root_dir = os.path.dirname(args.image_folder)

    print(f"ROOT: {root_dir}, Sequence: {seq_name}")

    dataset = Waymo_Multi(
        split=None,
        ROOT=root_dir,
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

    # 修复数据类型
    print("Fixing data types...")
    views = fix_views_data_types(views, device)

    # 转换为VGGT格式的batch
    print("Converting to VGGT batch format...")
    vggt_batch = safe_cut3r_batch_to_vggt(views, device)

    # 确保batch在正确的设备上
    for key, value in vggt_batch.items():
        if isinstance(value, torch.Tensor):
            vggt_batch[key] = value.to(device)

    # 计算GT flow和velocity点
    print("Computing GT flow and velocity points...")
    flow_data = compute_gt_flow_and_velocity_points(vggt_batch, device)

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