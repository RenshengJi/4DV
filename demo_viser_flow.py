#!/usr/bin/env python3
"""
demo_viser_flow.py - 展示SEA-RAFT光流以及GT pose+depth得到的gt velocity可视化

基于demo_viser.py和demo_stage1_inference.py的实现，展示通过SEA-RAFT光流和GT数据计算得到的velocity flow，
包含valid_current_pts和valid_warped_pts的可视化，以及它们之间的flow动画。
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
sys.path.append(os.path.join(os.path.dirname(__file__), "src/SEA-RAFT/core"))
from raft import RAFT
from vggt.utils.auxiliary import RAFTCfg, calc_flow
from dust3r.utils.image import scene_flow_to_rgb
from dust3r.utils.misc import tf32_off
from src.dust3r.inference import inference
from src.train import cut3r_batch_to_vggt
from accelerate.logging import get_logger


def load_flow_model(flow_model_path, device):
    """加载RAFT光流模型"""
    print(f"Loading RAFT flow model from {flow_model_path}...")

    # 创建RAFT配置
    flow_cfg = RAFTCfg(
        name="kitti-M",
        dataset="kitti",
        path=flow_model_path,
        use_var=True,
        var_min=0,
        var_max=10,
        pretrain="resnet34",
        initial_dim=64,
        block_dims=[64, 128, 256],
        radius=4,
        dim=128,
        num_blocks=2,
        iters=4,
        image_size=[432, 960],
        offload=False,
        geo_thresh=2,
        photo_thresh=-1
    )

    # 创建RAFT模型
    flow_model = RAFT(flow_cfg)

    # 加载权重
    if os.path.exists(flow_model_path):
        state_dict = torch.load(flow_model_path, map_location="cpu", weights_only=True)
        missing_keys, unexpected_keys = flow_model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            print(f"Warning: Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")

    flow_model.to(device)
    flow_model.eval()
    flow_model.requires_grad_(False)

    print("RAFT flow model loaded successfully")
    return flow_model


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
            if key in ['img', 'depthmap', 'camera_intrinsics', 'camera_pose', 'valid_mask', 'pts3d']:
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
        }

        print(f"Initial shapes - images: {vggt_batch['images'].shape}")
        if vggt_batch['depths'] is not None:
            print(f"depths: {vggt_batch['depths'].shape}")
        if vggt_batch['world_points'] is not None:
            print(f"world_points: {vggt_batch['world_points'].shape}")

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

        # 转置到[B, S, ...]格式
        vggt_batch['images'] = vggt_batch['images'].permute(1, 0, 2, 3, 4).contiguous()
        vggt_batch['depths'] = vggt_batch['depths'].permute(1, 0, 2, 3).contiguous() if vggt_batch['depths'] is not None else None
        vggt_batch['intrinsics'] = vggt_batch['intrinsics'].permute(1, 0, 2, 3).contiguous() if vggt_batch['intrinsics'] is not None else None
        vggt_batch['extrinsics'] = vggt_batch['extrinsics'].permute(1, 0, 2, 3).contiguous() if vggt_batch['extrinsics'] is not None else None
        vggt_batch['point_masks'] = vggt_batch['point_masks'].permute(1, 0, 2, 3).contiguous() if vggt_batch['point_masks'] is not None else None
        vggt_batch['world_points'] = vggt_batch['world_points'].permute(1, 0, 2, 3, 4).contiguous() if vggt_batch['world_points'] is not None else None

        print(f"Final shapes - images: {vggt_batch['images'].shape}")
        if vggt_batch['depths'] is not None:
            print(f"depths: {vggt_batch['depths'].shape}")
        if vggt_batch['world_points'] is not None:
            print(f"world_points: {vggt_batch['world_points'].shape}")

        print("Safe VGGT batch conversion completed successfully")
        return vggt_batch

    except Exception as e:
        print(f"Error in safe_cut3r_batch_to_vggt: {e}")
        import traceback
        traceback.print_exc()
        raise e


def compute_flow_and_velocity_points(vggt_batch, device, flow_model):
    """
    计算光流和velocity points，返回按帧组织的valid_current_pts和valid_warped_pts以及颜色信息

    Returns:
        dict: {
            'frames_data': list of dict, 每个dict包含一帧的数据:
                {
                    'valid_current_pts': numpy array [N_frame, 3] - 当前帧的有效3D点
                    'valid_warped_pts': numpy array [N_frame, 3] - 通过光流warp后的有效3D点
                    'velocity_colors': numpy array [N_frame, 3] - velocity的颜色编码
                    'velocity_magnitude': numpy array [N_frame] - velocity的magnitude
                    'frame_idx': int - 帧索引
                }
            'scene_center': numpy array [3] - 场景中心点用于重新定位
            'max_magnitude': float - 所有帧中velocity的最大magnitude
            'total_points': int - 所有帧的总点数
        }
    """
    try:
        print("Computing optical flow and velocity points...")

        images = vggt_batch["images"]  # [B, S, 3, H, W]
        B, S, C, H, W = images.shape

        # 计算光流
        print("Computing optical flow...")
        forward_flow, backward_flow, forward_consist_mask, _, _, _ = calc_flow(
            images, flow_model,
            check_consistency=True,
            geo_thresh=2,
            photo_thresh=-1,
            return_heatmap=False
        )

        # 获取GT数据
        depths = vggt_batch.get('depths')  # [B, S, H, W]
        intrinsics = vggt_batch.get('intrinsics')  # [B, S, 3, 3]
        extrinsics = vggt_batch.get('extrinsics')  # [B, S, 4, 4]
        point_masks = vggt_batch.get('point_masks')  # [B, S, H, W]

        if depths is None or intrinsics is None or extrinsics is None:
            print("Warning: Missing GT data for velocity computation")
            return None

        if point_masks is None:
            print("Warning: No point masks available, using all pixels")
            point_masks = torch.ones(B, S, H, W, device=device, dtype=torch.bool)

        print("Converting depth to world points...")
        # 参考flow_loss中的正确实现
        gt_depth_reshaped = depths.view(depths.shape[0]*depths.shape[1], depths.shape[2], depths.shape[3], 1)
        gt_world_points = depth_to_world_points(gt_depth_reshaped, intrinsics)
        gt_world_points = gt_world_points.view(gt_world_points.shape[0], gt_world_points.shape[1]*gt_world_points.shape[2], 3)

        extrinsic_inv = torch.linalg.inv(extrinsics)
        gt_xyz = torch.matmul(extrinsic_inv[0, :, :3, :3], gt_world_points.transpose(-1, -2)).transpose(-1, -2) + \
            extrinsic_inv[0, :, :3, 3:4].transpose(-1, -2)

        gt_gaussian_means = gt_xyz.reshape(B, S, H, W, 3).permute(0, 1, 4, 2, 3).contiguous()

        print("Computing GT velocity from optical flow...")
        # 计算GT velocity，同时获取valid points
        gt_fwd_vel, gt_fwd_mask = warp_pts3d_for_gt_velocity(
            gt_gaussian_means, point_masks,
            forward_flow, forward_consist_mask,
            direction="forward", interval=1
        )

        # 我们需要从warp_pts3d_for_gt_velocity函数内部获取valid_current_pts和valid_warped_pts
        # 这里需要重新实现部分逻辑来获取这些点
        print("Extracting valid current and warped points...")

        # 按照warp_pts3d_for_gt_velocity的逻辑重新计算，但按帧处理
        valid_frames = S - 1
        current_point_masks = point_masks[:, :-1]  # [B, S-1, H, W]
        next_point_masks = point_masks[:, 1:]      # [B, S-1, H, W]
        current_flow_mask = forward_consist_mask[:, :-1]      # [B, S-1, 1, H, W]
        current_flow = forward_flow[:, :-1]            # [B, S-1, 2, H, W]
        current_3d = gt_gaussian_means[:, :-1]     # [B, S-1, 3, H, W]
        next_3d = gt_gaussian_means[:, 1:]         # [B, S-1, 3, H, W]

        frames_data = []
        all_current_pts = []
        all_warped_pts = []
        global_max_magnitude = 0.0

        # 逐帧处理
        for frame_idx in range(valid_frames):
            print(f"Processing frame {frame_idx}...")

            # 获取当前帧的数据
            frame_current_point_mask = current_point_masks[0, frame_idx]  # [H, W]
            frame_next_point_mask = next_point_masks[0, frame_idx]        # [H, W]
            frame_flow_mask = current_flow_mask[0, frame_idx, 0]          # [H, W]
            frame_flow = current_flow[0, frame_idx]                       # [2, H, W]
            frame_current_3d = current_3d[0, frame_idx]                   # [3, H, W]
            frame_next_3d = next_3d[0, frame_idx]                         # [3, H, W]

            # 合并mask: point_mask & flow_mask
            frame_combined_mask = frame_current_point_mask & frame_flow_mask  # [H, W]

            # 获取有效位置的索引
            frame_inds = torch.nonzero(frame_combined_mask, as_tuple=True)  # (h_idx, w_idx)
            frame_init_h, frame_init_w = frame_inds

            if len(frame_init_h) == 0:
                print(f"Warning: No valid flow points found in frame {frame_idx}")
                continue

            # 获取这些位置的flow
            frame_flow_vals = frame_flow[:, frame_init_h, frame_init_w].T  # [N, 2]

            # 计算warped位置
            frame_warp_w = (frame_init_w + frame_flow_vals[:, 0]).round().long().clamp(min=0, max=W - 1)
            frame_warp_h = (frame_init_h + frame_flow_vals[:, 1]).round().long().clamp(min=0, max=H - 1)

            # 获取当前位置和warped位置的3D点
            frame_current_pts = frame_current_3d[:, frame_init_h, frame_init_w].T    # [N, 3]
            frame_warped_pts = frame_next_3d[:, frame_warp_h, frame_warp_w].T        # [N, 3]

            # 检查warped位置是否也有有效的depth
            frame_warped_valid = frame_next_point_mask[frame_warp_h, frame_warp_w]  # [N]

            # 只保留warped位置也有效的点
            frame_final_valid = frame_warped_valid.bool()
            if frame_final_valid.sum() == 0:
                print(f"Warning: No valid warped points found in frame {frame_idx}")
                continue

            # 过滤有效的点
            frame_valid_current_pts = frame_current_pts[frame_final_valid]    # [N_valid, 3]
            frame_valid_warped_pts = frame_warped_pts[frame_final_valid]      # [N_valid, 3]

            print(f"Frame {frame_idx}: Found {len(frame_valid_current_pts)} valid flow points")

            # 计算velocity
            frame_velocity = frame_valid_warped_pts - frame_valid_current_pts  # [N_valid, 3]

            # 坐标系调整一下(x=z, y=x, z=-y)，与demo_stage1_inference.py保持一致
            frame_velocity_adjusted = frame_velocity[:, [2, 0, 1]]
            frame_velocity_adjusted[:, 2] = -frame_velocity_adjusted[:, 2]

            # 计算velocity magnitude
            frame_velocity_magnitude = torch.norm(frame_velocity_adjusted, dim=1)
            frame_max_magnitude = frame_velocity_magnitude.max() if len(frame_velocity_magnitude) > 0 and frame_velocity_magnitude.max() > 0 else 1.0
            global_max_magnitude = max(global_max_magnitude, frame_max_magnitude)

            # 暂时保存帧数据，稍后统一生成颜色
            frames_data.append({
                'valid_current_pts': frame_valid_current_pts,
                'valid_warped_pts': frame_valid_warped_pts,
                'velocity': frame_velocity_adjusted,
                'velocity_magnitude': frame_velocity_magnitude,
                'frame_idx': frame_idx
            })

            # 收集所有点用于计算场景中心
            all_current_pts.append(frame_valid_current_pts)
            all_warped_pts.append(frame_valid_warped_pts)

        if len(frames_data) == 0:
            print("Warning: No valid frames found")
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
            frame_velocity = frame_data['velocity']
            frame_velocity_magnitude = frame_data['velocity_magnitude']
            N_valid = len(frame_velocity)
            total_points += N_valid

            if N_valid == 0:
                continue

            # 归一化velocity用于颜色计算
            velocity_normalized = frame_velocity / global_max_magnitude * 0.1

            # 生成颜色
            velocity_colors = torch.zeros(N_valid, 3, device=device)

            # 方法1：基于velocity的xyz分量生成RGB颜色
            velocity_colors[:, 0] = torch.clamp((frame_velocity[:, 0] / global_max_magnitude * 0.5 + 0.5), 0, 1)  # R
            velocity_colors[:, 1] = torch.clamp((frame_velocity[:, 1] / global_max_magnitude * 0.5 + 0.5), 0, 1)  # G
            velocity_colors[:, 2] = torch.clamp((frame_velocity[:, 2] / global_max_magnitude * 0.5 + 0.5), 0, 1)  # B

            # 方法2：使用scene_flow_to_rgb的颜色方案
            try:
                # 创建一个合理大小的velocity map用于颜色计算
                map_size = int(np.ceil(np.sqrt(N_valid)))
                if map_size == 0:
                    map_size = 1

                # 创建一个map_size x map_size的velocity map，用0填充不足的部分
                temp_vel_map = torch.zeros(map_size, map_size, 3, device=device)
                flat_indices = torch.arange(min(N_valid, map_size * map_size))
                row_indices = flat_indices // map_size
                col_indices = flat_indices % map_size
                temp_vel_map[row_indices, col_indices] = velocity_normalized[:len(flat_indices)]

                # 使用scene_flow_to_rgb生成颜色
                temp_colors = scene_flow_to_rgb(temp_vel_map.unsqueeze(0), 0.01)  # [1, H, W, 3]

                # 提取对应的颜色
                for i in range(min(N_valid, len(flat_indices))):
                    r, c = row_indices[i], col_indices[i]
                    velocity_colors[i] = temp_colors[0, r, c]

            except Exception as e:
                print(f"Frame {frame_data['frame_idx']}: Using simple color encoding due to error in scene_flow_to_rgb: {e}")
                # 保持方法1的结果

            # 将点重新定位到场景中心
            centered_current_pts = frame_data['valid_current_pts'] - scene_center
            centered_warped_pts = frame_data['valid_warped_pts'] - scene_center

            # 保存最终的帧数据
            final_frames_data.append({
                'valid_current_pts': centered_current_pts.cpu().numpy(),
                'valid_warped_pts': centered_warped_pts.cpu().numpy(),
                'velocity_colors': velocity_colors.cpu().numpy(),
                'velocity_magnitude': frame_velocity_magnitude.cpu().numpy(),
                'frame_idx': frame_data['frame_idx']
            })

        # 构造最终结果
        result = {
            'frames_data': final_frames_data,
            'scene_center': scene_center.cpu().numpy(),
            'max_magnitude': global_max_magnitude.cpu().item() if isinstance(global_max_magnitude, torch.Tensor) else global_max_magnitude,
            'total_points': total_points
        }

        print(f"Computed flow points for {len(final_frames_data)} frames")
        print(f"Total points: {total_points}")
        print(f"Global max velocity magnitude: {result['max_magnitude']:.6f}")

        return result

    except Exception as e:
        print(f"Error in compute_flow_and_velocity_points: {e}")
        import traceback
        traceback.print_exc()
        return None


def viser_wrapper_flow(
    flow_data: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,
    background_mode: bool = False,
):
    """
    使用viser可视化光流和velocity数据，支持按帧显示

    Args:
        flow_data (dict): 包含frames_data等的字典
        port (int): 端口号
        init_conf_threshold (float): 初始置信度阈值（这里用作点的显示比例）
        background_mode (bool): 是否后台运行
    """
    print(f"Starting viser flow server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # 解包流数据
    frames_data = flow_data['frames_data']
    max_magnitude = flow_data['max_magnitude']
    total_points = flow_data['total_points']

    num_frames = len(frames_data)
    print(f"Visualizing {num_frames} frames with {total_points} total valid flow points")

    # 构建GUI控件
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

    gui_flow_animation = server.gui.add_button("Animate Flow")
    gui_reset_view = server.gui.add_button("Reset to Current")

    # 添加信息显示
    gui_info = server.gui.add_text(
        "Flow Info",
        initial_value=f"Total Frames: {num_frames}\nTotal Points: {total_points}\nMax Velocity: {max_magnitude:.6f}m"
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
            frame_info += f" ({N} points)"

        gui_info.value = f"{frame_info}\nDisplaying: {len(display_indices)}/{N} points\nMax Velocity: {max_magnitude:.6f}m"

    def animate_flow():
        """动画显示从current到warped的流动"""
        print("Starting flow animation...")

        # 获取当前帧数据
        current_data = get_current_frame_data()
        valid_current_pts = current_data['valid_current_pts']
        valid_warped_pts = current_data['valid_warped_pts']
        velocity_colors = current_data['velocity_colors']
        velocity_magnitude = current_data['velocity_magnitude']

        N = len(valid_current_pts)
        if N == 0:
            print("No points to animate")
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
                server.scene.remove("animation_points")
            except:
                pass

            # 添加新的动画点云
            server.scene.add_point_cloud(
                name="animation_points",
                points=interpolated_pts,
                colors=display_colors,
                point_size=gui_point_size.value,
                point_shape="circle",
            )

            time.sleep(0.05)  # 50ms间隔

        # 动画结束后恢复正常显示
        try:
            server.scene.remove("animation_points")
        except:
            pass
        update_point_clouds()
        print("Flow animation completed")

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
        """启动流动画动画"""
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

    print("Starting viser flow server...")
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
    parser = argparse.ArgumentParser(description="VGGT flow demo with viser for 3D velocity visualization")
    parser.add_argument(
        "--image_folder",
        type=str,
        default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train/segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord",
        help="Path to folder containing images"
    )
    parser.add_argument(
        "--flow_model_path",
        type=str,
        default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/Tartan-C-T-TSKH-kitti432x960-M.pth",
        help="Path to the RAFT flow model checkpoint",
    )
    parser.add_argument(
        "--image_interval", type=int, default=1, help="Interval for selecting images from the folder"
    )
    parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
    parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
    parser.add_argument(
        "--conf_threshold", type=float, default=50.0, help="Initial percentage of points to display"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for additional logging")
    parser.add_argument("--num_views", type=int, default=8, help="Number of views for inference")
    parser.add_argument("--idx", type=int, default=9600, help="Index of the sequence to process")

    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(("localhost", 5678))
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载流模型
    print("Loading flow model...")
    flow_model = load_flow_model(args.flow_model_path, device)
    print("Flow model loaded successfully!")

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

    # 计算光流和velocity点
    print("Computing flow and velocity points...")
    flow_data = compute_flow_and_velocity_points(vggt_batch, device, flow_model)

    if flow_data is None:
        print("Failed to compute flow data. Exiting.")
        return

    print("Starting viser flow visualization...")
    viser_server = viser_wrapper_flow(
        flow_data,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        background_mode=args.background_mode,
    )
    print("Flow visualization complete")


if __name__ == "__main__":
    main()