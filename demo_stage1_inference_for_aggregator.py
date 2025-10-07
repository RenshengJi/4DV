#!/usr/bin/env python3
"""
Stage1推理代码 - 使用aggregator render方式输出所有帧
基于demo_stage1_inference.py，但使用aggregator_render_loss中的渲染方式
输出GT、aggregator_render、velocitymap、gt_velocitymap、skycolor五张拼接图片的视频
"""
import os
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
from copy import deepcopy
from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/vggt'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from vggt.models.vggt import VGGT
from dust3r.utils.misc import tf32_off
from src.dust3r.inference import inference
from src.train import cut3r_batch_to_vggt
from accelerate.logging import get_logger
import torch.multiprocessing
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 导入光流相关模块（保留以备将来使用）
sys.path.append(os.path.join(os.path.dirname(__file__), "src/SEA-RAFT/core"))
from raft import RAFT
from vggt.utils.auxiliary import RAFTCfg, calc_flow

torch.multiprocessing.set_sharing_strategy("file_system")

printer = get_logger(__name__, log_level="DEBUG")

# Set random seed for reproducibility.
random.seed(42)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Stage1 inference with aggregator render and generate GT+aggregator_render+velocitymap+skycolor videos."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/stage1_online_unfreeze+newsky+highvelocity+flownosky+gt+fixedextrinsic+detach/checkpoint-epoch_2_9765.pth",
        help="Path to the Stage1 model checkpoint",
    )
    parser.add_argument(
        "--flow_model_path",
        type=str,
        default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/Tartan-C-T-TSKH-kitti432x960-M.pth",
        help="Path to the RAFT flow model checkpoint",
    )
    parser.add_argument(
        "--seq_dir",
        type=str,
        default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train/segment-15795616688853411272_1245_000_1265_000_with_camera_labels",
        help="Path to the sequence directory or video file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./stage1_aggregator_inference_outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=1600,
        help="Index of the sequence to process (for single inference)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=8,
        help="Number of views for inference",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.05,
        help="Voxel size for aggregator render (default: 0.05 = 5cm)",
    )
    parser.add_argument(
        "--dynamic_threshold",
        type=float,
        default=0.1,
        help="Velocity threshold for dynamic-static separation in m/s (default: 0.1)",
    )

    # 批量推理参数
    parser.add_argument(
        "--batch_mode",
        action="store_true",
        help="Enable batch inference mode",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=1600,
        help="Starting index for batch inference",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=200,
        help="Ending index for batch inference",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="Step size for batch inference",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue batch processing even if some indices fail",
    )

    return parser.parse_args()


def load_stage1_model(model_path, device):
    """加载Stage1模型"""
    print(f"Loading Stage1 model from {model_path}...")

    # 创建模型
    model = VGGT(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        use_sky_token=True,  # 启用sky token以便生成skycolor
    )

    # 加载检查点（按照train.py中的正确方式）
    checkpoint = torch.load(model_path, map_location="cpu")
    ckpt = checkpoint['model']
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()

    print("Stage1 model loaded successfully")
    return model


def load_flow_model(flow_model_path, device):
    """加载RAFT光流模型，参考demo_stage1_inference_for_velocity.py中的加载方式"""
    print(f"Loading RAFT flow model from {flow_model_path}...")

    # 创建RAFT配置，使用RAFTCfg的正确参数
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


def generate_aggregator_render_images(model_preds, vggt_batch, device, voxel_size=0.05, sky_color_images=None, dynamic_threshold=0.1):
    """
    生成aggregator render图像
    使用aggregator_render_loss中的渲染方式（包含动静态分离），但渲染所有帧（不使用sampled_frame_indices）

    Args:
        model_preds: 模型预测结果
        vggt_batch: VGGT批次数据
        device: 设备
        voxel_size: 体素大小（默认0.05 = 5cm）
        sky_color_images: 天空颜色图像 [S, 3, H, W]，如果为None则生成
        dynamic_threshold: 动静态分离的速度阈值（metric scale, m/s，默认0.1）
    """
    try:
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from gsplat.rendering import rasterization
        from vggt.training.loss import depth_to_world_points, velocity_local_to_global, voxel_quantize_random_sampling
        import torch.nn.functional as F

        B, S, C, H, W = vggt_batch["images"].shape

        # 获取预测结果
        gaussian_params = model_preds.get("gaussian_params")  # [B, S, H, W, 15]
        pose_enc = model_preds.get("pose_enc")  # [B, S, 9]
        depth = model_preds.get("depth")  # [B, S, H, W, 1]
        velocity = model_preds.get("velocity")  # [B, S, H, W, 4]

        if gaussian_params is None or pose_enc is None or depth is None or velocity is None:
            print("Warning: Missing required predictions for aggregator render")
            return torch.zeros(S, 3, H, W, device=device)

        # 获取相机参数
        gt_extrinsic = vggt_batch.get("extrinsics")  # [B, S, 4, 4]
        gt_intrinsic = vggt_batch.get("intrinsics")  # [B, S, 3, 3]

        if gt_extrinsic is None or gt_intrinsic is None:
            print("Warning: Missing GT camera parameters")
            return torch.zeros(S, 3, H, W, device=device)

        # 提取gaussian参数（参考aggregator_render_loss的实现）
        # gaussian_params形状: [B, S, H, W, 14] (已激活)
        # 包含: scale(3:6), color(6:9), rotation(9:13), opacity(13)

        # 取velocity的前3维
        velocity_3d = velocity[..., :3]  # [B, S, H, W, 3]

        # 创建sky_masks (简化版本：假设没有sky mask，或使用一个默认的)
        # 在实际应用中，可以从模型预测中获取
        sky_masks = torch.zeros(B, S, H, W, dtype=torch.bool, device=device)

        # 获取gt_scale（从vggt_batch获取depth_scale_factor，与train.py一致）
        gt_scale = vggt_batch.get("depth_scale_factor", 1.0)
        if isinstance(gt_scale, torch.Tensor):
            gt_scale = gt_scale.item()  # 转换为标量

        print(f"Starting aggregator render for {S} frames...")
        print(f"Voxel size: {voxel_size}")
        print(f"GT scale factor: {gt_scale:.6f}")

        # === Step 1: Prepare data for all frames (参考aggregator_render_loss) ===
        depth_flat = depth.view(B * S, H, W, 1)  # [BS, H, W, 1]
        gaussian_params_flat = gaussian_params.view(B * S, H, W, 14)  # [BS, H, W, 14]
        velocity_flat = velocity_3d.view(B * S, H, W, 3)  # [BS, H, W, 3]
        sky_masks_flat = sky_masks.view(B * S, H, W)

        # Compute world points from depth
        world_points = depth_to_world_points(depth_flat, gt_intrinsic.view(B * S, 3, 3))  # [BS, H, W, 3]

        # Transform to first frame coordinate system
        extrinsic_inv = torch.linalg.inv(gt_extrinsic)  # [B, S, 4, 4]
        extrinsic_inv_flat = extrinsic_inv.view(B * S, 4, 4)

        world_points_reshaped = world_points.reshape(B * S, H * W, 3)
        xyz_per_frame = torch.matmul(
            extrinsic_inv_flat[:, :3, :3],
            world_points_reshaped.transpose(-1, -2)
        ).transpose(-1, -2) + extrinsic_inv_flat[:, :3, 3:4].transpose(-1, -2)  # [BS, H*W, 3]

        # Transform velocity to global frame
        velocity_reshaped = velocity_flat.reshape(B * S, H * W, 3)
        # Apply velocity activation (exp transform)
        velocity_activated = torch.sign(velocity_reshaped) * (torch.exp(torch.abs(velocity_reshaped)) - 1)
        velocity_global_per_frame = velocity_local_to_global(
            velocity_activated.reshape(-1, 3),
            extrinsic_inv.view(1, B * S, 4, 4)
        ).reshape(B * S, H * W, 3)  # [BS, H*W, 3]

        # Prepare viewmat and intrinsics
        viewmat = gt_extrinsic[0].unsqueeze(0)  # [1, S, 4, 4]
        K = gt_intrinsic[0].unsqueeze(0)  # [1, S, 3, 3]

        # Pre-allocate result tensors for ALL frames (不使用sampled_frame_indices)
        render_colors = torch.zeros(S, 3, H, W, device=device, dtype=gaussian_params.dtype)

        # Flatten all attributes once before the loop
        gaussian_params_all = gaussian_params_flat.reshape(B * S * H * W, 14)  # [N, 14]
        velocity_global_all = velocity_global_per_frame.reshape(B * S * H * W, 3)  # [N, 3]
        depth_all = depth_flat.reshape(B * S * H * W)  # [N]
        sky_mask_all = sky_masks_flat.reshape(B * S * H * W)  # [N]

        # === Step 1.5: Filter out sky pixels first ===
        print("Filtering sky pixels and separating dynamic/static gaussians...")
        non_sky_mask = ~sky_mask_all.bool()  # [N]
        xyz_per_frame_flat = xyz_per_frame.reshape(B * S * H * W, 3)  # [N, 3]

        # Apply sky filtering
        xyz_non_sky = xyz_per_frame_flat[non_sky_mask]  # [N_valid, 3]
        gaussian_params_non_sky = gaussian_params_all[non_sky_mask]  # [N_valid, 14]
        velocity_global_non_sky = velocity_global_all[non_sky_mask]  # [N_valid, 3]
        depth_non_sky = depth_all[non_sky_mask]  # [N_valid]

        # Keep track of which frame each point belongs to
        frame_indices = torch.arange(B * S, device=device).repeat_interleave(H * W)  # [N]
        frame_indices_non_sky = frame_indices[non_sky_mask]  # [N_valid]

        # === Step 2: Dynamic-static separation ===
        # Convert velocity to metric scale for threshold comparison
        if not isinstance(gt_scale, torch.Tensor):
            gt_scale_tensor = torch.tensor(gt_scale, device=device, dtype=velocity_global_non_sky.dtype)
        else:
            gt_scale_tensor = gt_scale

        velocity_global_non_sky_metric = velocity_global_non_sky / gt_scale_tensor
        velocity_magnitude_metric = torch.norm(velocity_global_non_sky_metric, dim=-1)  # [N_valid], in m/s

        is_dynamic = velocity_magnitude_metric > dynamic_threshold  # [N_valid]
        is_static = ~is_dynamic  # [N_valid]

        print(f"Total points: {len(xyz_non_sky)}, Static: {is_static.sum().item()}, Dynamic: {is_dynamic.sum().item()}")

        # Static gaussians (accumulated across all frames)
        xyz_static = xyz_non_sky[is_static]  # [N_static, 3]
        gaussian_params_static = gaussian_params_non_sky[is_static]  # [N_static, 14]
        depth_static = depth_non_sky[is_static]  # [N_static]

        # Dynamic gaussians (per-frame, for temporal window)
        xyz_dynamic_per_frame = []
        gaussian_params_dynamic_per_frame = []
        velocity_dynamic_per_frame = []
        depth_dynamic_per_frame = []

        for frame_idx in range(S):
            frame_mask = (frame_indices_non_sky == frame_idx) & is_dynamic
            xyz_dynamic_per_frame.append(xyz_non_sky[frame_mask])
            gaussian_params_dynamic_per_frame.append(gaussian_params_non_sky[frame_mask])
            velocity_dynamic_per_frame.append(velocity_global_non_sky[frame_mask])
            depth_dynamic_per_frame.append(depth_non_sky[frame_mask])

        # === Step 3: Voxel quantize static gaussians (globally) ===
        print("Quantizing static gaussians...")
        if voxel_size > 0 and xyz_static.shape[0] > 0:
            static_attributes = {
                'gaussian_params': gaussian_params_static,
                'depth': depth_static,
            }
            xyz_static_quantized, static_attrs_quantized = voxel_quantize_random_sampling(
                xyz_static, static_attributes, sky_mask=None,
                voxel_size=voxel_size, gt_scale=gt_scale
            )
            gaussian_params_static_quantized = static_attrs_quantized['gaussian_params']
            print(f"Static gaussians after quantization: {xyz_static_quantized.shape[0]}")
        else:
            xyz_static_quantized = xyz_static
            gaussian_params_static_quantized = gaussian_params_static
            print(f"No voxel quantization or no static points: {xyz_static_quantized.shape[0]}")

        # === Step 4: For each target frame, process dynamic gaussians and render ===
        print("Rendering all frames with dynamic-static separation...")
        for target_frame in range(S):
            if target_frame % 2 == 0:
                print(f"  Rendering frame {target_frame}/{S}...")

            # Define temporal window: [target_frame-1, target_frame, target_frame+1]
            temporal_window = []
            if target_frame > 0:
                temporal_window.append(target_frame - 1)
            temporal_window.append(target_frame)
            if target_frame < S - 1:
                temporal_window.append(target_frame + 1)

            # Collect dynamic gaussians from temporal window
            xyz_dynamic_list = []
            gaussian_params_dynamic_list = []

            for source_frame in temporal_window:
                if xyz_dynamic_per_frame[source_frame].shape[0] == 0:
                    continue

                # Transform to target frame: xyz - velocity * (target_frame - source_frame)
                time_offset = target_frame - source_frame
                xyz_transformed = xyz_dynamic_per_frame[source_frame] + \
                                velocity_dynamic_per_frame[source_frame] * time_offset

                xyz_dynamic_list.append(xyz_transformed)
                gaussian_params_dynamic_list.append(gaussian_params_dynamic_per_frame[source_frame])

            # Concatenate dynamic gaussians from temporal window
            if len(xyz_dynamic_list) > 0:
                xyz_dynamic_concat = torch.cat(xyz_dynamic_list, dim=0)
                gaussian_params_dynamic_concat = torch.cat(gaussian_params_dynamic_list, dim=0)

                # Voxel quantize dynamic gaussians
                if voxel_size > 0:
                    dynamic_attributes = {
                        'gaussian_params': gaussian_params_dynamic_concat,
                    }
                    xyz_dynamic_quantized, dynamic_attrs_quantized = voxel_quantize_random_sampling(
                        xyz_dynamic_concat, dynamic_attributes, sky_mask=None,
                        voxel_size=voxel_size, gt_scale=gt_scale
                    )
                    gaussian_params_dynamic_quantized = dynamic_attrs_quantized['gaussian_params']
                else:
                    xyz_dynamic_quantized = xyz_dynamic_concat
                    gaussian_params_dynamic_quantized = gaussian_params_dynamic_concat
            else:
                xyz_dynamic_quantized = torch.empty(0, 3, device=device, dtype=xyz_static_quantized.dtype)
                gaussian_params_dynamic_quantized = torch.empty(0, 14, device=device, dtype=gaussian_params_static_quantized.dtype)

            # Merge static and dynamic gaussians
            if xyz_static_quantized.shape[0] > 0 and xyz_dynamic_quantized.shape[0] > 0:
                selected_xyz = torch.cat([xyz_static_quantized, xyz_dynamic_quantized], dim=0)
                selected_gaussian_params = torch.cat([gaussian_params_static_quantized, gaussian_params_dynamic_quantized], dim=0)
            elif xyz_static_quantized.shape[0] > 0:
                selected_xyz = xyz_static_quantized
                selected_gaussian_params = gaussian_params_static_quantized
            elif xyz_dynamic_quantized.shape[0] > 0:
                selected_xyz = xyz_dynamic_quantized
                selected_gaussian_params = gaussian_params_dynamic_quantized
            else:
                print(f"  Frame {target_frame}: No valid gaussians, skipping")
                continue

            # Parse gaussian parameters (already activated in forward)
            scale = selected_gaussian_params[:, 3:6]  # [M, 3]
            color = selected_gaussian_params[:, 6:9].unsqueeze(-2)  # [M, 1, 3]
            rotations = selected_gaussian_params[:, 9:13]  # [M, 4]
            opacity = selected_gaussian_params[:, 13]  # [M]

            # Render to target frame
            render_output, _, _ = rasterization(
                selected_xyz, rotations, scale, opacity, color,
                viewmat[:, target_frame], K[:, target_frame], W, H,
                sh_degree=0, render_mode="RGB+ED",
                radius_clip=0, near_plane=0.0001,
                far_plane=1000.0,
                eps2d=0.3,
            )

            # Extract RGB from render output [H, W, 4]
            render_colors[target_frame] = render_output[0, ..., :3].permute(2, 0, 1)  # [3, H, W]

        # === Step 5: Add sky rendering (use pre-computed sky colors) ===
        if sky_color_images is not None:
            # Blend sky colors with rendered colors based on sky masks
            sky_masks_bool = sky_masks[0].bool()  # [S, H, W]
            render_colors = torch.where(
                sky_masks_bool.unsqueeze(1),  # [S, 1, H, W]
                sky_color_images,
                render_colors
            )

        print(f"Aggregator render completed for all {S} frames")
        return render_colors  # [S, 3, H, W]

    except Exception as e:
        print(f"Error in generate_aggregator_render_images: {e}")
        import traceback
        traceback.print_exc()
        B, S, C, H, W = vggt_batch["images"].shape
        return torch.zeros(S, 3, H, W, device=device)


def generate_velocity_map(model_preds, vggt_batch, device):
    """
    生成velocity map可视化
    将速度向量转换为颜色编码的图像
    """
    try:
        velocity = model_preds.get("velocity")  # [B, S, H, W, 4] 或 [B, S, H, W, 3]

        if velocity is None:
            print("Warning: No velocity predictions found")
            B, S, C, H, W = vggt_batch["images"].shape
            return torch.zeros(S, 3, H, W, device=device)

        B, S, H, W, vel_dim = velocity.shape

        # 取前3维作为xyz速度分量
        velocity_xyz = velocity[0, :, :, :, :3]  # [S, H, W, 3]

        # 应用与训练代码相同的速度变换
        velocity_xyz = torch.sign(velocity_xyz) * (torch.exp(torch.abs(velocity_xyz)) - 1)

        # 坐标系调整一下(x=z, y=x, z=-y)
        velocity_xyz = velocity_xyz[:, :, :, [2, 0, 1]]
        velocity_xyz[:, :, :, 2] = -velocity_xyz[:, :, :, 2]

        # 按照loss.py中cross_render_and_loss的方法实现velocity可视化
        from dust3r.utils.image import scene_flow_to_rgb

        velocity_img_forward = scene_flow_to_rgb(velocity_xyz.reshape(S, H, W, 3), 0.2).permute(0, 3, 1, 2)

        return velocity_img_forward  # [S, 3, H, W]

    except Exception as e:
        print(f"Error in generate_velocity_map: {e}")
        import traceback
        traceback.print_exc()
        B, S, C, H, W = vggt_batch["images"].shape
        return torch.zeros(S, 3, H, W, device=device)


def generate_gt_velocity_map(vggt_batch, device):
    """
    生成GT velocity map可视化
    使用vggt_batch中的flowmap（预处理好的GT 3D velocity）
    """
    try:
        from dust3r.utils.image import scene_flow_to_rgb

        print("Generating GT velocity map from flowmap...")

        flowmap = vggt_batch.get("flowmap")  # [B, S, H, W, 4]

        if flowmap is None:
            print("Warning: No flowmap found in vggt_batch")
            B, S, C, H, W = vggt_batch["images"].shape
            return torch.zeros(S, 3, H, W, device=device)

        B, S, H, W, _ = flowmap.shape

        # 提取GT velocity (前3维) 和 mask (第4维)
        gt_velocity_3d = flowmap[0, :, :, :, :3]  # [S, H, W, 3] - 取第一个batch
        gt_velocity_mask = flowmap[0, :, :, :, 3] != 0  # [S, H, W] - 有GT velocity的区域

        print(f"GT velocity statistics - min: {gt_velocity_3d.min():.4f}, max: {gt_velocity_3d.max():.4f}, mean: {gt_velocity_3d.mean():.4f}")
        print(f"Valid pixels: {gt_velocity_mask.sum().item()}/{gt_velocity_mask.numel()} ({gt_velocity_mask.float().mean()*100:.1f}%)")

        # 坐标系调整一下(x=z, y=x, z=-y)
        velocity_xyz = gt_velocity_3d[:, :, :, [2, 0, 1]]
        velocity_xyz[:, :, :, 2] = -velocity_xyz[:, :, :, 2]

        print("Generating velocity visualization...")
        # 按照loss.py中的方法实现velocity可视化
        velocity_img_forward = scene_flow_to_rgb(velocity_xyz, 0.2).permute(0, 3, 1, 2)

        print(f"GT velocity map generated with shape: {velocity_img_forward.shape}")
        return velocity_img_forward  # [S, 3, H, W]

    except Exception as e:
        print(f"Error in generate_gt_velocity_map: {e}")
        import traceback
        traceback.print_exc()
        B, S, C, H, W = vggt_batch["images"].shape
        return torch.zeros(S, 3, H, W, device=device)


def generate_sky_color_images(model_preds, vggt_batch, device, stage1_model=None):
    """
    生成sky color图像
    使用模型预测的sky token和相机参数生成天空颜色
    """
    try:
        # 检查所有可能的sky相关输出
        print("Available prediction keys:", list(model_preds.keys()))

        sky_colors = model_preds.get("pred_sky_colors")  # [B, S, H, W, 3]
        sky_token = model_preds.get("sky_token")

        if sky_colors is not None:
            print(f"Found pred_sky_colors with shape: {sky_colors.shape}")
            # 转换为正确的格式
            sky_colors = sky_colors[0]  # [S, H, W, 3]
            sky_colors = sky_colors.permute(0, 3, 1, 2)  # [S, 3, H, W]
            # 确保值在[0, 1]范围内
            sky_colors = torch.clamp(sky_colors, 0, 1)
            return sky_colors

        elif sky_token is not None:
            print(f"Found sky_token with shape: {sky_token.shape}, manually generating sky colors...")
            # 手动生成sky colors
            B, S, C, H, W = vggt_batch["images"].shape

            # 获取相机参数
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            pose_enc = model_preds.get("pose_enc")
            if pose_enc is not None:
                pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(
                    pose_enc.detach(), (H, W)
                )
                pred_extrinsics = torch.cat([
                    pred_extrinsics,
                    torch.tensor([0, 0, 0, 1], device=pred_extrinsics.device)[None,None,None,:].repeat(1,pred_extrinsics.shape[1],1,1)
                ], dim=-2)

                # 生成ray directions (简化版本)
                # 创建像素坐标网格
                y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
                rays = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()  # [H, W, 3]

                # 使用内参逆变换
                K_inv = torch.linalg.inv(pred_intrinsics[0, 0])  # [3, 3]
                ray_dirs = torch.matmul(rays, K_inv.T)  # [H, W, 3]

                # 归一化
                ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)

                # 使用sky_head生成颜色（如果模型有的话）
                if stage1_model is not None and hasattr(stage1_model, 'sky_head') and hasattr(stage1_model, 'generate_sky_color'):
                    sky_colors_hwc = stage1_model.generate_sky_color(
                        ray_dirs.unsqueeze(0),  # [1, H, W, 3]
                        sky_token[0, 0:1]  # [1, 1, embed_dim] - 使用第一帧的sky token
                    )
                    # 复制到所有帧
                    sky_colors_list = []
                    for s in range(S):
                        sky_colors_list.append(sky_colors_hwc[0].permute(2, 0, 1))  # [3, H, W]

                    sky_colors = torch.stack(sky_colors_list, dim=0)  # [S, 3, H, W]
                    sky_colors = torch.clamp(sky_colors, 0, 1)
                    return sky_colors

        print("Warning: No sky color predictions found and cannot generate manually")
        B, S, C, H, W = vggt_batch["images"].shape
        # 创建一个渐变的天空色彩作为fallback
        sky_fallback = torch.zeros(S, 3, H, W, device=device)
        for s in range(S):
            # 创建简单的天空渐变：从上到下从蓝色渐变到白色
            for h in range(H):
                intensity = 1.0 - (h / H) * 0.3  # 从上到下亮度递减
                sky_fallback[s, 0, h, :] = 0.5 * intensity  # R
                sky_fallback[s, 1, h, :] = 0.7 * intensity  # G
                sky_fallback[s, 2, h, :] = 1.0 * intensity  # B

        return sky_fallback

    except Exception as e:
        print(f"Error in generate_sky_color_images: {e}")
        import traceback
        traceback.print_exc()
        B, S, C, H, W = vggt_batch["images"].shape
        return torch.zeros(S, 3, H, W, device=device)


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
    """
    参考src/train.py中cut3r_batch_to_vggt的正确实现
    """
    try:
        from dust3r.utils.misc import tf32_off

        print(f"Processing {len(views)} views")

        # 完全按照train.py的方式构建：先构建[S, B, ...]格式
        imgs = [v['img'] for v in views]  # List of [B,3,H,W]
        imgs = torch.stack(imgs, dim=0)  # [S,B,3,H,W]

        # 完全按照train.py的方式构建vggt_batch
        # 注意：flowmap 需要确保在正确的设备上
        vggt_batch = {
            'images': imgs * 0.5 + 0.5,  # [S,B,3,H,W], 归一化到[0,1]
            'depths': torch.stack([v['depthmap'] for v in views], dim=0) if 'depthmap' in views[0] else None,
            'intrinsics': torch.stack([v['camera_intrinsics'] for v in views], dim=0) if 'camera_intrinsics' in views[0] else None,
            'extrinsics': torch.stack([v['camera_pose'] for v in views], dim=0) if 'camera_pose' in views[0] else None,
            'point_masks': torch.stack([v['valid_mask'] for v in views], dim=0) if 'valid_mask' in views[0] else None,
            'world_points': torch.stack([v['pts3d'] for v in views], dim=0) if 'pts3d' in views[0] else None,
            'flowmap': torch.stack([torch.from_numpy(v['flowmap']).float().to(device) if isinstance(v['flowmap'], np.ndarray) else v['flowmap'].float().to(device) for v in views], dim=0) if 'flowmap' in views[0] and views[0]['flowmap'] is not None else None,
        }

        print(f"Initial shapes - images: {vggt_batch['images'].shape}")
        if vggt_batch['depths'] is not None:
            print(f"depths: {vggt_batch['depths'].shape}")
        if vggt_batch['world_points'] is not None:
            print(f"world_points: {vggt_batch['world_points'].shape}")
        if vggt_batch['flowmap'] is not None:
            print(f"flowmap: {vggt_batch['flowmap'].shape}")

        # 完全按照train.py的处理方式，但需要先检查并添加batch维度
        with tf32_off(), torch.amp.autocast("cuda", enabled=False):
            # 转换world points的坐标系到第一帧相机坐标系
            if vggt_batch['world_points'] is not None:
                # 检查维度，如果是4维则添加batch维度
                if vggt_batch['world_points'].dim() == 4:  # [S, H, W, 3]
                    print("Adding batch dimension to data...")
                    vggt_batch['world_points'] = vggt_batch['world_points'].unsqueeze(1)  # [S, 1, H, W, 3]
                    vggt_batch['depths'] = vggt_batch['depths'].unsqueeze(1) if vggt_batch['depths'] is not None else None
                    vggt_batch['intrinsics'] = vggt_batch['intrinsics'].unsqueeze(1) if vggt_batch['intrinsics'] is not None else None
                    vggt_batch['extrinsics'] = vggt_batch['extrinsics'].unsqueeze(1) if vggt_batch['extrinsics'] is not None else None
                    vggt_batch['point_masks'] = vggt_batch['point_masks'].unsqueeze(1) if vggt_batch['point_masks'] is not None else None
                    vggt_batch['flowmap'] = vggt_batch['flowmap'].unsqueeze(1) if vggt_batch['flowmap'] is not None else None

                B, S, H, W, _ = vggt_batch['world_points'].shape
                print(f"Processing with shape - B: {B}, S: {S}, H: {H}, W: {W}")
                world_points = vggt_batch['world_points'].reshape(B, S, H*W, 3)
                world_points = torch.matmul(torch.linalg.inv(vggt_batch['extrinsics'][0])[:, :3, :3], world_points.transpose(-1, -2)).transpose(-1, -2) + \
                                           torch.linalg.inv(vggt_batch['extrinsics'][0])[:, :3, 3:4].transpose(-1, -2)
                vggt_batch['world_points'] = world_points.reshape(B, S, H, W, 3)

                # 处理flowmap
                if vggt_batch['flowmap'] is not None:
                    vggt_batch['flowmap'][..., :3] *=  0.1

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

                # 保存depth_scale_factor到batch中用于aggregator render (与train.py一致)
                vggt_batch['depth_scale_factor'] = depth_scale_factor

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


def run_stage1_inference(dataset, stage1_model, flow_model, device, args):
    """执行Stage1推理 - 生成GT、aggregator_render、velocitymap、gt_velocitymap、skycolor五种图像"""

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

    # 创建一个安全的cut3r_batch_to_vggt版本
    vggt_batch = safe_cut3r_batch_to_vggt(views, device)

    # 运行Stage1推理
    print("Running Stage1 inference...")
    start_time = time.time()
    with torch.no_grad():
        B, S, C, H, W = vggt_batch["images"].shape

        # 使用新的forward函数签名
        # 传入gt_extrinsics和gt_intrinsics以生成sky colors
        stage1_preds = stage1_model(
            vggt_batch["images"],
            query_points=None,
            gt_extrinsics=vggt_batch.get("extrinsics"),
            gt_intrinsics=vggt_batch.get("intrinsics"),
            frame_sample_ratio=1.0,  # 推理时使用所有帧（不采样）
        )
    stage1_time = time.time() - start_time
    print(f"Stage1 inference completed in {stage1_time:.2f} seconds")

    # 生成五种图像
    print("Generating GT images...")
    gt_images = vggt_batch["images"][0]  # [S, 3, H, W]

    print("Generating sky color images...")
    start_time = time.time()
    sky_color_images = generate_sky_color_images(stage1_preds, vggt_batch, device, stage1_model)
    print(f"Sky color generation completed in {time.time() - start_time:.2f} seconds")

    print("Generating aggregator render images...")
    start_time = time.time()
    aggregator_render_images = generate_aggregator_render_images(
        stage1_preds, vggt_batch, device,
        voxel_size=args.voxel_size,
        sky_color_images=sky_color_images,
        dynamic_threshold=args.dynamic_threshold
    )
    print(f"Aggregator render generation completed in {time.time() - start_time:.2f} seconds")

    print("Generating velocity map...")
    start_time = time.time()
    velocity_map = generate_velocity_map(stage1_preds, vggt_batch, device)
    print(f"Velocity map generation completed in {time.time() - start_time:.2f} seconds")

    print("Generating GT velocity map...")
    start_time = time.time()
    gt_velocity_map = generate_gt_velocity_map(vggt_batch, device)
    print(f"GT velocity map generation completed in {time.time() - start_time:.2f} seconds")

    return {
        'gt_images': gt_images,
        'aggregator_render_images': aggregator_render_images,
        'velocity_map': velocity_map,
        'gt_velocity_map': gt_velocity_map,
        'sky_color_images': sky_color_images,
        'views': views
    }


def save_results_as_video(results, args):
    """保存结果为视频 - 拼接GT、aggregator_render、velocitymap、gt_velocitymap、skycolor五张图片"""
    print("Saving concatenated results as video...")

    # 准备输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    gt_images = results['gt_images']  # [S, 3, H, W]
    aggregator_render_images = results['aggregator_render_images']  # [S, 3, H, W]
    velocity_map = results['velocity_map']  # [S, 3, H, W]
    gt_velocity_map = results['gt_velocity_map']  # [S, 3, H, W]
    sky_color_images = results['sky_color_images']  # [S, 3, H, W]
    views = results['views']

    # 转换为numpy格式并归一化到0-255
    def to_uint8(tensor):
        tensor = torch.clamp(tensor, 0, 1)
        return (tensor.detach().cpu().numpy() * 255).astype(np.uint8)

    gt_images_np = to_uint8(gt_images)
    aggregator_render_images_np = to_uint8(aggregator_render_images)
    velocity_map_np = to_uint8(velocity_map)
    gt_velocity_map_np = to_uint8(gt_velocity_map)
    sky_color_images_np = to_uint8(sky_color_images)

    # 创建视频 - 五列比较：GT | Aggregator Render | Velocity Map | GT Velocity | Sky Color
    video_path = os.path.join(
        args.output_dir, f"stage1_aggregator_inference_{args.idx}_{views[0]['label'].split('.')[0]}.mp4")

    with iio.get_writer(video_path, fps=10) as writer:
        num_frames = len(gt_images_np)

        for frame_idx in range(num_frames):
            # 获取当前帧的所有图像
            gt_img = gt_images_np[frame_idx].transpose(1, 2, 0)  # [H, W, 3]
            aggregator_render_img = aggregator_render_images_np[frame_idx].transpose(1, 2, 0)  # [H, W, 3]
            velocity_img = velocity_map_np[frame_idx].transpose(1, 2, 0)  # [H, W, 3]
            gt_velocity_img = gt_velocity_map_np[frame_idx].transpose(1, 2, 0)  # [H, W, 3]
            sky_color_img = sky_color_images_np[frame_idx].transpose(1, 2, 0)  # [H, W, 3]

            # 添加标题
            def add_title(img, title):
                img_with_title = img.copy()
                cv2.putText(img_with_title, title, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                return img_with_title

            gt_img = add_title(gt_img, "GT")
            aggregator_render_img = add_title(aggregator_render_img, "Aggregator Render")
            velocity_img = add_title(velocity_img, "Velocity Map")
            gt_velocity_img = add_title(gt_velocity_img, "GT Velocity")
            sky_color_img = add_title(sky_color_img, "Sky Color")

            # 水平拼接五个图像
            combined_frame = np.concatenate([
                gt_img, aggregator_render_img, velocity_img, gt_velocity_img, sky_color_img
            ], axis=1)  # [H, W*5, 3]

            writer.append_data(combined_frame)

    print(f"Stage1 aggregator inference video saved to: {video_path}")
    return video_path


def run_batch_inference(dataset, stage1_model, flow_model, device, args):
    """运行批量推理"""
    print("=" * 60)
    print("STARTING BATCH INFERENCE")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  Start IDX: {args.start_idx}")
    print(f"  End IDX: {args.end_idx}")
    print(f"  Step: {args.step}")
    print(f"  Continue on error: {args.continue_on_error}")
    print("")

    # 计算要处理的索引列表
    indices_to_process = list(range(args.start_idx, args.end_idx + 1, args.step))
    total_indices = len(indices_to_process)

    print(f"Will process {total_indices} indices: {indices_to_process}")
    print("")

    # 统计信息
    success_count = 0
    failed_count = 0
    failed_indices = []
    successful_videos = []

    # 批量处理
    for i, idx in enumerate(indices_to_process):
        print("=" * 40)
        print(f"Processing IDX {idx} ({i+1}/{total_indices})")
        print("=" * 40)

        try:
            # 临时修改args.idx为当前处理的索引
            original_idx = args.idx
            args.idx = idx

            # 运行单次推理
            with tf32_off():
                results = run_stage1_inference(dataset, stage1_model, flow_model, device, args)

            # 保存结果
            video_path = save_results_as_video(results, args)
            successful_videos.append(video_path)

            print(f"✓ IDX {idx} completed successfully")
            print(f"  Output: {video_path}")
            success_count += 1

            # 恢复原始idx
            args.idx = original_idx

        except Exception as e:
            print(f"✗ IDX {idx} failed with error: {e}")
            failed_count += 1
            failed_indices.append(idx)

            # 恢复原始idx
            args.idx = original_idx

            if not args.continue_on_error:
                print("Stopping batch inference due to error (use --continue_on_error to continue)")
                break

        # 简短休息避免GPU过热
        if i < total_indices - 1:  # 不是最后一个
            print("Waiting 1 second before next inference...")
            time.sleep(1)

        print("")

    # 输出最终统计
    print("=" * 60)
    print("BATCH INFERENCE COMPLETED")
    print("=" * 60)
    print(f"Summary:")
    print(f"  Total processed: {success_count + failed_count}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {failed_count}")

    if failed_indices:
        print(f"  Failed indices: {failed_indices}")

    if successful_videos:
        print(f"\nGenerated videos ({len(successful_videos)}):")
        for video in successful_videos:
            print(f"  - {os.path.basename(video)}")

    print(f"\nOutput directory: {args.output_dir}")

    return {
        'success_count': success_count,
        'failed_count': failed_count,
        'failed_indices': failed_indices,
        'successful_videos': successful_videos
    }


def main():
    args = parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 添加dust3r路径
    add_path_to_dust3r(args.model_path)

    # 加载数据集
    print(f"Loading dataset from: {args.seq_dir}")

    # 创建数据集对象
    from src.dust3r.datasets.waymo import Waymo_Multi

    # 提取序列名
    seq_name = os.path.basename(args.seq_dir)
    root_dir = os.path.dirname(args.seq_dir)

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

    # 加载模型
    print("Loading Stage1 model...")
    stage1_model = load_stage1_model(args.model_path, device)
    print("Stage1 model loaded successfully!")

    print("Loading flow model...")
    flow_model = load_flow_model(args.flow_model_path, device)
    print("Flow model loaded successfully!\n")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    if args.batch_mode:
        # 批量推理模式
        batch_results = run_batch_inference(dataset, stage1_model, flow_model, device, args)

        print(f"Batch inference completed!")
        if batch_results['successful_videos']:
            print(f"Generated {len(batch_results['successful_videos'])} videos successfully")

    else:
        # 单次推理模式
        print(f"Running single inference for IDX {args.idx}")

        with tf32_off():
            results = run_stage1_inference(dataset, stage1_model, flow_model, device, args)

        # 保存结果
        video_path = save_results_as_video(results, args)

        print(f"Single inference completed successfully!")
        print(f"Output video: {video_path}")


if __name__ == "__main__":
    main()
