#!/usr/bin/env python3
"""
Stage1推理代码 - 输出GT、self_render、velocitymap、gt_velocitymap、skycolor五张拼接图片的视频
基于demo_stage2_inference.py的数据读取方式和输出格式
支持GT velocity map生成，使用光流模型计算前向光流结合GT depth
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
        description="Run Stage1 inference and generate GT+self_render+velocitymap+skycolor videos."
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
        default="./stage1_inference_outputs",
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


def generate_self_render_images(model_preds, vggt_batch, device, sky_color_images=None, opacity_threshold=0.05):
    """
    生成self_render图像
    使用模型预测的gaussian参数进行渲染，基于训练代码中的实现
    在渲染前mask掉opacity小于阈值的gaussian，并在mask区域用sky_color替代

    Args:
        model_preds: 模型预测结果
        vggt_batch: VGGT批次数据
        device: 设备
        sky_color_images: 天空颜色图像 [S, 3, H, W]，如果为None则生成
        opacity_threshold: opacity阈值，小于此值的gaussian将被mask掉
    """
    try:
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from gsplat.rendering import rasterization
        from vggt.training.loss import depth_to_world_points

        B, S, C, H, W = vggt_batch["images"].shape

        # 获取预测结果
        gaussian_params = model_preds.get("gaussian_params")  # [B, S, H, W, 15]
        pose_enc = model_preds.get("pose_enc")  # [B, S, 9]
        depth = model_preds.get("depth")  # [B, S, H, W, 1]

        if gaussian_params is None or pose_enc is None or depth is None:
            print("Warning: Missing required predictions for self-render")
            return torch.zeros(S, 3, H, W, device=device)

        # 按照训练代码的方式处理相机参数
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, (H, W))
        extrinsics = torch.cat([
            extrinsics,
            torch.tensor([0, 0, 0, 1], device=extrinsics.device)[None, None, None, :].repeat(1, extrinsics.shape[1], 1, 1)
        ], dim=-2)

        # 按照训练代码的方式计算3D点
        depth_reshaped = depth.view(depth.shape[0] * depth.shape[1], depth.shape[2], depth.shape[3], 1)
        world_points = depth_to_world_points(depth_reshaped, intrinsics)
        world_points = world_points.view(world_points.shape[0], world_points.shape[1] * world_points.shape[2], 3)

        extrinsic_inv = torch.linalg.inv(extrinsics)
        xyz = torch.matmul(extrinsic_inv[0, :, :3, :3], world_points.transpose(-1, -2)).transpose(-1, -2) + \
              extrinsic_inv[0, :, :3, 3:4].transpose(-1, -2)
        xyz = xyz.reshape(xyz.shape[0], H * W, 3)  # [S, H*W, 3]

        # 处理gaussian参数
        gaussian_params = gaussian_params.reshape(gaussian_params.shape[0], gaussian_params.shape[1],
                                                 gaussian_params.shape[2] * gaussian_params.shape[3], gaussian_params.shape[4])
        gaussian_params = gaussian_params.squeeze(0)  # [S, H*W, 15]

        # 完全按照loss.py中的方式处理gaussian参数
        # gaussian_params现在是[S, H*W, dim]
        scale = gaussian_params[..., 3:6] if gaussian_params.shape[-1] > 6 else gaussian_params[..., :3]
        scale = (0.05 * torch.exp(scale)).clamp_max(0.3)  # [S, H*W, 3]

        color = gaussian_params[..., 6:9].unsqueeze(-2) if gaussian_params.shape[-1] > 9 else torch.sigmoid(gaussian_params[..., :3]).unsqueeze(-2)  # [S, H*W, 1, 3]

        rotation = gaussian_params[..., 9:13] if gaussian_params.shape[-1] > 13 else torch.tensor([1, 0, 0, 0], device=device).repeat(S, H*W, 1)
        # 旋转归一化
        rotation_norms = torch.norm(rotation, dim=-1, keepdim=True)
        rotation_norms = torch.clamp(rotation_norms, min=1e-8)
        rotation = rotation / rotation_norms

        opacity = gaussian_params[..., 13:14].sigmoid().squeeze(-1) if gaussian_params.shape[-1] > 13 else torch.ones(S, H*W, device=device) * 0.5  # [S, H*W]

        # 创建opacity mask - mask掉opacity小于阈值的gaussian
        opacity_mask = opacity >= opacity_threshold  # [S, H*W]
        print(f"Opacity statistics - min: {opacity.min():.4f}, max: {opacity.max():.4f}, mean: {opacity.mean():.4f}")
        print(f"Gaussians above threshold {opacity_threshold}: {opacity_mask.sum().item()}/{opacity.numel()} ({opacity_mask.float().mean()*100:.1f}%)")

        # 应用opacity mask - 将低于阈值的opacity设为0
        opacity_masked = opacity * opacity_mask.float()

        # 准备相机参数用于渲染 - 完全按照loss.py的方式
        # extrinsics: [1, S, 4, 4], intrinsics: [1, S, 3, 3]
        viewmat = extrinsics.permute(1, 0, 2, 3)  # [S, 1, 4, 4]
        K = intrinsics.permute(1, 0, 2, 3)  # [S, 1, 3, 3]

        print(f"viewmat shape: {viewmat.shape}, K shape: {K.shape}")
        print(f"gaussian_params shape: {gaussian_params.shape}")
        print(f"color shape: {color.shape}, scale shape: {scale.shape}, rotation shape: {rotation.shape}, opacity shape: {opacity.shape}")

        rendered_images = []

        for i in range(S):
            try:
                print(f"Frame {i} - rendering shapes:")
                print(f"  xyz: {xyz[i].shape}, rotation: {rotation[i].shape}")
                print(f"  scale: {scale[i].shape}, opacity: {opacity[i].shape}, color: {color[i].shape}")
                print(f"  viewmat: {viewmat[i].shape}, K: {K[i].shape}")

                # 完全按照loss.py的方式使用gsplat进行渲染，使用masked opacity
                render_color, _, _ = rasterization(
                    xyz[i],  # mean positions [H*W, 3]
                    rotation[i],  # rotations [H*W, 4]
                    scale[i],  # scales [H*W, 3]
                    opacity_masked[i],  # 使用masked后的opacities [H*W]
                    color[i],  # colors [H*W, 1, 3]
                    viewmat[i],  # view matrix [1, 4, 4]
                    K[i],  # camera intrinsics [1, 3, 3]
                    W, H,  # image dimensions (注意loss.py中是image_width, image_height)
                    sh_degree=0,
                    render_mode="RGB+ED",
                    radius_clip=0,
                    near_plane=0.0001,
                    far_plane=1000.0,
                    eps2d=0.3,
                )

                # 按照loss.py的方式处理render_color
                # render_color[0]的形状是[H, W, 4]，包含RGB+深度
                render_rgb = render_color[0][..., :3]  # [H, W, 3] - 提取RGB
                render_rgb = render_rgb.permute(2, 0, 1)  # [3, H, W] - 转换为CHW格式
                render_rgb = torch.clamp(render_rgb, min=0, max=1)  # 限制在[0,1]范围

                # 创建mask区域并用sky_color替代
                if sky_color_images is not None and i < sky_color_images.shape[0]:
                    # 将opacity_mask转换为图像mask [H, W]
                    opacity_image_mask = opacity_mask[i].view(H, W).float()  # [H, W] - 转换为float
                    # 扩展到3个通道 [3, H, W]
                    opacity_image_mask = opacity_image_mask.unsqueeze(0).repeat(3, 1, 1)

                    # 在opacity小于阈值的区域使用sky_color
                    sky_color_frame = sky_color_images[i]  # [3, H, W]
                    render_rgb = render_rgb * opacity_image_mask + sky_color_frame * (1 - opacity_image_mask)

                    print(f"Frame {i}: Applied sky color masking")

                rendered_images.append(render_rgb)
                print(f"Frame {i}: Rendering successful, RGB shape: {render_rgb.shape}")

            except Exception as e:
                print(f"Error rendering frame {i}: {e}")
                import traceback
                traceback.print_exc()
                rendered_images.append(torch.zeros(3, H, W, device=device))

        if len(rendered_images) == 0:
            return torch.zeros(S, 3, H, W, device=device)

        return torch.stack(rendered_images, dim=0)  # [S, 3, H, W]

    except Exception as e:
        print(f"Error in generate_self_render_images: {e}")
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
            'flowmap': torch.stack([torch.from_numpy(v['flowmap']).float().to(device) if isinstance(v['flowmap'], np.ndarray) else v['flowmap'].float().to(device) for v in views], dim=0) if 'flowmap' in views[0] and views[0]['flowmap'] is not None else None,
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
                    vggt_batch['flowmap'] = vggt_batch['flowmap'].unsqueeze(1) if vggt_batch['flowmap'] is not None else None
                    print(f"Added batch dimension - world_points: {vggt_batch['world_points'].shape}")

                B, S, H, W, _ = vggt_batch['world_points'].shape
                world_points = vggt_batch['world_points'].reshape(B, S, H*W, 3)
                world_points = torch.matmul(torch.linalg.inv(vggt_batch['extrinsics'][0])[:, :3, :3], world_points.transpose(-1, -2)).transpose(-1, -2) + \
                                           torch.linalg.inv(vggt_batch['extrinsics'][0])[:, :3, 3:4].transpose(-1, -2)
                vggt_batch['world_points'] = world_points.reshape(B, S, H, W, 3)

                # 处理flowmap - 应用缩放
                if vggt_batch['flowmap'] is not None:
                    vggt_batch['flowmap'][..., :3] *= 0.1

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

        print("Safe VGGT batch conversion completed successfully")
        return vggt_batch

    except Exception as e:
        print(f"Error in safe_cut3r_batch_to_vggt: {e}")
        import traceback
        traceback.print_exc()
        raise e


def run_stage1_inference(dataset, stage1_model, flow_model, device, args):
    """执行Stage1推理 - 生成GT、self_render、velocitymap、gt_velocitymap、skycolor五种图像"""

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
        # 创建dummy sky_masks以触发sky color计算
        B, S, C, H, W = vggt_batch["images"].shape
        dummy_sky_masks = torch.ones(B, S, H, W, device=device) * 0.5  # 创建假的sky masks

        stage1_preds = stage1_model(
            vggt_batch["images"],
            compute_sky_color_loss=True,  # 计算sky color
            sky_masks=dummy_sky_masks,  # 传入dummy masks而不是None
            gt_images=vggt_batch["images"],
        )
    stage1_time = time.time() - start_time
    print(f"Stage1 inference completed in {stage1_time:.2f} seconds")

    # 生成四种图像
    print("Generating GT images...")
    gt_images = vggt_batch["images"][0]  # [S, 3, H, W]

    print("Generating sky color images...")
    start_time = time.time()
    sky_color_images = generate_sky_color_images(stage1_preds, vggt_batch, device, stage1_model)
    print(f"Sky color generation completed in {time.time() - start_time:.2f} seconds")

    print("Generating self-render images...")
    start_time = time.time()
    self_render_images = generate_self_render_images(stage1_preds, vggt_batch, device, sky_color_images)
    print(f"Self-render generation completed in {time.time() - start_time:.2f} seconds")

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
        'self_render_images': self_render_images,
        'velocity_map': velocity_map,
        'gt_velocity_map': gt_velocity_map,
        'sky_color_images': sky_color_images,
        'views': views
    }


def save_results_as_video(results, args):
    """保存结果为视频 - 拼接GT、self_render、velocitymap、gt_velocitymap、skycolor五张图片"""
    print("Saving concatenated results as video...")

    # 准备输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    gt_images = results['gt_images']  # [S, 3, H, W]
    self_render_images = results['self_render_images']  # [S, 3, H, W]
    velocity_map = results['velocity_map']  # [S, 3, H, W]
    gt_velocity_map = results['gt_velocity_map']  # [S, 3, H, W]
    sky_color_images = results['sky_color_images']  # [S, 3, H, W]
    views = results['views']

    # 转换为numpy格式并归一化到0-255
    def to_uint8(tensor):
        tensor = torch.clamp(tensor, 0, 1)
        return (tensor.cpu().numpy() * 255).astype(np.uint8)

    gt_images_np = to_uint8(gt_images)
    self_render_images_np = to_uint8(self_render_images)
    velocity_map_np = to_uint8(velocity_map)
    gt_velocity_map_np = to_uint8(gt_velocity_map)
    sky_color_images_np = to_uint8(sky_color_images)

    # 创建视频 - 五列比较：GT | Self-Render | Velocity Map | GT Velocity | Sky Color
    video_path = os.path.join(
        args.output_dir, f"stage1_inference_{args.idx}_{views[0]['label'].split('.')[0]}.mp4")

    with iio.get_writer(video_path, fps=10) as writer:
        num_frames = len(gt_images_np)

        for frame_idx in range(num_frames):
            # 获取当前帧的所有图像
            gt_img = gt_images_np[frame_idx].transpose(1, 2, 0)  # [H, W, 3]
            self_render_img = self_render_images_np[frame_idx].transpose(1, 2, 0)  # [H, W, 3]
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
            self_render_img = add_title(self_render_img, "Self-Render")
            velocity_img = add_title(velocity_img, "Velocity Map")
            gt_velocity_img = add_title(gt_velocity_img, "GT Velocity")
            sky_color_img = add_title(sky_color_img, "Sky Color")

            # 水平拼接五个图像
            combined_frame = np.concatenate([
                gt_img, self_render_img, velocity_img, gt_velocity_img, sky_color_img
            ], axis=1)  # [H, W*5, 3]

            writer.append_data(combined_frame)

    print(f"Stage1 inference video saved to: {video_path}")
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