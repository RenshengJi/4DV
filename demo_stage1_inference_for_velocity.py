#!/usr/bin/env python3
"""
Stage1推理代码 - 专门用于velocity分析
输出GT、velocity_map、forward_flow_heatmap三张拼接图片的视频
基于demo_stage1_inference.py，但专注于velocity和flow分析
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

# 导入光流相关模块
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
        description="Run Stage1 inference for velocity analysis - generate GT+velocity_map+flow_heatmap videos."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo/step2(true+fixmodel+lowlr!+nolpips+onlyflow+velocitylocal+fromscratch)/checkpoint-epoch_2_17880.pth",
        help="Path to the Stage1 model checkpoint",
    )
    parser.add_argument(
        "--flow_model_path",
        type=str,
        default="Tartan-C-T-TSKH-kitti432x960-M.pth",
        help="Path to the RAFT flow model checkpoint",
    )
    parser.add_argument(
        "--seq_dir",
        type=str,
        default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/test/segment-11717495969710734380_2440_000_2460_000_with_camera_labels",
        help="Path to the sequence directory or video file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./stage1_inference_velocity_outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=0,
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
        default=24,
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
        default=150,
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
        use_sky_token=False  # 不需要sky token，专注于velocity
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
    """加载RAFT光流模型，参考train.py中的加载方式"""
    print(f"Loading RAFT flow model from {flow_model_path}...")

    # 创建RAFT配置，与train.py中的配置保持一致
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
        print(f"Flow model - Unexpected keys: {len(unexpected_keys)}, Missing keys: {len(missing_keys)}")
    else:
        print(f"Warning: Flow model checkpoint not found at {flow_model_path}")

    flow_model.to(device)
    flow_model.eval()
    flow_model.requires_grad_(False)

    print("RAFT flow model loaded successfully")
    return flow_model


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

        # 按照loss.py中cross_render_and_loss的方法实现velocity可视化
        from dust3r.utils.image import scene_flow_to_rgb

        velocity_img_forward = scene_flow_to_rgb(velocity_xyz.reshape(S, H, W, 3), 0.01).permute(0, 3, 1, 2)

        return velocity_img_forward  # [S, 3, H, W]

    except Exception as e:
        print(f"Error in generate_velocity_map: {e}")
        import traceback
        traceback.print_exc()
        B, S, C, H, W = vggt_batch["images"].shape
        return torch.zeros(S, 3, H, W, device=device)


def generate_flow_heatmap_and_consistency(images, flow_model, device):
    """
    计算前向光流并生成置信度热力图可视化以及一致性掩码
    """
    try:
        print("Computing forward flow, heatmap, and consistency masks...")

        # 使用calc_flow计算光流、热力图和一致性掩码
        forward_flow, backward_flow, forward_heatmap, backward_heatmap, forward_consist_mask, backward_consist_mask, forward_in_bound_mask, backward_in_bound_mask = calc_flow(
            images, flow_model,
            check_consistency=True,  # 启用一致性检查
            geo_thresh=flow_model.args.geo_thresh,
            photo_thresh=flow_model.args.photo_thresh,
            return_heatmap=True
        )

        print(f"Flow computation completed.")
        print(f"Forward heatmap shape: {forward_heatmap.shape}")
        print(f"Forward consistency mask shape: {forward_consist_mask.shape}")

        # 处理热力图 [B, S, 1, H, W]
        heatmap = forward_heatmap[0]  # [S, 1, H, W]

        # 处理一致性掩码 [B, S, 1, H, W]
        consist_mask = forward_consist_mask[0]  # [S, 1, H, W]

        # 将单通道热力图转换为彩色可视化
        S, _, H, W = heatmap.shape
        heatmap_vis = torch.zeros(S, 3, H, W, device=device)

        for s in range(S):
            # 归一化热力图到 [0, 1]
            heatmap_frame = heatmap[s, 0]  # [H, W]
            heatmap_min = heatmap_frame.min()
            heatmap_max = heatmap_frame.max()

            if heatmap_max > heatmap_min:
                heatmap_norm = (heatmap_frame - heatmap_min) / (heatmap_max - heatmap_min)
            else:
                heatmap_norm = torch.zeros_like(heatmap_frame)

            # 转换为numpy进行colormap处理
            heatmap_np = heatmap_norm.cpu().numpy()

            # 使用jet colormap生成彩色热力图
            import matplotlib.cm as cm
            colored_heatmap = cm.jet(heatmap_np)[:, :, :3]  # [H, W, 3], 去掉alpha通道

            # 转换回tensor
            colored_heatmap_tensor = torch.from_numpy(colored_heatmap).float().to(device)
            heatmap_vis[s] = colored_heatmap_tensor.permute(2, 0, 1)  # [3, H, W]

        return heatmap_vis, consist_mask  # [S, 3, H, W], [S, 1, H, W]

    except Exception as e:
        print(f"Error in generate_flow_heatmap_and_consistency: {e}")
        import traceback
        traceback.print_exc()
        B, S, C, H, W = images.shape
        return torch.zeros(S, 3, H, W, device=device), torch.zeros(S, 1, H, W, device=device)


def generate_confidence_mask(model_preds, device):
    """
    生成深度置信度掩码 (conf)，与loss.py中的逻辑一致
    """
    try:
        depth_conf = model_preds.get("depth_conf")  # [B, S, H, W]

        if depth_conf is None:
            print("Warning: No depth_conf predictions found")
            # 尝试从depth中推断
            depth = model_preds.get("depth")
            if depth is not None:
                B, S, H, W, _ = depth.shape
                return torch.zeros(S, 1, H, W, device=device)
            else:
                return None

        print(f"Depth confidence shape: {depth_conf.shape}")

        # 应用阈值生成置信度掩码，与loss.py中一致: conf = preds["depth_conf"] > 2
        conf = depth_conf > 2  # [B, S, H, W]
        conf = conf[0]  # [S, H, W]
        conf = conf.unsqueeze(1)  # [S, 1, H, W] 添加通道维度

        print(f"Generated confidence mask shape: {conf.shape}")

        return conf

    except Exception as e:
        print(f"Error in generate_confidence_mask: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_mask_as_grayscale(mask, device, mask_name="mask"):
    """
    将布尔mask转换为灰度图像显示

    Args:
        mask: [S, 1, H, W] 布尔张量或浮点张量
        device: 设备
        mask_name: mask名称，用于调试

    Returns:
        [S, 3, H, W] RGB图像（灰度显示，白色=True/高值，黑色=False/低值）
    """
    try:
        if mask is None:
            print(f"Warning: {mask_name} is None, generating black image")
            return torch.zeros(1, 3, 256, 256, device=device)

        print(f"Visualizing {mask_name} with shape: {mask.shape}")

        S, _, H, W = mask.shape
        mask_vis = torch.zeros(S, 3, H, W, device=device)

        for s in range(S):
            # 获取单帧mask [H, W]
            mask_frame = mask[s, 0]  # [H, W]

            # 如果是布尔类型，转换为浮点
            if mask_frame.dtype == torch.bool:
                mask_frame = mask_frame.float()

            # 确保值在[0, 1]范围内
            mask_frame = torch.clamp(mask_frame, 0, 1)

            # 复制到三个通道以形成灰度图像
            mask_vis[s, 0] = mask_frame  # R
            mask_vis[s, 1] = mask_frame  # G
            mask_vis[s, 2] = mask_frame  # B

        return mask_vis  # [S, 3, H, W]

    except Exception as e:
        print(f"Error in visualize_mask_as_grayscale for {mask_name}: {e}")
        import traceback
        traceback.print_exc()
        # 返回默认大小的黑色图像
        return torch.zeros(1, 3, 256, 256, device=device)


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
    安全版本的cut3r_batch_to_vggt，避免维度不匹配的问题
    """
    try:
        from dust3r.utils.misc import tf32_off

        print(f"Processing {len(views)} views")

        # 收集所有的tensor
        imgs = []
        depths = []
        intrinsics = []
        extrinsics = []
        point_masks = []
        world_points = []

        for i, view in enumerate(views):
            # 处理图像
            img = view.get('img')
            if img is not None:
                if img.dim() == 3:  # [3, H, W] -> [1, 3, H, W]
                    img = img.unsqueeze(0)
                imgs.append(img)
                print(f"View {i} img shape: {img.shape}")

            # 处理其他数据
            if 'depthmap' in view and view['depthmap'] is not None:
                depths.append(view['depthmap'])

            if 'camera_intrinsics' in view and view['camera_intrinsics'] is not None:
                intrinsics.append(view['camera_intrinsics'])

            if 'camera_pose' in view and view['camera_pose'] is not None:
                extrinsics.append(view['camera_pose'])

            if 'valid_mask' in view and view['valid_mask'] is not None:
                point_masks.append(view['valid_mask'])

            if 'pts3d' in view and view['pts3d'] is not None:
                world_points.append(view['pts3d'])

        # 构建vggt_batch
        vggt_batch = {}

        # 处理图像 - 应该是 [1, S, 3, H, W]
        if imgs:
            imgs_tensor = torch.stack(imgs, dim=0)  # [S, 1, 3, H, W]
            imgs_tensor = imgs_tensor.squeeze(1)    # [S, 3, H, W]
            imgs_tensor = imgs_tensor.unsqueeze(0)  # [1, S, 3, H, W]
            vggt_batch['images'] = imgs_tensor * 0.5 + 0.5  # 归一化到[0,1]
            print(f"Final images shape: {vggt_batch['images'].shape}")

        # 处理深度
        if depths:
            depths_tensor = torch.stack(depths, dim=0)  # [S, H, W]
            vggt_batch['depths'] = depths_tensor.unsqueeze(0)  # [1, S, H, W]
            print(f"Depths shape: {vggt_batch['depths'].shape}")
        else:
            vggt_batch['depths'] = None

        # 处理内参
        if intrinsics:
            intrinsics_tensor = torch.stack(intrinsics, dim=0)  # [S, 3, 3]
            vggt_batch['intrinsics'] = intrinsics_tensor.unsqueeze(0)  # [1, S, 3, 3]
            print(f"Intrinsics shape: {vggt_batch['intrinsics'].shape}")
        else:
            vggt_batch['intrinsics'] = None

        # 处理外参
        if extrinsics:
            extrinsics_tensor = torch.stack(extrinsics, dim=0)  # [S, 4, 4]
            vggt_batch['extrinsics'] = extrinsics_tensor.unsqueeze(0)  # [1, S, 4, 4]
            print(f"Extrinsics shape: {vggt_batch['extrinsics'].shape}")
        else:
            vggt_batch['extrinsics'] = None

        # 处理点掩码
        if point_masks:
            masks_tensor = torch.stack(point_masks, dim=0)  # [S, H, W]
            vggt_batch['point_masks'] = masks_tensor.unsqueeze(0)  # [1, S, H, W]
            print(f"Point masks shape: {vggt_batch['point_masks'].shape}")
        else:
            vggt_batch['point_masks'] = None

        # 处理世界坐标点
        if world_points:
            points_tensor = torch.stack(world_points, dim=0)  # [S, H, W, 3]
            vggt_batch['world_points'] = points_tensor.unsqueeze(0)  # [1, S, H, W, 3]
            print(f"World points shape: {vggt_batch['world_points'].shape}")
        else:
            vggt_batch['world_points'] = None

        # 执行坐标转换（如果有必要的数据）
        if vggt_batch['world_points'] is not None and vggt_batch['extrinsics'] is not None:
            with tf32_off(), torch.amp.autocast("cuda", enabled=False):
                print("Performing coordinate transformation...")
                B, S, H, W, _ = vggt_batch['world_points'].shape
                print(f"world_points shape for transformation: B={B}, S={S}, H={H}, W={W}")

                world_points_reshaped = vggt_batch['world_points'].reshape(B, S, H*W, 3)
                extrinsics_inv = torch.linalg.inv(vggt_batch['extrinsics'])

                # 转换到第一帧坐标系
                transformed_points = torch.matmul(
                    extrinsics_inv[0, :, :3, :3],
                    world_points_reshaped.transpose(-1, -2)
                ).transpose(-1, -2) + extrinsics_inv[0, :, :3, 3:4].transpose(-1, -2)

                vggt_batch['world_points'] = transformed_points.reshape(B, S, H, W, 3)

                # 转换外参
                vggt_batch['extrinsics'] = torch.matmul(
                    extrinsics_inv,
                    vggt_batch['extrinsics'][0]
                )

        print("Safe VGGT batch conversion completed successfully")
        return vggt_batch

    except Exception as e:
        print(f"Error in safe_cut3r_batch_to_vggt: {e}")
        import traceback
        traceback.print_exc()
        raise e


def run_stage1_inference_for_velocity(dataset, stage1_model, flow_model, device, args):
    """执行Stage1推理 - 专门用于velocity分析，生成GT、velocity_map、flow_heatmap、consist_mask、conf_mask五种图像"""

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

    # 运行Stage1推理
    print("Running Stage1 inference...")
    start_time = time.time()
    with torch.no_grad():
        stage1_preds = stage1_model(
            vggt_batch["images"],
            compute_sky_color_loss=False,  # 不需要sky color
            sky_masks=None,
            gt_images=vggt_batch["images"],
        )
    stage1_time = time.time() - start_time
    print(f"Stage1 inference completed in {stage1_time:.2f} seconds")

    # 生成五种图像
    print("Generating GT images...")
    gt_images = vggt_batch["images"][0]  # [S, 3, H, W]

    print("Generating velocity map...")
    start_time = time.time()
    velocity_map = generate_velocity_map(stage1_preds, vggt_batch, device)
    print(f"Velocity map generation completed in {time.time() - start_time:.2f} seconds")

    print("Generating flow heatmap and consistency masks...")
    start_time = time.time()
    flow_heatmap, forward_consist_mask = generate_flow_heatmap_and_consistency(vggt_batch["images"], flow_model, device)
    print(f"Flow heatmap and consistency generation completed in {time.time() - start_time:.2f} seconds")

    print("Generating confidence mask...")
    start_time = time.time()
    conf_mask = generate_confidence_mask(stage1_preds, device)
    print(f"Confidence mask generation completed in {time.time() - start_time:.2f} seconds")

    # 可视化mask为灰度图像
    print("Converting masks to grayscale visualizations...")
    consist_mask_vis = visualize_mask_as_grayscale(forward_consist_mask, device, "forward_consist_mask")
    conf_mask_vis = visualize_mask_as_grayscale(conf_mask, device, "conf_mask")

    return {
        'gt_images': gt_images,
        'velocity_map': velocity_map,
        'flow_heatmap': flow_heatmap,
        'consist_mask': consist_mask_vis,
        'conf_mask': conf_mask_vis,
        'views': views
    }


def save_results_as_video(results, args):
    """保存结果为视频 - 拼接GT、velocity_map、flow_heatmap、consist_mask、conf_mask五张图片"""
    print("Saving concatenated results as video...")

    # 准备输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    gt_images = results['gt_images']  # [S, 3, H, W]
    velocity_map = results['velocity_map']  # [S, 3, H, W]
    flow_heatmap = results['flow_heatmap']  # [S, 3, H, W]
    consist_mask = results['consist_mask']  # [S, 3, H, W]
    conf_mask = results['conf_mask']  # [S, 3, H, W]
    views = results['views']

    # 转换为numpy格式并归一化到0-255
    def to_uint8(tensor):
        tensor = torch.clamp(tensor, 0, 1)
        return (tensor.cpu().numpy() * 255).astype(np.uint8)

    gt_images_np = to_uint8(gt_images)
    velocity_map_np = to_uint8(velocity_map)
    flow_heatmap_np = to_uint8(flow_heatmap)
    consist_mask_np = to_uint8(consist_mask)
    conf_mask_np = to_uint8(conf_mask)

    # 创建视频 - 五列比较：GT | Velocity Map | Flow Heatmap | Consist Mask | Conf Mask
    video_path = os.path.join(
        args.output_dir, f"stage1_velocity_inference_{args.idx}_{views[0]['label'].split('.')[0]}.mp4")

    with iio.get_writer(video_path, fps=10) as writer:
        num_frames = len(gt_images_np)

        for frame_idx in range(num_frames):
            # 获取当前帧的所有图像
            gt_img = gt_images_np[frame_idx].transpose(1, 2, 0)  # [H, W, 3]
            velocity_img = velocity_map_np[frame_idx].transpose(1, 2, 0)  # [H, W, 3]
            heatmap_img = flow_heatmap_np[frame_idx].transpose(1, 2, 0)  # [H, W, 3]
            consist_img = consist_mask_np[frame_idx].transpose(1, 2, 0)  # [H, W, 3]
            conf_img = conf_mask_np[frame_idx].transpose(1, 2, 0)  # [H, W, 3]

            # 添加标题
            def add_title(img, title, font_scale=0.7):
                img_with_title = img.copy()
                cv2.putText(img_with_title, title, (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
                return img_with_title

            gt_img = add_title(gt_img, "GT")
            velocity_img = add_title(velocity_img, "Velocity Map")
            heatmap_img = add_title(heatmap_img, "Flow Heatmap")
            consist_img = add_title(consist_img, "Consist Mask")
            conf_img = add_title(conf_img, "Conf Mask")

            # 水平拼接五个图像
            combined_frame = np.concatenate([
                gt_img, velocity_img, heatmap_img, consist_img, conf_img
            ], axis=1)  # [H, W*5, 3]

            writer.append_data(combined_frame)

    print(f"Stage1 velocity inference video saved to: {video_path}")
    return video_path


def run_batch_inference(dataset, stage1_model, flow_model, device, args):
    """运行批量推理"""
    print("=" * 60)
    print("STARTING BATCH VELOCITY INFERENCE")
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
                results = run_stage1_inference_for_velocity(dataset, stage1_model, flow_model, device, args)

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
    print("BATCH VELOCITY INFERENCE COMPLETED")
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
        valid_camera_id_list=["1"],
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

    print("Loading RAFT flow model...")
    flow_model = load_flow_model(args.flow_model_path, device)
    print("RAFT flow model loaded successfully!")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    if args.batch_mode:
        # 批量推理模式
        batch_results = run_batch_inference(dataset, stage1_model, flow_model, device, args)

        print(f"Batch velocity inference completed!")
        if batch_results['successful_videos']:
            print(f"Generated {len(batch_results['successful_videos'])} videos successfully")

    else:
        # 单次推理模式
        print(f"Running single velocity inference for IDX {args.idx}")

        with tf32_off():
            results = run_stage1_inference_for_velocity(dataset, stage1_model, flow_model, device, args)

        # 保存结果
        video_path = save_results_as_video(results, args)

        print(f"Single velocity inference completed successfully!")
        print(f"Output video: {video_path}")


if __name__ == "__main__":
    main()