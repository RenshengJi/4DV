#!/usr/bin/env python3
"""
Novel View Synthesis Inference Script
生成3x3视频布局,展示不同视角的渲染结果:
- 中间(第2排第2列): 原视角
- 上方(第1排第2列): 视角向上移动
- 下方(第3排第2列): 视角向下移动
- 左侧(第2排第1列): 视角向左移动
- 右侧(第2排第3列): 视角向右移动
- 其他位置(对角线): 组合视角移动

只输出RGB,不输出depth
"""

import os
import sys
import numpy as np
import torch
import argparse
import cv2
import imageio.v2 as iio
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from copy import deepcopy

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/vggt'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from vggt.models.vggt import VGGT
from dust3r.utils.misc import tf32_off
from src.dust3r.inference import inference
from src.train import cut3r_batch_to_vggt
from src.online_dynamic_processor import OnlineDynamicProcessor


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Novel View Synthesis Inference")

    # 基础参数
    parser.add_argument("--model_path", type=str, required=True, help="Path to Stage1 model checkpoint")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument("--output_dir", type=str, default="./inference_nvs_outputs", help="Output directory")
    parser.add_argument("--idx", type=int, default=0, help="Sequence index (single mode)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--num_views", type=int, default=8, help="Number of views")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS")

    # 批量推理参数
    parser.add_argument("--batch_mode", action="store_true", help="Enable batch inference mode")
    parser.add_argument("--start_idx", type=int, default=150, help="Start index for batch mode")
    parser.add_argument("--end_idx", type=int, default=200, help="End index for batch mode")
    parser.add_argument("--step", type=int, default=5, help="Step size for batch mode")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue on error in batch mode")

    # Novel View Synthesis参数
    parser.add_argument("--translation_offset", type=float, default=3.0,
                       help="Translation offset in meters for NVS")

    # Dynamic processor参数
    parser.add_argument("--velocity_transform_mode", type=str, default="procrustes",
                       choices=["simple", "procrustes"], help="Velocity transformation mode")

    # 聚类参数
    parser.add_argument("--velocity_threshold", type=float, default=0.1,
                       help="Velocity threshold for clustering")
    parser.add_argument("--clustering_eps", type=float, default=0.3,
                       help="DBSCAN eps parameter (meters)")
    parser.add_argument("--clustering_min_samples", type=int, default=10,
                       help="DBSCAN min_samples parameter")
    parser.add_argument("--min_object_size", type=int, default=500,
                       help="Minimum object size (points)")
    parser.add_argument("--tracking_position_threshold", type=float, default=2.0,
                       help="Position threshold for tracking")
    parser.add_argument("--tracking_velocity_threshold", type=float, default=0.2,
                       help="Velocity threshold for tracking")

    # VGGT模型配置参数
    parser.add_argument("--sh_degree", type=int, default=0, help="Spherical harmonics degree")
    parser.add_argument("--use_gs_head", action="store_true", default=True, help="Use DPTGSHead for gaussian_head")
    parser.add_argument("--use_gs_head_velocity", action="store_true", default=False, help="Use DPTGSHead for velocity_head")
    parser.add_argument("--use_gt_camera", action="store_true", help="Use GT camera parameters")
    
    return parser.parse_args()


def load_model(model_path, device, args):
    """加载Stage1模型"""
    print(f"Loading model from: {model_path}")
    print(f"Model config: sh_degree={args.sh_degree}, use_gs_head={args.use_gs_head}, use_gs_head_velocity={args.use_gs_head_velocity}")

    model = VGGT(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        use_sky_token=True,
        sh_degree=args.sh_degree,
        use_gs_head=args.use_gs_head,
        use_gs_head_velocity=args.use_gs_head_velocity,
        use_gt_camera=args.use_gt_camera
    )

    checkpoint = torch.load(model_path, map_location="cpu")
    ckpt = checkpoint.get('model', checkpoint)
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model


def load_dataset(dataset_root, num_views):
    """加载数据集"""
    from src.dust3r.datasets.waymo import Waymo_Multi

    seq_name = os.path.basename(dataset_root)
    root_dir = os.path.dirname(dataset_root)

    print(f"Loading dataset - Root: {root_dir}, Sequence: {seq_name}")

    dataset = Waymo_Multi(
        split=None,
        ROOT=root_dir,
        img_ray_mask_p=[1.0, 0.0, 0.0],
        valid_camera_id_list=["1", "2", "3"],
        resolution=[(518, 378), (518, 336), (518, 294), (518, 252), (518, 210),
                    (518, 140), (378, 518), (336, 518), (294, 518), (252, 518)],
        num_views=num_views,
        seed=42,
        n_corres=0,
        seq_aug_crop=True
    )

    return dataset


def ensure_tensor(data, device):
    """确保数据是tensor格式"""
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


def modify_camera_extrinsics(extrinsics, offset_x=0.0, offset_y=0.0, offset_z=0.0):
    """
    修改相机外参,实现视角平移

    Args:
        extrinsics: [S, 4, 4] 相机外参矩阵 (world-to-camera变换)
        offset_x: 相机在世界坐标系X方向的偏移 (米)
        offset_y: 相机在世界坐标系Y方向的偏移 (米)
        offset_z: 相机在世界坐标系Z方向的偏移 (米)

    Returns:
        modified_extrinsics: [S, 4, 4] 修改后的相机外参

    相机坐标系(OpenCV):
        - X: 右
        - Y: 下
        - Z: 前

    世界坐标系(OpenCV标准 - VGGT使用):
        - X: 右 (Right)
        - Y: 下 (Down)
        - Z: 前 (Forward)

    视角移动映射:
        - 视角向上: offset_y < 0 (Y负方向)
        - 视角向下: offset_y > 0 (Y正方向)
        - 视角向左: offset_x < 0 (X负方向)
        - 视角向右: offset_x > 0 (X正方向)
    """
    S = extrinsics.shape[0]
    device = extrinsics.device

    modified_extrinsics = extrinsics.clone()

    # 创建平移向量 (世界坐标系)
    translation_world = torch.tensor([offset_x, offset_y, offset_z],
                                     device=device, dtype=extrinsics.dtype)

    for s in range(S):
        # extrinsics[s] 是 4x4 矩阵: [R | t]
        #                           [0 | 1]
        # 其中:
        #   R 是旋转矩阵(world to camera)
        #   t = -R @ C_world，其中C_world是相机中心在世界坐标系的位置

        # 提取旋转矩阵和平移向量
        R = modified_extrinsics[s, :3, :3]  # [3, 3]
        t = modified_extrinsics[s, :3, 3]   # [3]

        # 移动相机:
        # C_world_new = C_world + offset
        # t_new = -R @ C_world_new = -R @ (C_world + offset)
        #       = -R @ C_world - R @ offset
        #       = t - R @ offset
        # 注意: 这里是减号，因为我们移动相机，而不是移动世界
        t_offset_camera = R @ translation_world
        t_new = t - t_offset_camera  # 注意符号是负号

        modified_extrinsics[s, :3, 3] = t_new

    return modified_extrinsics


def render_gaussians_with_sky(scene, intrinsics, extrinsics, sky_colors, sampled_frame_indices, H, W, device):
    """
    渲染gaussian场景 (与inference.py相同的实现)
    包括sky_color的alpha blending合成
    逐帧渲染以正确处理动态物体的变换
    """
    from gsplat import rasterization
    import torch.nn.functional as F

    S = intrinsics.shape[0]
    rendered_images = []

    # 逐帧渲染
    for frame_idx in range(S):
        # 每帧收集gaussians
        all_means = []
        all_scales = []
        all_colors = []
        all_rotations = []
        all_opacities = []

        # Static gaussians
        if scene.get('static_gaussians') is not None:
            static_gaussians = scene['static_gaussians']  # [N, 14]
            if static_gaussians.shape[0] > 0:
                all_means.append(static_gaussians[:, :3])
                all_scales.append(static_gaussians[:, 3:6])
                all_colors.append(static_gaussians[:, 6:9].unsqueeze(-2))
                all_rotations.append(static_gaussians[:, 9:13])
                all_opacities.append(static_gaussians[:, 13])

        # Dynamic objects - Cars (使用canonical空间+变换)
        dynamic_objects_cars = scene.get('dynamic_objects_cars', [])
        for obj_data in dynamic_objects_cars:
            # 检查物体是否在当前帧存在
            if not _object_exists_in_frame(obj_data, frame_idx):
                continue

            # 获取物体在正规空间(canonical)的Gaussian参数
            canonical_gaussians = obj_data.get('canonical_gaussians')  # [N, 14]
            if canonical_gaussians is None or canonical_gaussians.shape[0] == 0:
                continue

            # 获取从canonical空间到当前帧的变换
            frame_transform = _get_object_transform_to_frame(obj_data, frame_idx)
            if frame_transform is None:
                transformed_gaussians = canonical_gaussians
            else:
                transformed_gaussians = _apply_transform_to_gaussians(
                    canonical_gaussians, frame_transform
                )

            # 添加到渲染列表
            if transformed_gaussians.shape[0] > 0:
                all_means.append(transformed_gaussians[:, :3])
                all_scales.append(transformed_gaussians[:, 3:6])
                all_colors.append(transformed_gaussians[:, 6:9].unsqueeze(-2))
                all_rotations.append(transformed_gaussians[:, 9:13])
                all_opacities.append(transformed_gaussians[:, 13])

        # Dynamic objects - People (每帧单独的Gaussians，不使用变换)
        dynamic_objects_people = scene.get('dynamic_objects_people', [])
        for obj_data in dynamic_objects_people:
            # 检查物体是否在当前帧存在
            frame_gaussians = obj_data.get('frame_gaussians', {})
            if frame_idx not in frame_gaussians:
                continue

            # 直接使用当前帧的Gaussians（不进行变换）
            current_frame_gaussians = frame_gaussians[frame_idx]  # [N, 14]
            if current_frame_gaussians is None or current_frame_gaussians.shape[0] == 0:
                continue

            # 添加到渲染列表
            all_means.append(current_frame_gaussians[:, :3])
            all_scales.append(current_frame_gaussians[:, 3:6])
            all_colors.append(current_frame_gaussians[:, 6:9].unsqueeze(-2))
            all_rotations.append(current_frame_gaussians[:, 9:13])
            all_opacities.append(current_frame_gaussians[:, 13])

        if len(all_means) == 0:
            # 如果没有Gaussian，返回空图像
            rendered_images.append(torch.zeros(3, H, W, device=device))
            continue

        # Concatenate
        means = torch.cat(all_means, dim=0)  # [N, 3]
        scales = torch.cat(all_scales, dim=0)  # [N, 3]
        colors = torch.cat(all_colors, dim=0)  # [N, 1, 3]
        rotations = torch.cat(all_rotations, dim=0)  # [N, 4]
        opacities = torch.cat(all_opacities, dim=0)  # [N]

        # Fix NaN/Inf
        means = torch.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
        scales = torch.nan_to_num(scales, nan=0.01, posinf=1.0, neginf=0.01)
        colors = torch.nan_to_num(colors, nan=0.5, posinf=1.0, neginf=0.0)
        rotations = torch.nan_to_num(rotations, nan=0.0, posinf=1.0, neginf=-1.0)
        opacities = torch.nan_to_num(opacities, nan=0.5, posinf=1.0, neginf=0.0)

        K = intrinsics[frame_idx]
        w2c = extrinsics[frame_idx]

        try:
            render_result = rasterization(
                means, rotations, scales, opacities, colors,
                w2c.unsqueeze(0), K.unsqueeze(0), W, H,
                sh_degree=0, render_mode="RGB+ED",
                radius_clip=0, near_plane=0.0001,
                far_plane=1000.0,
                eps2d=0.3,
            )

            rendered_image = render_result[0][0, :, :, :3].permute(2, 0, 1)
            rendered_alpha = render_result[1][0, :, :, 0] if len(render_result) > 1 and render_result[1] is not None else torch.ones(H, W, device=device)

            rendered_image = torch.clamp(rendered_image, min=0, max=1)

            # Composite with sky
            if sky_colors is not None and sampled_frame_indices is not None:
                if not isinstance(sampled_frame_indices, torch.Tensor):
                    sampled_frame_indices = torch.tensor(sampled_frame_indices, device=device)

                matches = (sampled_frame_indices == frame_idx)
                if matches.any():
                    sky_idx = matches.nonzero(as_tuple=True)[0].item()
                    frame_sky_color = sky_colors[sky_idx]  # [3, H, W]
                    alpha_3ch = rendered_alpha.unsqueeze(0)  # [1, H, W]
                    rendered_image = alpha_3ch * rendered_image + (1 - alpha_3ch) * frame_sky_color
                    rendered_image = torch.clamp(rendered_image, min=0, max=1)

            rendered_images.append(rendered_image)

        except Exception as e:
            print(f"Error rendering frame {frame_idx}: {e}")
            rendered_images.append(torch.zeros(3, H, W, device=device))

    return torch.stack(rendered_images, dim=0)


def _object_exists_in_frame(obj_data, frame_idx):
    """检查动态物体是否在指定帧中存在"""
    if 'frame_transforms' in obj_data:
        frame_transforms = obj_data['frame_transforms']
        if frame_idx in frame_transforms:
            return True
    return False


def _get_object_transform_to_frame(obj_data, frame_idx):
    """获取从canonical空间到指定帧的变换矩阵"""
    reference_frame = obj_data.get('reference_frame', 0)
    if frame_idx == reference_frame:
        return None

    if 'frame_transforms' in obj_data:
        frame_transforms = obj_data['frame_transforms']
        if frame_idx in frame_transforms:
            frame_to_canonical = frame_transforms[frame_idx]
            canonical_to_frame = torch.inverse(frame_to_canonical)
            return canonical_to_frame

    return None


def _apply_transform_to_gaussians(gaussians, transform):
    """将变换应用到Gaussian参数"""
    if torch.allclose(transform, torch.zeros_like(transform), atol=1e-6):
        print(f"⚠️  检测到零变换矩阵！使用单位矩阵替代")
        transform = torch.eye(4, dtype=transform.dtype, device=transform.device)
    else:
        det_val = torch.det(transform[:3, :3].float()).abs()
        if det_val < 1e-8:
            print(f"⚠️  变换矩阵奇异(det={det_val:.2e})！")

    transformed_gaussians = gaussians.clone()

    # 变换位置
    positions = gaussians[:, :3]  # [N, 3]
    positions_homo = torch.cat([positions, torch.ones(
        positions.shape[0], 1, device=positions.device)], dim=1)  # [N, 4]
    transformed_positions = torch.mm(
        transform, positions_homo.T).T[:, :3]  # [N, 3]
    transformed_gaussians[:, :3] = transformed_positions

    return transformed_gaussians


def create_nvs_grid(rendered_views, translation_offset):
    """
    创建3x3的NVS可视化网格

    Args:
        rendered_views: dict包含9个视角的渲染结果，每个是[S, 3, H, W]
        translation_offset: 平移偏移量(用于显示)

    Returns:
        grid_frames: list of [H_grid, W_grid, 3] numpy arrays
    """
    # 获取视频参数
    center_view = rendered_views['center']  # [S, 3, H, W]
    S, C, H, W = center_view.shape

    grid_frames = []

    for s in range(S):
        # 提取每个视角的当前帧
        # Row 1: up-left, up, up-right
        up_left = rendered_views['up_left'][s].permute(1, 2, 0).detach().cpu().numpy()
        up = rendered_views['up'][s].permute(1, 2, 0).detach().cpu().numpy()
        up_right = rendered_views['up_right'][s].permute(1, 2, 0).detach().cpu().numpy()

        # Row 2: left, center, right
        left = rendered_views['left'][s].permute(1, 2, 0).detach().cpu().numpy()
        center = rendered_views['center'][s].permute(1, 2, 0).detach().cpu().numpy()
        right = rendered_views['right'][s].permute(1, 2, 0).detach().cpu().numpy()

        # Row 3: down-left, down, down-right
        down_left = rendered_views['down_left'][s].permute(1, 2, 0).detach().cpu().numpy()
        down = rendered_views['down'][s].permute(1, 2, 0).detach().cpu().numpy()
        down_right = rendered_views['down_right'][s].permute(1, 2, 0).detach().cpu().numpy()

        # 创建网格: 3 rows x 3 columns
        row1 = np.concatenate([up_left, up, up_right], axis=1)
        row2 = np.concatenate([left, center, right], axis=1)
        row3 = np.concatenate([down_left, down, down_right], axis=1)

        grid = np.concatenate([row1, row2, row3], axis=0)
        grid = (np.clip(grid, 0, 1) * 255).astype(np.uint8)

        grid_frames.append(grid)

    return grid_frames


def run_single_inference(model, dataset, dynamic_processor, idx, num_views, device, args):
    """运行单次推理并生成NVS视频"""
    print(f"\n{'='*60}")
    print(f"Processing sequence index: {idx}")
    print(f"Translation offset: {args.translation_offset}m")
    print(f"{'='*60}\n")

    try:
        # Load data
        views = dataset.__getitem__((idx, 2, num_views))

        # 运行Stage1推理
        with torch.no_grad():
            outputs, batch = inference(views, model, device)

        # 转换为vggt batch
        vggt_batch = cut3r_batch_to_vggt(views)

        # Vggt forward
        with torch.no_grad():
            preds = model(
                vggt_batch['images'],
                gt_extrinsics=vggt_batch['extrinsics'],
                gt_intrinsics=vggt_batch['intrinsics'],
                frame_sample_ratio=1.0
            )

        # 处理use_gt_camera参数
        preds_for_dynamic = preds.copy() if isinstance(preds, dict) else preds
        if hasattr(args, 'use_gt_camera') and args.use_gt_camera and 'pose_enc' in preds_for_dynamic:
            from vggt.utils.pose_enc import extri_intri_to_pose_encoding
            gt_extrinsics = vggt_batch['extrinsics']
            gt_intrinsics = vggt_batch['intrinsics']
            image_size_hw = vggt_batch['images'].shape[-2:]

            gt_pose_enc = extri_intri_to_pose_encoding(
                gt_extrinsics, gt_intrinsics, image_size_hw, pose_encoding_type="absT_quaR_FoV"
            )
            preds_for_dynamic['pose_enc'] = gt_pose_enc
            print(f"[INFO] Using GT camera parameters")
        else:
            print(f"[INFO] Using predicted camera parameters")

        # 创建空的辅助模型字典
        auxiliary_models = {}

        # Process dynamic objects
        dynamic_objects_data = dynamic_processor.process_dynamic_objects(
            preds_for_dynamic, vggt_batch, auxiliary_models
        )

        # Extract data
        B, S, C, H, W = vggt_batch['images'].shape

        # Build scene (updated to support cars and people separately)
        dynamic_objects_cars = dynamic_objects_data.get('dynamic_objects_cars', []) if dynamic_objects_data is not None else []
        dynamic_objects_people = dynamic_objects_data.get('dynamic_objects_people', []) if dynamic_objects_data is not None else []
        static_gaussians = dynamic_objects_data.get('static_gaussians') if dynamic_objects_data is not None else None

        scene = {
            'static_gaussians': static_gaussians,
            'dynamic_objects_cars': dynamic_objects_cars,
            'dynamic_objects_people': dynamic_objects_people
        }

        print(f"[INFO] Detected {len(dynamic_objects_cars)} cars, {len(dynamic_objects_people)} people")

        # Get camera parameters
        intrinsics = vggt_batch['intrinsics'][0]  # [S, 3, 3]
        extrinsics_original = vggt_batch['extrinsics'][0]  # [S, 4, 4]

        # Get sky colors
        sky_colors = preds.get('sky_colors', None)
        sampled_frame_indices = preds.get('sampled_frame_indices', None)

        if sky_colors is not None:
            sky_colors = sky_colors[0]  # [num_sampled, 3, H, W]

        # 获取GT的scale factor,将metric尺度的translation_offset转换到非metric尺度
        # vggt_batch['depth_scale_factor'] = 1 / dist_avg，用于将metric尺度归一化
        # 因此要从非metric转到metric需要除以scale_factor，从metric转到非metric需要乘以scale_factor
        depth_scale_factor = vggt_batch.get('depth_scale_factor', None)

        if depth_scale_factor is not None:
            # 用户提供的translation_offset是metric尺度(米)
            # 需要转换到预测的非metric尺度
            offset = args.translation_offset * depth_scale_factor.item()
            print(f"[Scale Info] GT depth_scale_factor: {depth_scale_factor.item():.6f}")
            print(f"[Scale Info] Metric translation offset: {args.translation_offset:.3f}m")
            print(f"[Scale Info] Non-metric translation offset: {offset:.6f}")
        else:
            offset = args.translation_offset
            print(f"[Warning] No depth_scale_factor found, using translation_offset directly: {offset:.3f}")

        # 定义9个视角的相机偏移 (使用OpenCV坐标系: X=右, Y=下, Z=前)
        camera_offsets = {
            'up_left':    (-offset, -offset, 0.0),   # 左上 (X负=左, Y负=上)
            'up':         (0.0, -offset, 0.0),       # 上 (Y负=上)
            'up_right':   (offset, -offset, 0.0),    # 右上 (X正=右, Y负=上)
            'left':       (-offset, 0.0, 0.0),       # 左 (X负=左)
            'center':     (0.0, 0.0, 0.0),           # 中心(原视角)
            'right':      (offset, 0.0, 0.0),        # 右 (X正=右)
            'down_left':  (-offset, offset, 0.0),    # 左下 (X负=左, Y正=下)
            'down':       (0.0, offset, 0.0),        # 下 (Y正=下)
            'down_right': (offset, offset, 0.0),     # 右下 (X正=右, Y正=下)
        }

        # 渲染9个视角
        print("Rendering 9 novel views...")
        rendered_views = {}

        for view_name, (offset_x, offset_y, offset_z) in camera_offsets.items():
            print(f"  Rendering {view_name} view (offset: x={offset_x:.2f}, y={offset_y:.2f}, z={offset_z:.2f})...")

            # 修改相机外参
            extrinsics_modified = modify_camera_extrinsics(
                extrinsics_original, offset_x, offset_y, offset_z
            )

            # 渲染
            rendered_rgb = render_gaussians_with_sky(
                scene, intrinsics, extrinsics_modified,
                sky_colors, sampled_frame_indices, H, W, device
            )

            rendered_views[view_name] = rendered_rgb

        return {
            'rendered_views': rendered_views,
            'num_objects': len(dynamic_objects),
            'success': True
        }

    except Exception as e:
        print(f"Error processing idx {idx}: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def run_batch_inference(model, dataset, dynamic_processor, args, device):
    """运行批量推理"""
    print(f"\n{'='*60}")
    print(f"Batch Inference Mode (NVS)")
    print(f"Range: {args.start_idx} to {args.end_idx}, step {args.step}")
    print(f"{'='*60}\n")

    successful = []
    failed = []

    indices = range(args.start_idx, args.end_idx, args.step)

    for idx in tqdm(indices, desc="Batch processing"):
        result = run_single_inference(model, dataset, dynamic_processor, idx, args.num_views, device, args)

        if result['success']:
            # Save video
            grid_frames = create_nvs_grid(
                result['rendered_views'],
                args.translation_offset
            )

            seq_name = os.path.basename(args.dataset_root)
            output_path = os.path.join(args.output_dir, f"{seq_name}_nvs_idx{idx}.mp4")

            save_video(grid_frames, output_path, fps=args.fps)
            successful.append(idx)
            print(f"✓ idx {idx}: {result['num_objects']} objects")
        else:
            failed.append(idx)
            print(f"✗ idx {idx}: {result['error']}")
            if not args.continue_on_error:
                break

    print(f"\n{'='*60}")
    print(f"Batch complete: {len(successful)} successful, {len(failed)} failed")
    print(f"{'='*60}\n")


def save_video(grid_frames, output_path, fps=10):
    """保存视频"""
    print(f"Saving video to: {output_path}")

    with iio.get_writer(output_path, fps=fps, codec='libx264', quality=8) as writer:
        for frame in grid_frames:
            writer.append_data(frame)

    print(f"Video saved! Frames: {len(grid_frames)}")


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load model and dataset
    model = load_model(args.model_path, device, args)
    dataset = load_dataset(args.dataset_root, args.num_views)

    # Create dynamic processor
    dynamic_processor = OnlineDynamicProcessor(
        device=device,
        velocity_transform_mode=args.velocity_transform_mode,
        velocity_threshold=args.velocity_threshold,
        clustering_eps=args.clustering_eps,
        clustering_min_samples=args.clustering_min_samples,
        min_object_size=args.min_object_size,
        tracking_position_threshold=args.tracking_position_threshold,
        tracking_velocity_threshold=args.tracking_velocity_threshold
    )

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Run inference
    with tf32_off():
        if args.batch_mode:
            run_batch_inference(model, dataset, dynamic_processor, args, device)
        else:
            result = run_single_inference(model, dataset, dynamic_processor, args.idx, args.num_views, device, args)

            if result['success']:
                grid_frames = create_nvs_grid(
                    result['rendered_views'],
                    args.translation_offset
                )

                seq_name = os.path.basename(args.dataset_root)
                output_path = os.path.join(args.output_dir, f"{seq_name}_nvs_idx{args.idx}.mp4")

                save_video(grid_frames, output_path, fps=args.fps)

                print(f"\n{'='*60}")
                print(f"Success! Objects: {result['num_objects']}")
                print(f"Output: {output_path}")
                print(f"{'='*60}\n")
            else:
                print(f"Failed: {result['error']}")


if __name__ == "__main__":
    main()
