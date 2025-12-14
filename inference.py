#!/usr/bin/env python3
"""
Complete Stage1 Inference Script
输出 4x2 布局的可视化结果（参考demo_stage2_inference.py的正确实现）：
- Row 1: GT RGB | Rendered RGB (with sky)
- Row 2: GT Depth | Rendered Depth
- Row 3: GT Velocity | GT RGB + Pred Velocity 融合 (加权叠加)
- Row 4: GT Segmentation | Pred Segmentation
- Row 5: Dynamic Clustering (full width)
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
from contextlib import contextmanager
from lpips import LPIPS
from skimage.metrics import structural_similarity

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/vggt'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# from add_ckpt_path import add_path_to_dust3r
from vggt.models.vggt import VGGT
from dust3r.utils.misc import tf32_off
from src.dust3r.inference import inference
from src.train import cut3r_batch_to_vggt
from src.online_dynamic_processor import OnlineDynamicProcessor
from vggt.training.loss import self_render_and_loss
from vggt.training.stage2_loss import prune_gaussians_by_voxel


# ========== Metric Computation Functions (from NoPoSplat/evaluation/metrics.py) ==========

@contextmanager
def tf32_metric_off():
    """Temporarily disable TF32 for metric computation"""
    original = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original


_lpips_cache = {}

def get_lpips(device: torch.device) -> LPIPS:
    """Get or create LPIPS model"""
    if device not in _lpips_cache:
        _lpips_cache[device] = LPIPS(net="vgg").to(device)
    return _lpips_cache[device]


@torch.no_grad()
def compute_psnr(ground_truth, predicted):
    """
    Compute PSNR between ground truth and predicted images

    Args:
        ground_truth: [batch, channel, height, width] tensor
        predicted: [batch, channel, height, width] tensor

    Returns:
        [batch] tensor of PSNR values
    """
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = ((ground_truth - predicted) ** 2).mean(dim=[1, 2, 3])
    return -10 * mse.log10()


@torch.no_grad()
def compute_lpips(ground_truth, predicted):
    """
    Compute LPIPS between ground truth and predicted images

    Args:
        ground_truth: [batch, channel, height, width] tensor
        predicted: [batch, channel, height, width] tensor

    Returns:
        [batch] tensor of LPIPS values
    """
    value = get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    return value[:, 0, 0, 0]


@torch.no_grad()
def compute_ssim(ground_truth, predicted):
    """
    Compute SSIM between ground truth and predicted images

    Args:
        ground_truth: [batch, channel, height, width] tensor
        predicted: [batch, channel, height, width] tensor

    Returns:
        [batch] tensor of SSIM values
    """
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)

# ========== End of Metric Computation Functions ==========


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Complete Stage1 Inference with Visualization")

    # 基础参数
    parser.add_argument("--model_path", type=str, required=True, help="Path to Stage1 model checkpoint")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument("--output_dir", type=str, default="./inference_outputs", help="Output directory")
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

    # 可视化参数
    parser.add_argument("--velocity_alpha", type=float, default=1.0,
                       help="Weight for pred velocity in fusion (0-1), default 0.5 means 50% each")

    # VGGT模型配置参数
    parser.add_argument("--sh_degree", type=int, default=0, help="Spherical harmonics degree")
    parser.add_argument("--use_gs_head", action="store_true", default=True, help="Use DPTGSHead for gaussian_head")
    parser.add_argument("--use_gs_head_velocity", action="store_true", default=False, help="Use DPTGSHead for velocity_head")
    parser.add_argument("--use_gt_camera", action="store_true", help="Use GT camera parameters")

    return parser.parse_args()


def load_model(model_path, device, args):
    """加载Stage1模型（参考demo_stage2_inference.py）"""
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


def fix_views_data_types(views, device):
    """修复views中的数据类型，确保所有数组都是tensor（参考demo_stage1_inference.py）"""
    fixed_views = []

    for i, view in enumerate(views):
        fixed_view = {}
        for key, value in view.items():
            if key in ['img', 'depthmap', 'camera_intrinsics', 'camera_pose', 'valid_mask', 'pts3d']:
                # 这些字段需要是tensor
                if value is not None:
                    tensor_value = ensure_tensor(value, device)
                    fixed_view[key] = tensor_value
                else:
                    fixed_view[key] = value
            else:
                # 其他字段保持原样
                fixed_view[key] = value

        fixed_views.append(fixed_view)

    return fixed_views


def render_gaussians_with_sky(scene, intrinsics, extrinsics, sky_colors, sampled_frame_indices, H, W, device,
                              enable_voxel_pruning=True, voxel_size=0.002, depth_scale_factor=None):
    """
    渲染gaussian场景（与train.py的Stage2RenderLoss相同逻辑）
    包括sky_color的alpha blending合成
    逐帧渲染以正确处理动态物体的变换

    Args:
        enable_voxel_pruning: bool 是否启用voxel剪枝
        voxel_size: float voxel大小（metric尺度，单位米）
        depth_scale_factor: float 深度缩放因子
    """
    from gsplat import rasterization
    import torch.nn.functional as F

    S = intrinsics.shape[0]
    rendered_images = []
    rendered_depths = []

    # 逐帧渲染（参考stage2_loss.py的_render_frame逻辑）
    for frame_idx in range(S):
        # 每帧收集gaussians
        all_means = []
        all_scales = []
        all_colors = []
        all_rotations = []
        all_opacities = []

        # Static gaussians (参考stage2_loss.py line 493-499)
        if scene.get('static_gaussians') is not None:
            static_gaussians = scene['static_gaussians']  # [N, 14]
            if static_gaussians.shape[0] > 0:
                all_means.append(static_gaussians[:, :3])
                all_scales.append(static_gaussians[:, 3:6])
                all_colors.append(static_gaussians[:, 6:9].unsqueeze(-2))
                all_rotations.append(static_gaussians[:, 9:13])
                all_opacities.append(static_gaussians[:, 13])

        # Dynamic objects - Cars (使用canonical空间+变换，参考stage2_loss.py line 436-467)
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
                # 如果没有变换信息，直接使用原始Gaussians（假设当前帧就是参考帧）
                transformed_gaussians = canonical_gaussians
            else:
                # 应用变换：将canonical空间的Gaussians变换到当前帧
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

        # Dynamic objects - People (每帧单独的Gaussians，不使用变换，参考stage2_loss.py line 469-487)
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
            rendered_depths.append(torch.zeros(H, W, device=device))
            continue

        # Concatenate
        means = torch.cat(all_means, dim=0)  # [N, 3]
        scales = torch.cat(all_scales, dim=0)  # [N, 3]
        colors = torch.cat(all_colors, dim=0)  # [N, 1, 3]
        rotations = torch.cat(all_rotations, dim=0)  # [N, 4]
        opacities = torch.cat(all_opacities, dim=0)  # [N]

        # Apply voxel pruning (如果启用)
        if enable_voxel_pruning and means.shape[0] > 0:
            means, scales, rotations, opacities, colors = prune_gaussians_by_voxel(
                means, scales, rotations, opacities, colors,
                voxel_size=voxel_size,
                depth_scale_factor=depth_scale_factor
            )

        # Fix NaN/Inf (参考stage2_loss.py line 550-555)
        means = torch.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
        scales = torch.nan_to_num(scales, nan=0.01, posinf=1.0, neginf=0.01)
        colors = torch.nan_to_num(colors, nan=0.5, posinf=1.0, neginf=0.0)
        rotations = torch.nan_to_num(rotations, nan=0.0, posinf=1.0, neginf=-1.0)
        opacities = torch.nan_to_num(opacities, nan=0.5, posinf=1.0, neginf=0.0)

        K = intrinsics[frame_idx]
        w2c = extrinsics[frame_idx]

        try:
            # 参考stage2_loss.py line 562-569
            render_result = rasterization(
                means, rotations, scales, opacities, colors,
                w2c.unsqueeze(0), K.unsqueeze(0), W, H,
                sh_degree=0, render_mode="RGB+ED",
                radius_clip=0, near_plane=0.0001,
                far_plane=1000.0,
                eps2d=0.3,
            )

            rendered_image = render_result[0][0, :, :, :3].permute(2, 0, 1)
            rendered_depth = render_result[0][0, :, :, -1]
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
            rendered_depths.append(rendered_depth)

        except Exception as e:
            print(f"Error rendering frame {frame_idx}: {e}")
            rendered_images.append(torch.zeros(3, H, W, device=device))
            rendered_depths.append(torch.zeros(H, W, device=device))

    return torch.stack(rendered_images, dim=0), torch.stack(rendered_depths, dim=0)


def render_self(vggt_batch, preds, sh_degree=0, enable_voxel_pruning=True, voxel_size=0.002):
    """
    调用self_render_and_loss生成自渲染结果（用于可视化）

    Returns:
        rendered_rgb: [S, 3, H, W] 渲染的RGB图像
        rendered_depth: [S, H, W] 渲染的深度图像
    """
    # 调用self_render_and_loss
    _, img_dict = self_render_and_loss(
        vggt_batch, preds,
        sampled_frame_indices=None,  # 渲染所有帧
        sh_degree=sh_degree,
        enable_voxel_pruning=enable_voxel_pruning,
        voxel_size=voxel_size
    )

    # 提取渲染结果
    rendered_rgb = img_dict['self_rgb_pred']  # [S, 3, H, W]
    rendered_depth = img_dict['self_depth_pred'].squeeze(1)  # [S, H, W]

    return rendered_rgb, rendered_depth


def _object_exists_in_frame(obj_data, frame_idx):
    """检查动态物体是否在指定帧中存在（参考stage2_loss.py line 650-655）"""
    if 'frame_transforms' in obj_data:
        frame_transforms = obj_data['frame_transforms']
        if frame_idx in frame_transforms:
            return True
    return False


def _get_object_transform_to_frame(obj_data, frame_idx):
    """获取从canonical空间到指定帧的变换矩阵（参考stage2_loss.py line 657-680）"""
    # 检查参考帧
    reference_frame = obj_data.get('reference_frame', 0)
    if frame_idx == reference_frame:
        # 如果要渲染的就是参考帧（canonical帧），不需要变换
        return None

    # 获取变换：frame_transforms存储的是从各帧到reference_frame的变换
    # 但我们需要从reference_frame到frame_idx的变换，所以需要求逆
    if 'frame_transforms' in obj_data:
        frame_transforms = obj_data['frame_transforms']
        if frame_idx in frame_transforms:
            # 存储的变换：frame_idx -> reference_frame
            # 我们需要的变换：reference_frame -> frame_idx（即逆变换）
            frame_to_canonical = frame_transforms[frame_idx]  # [4, 4] 变换矩阵
            canonical_to_frame = torch.inverse(frame_to_canonical)
            return canonical_to_frame

    return None


def _apply_transform_to_gaussians(gaussians, transform):
    """将变换应用到Gaussian参数（参考stage2_loss.py line 768-802）"""
    # gaussians: [N, 14] - [xyz(3), scale(3), color(3), quat(4), opacity(1)]
    # transform: [4, 4] 变换矩阵

    # 检查变换矩阵是否异常
    if torch.allclose(transform, torch.zeros_like(transform), atol=1e-6):
        print(f"⚠️  检测到零变换矩阵！使用单位矩阵替代")
        transform = torch.eye(4, dtype=transform.dtype, device=transform.device)
    else:
        # 转换为float32以支持torch.det
        det_val = torch.det(transform[:3, :3].float()).abs()
        if det_val < 1e-8:
            print(f"⚠️  变换矩阵奇异(det={det_val:.2e})！")

    transformed_gaussians = gaussians.clone()

    # 变换位置（参考stage2_loss.py line 794-800）
    positions = gaussians[:, :3]  # [N, 3]
    positions_homo = torch.cat([positions, torch.ones(
        positions.shape[0], 1, device=positions.device)], dim=1)  # [N, 4]
    transformed_positions = torch.mm(
        transform, positions_homo.T).T[:, :3]  # [N, 3]
    transformed_gaussians[:, :3] = transformed_positions

    # 注意：stage2_loss.py 中只变换了位置，没有变换旋转和尺度
    # 这是简化处理，完整实现需要变换quaternion和scale

    return transformed_gaussians


def visualize_velocity(velocity, scale=0.2):
    """可视化velocity为RGB图像"""
    from dust3r.utils.image import scene_flow_to_rgb

    S, H, W, _ = velocity.shape
    velocity_rgb = scene_flow_to_rgb(velocity.detach(), scale).permute(0, 3, 1, 2)
    return velocity_rgb


def visualize_depth(depth):
    """可视化depth为RGB图像（viridis colormap）"""
    S, H, W = depth.shape
    depth_vis = []

    for s in range(S):
        d = depth[s].detach().cpu().numpy()
        d_min, d_max = d.min(), d.max()
        if d_max > d_min:
            d_norm = (d - d_min) / (d_max - d_min)
        else:
            d_norm = d * 0

        colored = cm.viridis(d_norm)[:, :, :3]
        depth_vis.append(torch.from_numpy(colored).permute(2, 0, 1))

    return torch.stack(depth_vis, dim=0)


def visualize_clustering_results(clustering_results, num_colors=20):
    """
    为聚类结果生成可视化颜色（参考demo_stage2_inference.py line 612-662）
    """
    import matplotlib.pyplot as plt

    # 生成颜色映射
    colors = plt.cm.tab20(np.linspace(0, 1, num_colors))  # 使用tab20颜色映射
    colors = (colors[:, :3] * 255).astype(np.uint8)  # 转换为0-255范围

    colored_results = []

    for frame_result in clustering_results:
        points = frame_result['points']  # [H*W, 3]
        labels = frame_result['labels']  # [H*W]
        global_ids = frame_result.get('global_ids', [])

        # 初始化颜色数组（默认黑色背景）
        point_colors = np.zeros((len(points), 3), dtype=np.uint8)

        # 为每个聚类分配颜色（基于全局ID）
        unique_labels = torch.unique(labels)
        colors_assigned = 0
        for label in unique_labels:
            if label == -1:
                continue  # 跳过噪声点

            label_val = label.item()
            if label_val < len(global_ids):
                global_id = global_ids[label_val]
                if global_id != -1:
                    color_idx = global_id % num_colors
                    color = colors[color_idx]

                    mask = labels == label
                    point_colors[mask] = color
                    colors_assigned += 1

        colored_results.append({
            'points': points,
            'colors': point_colors,
            'num_clusters': colors_assigned
        })

    return colored_results


def create_clustering_visualization(matched_clustering_results, vggt_batch, fusion_alpha=0.7):
    """
    从matched_clustering_results创建可视化图像（参考demo_stage2_inference.py line 663-735）
    """
    import cv2

    try:
        if not matched_clustering_results or len(matched_clustering_results) == 0:
            # 返回源RGB图像
            B, S, C, H, W = vggt_batch["images"].shape
            return vggt_batch["images"][0]  # [S, 3, H, W]

        # 生成可视化颜色
        colored_results = visualize_clustering_results(matched_clustering_results, num_colors=20)

        B, S, C, image_height, image_width = vggt_batch["images"].shape

        # 将聚类结果与源RGB图像融合
        clustering_images = []
        for frame_idx, colored_result in enumerate(colored_results):
            # 获取源RGB图像
            source_rgb = vggt_batch["images"][0, frame_idx].permute(1, 2, 0)  # [H, W, 3]
            source_rgb = (source_rgb * 255).cpu().numpy().astype(np.uint8)  # 转换为0-255范围

            # 检查是否有动态物体
            if colored_result['num_clusters'] > 0:
                # 将点云颜色重塑为图像格式
                point_colors = colored_result['colors']  # [H*W, 3]
                clustering_image = point_colors.reshape(image_height, image_width, 3)  # [H, W, 3]

                # 融合：将动态聚类结果叠加到源RGB图像上
                # 只有非黑色（非静态）的点才覆盖源图像
                mask = np.any(clustering_image > 0, axis=2)  # [H, W] 布尔掩码，True表示动态点
                mask = mask[:, :, np.newaxis]  # [H, W, 1] 扩展维度

                # 透明度混合：动态点使用聚类颜色与源RGB混合，静态点使用源RGB
                fused_image = np.where(mask,
                                     (fusion_alpha * clustering_image + (1 - fusion_alpha) * source_rgb).astype(np.uint8),
                                     source_rgb)
            else:
                # 没有动态物体时，显示源RGB图像并添加文本提示
                fused_image = source_rgb.copy()
                # 在图像上添加文本提示
                cv2.putText(fused_image, "No Dynamic Objects", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            clustering_images.append(fused_image)

        # 转换为tensor格式，并归一化到0-1范围（与其他可视化数据一致）
        clustering_tensor = torch.stack([torch.from_numpy(img) for img in clustering_images], dim=0)  # [S, H, W, 3]
        clustering_tensor = clustering_tensor.float() / 255.0  # 归一化到0-1范围
        clustering_tensor = clustering_tensor.permute(0, 3, 1, 2)  # [S, 3, H, W]

        return clustering_tensor

    except Exception as e:
        print(f"Error creating clustering visualization: {e}")
        import traceback
        traceback.print_exc()
        # 返回源RGB图像作为备用
        B, S, C, H, W = vggt_batch["images"].shape
        return vggt_batch["images"][0]  # [S, 3, H, W]


def get_segmentation_colormap(num_classes=4):
    """生成分割颜色映射表（基于 Waymo 类别和 Cityscapes 风格）

    Waymo 类别定义（4类）:
        0: background/unlabeled
        1: vehicle
        2: sign
        3: pedestrian + cyclist

    Args:
        num_classes: 类别数量（默认4）

    Returns:
        colormap: [num_classes, 3] numpy array of RGB colors
    """
    colormap = np.zeros((num_classes, 3), dtype=np.uint8)

    # 基于 Cityscapes 风格的颜色方案
    colormap[0] = [255, 255, 255]     # background/unlabeled - 白色
    colormap[1] = [0, 0, 142]         # vehicle - 深蓝色（Cityscapes car color）
    colormap[2] = [220, 220, 0]       # sign - 黄色（Cityscapes traffic sign color）
    colormap[3] = [220, 20, 60]       # pedestrian+cyclist - 红色（Cityscapes person color）

    return colormap


def visualize_segmentation(seg_labels, seg_mask=None, num_classes=4):
    """将分割标签可视化为RGB图像

    Args:
        seg_labels: [S, H, W] or [H, W] - Segmentation labels (class indices)
        seg_mask: [S, H, W] or [H, W] - Optional validity mask (1=valid, 0=invalid)
        num_classes: Number of segmentation classes (default 4 for Waymo)

    Returns:
        seg_rgb: [S, 3, H, W] - RGB visualization
    """
    colormap = get_segmentation_colormap(num_classes)

    # Convert labels to numpy if needed
    if isinstance(seg_labels, torch.Tensor):
        seg_labels = seg_labels.cpu().numpy()

    if seg_mask is not None and isinstance(seg_mask, torch.Tensor):
        seg_mask = seg_mask.cpu().numpy()

    # Handle both [S, H, W] and [H, W] shapes
    if seg_labels.ndim == 2:
        # [H, W] -> add batch dimension -> [1, H, W]
        seg_labels = seg_labels[np.newaxis, ...]
        if seg_mask is not None:
            seg_mask = seg_mask[np.newaxis, ...]
    elif seg_labels.ndim != 3:
        raise ValueError(f"Expected seg_labels to have 2 or 3 dimensions, got {seg_labels.ndim} (shape: {seg_labels.shape})")

    S, H, W = seg_labels.shape
    seg_rgb = np.zeros((S, H, W, 3), dtype=np.float32)

    # Map each class to its color
    for s in range(S):
        for class_id in range(num_classes):
            mask = (seg_labels[s] == class_id)
            seg_rgb[s][mask] = colormap[class_id] / 255.0

        # Set invalid regions to gray if mask is provided
        if seg_mask is not None:
            invalid_mask = (seg_mask[s] == 0)
            seg_rgb[s][invalid_mask] = [0.5, 0.5, 0.5]  # Gray for invalid regions

    # Convert to [S, 3, H, W] format
    seg_rgb = torch.from_numpy(seg_rgb).permute(0, 3, 1, 2)

    return seg_rgb


def create_visualization_grid(gt_rgb, rendered_rgb, gt_depth, rendered_depth,
                              gt_velocity, pred_velocity, clustering_vis,
                              gt_seg=None, pred_seg=None,
                              self_render_rgb=None, velocity_alpha=0.5):
    """创建5行可视化网格

    Args:
        velocity_alpha: Pred velocity在融合中的权重 (0-1)，默认0.5表示各占50%
        self_render_rgb: [S, 3, H, W] self_render的RGB渲染结果
        gt_seg: [S, 3, H, W] GT segmentation可视化
        pred_seg: [S, 3, H, W] Pred segmentation可视化
    """
    S = gt_rgb.shape[0]
    _, H, W = gt_rgb.shape[1:]

    grid_frames = []

    for s in range(S):
        # Convert to numpy [H, W, 3] (添加 detach 避免梯度错误)
        gt_rgb_np = gt_rgb[s].detach().permute(1, 2, 0).cpu().numpy()
        rendered_rgb_np = rendered_rgb[s].detach().permute(1, 2, 0).cpu().numpy()
        gt_depth_np = gt_depth[s].detach().permute(1, 2, 0).cpu().numpy()
        rendered_depth_np = rendered_depth[s].detach().permute(1, 2, 0).cpu().numpy()
        gt_velocity_np = gt_velocity[s].detach().permute(1, 2, 0).cpu().numpy()
        pred_velocity_np = pred_velocity[s].detach().permute(1, 2, 0).cpu().numpy()
        clustering_vis_np = clustering_vis[s].detach().permute(1, 2, 0).cpu().numpy()

        # 创建GT RGB和Pred Velocity的加权融合图像
        fused_velocity_np = velocity_alpha * pred_velocity_np + (1 - velocity_alpha) * gt_rgb_np
        fused_velocity_np = np.clip(fused_velocity_np, 0, 1)

        # 准备segmentation可视化（如果提供）
        if gt_seg is not None and pred_seg is not None:
            gt_seg_np = gt_seg[s].detach().permute(1, 2, 0).cpu().numpy()
            pred_seg_np = pred_seg[s].detach().permute(1, 2, 0).cpu().numpy()
        else:
            # 如果没有提供segmentation，使用黑色占位符
            gt_seg_np = np.zeros_like(gt_rgb_np)
            pred_seg_np = np.zeros_like(gt_rgb_np)

        # 准备self_render可视化（如果提供）
        if self_render_rgb is not None:
            self_render_np = self_render_rgb[s].detach().permute(1, 2, 0).cpu().numpy()
        else:
            # 如果没有提供self_render，使用黑色占位符
            self_render_np = np.zeros_like(gt_rgb_np)

        # Create grid: 5 rows x 2 columns
        row1 = np.concatenate([gt_rgb_np, rendered_rgb_np], axis=1)
        row2 = np.concatenate([gt_depth_np, rendered_depth_np], axis=1)
        # Row 3: GT Velocity | GT RGB + Pred Velocity 融合
        row3 = np.concatenate([gt_velocity_np, fused_velocity_np], axis=1)
        # Row 4: GT Segmentation | Pred Segmentation
        row4 = np.concatenate([gt_seg_np, pred_seg_np], axis=1)
        # Row 5: Dynamic Clustering | Self Render
        row5 = np.concatenate([clustering_vis_np, self_render_np], axis=1)

        grid = np.concatenate([row1, row2, row3, row4, row5], axis=0)
        grid = (np.clip(grid, 0, 1) * 255).astype(np.uint8)

        grid_frames.append(grid)

    return grid_frames


def compute_metrics(gt_rgb, rendered_rgb):
    """
    计算 GT RGB 和 Rendered RGB 之间的 photometric 指标

    Args:
        gt_rgb: [S, 3, H, W] tensor, GT RGB images
        rendered_rgb: [S, 3, H, W] tensor, Rendered RGB images

    Returns:
        dict with per-frame and average metrics
    """
    # Ensure inputs are in [0, 1] range
    gt_rgb = torch.clamp(gt_rgb, 0, 1)
    rendered_rgb = torch.clamp(rendered_rgb, 0, 1)

    # Compute metrics (input format: [batch, channel, height, width])
    psnr_values = compute_psnr(gt_rgb, rendered_rgb)  # [S]
    ssim_values = compute_ssim(gt_rgb, rendered_rgb)  # [S]
    lpips_values = compute_lpips(gt_rgb, rendered_rgb)  # [S]

    # Convert to numpy for easier handling
    psnr_values = psnr_values.detach().cpu().numpy()
    ssim_values = ssim_values.detach().cpu().numpy()
    lpips_values = lpips_values.detach().cpu().numpy()

    return {
        'psnr': psnr_values,
        'ssim': ssim_values,
        'lpips': lpips_values,
        'psnr_mean': np.mean(psnr_values),
        'ssim_mean': np.mean(ssim_values),
        'lpips_mean': np.mean(lpips_values)
    }


def run_single_inference(model, dataset, dynamic_processor, idx, num_views, device, args=None):
    """运行单次推理"""
    print(f"\n{'='*60}")
    print(f"Processing sequence index: {idx}")
    print(f"{'='*60}\n")

    try:
        # Load data（参考demo_stage2_inference.py）
        views = dataset.__getitem__((idx, 2, num_views))

        # 运行Stage1推理（参考demo_stage2_inference.py）
        with torch.no_grad():
            outputs, batch = inference(views, model, device)

        # 转换为vggt batch（参考demo_stage2_inference.py）
        vggt_batch = cut3r_batch_to_vggt(views)

        # Vggt forward（参考demo_stage2_inference.py）
        # 为了获取sky_colors，需要传入gt_extrinsics和gt_intrinsics
        # 同时设置frame_sample_ratio=1确保所有帧都被采样
        with torch.no_grad():
            preds = model(
                vggt_batch['images'],
                gt_extrinsics=vggt_batch['extrinsics'],
                gt_intrinsics=vggt_batch['intrinsics'],
                frame_sample_ratio=1.0
            )

        # 处理use_gt_camera参数（参考demo_stage2_inference.py）
        preds_for_dynamic = preds.copy() if isinstance(preds, dict) else preds
        if hasattr(args, 'use_gt_camera') and args.use_gt_camera and 'pose_enc' in preds_for_dynamic:
            # 使用GT相机参数替换预测的pose_enc
            from vggt.utils.pose_enc import extri_intri_to_pose_encoding
            gt_extrinsics = vggt_batch['extrinsics']  # [B, S, 4, 4]
            gt_intrinsics = vggt_batch['intrinsics']  # [B, S, 3, 3]
            image_size_hw = vggt_batch['images'].shape[-2:]

            gt_pose_enc = extri_intri_to_pose_encoding(
                gt_extrinsics, gt_intrinsics, image_size_hw, pose_encoding_type="absT_quaR_FoV"
            )
            preds_for_dynamic['pose_enc'] = gt_pose_enc
            print(f"[INFO] Using GT camera parameters for dynamic object processing")
        else:
            print(f"[INFO] Using predicted camera parameters for dynamic object processing")

        # 创建空的辅助模型字典（参考demo_stage2_inference.py）
        auxiliary_models = {}

        # Process dynamic objects（参考demo_stage2_inference.py）
        dynamic_objects_data = dynamic_processor.process_dynamic_objects(
            preds_for_dynamic, vggt_batch, auxiliary_models
        )

        # Extract data
        B, S, C, H, W = vggt_batch['images'].shape

        gt_rgb = vggt_batch['images'][0]  # [S, 3, H, W]
        # gt_depth should be [S, H, W]
        if 'depths' in vggt_batch and vggt_batch['depths'] is not None:
            gt_depth = vggt_batch['depths'][0]  # [S, H, W]
        else:
            gt_depth = torch.ones(S, H, W, device=device) * 5.0

        gt_velocity = vggt_batch.get('flowmap', None)

        if gt_velocity is not None:
            gt_velocity = gt_velocity[0, :, :, :, :3]
            # Apply same coordinate transformation as pred_velocity
            gt_velocity = gt_velocity[:, :, :, [2, 0, 1]]
            gt_velocity[:, :, :, 2] = -gt_velocity[:, :, :, 2]
        else:
            gt_velocity = torch.zeros(S, H, W, 3, device=device)

        pred_velocity = preds.get('velocity', torch.zeros(1, S, H, W, 3, device=device))[0]

        # velocity已在模型forward中激活，这里直接使用
        pred_velocity = pred_velocity[:, :, :, [2, 0, 1]]
        pred_velocity[:, :, :, 2] = -pred_velocity[:, :, :, 2]

        # Build scene (updated to support cars and people separately)
        dynamic_objects_cars = dynamic_objects_data.get('dynamic_objects_cars', []) if dynamic_objects_data is not None else []
        dynamic_objects_people = dynamic_objects_data.get('dynamic_objects_people', []) if dynamic_objects_data is not None else []
        static_gaussians = dynamic_objects_data.get('static_gaussians') if dynamic_objects_data is not None else None

        scene = {
            'static_gaussians': static_gaussians,
            'dynamic_objects_cars': dynamic_objects_cars,
            'dynamic_objects_people': dynamic_objects_people
        }

        # Log object counts
        print(f"[INFO] Detected {len(dynamic_objects_cars)} cars, {len(dynamic_objects_people)} people")

        # Get camera parameters
        intrinsics = vggt_batch['intrinsics'][0]
        extrinsics = vggt_batch['extrinsics'][0]

        # Get sky colors
        sky_colors_full = preds.get('sky_colors', None)  # [B, num_sampled, 3, H, W]
        sampled_frame_indices = preds.get('sampled_frame_indices', None)

        # Extract sky colors for aggregator rendering (remove batch dimension)
        if sky_colors_full is not None:
            sky_colors = sky_colors_full[0]  # [num_sampled, 3, H, W]
        else:
            sky_colors = None

        # Get depth_scale_factor for voxel pruning
        depth_scale_factor = vggt_batch.get('depth_scale_factor', None)
        if depth_scale_factor is not None and torch.is_tensor(depth_scale_factor):
            depth_scale_factor = depth_scale_factor.item()

        # Render aggregator (with voxel pruning)
        print("Rendering gaussians with sky (aggregator render with voxel)...")
        rendered_rgb, rendered_depth = render_gaussians_with_sky(
            scene, intrinsics, extrinsics, sky_colors, sampled_frame_indices, H, W, device,
            enable_voxel_pruning=False, voxel_size=0.05, depth_scale_factor=depth_scale_factor
        )

        # Render self (with voxel pruning)
        # Note: preds still contains the full sky_colors [B, num_frames, 3, H, W] for self_render_and_loss
        print("Rendering self...")
        self_render_rgb, self_render_depth = render_self(
            vggt_batch, preds,
            sh_degree=args.sh_degree if hasattr(args, 'sh_degree') else 0,
            enable_voxel_pruning=True,
            voxel_size=0.05
        )

        # Compute photometric metrics
        print("Computing photometric metrics...")
        metrics = compute_metrics(gt_rgb, rendered_rgb)

        # Print per-frame metrics
        print(f"\n{'='*60}")
        print("Per-frame Metrics (GT RGB vs Rendered RGB with sky):")
        print(f"{'='*60}")
        for i in range(len(metrics['psnr'])):
            print(f"Frame {i:2d} - PSNR: {metrics['psnr'][i]:.4f}, SSIM: {metrics['ssim'][i]:.4f}, LPIPS: {metrics['lpips'][i]:.4f}")

        # Print average metrics
        print(f"\n{'='*60}")
        print("Average Metrics:")
        print(f"{'='*60}")
        print(f"PSNR:  {metrics['psnr_mean']:.4f}")
        print(f"SSIM:  {metrics['ssim_mean']:.4f}")
        print(f"LPIPS: {metrics['lpips_mean']:.4f}")
        print(f"{'='*60}\n")

        # Visualize
        print("Creating visualizations...")
        gt_velocity_vis = visualize_velocity(gt_velocity, scale=0.1)
        pred_velocity_vis = visualize_velocity(pred_velocity, scale=0.1)
        gt_depth_vis = visualize_depth(gt_depth)  # gt_depth is already [S, H, W]
        rendered_depth_vis = visualize_depth(rendered_depth)

        # Create Segmentation visualization
        print("Creating segmentation visualizations...")
        # Get GT segmentation
        gt_seg_labels = vggt_batch.get('segment_label', None)
        gt_seg_mask = vggt_batch.get('segment_mask', None)

        # Get predicted segmentation
        pred_seg_logits = preds.get('segment_logits', None)

        if gt_seg_labels is not None and pred_seg_logits is not None:
            # Visualize GT segmentation
            gt_seg_vis = visualize_segmentation(gt_seg_labels, gt_seg_mask, num_classes=4)

            # Visualize predicted segmentation
            pred_seg_probs = torch.softmax(pred_seg_logits[0], dim=-1)  # [S, H, W, 4]
            pred_seg_labels = torch.argmax(pred_seg_probs, dim=-1)  # [S, H, W]
            pred_seg_vis = visualize_segmentation(pred_seg_labels, num_classes=4)
        else:
            # If segmentation not available, set to None
            gt_seg_vis = None
            pred_seg_vis = None
            if gt_seg_labels is None:
                print("Warning: No GT segmentation found")
            if pred_seg_logits is None:
                print("Warning: No predicted segmentation found")

        # Create Dynamic Clustering visualization（参考demo_stage2_inference.py）
        matched_clustering_results = dynamic_objects_data.get('matched_clustering_results', []) if dynamic_objects_data is not None else []
        clustering_vis = create_clustering_visualization(matched_clustering_results, vggt_batch)

        return {
            'gt_rgb': gt_rgb,
            'rendered_rgb': rendered_rgb,
            'gt_depth': gt_depth_vis,
            'rendered_depth': rendered_depth_vis,
            'gt_velocity': gt_velocity_vis,
            'pred_velocity': pred_velocity_vis,
            'gt_seg': gt_seg_vis,
            'pred_seg': pred_seg_vis,
            'clustering': clustering_vis,
            'self_render_rgb': self_render_rgb,
            'num_objects': len(dynamic_objects_cars) + len(dynamic_objects_people),
            'num_cars': len(dynamic_objects_cars),
            'num_people': len(dynamic_objects_people),
            'metrics': metrics,
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
    print(f"Batch Inference Mode")
    print(f"Range: {args.start_idx} to {args.end_idx}, step {args.step}")
    print(f"{'='*60}\n")

    successful = []
    failed = []
    all_metrics = []

    indices = range(args.start_idx, args.end_idx, args.step)

    for idx in tqdm(indices, desc="Batch processing"):
        result = run_single_inference(model, dataset, dynamic_processor, idx, args.num_views, device, args)

        if result['success']:
            # Save video
            grid_frames = create_visualization_grid(
                result['gt_rgb'], result['rendered_rgb'],
                result['gt_depth'], result['rendered_depth'],
                result['gt_velocity'], result['pred_velocity'],
                result['clustering'],
                gt_seg=result.get('gt_seg'),
                pred_seg=result.get('pred_seg'),
                self_render_rgb=result.get('self_render_rgb'),
                velocity_alpha=args.velocity_alpha
            )

            seq_name = os.path.basename(args.dataset_root)
            output_path = os.path.join(args.output_dir, f"{seq_name}_idx{idx}.mp4")

            save_video(grid_frames, output_path, fps=args.fps)
            successful.append(idx)
            all_metrics.append(result['metrics'])
            print(f"✓ idx {idx}: {result['num_objects']} objects, PSNR: {result['metrics']['psnr_mean']:.4f}, SSIM: {result['metrics']['ssim_mean']:.4f}, LPIPS: {result['metrics']['lpips_mean']:.4f}")
        else:
            failed.append(idx)
            print(f"✗ idx {idx}: {result['error']}")
            if not args.continue_on_error:
                break

    print(f"\n{'='*60}")
    print(f"Batch complete: {len(successful)} successful, {len(failed)} failed")

    # Compute overall average metrics
    if all_metrics:
        avg_psnr = np.mean([m['psnr_mean'] for m in all_metrics])
        avg_ssim = np.mean([m['ssim_mean'] for m in all_metrics])
        avg_lpips = np.mean([m['lpips_mean'] for m in all_metrics])
        print(f"\nOverall Average Metrics across all sequences:")
        print(f"PSNR:  {avg_psnr:.4f}")
        print(f"SSIM:  {avg_ssim:.4f}")
        print(f"LPIPS: {avg_lpips:.4f}")
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

    # import debugpy
    # debugpy.listen(5697)
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()

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
                grid_frames = create_visualization_grid(
                    result['gt_rgb'], result['rendered_rgb'],
                    result['gt_depth'], result['rendered_depth'],
                    result['gt_velocity'], result['pred_velocity'],
                    result['clustering'],
                    gt_seg=result.get('gt_seg'),
                    pred_seg=result.get('pred_seg'),
                    self_render_rgb=result.get('self_render_rgb'),
                    velocity_alpha=args.velocity_alpha
                )

                seq_name = os.path.basename(args.dataset_root)
                output_path = os.path.join(args.output_dir, f"{seq_name}_idx{args.idx}.mp4")

                save_video(grid_frames, output_path, fps=args.fps)

                print(f"\n{'='*60}")
                print(f"Success! Objects: {result['num_objects']}")
                print(f"Metrics - PSNR: {result['metrics']['psnr_mean']:.4f}, SSIM: {result['metrics']['ssim_mean']:.4f}, LPIPS: {result['metrics']['lpips_mean']:.4f}")
                print(f"Output: {output_path}")
                print(f"{'='*60}\n")
            else:
                print(f"Failed: {result['error']}")


if __name__ == "__main__":
    main()
