#!/usr/bin/env python3
"""
Confidence Inference Script
输出单行可视化结果：
- GT RGB | Depth Confidence | Velocity Confidence
"""

import os
import sys
import numpy as np
import torch
import argparse
import imageio.v2 as iio
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/vggt'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from vggt.models.vggt import VGGT
from dust3r.utils.misc import tf32_off
from src.dust3r.inference import inference
from src.train import cut3r_batch_to_vggt


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Confidence Inference with Visualization")

    # 基础参数
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument("--output_dir", type=str, default="./inference_conf_outputs", help="Output directory")
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

    # VGGT模型配置参数
    parser.add_argument("--sh_degree", type=int, default=0, help="Spherical harmonics degree")
    parser.add_argument("--use_gs_head", action="store_true", default=True, help="Use DPTGSHead for gaussian_head")
    parser.add_argument("--use_gs_head_velocity", action="store_true", default=False, help="Use DPTGSHead for velocity_head")
    parser.add_argument("--use_gt_camera", action="store_true", help="Use GT camera parameters")

    return parser.parse_args()


def load_model(model_path, device, args):
    """加载模型"""
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


def visualize_confidence(conf):
    """可视化confidence为RGB图像（使用viridis colormap）

    Args:
        conf: [S, H, W] tensor - Confidence values

    Returns:
        [S, 3, H, W] tensor - Confidence visualization
    """
    S, H, W = conf.shape
    conf_vis = []

    for s in range(S):
        c = conf[s].detach().cpu().numpy()

        # Normalize to [0, 1] range
        c_min, c_max = c.min(), c.max()
        if c_max > c_min:
            c_norm = (c - c_min) / (c_max - c_min)
        else:
            c_norm = c * 0

        # Apply viridis colormap
        colored = cm.viridis(c_norm)[:, :, :3]  # [H, W, 3]
        conf_vis.append(torch.from_numpy(colored).permute(2, 0, 1))  # [3, H, W]

    return torch.stack(conf_vis, dim=0)  # [S, 3, H, W]


def create_visualization_grid(gt_rgb, depth_conf_vis, velocity_conf_vis):
    """创建单行可视化网格：GT RGB | Depth Confidence | Velocity Confidence

    Args:
        gt_rgb: [S, 3, H, W] - Ground truth RGB images
        depth_conf_vis: [S, 3, H, W] - Depth confidence visualization
        velocity_conf_vis: [S, 3, H, W] - Velocity confidence visualization
    """
    S = gt_rgb.shape[0]
    _, H, W = gt_rgb.shape[1:]

    grid_frames = []

    for s in range(S):
        # Convert to numpy [H, W, 3]
        gt_rgb_np = gt_rgb[s].detach().permute(1, 2, 0).cpu().numpy()
        depth_conf_np = depth_conf_vis[s].detach().permute(1, 2, 0).cpu().numpy()
        velocity_conf_np = velocity_conf_vis[s].detach().permute(1, 2, 0).cpu().numpy()

        # Create single row: GT RGB | Depth Confidence | Velocity Confidence
        row = np.concatenate([gt_rgb_np, depth_conf_np, velocity_conf_np], axis=1)

        # Convert to uint8
        grid = (np.clip(row, 0, 1) * 255).astype(np.uint8)
        grid_frames.append(grid)

    return grid_frames


def run_single_inference(model, dataset, idx, num_views, device, args=None):
    """运行单次推理"""
    print(f"\n{'='*60}")
    print(f"Processing sequence index: {idx}")
    print(f"{'='*60}\n")

    try:
        # Load data
        views = dataset.__getitem__((idx, 2, num_views))

        # 运行推理
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

        # Extract data
        B, S, C, H, W = vggt_batch['images'].shape

        gt_rgb = vggt_batch['images'][0]  # [S, 3, H, W]

        # Get depth confidence - shape: [B, S, H, W]
        depth_conf = preds.get('depth_conf', None)
        if depth_conf is None:
            print("Warning: depth_conf not found in predictions, using zeros")
            depth_conf = torch.zeros(B, S, H, W, device=device)
        depth_conf = depth_conf[0]  # [S, H, W]

        # Get velocity confidence - shape: [B, S, H, W, 1] or [B, S, H, W]
        velocity_conf = preds.get('velocity_conf', None)
        if velocity_conf is None:
            print("Warning: velocity_conf not found in predictions, using zeros")
            velocity_conf = torch.zeros(B, S, H, W, device=device)
        else:
            velocity_conf = velocity_conf[0]  # [S, H, W, 1] or [S, H, W]
            # If has extra dimension, squeeze it
            if velocity_conf.dim() == 4 and velocity_conf.shape[-1] == 1:
                velocity_conf = velocity_conf.squeeze(-1)  # [S, H, W]

        # Visualize confidences
        print("Creating visualizations...")
        print(f"Depth confidence - min: {depth_conf.min().item():.4f}, max: {depth_conf.max().item():.4f}, mean: {depth_conf.mean().item():.4f}")
        print(f"Velocity confidence - min: {velocity_conf.min().item():.4f}, max: {velocity_conf.max().item():.4f}, mean: {velocity_conf.mean().item():.4f}")

        depth_conf_vis = visualize_confidence(depth_conf)
        velocity_conf_vis = visualize_confidence(velocity_conf)

        return {
            'gt_rgb': gt_rgb,
            'depth_conf_vis': depth_conf_vis,
            'velocity_conf_vis': velocity_conf_vis,
            'success': True
        }

    except Exception as e:
        print(f"Error processing idx {idx}: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def run_batch_inference(model, dataset, args, device):
    """运行批量推理"""
    print(f"\n{'='*60}")
    print(f"Batch Inference Mode")
    print(f"Range: {args.start_idx} to {args.end_idx}, step {args.step}")
    print(f"{'='*60}\n")

    successful = []
    failed = []

    indices = range(args.start_idx, args.end_idx, args.step)

    for idx in tqdm(indices, desc="Batch processing"):
        result = run_single_inference(model, dataset, idx, args.num_views, device, args)

        if result['success']:
            # Save video
            grid_frames = create_visualization_grid(
                result['gt_rgb'],
                result['depth_conf_vis'],
                result['velocity_conf_vis']
            )

            seq_name = os.path.basename(args.dataset_root)
            output_path = os.path.join(args.output_dir, f"{seq_name}_idx{idx}.mp4")

            save_video(grid_frames, output_path, fps=args.fps)
            successful.append(idx)
            print(f"✓ idx {idx}")
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

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Run inference
    with tf32_off():
        if args.batch_mode:
            run_batch_inference(model, dataset, args, device)
        else:
            result = run_single_inference(model, dataset, args.idx, args.num_views, device, args)

            if result['success']:
                grid_frames = create_visualization_grid(
                    result['gt_rgb'],
                    result['depth_conf_vis'],
                    result['velocity_conf_vis']
                )

                seq_name = os.path.basename(args.dataset_root)
                output_path = os.path.join(args.output_dir, f"{seq_name}_idx{args.idx}.mp4")

                save_video(grid_frames, output_path, fps=args.fps)

                print(f"\n{'='*60}")
                print(f"Success!")
                print(f"Output: {output_path}")
                print(f"{'='*60}\n")
            else:
                print(f"Failed: {result['error']}")


if __name__ == "__main__":
    main()
