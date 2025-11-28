#!/usr/bin/env python3
"""
Velocity Inference Script
输出单行可视化结果：
- GT RGB | GT Velocity | GT RGB + Pred Velocity 融合 (加权叠加)
"""

import os
import sys
import numpy as np
import torch
import argparse
import imageio.v2 as iio
from tqdm import tqdm

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/vggt'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from vggt.models.vggt import VGGT
from dust3r.utils.misc import tf32_off
from src.dust3r.inference import inference
from src.train import cut3r_batch_to_vggt


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Velocity Inference with Visualization")

    # 基础参数
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--seq_dir", type=str, required=True, help="Path to sequence directory")
    parser.add_argument("--output_dir", type=str, default="./inference_velocity_outputs", help="Output directory")
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

    # 可视化参数
    parser.add_argument("--velocity_alpha", type=float, default=0.5,
                       help="Weight for pred velocity in fusion (0-1), default 0.5 means 50% each")
    parser.add_argument("--velocity_scale", type=float, default=0.1,
                       help="Scale factor for velocity visualization")

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


def load_dataset(seq_dir, num_views):
    """加载数据集"""
    from src.dust3r.datasets.waymo import Waymo_Multi

    seq_name = os.path.basename(seq_dir)
    root_dir = os.path.dirname(seq_dir)

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


def visualize_velocity(velocity, scale=0.1):
    """可视化velocity为RGB图像"""
    from dust3r.utils.image import scene_flow_to_rgb

    S, H, W, _ = velocity.shape
    velocity_rgb = scene_flow_to_rgb(velocity.detach(), scale).permute(0, 3, 1, 2)
    return velocity_rgb


def create_visualization_grid(gt_rgb, gt_velocity, pred_velocity, velocity_alpha=0.5):
    """创建单行可视化网格：GT RGB | GT Velocity | GT RGB + Pred Velocity 融合

    Args:
        gt_rgb: [S, 3, H, W] - Ground truth RGB images
        gt_velocity: [S, 3, H, W] - Ground truth velocity visualization
        pred_velocity: [S, 3, H, W] - Predicted velocity visualization
        velocity_alpha: Pred velocity在融合中的权重 (0-1)，默认0.5表示各占50%
    """
    S = gt_rgb.shape[0]
    _, H, W = gt_rgb.shape[1:]

    grid_frames = []

    for s in range(S):
        # Convert to numpy [H, W, 3]
        gt_rgb_np = gt_rgb[s].detach().permute(1, 2, 0).cpu().numpy()
        gt_velocity_np = gt_velocity[s].detach().permute(1, 2, 0).cpu().numpy()
        pred_velocity_np = pred_velocity[s].detach().permute(1, 2, 0).cpu().numpy()

        # 创建GT RGB和Pred Velocity的加权融合图像
        fused_velocity_np = velocity_alpha * pred_velocity_np + (1 - velocity_alpha) * gt_rgb_np
        fused_velocity_np = np.clip(fused_velocity_np, 0, 1)

        # Create single row: GT RGB | GT Velocity | GT RGB + Pred Velocity 融合
        row = np.concatenate([gt_rgb_np, gt_velocity_np, fused_velocity_np], axis=1)

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

        # Get GT velocity
        gt_velocity = vggt_batch.get('flowmap', None)
        gt_velocity = gt_velocity[0, :, :, :, :3]
        # Apply coordinate transformation
        gt_velocity = gt_velocity[:, :, :, [2, 0, 1]]
        gt_velocity[:, :, :, 2] = -gt_velocity[:, :, :, 2]

        # Get predicted velocity
        pred_velocity = preds.get('velocity', torch.zeros(1, S, H, W, 3, device=device))[0]

        # Apply coordinate transformation
        pred_velocity = pred_velocity[:, :, :, [2, 0, 1]]
        pred_velocity[:, :, :, 2] = -pred_velocity[:, :, :, 2]

        # Visualize velocities
        print("Creating visualizations...")
        gt_velocity_vis = visualize_velocity(gt_velocity, scale=args.velocity_scale)
        pred_velocity_vis = visualize_velocity(pred_velocity, scale=args.velocity_scale)

        return {
            'gt_rgb': gt_rgb,
            'gt_velocity': gt_velocity_vis,
            'pred_velocity': pred_velocity_vis,
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
                result['gt_velocity'],
                result['pred_velocity'],
                velocity_alpha=args.velocity_alpha
            )

            seq_name = os.path.basename(args.seq_dir)
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
    dataset = load_dataset(args.seq_dir, args.num_views)

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
                    result['gt_velocity'],
                    result['pred_velocity'],
                    velocity_alpha=args.velocity_alpha
                )

                seq_name = os.path.basename(args.seq_dir)
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
