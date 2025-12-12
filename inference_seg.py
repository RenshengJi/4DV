#!/usr/bin/env python3
"""
Segmentation Inference Script
输出单行可视化结果：
- GT RGB | GT segmentation | Pred segmentation
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
    parser = argparse.ArgumentParser(description="Segmentation Inference with Visualization")

    # 基础参数
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument("--output_dir", type=str, default="./inference_seg_outputs", help="Output directory")
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
    parser.add_argument("--num_classes", type=int, default=4, help="Number of segmentation classes (Waymo: 4)")

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
    colormap[0] = [128, 128, 128]     # background/unlabeled - 灰色
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


def create_visualization_grid(gt_rgb, gt_seg, pred_seg):
    """创建单行可视化网格：GT RGB | GT segmentation | Pred segmentation

    Args:
        gt_rgb: [S, 3, H, W] - Ground truth RGB images
        gt_seg: [S, 3, H, W] - Ground truth segmentation visualization
        pred_seg: [S, 3, H, W] - Predicted segmentation visualization
    """
    S = gt_rgb.shape[0]
    _, H, W = gt_rgb.shape[1:]

    grid_frames = []

    for s in range(S):
        # Convert to numpy [H, W, 3]
        gt_rgb_np = gt_rgb[s].detach().permute(1, 2, 0).cpu().numpy()
        gt_seg_np = gt_seg[s].detach().permute(1, 2, 0).cpu().numpy()
        pred_seg_np = pred_seg[s].detach().permute(1, 2, 0).cpu().numpy()

        # Create single row: GT RGB | GT Segmentation | Pred Segmentation
        row = np.concatenate([gt_rgb_np, gt_seg_np, pred_seg_np], axis=1)

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

        # Get GT segmentation (from vggt_batch, not views)
        # Shape is [S, H, W] without batch dimension
        gt_seg_labels = vggt_batch.get('segment_label', None)
        gt_seg_mask = vggt_batch.get('segment_mask', None)

        if gt_seg_labels is None:
            print("Warning: No GT segment_label found in batch")
            gt_seg_labels = torch.zeros(S, H, W, dtype=torch.long, device=device)

        if gt_seg_mask is None:
            print("Warning: No GT segment_mask found in batch")
            gt_seg_mask = None

        # Get predicted segmentation (soft logits need softmax)
        # pred_seg_logits: [B, S, H, W, num_classes]
        pred_seg_logits = preds.get('segment_logits', None)
        if pred_seg_logits is not None:
            pred_seg_probs = torch.softmax(pred_seg_logits[0], dim=-1)  # [S, H, W, num_classes]
            pred_seg_labels = torch.argmax(pred_seg_probs, dim=-1)  # [S, H, W]
        else:
            print("Warning: No predicted segment_logits found in output")
            pred_seg_labels = torch.zeros(S, H, W, dtype=torch.long, device=device)

        # Visualize segmentations
        print("Creating visualizations...")
        gt_seg_vis = visualize_segmentation(gt_seg_labels, gt_seg_mask, num_classes=args.num_classes)
        pred_seg_vis = visualize_segmentation(pred_seg_labels, num_classes=args.num_classes)

        return {
            'gt_rgb': gt_rgb,
            'gt_seg': gt_seg_vis,
            'pred_seg': pred_seg_vis,
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
                result['gt_seg'],
                result['pred_seg']
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

    # import debugpy
    # debugpy.listen(5697)
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()

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
                    result['gt_seg'],
                    result['pred_seg']
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
