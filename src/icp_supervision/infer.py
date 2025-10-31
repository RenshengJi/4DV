#!/usr/bin/env python3
"""
ICP Refiner 推理脚本 - 可视化单个样本的细化结果
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from vggt.vggt.heads.sparse_conv_refine_head import GaussianRefineHeadSparseConv
from icp_supervision.utils import load_sample_pair, gaussians_to_pointcloud


def load_model(checkpoint_path: str, device: str = 'cuda:0'):
    """
    加载训练好的模型

    Args:
        checkpoint_path: checkpoint 文件路径
        device: 设备

    Returns:
        model: 加载好的模型
        config: 训练配置
    """
    print(f"Loading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})

    # 创建模型
    model = GaussianRefineHeadSparseConv(
        input_gaussian_dim=config.get('input_gaussian_dim', 14),
        output_gaussian_dim=config.get('output_gaussian_dim', 14),
        feature_dim=config.get('gaussian_feature_dim', 384),
        num_conv_layers=config.get('gaussian_num_conv_layers', 10),
        voxel_size=config.get('gaussian_voxel_size', 0.05),
    )

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  ✓ Model loaded (epoch {checkpoint['epoch']})")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")

    return model, config


def infer_sample(
    model: torch.nn.Module,
    sample_path: str,
    device: str = 'cuda:0',
    use_metric_scale: bool = True
):
    """
    对单个样本进行推理

    Args:
        model: 训练好的模型
        sample_path: .npz 样本路径
        device: 设备
        use_metric_scale: 是否转换到 metric 尺度（用于可视化）

    Returns:
        input_gaussians: [N, 14] 输入 Gaussians (numpy)
        pred_gaussians: [N, 14] 预测 Gaussians (numpy)
        target_gaussians: [N, 14] 目标 Gaussians (numpy)
        pred_scale: float
    """
    print(f"\nInferring sample: {sample_path}")

    # 加载样本
    data = load_sample_pair(sample_path)
    if data is None:
        raise ValueError(f"Failed to load sample from {sample_path}")

    input_gaussians_np = data['input_gaussians']
    target_gaussians_np = data['target_gaussians']
    pred_scale = data['pred_scale']

    print(f"  Loaded {input_gaussians_np.shape[0]} Gaussians")
    print(f"  pred_scale: {pred_scale}")

    # 转换到 torch
    input_gaussians = torch.from_numpy(input_gaussians_np).float().to(device)
    pred_scale_tensor = torch.tensor([pred_scale]).float().to(device)

    # 推理
    with torch.no_grad():
        pred_gaussians_delta = model(input_gaussians, pred_scale_tensor)
        pred_gaussians = model.apply_deltas(input_gaussians, pred_gaussians_delta)

    # 转换回 numpy
    pred_gaussians_np = pred_gaussians.cpu().numpy()

    print(f"  ✓ Inference complete")

    return input_gaussians_np, pred_gaussians_np, target_gaussians_np, pred_scale


def save_gaussian_as_ply(
    gaussians: np.ndarray,
    output_path: str,
    pred_scale: float = None,
    use_metric_scale: bool = True
):
    """
    将 Gaussians 保存为带颜色的点云 PLY 文件

    Args:
        gaussians: [N, 14] Gaussian 参数
        output_path: 输出 PLY 文件路径
        pred_scale: 用于尺度转换（如果提供）
        use_metric_scale: 是否转换到 metric 尺度
    """
    # 转换为点云
    pcd = gaussians_to_pointcloud(
        gaussians,
        use_colors=True,
        pred_scale=pred_scale if use_metric_scale else None
    )

    # 保存为 PLY
    import open3d as o3d
    o3d.io.write_point_cloud(output_path, pcd)

    print(f"  Saved: {output_path}")


def visualize_comparison(
    input_path: str,
    pred_path: str,
    target_path: str
):
    """
    可视化对比（可选）

    Args:
        input_path: 输入点云路径
        pred_path: 预测点云路径
        target_path: 目标点云路径
    """
    import open3d as o3d

    # 加载点云
    input_pcd = o3d.io.read_point_cloud(input_path)
    pred_pcd = o3d.io.read_point_cloud(pred_path)
    target_pcd = o3d.io.read_point_cloud(target_path)

    # 设置不同的位置以便对比
    # Input: 左侧
    input_pcd.translate([-2, 0, 0])

    # Prediction: 中间（原位置）
    # pred_pcd 保持原位

    # Target: 右侧
    target_pcd.translate([2, 0, 0])

    # 可视化
    print("\nVisualizing comparison...")
    print("  Left: Input (粗糙)")
    print("  Center: Prediction (细化)")
    print("  Right: Target (GT)")

    o3d.visualization.draw_geometries(
        [input_pcd, pred_pcd, target_pcd],
        window_name="ICP Refiner Comparison",
        width=1920,
        height=1080
    )


def compute_metrics(pred_gaussians: np.ndarray, target_gaussians: np.ndarray):
    """
    计算评估指标

    Args:
        pred_gaussians: [N, 14] 预测 Gaussians
        target_gaussians: [N, 14] 目标 Gaussians

    Returns:
        metrics: 指标字典
    """
    metrics = {}

    # Position error (最重要)
    pos_pred = pred_gaussians[:, :3]
    pos_target = target_gaussians[:, :3]
    pos_error = np.linalg.norm(pos_pred - pos_target, axis=1)
    metrics['position_mae'] = np.mean(pos_error)
    metrics['position_rmse'] = np.sqrt(np.mean(pos_error ** 2))
    metrics['position_max'] = np.max(pos_error)

    # Scale error
    scale_pred = pred_gaussians[:, 3:6]
    scale_target = target_gaussians[:, 3:6]
    scale_error = np.linalg.norm(scale_pred - scale_target, axis=1)
    metrics['scale_mae'] = np.mean(scale_error)

    # Color error
    color_pred = pred_gaussians[:, 6:9]
    color_target = target_gaussians[:, 6:9]
    color_error = np.linalg.norm(color_pred - color_target, axis=1)
    metrics['color_mae'] = np.mean(color_error)

    # Rotation error (quaternion)
    rot_pred = pred_gaussians[:, 9:13]
    rot_target = target_gaussians[:, 9:13]
    rot_error = np.linalg.norm(rot_pred - rot_target, axis=1)
    metrics['rotation_mae'] = np.mean(rot_error)

    # Opacity error
    opacity_pred = pred_gaussians[:, 13]
    opacity_target = target_gaussians[:, 13]
    opacity_error = np.abs(opacity_pred - opacity_target)
    metrics['opacity_mae'] = np.mean(opacity_error)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="ICP Refiner 推理 - 可视化单个样本")

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='训练好的 checkpoint 路径')
    parser.add_argument('--sample', type=str, required=True,
                        help='输入 .npz 样本路径')
    parser.add_argument('--output_dir', type=str, default='./inference_output',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备 (cuda:0 or cpu)')
    parser.add_argument('--visualize', action='store_true',
                        help='使用 Open3D 可视化对比')
    parser.add_argument('--no_metric_scale', action='store_true',
                        help='不转换到 metric 尺度（保持归一化尺度）')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("ICP Refiner Inference")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Sample: {args.sample}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print("="*80)

    # 加载模型
    model, config = load_model(args.checkpoint, args.device)

    # 推理
    use_metric_scale = not args.no_metric_scale
    input_gaussians, pred_gaussians, target_gaussians, pred_scale = infer_sample(
        model, args.sample, args.device, use_metric_scale
    )

    # 保存为 PLY 文件
    print("\nSaving point clouds...")

    sample_name = Path(args.sample).stem

    input_ply = output_dir / f"{sample_name}_input.ply"
    pred_ply = output_dir / f"{sample_name}_pred.ply"
    target_ply = output_dir / f"{sample_name}_target.ply"

    save_gaussian_as_ply(
        input_gaussians, str(input_ply),
        pred_scale if use_metric_scale else None,
        use_metric_scale
    )
    save_gaussian_as_ply(
        pred_gaussians, str(pred_ply),
        pred_scale if use_metric_scale else None,
        use_metric_scale
    )
    save_gaussian_as_ply(
        target_gaussians, str(target_ply),
        pred_scale if use_metric_scale else None,
        use_metric_scale
    )

    # 计算指标
    print("\nComputing metrics...")
    metrics = compute_metrics(pred_gaussians, target_gaussians)

    print("\nMetrics:")
    print(f"  Position MAE:  {metrics['position_mae']:.6f}")
    print(f"  Position RMSE: {metrics['position_rmse']:.6f}")
    print(f"  Position Max:  {metrics['position_max']:.6f}")
    print(f"  Scale MAE:     {metrics['scale_mae']:.6f}")
    print(f"  Color MAE:     {metrics['color_mae']:.6f}")
    print(f"  Rotation MAE:  {metrics['rotation_mae']:.6f}")
    print(f"  Opacity MAE:   {metrics['opacity_mae']:.6f}")

    # 保存指标到文本文件
    metrics_file = output_dir / f"{sample_name}_metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write(f"Sample: {args.sample}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"pred_scale: {pred_scale}\n")
        f.write(f"use_metric_scale: {use_metric_scale}\n")
        f.write(f"\nMetrics:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.6f}\n")

    print(f"\n  Saved metrics to: {metrics_file}")

    # 可视化
    if args.visualize:
        visualize_comparison(str(input_ply), str(pred_ply), str(target_ply))

    print("\n" + "="*80)
    print("✓ Inference complete!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  Input:  {input_ply}")
    print(f"  Pred:   {pred_ply}")
    print(f"  Target: {target_ply}")
    print(f"  Metrics: {metrics_file}")
    print("\n可以使用 CloudCompare 或 MeshLab 打开 .ply 文件查看")


if __name__ == "__main__":
    main()
