"""
Visualization tools for verifying ICP GT correctness
"""

import os
import sys
import argparse
import numpy as np
import open3d as o3d
from typing import List, Tuple
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from icp_supervision.utils import (
    load_sample_pair,
    gaussians_to_pointcloud,
    compute_chamfer_distance
)


def visualize_sample_pair(
    npz_path: str,
    show_input: bool = True,
    show_target: bool = True,
    show_comparison: bool = True
):
    """
    可视化单个ICP样本对

    Args:
        npz_path: .npz文件路径
        show_input: 是否显示输入点云
        show_target: 是否显示目标点云
        show_comparison: 是否显示对比视图
    """
    print(f"Loading sample from: {npz_path}")

    # 加载数据
    data = load_sample_pair(npz_path)

    if data is None:
        print("Failed to load sample")
        return

    print(f"  Object ID: {data['object_id']}")
    print(f"  Pred scale: {data['pred_scale']}")
    print(f"  Num points: {data['input_gaussians'].shape[0]}")

    # 转换为点云
    input_pcd = gaussians_to_pointcloud(data['input_gaussians'], use_colors=True)
    target_pcd = gaussians_to_pointcloud(data['target_gaussians'], use_colors=True)

    # 计算Chamfer距离
    chamfer_dist = compute_chamfer_distance(input_pcd, target_pcd)
    print(f"  Chamfer distance: {chamfer_dist:.6f}")

    # 可视化
    if show_input:
        print("\nShowing input point cloud (original)...")
        o3d.visualization.draw_geometries(
            [input_pcd],
            window_name=f"Input - Object {data['object_id']}",
            width=800, height=600
        )

    if show_target:
        print("\nShowing target point cloud (ICP refined)...")
        o3d.visualization.draw_geometries(
            [target_pcd],
            window_name=f"Target (ICP GT) - Object {data['object_id']}",
            width=800, height=600
        )

    if show_comparison:
        print("\nShowing comparison (red=input, green=target)...")

        # 复制点云并着色
        input_pcd_colored = o3d.geometry.PointCloud(input_pcd)
        target_pcd_colored = o3d.geometry.PointCloud(target_pcd)

        input_pcd_colored.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
        target_pcd_colored.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色

        o3d.visualization.draw_geometries(
            [input_pcd_colored, target_pcd_colored],
            window_name=f"Comparison - Object {data['object_id']} (Red=Input, Green=Target)",
            width=800, height=600
        )


def batch_visualize_samples(
    data_dir: str,
    num_samples: int = 5,
    mode: str = 'comparison'
):
    """
    批量可视化多个样本

    Args:
        data_dir: 数据目录
        num_samples: 可视化样本数量
        mode: 'input', 'target', or 'comparison'
    """
    # 查找所有.npz文件
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

    if len(npz_files) == 0:
        print(f"No .npz files found in {data_dir}")
        return

    print(f"Found {len(npz_files)} samples")
    print(f"Visualizing first {min(num_samples, len(npz_files))} samples\n")

    for i, npz_path in enumerate(npz_files[:num_samples]):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}/{min(num_samples, len(npz_files))}")
        print(f"{'='*60}")

        if mode == 'input':
            visualize_sample_pair(npz_path, show_input=True, show_target=False, show_comparison=False)
        elif mode == 'target':
            visualize_sample_pair(npz_path, show_input=False, show_target=True, show_comparison=False)
        elif mode == 'comparison':
            visualize_sample_pair(npz_path, show_input=False, show_target=False, show_comparison=True)
        else:
            visualize_sample_pair(npz_path, show_input=True, show_target=True, show_comparison=True)


def compute_dataset_statistics(data_dir: str):
    """
    计算数据集统计信息

    Args:
        data_dir: 数据目录
    """
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

    if len(npz_files) == 0:
        print(f"No .npz files found in {data_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Dataset Statistics")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    print(f"Total samples: {len(npz_files)}\n")

    num_points_list = []
    chamfer_distances = []
    pred_scales = []
    object_ids = set()

    for npz_path in npz_files:
        data = load_sample_pair(npz_path)
        if data is None:
            continue

        num_points = data['input_gaussians'].shape[0]
        num_points_list.append(num_points)
        pred_scales.append(data['pred_scale'])
        object_ids.add(data['object_id'])

        # 计算Chamfer距离
        input_pcd = gaussians_to_pointcloud(data['input_gaussians'])
        target_pcd = gaussians_to_pointcloud(data['target_gaussians'])
        chamfer_dist = compute_chamfer_distance(input_pcd, target_pcd)
        chamfer_distances.append(chamfer_dist)

    # 打印统计
    print(f"Number of unique objects: {len(object_ids)}")
    print(f"\nPoints per sample:")
    print(f"  Mean: {np.mean(num_points_list):.1f}")
    print(f"  Std: {np.std(num_points_list):.1f}")
    print(f"  Min: {np.min(num_points_list)}")
    print(f"  Max: {np.max(num_points_list)}")

    print(f"\nPred scale:")
    print(f"  Mean: {np.mean(pred_scales):.6f}")
    print(f"  Std: {np.std(pred_scales):.6f}")
    print(f"  Min: {np.min(pred_scales):.6f}")
    print(f"  Max: {np.max(pred_scales):.6f}")

    print(f"\nChamfer distance (input vs ICP GT):")
    print(f"  Mean: {np.mean(chamfer_distances):.6f}")
    print(f"  Std: {np.std(chamfer_distances):.6f}")
    print(f"  Min: {np.min(chamfer_distances):.6f}")
    print(f"  Max: {np.max(chamfer_distances):.6f}")

    print(f"\n{'='*60}\n")


def export_pointclouds_for_meshlab(
    npz_path: str,
    output_dir: str = "./exported_pointclouds"
):
    """
    导出点云为.ply文件，用于在MeshLab等工具中查看

    Args:
        npz_path: .npz文件路径
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    data = load_sample_pair(npz_path)
    if data is None:
        print("Failed to load sample")
        return

    object_id = data['object_id']
    basename = os.path.splitext(os.path.basename(npz_path))[0]

    # 转换为点云
    input_pcd = gaussians_to_pointcloud(data['input_gaussians'], use_colors=True)
    target_pcd = gaussians_to_pointcloud(data['target_gaussians'], use_colors=True)

    # 保存
    input_path = os.path.join(output_dir, f"{basename}_input.ply")
    target_path = os.path.join(output_dir, f"{basename}_target.ply")

    o3d.io.write_point_cloud(input_path, input_pcd)
    o3d.io.write_point_cloud(target_path, target_pcd)

    print(f"Exported point clouds to:")
    print(f"  Input: {input_path}")
    print(f"  Target: {target_path}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Visualize ICP supervision data")

    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory containing .npz files')
    parser.add_argument('--mode', type=str, default='comparison',
                        choices=['input', 'target', 'comparison', 'all', 'stats'],
                        help='Visualization mode')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--sample_idx', type=int, default=None,
                        help='Visualize specific sample index')
    parser.add_argument('--export', action='store_true',
                        help='Export point clouds as .ply files')
    parser.add_argument('--export_dir', type=str, default='./exported_pointclouds',
                        help='Export directory')

    args = parser.parse_args()

    if args.mode == 'stats':
        # 计算统计信息
        compute_dataset_statistics(args.data_dir)
    elif args.sample_idx is not None:
        # 可视化指定样本
        npz_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
        if args.sample_idx < len(npz_files):
            npz_path = npz_files[args.sample_idx]
            print(f"Visualizing sample {args.sample_idx}: {npz_path}")

            if args.export:
                export_pointclouds_for_meshlab(npz_path, args.export_dir)
            else:
                visualize_sample_pair(
                    npz_path,
                    show_input=(args.mode in ['input', 'all']),
                    show_target=(args.mode in ['target', 'all']),
                    show_comparison=(args.mode in ['comparison', 'all'])
                )
        else:
            print(f"Sample index {args.sample_idx} out of range")
    else:
        # 批量可视化
        batch_visualize_samples(args.data_dir, args.num_samples, args.mode)


if __name__ == "__main__":
    main()
