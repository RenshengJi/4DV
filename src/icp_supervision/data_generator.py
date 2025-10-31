"""
ICP Data Generator - 离线构建ICP GT样本对

核心流程:
1. 从Stage2训练数据中提取dynamic objects的粗糙Gaussians
2. 将Gaussians转换为点云
3. 使用ICP对多帧点云进行配准，得到refined点云
4. 将refined点云转换回Gaussians作为GT
5. 保存(input, GT)样本对到.npz文件
6. (可选) 保存点云文件用于可视化验证
"""

import os
import sys
import numpy as np
import torch
import open3d as o3d
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import argparse
import pickle
from pathlib import Path
import copy

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入现有ICP实现
from point_cloud_registration_save import AdvancedPointCloudRegistration

# 导入utils
from icp_supervision.utils import (
    gaussians_to_pointcloud,
    pointcloud_to_gaussians,
    save_pointcloud_visualization,
    validate_gaussian_params,
    torch_to_numpy,
    compute_chamfer_distance
)


class ICPDataGenerator:
    """
    ICP数据生成器 - 从Stage2训练数据构建ICP GT样本对
    """

    def __init__(
        self,
        output_dir: str = "./icp_supervision_data",
        voxel_size: float = 0.01,
        max_icp_iterations: int = 50,
        save_pointcloud_files: bool = False,
        use_color_features: bool = False,
        min_frames_per_object: int = 2,
        max_frames_per_object: int = 10,
        ransac_max_iteration: int = 5000,
        ransac_confidence: int = 5,
        device: str = 'cuda:0'
    ):
        """
        Args:
            output_dir: 输出目录
            voxel_size: ICP体素大小
            max_icp_iterations: ICP最大迭代次数
            save_pointcloud_files: 是否保存点云文件用于可视化
            use_color_features: 是否使用颜色特征进行ICP
            min_frames_per_object: 每个物体最少帧数（少于此数跳过）
            max_frames_per_object: 每个物体最多使用的帧数
            ransac_max_iteration: RANSAC最大迭代次数
            ransac_confidence: RANSAC置信度参数
            device: PyTorch设备
        """
        self.output_dir = output_dir
        self.voxel_size = voxel_size
        self.max_icp_iterations = max_icp_iterations
        self.save_pointcloud_files = save_pointcloud_files
        self.min_frames_per_object = min_frames_per_object
        self.max_frames_per_object = max_frames_per_object
        self.device = device

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        if save_pointcloud_files:
            self.pcd_dir = os.path.join(output_dir, 'pointclouds')
            os.makedirs(self.pcd_dir, exist_ok=True)

        # 初始化ICP配准器
        self.icp_registrator = AdvancedPointCloudRegistration(
            voxel_size=voxel_size,
            max_iterations=max_icp_iterations,
            use_coarse_registration=True,
            coarse_registration_threshold=0.1,
            use_color_features=use_color_features,
            ransac_max_iteration=ransac_max_iteration,
            ransac_confidence=ransac_confidence
        )

        # 统计信息
        self.stats = {
            'total_objects_processed': 0,
            'successful_registrations': 0,
            'failed_registrations': 0,
            'total_samples_generated': 0,
            'avg_chamfer_distance': [],
        }

    def extract_dynamic_objects_from_stage2_batch(
        self,
        dynamic_objects: List[Dict],
        pred_scale: float,
        frame_indices: Optional[List[int]] = None
    ) -> Dict[int, List[Dict]]:
        """
        从Stage2训练batch中提取dynamic objects

        Args:
            dynamic_objects: Stage2的dynamic_objects列表
            pred_scale: 预测的scale因子，来自preds['pred_scale']
            frame_indices: 需要提取的帧索引列表 (None表示全部)

        Returns:
            object_dict: {object_id: [frame_data_list]}
                每个frame_data包含:
                - gaussians: [N, 14]
                - frame_idx: int
                - pred_scale: float
        """
        object_dict = {}

        for obj_idx, obj_data in enumerate(dynamic_objects):
            # 使用aggregated_gaussians而不是canonical_gaussians
            if 'canonical_gaussians' not in obj_data:
                continue

            # 获取物体ID (如果没有则使用索引)
            object_id = obj_data.get('object_id', obj_idx)

            # 获取aggregated_gaussians作为input (来自光流聚合的结果)
            aggregated_gaussians = obj_data.get('canonical_gaussians')  # 注意：这里实际是aggregated的
            if aggregated_gaussians is None:
                print(f"Object {object_id}: no aggregated_gaussians, skipping")
                continue

            if isinstance(aggregated_gaussians, torch.Tensor):
                aggregated_gaussians = torch_to_numpy(aggregated_gaussians)

            # 获取frame_gaussians字典（每一帧的真实高斯参数）
            frame_gaussians_dict = obj_data.get('frame_gaussians', {})

            # 如果没有frame_gaussians，尝试从frame_pixel_indices获取可用帧
            if not frame_gaussians_dict:
                frame_pixel_indices = obj_data.get('frame_pixel_indices', {})
                available_frames = sorted(frame_pixel_indices.keys())
            else:
                available_frames = sorted(frame_gaussians_dict.keys())

            # 过滤帧
            if frame_indices is not None:
                available_frames = [f for f in available_frames if f in frame_indices]

            if len(available_frames) < self.min_frames_per_object:
                print(f"Object {object_id}: only {len(available_frames)} frames, skipping")
                continue

            # 限制最大帧数
            if len(available_frames) > self.max_frames_per_object:
                # 均匀采样
                step = len(available_frames) / self.max_frames_per_object
                selected_frames = [available_frames[int(i * step)] for i in range(self.max_frames_per_object)]
            else:
                selected_frames = available_frames

            # 为每一帧创建数据
            object_dict[object_id] = []
            for frame_idx in selected_frames:
                # 优先使用frame_gaussians中的真实高斯参数
                if frame_gaussians_dict and frame_idx in frame_gaussians_dict:
                    frame_gaussian = frame_gaussians_dict[frame_idx]
                    if isinstance(frame_gaussian, torch.Tensor):
                        frame_gaussian = torch_to_numpy(frame_gaussian)
                else:
                    # 回退到aggregated_gaussians
                    frame_gaussian = aggregated_gaussians.copy()

                frame_data = {
                    'gaussians': frame_gaussian,
                    'frame_idx': frame_idx,
                    'pred_scale': pred_scale
                }
                object_dict[object_id].append(frame_data)

        return object_dict

    def run_icp_on_object_frames(
        self,
        object_id: int,
        frame_data_list: List[Dict],
        pred_scale: float = 1.0
    ) -> Optional[np.ndarray]:
        """
        对一个物体的多帧Gaussians运行ICP配准，生成refined的target

        Args:
            object_id: 物体ID
            frame_data_list: 该物体的多帧数据
            pred_scale: 预测的scale因子，用于将归一化坐标转换到metric尺度

        Returns:
            all_aligned_points: ICP配准后的点云列表 (在metric尺度下)，如果失败则返回None
        """
        print(f"\n{'='*60}")
        print(f"Processing Object {object_id} with {len(frame_data_list)} frames")
        print(f"{'='*60}")
        print(f"  Using pred_scale: {pred_scale}")

        # 将每一帧的Gaussians转换为点云（转换到metric尺度）
        point_clouds = []
        for i, frame_data in enumerate(frame_data_list):
            gaussians = frame_data['gaussians']
            # 传入pred_scale，将归一化坐标转换到metric尺度
            pcd = gaussians_to_pointcloud(
                gaussians,
                use_colors=self.icp_registrator.use_color_features,
                pred_scale=pred_scale
            )
            point_clouds.append(pcd)
            print(f"  Frame {i}: {len(pcd.points)} points")

        # 运行ICP配准：将所有帧配准到中间帧（与aggregator保持一致）
        try:
            # 选择中间帧作为参考帧
            middle_idx = len(point_clouds) // 2
            print(f"\nRunning ICP registration to align all frames to middle frame (frame {middle_idx})...")

            # 中间帧作为目标
            target_pcd = point_clouds[middle_idx]

            # 将所有帧配准到中间帧，然后合并所有点
            # 首先添加中间帧（参考帧）
            all_aligned_points = []
            all_aligned_points.append(np.asarray(target_pcd.points))
            print(f"  Frame {middle_idx}: reference frame (middle), {len(target_pcd.points)} points")

            # 然后配准其他帧
            for i in range(len(point_clouds)):
                if i == middle_idx:
                    # 中间帧已经添加，跳过
                    continue

                source_pcd = point_clouds[i]

                # 运行ICP配准 - simple_pairwise_registration返回变换矩阵
                try:
                    transformation = self.icp_registrator.simple_pairwise_registration(source_pcd, target_pcd)

                    # 应用变换到源点云 (深拷贝避免修改原始点云)
                    source_pcd_copy = copy.deepcopy(source_pcd)
                    source_pcd_copy.transform(transformation)

                    aligned_points = np.asarray(source_pcd_copy.points)
                    all_aligned_points.append(aligned_points)
                    print(f"  Frame {i}: aligned {len(aligned_points)} points")

                except Exception as e:
                    print(f"  Frame {i}: ICP failed ({e}), skipping this frame")
                    continue

            # 检查是否有足够的成功配准
            if len(all_aligned_points) == 0:
                print(f"  ✗ No successful alignments, registration failed")
                self.stats['failed_registrations'] += 1
                return None

            self.stats['successful_registrations'] += 1
            return all_aligned_points

        except Exception as e:
            print(f"  ✗ Registration error: {e}")
            import traceback
            traceback.print_exc()
            self.stats['failed_registrations'] += 1
            return None

    def save_sample_pair(
        self,
        object_id: int,
        input_gaussians: np.ndarray,
        target_gaussians: np.ndarray,
        pred_scale: float,
        sample_idx: int
    ) -> str:
        """
        保存样本对到.npz文件

        Args:
            object_id: 物体ID
            input_gaussians: 输入Gaussians
            target_gaussians: GT Gaussians
            pred_scale: 用于体素化的scale
            sample_idx: 样本索引

        Returns:
            npz_path: 保存的文件路径
        """
        # 生成文件名
        filename = f"object_{object_id:06d}_sample_{sample_idx:06d}.npz"
        npz_path = os.path.join(self.output_dir, filename)

        # 准备保存数据
        save_dict = {
            'input_gaussians': input_gaussians,
            'target_gaussians': target_gaussians,
            'pred_scale': pred_scale,
            'object_id': object_id,
        }

        # 如果需要，保存点云文件（用于可视化，转换到metric尺度）
        if self.save_pointcloud_files:
            # 输入点云 - 转换到metric尺度用于可视化
            input_pcd = gaussians_to_pointcloud(input_gaussians, pred_scale=pred_scale)
            input_pcd_name = f"object_{object_id:06d}_sample_{sample_idx:06d}_input"
            input_pcd_path = os.path.join(self.pcd_dir, input_pcd_name)
            save_pointcloud_visualization(input_pcd, input_pcd_path, format='ply')
            save_dict['input_pcd_path'] = f"{input_pcd_path}.ply"

            # GT点云 - 转换到metric尺度用于可视化
            target_pcd = gaussians_to_pointcloud(target_gaussians, pred_scale=pred_scale)
            target_pcd_name = f"object_{object_id:06d}_sample_{sample_idx:06d}_target"
            target_pcd_path = os.path.join(self.pcd_dir, target_pcd_name)
            save_pointcloud_visualization(target_pcd, target_pcd_path, format='ply')
            save_dict['target_pcd_path'] = f"{target_pcd_path}.ply"

        # 保存npz文件
        np.savez_compressed(npz_path, **save_dict)
        print(f"  Saved sample pair to: {npz_path}")

        self.stats['total_samples_generated'] += 1
        return npz_path

    def generate_from_stage2_data(
        self,
        dynamic_objects: List[Dict],
        frame_indices: Optional[List[int]] = None
    ) -> List[str]:
        """
        从Stage2数据生成ICP监督样本对

        Args:
            dynamic_objects: Stage2的dynamic_objects列表
            frame_indices: 要处理的帧索引

        Returns:
            generated_files: 生成的.npz文件路径列表
        """
        print(f"\n{'='*80}")
        print(f"Starting ICP Data Generation")
        print(f"{'='*80}")
        print(f"Output directory: {self.output_dir}")
        print(f"Voxel size: {self.voxel_size}")
        print(f"Max ICP iterations: {self.max_icp_iterations}")
        print(f"Save pointcloud files: {self.save_pointcloud_files}")
        print(f"Min frames per object: {self.min_frames_per_object}")
        print(f"Max frames per object: {self.max_frames_per_object}")
        print(f"{'='*80}\n")

        # 提取dynamic objects
        object_dict = self.extract_dynamic_objects_from_stage2_batch(
            dynamic_objects, frame_indices
        )

        if len(object_dict) == 0:
            print("No valid objects found!")
            return []

        print(f"Found {len(object_dict)} objects to process\n")

        # 处理每个物体
        generated_files = []
        for object_id, frame_data_list in tqdm(object_dict.items(), desc="Processing objects"):
            self.stats['total_objects_processed'] += 1

            # 运行ICP
            input_gaussians, target_gaussians = self.run_icp_on_object_frames(
                object_id, frame_data_list
            )

            if input_gaussians is None or target_gaussians is None:
                continue

            # 保存样本对
            pred_scale = frame_data_list[0]['pred_scale']
            npz_path = self.save_sample_pair(
                object_id=object_id,
                input_gaussians=input_gaussians,
                target_gaussians=target_gaussians,
                pred_scale=pred_scale,
                sample_idx=self.stats['total_samples_generated']
            )
            generated_files.append(npz_path)

        # 打印统计信息
        self.print_stats()

        return generated_files

    def print_stats(self):
        """打印统计信息"""
        print(f"\n{'='*80}")
        print(f"ICP Data Generation Statistics")
        print(f"{'='*80}")
        print(f"Total objects processed: {self.stats['total_objects_processed']}")
        print(f"Successful registrations: {self.stats['successful_registrations']}")
        print(f"Failed registrations: {self.stats['failed_registrations']}")
        print(f"Total samples generated: {self.stats['total_samples_generated']}")

        if len(self.stats['avg_chamfer_distance']) > 0:
            avg_cd = np.mean(self.stats['avg_chamfer_distance'])
            print(f"Average Chamfer distance: {avg_cd:.6f}")

        success_rate = 0.0
        if self.stats['total_objects_processed'] > 0:
            success_rate = self.stats['successful_registrations'] / self.stats['total_objects_processed'] * 100
        print(f"Success rate: {success_rate:.1f}%")
        print(f"{'='*80}\n")


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="Generate ICP supervision data from Stage2 training data")

    parser.add_argument('--input', type=str, required=True,
                        help='Input pickle file containing Stage2 dynamic_objects')
    parser.add_argument('--output_dir', type=str, default='./icp_supervision_data',
                        help='Output directory for generated samples')
    parser.add_argument('--voxel_size', type=float, default=0.01,
                        help='Voxel size for ICP')
    parser.add_argument('--max_icp_iterations', type=int, default=50,
                        help='Maximum ICP iterations')
    parser.add_argument('--save_pointclouds', action='store_true',
                        help='Save pointcloud files for visualization')
    parser.add_argument('--use_color_features', action='store_true',
                        help='Use color features in ICP')
    parser.add_argument('--min_frames', type=int, default=2,
                        help='Minimum frames per object')
    parser.add_argument('--max_frames', type=int, default=10,
                        help='Maximum frames per object')

    args = parser.parse_args()

    # 加载输入数据
    print(f"Loading data from: {args.input}")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    if 'dynamic_objects' not in data:
        print("Error: Input file does not contain 'dynamic_objects' key")
        return

    dynamic_objects = data['dynamic_objects']

    # 创建生成器
    generator = ICPDataGenerator(
        output_dir=args.output_dir,
        voxel_size=args.voxel_size,
        max_icp_iterations=args.max_icp_iterations,
        save_pointcloud_files=args.save_pointclouds,
        use_color_features=args.use_color_features,
        min_frames_per_object=args.min_frames,
        max_frames_per_object=args.max_frames
    )

    # 生成数据
    generated_files = generator.generate_from_stage2_data(dynamic_objects)

    print(f"\nGenerated {len(generated_files)} sample pairs")
    print(f"Data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
