"""
从Stage2训练数据生成dynamic_objects并保存

这个脚本会：
1. 加载Stage2训练数据（checkpoint或推理结果）
2. 运行OnlineDynamicProcessor生成dynamic_objects
3. 保存dynamic_objects到pickle文件
4. 为ICP supervision数据生成做准备
"""

import os
import sys
import torch
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dust3r.utils.misc import tf32_off
from online_dynamic_processor import OnlineDynamicProcessor
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.training.loss import depth_to_world_points


def load_stage1_predictions(checkpoint_path: str, device='cuda:0'):
    """
    加载Stage1模型的checkpoint并准备推理

    Args:
        checkpoint_path: Stage1 checkpoint路径
        device: 设备

    Returns:
        model: 加载好的模型
    """
    print(f"Loading Stage1 checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 这里需要根据实际的模型结构来加载
    # 简化版本：假设我们已经有了预测结果
    print(f"Checkpoint loaded (epoch: {checkpoint.get('epoch', 'unknown')})")

    return checkpoint


def run_inference_and_extract_objects(
    dataset,
    processor: OnlineDynamicProcessor,
    num_samples: int = 100,
    device: str = 'cuda:0',
    save_interval: int = 10
):
    """
    运行推理并提取dynamic_objects

    Args:
        dataset: 数据集
        processor: OnlineDynamicProcessor实例
        num_samples: 要处理的样本数量
        device: 设备
        save_interval: 每隔多少个样本保存一次中间结果

    Returns:
        all_dynamic_objects: 所有样本的dynamic_objects列表
    """
    all_dynamic_objects = []

    print(f"\nProcessing {num_samples} samples...")

    for idx in tqdm(range(min(num_samples, len(dataset)))):
        try:
            # 获取一个batch的数据
            batch = dataset[idx]

            # 这里需要实际运行Stage1模型推理
            # 简化版本：假设我们已经有了predictions
            # 在实际使用中，你需要：
            # 1. 加载Stage1模型
            # 2. 运行前向传播得到preds
            # 3. 传入processor

            # 模拟的predictions结构
            # 实际使用时需要替换为真实的模型输出
            preds = {
                'depth': batch.get('depth'),  # 需要从实际推理得到
                'pose_enc': batch.get('pose_enc'),  # 需要从实际推理得到
                'pts3d': batch.get('pts3d'),  # 需要从实际推理得到
                'velocity': batch.get('velocity'),  # 需要从实际推理得到
                # ... 其他需要的字段
            }

            vggt_batch = {
                'images': batch.get('images'),
                # ... 其他需要的字段
            }

            conf = batch.get('conf', None)

            # 运行processor
            result = processor.process_dynamic_objects(
                preds=preds,
                vggt_batch=vggt_batch,
                conf=conf
            )

            dynamic_objects = result.get('dynamic_objects', [])

            if len(dynamic_objects) > 0:
                all_dynamic_objects.extend(dynamic_objects)
                print(f"Sample {idx}: Found {len(dynamic_objects)} dynamic objects")

            # 定期保存中间结果
            if (idx + 1) % save_interval == 0:
                print(f"\nSaving intermediate results at sample {idx + 1}...")
                save_dynamic_objects(all_dynamic_objects, f'dynamic_objects_temp_{idx+1}.pkl')

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return all_dynamic_objects


def save_dynamic_objects(dynamic_objects, output_path):
    """
    保存dynamic_objects到pickle文件

    Args:
        dynamic_objects: dynamic_objects列表
        output_path: 输出文件路径
    """
    data = {
        'dynamic_objects': dynamic_objects,
        'num_objects': len(dynamic_objects),
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    file_size = os.path.getsize(output_path)
    print(f"Saved {len(dynamic_objects)} objects to {output_path}")
    print(f"File size: {file_size / 1024 / 1024:.2f} MB")


def extract_from_existing_results(
    results_dir: str,
    num_samples: int = 100,
    output_path: str = 'dynamic_objects.pkl'
):
    """
    从已有的推理结果中提取dynamic_objects

    这个函数假设你已经运行过推理，并保存了包含dynamic_objects的结果

    Args:
        results_dir: 推理结果目录
        num_samples: 要提取的样本数量
        output_path: 输出文件路径
    """
    print(f"Extracting dynamic_objects from existing results: {results_dir}")

    all_dynamic_objects = []

    # 查找所有的结果文件
    result_files = sorted(Path(results_dir).glob('*.pkl'))

    print(f"Found {len(result_files)} result files")

    for i, result_file in enumerate(tqdm(result_files[:num_samples])):
        try:
            with open(result_file, 'rb') as f:
                data = pickle.load(f)

            # 提取dynamic_objects
            if 'dynamic_objects' in data:
                dynamic_objects = data['dynamic_objects']
                if len(dynamic_objects) > 0:
                    all_dynamic_objects.extend(dynamic_objects)
                    print(f"File {i}: Found {len(dynamic_objects)} objects")
            elif 'clustering_results' in data:
                # 如果是clustering_results格式，需要转换
                print(f"File {i}: Converting from clustering_results format")
                # 这里需要实现转换逻辑
                pass

        except Exception as e:
            print(f"Error loading {result_file}: {e}")
            continue

    print(f"\nTotal objects collected: {len(all_dynamic_objects)}")

    # 保存
    save_dynamic_objects(all_dynamic_objects, output_path)

    return all_dynamic_objects


def create_from_scratch(
    checkpoint_path: str,
    dataset_config: dict,
    processor_config: dict,
    num_samples: int = 100,
    output_path: str = 'dynamic_objects.pkl',
    device: str = 'cuda:0'
):
    """
    从头运行推理并提取dynamic_objects

    Args:
        checkpoint_path: Stage1模型checkpoint
        dataset_config: 数据集配置
        processor_config: Processor配置
        num_samples: 样本数量
        output_path: 输出路径
        device: 设备
    """
    print("Creating dynamic_objects from scratch...")

    # 1. 加载模型
    # model = load_stage1_model(checkpoint_path, device)

    # 2. 创建数据集
    # dataset = create_dataset(dataset_config)

    # 3. 创建processor
    processor = OnlineDynamicProcessor(**processor_config)

    # 4. 运行推理并提取
    # all_dynamic_objects = run_inference_and_extract_objects(
    #     dataset, processor, num_samples, device
    # )

    # 5. 保存
    # save_dynamic_objects(all_dynamic_objects, output_path)

    print("Not fully implemented yet - please use extract_from_existing_results")
    print("Or manually run Stage2 training/inference with save logic")


def main():
    parser = argparse.ArgumentParser(
        description="Generate dynamic_objects data for ICP supervision"
    )

    parser.add_argument('--mode', type=str, required=True,
                        choices=['extract', 'create'],
                        help='extract: from existing results, create: run inference from scratch')

    # Extract mode
    parser.add_argument('--results_dir', type=str,
                        help='Directory containing inference results (for extract mode)')

    # Create mode
    parser.add_argument('--checkpoint', type=str,
                        help='Stage1 checkpoint path (for create mode)')
    parser.add_argument('--dataset_root', type=str,
                        help='Dataset root path (for create mode)')

    # Common
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to process')
    parser.add_argument('--output', type=str, default='dynamic_objects.pkl',
                        help='Output pickle file path')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')

    args = parser.parse_args()

    if args.mode == 'extract':
        if not args.results_dir:
            print("Error: --results_dir is required for extract mode")
            return

        extract_from_existing_results(
            results_dir=args.results_dir,
            num_samples=args.num_samples,
            output_path=args.output
        )

    elif args.mode == 'create':
        if not args.checkpoint or not args.dataset_root:
            print("Error: --checkpoint and --dataset_root are required for create mode")
            return

        print("Create mode - please implement the model loading and inference logic")
        print("For now, please use extract mode with existing results")

        # processor_config = {
        #     'velocity_threshold': 0.1,
        #     'clustering_eps': 0.02,
        #     'clustering_min_samples': 10,
        #     # ... 其他配置
        # }
        #
        # create_from_scratch(
        #     checkpoint_path=args.checkpoint,
        #     dataset_config={'root': args.dataset_root},
        #     processor_config=processor_config,
        #     num_samples=args.num_samples,
        #     output_path=args.output,
        #     device=args.device
        # )


if __name__ == "__main__":
    main()
