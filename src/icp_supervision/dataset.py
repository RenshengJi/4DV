"""
ICP Supervision Dataset - 用于加载ICP样本对进行训练
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import glob
from pathlib import Path

from icp_supervision.utils import load_sample_pair, numpy_to_torch


class ICPSupervisionDataset(Dataset):
    """
    ICP监督数据集

    加载由ICPDataGenerator生成的.npz样本对文件
    每个样本包含:
    - input_gaussians: [N, 14] 粗糙Gaussian参数
    - target_gaussians: [N, 14] ICP配准后的GT参数
    - pred_scale: float, 用于体素化
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        train_ratio: float = 0.8,
        seed: int = 42,
        device: str = 'cuda:0',
        cache_in_memory: bool = False
    ):
        """
        Args:
            data_dir: 数据目录 (包含.npz文件)
            split: 'train' or 'val'
            train_ratio: 训练集比例
            seed: 随机种子
            device: PyTorch设备（在DataLoader中会被忽略，数据在CPU上加载）
            cache_in_memory: 是否缓存所有数据到内存
        """
        self.data_dir = data_dir
        self.split = split
        # 在 DataLoader workers 中不使用 CUDA，数据在 CPU 上加载
        self.device = 'cpu'  # 强制使用 CPU
        self.cache_in_memory = cache_in_memory

        # 查找所有.npz文件
        self.npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

        if len(self.npz_files) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")

        # 划分train/val
        np.random.seed(seed)
        indices = np.random.permutation(len(self.npz_files))
        split_idx = int(len(self.npz_files) * train_ratio)

        if split == 'train':
            self.npz_files = [self.npz_files[i] for i in indices[:split_idx]]
        else:  # val
            self.npz_files = [self.npz_files[i] for i in indices[split_idx:]]

        print(f"ICPSupervisionDataset [{split}]: {len(self.npz_files)} samples from {data_dir}")

        # 缓存数据
        self.cache = {}
        if cache_in_memory:
            print("Caching all data to memory...")
            for idx in range(len(self.npz_files)):
                self.cache[idx] = self._load_sample(idx)
            print("Caching complete")

    def _load_sample(self, idx: int) -> Dict:
        """
        加载单个样本

        Returns:
            sample: 包含以下键的字典
                - input_gaussians: torch.Tensor [N, 14]
                - target_gaussians: torch.Tensor [N, 14]
                - pred_scale: torch.Tensor [1]
                - object_id: int
                - sample_idx: int
        """
        npz_path = self.npz_files[idx]
        data = load_sample_pair(npz_path)

        if data is None:
            # 如果加载失败，返回一个dummy样本（在CPU上）
            return {
                'input_gaussians': torch.zeros(1, 14),
                'target_gaussians': torch.zeros(1, 14),
                'pred_scale': torch.tensor([1.0]),
                'object_id': -1,
                'sample_idx': idx,
            }

        # 转换为torch tensor（在CPU上，DataLoader会自动处理设备转移）
        sample = {
            'input_gaussians': numpy_to_torch(data['input_gaussians'], 'cpu').float(),
            'target_gaussians': numpy_to_torch(data['target_gaussians'], 'cpu').float(),
            'pred_scale': torch.tensor([data['pred_scale']]).float(),
            'object_id': data['object_id'],
            'sample_idx': idx,
        }

        return sample

    def __len__(self) -> int:
        return len(self.npz_files)

    def __getitem__(self, idx: int) -> Dict:
        """
        获取样本

        Returns:
            sample: 字典，包含input_gaussians, target_gaussians, pred_scale等
        """
        if self.cache_in_memory and idx in self.cache:
            return self.cache[idx]
        else:
            return self._load_sample(idx)

    def get_sample_info(self, idx: int) -> Dict:
        """获取样本的基本信息（不加载数据）"""
        npz_path = self.npz_files[idx]
        data = np.load(npz_path, allow_pickle=True)

        info = {
            'npz_path': npz_path,
            'object_id': int(data['object_id']),
            'num_points': data['input_gaussians'].shape[0],
            'pred_scale': float(data['pred_scale']),
        }

        if 'input_pcd_path' in data:
            info['input_pcd_path'] = str(data['input_pcd_path'])
        if 'target_pcd_path' in data:
            info['target_pcd_path'] = str(data['target_pcd_path'])

        return info


def collate_fn_icp(batch: List[Dict]) -> Dict:
    """
    自定义collate函数，用于DataLoader

    由于每个object的点数N不同，我们不能简单stack
    策略: 返回list of samples

    Args:
        batch: List of samples

    Returns:
        batched_data: 字典，包含列表形式的数据
    """
    # 过滤掉无效样本 (object_id == -1)
    valid_batch = [sample for sample in batch if sample['object_id'] != -1]

    if len(valid_batch) == 0:
        # 如果没有有效样本，返回一个dummy batch
        return {
            'input_gaussians': [],
            'target_gaussians': [],
            'pred_scale': [],
            'object_id': [],
            'sample_idx': [],
            'batch_size': 0,
        }

    # 返回list of tensors
    batched_data = {
        'input_gaussians': [sample['input_gaussians'] for sample in valid_batch],
        'target_gaussians': [sample['target_gaussians'] for sample in valid_batch],
        'pred_scale': [sample['pred_scale'] for sample in valid_batch],
        'object_id': [sample['object_id'] for sample in valid_batch],
        'sample_idx': [sample['sample_idx'] for sample in valid_batch],
        'batch_size': len(valid_batch),
    }

    return batched_data


def create_icp_dataloaders(
    data_dir: str,
    batch_size: int = 1,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    seed: int = 42,
    device: str = 'cuda:0',
    cache_in_memory: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证DataLoader

    Args:
        data_dir: 数据目录
        batch_size: batch大小 (注意: 由于点数不同，实际batch会是list)
        num_workers: 数据加载线程数
        train_ratio: 训练集比例
        seed: 随机种子
        device: PyTorch设备
        cache_in_memory: 是否缓存数据到内存

    Returns:
        (train_loader, val_loader)
    """
    # 创建数据集
    train_dataset = ICPSupervisionDataset(
        data_dir=data_dir,
        split='train',
        train_ratio=train_ratio,
        seed=seed,
        device=device,
        cache_in_memory=cache_in_memory
    )

    val_dataset = ICPSupervisionDataset(
        data_dir=data_dir,
        split='val',
        train_ratio=train_ratio,
        seed=seed,
        device=device,
        cache_in_memory=cache_in_memory
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_icp,
        pin_memory=False  # 已经在dataset中移到device
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_icp,
        pin_memory=False
    )

    return train_loader, val_loader


# 测试代码
if __name__ == "__main__":
    # 测试数据集加载
    data_dir = "./icp_supervision_data"

    if os.path.exists(data_dir):
        dataset = ICPSupervisionDataset(data_dir, split='train')

        print(f"Dataset size: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print("\nFirst sample:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                else:
                    print(f"  {key}: {value}")

            # 测试DataLoader
            train_loader, val_loader = create_icp_dataloaders(data_dir, batch_size=2)

            print(f"\nTrain loader: {len(train_loader)} batches")
            print(f"Val loader: {len(val_loader)} batches")

            batch = next(iter(train_loader))
            print(f"\nFirst batch:")
            print(f"  batch_size: {batch['batch_size']}")
            if batch['batch_size'] > 0:
                print(f"  input_gaussians[0].shape: {batch['input_gaussians'][0].shape}")
                print(f"  target_gaussians[0].shape: {batch['target_gaussians'][0].shape}")
    else:
        print(f"Data directory {data_dir} does not exist")
        print("Please run data_generator.py first to generate ICP supervision data")
