#!/usr/bin/env python3
"""
调试数据类型错误的测试脚本
"""
import os
import sys
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src/vggt'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_data_loading():
    """测试数据加载和类型转换"""
    try:
        from src.dust3r.datasets.waymo import Waymo_Multi
        from src.train import cut3r_batch_to_vggt

        # 寻找可用的数据集目录
        possible_dirs = [
            "/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train",
            "/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/test"
        ]

        seq_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                # 查找第一个序列目录
                subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d)) and d.startswith('segment')]
                if subdirs:
                    seq_dir = os.path.join(dir_path, subdirs[0])
                    break

        if seq_dir is None:
            print("✗ No valid dataset directory found")
            return False

        print(f"Using dataset: {seq_dir}")
        root_dir = os.path.dirname(seq_dir)

        # 创建数据集
        dataset = Waymo_Multi(
            split=None,
            ROOT=root_dir,
            img_ray_mask_p=[1.0, 0.0, 0.0],
            valid_camera_id_list=["1"],
            resolution=[(518, 378)],
            num_views=8,
            seed=42,
            n_corres=0,
            seq_aug_crop=True
        )

        # 尝试获取一个样本
        print("Loading sample data...")
        views = dataset.__getitem__((0, 2, 8))

        print(f"Number of views: {len(views)}")

        # 检查每个view的数据类型
        for i, view in enumerate(views):
            print(f"\nView {i}:")
            for key, value in view.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {type(value).__name__} {value.shape}")
                else:
                    print(f"  {key}: {type(value).__name__}")

        # 测试直接使用原始的cut3r_batch_to_vggt
        print("\nTesting original cut3r_batch_to_vggt...")
        try:
            vggt_batch = cut3r_batch_to_vggt(views)
            print("✓ Original function succeeded")
            return True
        except Exception as e:
            print(f"✗ Original function failed: {e}")

            # 尝试修复数据类型
            print("\nTrying to fix data types...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            def ensure_tensor(data, device):
                """确保数据是tensor格式"""
                if isinstance(data, np.ndarray):
                    return torch.from_numpy(data).to(device)
                elif isinstance(data, torch.Tensor):
                    return data.to(device)
                else:
                    return data

            fixed_views = []
            for i, view in enumerate(views):
                fixed_view = {}
                for key, value in view.items():
                    if key in ['img', 'depthmap', 'camera_intrinsics', 'camera_pose', 'valid_mask', 'pts3d']:
                        if value is not None:
                            tensor_value = ensure_tensor(value, device)

                            # 处理img维度
                            if key == 'img':
                                print(f"  Original img shape: {tensor_value.shape}")
                                if tensor_value.dim() == 3:  # [3, H, W]
                                    tensor_value = tensor_value.unsqueeze(0)  # [1, 3, H, W]
                                print(f"  Fixed img shape: {tensor_value.shape}")

                            fixed_view[key] = tensor_value
                        else:
                            fixed_view[key] = value
                    else:
                        fixed_view[key] = value
                fixed_views.append(fixed_view)

            # 再次尝试
            try:
                vggt_batch = cut3r_batch_to_vggt(fixed_views)
                print("✓ Fixed function succeeded")
                return True
            except Exception as e2:
                print(f"✗ Fixed function also failed: {e2}")
                return False

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Testing data loading and type conversion...")
    success = test_data_loading()

    if success:
        print("\n🎉 Data loading test passed!")
    else:
        print("\n⚠️ Data loading test failed!")

    return success

if __name__ == "__main__":
    main()