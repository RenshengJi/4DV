#!/usr/bin/env python3
"""
简化的Stage1推理测试脚本
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/vggt'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 测试基本的导入和模型加载
def test_imports():
    print("Testing imports...")
    try:
        from vggt.models.vggt import VGGT
        print("✓ VGGT import successful")

        from src.dust3r.datasets.waymo import Waymo_Multi
        print("✓ Waymo_Multi import successful")

        from src.train import cut3r_batch_to_vggt
        print("✓ cut3r_batch_to_vggt import successful")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_loading():
    print("\nTesting model loading...")
    try:
        import torch
        from vggt.models.vggt import VGGT

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # 创建模型（不加载权重）
        model = VGGT(
            img_size=518,
            patch_size=14,
            embed_dim=1024,
            use_sky_token=True
        )
        model.to(device)
        model.eval()

        print("✓ Model creation successful")

        # 测试一个简单的前向传播
        with torch.no_grad():
            dummy_input = torch.randn(1, 8, 3, 518, 378, device=device)
            try:
                output = model(dummy_input, compute_sky_color_loss=True, sky_masks=None, gt_images=dummy_input)
                print("✓ Forward pass successful")
                print(f"Output keys: {list(output.keys())}")

                # 检查输出维度
                for key, value in output.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.shape}")

                return True
            except Exception as e:
                print(f"✗ Forward pass failed: {e}")
                return False

    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_dataset_loading():
    print("\nTesting dataset loading...")
    try:
        from src.dust3r.datasets.waymo import Waymo_Multi

        seq_dir = "/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/test/segment-11717495969710734380_2440_000_2460_000_with_camera_labels"
        root_dir = os.path.dirname(seq_dir)

        if not os.path.exists(seq_dir):
            print(f"✗ Dataset directory not found: {seq_dir}")
            return False

        dataset = Waymo_Multi(
            split=None,
            ROOT=root_dir,
            img_ray_mask_p=[1.0, 0.0, 0.0],
            valid_camera_id_list=["1"],
            resolution=[(518, 378)],  # 简化分辨率
            num_views=8,  # 减少视图数量
            seed=42,
            n_corres=0,
            seq_aug_crop=True
        )

        print("✓ Dataset creation successful")

        # 测试获取一个样本
        try:
            views = dataset.__getitem__((0, 2, 8))
            print("✓ Dataset sample loading successful")
            print(f"Number of views: {len(views)}")

            return True
        except Exception as e:
            print(f"✗ Dataset sample loading failed: {e}")
            return False

    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False

def main():
    print("=" * 50)
    print("Stage1 Inference Test Suite")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Model Loading", test_model_loading),
        ("Dataset Loading", test_dataset_loading),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n[{passed+1}/{total}] Running {test_name} test...")
        if test_func():
            passed += 1
            print(f"✓ {test_name} test PASSED")
        else:
            print(f"✗ {test_name} test FAILED")

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! Stage1 inference should work.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

    return passed == total

if __name__ == "__main__":
    main()