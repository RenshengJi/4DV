#!/usr/bin/env python3
"""
简单测试VGGT内存优化
"""

import torch
import sys
sys.path.append('src/vggt')
from vggt.models.vggt import VGGT

def test_simple():
    """简单测试"""
    print("简单测试VGGT内存优化...")

    device = "cpu"  # 使用CPU避免GPU内存问题

    # 创建小规模测试数据
    images = torch.randn(1, 10, 3, 378, 518, device=device)

    print("测试内存优化模式:")
    try:
        model_opt = VGGT(memory_efficient=True).to(device)
        model_opt.eval()

        with torch.no_grad():
            predictions = model_opt(images)

        print("成功！预测键:", list(predictions.keys()))
        print("形状:")
        for key, value in predictions.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")

    except Exception as e:
        print(f"内存优化模式失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n测试原始模式:")
    try:
        model_orig = VGGT(memory_efficient=False).to(device)
        model_orig.eval()

        with torch.no_grad():
            predictions = model_orig(images)

        print("成功！预测键:", list(predictions.keys()))
        print("形状:")
        for key, value in predictions.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")

    except Exception as e:
        print(f"原始模式失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple()