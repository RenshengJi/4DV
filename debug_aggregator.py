#!/usr/bin/env python3
"""
调试aggregator的输出
"""

import torch
import sys
sys.path.append('src/vggt')
from vggt.models.aggregator import Aggregator

def debug_aggregator():
    """调试aggregator的层输出"""
    print("调试aggregator...")

    # 创建小规模测试
    aggregator_orig = Aggregator(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        memory_efficient=False,
        output_layers=None
    )

    aggregator_opt = Aggregator(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        memory_efficient=True,
        output_layers=[4, 11, 17, 23]
    )

    print(f"原始模式 output_layers: {aggregator_orig.output_layers}")
    print(f"优化模式 output_layers: {aggregator_opt.output_layers}")
    print(f"优化模式 depth: {aggregator_opt.depth}")
    print(f"优化模式 aa_block_num: {aggregator_opt.aa_block_num}")
    print(f"优化模式 aa_order: {aggregator_opt.aa_order}")
    print(f"优化模式 aa_block_size: {aggregator_opt.aa_block_size}")

    # 创建测试数据
    device = "cpu"  # 使用CPU避免GPU内存问题
    images = torch.randn(1, 5, 3, 378, 518, device=device)

    aggregator_orig = aggregator_orig.to(device)
    aggregator_opt = aggregator_opt.to(device)

    print("\n原始模式测试:")
    try:
        with torch.no_grad():
            output_orig, patch_idx_orig = aggregator_orig(images)
        print(f"原始模式输出数量: {len(output_orig)}")
        print(f"原始模式输出形状: {[x.shape for x in output_orig]}")
    except Exception as e:
        print(f"原始模式错误: {e}")

    print("\n优化模式测试:")
    try:
        with torch.no_grad():
            output_opt, patch_idx_opt = aggregator_opt(images)
        print(f"优化模式输出数量: {len(output_opt)}")
        print(f"优化模式输出形状: {[x.shape for x in output_opt]}")
    except Exception as e:
        print(f"优化模式错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_aggregator()