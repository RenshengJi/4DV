#!/usr/bin/env python3
"""
测试VGGT内存优化效果的脚本
比较原始实现和内存优化实现的内存使用情况
"""

import torch
import time
import gc
import psutil
import os
from contextlib import contextmanager
import sys
sys.path.append('src/vggt')
from vggt.models.vggt import VGGT


@contextmanager
def memory_monitor(label):
    """监控内存使用情况的上下文管理器"""
    # 清理内存
    gc.collect()
    torch.cuda.empty_cache()

    # 获取初始内存
    process = psutil.Process(os.getpid())
    initial_ram = process.memory_info().rss / 1024 / 1024  # MB
    initial_gpu = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0  # MB

    print(f"\n{label} - 初始内存:")
    print(f"  RAM: {initial_ram:.1f} MB")
    print(f"  GPU: {initial_gpu:.1f} MB")

    start_time = time.time()

    try:
        yield
    finally:
        # 获取最终内存
        final_ram = process.memory_info().rss / 1024 / 1024  # MB
        final_gpu = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0  # MB
        peak_gpu = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0  # MB

        duration = time.time() - start_time

        print(f"{label} - 最终内存:")
        print(f"  RAM: {final_ram:.1f} MB (增加 {final_ram - initial_ram:.1f} MB)")
        print(f"  GPU: {final_gpu:.1f} MB (增加 {final_gpu - initial_gpu:.1f} MB)")
        print(f"  GPU Peak: {peak_gpu:.1f} MB")
        print(f"  执行时间: {duration:.2f}s")


def test_model_memory(memory_efficient=True, sequence_length=50, batch_size=1):
    """测试模型的内存使用情况"""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建模型
    model = VGGT(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        memory_efficient=memory_efficient
    ).to(device)

    model.eval()

    # 创建测试数据
    H, W = 378, 518
    images = torch.randn(batch_size, sequence_length, 3, H, W, device=device)

    mode_name = "内存优化模式" if memory_efficient else "原始模式"

    with memory_monitor(f"{mode_name} (S={sequence_length})"):
        with torch.no_grad():
            try:
                predictions = model(images)
                print(f"  成功处理 {sequence_length} 帧")
                print(f"  输出中间层数量: {len(model.aggregator.output_layers) if hasattr(model.aggregator, 'output_layers') else 'N/A'}")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  内存不足! 无法处理 {sequence_length} 帧")
                else:
                    print(f"  错误: {e}")

    # 清理
    del model, images
    if 'predictions' in locals():
        del predictions
    gc.collect()
    torch.cuda.empty_cache()


def test_different_sequence_lengths():
    """测试不同序列长度下的内存使用情况"""
    print("=" * 60)
    print("VGGT内存优化测试")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU测试")

    # 测试不同的序列长度
    sequence_lengths = [20, 50, 100, 150, 200]

    for seq_len in sequence_lengths:
        print(f"\n{'='*60}")
        print(f"测试序列长度: {seq_len}")
        print('='*60)

        # 测试原始模式
        try:
            test_model_memory(memory_efficient=False, sequence_length=seq_len)
        except Exception as e:
            print(f"原始模式失败: {e}")

        # 测试内存优化模式
        try:
            test_model_memory(memory_efficient=True, sequence_length=seq_len)
        except Exception as e:
            print(f"内存优化模式失败: {e}")


def test_layer_outputs():
    """测试输出层数量的差异"""
    print(f"\n{'='*60}")
    print("层输出数量测试")
    print('='*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建小规模测试数据
    images = torch.randn(1, 10, 3, 378, 518, device=device)

    # 测试原始模式
    model_original = VGGT(memory_efficient=False).to(device)
    model_original.eval()

    with torch.no_grad():
        aggregated_tokens_list, _ = model_original.aggregator(images)

    print(f"原始模式输出层数量: {len(aggregated_tokens_list)}")

    # 测试内存优化模式
    model_optimized = VGGT(memory_efficient=True).to(device)
    model_optimized.eval()

    with torch.no_grad():
        aggregated_tokens_list_opt, _ = model_optimized.aggregator(images)

    print(f"内存优化模式输出层数量: {len(aggregated_tokens_list_opt)}")
    print(f"内存优化输出层索引: {sorted(model_optimized.aggregator.output_layers)}")

    # 清理
    del model_original, model_optimized, images, aggregated_tokens_list, aggregated_tokens_list_opt
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # 测试层输出数量
    test_layer_outputs()

    # 测试不同序列长度
    test_different_sequence_lengths()

    print(f"\n{'='*60}")
    print("测试完成")
    print('='*60)