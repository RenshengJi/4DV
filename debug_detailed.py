#!/usr/bin/env python3
"""
详细调试aggregator的执行过程
"""

import torch
import sys
sys.path.append('src/vggt')
from vggt.models.aggregator import Aggregator

def debug_detailed():
    """详细调试aggregator"""
    print("详细调试aggregator...")

    # 创建优化模式aggregator
    aggregator = Aggregator(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        memory_efficient=True,
        output_layers=[4, 11, 17, 23]
    )

    # 添加调试打印到forward方法
    original_forward = aggregator.forward

    def debug_forward(images):
        B, S, C_in, H, W = images.shape
        print(f"Input shape: B={B}, S={S}, C={C_in}, H={H}, W={W}")

        # 重复前面的步骤...
        images = (images - aggregator._resnet_mean) / aggregator._resnet_std
        images = images.view(B * S, C_in, H, W)
        patch_tokens = aggregator.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape
        print(f"Patch tokens shape: {patch_tokens.shape}")

        # ... 添加tokens...

        # 核心循环
        frame_idx = 0
        global_idx = 0
        output_list = []
        layer_outputs = {}
        current_layer_idx = 0

        print(f"aa_block_num: {aggregator.aa_block_num}")
        print(f"output_layers: {aggregator.output_layers}")

        for block_num in range(aggregator.aa_block_num):
            print(f"\nProcessing block {block_num}")
            frame_intermediates = []
            global_intermediates = []

            for attn_type in aggregator.aa_order:
                print(f"  Processing {attn_type} attention, current_layer_idx: {current_layer_idx}")

                if attn_type == "frame":
                    # 简化调用以查看层处理
                    for i in range(aggregator.aa_block_size):
                        layer_num = current_layer_idx + i
                        should_output = layer_num in aggregator.output_layers
                        print(f"    Frame layer {layer_num}: {'SAVE' if should_output else 'SKIP'}")
                        if should_output:
                            # 模拟保存 - 添加一个假的tensor
                            fake_tensor = torch.zeros(B, S, P, C)
                            frame_intermediates.append(fake_tensor)
                        else:
                            frame_intermediates.append(None)
                elif attn_type == "global":
                    for i in range(aggregator.aa_block_size):
                        layer_num = current_layer_idx + i
                        should_output = layer_num in aggregator.output_layers
                        print(f"    Global layer {layer_num}: {'SAVE' if should_output else 'SKIP'}")
                        if should_output:
                            # 模拟保存 - 添加一个假的tensor
                            fake_tensor = torch.zeros(B, S, P, C)
                            global_intermediates.append(fake_tensor)
                        else:
                            global_intermediates.append(None)

                current_layer_idx += aggregator.aa_block_size

            print(f"  Frame intermediates: {len([x for x in frame_intermediates if x is not None])}")
            print(f"  Global intermediates: {len([x for x in global_intermediates if x is not None])}")

            # Store outputs
            for i in range(len(frame_intermediates)):
                if frame_intermediates[i] is not None and global_intermediates[i] is not None:
                    layer_idx = block_num * len(aggregator.aa_order) * aggregator.aa_block_size + i
                    print(f"  Storing output for layer_idx {layer_idx}")
                    # 模拟concat
                    concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                    layer_outputs[layer_idx] = concat_inter

        print(f"\nTotal layer_outputs: {len(layer_outputs)}")
        print(f"Layer indices: {sorted(layer_outputs.keys())}")

        # Build output list
        for layer_idx in sorted(layer_outputs.keys()):
            output_list.append(layer_outputs[layer_idx])

        print(f"Final output count: {len(output_list)}")
        return output_list, aggregator.patch_start_idx

    # 使用调试版本
    aggregator.forward = debug_forward

    # 测试
    device = "cpu"
    images = torch.randn(1, 5, 3, 378, 518, device=device)
    aggregator = aggregator.to(device)

    with torch.no_grad():
        output, _ = aggregator(images)

if __name__ == "__main__":
    debug_detailed()