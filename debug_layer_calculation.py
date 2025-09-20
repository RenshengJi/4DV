#!/usr/bin/env python3
"""
调试层计算逻辑
"""

def debug_layer_calculation():
    """调试层计算"""
    print("调试层计算逻辑...")

    # VGGT参数
    depth = 24
    aa_order = ["frame", "global"]
    aa_block_size = 1
    aa_block_num = depth // aa_block_size  # 24

    print(f"depth: {depth}")
    print(f"aa_order: {aa_order}")
    print(f"aa_block_size: {aa_block_size}")
    print(f"aa_block_num: {aa_block_num}")

    # 模拟原始逻辑
    print("\n模拟原始逻辑:")
    frame_idx = 0
    global_idx = 0
    output_count = 0

    for block_num in range(aa_block_num):
        print(f"\nblock_num: {block_num}")
        frame_intermediates_count = 0
        global_intermediates_count = 0

        for attn_type in aa_order:
            if attn_type == "frame":
                # 模拟_process_frame_attention
                for i in range(aa_block_size):
                    frame_intermediates_count += 1
                    frame_idx += 1
                    print(f"  frame block {frame_idx-1} processed")
            elif attn_type == "global":
                # 模拟_process_global_attention
                for i in range(aa_block_size):
                    global_intermediates_count += 1
                    global_idx += 1
                    print(f"  global block {global_idx-1} processed")

        # 模拟输出添加
        intermediates_count = max(frame_intermediates_count, global_intermediates_count)
        print(f"  frame_intermediates: {frame_intermediates_count}")
        print(f"  global_intermediates: {global_intermediates_count}")
        print(f"  will add {intermediates_count} outputs")
        output_count += intermediates_count

    print(f"\n总输出数量: {output_count}")

    # 分析为什么原始输出是12个
    print(f"\n分析:")
    print(f"原始应该输出: {aa_block_num} * {aa_block_size} = {aa_block_num * aa_block_size} 个")
    print(f"但实际输出: 12 个")
    print(f"可能的解释: aa_block_num = {aa_block_num // 2} (每2个block算一组)")

    # 测试我的新逻辑
    print(f"\n测试新逻辑:")
    output_layers = {4, 11, 17, 23}

    for block_num in range(aa_block_num):
        current_block_layer = block_num * len(aa_order) * aa_block_size
        print(f"\nblock_num: {block_num}, current_block_layer: {current_block_layer}")

        for attn_idx, attn_type in enumerate(aa_order):
            current_layer = current_block_layer + attn_idx * aa_block_size
            print(f"  {attn_type} attention, current_layer: {current_layer}")

            for i in range(aa_block_size):
                layer_num = current_layer + i
                should_output = layer_num in output_layers
                print(f"    layer {layer_num}: {'OUTPUT' if should_output else 'skip'}")

if __name__ == "__main__":
    debug_layer_calculation()