#!/usr/bin/env python3
"""
快速测试Stage1推理脚本的修复
"""
import sys
import os
sys.path.append('.')

# 测试单次推理，使用一个小的idx
if __name__ == "__main__":
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # 导入并运行demo_stage1_inference的main函数，但修改参数
    import demo_stage1_inference

    # 模拟命令行参数
    class Args:
        def __init__(self):
            self.model_path = "/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo/step2(true+fixmodel+lowlr!+nolpips+onlyflow+velocitylocal+fromscratch)/checkpoint-epoch_2_17880.pth"
            self.seq_dir = "/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train"
            self.output_dir = "./stage1_test_outputs"
            self.idx = 0  # 使用idx 0
            self.device = "cuda"
            self.num_views = 4  # 减少views数量以加快测试
            self.batch_mode = False
            self.start_idx = 0
            self.end_idx = 0
            self.step = 1
            self.continue_on_error = False

    args = Args()

    print("Quick test of Stage1 inference...")
    print(f"Using idx: {args.idx}")
    print(f"Using num_views: {args.num_views}")

    try:
        demo_stage1_inference.main.__globals__['args'] = args
        demo_stage1_inference.main()
        print("✓ Test completed successfully!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()