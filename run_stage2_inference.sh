#!/bin/bash

# Stage2推理运行脚本 - 增强版本，输出RGB比较和动态多帧匹配聚类
# 输出格式：四列比較 GT | Initial | Refined | Dynamic Clustering

# 配置参数
START_IDX=${1:-150}     # 起始idx，默认150
STEP=${2:-5}            # 步长，默认5
END_IDX=${3:-200}       # 结束idx，默认200

echo "Starting Stage2 comparison inference (optimized Python version)..."
echo "Parameters:"
echo "  Start IDX: ${START_IDX}"
echo "  End IDX: ${END_IDX}"
echo "  Step: ${STEP}"
echo "  Mode: Enhanced comparison with dynamic clustering"
echo "  Output: 4-column layout (GT | Initial | Refined | Dynamic Clustering)"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 模型路径
# STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo/step2(true+fixmodel+lowlr!+nolpips+onlyflow+velocitylocal+fromscratch)/checkpoint-epoch_2_17880.pth"
STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/stage1_online_unfreeze+newsky+highvelocity+flownosky+gt+fixedextrinsic+detach/checkpoint-epoch_0_65100.pth"

STAGE2_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage2_online/stage2-checkpoint-epoch_50_96.pth"

# 数据路径
# SEQ_DIR="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/test/segment-11717495969710734380_2440_000_2460_000_with_camera_labels"
SEQ_DIR="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train/segment-15795616688853411272_1245_000_1265_000_with_camera_labels"



# 输出目录
OUTPUT_DIR="./stage2_inference_outputs"

# 构建Python批量推理命令 - 现在总是生成比较视频
PYTHON_CMD="/opt/miniconda/envs/vggt/bin/python demo_stage2_inference.py \
    --batch_mode \
    --start_idx ${START_IDX} \
    --end_idx ${END_IDX} \
    --step ${STEP} \
    --model_path \"${STAGE1_MODEL_PATH}\" \
    --stage2_model_path \"${STAGE2_MODEL_PATH}\" \
    --seq_dir \"${SEQ_DIR}\" \
    --output_dir \"${OUTPUT_DIR}\" \
    --device cuda \
    --num_views 8 \
    --continue_on_error"

echo "Generating enhanced videos with dynamic clustering"
echo "  - GT: Ground truth images"
echo "  - Initial: Without Stage2 refinement"
echo "  - Refined: With Stage2 refinement"
echo "  - Dynamic Clustering: Multi-frame object tracking visualization"

echo "Running optimized batch inference..."
echo "Command: ${PYTHON_CMD}"
echo ""

# 执行Python批量推理（模型只加载一次，大幅提升效率！）
eval $PYTHON_CMD

echo ""
echo "Batch inference script completed!"
echo ""
echo "Features of this enhanced version:"
echo "  ✓ Models loaded only once (much faster!)"
echo "  ✓ Dataset initialized once (saves time)"
echo "  ✓ RGB comparison (GT | Initial | Refined)"
echo "  ✓ Dynamic object clustering and tracking"
echo "  ✓ Multi-frame correspondence visualization"
echo "  ✓ Better error handling and progress tracking"
echo ""
echo "Usage examples:"
echo "  ./run_stage2_inference.sh                    # 默认: idx 150-200, step 5, 增强模式"
echo "  ./run_stage2_inference.sh 100 10 150        # idx 100-150, step 10, 增强模式"
echo "  ./run_stage2_inference.sh 0 1 10            # idx 0-10, step 1, 增强模式"
echo ""
echo "Or use Python directly:"
echo "  /opt/miniconda/envs/vggt/bin/python demo_stage2_inference.py --batch_mode --start_idx 150 --end_idx 200 --step 5"
echo "  /opt/miniconda/envs/vggt/bin/python demo_stage2_inference.py --idx 150  # single inference"