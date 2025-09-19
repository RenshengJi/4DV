#!/bin/bash

# Stage1推理运行脚本
# 输出格式：四列比較 GT | Self-Render | Velocity Map | Sky Color

# 配置参数
START_IDX=${1:-150}     # 起始idx，默认150
STEP=${2:-5}            # 步长，默认5
END_IDX=${3:-200}       # 结束idx，默认200

echo "Starting Stage1 inference..."
echo "Parameters:"
echo "  Start IDX: ${START_IDX}"
echo "  End IDX: ${END_IDX}"
echo "  Step: ${STEP}"
echo "  Output: 4-column layout (GT | Self-Render | Velocity Map | Sky Color)"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 模型路径
# STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo/step2(true+fixmodel+lowlr!+nolpips+onlyflow+velocitylocal+fromscratch)/checkpoint-epoch_2_17880.pth"
STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/stage1_online/checkpoint-epoch_0_58590.pth"


# 数据路径
# SEQ_DIR="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/test/segment-11717495969710734380_2440_000_2460_000_with_camera_labels"
SEQ_DIR="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train/segment-15795616688853411272_1245_000_1265_000_with_camera_labels"

# 输出目录
OUTPUT_DIR="./stage1_inference_outputs"

# 构建Python批量推理命令
PYTHON_CMD="/opt/miniconda/envs/vggt/bin/python demo_stage1_inference.py \
    --batch_mode \
    --start_idx ${START_IDX} \
    --end_idx ${END_IDX} \
    --step ${STEP} \
    --model_path \"${STAGE1_MODEL_PATH}\" \
    --seq_dir \"${SEQ_DIR}\" \
    --output_dir \"${OUTPUT_DIR}\" \
    --device cuda \
    --num_views 8 \
    --continue_on_error"

echo "Generating Stage1 inference videos"
echo "  - GT: Ground truth images"
echo "  - Self-Render: Gaussian splatting rendered images"
echo "  - Velocity Map: Velocity field visualization"
echo "  - Sky Color: Sky color prediction visualization"

echo "Running batch inference..."
echo "Command: ${PYTHON_CMD}"
echo ""

# 执行Python批量推理
eval $PYTHON_CMD

echo ""
echo "Stage1 batch inference script completed!"
echo ""
echo "Features of this Stage1 inference:"
echo "  ✓ Model loaded only once (efficient!)"
echo "  ✓ Dataset initialized once (saves time)"
echo "  ✓ Four-way comparison (GT | Self-Render | Velocity | Sky)"
echo "  ✓ Gaussian splatting based rendering"
echo "  ✓ Velocity field visualization"
echo "  ✓ Sky color prediction visualization"
echo "  ✓ Better error handling and progress tracking"
echo ""
echo "Usage examples:"
echo "  ./run_stage1_inference.sh                    # 默认: idx 150-200, step 5"
echo "  ./run_stage1_inference.sh 100 10 150        # idx 100-150, step 10"
echo "  ./run_stage1_inference.sh 0 1 10            # idx 0-10, step 1"
echo ""
echo "Or use Python directly:"
echo "  /opt/miniconda/envs/vggt/bin/python demo_stage1_inference.py --batch_mode --start_idx 150 --end_idx 200 --step 5"
echo "  /opt/miniconda/envs/vggt/bin/python demo_stage1_inference.py --idx 150  # single inference"
echo ""
echo "Output videos will be saved to: ${OUTPUT_DIR}"