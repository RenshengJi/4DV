#!/bin/bash

# Stage1 Aggregator Render推理运行脚本
# 输出格式：五列比较 GT | Aggregator Render | Velocity Map | GT Velocity | Sky Color
# 使用aggregator_render_loss中的渲染方式

# 配置参数
START_IDX=${1:-150}            # 起始idx，默认150
STEP=${2:-5}                   # 步长，默认5
END_IDX=${3:-200}              # 结束idx，默认200
VOXEL_SIZE=${4:-0.05}          # 体素大小，默认0.05 (5cm)
DYNAMIC_THRESHOLD=${5:-0.01}    # 动静态分离阈值，默认0.01 (m/s)


echo "Starting Stage1 Aggregator Render inference..."
echo "Parameters:"
echo "  Start IDX: ${START_IDX}"
echo "  End IDX: ${END_IDX}"
echo "  Step: ${STEP}"
echo "  Voxel Size: ${VOXEL_SIZE}"
echo "  Dynamic Threshold: ${DYNAMIC_THRESHOLD} m/s"
echo "  Output: 5-column layout (GT | Aggregator Render | Velocity Map | GT Velocity | Sky Color)"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 模型路径
# STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/stage1_gtflow+depthgrad(true)+depth+flowgradconf+aggregatorenderloss+fixopacity+no1opacityloss+fixdirection+fromv/checkpoint-epoch_0_22784.pth"
# STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/stage1_gtflow+depthgrad(true)+depth+flowgradconf+aggregatorenderloss+fixopacity+no1opacityloss+fixdirection+fromvresume+lpips+noquantize/checkpoint-epoch_0_19530.pth"
# STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/stage1_gtflow+depthgrad(true)+depth+flowgradconf+aggregatorenderloss+fixopacity+no1opacityloss+fixdirection+fromvresume+lpips+quantize0.05/checkpoint-epoch_0_8544.pth"
STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/aggregator_resume_noflowgrad_nearestdynamic/checkpoint-epoch_0_26040.pth"

# 光流模型路径
FLOW_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/Tartan-C-T-TSKH-kitti432x960-M.pth"

# 数据路径
SEQ_DIR="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train_with_flow/segment-15795616688853411272_1245_000_1265_000_with_camera_labels"

# 输出目录
OUTPUT_DIR="./stage1_aggregator_inference_outputs"

# 构建Python批量推理命令
PYTHON_CMD="/opt/miniconda/envs/vggt/bin/python demo_stage1_inference_for_aggregator.py \
    --batch_mode \
    --start_idx ${START_IDX} \
    --end_idx ${END_IDX} \
    --step ${STEP} \
    --model_path \"${STAGE1_MODEL_PATH}\" \
    --flow_model_path \"${FLOW_MODEL_PATH}\" \
    --seq_dir \"${SEQ_DIR}\" \
    --output_dir \"${OUTPUT_DIR}\" \
    --device cuda \
    --num_views 24 \
    --voxel_size ${VOXEL_SIZE} \
    --dynamic_threshold ${DYNAMIC_THRESHOLD} \
    --continue_on_error"

echo "Generating Stage1 Aggregator Render inference videos"
echo "  - GT: Ground truth images"
echo "  - Aggregator Render: Multi-frame aggregated gaussian rendering (voxel quantized)"
echo "  - Velocity Map: Velocity field visualization"
echo "  - GT Velocity: Ground truth velocity visualization"
echo "  - Sky Color: Sky color prediction visualization"

echo "Running batch inference..."
echo "Command: ${PYTHON_CMD}"
echo ""

# 执行Python批量推理
eval $PYTHON_CMD

echo ""
echo "Stage1 Aggregator Render batch inference script completed!"
echo ""
echo "Features of this Stage1 Aggregator Render inference:"
echo "  ✓ Model loaded only once (efficient!)"
echo "  ✓ Dataset initialized once (saves time)"
echo "  ✓ Five-way comparison (GT | Aggregator Render | Velocity | GT Velocity | Sky)"
echo "  ✓ Aggregator rendering with voxel quantization (voxel_size=${VOXEL_SIZE})"
echo "  ✓ Dynamic-static separation (threshold=${DYNAMIC_THRESHOLD} m/s)"
echo "  ✓ Static gaussians: global voxel quantization (all frames)"
echo "  ✓ Dynamic gaussians: temporal window (prev/curr/next frame only)"
echo "  ✓ Multi-frame gaussian aggregation with temporal propagation"
echo "  ✓ Velocity field visualization"
echo "  ✓ GT velocity visualization for comparison"
echo "  ✓ Sky color prediction visualization"
echo "  ✓ Better error handling and progress tracking"
echo ""
echo "Usage examples:"
echo "  ./run_stage1_inference_for_aggregator.sh                          # 默认: idx 150-200, step 5, voxel 0.05, threshold 0.1"
echo "  ./run_stage1_inference_for_aggregator.sh 100 10 150               # idx 100-150, step 10, voxel 0.05, threshold 0.1"
echo "  ./run_stage1_inference_for_aggregator.sh 0 1 10 0.1               # idx 0-10, step 1, voxel 0.1, threshold 0.1"
echo "  ./run_stage1_inference_for_aggregator.sh 150 5 200 0.05 0.2       # idx 150-200, step 5, voxel 0.05, threshold 0.2"
echo "  ./run_stage1_inference_for_aggregator.sh 150 5 200 0 0.1          # idx 150-200, step 5, no voxel quantization, threshold 0.1"
echo ""
echo "Or use Python directly:"
echo "  /opt/miniconda/envs/vggt/bin/python demo_stage1_inference_for_aggregator.py --batch_mode --start_idx 150 --end_idx 200 --step 5 --voxel_size 0.05 --dynamic_threshold 0.1"
echo "  /opt/miniconda/envs/vggt/bin/python demo_stage1_inference_for_aggregator.py --idx 150 --voxel_size 0.05 --dynamic_threshold 0.1  # single inference"
echo ""
echo "Output videos will be saved to: ${OUTPUT_DIR}"
