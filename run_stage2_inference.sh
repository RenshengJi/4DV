#!/bin/bash

# Stage2推理运行脚本 - 增强版本，输出RGB比较和动态多帧匹配聚类
# 输出格式：四列比較 GT | Initial | Refined | Dynamic Clustering

# 配置参数
START_IDX=${1:-150}     # 起始idx，默认150
STEP=${2:-5}            # 步长，默认5
END_IDX=${3:-200}       # 结束idx，默认200
USE_VELOCITY=${4:-false}  # 是否使用velocity-based方法，默认false（使用光流）

# 动态物体聚类和跟踪参数
VELOCITY_THRESHOLD=${5:-0.1}              # 速度阈值，默认0.1
CLUSTERING_EPS=${6:-0.02}                 # DBSCAN聚类eps参数，默认0.02
CLUSTERING_MIN_SAMPLES=${7:-10}           # DBSCAN聚类min_samples参数，默认10
TRACKING_POSITION_THRESHOLD=${8:-2.0}     # 跟踪位置阈值，默认2.0
TRACKING_VELOCITY_THRESHOLD=${9:-0.2}     # 跟踪速度阈值，默认0.2

echo "Starting Stage2 comparison inference (optimized Python version)..."
echo "Parameters:"
echo "  Start IDX: ${START_IDX}"
echo "  End IDX: ${END_IDX}"
echo "  Step: ${STEP}"
echo "  Transformation Method: $([ "$USE_VELOCITY" = "true" ] && echo "Velocity-based" || echo "Flow-based")"
echo "  Mode: Enhanced comparison with dynamic clustering"
echo "  Output: 4-column layout (GT | Initial | Refined | Dynamic Clustering)"
echo ""
echo "Clustering & Tracking Parameters:"
echo "  Velocity Threshold: ${VELOCITY_THRESHOLD}"
echo "  Clustering EPS: ${CLUSTERING_EPS}"
echo "  Clustering Min Samples: ${CLUSTERING_MIN_SAMPLES}"
echo "  Tracking Position Threshold: ${TRACKING_POSITION_THRESHOLD}"
echo "  Tracking Velocity Threshold: ${TRACKING_VELOCITY_THRESHOLD}"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=7

# 模型路径
STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/aggregator_resume_noflowgrad_nearestdynamic_resume_0point1_novoxel/checkpoint-epoch_0_45568.pth"
# STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo/step2(true+fixmodel+lowlr!+nolpips+onlyflow+velocitylocal+fromscratch)/checkpoint-epoch_2_17880.pth"

# STAGE2_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage2_online/stage2_train+lr5e-4+0.00001init+poserefine(cross+9)+fixcuml+fixposeinit/stage2-checkpoint-epoch_0_19915.pth"
STAGE2_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage2_online/stage2_train+lr5e-4+0.00001init+poserefine(cross+9)+fixcuml+fixposeinit+fixvelocitytop+fixddp+lr5e-2(true)+larger+fixvoxel+biggermodel/stage2-checkpoint-epoch_0_569.pth"

# 数据路径
# SEQ_DIR="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/test/segment-11717495969710734380_2440_000_2460_000_with_camera_labels"
SEQ_DIR="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train_with_flow/segment-15795616688853411272_1245_000_1265_000_with_camera_labels"



# 输出目录
OUTPUT_DIR="./stage2_inference_outputs"

# 构建Python批量推理命令 - 现在总是生成比较视频
# 根据USE_VELOCITY参数添加--use_velocity_based_transform标志
if [ "$USE_VELOCITY" = "true" ]; then
    VELOCITY_FLAG="--use_velocity_based_transform"
else
    VELOCITY_FLAG=""
fi

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
    --continue_on_error \
    --velocity_threshold ${VELOCITY_THRESHOLD} \
    --clustering_eps ${CLUSTERING_EPS} \
    --clustering_min_samples ${CLUSTERING_MIN_SAMPLES} \
    --tracking_position_threshold ${TRACKING_POSITION_THRESHOLD} \
    --tracking_velocity_threshold ${TRACKING_VELOCITY_THRESHOLD} \
    ${VELOCITY_FLAG}"

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
echo "  ./run_stage2_inference.sh                                    # 默认参数"
echo "  ./run_stage2_inference.sh 100 10 150                        # idx 100-150, step 10"
echo "  ./run_stage2_inference.sh 100 10 150 true                   # Velocity-based"
echo "  ./run_stage2_inference.sh 100 10 150 false 0.15             # 自定义velocity_threshold=0.15"
echo "  ./run_stage2_inference.sh 100 10 150 false 0.1 0.03         # 自定义clustering_eps=0.03"
echo "  ./run_stage2_inference.sh 100 10 150 false 0.1 0.02 15      # 自定义min_samples=15"
echo "  ./run_stage2_inference.sh 100 10 150 false 0.1 0.02 10 3.0  # 自定义tracking_position=3.0"
echo "  ./run_stage2_inference.sh 100 10 150 false 0.1 0.02 10 2.0 0.3  # 完整自定义"
echo ""
echo "Parameters:"
echo "  \$1: START_IDX (default: 150)"
echo "  \$2: STEP (default: 5)"
echo "  \$3: END_IDX (default: 200)"
echo "  \$4: USE_VELOCITY - true=Velocity-based, false=Flow-based (default: false)"
echo "  \$5: VELOCITY_THRESHOLD (default: 0.1)"
echo "  \$6: CLUSTERING_EPS (default: 0.02)"
echo "  \$7: CLUSTERING_MIN_SAMPLES (default: 10)"
echo "  \$8: TRACKING_POSITION_THRESHOLD (default: 2.0)"
echo "  \$9: TRACKING_VELOCITY_THRESHOLD (default: 0.2)"
echo ""
echo "Or use Python directly:"
echo "  /opt/miniconda/envs/vggt/bin/python demo_stage2_inference.py --batch_mode --start_idx 150 --end_idx 200 --step 5"
echo "  /opt/miniconda/envs/vggt/bin/python demo_stage2_inference.py --batch_mode --start_idx 150 --end_idx 200 --step 5 --use_velocity_based_transform"
echo "  /opt/miniconda/envs/vggt/bin/python demo_stage2_inference.py --batch_mode --start_idx 150 --end_idx 200 --step 5 --velocity_threshold 0.15 --clustering_eps 0.03"
echo "  /opt/miniconda/envs/vggt/bin/python demo_stage2_inference.py --idx 150  # single inference"