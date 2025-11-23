#!/bin/bash

# 完整的inference推理运行脚本（参考run_stage2_inference.sh）
# 输出格式：4x2网格布局
#   Row 1: GT RGB | Rendered RGB (with sky)
#   Row 2: GT Depth | Rendered Depth
#   Row 3: GT Velocity | Pred Velocity
#   Row 4: Dynamic Clustering | (Black)

# 配置参数
START_IDX=${1:-100}     # 起始idx，默认150
STEP=${2:-100}            # 步长，默认5
END_IDX=${3:-100000}       # 结束idx，默认200
USE_VELOCITY=${4:-true}  # 是否使用velocity-based方法，默认true
VELOCITY_TRANSFORM_MODE=${5:-"procrustes"}  # velocity变换模式: "simple"或"procrustes"
USE_GT_CAMERA=${6:-true}  # 是否使用GT camera参数，默认true

# 动态物体聚类和跟踪参数
VELOCITY_THRESHOLD=${7:-0.1}              # 速度阈值，默认0.1
CLUSTERING_EPS=${8:-0.3}                  # DBSCAN聚类eps参数，默认0.3 (米)
CLUSTERING_MIN_SAMPLES=${9:-10}           # DBSCAN聚类min_samples参数，默认10
MIN_OBJECT_SIZE=${10:-500}                # 最小物体尺寸（点数），默认500
TRACKING_POSITION_THRESHOLD=${11:-2.0}    # 跟踪位置阈值，默认2.0
TRACKING_VELOCITY_THRESHOLD=${12:-0.2}    # 跟踪速度阈值，默认0.2

echo "Starting Complete Inference (Batch Mode)..."
echo "Parameters:"
echo "  Start IDX: ${START_IDX}"
echo "  End IDX: ${END_IDX}"
echo "  Step: ${STEP}"
echo "  Use Velocity Transform: ${USE_VELOCITY}"
echo "  Velocity Transform Mode: ${VELOCITY_TRANSFORM_MODE}"
echo "  Use GT Camera: ${USE_GT_CAMERA}"
echo "  Velocity Threshold: ${VELOCITY_THRESHOLD}"
echo "  Clustering EPS: ${CLUSTERING_EPS}"
echo "  Clustering Min Samples: ${CLUSTERING_MIN_SAMPLES}"
echo "  Min Object Size: ${MIN_OBJECT_SIZE}"
echo "  Tracking Position Threshold: ${TRACKING_POSITION_THRESHOLD}"
echo "  Tracking Velocity Threshold: ${TRACKING_VELOCITY_THRESHOLD}"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 模型路径
# STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/fromaggregator_all_lr1e-5_procrustes_area500_velocityconstraint0.05_gtcamera_xyzgrad+fixdbscan+sky+fixepsmetric/checkpoint-epoch_0_22785.pth"
# STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/fromaggregator_all_lr1e-5_procrustes_area500_velocityconstraint0.05_gtcamera_xyzgrad+fixdbscan+sky+fixepsmetric+noconf+novelocity!/checkpoint-epoch_0_75960.pth"
# STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/stage1_gtflow/checkpoint-epoch_7_11392.pth"
# STAGE1_MODEL_PATH="src/checkpoints/waymo_stage1_online/aggregator_all_resume_procrustes_depthconf0.2+fixcamera+velocityconstrain_detach/checkpoint-epoch_0_39060.pth"
STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_new/segmetation/checkpoint-epoch_0_58590.pth"
    

# 数据路径
SEQ_DIR="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train_full/"

# 输出目录
OUTPUT_DIR="./results/seg_5.5w"

# 其他参数
NUM_VIEWS=8
DEVICE="cuda"
FPS=10

# 构建velocity参数
VELOCITY_ARGS=""
if [ "$USE_VELOCITY" = "true" ]; then
    VELOCITY_ARGS="$VELOCITY_ARGS --use_velocity_based_transform"
fi
VELOCITY_ARGS="$VELOCITY_ARGS --velocity_transform_mode ${VELOCITY_TRANSFORM_MODE}"

if [ "$USE_GT_CAMERA" = "true" ]; then
    VELOCITY_ARGS="$VELOCITY_ARGS --use_gt_camera"
fi

# 构建Python批量推理命令
PYTHON_CMD="/opt/miniconda/envs/vggt/bin/python inference.py \
    --batch_mode \
    --start_idx ${START_IDX} \
    --end_idx ${END_IDX} \
    --step ${STEP} \
    --model_path \"${STAGE1_MODEL_PATH}\" \
    --seq_dir \"${SEQ_DIR}\" \
    --output_dir \"${OUTPUT_DIR}\" \
    --device ${DEVICE} \
    --num_views ${NUM_VIEWS} \
    --fps ${FPS} \
    ${VELOCITY_ARGS} \
    --velocity_threshold ${VELOCITY_THRESHOLD} \
    --clustering_eps ${CLUSTERING_EPS} \
    --clustering_min_samples ${CLUSTERING_MIN_SAMPLES} \
    --min_object_size ${MIN_OBJECT_SIZE} \
    --tracking_position_threshold ${TRACKING_POSITION_THRESHOLD} \
    --tracking_velocity_threshold ${TRACKING_VELOCITY_THRESHOLD} \
    --continue_on_error"

echo "Generating complete visualization videos"
echo "  Row 1: GT RGB | Rendered RGB (with sky color composition)"
echo "  Row 2: GT Depth | Rendered Depth (viridis colormap)"
echo "  Row 3: GT Velocity | Pred Velocity (scene_flow_to_rgb)"
echo "  Row 4: Dynamic Clustering (HSV colors per cluster) | (Black)"
echo ""

echo "Running batch inference..."
echo "Command: ${PYTHON_CMD}"
echo ""

# 执行Python推理
eval $PYTHON_CMD

echo ""
echo "Batch inference script completed!"
echo ""
echo "Features of this inference:"
echo "  ✓ 4x2 grid layout for comprehensive visualization"
echo "  ✓ GT/Pred side-by-side comparison"
echo "  ✓ Aggregator_all rendering with sky color composition"
echo "  ✓ Dynamic objects rendered (without Stage2 refinement)"
echo "  ✓ Dynamic clustering visualization (Row 4 left column)"
echo "  ✓ Pure Stage1 output"
echo "  ✓ Configurable clustering parameters"
echo "  ✓ Batch processing with error recovery"
echo ""
echo "Usage examples:"
echo "  ./run_inference.sh                                    # 默认: idx 150-200, step 5"
echo "  ./run_inference.sh 100 10 150                        # idx 100-150, step 10"
echo "  ./run_inference.sh 0 1 10 true simple false         # with velocity transform"
echo "  ./run_inference.sh 150 5 200 false simple true      # with GT camera"
echo ""
echo "Full parameter list:"
echo "  \$1: START_IDX (default: 150)"
echo "  \$2: STEP (default: 5)"
echo "  \$3: END_IDX (default: 200)"
echo "  \$4: USE_VELOCITY (default: false)"
echo "  \$5: VELOCITY_TRANSFORM_MODE (default: simple)"
echo "  \$6: USE_GT_CAMERA (default: false)"
echo "  \$7: VELOCITY_THRESHOLD (default: 0.1)"
echo "  \$8: CLUSTERING_EPS (default: 0.3)"
echo "  \$9: CLUSTERING_MIN_SAMPLES (default: 10)"
echo "  \$10: MIN_OBJECT_SIZE (default: 500)"
echo "  \$11: TRACKING_POSITION_THRESHOLD (default: 2.0)"
echo "  \$12: TRACKING_VELOCITY_THRESHOLD (default: 0.2)"
echo ""
echo "Or use Python directly:"
echo "  /opt/miniconda/envs/vggt/bin/python inference.py --batch_mode --start_idx 150 --end_idx 200"
echo "  /opt/miniconda/envs/vggt/bin/python inference.py --idx 1600  # single inference"
echo ""
echo "Output videos will be saved to: ${OUTPUT_DIR}"
