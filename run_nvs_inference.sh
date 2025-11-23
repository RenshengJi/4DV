#!/bin/bash

# Novel View Synthesis (NVS) Inference Script (Batch Mode)
# 生成3x3视频布局，展示不同视角的渲染结果
#   - 中间(2,2): 原视角
#   - 上方(1,2): 视角向上移动
#   - 下方(3,2): 视角向下移动
#   - 左侧(2,1): 视角向左移动
#   - 右侧(2,3): 视角向右移动
#   - 对角线: 组合视角移动

# 配置参数（与run_inference.sh对齐）
START_IDX=${1:-100}                        # 起始idx，默认100
STEP=${2:-100}                             # 步长，默认100
END_IDX=${3:-100000}                       # 结束idx，默认100000
TRANSLATION_OFFSET=${4:-0.1}               # 平移偏移量(米)，默认3.0
USE_VELOCITY=${5:-true}                    # 是否使用velocity-based方法，默认true
VELOCITY_TRANSFORM_MODE=${6:-"procrustes"} # velocity变换模式: "simple"或"procrustes"
USE_GT_CAMERA=${7:-true}                   # 是否使用GT camera参数，默认true

# 动态物体聚类和跟踪参数
VELOCITY_THRESHOLD=${8:-0.1}               # 速度阈值，默认0.1
CLUSTERING_EPS=${9:-0.3}                   # DBSCAN聚类eps参数，默认0.3 (米)
CLUSTERING_MIN_SAMPLES=${10:-10}           # DBSCAN聚类min_samples参数，默认10
MIN_OBJECT_SIZE=${11:-500}                 # 最小物体尺寸（点数），默认500
TRACKING_POSITION_THRESHOLD=${12:-2.0}     # 跟踪位置阈值，默认2.0
TRACKING_VELOCITY_THRESHOLD=${13:-0.2}     # 跟踪速度阈值，默认0.2

echo "Starting Novel View Synthesis Inference (Batch Mode)..."
echo "Parameters:"
echo "  Start IDX: ${START_IDX}"
echo "  End IDX: ${END_IDX}"
echo "  Step: ${STEP}"
echo "  Translation Offset: ${TRANSLATION_OFFSET}m"
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

# 模型路径 (使用与inference.py相同的模型)
STAGE1_MODEL_PATH="src/checkpoints/waymo_new/segmetation/checkpoint-epoch_0_55335.pth"

# 数据路径
SEQ_DIR="data/waymo/train_full/"

# 输出目录
OUTPUT_DIR="./results/nvs_5.5w"

# 其他参数
NUM_VIEWS=8
DEVICE="cuda"
FPS=10

# 聚类参数
VELOCITY_THRESHOLD=0.1
CLUSTERING_EPS=0.3
CLUSTERING_MIN_SAMPLES=10
MIN_OBJECT_SIZE=500
TRACKING_POSITION_THRESHOLD=2.0
TRACKING_VELOCITY_THRESHOLD=0.2

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
PYTHON_CMD="/opt/miniconda/envs/vggt/bin/python inference_nvs.py \
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
    --translation_offset ${TRANSLATION_OFFSET} \
    ${VELOCITY_ARGS} \
    --velocity_threshold ${VELOCITY_THRESHOLD} \
    --clustering_eps ${CLUSTERING_EPS} \
    --clustering_min_samples ${CLUSTERING_MIN_SAMPLES} \
    --min_object_size ${MIN_OBJECT_SIZE} \
    --tracking_position_threshold ${TRACKING_POSITION_THRESHOLD} \
    --tracking_velocity_threshold ${TRACKING_VELOCITY_THRESHOLD} \
    --continue_on_error"

echo "Generating 3x3 Novel View Synthesis video"
echo "  Layout:"
echo "    [Up-Left]  [Up]      [Up-Right]"
echo "    [Left]     [Center]  [Right]"
echo "    [Down-Left][Down]    [Down-Right]"
echo ""

echo "Running inference..."
echo "Command: ${PYTHON_CMD}"
echo ""

# 执行Python推理
eval $PYTHON_CMD

echo ""
echo "NVS inference script completed!"
echo ""
echo "Features:"
echo "  ✓ 3x3 grid layout for novel view synthesis"
echo "  ✓ Camera translation in world coordinates"
echo "  ✓ RGB-only output (no depth)"
echo "  ✓ Dynamic objects rendered with transformation"
echo "  ✓ Sky color composition"
echo ""
echo "Usage examples:"
echo "  ./run_nvs_inference.sh                              # 默认: start=100, step=100, end=100000, offset=3.0m"
echo "  ./run_nvs_inference.sh 100 100 100000              # start=100, step=100, end=100000, offset=3.0m (默认)"
echo "  ./run_nvs_inference.sh 100 100 100000 3.0          # 明确指定offset=3.0m"
echo "  ./run_nvs_inference.sh 0 5 50 2.0                  # start=0, step=5, end=50, offset=2.0m"
echo "  ./run_nvs_inference.sh 100 10 200 1.5 true simple  # with velocity transform mode"
echo ""
echo "Full parameter list:"
echo "  \$1: START_IDX (default: 100)"
echo "  \$2: STEP (default: 100)"
echo "  \$3: END_IDX (default: 100000)"
echo "  \$4: TRANSLATION_OFFSET (default: 3.0m)"
echo "  \$5: USE_VELOCITY (default: true)"
echo "  \$6: VELOCITY_TRANSFORM_MODE (default: procrustes)"
echo "  \$7: USE_GT_CAMERA (default: true)"
echo "  \$8: VELOCITY_THRESHOLD (default: 0.1)"
echo "  \$9: CLUSTERING_EPS (default: 0.3)"
echo "  \$10: CLUSTERING_MIN_SAMPLES (default: 10)"
echo "  \$11: MIN_OBJECT_SIZE (default: 500)"
echo "  \$12: TRACKING_POSITION_THRESHOLD (default: 2.0)"
echo "  \$13: TRACKING_VELOCITY_THRESHOLD (default: 0.2)"
echo ""
echo "Output videos saved to: ${OUTPUT_DIR}"


