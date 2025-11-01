#!/bin/bash

# Stage2推理运行脚本 - 增强版本，输出RGB比较和动态多帧匹配聚类
# 输出格式：四列比較 GT | Initial | Refined | Dynamic Clustering

# 配置参数
START_IDX=${1:-150}     # 起始idx，默认150
STEP=${2:-5}            # 步长，默认5
END_IDX=${3:-200}       # 结束idx，默认200
USE_VELOCITY=${4:-false}  # 是否使用velocity-based方法，默认false（使用光流）
VELOCITY_TRANSFORM_MODE=${5:-"simple"}  # velocity变换模式: "simple"或"procrustes"，默认"simple"
USE_GT_CAMERA=${6:-false}  # 是否使用GT camera参数，默认false（使用预测的）

# 动态物体聚类和跟踪参数
VELOCITY_THRESHOLD=${7:-0.1}              # 速度阈值，默认0.1
CLUSTERING_EPS=${8:-0.3}                  # DBSCAN聚类eps参数，默认0.3 (米)
CLUSTERING_MIN_SAMPLES=${9:-10}           # DBSCAN聚类min_samples参数，默认10
MIN_OBJECT_SIZE=${10:-500}                # 最小物体尺寸（点数），默认100
TRACKING_POSITION_THRESHOLD=${11:-2.0}    # 跟踪位置阈值，默认2.0
TRACKING_VELOCITY_THRESHOLD=${12:-0.2}    # 跟踪速度阈值，默认0.2

echo "Starting Stage2 comparison inference (optimized Python version)..."
echo "Parameters:"
echo "  Start IDX: ${START_IDX}"
echo "  End IDX: ${END_IDX}"
echo "  Step: ${STEP}"
echo "  Transformation Method: $([ "$USE_VELOCITY" = "true" ] && echo "Velocity-based (${VELOCITY_TRANSFORM_MODE})" || echo "Flow-based")"
echo "  Camera Parameters: $([ "$USE_GT_CAMERA" = "true" ] && echo "Ground Truth" || echo "Predicted")"
echo "  Mode: Enhanced comparison with dynamic clustering"
echo "  Output: 4-column layout (GT | Initial | Refined | Dynamic Clustering)"
echo ""
echo "Clustering & Tracking Parameters:"
echo "  Velocity Threshold: ${VELOCITY_THRESHOLD} m/s"
echo "  Clustering EPS: ${CLUSTERING_EPS} m"
echo "  Clustering Min Samples: ${CLUSTERING_MIN_SAMPLES}"
echo "  Min Object Size: ${MIN_OBJECT_SIZE} points"
echo "  Tracking Position Threshold: ${TRACKING_POSITION_THRESHOLD} m"
echo "  Tracking Velocity Threshold: ${TRACKING_VELOCITY_THRESHOLD} m/s"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/stage1_gtflow/checkpoint-epoch_7_9968.pth"
# STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/aggregator_resume_noflowgrad_nearestdynamic_resume_0point1_novoxel/checkpoint-epoch_0_19936.pth"
# STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/aggregator_from_scratch/checkpoint-epoch_0_2848.pth"
# STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/aggregator_all_resume_procrustes_depthconf0.2+fixcamera+velocityconstrain_predcamera+nodetach/checkpoint-epoch_0_5696.pth"
# STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/aggregator_all_resume_procrustes_depthconf0.2+fixcamera+velocityconstrain_detach/checkpoint-epoch_0_32550.pth"
# STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/fix_velocity/checkpoint-epoch_0_9765.pth"
# STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/fix_velocity_fix_padding/checkpoint-epoch_0_8544.pth"
STAGE1_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage1_online/fromaggregator_all_lr1e-5_procrustes_area500_velocityconstraint0.05_gtcamera_xyzgrad+fixdbscan+sky+lpips/checkpoint-epoch_0_3255.pth"

STAGE2_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage2_online/stage2_train+5e-4+biggermodel+onlydynamic+poseinput+gaussian_only+flow+10layers+onlymiddle/stage2-checkpoint-epoch_0_23329.pth"
# STAGE2_MODEL_PATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo_stage2_online/stage2_train+5e-4+biggermodel+onlydynamic+poseinput+joint+flow+10layers/stage2-checkpoint-epoch_0_2604.pth"

# 数据路径
# SEQ_DIR="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/test/segment-11717495969710734380_2440_000_2460_000_with_camera_labels"
SEQ_DIR="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train_with_flow/segment-15795616688853411272_1245_000_1265_000_with_camera_labels"



# 输出目录
OUTPUT_DIR="./stage2_inference_outputs"

# 构建Python批量推理命令 - 现在总是生成比较视频
# 根据USE_VELOCITY参数添加--use_velocity_based_transform标志
if [ "$USE_VELOCITY" = "true" ]; then
    VELOCITY_FLAG="--use_velocity_based_transform --velocity_transform_mode ${VELOCITY_TRANSFORM_MODE}"
else
    VELOCITY_FLAG=""
fi

# 根据USE_GT_CAMERA参数添加--use_gt_camera标志
if [ "$USE_GT_CAMERA" = "true" ]; then
    CAMERA_FLAG="--use_gt_camera"
else
    CAMERA_FLAG=""
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
    --min_object_size ${MIN_OBJECT_SIZE} \
    --tracking_position_threshold ${TRACKING_POSITION_THRESHOLD} \
    --tracking_velocity_threshold ${TRACKING_VELOCITY_THRESHOLD} \
    ${VELOCITY_FLAG} \
    ${CAMERA_FLAG}"

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
echo "  ./run_stage2_inference.sh                                          # 默认参数"
echo "  ./run_stage2_inference.sh 100 10 150                              # idx 100-150, step 10"
echo "  ./run_stage2_inference.sh 100 10 150 true                         # Velocity-based (simple mode)"
echo "  ./run_stage2_inference.sh 100 10 150 true procrustes              # Velocity-based (procrustes mode)"
echo "  ./run_stage2_inference.sh 100 10 150 true procrustes true         # Velocity-based + GT camera"
echo "  ./run_stage2_inference.sh 100 10 150 false simple false           # Flow-based + predicted camera"
echo "  ./run_stage2_inference.sh 100 10 150 false simple true 0.15       # Flow-based + GT camera, velocity_threshold=0.15"
echo "  ./run_stage2_inference.sh 100 10 150 false simple false 0.1 0.03  # 自定义clustering_eps=0.03"
echo "  ./run_stage2_inference.sh 100 10 150 false simple false 0.1 0.02 15     # 自定义min_samples=15"
echo "  ./run_stage2_inference.sh 100 10 150 false simple false 0.1 0.3 10 100  # 自定义min_object_size=100"
echo "  ./run_stage2_inference.sh 100 10 150 false simple false 0.1 0.3 10 500 3.0  # 自定义tracking_position=3.0"
echo "  ./run_stage2_inference.sh 100 10 150 false simple false 0.1 0.3 10 100 2.0 0.3  # 完整自定义"
echo ""
echo "Parameters:"
echo "  \$1: START_IDX (default: 150)"
echo "  \$2: STEP (default: 5)"
echo "  \$3: END_IDX (default: 200)"
echo "  \$4: USE_VELOCITY - true=Velocity-based, false=Flow-based (default: false)"
echo "  \$5: VELOCITY_TRANSFORM_MODE - \"simple\" or \"procrustes\" (default: \"simple\")"
echo "  \$6: USE_GT_CAMERA - true=GT camera, false=Predicted camera (default: false)"
echo "  \$7: VELOCITY_THRESHOLD in m/s (default: 0.1)"
echo "  \$8: CLUSTERING_EPS in meters (default: 0.3)"
echo "  \$9: CLUSTERING_MIN_SAMPLES (default: 10)"
echo "  \$10: MIN_OBJECT_SIZE in points (default: 100)"
echo "  \$11: TRACKING_POSITION_THRESHOLD in meters (default: 2.0)"
echo "  \$12: TRACKING_VELOCITY_THRESHOLD in m/s (default: 0.2)"
echo ""
echo "Velocity Transform Modes:"
echo "  simple:     Fast, translation only (R=I, T=mean_velocity)"
echo "  procrustes: Accurate, estimates R and T using xyz+velocity+Procrustes"
echo ""
echo "Camera Parameters:"
echo "  GT camera:        Use ground truth camera from dataset (eliminates camera prediction errors)"
echo "  Predicted camera: Use camera predicted by Stage1 model (default)"
echo ""
echo "Or use Python directly:"
echo "  /opt/miniconda/envs/vggt/bin/python demo_stage2_inference.py --batch_mode --start_idx 150 --end_idx 200 --step 5"
echo "  /opt/miniconda/envs/vggt/bin/python demo_stage2_inference.py --batch_mode --start_idx 150 --end_idx 200 --step 5 --use_velocity_based_transform"
echo "  /opt/miniconda/envs/vggt/bin/python demo_stage2_inference.py --batch_mode --start_idx 150 --end_idx 200 --step 5 --use_velocity_based_transform --velocity_transform_mode procrustes"
echo "  /opt/miniconda/envs/vggt/bin/python demo_stage2_inference.py --batch_mode --start_idx 150 --end_idx 200 --step 5 --use_gt_camera"
echo "  /opt/miniconda/envs/vggt/bin/python demo_stage2_inference.py --batch_mode --start_idx 150 --end_idx 200 --step 5 --min_object_size 500"
echo "  /opt/miniconda/envs/vggt/bin/python demo_stage2_inference.py --batch_mode --start_idx 150 --end_idx 200 --step 5 --velocity_threshold 0.15 --clustering_eps 0.5"
echo "  /opt/miniconda/envs/vggt/bin/python demo_stage2_inference.py --idx 150  # single inference"