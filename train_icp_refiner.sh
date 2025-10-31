#!/bin/bash
#
# ICP Refiner 训练脚本
# 使用ICP监督数据训练Gaussian Refine网络
#

set -e

echo "========================================="
echo "ICP Refiner Training"
echo "========================================="
echo ""

# 配置
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src:$PYTHONPATH

# 切换到项目目录
cd /mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo

# 检查数据目录
DATA_DIR="./icp_supervision_data_real"
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR does not exist!"
    exit 1
fi

# 统计样本数量
NUM_SAMPLES=$(ls -1 $DATA_DIR/*.npz 2>/dev/null | wc -l)
echo "Found $NUM_SAMPLES training samples in $DATA_DIR"
echo ""

if [ $NUM_SAMPLES -eq 0 ]; then
    echo "Error: No .npz files found in $DATA_DIR"
    echo "Please run collect_icp_data.sh first to collect training data"
    exit 1
fi

# 运行训练
echo "Starting training..."
echo "Config: src/icp_supervision/config/icp_train.yaml"
echo ""

/opt/miniconda/envs/vggt/bin/python src/icp_supervision/train.py \
    --config src/icp_supervision/config/icp_train.yaml

echo ""
echo "✓ Training complete!"
echo ""
