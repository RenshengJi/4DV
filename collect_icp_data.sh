#!/bin/bash
#
# 独立的ICP数据收集脚本 - 逐样本处理版本
# 不修改任何现有代码，直接运行Stage2推理收集dynamic_objects
#

set -e

echo "========================================="
echo "ICP Data Collection - 逐样本处理"
echo "========================================="
echo ""

# 配置
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src:$PYTHONPATH

# 运行数据收集
cd /mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo

echo "Using config: config/waymo/stage2_icp_collect.yaml"
echo "Processing samples one by one..."
echo ""

/opt/miniconda/envs/vggt/bin/python src/icp_supervision/collect_stage2_data_simple.py \
    --config-path ../../config/waymo \
    --config-name stage2_icp_collect

echo ""
echo "✓ Done!"
echo ""
echo "Output:"
echo "  - ICP training data: ./icp_supervision_data_real/"
echo ""
