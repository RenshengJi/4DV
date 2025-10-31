#!/bin/bash
#
# ICP Refiner 推理脚本 - 快速启动
#

set -e

# 默认参数
CHECKPOINT="./icp_train_output_largelr/checkpoints/epoch_0099.pth"
# SAMPLE="./icp_supervision_data_real/object_000000_sample_000008.npz"
SAMPLE="./icp_supervision_data_real/object_000000_sample_000033.npz"
OUTPUT_DIR="./inference_output"
DEVICE="cuda:0"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --sample)
            SAMPLE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --visualize)
            VISUALIZE="--visualize"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "ICP Refiner Inference"
echo "========================================="
echo ""
echo "Checkpoint: $CHECKPOINT"
echo "Sample:     $SAMPLE"
echo "Output:     $OUTPUT_DIR"
echo "Device:     $DEVICE"
echo ""

# 检查文件是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    echo "Please train the model first or specify a valid checkpoint path"
    exit 1
fi

if [ ! -f "$SAMPLE" ]; then
    echo "Error: Sample not found: $SAMPLE"
    exit 1
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src:$PYTHONPATH

# 运行推理
echo "Starting inference..."
echo ""

/opt/miniconda/envs/vggt/bin/python src/icp_supervision/infer.py \
    --checkpoint "$CHECKPOINT" \
    --sample "$SAMPLE" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    $VISUALIZE

echo ""
echo "✓ Inference complete!"
echo ""
echo "Output files saved to: $OUTPUT_DIR"
echo ""
echo "You can view the .ply files using:"
echo "  - CloudCompare"
echo "  - MeshLab"
echo "  - Open3D viewer"
echo ""
