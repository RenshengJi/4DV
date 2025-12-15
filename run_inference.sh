#!/bin/bash
# 使用 Hydra 配置运行推理的示例脚本

# 批量推理模式 - 使用默认配置（config/waymo/infer.yaml）
CUDA_VISIBLE_DEVICES=2 /opt/miniconda/envs/vggt/bin/python inference.py

# 批量推理模式 - 自定义参数（命令行覆盖配置文件）
# CUDA_VISIBLE_DEVICES=2 /opt/miniconda/envs/vggt/bin/python inference.py \
#     batch_mode=true \
#     start_idx=1 \
#     end_idx=798 \
#     step=1 \
#     model_path="src/checkpoints/new/start/checkpoint-epoch_0_19533.pth" \
#     output_dir="./results/seperate" \
#     output_prefix="waymo_train"

# 单样本模式
# CUDA_VISIBLE_DEVICES=2 /opt/miniconda/envs/vggt/bin/python inference.py \
#     batch_mode=false \
#     single_idx=50

# 修改动态物体处理参数
# CUDA_VISIBLE_DEVICES=2 /opt/miniconda/envs/vggt/bin/python inference.py \
#     velocity_threshold=0.2 \
#     clustering_eps=0.5 \
#     min_object_size=1000

# 修改数据集配置（使用不同的分辨率或相机）
# CUDA_VISIBLE_DEVICES=2 /opt/miniconda/envs/vggt/bin/python inference.py \
#     'infer_dataset=Waymo_Multi(ROOT="data/waymo/val", valid_camera_id_list=["1"], intervals=[1], resolution=[(518, 336)], transform=ImgNorm, num_views=8, zero_ground_velocity=True)'

