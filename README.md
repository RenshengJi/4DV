# Feedforward 4D Reconstruction for Driving Scenes

## Install

```bash
conda create -n 4dv python=3.10 -y
conda activate 4dv
pip install -r requirements.txt
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git
```

## Dataset

Dataset: https://huggingface.co/datasets/renshengjihe/waymo-flow-seg

```bash
python datasets_preprocess/extract_tars.py ../data/waymo/tar --num-workers 64
```

## Train

```bash
cd src
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py --config-path ../config/waymo --config-name stage1_online
```

## Inference

### 1. 完整推理（4x2网格布局）

生成包含GT/Pred对比的可视化：RGB、Depth、Velocity、Dynamic Clustering

```bash
CUDA_VISIBLE_DEVICES=0 /opt/miniconda/envs/vggt/bin/python inference.py \
    --batch_mode \
    --start_idx 100 --end_idx 100000 --step 100 \
    --model_path "src/checkpoints/waymo_stage1_online/fromaggregator_all_lr1e-5_procrustes_area500_velocityconstraint0.05_gtcamera_xyzgrad+fixdbscan+sky+fixepsmetric+noconf/checkpoint-epoch_0_37980.pth" \
    --seq_dir "data/waymo/train_full/" \
    --output_dir "./results/no_conf" \
    --device cuda --num_views 8 --fps 10 \
    --use_velocity_based_transform --velocity_transform_mode procrustes --use_gt_camera \
    --velocity_threshold 0.1 --clustering_eps 0.3 --clustering_min_samples 10 \
    --min_object_size 500 --tracking_position_threshold 2.0 --tracking_velocity_threshold 0.2 \
    --continue_on_error
```

**单帧推理**
```bash
/opt/miniconda/envs/vggt/bin/python inference.py --idx 1600
```

### 2. 新视角合成（3x3网格布局）

生成9个不同视角的RGB渲染结果

```bash
CUDA_VISIBLE_DEVICES=0 /opt/miniconda/envs/vggt/bin/python inference_nvs.py \
    --batch_mode \
    --start_idx 100 --end_idx 100000 --step 100 \
    --model_path "src/checkpoints/waymo_stage1_online/fromaggregator_all_lr1e-5_procrustes_area500_velocityconstraint0.05_gtcamera_xyzgrad+fixdbscan+sky+fixepsmetric+noconf/checkpoint-epoch_0_37980.pth" \
    --seq_dir "data/waymo/train_full/" \
    --output_dir "./results/nvs_no_Conf" \
    --device cuda --num_views 8 --fps 10 \
    --translation_offset 0.1 \
    --use_velocity_based_transform --velocity_transform_mode procrustes --use_gt_camera \
    --velocity_threshold 0.1 --clustering_eps 0.3 --clustering_min_samples 10 \
    --min_object_size 500 --tracking_position_threshold 2.0 --tracking_velocity_threshold 0.2 \
    --continue_on_error
```

### 主要参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--start_idx / --end_idx / --step` | 批量推理的帧范围和步长 | 100 / 100000 / 100 |
| `--idx` | 单帧推理的帧索引 | - |
| `--velocity_transform_mode` | 速度变换模式：`simple` 或 `procrustes` | procrustes |
| `--translation_offset` | NVS相机平移偏移量(米) | 0.1 |
| `--velocity_threshold` | 动态物体速度阈值 | 0.1 |
| `--clustering_eps` | DBSCAN聚类距离阈值(米) | 0.3 |
| `--min_object_size` | 最小物体点数 | 500 |
| `--tracking_position_threshold` | 跟踪位置阈值 | 2.0 |
