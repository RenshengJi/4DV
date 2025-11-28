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
    --sh_degree 0 --use_gs_head --use_gs_head_velocity \
    --use_velocity_based_transform --velocity_transform_mode procrustes --use_gt_camera \
    --velocity_threshold 0.1 --clustering_eps 0.3 --clustering_min_samples 10 \
    --min_object_size 500 --tracking_position_threshold 2.0 --tracking_velocity_threshold 0.2 \
    --continue_on_error
```

**单帧推理**
```bash
/opt/miniconda/envs/vggt/bin/python inference.py --idx 1600
```

### 2. Velocity可视化（单行布局）

生成速度预测可视化：GT RGB | GT Velocity | GT RGB + Pred Velocity 融合

```bash
CUDA_VISIBLE_DEVICES=0 /opt/miniconda/envs/vggt/bin/python inference_velocity.py \
    --batch_mode \
    --start_idx 100 --end_idx 100000 --step 100 \
    --model_path "src/checkpoints/waymo_stage1_online/fromaggregator_all_lr1e-5_procrustes_area500_velocityconstraint0.05_gtcamera_xyzgrad+fixdbscan+sky+fixepsmetric+noconf/checkpoint-epoch_0_37980.pth" \
    --seq_dir "data/waymo/train_full/" \
    --output_dir "./results/velocity_outputs" \
    --device cuda --num_views 8 --fps 10 \
    --sh_degree 0 --use_gs_head --use_gs_head_velocity --use_gt_camera \
    --velocity_alpha 0.5 --velocity_scale 0.1 \
    --continue_on_error
```

**单帧推理**
```bash
/opt/miniconda/envs/vggt/bin/python inference_velocity.py \
    --idx 1600 \
    --model_path "path/to/checkpoint.pth" \
    --seq_dir "data/waymo/train_full/" \
    --velocity_alpha 0.5
```

### 3. 新视角合成（3x3网格布局）

生成9个不同视角的RGB渲染结果

```bash
CUDA_VISIBLE_DEVICES=0 /opt/miniconda/envs/vggt/bin/python inference_nvs.py \
    --batch_mode \
    --start_idx 100 --end_idx 100000 --step 100 \
    --model_path "src/checkpoints/waymo_stage1_online/fromaggregator_all_lr1e-5_procrustes_area500_velocityconstraint0.05_gtcamera_xyzgrad+fixdbscan+sky+fixepsmetric+noconf/checkpoint-epoch_0_37980.pth" \
    --seq_dir "data/waymo/train_full/" \
    --output_dir "./results/nvs_no_Conf" \
    --device cuda --num_views 8 --fps 10 \
    --sh_degree 0 --use_gs_head --use_gs_head_velocity \
    --translation_offset 3 \
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
| `--sh_degree` | 球谐函数阶数 (0/1/2/3) | 0 |
| `--use_gs_head` | gaussian_head 使用 DPTGSHead | True |
| `--use_gs_head_velocity` | velocity_head 使用 DPTGSHead | False |
| `--velocity_transform_mode` | 速度变换模式：`simple` 或 `procrustes` | procrustes |
| `--velocity_alpha` | Pred velocity在融合中的权重 (0-1) | 0.5 |
| `--velocity_scale` | Velocity可视化缩放因子 | 0.1 |
| `--translation_offset` | NVS相机平移偏移量(米) | 0.1 |
| `--velocity_threshold` | 动态物体速度阈值 | 0.1 |
| `--clustering_eps` | DBSCAN聚类距离阈值(米) | 0.3 |
| `--min_object_size` | 最小物体点数 | 500 |
| `--tracking_position_threshold` | 跟踪位置阈值 | 2.0 |

**VGGT模型配置参数说明：**
- `--sh_degree`: 球谐函数阶数，控制颜色表示的复杂度
  - 0: 只有DC分量 (3个参数)
  - 1: DC + 方向性 (12个参数)
  - 2: 27个参数
  - 3: 48个参数
- `--use_gs_head`: 控制 gaussian_head 使用的网络架构
  - True: 使用 DPTGSHead (默认)
  - False: 使用 DPTHead
- `--use_gs_head_velocity`: 控制 velocity_head 使用的网络架构
  - True: 使用 DPTGSHead
  - False: 使用 DPTHead (默认)
