# ICP Supervision Training Guide

## 概述

使用 ICP 配准生成的 GT 数据训练 Gaussian Refine 网络，用于 Stage2 的高斯参数细化。

## 数据收集

### 1. 收集 ICP 训练数据

```bash
bash collect_icp_data.sh
```

这会：
- 运行 Stage2 推理提取动态物体
- 对每个物体的多帧点云进行 ICP 配准
- 生成 `.npz` 样本对文件保存到 `icp_supervision_data_real/`

每个样本包含：
- `input_gaussians`: [N, 14] 粗糙的 Gaussian 参数（来自光流聚合）
- `target_gaussians`: [N, 14] ICP 配准后的精细 Gaussian 参数（GT）
- `pred_scale`: 用于尺度归一化的 scale 因子

### 2. 配置参数

配置文件位于 `config/waymo/stage2_icp_collect.yaml`，可以调整：

```yaml
# ICP配准参数
use_color_features: true    # 是否启用颜色特征（SIFT）
voxel_size: 0.01           # 体素大小（影响下采样精度）
ransac_max_iteration: 5000 # RANSAC 最大迭代次数
ransac_confidence: 5       # RANSAC 置信度参数

# 数据收集参数
icp_max_batches: 10000     # 最多处理多少个样本
```

### 3. 查看收集的数据

```bash
# 统计样本数量
ls -1 icp_supervision_data_real/*.npz | wc -l

# 查看样本大小
du -sh icp_supervision_data_real/
```

## 训练

### 快速开始

```bash
# 使用默认配置启动训练
bash train_icp_refiner.sh
```

### 手动启动

```bash
cd /mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo

export CUDA_VISIBLE_DEVICES=0

python src/icp_supervision/train.py \
    --config src/icp_supervision/config/icp_train.yaml
```

### 训练配置

配置文件位于 `src/icp_supervision/config/icp_train.yaml`：

```yaml
# 数据配置
data_dir: './icp_supervision_data_real'
train_ratio: 0.8                    # 训练/验证集划分比例
batch_size: 4                       # 批次大小

# 模型配置
gaussian_feature_dim: 384           # 特征维度
gaussian_num_conv_layers: 10        # 卷积层数
gaussian_voxel_size: 0.05           # 体素大小（米）

# 损失函数权重
position_weight: 10.0               # 位置损失权重（最重要）
scale_weight: 1.0                   # 尺度损失权重
rotation_weight: 1.0                # 旋转损失权重
color_weight: 1.0                   # 颜色损失权重
opacity_weight: 1.0                 # 不透明度损失权重

# 训练参数
epochs: 100                         # 训练轮数
lr: 5e-4                           # 学习率
optimizer: 'adamw'                  # 优化器
scheduler: 'cosine'                 # 学习率调度器
```

### 从断点恢复训练

```bash
python src/icp_supervision/train.py \
    --config src/icp_supervision/config/icp_train.yaml \
    --resume ./icp_train_output/checkpoints/latest.pth
```

## 输出

### 训练输出目录结构

```
icp_train_output/
├── checkpoints/
│   ├── best.pth           # 最佳模型（验证集损失最低）
│   ├── latest.pth         # 最新模型
│   └── epoch_*.pth        # 定期保存的模型
└── logs/                  # TensorBoard 日志
```

### 查看训练日志

```bash
# 启动 TensorBoard
tensorboard --logdir=./icp_train_output/logs --port=6006

# 在浏览器中访问
# http://localhost:6006
```

## 数据流程

### 完整流程

1. **Stage1 推理** → 粗糙的动态物体 Gaussians
2. **ICP 配准** → 精细的 GT Gaussians
3. **训练 Refiner** → 学习从粗糙到精细的映射
4. **集成到 Stage2** → 使用训练好的 Refiner 细化 Gaussians

### 尺度转换

由于 Gaussians 在训练中使用归一化的非 metric 尺度，ICP 流程包含尺度转换：

1. **输入**: Gaussians (归一化尺度) + `pred_scale`
2. **转换到 metric**: `xyz_metric = xyz_normalized / pred_scale`
3. **ICP 配准**: 在真实物理尺度下进行
4. **转换回归一化**: `xyz_normalized = xyz_metric * pred_scale`
5. **保存**: Gaussians (归一化尺度)

这确保 ICP 在真实物理空间中进行，配准结果更准确。

## 常见问题

### Q: 样本数量不够怎么办？

增加 `icp_max_batches` 参数：
```yaml
icp_max_batches: 20000  # 收集更多样本
```

### Q: 训练速度太慢？

1. 增加 `batch_size`（如果 GPU 内存允许）
2. 减少 `gaussian_num_conv_layers`
3. 减少 `gaussian_feature_dim`
4. 启用 `cache_in_memory: true`（如果内存足够）

### Q: 验证损失不下降？

1. 检查数据质量：查看 ICP 配准是否成功
2. 调整学习率：尝试更小的 `lr` (e.g., 1e-4)
3. 调整损失权重：增加 `position_weight`
4. 增加训练样本数量

### Q: 如何可视化结果？

使用提供的可视化脚本：
```bash
python src/icp_supervision/visualize.py \
    --checkpoint ./icp_train_output/checkpoints/best.pth \
    --data_dir ./icp_supervision_data_real \
    --output_dir ./visualization
```

## 修复记录

### 2024-10-26: 移除空洞卷积参数

**问题**: 训练报错 `GaussianRefineHeadSparseConv.__init__() got an unexpected keyword argument 'use_dilated_conv'`

**原因**: 网络已经移除了空洞卷积相关功能，但配置文件仍包含这些参数

**修复**:
- 从配置文件删除：`use_dilated_conv`, `gaussian_dilation_rates`, `max_num_points_per_voxel`
- 更新训练脚本的 `_create_model` 方法

**修改文件**:
- `src/icp_supervision/config/icp_train.yaml`
- `src/icp_supervision/train.py`
