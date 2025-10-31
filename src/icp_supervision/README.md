# ICP Supervision Module for Gaussian Refinement

## 概述

本模块提供了基于 **ICP (Iterative Closest Point)** 配准的强监督方式，用于训练 Gaussian Refinement 网络。通过离线生成 ICP 配准的 GT (Ground Truth)，我们可以直接监督网络学习如何精细化 Gaussian 参数。

## 核心特点

- ✅ **强监督**: 使用 ICP 配准结果作为 GT，提供点对点的强监督信号
- ✅ **离线生成**: 预先构建样本对，避免在线 ICP 的速度问题
- ✅ **一一对应**: Gaussians 参数一一对应，实现精确的参数级别监督
- ✅ **独立训练**: 完全独立的训练流程，不修改原有 Stage2 代码
- ✅ **可视化验证**: 提供点云可视化工具，验证 ICP GT 的正确性
- ✅ **灵活配置**: 支持多种损失函数组合和训练策略

## 目录结构

```
src/icp_supervision/
├── __init__.py              # 模块初始化
├── data_generator.py        # 离线数据生成：从 Stage2 数据构建 ICP GT 样本对
├── dataset.py               # PyTorch Dataset 和 DataLoader
├── icp_loss.py              # ICP 监督损失函数
├── train.py                 # 独立训练脚本
├── visualize.py             # 可视化工具
├── utils.py                 # 工具函数
├── config/
│   └── icp_train.yaml       # 训练配置文件
└── README.md                # 本文档
```

## 使用流程

### 步骤 1: 生成 ICP GT 样本对

首先，需要从 Stage2 训练数据中提取 dynamic objects 并运行 ICP 配准生成 GT。

```bash
cd src

# 从 Stage2 训练过程中保存的 dynamic_objects 数据生成 ICP GT
python -m icp_supervision.data_generator \
    --input /path/to/stage2_dynamic_objects.pkl \
    --output_dir ./icp_supervision_data \
    --voxel_size 0.01 \
    --max_icp_iterations 50 \
    --min_frames 2 \
    --max_frames 10 \
    --save_pointclouds  # 可选：保存点云用于可视化验证
```

**参数说明:**
- `--input`: Stage2 保存的包含 `dynamic_objects` 的 pickle 文件
- `--output_dir`: 输出目录，存储生成的 `.npz` 样本对
- `--voxel_size`: ICP 配准的体素大小
- `--max_icp_iterations`: ICP 最大迭代次数
- `--min_frames`: 每个物体最少帧数（少于此数跳过）
- `--max_frames`: 每个物体最多使用的帧数（避免过多帧导致 ICP 速度慢）
- `--save_pointclouds`: 是否保存点云文件（.ply 格式）用于可视化验证
- `--use_color_features`: 是否在 ICP 中使用颜色特征

**输出:**
- `*.npz` 文件：每个物体一个样本对，包含:
  - `input_gaussians`: [N, 14] 粗糙 Gaussian 参数
  - `target_gaussians`: [N, 14] ICP 配准后的 GT 参数
  - `pred_scale`: float, 用于体素化
  - `object_id`: int, 物体 ID
  - (可选) `input_pcd_path`, `target_pcd_path`: 点云文件路径

### 步骤 2: 验证 ICP GT 正确性 (可选但推荐)

使用可视化工具检查生成的 ICP GT 是否正确。

```bash
# 查看数据集统计信息
python -m icp_supervision.visualize \
    --data_dir ./icp_supervision_data \
    --mode stats

# 可视化前 5 个样本的对比视图（红色=input，绿色=target）
python -m icp_supervision.visualize \
    --data_dir ./icp_supervision_data \
    --mode comparison \
    --num_samples 5

# 可视化特定样本
python -m icp_supervision.visualize \
    --data_dir ./icp_supervision_data \
    --sample_idx 0 \
    --mode all

# 导出点云为 .ply 文件，用于 MeshLab 等工具查看
python -m icp_supervision.visualize \
    --data_dir ./icp_supervision_data \
    --sample_idx 0 \
    --export \
    --export_dir ./exported_pointclouds
```

**可视化模式:**
- `stats`: 计算并打印数据集统计信息
- `input`: 仅显示输入点云
- `target`: 仅显示 ICP GT 点云
- `comparison`: 对比视图（红色=input，绿色=target）
- `all`: 显示所有视图

### 步骤 3: 训练 Gaussian Refine 网络

使用生成的 ICP GT 样本对训练网络。

```bash
# 使用默认配置训练
python -m icp_supervision.train \
    --config icp_supervision/config/icp_train.yaml \
    --data_dir ./icp_supervision_data \
    --output_dir ./icp_train_output

# 从 checkpoint 继续训练
python -m icp_supervision.train \
    --config icp_supervision/config/icp_train.yaml \
    --resume ./icp_train_output/checkpoints/latest.pth
```

**配置文件** (`config/icp_train.yaml`) **重要参数:**

```yaml
# 数据
data_dir: './icp_supervision_data'
train_ratio: 0.8

# 模型 (与 Stage2 保持一致)
gaussian_feature_dim: 384
gaussian_num_conv_layers: 10
gaussian_voxel_size: 0.05

# 损失权重 (position 最重要)
position_weight: 10.0
scale_weight: 1.0
rotation_weight: 1.0
color_weight: 1.0
opacity_weight: 1.0

position_only: false  # 设为 true 则只监督位置

# 训练
epochs: 100
batch_size: 4
lr: 5e-4
optimizer: 'adamw'
scheduler: 'cosine'
```

**训练输出:**
- Checkpoints: `output_dir/checkpoints/`
  - `latest.pth`: 最新 checkpoint
  - `best.pth`: 验证集最佳 checkpoint
  - `epoch_XXXX.pth`: 定期保存的 checkpoint
- TensorBoard logs: `output_dir/logs/`

**监控训练:**
```bash
tensorboard --logdir ./icp_train_output/logs
```

### 步骤 4: 评估和使用训练好的模型

训练完成后，可以加载最佳模型用于推理。

```python
import torch
from vggt.vggt.heads.sparse_conv_refine_head import GaussianRefineHeadSparseConv

# 加载模型
model = GaussianRefineHeadSparseConv(
    input_gaussian_dim=14,
    output_gaussian_dim=14,
    feature_dim=384,
    num_conv_layers=10,
    voxel_size=0.05,
    # ... 其他参数
)

checkpoint = torch.load('./icp_train_output/checkpoints/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
with torch.no_grad():
    refined_gaussians = model(input_gaussians, pred_scale)
```

## 数据格式

### .npz 样本对文件格式

每个 `.npz` 文件包含一个物体的样本对:

```python
{
    'input_gaussians': np.ndarray,    # [N, 14] 粗糙 Gaussian 参数
    'target_gaussians': np.ndarray,   # [N, 14] ICP GT Gaussian 参数
    'pred_scale': float,              # 用于体素化的 scale
    'object_id': int,                 # 物体 ID
    'input_pcd_path': str (optional), # 输入点云路径
    'target_pcd_path': str (optional) # GT 点云路径
}
```

### Gaussian 参数格式 [N, 14]

```
[0:3]   - xyz positions (means)
[3:6]   - scales
[6:9]   - RGB colors (range [0, 1])
[9:13]  - quaternion rotations (w, x, y, z, normalized)
[13:14] - opacity (range [0, 1])
```

## 损失函数

### 主要损失: ICPSupervisionLoss

提供参数空间的直接监督:

1. **Position Loss** (MSE): xyz 位置，最重要
2. **Scale Loss** (MSE): 尺度参数
3. **Rotation Loss** (Quaternion distance): 旋转四元数
4. **Color Loss** (MSE): RGB 颜色
5. **Opacity Loss** (MSE): 透明度

### 备选损失: ICPChamferLoss

点云空间的 Chamfer 距离，可以与主损失联合使用:

```yaml
use_chamfer_loss: true
chamfer_weight: 1.0
```

## 关键设计考虑

### 1. 为什么使用 ICP GT？

- **强监督**: 相比 Stage2 的渲染损失，ICP 提供点级别的直接监督
- **验证网络**: 可以验证当前网络结构是否有能力学习 Gaussian 精细化
- **快速收敛**: 强监督信号可能带来更快的收敛速度

### 2. 为什么离线生成？

- **速度**: ICP 配准较慢，在线计算会严重拖累训练速度
- **稳定性**: 离线生成保证所有样本的 GT 质量
- **可重复性**: 预先生成的数据集保证实验可重复

### 3. Gaussians 一一对应

- ICP 配准保持点的数量和顺序不变，只改变位置
- 这使得可以实现 **点对点的强监督**，而不是集合级别的弱监督
- 其他参数（scale, rotation, color, opacity）也保持对应

### 4. 只监督位置 vs 全参数监督

- **只监督位置** (`position_only=True`):
  - 适合快速验证网络能否学习位置精细化
  - 与当前 Stage2 实现一致（只调整 means）

- **全参数监督** (`position_only=False`):
  - 理论上可以学习更完整的 Gaussian 精细化
  - 需要确保 ICP 后其他参数的变化有意义

## 常见问题

### Q1: 如何获取 Stage2 的 dynamic_objects 数据？

在 Stage2 训练代码中添加保存逻辑:

```python
# 在 stage2_loss.py 的 forward 函数中
import pickle

# 保存 dynamic_objects
with open('dynamic_objects.pkl', 'wb') as f:
    pickle.dump({
        'dynamic_objects': dynamic_objects,
        # 可以保存其他需要的数据
    }, f)
```

### Q2: ICP 配准质量不好怎么办？

调整 ICP 参数:
- 增加 `--max_icp_iterations`
- 调整 `--voxel_size`
- 启用 `--use_color_features`
- 增加 `--min_frames` 确保有足够的帧数

### Q3: 训练收敛慢或不收敛？

- 检查学习率是否合适
- 尝试调整损失权重，增大 `position_weight`
- 启用 `use_smooth_l1: true` 使用更鲁棒的损失
- 检查数据集质量，使用可视化工具验证 ICP GT

### Q4: 内存不足？

- 减小 `batch_size`
- 设置 `cache_in_memory: false`
- 减少 `num_workers`
- 限制每个物体的最大点数

## 进阶使用

### 自定义损失函数

在 `icp_loss.py` 中可以添加自定义损失:

```python
class CustomICPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Your custom loss

    def forward(self, pred, target):
        # Your loss computation
        return loss
```

### 数据增强

在 `dataset.py` 的 `__getitem__` 中添加数据增强:

```python
# 例如：添加随机噪声
input_gaussians = input_gaussians + torch.randn_like(input_gaussians) * 0.01
```

## 实验建议

1. **首先验证 ICP GT 质量**: 使用可视化工具检查配准效果
2. **从小数据集开始**: 先用少量样本验证训练流程
3. **只监督位置**: 初始实验建议 `position_only=True`
4. **监控 Chamfer 距离**: 在验证集上计算 Chamfer 距离评估质量
5. **对比 Stage2**: 将 ICP 监督训练的模型与 Stage2 训练的模型对比

## 引用

如果使用 ICP 配准，相关参考:

- Besl, P. J., & McKay, N. D. (1992). "A method for registration of 3-D shapes". PAMI.
- Park, J., Zhou, Q. Y., & Koltun, V. (2017). "Colored point cloud registration revisited". ICCV.

## License

与主项目保持一致。

## 作者

VGGT Team

---

**联系方式**: 如有问题请提交 Issue 或联系项目维护者。
