# ICP Supervision 快速开始指南

## 5分钟快速上手

### 1. 验证模块安装

```bash
cd /mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo

# 运行测试脚本
/opt/miniconda/envs/vggt/bin/python src/icp_supervision/test_module.py
```

如果看到 "✓ All tests passed!" 说明模块已正确安装。

### 2. 准备数据

你需要从 Stage2 训练中获取 `dynamic_objects` 数据。在 Stage2 训练代码中添加：

```python
# 在 stage2_loss.py 或训练循环中
import pickle

# 保存 dynamic_objects (建议每隔一定 iterations 保存一次)
if iteration % save_interval == 0:
    save_data = {
        'dynamic_objects': dynamic_objects,  # 来自 processor 的输出
        # 可选：保存其他需要的数据
    }

    with open(f'dynamic_objects_iter_{iteration}.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Saved dynamic_objects to dynamic_objects_iter_{iteration}.pkl")
```

### 3. 生成 ICP GT 数据

```bash
cd src

# 基础用法
/opt/miniconda/envs/vggt/bin/python -m icp_supervision.data_generator \
    --input /path/to/dynamic_objects.pkl \
    --output_dir ./icp_supervision_data \
    --save_pointclouds

# 推荐参数（更快速）
/opt/miniconda/envs/vggt/bin/python -m icp_supervision.data_generator \
    --input /path/to/dynamic_objects.pkl \
    --output_dir ./icp_supervision_data \
    --voxel_size 0.02 \
    --max_icp_iterations 30 \
    --min_frames 2 \
    --max_frames 5 \
    --save_pointclouds
```

**重要提示:**
- `--input`: 替换为你实际的 pickle 文件路径
- ICP 配准较慢，建议从小数据集开始（few objects）
- 第一次运行建议添加 `--save_pointclouds` 用于验证

### 4. 验证 ICP GT 质量

```bash
# 查看统计信息
/opt/miniconda/envs/vggt/bin/python -m icp_supervision.visualize \
    --data_dir ./icp_supervision_data \
    --mode stats

# 可视化前3个样本
/opt/miniconda/envs/vggt/bin/python -m icp_supervision.visualize \
    --data_dir ./icp_supervision_data \
    --mode comparison \
    --num_samples 3
```

**检查点:**
- Chamfer distance 应该较小（< 0.1 为好）
- 对比视图中红色和绿色点云应该大致重合
- 如果质量不好，调整 ICP 参数重新生成

### 5. 训练模型

```bash
# 首先编辑配置文件
vim src/icp_supervision/config/icp_train.yaml

# 修改以下关键参数:
# - data_dir: './icp_supervision_data'  # 改为实际路径
# - output_dir: './icp_train_output'
# - epochs: 50  # 可以先用较少 epochs 测试
# - batch_size: 2  # 根据 GPU 内存调整

# 开始训练
/opt/miniconda/envs/vggt/bin/python -m icp_supervision.train \
    --config src/icp_supervision/config/icp_train.yaml
```

### 6. 监控训练

在另一个终端中：

```bash
cd /mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo

# 启动 TensorBoard
tensorboard --logdir ./icp_train_output/logs --port 6006
```

然后在浏览器中访问: `http://localhost:6006`

### 7. 使用训练好的模型

```python
import torch
from vggt.vggt.heads.sparse_conv_refine_head import GaussianRefineHeadSparseConv

# 1. 创建模型（参数需与配置文件一致）
model = GaussianRefineHeadSparseConv(
    input_gaussian_dim=14,
    output_gaussian_dim=14,
    feature_dim=384,
    num_conv_layers=10,
    voxel_size=0.05,
    use_dilated_conv=True,
    dilation_rates=[1, 2, 2, 4, 4, 8, 8, 16, 16, 16],
    max_num_points_per_voxel=5,
)

# 2. 加载最佳 checkpoint
checkpoint = torch.load('./icp_train_output/checkpoints/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.cuda()
model.eval()

print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")

# 3. 推理
input_gaussians = ...  # [N, 14] 来自 Stage2
pred_scale = ...       # [1] scalar

with torch.no_grad():
    refined_gaussians = model(
        input_gaussians.unsqueeze(0).cuda(),  # [1, N, 14]
        pred_scale.cuda()                      # [1]
    )
    refined_gaussians = refined_gaussians.squeeze(0)  # [N, 14]

print(f"Input: {input_gaussians.shape}")
print(f"Output: {refined_gaussians.shape}")
```

## 常见问题快速解决

### 问题 1: ImportError

```bash
# 确保在正确的环境中
conda activate vggt

# 确保路径正确
export PYTHONPATH="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo:$PYTHONPATH"
```

### 问题 2: CUDA out of memory

编辑 `config/icp_train.yaml`:
```yaml
batch_size: 1  # 减小 batch size
cache_in_memory: false  # 不缓存到内存
```

### 问题 3: ICP 配准太慢

```bash
# 使用更激进的参数
/opt/miniconda/envs/vggt/bin/python -m icp_supervision.data_generator \
    --input /path/to/dynamic_objects.pkl \
    --output_dir ./icp_supervision_data \
    --voxel_size 0.05 \              # 更大的体素
    --max_icp_iterations 20 \        # 更少迭代
    --min_frames 2 \
    --max_frames 3                   # 更少帧数
```

### 问题 4: No .npz files found

确保:
1. `data_generator.py` 运行成功
2. `--output_dir` 路径正确
3. 至少有一个物体满足 `min_frames` 要求

## 进阶配置

### 只监督位置（推荐用于初始实验）

编辑 `config/icp_train.yaml`:
```yaml
position_only: true
position_weight: 10.0
```

### 添加 Chamfer Loss

编辑 `config/icp_train.yaml`:
```yaml
use_chamfer_loss: true
chamfer_weight: 1.0
```

### 调整学习率

编辑 `config/icp_train.yaml`:
```yaml
lr: 1e-3           # 更大的学习率（更快收敛但可能不稳定）
scheduler: 'step'  # 使用 step scheduler
lr_decay_step: 20
lr_decay_gamma: 0.5
```

## 预期结果

### 数据生成阶段
- **时间**: 取决于物体数量和帧数，通常每个物体 10-30 秒
- **成功率**: 应该 > 80%，如果太低说明 ICP 参数需要调整
- **Chamfer 距离**: 平均应 < 0.1

### 训练阶段
- **Loss 下降**: 通常在前 10-20 epochs 快速下降
- **收敛时间**: 50-100 epochs（取决于数据集大小）
- **最终 position_loss**: 应降到 < 0.01（对于归一化场景）

### 对比 Stage2
- ICP 监督应该收敛更快
- 验证时可以比较 refinement 前后的 Chamfer 距离

## 下一步

1. **评估模型**: 在测试集上计算 Chamfer 距离
2. **可视化结果**: 对比 input, ICP GT, 和 model prediction
3. **集成到 Stage2**: 如果效果好，可以考虑将训练好的模型集成到 Stage2 pipeline
4. **消融实验**: 尝试不同的损失权重组合

## 需要帮助？

- 查看详细文档: [README.md](README.md)
- 查看代码注释: 每个文件都有详细的 docstring
- 运行测试: `python src/icp_supervision/test_module.py`

## Checklist

- [ ] 测试脚本通过
- [ ] 准备好 `dynamic_objects.pkl` 文件
- [ ] 生成 ICP GT 数据成功
- [ ] 可视化验证 GT 质量
- [ ] 配置文件参数已调整
- [ ] 训练启动并正常运行
- [ ] TensorBoard 正常工作
- [ ] 保存了最佳 checkpoint

完成以上步骤后，你就成功设置好了 ICP 监督训练！🎉
