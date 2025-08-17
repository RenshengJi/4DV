# 4D视频点云处理工作流

本工作流用于处理4D视频数据，提取动态物体的点云并进行配准，最终生成完整的3D点云模型。

## 工作流概述

工作流分为两个主要阶段：

1. **点云提取和聚类** (`demo_video_with_pointcloud_save.py`)
   - 从4D视频中提取深度和RGB信息
   - 进行动态物体检测和聚类
   - 跨帧物体匹配和跟踪
   - 保存每个物体的多帧点云数据

2. **点云配准** (`point_cloud_registration_save.py`)
   - 读取第一阶段保存的点云数据
   - 对每个物体的多帧点云进行配准
   - 生成完整的3D点云模型
   - 保存为可查看的格式（PLY、PCD、OBJ）

## 文件结构

```
├── demo_video_with_pointcloud_save.py    # 第一阶段：点云提取和聚类
├── point_cloud_registration_save.py      # 第二阶段：点云配准
├── test_workflow.py                      # 工作流测试脚本
├── WORKFLOW_README.md                    # 本说明文件
├── results_26040_8views_true/            # 推理结果输出目录
│   ├── pointcloud_data_*.pkl             # 保存的点云数据文件
│   └── *.mp4                             # 可视化视频文件
└── saved_pointclouds/                    # 配准结果输出目录
    ├── *_Object_*_Registered.ply         # 配准后的点云文件
    └── registration_report.txt           # 配准报告
```

## 使用方法

### 第一阶段：点云提取和聚类

```bash
python demo_video_with_pointcloud_save.py \
    --save_pointcloud_data \
    --idx 600 \
    --velocity_threshold 0.01 \
    --dbscan_eps 0.005 \
    --dbscan_min_samples 10 \
    --position_threshold 1.0 \
    --velocity_threshold_match 0.5 \
    --fusion_alpha 0.7 \
    --area_threshold 100
```

**主要参数说明：**
- `--save_pointcloud_data`: 启用点云数据保存
- `--idx`: 起始视频帧索引
- `--velocity_threshold`: 速度阈值，用于过滤静态背景
- `--dbscan_eps`: DBSCAN聚类半径
- `--dbscan_min_samples`: DBSCAN最小样本数
- `--position_threshold`: 帧间匹配的位置阈值
- `--velocity_threshold_match`: 帧间匹配的速度阈值
- `--fusion_alpha`: 动态物体与源图像的融合透明度
- `--area_threshold`: 面积阈值，过滤小聚类

### 第二阶段：点云配准

```bash
python point_cloud_registration_save.py \
    --input_dir ./results_26040_8views_true \
    --output_dir ./saved_pointclouds \
    --voxel_size 0.01 \
    --max_iterations 50 \
    --save_format ply
```

**主要参数说明：**
- `--input_dir`: 输入数据目录（第一阶段输出）
- `--output_dir`: 输出目录
- `--voxel_size`: 体素大小，用于下采样
- `--max_iterations`: 最大迭代次数
- `--save_format`: 保存格式（ply, pcd, obj）

### 测试工作流

```bash
# 测试现有数据
python test_workflow.py

# 运行完整测试（包括推理）
python test_workflow.py --run_inference --idx 600
```

## 输出文件说明

### 第一阶段输出

1. **点云数据文件** (`pointcloud_data_*.pkl`)
   - 包含原始预测结果、VGGT批次数据、置信度掩码
   - 包含聚类结果：每帧的物体ID、聚类索引、中心点等
   - 用于第二阶段点云重建和配准

2. **可视化视频** (`*.mp4`)
   - 包含源图像、预测结果、深度图、速度图
   - 包含动态物体聚类结果（与源图像融合）

### 第二阶段输出

1. **配准后的点云文件**
   - `*_Object_*_Registered.ply`: 多帧配准后的完整点云
   - `*_Object_*_Single.ply`: 单帧点云（当物体只出现在一帧时）

2. **配准报告** (`registration_report.txt`)
   - 处理时间、文件统计、配准成功率等信息

## 技术细节

### 动态物体检测和聚类

1. **速度过滤**: 使用速度阈值过滤静态背景点
2. **DBSCAN聚类**: 对动态点进行空间聚类
3. **面积过滤**: 过滤掉面积过小的聚类
4. **跨帧匹配**: 使用匈牙利算法进行物体跟踪

### 点云配准

1. **预处理**: 去噪、下采样、法向量估计
2. **特征提取**: 使用FPFH特征描述子
3. **RANSAC配准**: 基于特征的粗配准
4. **ICP精细配准**: 点对平面ICP算法
5. **全局优化**: 多帧点云的全局配准

## 故障排除

### 常见问题

1. **没有检测到动态物体**
   - 调整 `velocity_threshold` 参数
   - 检查输入视频是否包含运动物体

2. **聚类结果不理想**
   - 调整 `dbscan_eps` 和 `dbscan_min_samples` 参数
   - 增加 `area_threshold` 过滤小聚类

3. **配准失败**
   - 减少 `voxel_size` 提高精度
   - 增加 `max_iterations` 提高收敛性
   - 检查点云质量，确保有足够的重叠区域

4. **内存不足**
   - 增加 `voxel_size` 减少点云密度
   - 减少 `max_iterations` 降低计算量

### 性能优化

1. **GPU加速**: 确保使用CUDA设备进行推理
2. **并行处理**: 可以同时处理多个视频序列
3. **参数调优**: 根据具体场景调整聚类和配准参数

## 扩展功能

1. **多物体跟踪**: 支持复杂场景中的多个动态物体
2. **时序一致性**: 确保物体在时间上的连续性
3. **质量评估**: 自动评估点云质量和配准精度
4. **可视化工具**: 提供交互式点云查看器

## 依赖库

- PyTorch
- Open3D
- NumPy
- OpenCV
- scikit-learn
- scipy
- matplotlib
- imageio

确保安装所有必要的依赖库：

```bash
pip install -r requirements.txt
``` 