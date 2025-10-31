# Stage2 Inference Script Usage Guide

## Quick Start

The `run_stage2_inference.sh` script now supports the new `velocity_transform_mode` parameter!

## Basic Usage

```bash
# 默认参数 (Flow-based, simple mode)
./run_stage2_inference.sh

# Velocity-based with simple mode (fast, translation only)
./run_stage2_inference.sh 100 10 150 true

# Velocity-based with procrustes mode (accurate, R+T)
./run_stage2_inference.sh 100 10 150 true procrustes
```

## All Parameters

```bash
./run_stage2_inference.sh START_IDX STEP END_IDX USE_VELOCITY VELOCITY_TRANSFORM_MODE VELOCITY_THRESHOLD CLUSTERING_EPS CLUSTERING_MIN_SAMPLES TRACKING_POSITION_THRESHOLD TRACKING_VELOCITY_THRESHOLD
```

### Parameter Details

| Position | Parameter | Default | Options | Description |
|----------|-----------|---------|---------|-------------|
| $1 | START_IDX | 150 | any int | 起始帧索引 |
| $2 | STEP | 5 | any int | 帧间隔 |
| $3 | END_IDX | 200 | any int | 结束帧索引 |
| $4 | USE_VELOCITY | false | true/false | 是否使用velocity-based方法 |
| $5 | VELOCITY_TRANSFORM_MODE | "simple" | "simple"/"procrustes" | Velocity变换模式 |
| $6 | VELOCITY_THRESHOLD | 0.1 | any float | 速度阈值 (m/s) |
| $7 | CLUSTERING_EPS | 0.02 | any float | DBSCAN邻域半径 |
| $8 | CLUSTERING_MIN_SAMPLES | 10 | any int | DBSCAN最小样本数 |
| $9 | TRACKING_POSITION_THRESHOLD | 2.0 | any float | 跟踪位置阈值 |
| $10 | TRACKING_VELOCITY_THRESHOLD | 0.2 | any float | 跟踪速度阈值 |

## Common Use Cases

### 1. Flow-based (Baseline)
```bash
./run_stage2_inference.sh 100 10 150 false
```
- Uses optical flow for transformation
- Most accurate when flow is available
- Slower

### 2. Velocity-based Simple (Fast)
```bash
./run_stage2_inference.sh 100 10 150 true simple
```
- Only estimates translation (T)
- Rotation is identity (R = I)
- Very fast
- Good for objects with minimal rotation

### 3. Velocity-based Procrustes (Accurate)
```bash
./run_stage2_inference.sh 100 10 150 true procrustes
```
- Estimates both rotation (R) and translation (T)
- Uses xyz + velocity + Procrustes algorithm
- More accurate for rotating objects
- Slightly slower than simple mode

### 4. Custom Parameters
```bash
# Adjust velocity threshold
./run_stage2_inference.sh 100 10 150 true procrustes 0.15

# Adjust clustering
./run_stage2_inference.sh 100 10 150 true procrustes 0.1 0.03

# Full customization
./run_stage2_inference.sh 100 10 150 true procrustes 0.1 0.02 15 3.0 0.3
```

## Velocity Transform Modes Explained

### Simple Mode
- **Algorithm**: Average velocity → Translation only
- **Speed**: ★★★★★ Very Fast
- **Accuracy**: ★★★☆☆ Good for translation-dominant motion
- **Formula**: `T[:3, 3] = mean(velocity)`

### Procrustes Mode
- **Algorithm**: xyz + velocity → Procrustes/Kabsch → R + T
- **Speed**: ★★★★☆ Fast
- **Accuracy**: ★★★★☆ Better for complex motion
- **Formula**: `R, T = Procrustes(xyz_src, xyz_src + velocity)`

## Output

The script will generate:
- 4-column comparison videos: GT | Initial | Refined | Dynamic Clustering
- Per-frame visualizations in `./stage2_inference_outputs/`

## Direct Python Usage

You can also call the Python script directly:

```bash
# Simple mode
/opt/miniconda/envs/vggt/bin/python demo_stage2_inference.py \
    --batch_mode \
    --start_idx 100 --end_idx 150 --step 10 \
    --use_velocity_based_transform \
    --velocity_transform_mode simple

# Procrustes mode
/opt/miniconda/envs/vggt/bin/python demo_stage2_inference.py \
    --batch_mode \
    --start_idx 100 --end_idx 150 --step 10 \
    --use_velocity_based_transform \
    --velocity_transform_mode procrustes
```

## Tips

1. **Start with simple mode** for quick prototyping
2. **Use procrustes mode** for final results or when objects rotate
3. **Use flow-based** when optical flow is available and reliable
4. **Adjust velocity_threshold** if no objects are detected (try 0.05)
5. **Monitor console output** for transformation method confirmation

## Troubleshooting

### No dynamic objects detected
- Lower `VELOCITY_THRESHOLD` (try 0.05)
- Check that velocity predictions are in metric scale (m/s)

### Procrustes mode falling back to simple
- Check console for "Procrustes估计失败" messages
- This is normal for challenging cases
- The fallback ensures robustness

### Performance issues
- Use simple mode for faster processing
- Reduce number of frames or objects

## Examples

```bash
# Quick test with simple mode
./run_stage2_inference.sh 150 5 160 true simple

# Thorough test with procrustes mode
./run_stage2_inference.sh 100 5 200 true procrustes

# Custom velocity threshold with procrustes
./run_stage2_inference.sh 100 5 150 true procrustes 0.15

# Compare all three methods
./run_stage2_inference.sh 100 5 150 false        # Flow-based
./run_stage2_inference.sh 100 5 150 true simple  # Velocity simple
./run_stage2_inference.sh 100 5 150 true procrustes  # Velocity procrustes
```

## See Also

- [VELOCITY_TRANSFORM_MODES.md](VELOCITY_TRANSFORM_MODES.md) - Detailed mode documentation
- [TORCH_CONVERSION_SUMMARY.md](TORCH_CONVERSION_SUMMARY.md) - Technical implementation details
