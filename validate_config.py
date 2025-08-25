#!/usr/bin/env python3
"""
配置文件验证脚本

验证Waymo Stage2配置文件的有效性和完整性
"""

import sys
import os
from pathlib import Path

def validate_waymo_stage2_config():
    """验证Waymo Stage2在线配置文件"""
    
    config_path = "config/waymo/stage2_online.yaml"
    
    print("🔍 验证Waymo Stage2配置文件...")
    print(f"📁 配置文件路径: {config_path}")
    print("=" * 50)
    
    # 1. 检查文件存在
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    print("✅ 配置文件存在")
    
    # 2. 验证YAML语法
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ YAML语法正确")
    except yaml.YAMLError as e:
        print(f"❌ YAML语法错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        return False
    
    # 3. 检查关键配置项
    required_keys = [
        'enable_stage2',
        'stage2_start_epoch', 
        'stage2_frequency',
        'dataset_waymo',
        'train_dataset',
        'exp_name'
    ]
    
    missing_keys = []
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    
    if missing_keys:
        print(f"❌ 缺少必需的配置项: {missing_keys}")
        return False
    
    print("✅ 必需配置项完整")
    
    # 4. 验证Hydra配置加载
    try:
        from omegaconf import OmegaConf
        hydra_config = OmegaConf.load(config_path)
        print("✅ Hydra配置加载成功")
    except Exception as e:
        print(f"❌ Hydra配置加载失败: {e}")
        return False
    
    # 5. 显示关键配置信息
    print("\n📋 关键配置信息:")
    print(f"  🎯 实验名称: {config.get('exp_name', 'N/A')}")
    print(f"  🚀 第二阶段启用: {config.get('enable_stage2', False)}")
    print(f"  📊 开始轮次: {config.get('stage2_start_epoch', 'N/A')}")
    print(f"  🔄 训练频率: {config.get('stage2_frequency', 'N/A')}")
    print(f"  🎓 批次大小: {config.get('batch_size', 'N/A')}")
    print(f"  📈 学习率: {config.get('lr', 'N/A')}")
    print(f"  🏁 总轮次: {config.get('epochs', 'N/A')}")
    
    # 6. 验证损失权重
    loss_weights = {
        'rgb_loss_weight': config.get('rgb_loss_weight'),
        'depth_loss_weight': config.get('depth_loss_weight'),
        'lpips_loss_weight': config.get('lpips_loss_weight'),
        'consistency_loss_weight': config.get('consistency_loss_weight'),
        'gaussian_reg_weight': config.get('gaussian_reg_weight'),
        'pose_reg_weight': config.get('pose_reg_weight'),
        'temporal_smooth_weight': config.get('temporal_smooth_weight')
    }
    
    print("\n⚖️  损失权重配置:")
    for weight_name, weight_value in loss_weights.items():
        if weight_value is not None:
            print(f"  {weight_name}: {weight_value}")
        else:
            print(f"  ⚠️  {weight_name}: 未设置")
    
    # 7. 检查数据集配置
    if 'dataset_waymo' in config:
        print(f"\n📦 数据集配置:")
        dataset_info = str(config['dataset_waymo'])
        print(f"  {dataset_info}")
    
    print("\n" + "=" * 50)
    print("✅ 配置文件验证通过！")
    print("\n🚀 可以使用以下命令开始训练:")
    print("python src/train.py --config-name waymo/stage2_online")
    
    return True

if __name__ == "__main__":
    success = validate_waymo_stage2_config()
    sys.exit(0 if success else 1)