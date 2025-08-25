#!/usr/bin/env python3
"""
Waymo Stage2 Online Training Script

专门用于Waymo数据集的第二阶段在线训练的便捷脚本
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Waymo Stage2 Online Training')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-6, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--exp-name', type=str, default='waymo_stage2_online', help='Experiment name')
    parser.add_argument('--stage2-start-epoch', type=int, default=5, help='Stage2 start epoch')
    parser.add_argument('--stage2-frequency', type=int, default=3, help='Stage2 training frequency')
    
    args = parser.parse_args()
    
    # 构建训练命令
    cmd = [
        'python', 'src/train.py',
        '--config-name', 'waymo/stage2_online',
        f'batch_size={args.batch_size}',
        f'epochs={args.epochs}',
        f'lr={args.lr}',
        f'exp_name={args.exp_name}',
        f'stage2_start_epoch={args.stage2_start_epoch}',
        f'stage2_frequency={args.stage2_frequency}'
    ]
    
    if args.resume:
        cmd.append(f'resume={args.resume}')
    
    # 显示配置信息
    print("=" * 60)
    print("🚗 Waymo Stage2 Online Training Configuration")
    print("=" * 60)
    print(f"📦 Batch Size: {args.batch_size}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📈 Learning Rate: {args.lr}")
    print(f"🎯 Experiment Name: {args.exp_name}")
    print(f"🚀 Stage2 Start Epoch: {args.stage2_start_epoch}")
    print(f"⚡ Stage2 Frequency: {args.stage2_frequency}")
    if args.resume:
        print(f"🔄 Resume from: {args.resume}")
    print("=" * 60)
    
    # 显示命令
    print("💻 Training Command:")
    print(" ".join(cmd))
    print("=" * 60)
    
    # 确认启动
    response = input("🤔 Start training? (y/N): ")
    if response.lower() in ['y', 'yes']:
        print("🚀 Starting Waymo Stage2 training...")
        
        # 执行训练
        try:
            subprocess.run(cmd, check=True)
            print("✅ Training completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed with exit code {e.returncode}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n⏹️ Training interrupted by user")
            sys.exit(0)
    else:
        print("❌ Training cancelled")

if __name__ == "__main__":
    main()