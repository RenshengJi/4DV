#!/usr/bin/env python3
"""
光流配准系统使用示例
展示如何使用基于光流的点云配准功能
"""

import os
import argparse
import glob
from optical_flow_registration import OpticalFlowRegistration


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="光流配准系统使用示例")
    parser.add_argument("--data_dir", type=str, 
                       default="./results",
                       help="包含点云数据文件的目录")
    parser.add_argument("--output_dir", type=str, 
                       default="./optical_flow_results",
                       help="输出目录")
    parser.add_argument("--flow_model", type=str, default="raft",
                       choices=["raft", "pwc", "opencv"],
                       help="光流模型")
    parser.add_argument("--use_pnp", action="store_true",
                       help="使用3DPnP方法（复杂版），否则使用直接平移（简单版）")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备")
    parser.add_argument("--min_inliers_ratio", type=float, default=0.3,
                       help="最小内点比例")
    parser.add_argument("--ransac_threshold", type=float, default=3.0,
                       help="RANSAC阈值")
    parser.add_argument("--max_flow_magnitude", type=float, default=100.0,
                       help="最大光流幅度阈值")
    parser.add_argument("--process_all", action="store_true",
                       help="处理目录中的所有点云数据文件")
    parser.add_argument("--data_file", type=str, default=None,
                       help="指定单个点云数据文件路径")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化配准器
    print("初始化光流配准器...")
    registration = OpticalFlowRegistration(
        flow_model_name=args.flow_model,
        device=args.device,
        use_pnp=args.use_pnp,
        min_inliers_ratio=args.min_inliers_ratio,
        ransac_threshold=args.ransac_threshold,
        max_flow_magnitude=args.max_flow_magnitude
    )
    
    # 确定要处理的文件
    if args.data_file:
        # 处理单个文件
        data_files = [args.data_file]
    elif args.process_all:
        # 处理目录中的所有文件
        pattern = os.path.join(args.data_dir, "pointcloud_data_*.pkl")
        data_files = glob.glob(pattern)
        data_files.sort()
    else:
        # 处理最新的文件
        pattern = os.path.join(args.data_dir, "pointcloud_data_*.pkl")
        data_files = glob.glob(pattern)
        if data_files:
            data_files = [max(data_files, key=os.path.getctime)]
        else:
            print(f"在目录 {args.data_dir} 中未找到点云数据文件")
            return
    
    print(f"找到 {len(data_files)} 个点云数据文件")
    
    # 处理每个文件
    for i, data_file in enumerate(data_files):
        print(f"\n{'='*50}")
        print(f"处理文件 {i+1}/{len(data_files)}: {os.path.basename(data_file)}")
        print(f"{'='*50}")
        
        # 为每个文件创建单独的输出目录
        file_name = os.path.splitext(os.path.basename(data_file))[0]
        file_output_dir = os.path.join(args.output_dir, file_name)
        os.makedirs(file_output_dir, exist_ok=True)
        
        # try:
        # 处理点云数据
        results = registration.process_pointcloud_data(data_file, file_output_dir)
        
        # 打印结果统计
        if results:
            total_points = sum(result['num_points'] for result in results.values())
            avg_points = total_points / len(results)
            print(f"\n文件 {file_name} 处理完成:")
            print(f"  成功处理物体数: {len(results)}")
            print(f"  总点数: {total_points}")
            print(f"  平均每物体点数: {avg_points:.1f}")
            
            # 打印每个物体的详细信息
            for global_id, result in results.items():
                print(f"  物体 {global_id}: {result['num_frames']} 帧, {result['num_points']} 点, 中间帧 {result['middle_frame']}")
            else:
                print(f"文件 {file_name} 处理完成，但未找到有效的物体")
                
        # except Exception as e:
        #     print(f"处理文件 {file_name} 时出错: {e}")
        #     continue
    
    print(f"\n所有文件处理完成！结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main() 