#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# SAM preprocessing for WayMo Open dataset
# This script adds SAM mask generation to WayMo dataset preprocessing
# --------------------------------------------------------

import sys
import os
import os.path as osp
import shutil
import json
from tqdm import tqdm
import PIL.Image
import numpy as np
import argparse
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import sys
# 添加vggt路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))
import cv2

import tensorflow.compat.v1 as tf

tf.enable_eager_execution()

import path_to_root  # noqa
from src.dust3r.utils.geometry import geotrf, inv
from src.dust3r.utils.image import imread_cv2
from src.dust3r.utils.parallel import parallel_processes as parallel_map
from datasets_preprocess.utils import cropping

# 导入SAM预处理模块
from sam_preprocessing import SAMPreprocessor


def get_parser():
    parser = argparse.ArgumentParser(description="SAM preprocessing for WayMo dataset")
    parser.add_argument("--waymo_dir", default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/test", help="Path to WayMo dataset directory")
    parser.add_argument("--output_dir", default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/test", help="Output directory for processed data")
    parser.add_argument("--workers", type=int, default=40, help="Number of parallel workers")
    parser.add_argument("--resolution", type=int, default=512, help="Output image resolution")
    
    # SAM相关参数
    parser.add_argument("--sam_model_type", default="sam2", choices=["sam2", "sam"], 
                       help="SAM model type")
    parser.add_argument("--sam_device", default="cuda", help="Device for SAM model")
    parser.add_argument("--sam_config_file", default="configs/sam2.1/sam2.1_hiera_t.yaml", help="SAM2 config file path")
    parser.add_argument("--sam_ckpt_path", default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/sam2.1_hiera_tiny.pt", help="SAM model checkpoint path")
    
    # 处理选项
    parser.add_argument("--skip_existing", action="store_true", 
                       help="Skip existing SAM mask files")
    parser.add_argument("--dry_run", action="store_true", 
                       help="Dry run - only show what would be processed")
    
    return parser


def process_image_with_sam(image_path, sam_preprocessor, output_path, dry_run=False):
    """处理单张图片并生成SAM掩码"""
    try:
        if dry_run:
            print(f"Would process: {image_path} -> {output_path}")
            return True
            
        # 读取图片
        image = PIL.Image.open(image_path).convert('RGB')
        
        # 生成SAM掩码
        masks = sam_preprocessor.generate_masks(image)
        
        # 保存掩码
        sam_preprocessor.save_masks(masks, output_path)
        
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def process_sequence_directory(seq_dir, sam_preprocessor, skip_existing=False, dry_run=False):
    """处理单个序列目录中的所有图片"""
    # 创建SAM掩码输出目录
    sam_output_dir = seq_dir
    if not dry_run:
        os.makedirs(sam_output_dir, exist_ok=True)
    
    # 查找所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(seq_dir).glob(f"*{ext}"))
    
    print(f"Found {len(image_files)} images in {seq_dir}")
    
    processed_count = 0
    for image_file in tqdm(image_files, desc=f"Processing {osp.basename(seq_dir)}"):
        # 生成输出路径
        output_name = image_file.stem + ".npy"
        output_path = osp.join(sam_output_dir, output_name)
        
        # 检查是否跳过已存在的文件
        if skip_existing and osp.exists(output_path):
            continue
            
        # 处理图片
        if process_image_with_sam(str(image_file), sam_preprocessor, output_path, dry_run):
            processed_count += 1
    
    print(f"Processed {processed_count} images in {seq_dir}")
    return processed_count


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    print("="*60)
    print("WayMo SAM Preprocessing")
    print("="*60)
    print(f"Input directory: {args.waymo_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Workers: {args.workers}")
    print(f"Resolution: {args.resolution}")
    print(f"SAM model type: {args.sam_model_type}")
    print(f"SAM device: {args.sam_device}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Dry run: {args.dry_run}")
    print("="*60)
    
    # 初始化SAM预处理器
    sam_preprocessor = None
    if not args.dry_run:
        try:
            sam_preprocessor = SAMPreprocessor(
                model_type=args.sam_model_type,
                device=args.sam_device,
                config_file=args.sam_config_file,
                ckpt_path=args.sam_ckpt_path
            )
            print(f"✅ Successfully initialized SAM preprocessor with model type: {args.sam_model_type}")
        except Exception as e:
            print(f"❌ Failed to initialize SAM preprocessor: {e}")
            print("Continuing without SAM preprocessing")
            sam_preprocessor = None
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查找所有序列目录
    sequence_dirs = []
    for item in os.listdir(args.waymo_dir):
        item_path = osp.join(args.waymo_dir, item)
        if osp.isdir(item_path):
            sequence_dirs.append(item_path)
    
    print(f"Found {len(sequence_dirs)} sequence directories")
    
    # 处理每个序列目录
    total_processed = 0
    for seq_dir in sequence_dirs:
        print(f"\nProcessing sequence: {osp.basename(seq_dir)}")
        processed = process_sequence_directory(
            seq_dir, sam_preprocessor, args.skip_existing, args.dry_run
        )
        total_processed += processed
    
    print(f"\n✅ Done! Total processed images: {total_processed}")
    
    if not args.dry_run and sam_preprocessor is not None:
        print("📊 SAM masks have been generated for all images")
        print("📁 SAM masks are saved in 'sam_masks' subdirectories")


if __name__ == "__main__":
    main() 

