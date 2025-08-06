#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# SAM preprocessing module for datasets
# This module provides functionality to generate SAM masks for images
# --------------------------------------------------------

import os
import sys
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from pathlib import Path

# 添加sam2路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/sam2'))

try:
    # 确保Hydra正确初始化
    import sam2
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    SAM2_AVAILABLE = True
except ImportError:
    print("Warning: SAM2 not available, falling back to SAM")
    SAM2_AVAILABLE = False
    try:
        import segment_anything as sam
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        SAM_AVAILABLE = True
    except ImportError:
        print("Warning: Neither SAM2 nor SAM available")
        SAM_AVAILABLE = False


class SAMPreprocessor:
    """SAM预处理器，用于为图像生成掩码"""
    
    def __init__(self, model_type="sam2", device="cuda", **kwargs):
        """
        初始化SAM预处理器
        
        Args:
            model_type: "sam2" 或 "sam"
            device: 计算设备
            **kwargs: 其他参数
        """
        self.device = device
        self.model_type = model_type
        
        if model_type == "sam2" and SAM2_AVAILABLE:
            self._init_sam2(**kwargs)
        elif model_type == "sam" and SAM_AVAILABLE:
            self._init_sam(**kwargs)
        else:
            raise ValueError(f"Model type {model_type} not available")
    
    def _init_sam2(self, config_file=None, ckpt_path=None, **kwargs):
        """初始化SAM2模型"""
        

        
        # 构建SAM2模型
        self.model = build_sam2(
            config_file=config_file,
            ckpt_path=ckpt_path,
            device=self.device,
            mode="eval"
        )
        
        self.model.eval()
        self.model.requires_grad_(False)
        
        # 创建SAM2AutomaticMaskGenerator
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=32,
            points_per_batch=640,
            pred_iou_thresh=0.1,
            stability_score_thresh=0.9,
            mask_threshold=0.0,
            box_nms_thresh=0.5,
            min_mask_region_area=10,
            output_mode="binary_mask"
        )
    
    def _init_sam(self, model_type="vit_h", ckpt_path=None, **kwargs):
        """初始化SAM模型"""
        if ckpt_path is None:
            # 使用绝对路径，指向src目录下的模型文件
            ckpt_path = os.path.join(os.path.dirname(__file__), "../src/sam_vit_h_4b8939.pth")
        
        # 构建SAM模型
        self.model = sam_model_registry[model_type](checkpoint=ckpt_path)
        self.model.to(device=self.device)
        self.model.eval()
        
        # 创建SamAutomaticMaskGenerator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.model,
            points_per_side=32,
            pred_iou_thresh=0.3,
            stability_score_thresh=0.95,
            mask_threshold=0.0,
            box_nms_thresh=0.3,
            min_mask_region_area=100,
            output_mode="binary_mask"
        )
    
    def generate_masks(self, image):
        """为图像生成掩码"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')

        # 将image转换为tensor
        image = np.array(image)
        
        # 生成掩码
        masks = self.mask_generator.generate(image)
        return masks
    
    def save_masks(self, masks, output_path):
        """保存掩码到文件，以numpy向量形式保存"""
        # 将masks转换为numpy数组
        if isinstance(masks, list) and len(masks) > 0:
            # 提取所有mask的segmentation数据
            mask_arrays = []
            for mask in masks:
                if 'segmentation' in mask:
                    mask_arrays.append(mask['segmentation'].astype(np.uint8))
            
            if mask_arrays:
                # 将所有mask堆叠成一个3D数组 (num_masks, height, width)
                mask_tensor = np.stack(mask_arrays, axis=0)
                np.save(output_path, mask_tensor)
            else:
                # 如果没有有效的mask，保存空数组
                np.save(output_path, np.array([]))
        else:
            # 如果没有mask，保存空数组
            np.save(output_path, np.array([]))
    
    def process_image_file(self, image_path, output_path):
        """处理单个图像文件"""
        try:
            # 生成掩码
            masks = self.generate_masks(image_path)
            
            # 保存掩码
            self.save_masks(masks, output_path)
            
            return True
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False
    
    def process_directory(self, input_dir, output_dir=None, image_extensions=('.jpg', '.jpeg', '.png')):
        """处理目录中的所有图像"""
        # 如果没有指定output_dir，则使用input_dir
        if output_dir is None:
            output_dir = input_dir
        
        # 查找所有图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
        
        print(f"Found {len(image_files)} images in {input_dir}")
        
        # 处理每个图像
        for image_file in tqdm(image_files, desc="Processing images"):
            # 使用与图像相同的名称，但扩展名为.npy
            output_name = image_file.stem + ".npy"
            output_path = os.path.join(output_dir, output_name)
            
            self.process_image_file(str(image_file), output_path)


def main():
    """主函数，用于测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SAM preprocessing")
    parser.add_argument("--input_dir", required=True, help="Input directory")
    parser.add_argument("--output_dir", help="Output directory (default: same as input_dir)")
    parser.add_argument("--model_type", default="sam2", choices=["sam2", "sam"], help="Model type")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--config_file", help="Config file (for SAM2)")
    parser.add_argument("--ckpt_path", help="Checkpoint path")
    
    args = parser.parse_args()
    
    # 创建预处理器
    preprocessor = SAMPreprocessor(
        model_type=args.model_type,
        device=args.device,
        config_file=args.config_file,
        ckpt_path=args.ckpt_path
    )
    
    # 处理目录
    preprocessor.process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main() 