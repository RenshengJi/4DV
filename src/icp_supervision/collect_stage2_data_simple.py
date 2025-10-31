#!/usr/bin/env python3
"""
简化版ICP数据收集脚本

逐个样本处理，不使用batch
每次只处理一个场景，提取动态物体，生成ICP数据
"""

import os
import sys
import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

# 添加src到路径
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

# 导入需要的模块
from dust3r.datasets import get_data_loader
from dust3r.model import strip_module
from online_dynamic_processor import OnlineDynamicProcessor
from icp_supervision.data_generator import ICPDataGenerator

# 导入数据集类
from dust3r.datasets.waymo import Waymo_Multi
from dust3r.datasets.utils.transforms import ImgNorm

# 需要导入VGGT模型（因为eval会用到）
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'vggt'))
from vggt.vggt.models.vggt import VGGT

# 导入cut3r_batch_to_vggt函数 - 从train.py复制
from dust3r.utils.misc import tf32_off
import numpy as np


def cut3r_batch_to_vggt(views, device='cuda'):
    """将CUT3R的batch转换为VGGT格式 - 从train.py复制，添加numpy处理"""
    # Helper function to convert to tensor on the specified device
    def to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(device)
        elif isinstance(x, torch.Tensor):
            return x.to(device)
        return x

    # views: List[Dict], 长度为num_views
    # 目标: [1, S, 3, H, W] (B=1, S=num_views)
    imgs = [to_tensor(v['img']) for v in views]  # List of tensors
    imgs = torch.stack(imgs, dim=0)  # [S, ...]

    # 处理imgs的维度：可能是[S,3,H,W]或[S,B,3,H,W]
    if imgs.dim() == 4:  # [S,3,H,W] - 添加batch维度
        imgs = imgs.unsqueeze(1)  # [S,1,3,H,W]

    vggt_batch = {
        'images': imgs * 0.5 + 0.5,  # [S,B,3,H,W], 归一化到[0,1]
        'depths': torch.stack([to_tensor(v['depthmap']) for v in views], dim=0) if 'depthmap' in views[0] else None,
        'intrinsics': torch.stack([to_tensor(v['camera_intrinsics']) for v in views], dim=0) if 'camera_intrinsics' in views[0] else None,
        'extrinsics': torch.stack([to_tensor(v['camera_pose']) for v in views], dim=0) if 'camera_pose' in views[0] else None,
        'point_masks': torch.stack([to_tensor(v['valid_mask']) for v in views], dim=0) if 'valid_mask' in views[0] else None,
        'world_points': torch.stack([to_tensor(v['pts3d']) for v in views], dim=0) if 'pts3d' in views[0] else None,
        'flowmap': torch.stack([to_tensor(v['flowmap']) for v in views], dim=0) if 'flowmap' in views[0] and views[0]['flowmap'] is not None else None,
    }

    # 统一处理其他字段的维度
    for key in ['depths', 'intrinsics', 'extrinsics', 'point_masks']:
        if vggt_batch[key] is not None and vggt_batch[key].dim() == 3:
            # [S, H, W] -> [S, 1, H, W]
            vggt_batch[key] = vggt_batch[key].unsqueeze(1)

    with tf32_off(), torch.amp.autocast("cuda", enabled=False):
        # 转换world points的坐标系到第一帧相机坐标系
        if vggt_batch['world_points'] is not None:
            # world_points可能是4维或5维
            if vggt_batch['world_points'].dim() == 4:
                # [S, H, W, 3] - 添加batch维度
                vggt_batch['world_points'] = vggt_batch['world_points'].unsqueeze(1)  # [S, 1, H, W, 3]
            B, S, H, W, _ = vggt_batch['world_points'].shape
            world_points = vggt_batch['world_points'].reshape(B, S, H*W, 3)
            world_points = torch.matmul(torch.linalg.inv(vggt_batch['extrinsics'][0])[:, :3, :3], world_points.transpose(-1, -2)).transpose(-1, -2) + \
                                       torch.linalg.inv(vggt_batch['extrinsics'][0])[:, :3, 3:4].transpose(-1, -2)
            vggt_batch['world_points'] = world_points.reshape(B, S, H, W, 3)

            # 处理flowmap
            if vggt_batch['flowmap'] is not None:
                vggt_batch['flowmap'][..., :3] *=  0.1

            # 转换extrinsics的坐标系到第一帧相机坐标系
            vggt_batch['extrinsics'] = torch.matmul(
                    torch.linalg.inv(vggt_batch['extrinsics']),
                    vggt_batch['extrinsics'][0]
                )

            # 将extrinsics(中的T)以及world_points、depth进行非metric化
            world_points_flatten = vggt_batch['world_points'].reshape(-1, 3)
            world_points_mask_flatten = vggt_batch['point_masks'].reshape(-1).bool() if vggt_batch['point_masks'] is not None else torch.ones_like(world_points_flatten[:, 0], dtype=torch.bool)
            dist_avg = world_points_flatten[world_points_mask_flatten].norm(dim=-1).mean()
            depth_scale_factor = 1 / dist_avg
            pose_scale_factor = depth_scale_factor

            # 保存depth_scale_factor到batch中用于scale loss监督
            vggt_batch['depth_scale_factor'] = depth_scale_factor

            # 应用非metric化
            vggt_batch['depths'] = vggt_batch['depths'] * depth_scale_factor
            vggt_batch['extrinsics'][:, :, :3, 3] = vggt_batch['extrinsics'][:, :, :3, 3] * pose_scale_factor
            vggt_batch['world_points'] = vggt_batch['world_points'] * depth_scale_factor

            # 对flowmap应用非metric化：只对velocity magnitude进行缩放
            if vggt_batch['flowmap'] is not None:
                vggt_batch['flowmap'][..., :3] = vggt_batch['flowmap'][..., :3] * depth_scale_factor


    vggt_batch['images'] = vggt_batch['images'].permute(1, 0, 2, 3, 4).contiguous()
    vggt_batch['depths'] = vggt_batch['depths'].permute(1, 0, 2, 3).contiguous() if vggt_batch['depths'] is not None else None
    vggt_batch['intrinsics'] = vggt_batch['intrinsics'].permute(1, 0, 2, 3).contiguous() if vggt_batch['intrinsics'] is not None else None
    vggt_batch['extrinsics'] = vggt_batch['extrinsics'].permute(1, 0, 2, 3).contiguous() if vggt_batch['extrinsics'] is not None else None
    vggt_batch['point_masks'] = vggt_batch['point_masks'].permute(1, 0, 2, 3).contiguous() if vggt_batch['point_masks'] is not None else None
    vggt_batch['world_points'] = vggt_batch['world_points'].permute(1, 0, 2, 3, 4).contiguous() if vggt_batch['world_points'] is not None else None

    # flowmap处理：根据维度判断是否需要permute
    if vggt_batch['flowmap'] is not None:
        if vggt_batch['flowmap'].dim() == 5:
            vggt_batch['flowmap'] = vggt_batch['flowmap'].permute(1, 0, 2, 3, 4).contiguous()

    return vggt_batch


class SimpleStage2DataCollector:
    """简化的Stage2数据收集器 - 逐样本处理"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')

        print("="*60)
        print("Simple Stage2 Data Collector - 逐样本处理")
        print("="*60)
        print(f"Device: {self.device}")

        # 初始化模型
        self._init_model()

        # 初始化数据集（不用dataloader，直接用dataset）
        self._init_dataset()

        # 初始化动态物体处理器
        self._init_dynamic_processor()

        # 初始化ICP数据生成器
        self._init_icp_generator()

    def _init_model(self):
        """初始化Stage2模型"""
        print("\n[1/4] Initializing model...")

        # 加载模型
        print(f"  Loading model: {self.cfg.model}")
        self.model = eval(self.cfg.model)
        self.model.to(self.device)

        # 加载预训练权重
        if hasattr(self.cfg, 'pretrained_velocity') and self.cfg.pretrained_velocity:
            print(f"  Loading pretrained from: {self.cfg.pretrained_velocity}")
            checkpoint = torch.load(self.cfg.pretrained_velocity, map_location=self.device)
            ckpt = strip_module(checkpoint.get('model', checkpoint))
            self.model.load_state_dict(ckpt, strict=False)
            del ckpt, checkpoint

        self.model.eval()
        print(f"  ✓ Model loaded")

    def _init_dataset(self):
        """初始化数据集（直接使用dataset，不用dataloader）"""
        print("\n[2/4] Initializing dataset...")

        # 直接eval数据集字符串
        self.dataset = eval(self.cfg.train_dataset)

        print(f"  ✓ Dataset initialized: {len(self.dataset)} samples")

    def _init_dynamic_processor(self):
        """初始化动态物体处理器"""
        print("\n[3/4] Initializing dynamic processor...")

        self.dynamic_processor = OnlineDynamicProcessor(
            device=self.device,
            min_object_size=self.cfg.get('min_object_size', 100),
            max_objects_per_frame=self.cfg.get('max_objects_per_frame', 10),
            velocity_threshold=self.cfg.get('velocity_threshold', 0.1),
            clustering_eps=self.cfg.get('clustering_eps', 0.02),
            clustering_min_samples=self.cfg.get('clustering_min_samples', 10),
            tracking_position_threshold=self.cfg.get('tracking_position_threshold', 2.0),
            tracking_velocity_threshold=self.cfg.get('tracking_velocity_threshold', 0.2),
            use_optical_flow_aggregation=self.cfg.get('use_optical_flow_aggregation', True),
            use_velocity_based_transform=self.cfg.get('use_velocity_based_transform', True),
        )

        print(f"  ✓ Dynamic processor initialized")

    def _init_icp_generator(self):
        """初始化ICP数据生成器"""
        print("\n[4/4] Initializing ICP generator...")

        output_dir = self.cfg.get('icp_data_dir', './icp_supervision_data_real')
        use_color_features = self.cfg.get('use_color_features', False)
        voxel_size = self.cfg.get('voxel_size', 0.01)
        ransac_max_iteration = self.cfg.get('ransac_max_iteration', 5000)
        ransac_confidence = self.cfg.get('ransac_confidence', 5)

        self.icp_generator = ICPDataGenerator(
            output_dir=output_dir,
            min_frames_per_object=2,
            max_frames_per_object=5,
            save_pointcloud_files=True,
            use_color_features=use_color_features,
            voxel_size=voxel_size,
            ransac_max_iteration=ransac_max_iteration,
            ransac_confidence=ransac_confidence,
        )

        print(f"  ✓ ICP generator initialized")
        print(f"  Output directory: {output_dir}")
        print(f"  Use color features: {use_color_features}")
        print(f"  Voxel size: {voxel_size}")
        print(f"  RANSAC max iteration: {ransac_max_iteration}")
        print(f"  RANSAC confidence: {ransac_confidence}")

    def process_one_sample(self, sample_idx):
        """
        处理单个样本

        Args:
            sample_idx: 样本索引

        Returns:
            num_objects: 收集到的objects数量
        """
        # 获取样本 - sample是tuple of dicts
        sample = self.dataset[sample_idx]

        # 将样本移到GPU
        sample_gpu = []
        for view in sample:
            view_gpu = {}
            for key, value in view.items():
                if isinstance(value, torch.Tensor):
                    view_gpu[key] = value.to(self.device, non_blocking=True)
                else:
                    view_gpu[key] = value
            sample_gpu.append(view_gpu)

        # 使用cut3r_batch_to_vggt转换batch格式 - 完全参考train.py
        with torch.no_grad():
            try:
                # 将sample_gpu转换为vggt_batch格式
                vggt_batch = cut3r_batch_to_vggt(sample_gpu, device=self.device)

                # 调用模型 - 参考train.py
                preds = self.model(
                    vggt_batch["images"],
                    gt_extrinsics=vggt_batch.get("extrinsics"),
                    gt_intrinsics=vggt_batch.get("intrinsics"),
                    frame_sample_ratio=0.25
                )

                # 【打补丁】修复velocity map前三行的异常高速度问题
                # Stage1的VGGT输出的velocity前三行容易出现异常，直接置零
                if 'velocity' in preds and preds['velocity'] is not None:
                    velocity = preds['velocity']  # [B, S, H, W, 3]
                    if velocity.dim() == 5 and velocity.shape[2] >= 3:
                        # 将前三行的速度向量置为0 (所有3个分量: vx, vy, vz)
                        velocity[:, :, :5, :, :] = 0.0  # TODO: 解决丑陋的补丁
                        # 更新preds（如果velocity不是inplace修改）
                        preds['velocity'] = velocity

            except Exception as e:
                print(f"  ✗ Model forward failed: {e}")
                import traceback
                traceback.print_exc()
                return 0

            # 处理动态物体
            try:
                dynamic_result = self.dynamic_processor.process_dynamic_objects(
                    preds=preds,
                    vggt_batch=vggt_batch,
                    auxiliary_models={}  # 不使用辅助模型
                )

                dynamic_objects = dynamic_result.get('dynamic_objects', [])

            except Exception as e:
                print(f"  ✗ Dynamic processing failed: {e}")
                import traceback
                traceback.print_exc()
                return 0

        # 转换为CPU
        cpu_objects = self._convert_to_cpu(dynamic_objects)

        # 转换preds到CPU
        cpu_preds = {}
        for key, value in preds.items():
            if isinstance(value, torch.Tensor):
                cpu_preds[key] = value.detach().cpu()
            else:
                cpu_preds[key] = value

        # 立即生成ICP数据
        if len(cpu_objects) > 0:
            print(f"  Found {len(cpu_objects)} dynamic objects in sample {sample_idx}")
            self._generate_icp_for_objects(cpu_objects, cpu_preds, sample_idx)

        return len(cpu_objects)

    def _views_to_batch(self, views):
        """将views列表转换为batch格式"""
        # views是list of dicts
        # 需要转换为 {'images': [B, S, 3, H, W], ...}

        batch = {}

        # 堆叠images
        if 'img' in views[0]:
            # views中每个dict的'img'可能是tensor或list
            imgs = []
            for v in views:
                img = v['img']
                if isinstance(img, list):
                    # 如果是list，转换为tensor
                    img = torch.stack(img, dim=0) if all(isinstance(x, torch.Tensor) for x in img) else img
                imgs.append(img)

            # 现在imgs是list of tensors
            # 每个tensor是[3, H, W]或者已经是合并的
            # 我们需要构造[B=1, S, 3, H, W]格式
            if len(imgs) > 0 and isinstance(imgs[0], torch.Tensor):
                if imgs[0].dim() == 3:  # [3, H, W]
                    # 每个view一张图，stack成[num_views, 3, H, W]，然后加batch维度
                    stacked = torch.stack(imgs, dim=0)  # [num_views, 3, H, W]
                    batch['images'] = stacked.unsqueeze(0)  # [1, num_views, 3, H, W]
                elif imgs[0].dim() == 4:  # [S, 3, H, W]
                    # 多帧数据，取第一个
                    batch['images'] = imgs[0].unsqueeze(0)  # [1, S, 3, H, W]

        return batch

    def _convert_to_cpu(self, dynamic_objects):
        """将dynamic_objects转换为CPU张量"""
        cpu_objects = []
        for obj in dynamic_objects:
            cpu_obj = {}
            for key, value in obj.items():
                if isinstance(value, torch.Tensor):
                    cpu_obj[key] = value.detach().cpu()
                elif isinstance(value, dict):
                    # 处理frame_gaussians等嵌套字典
                    cpu_dict = {}
                    for k, v in value.items():
                        if isinstance(v, torch.Tensor):
                            cpu_dict[k] = v.detach().cpu()
                        else:
                            cpu_dict[k] = v
                    cpu_obj[key] = cpu_dict
                else:
                    cpu_obj[key] = value
            cpu_objects.append(cpu_obj)
        return cpu_objects

    def _generate_icp_for_objects(self, dynamic_objects, preds, sample_idx):
        """为收集到的objects生成ICP数据"""
        try:
            # 从preds获取pred_scale
            pred_scale = preds.get('scale', 1.0)
            if isinstance(pred_scale, torch.Tensor):
                pred_scale = float(pred_scale.item())

            print(f"  Processing {len(dynamic_objects)} dynamic objects")

            # 直接遍历dynamic_objects，不需要中间的extract步骤
            for obj_idx, obj_data in enumerate(dynamic_objects):
                object_id = obj_data.get('object_id', obj_idx)

                # 获取canonical_gaussians作为input (粗糙的光流聚合结果)
                canonical_gaussians = obj_data.get('canonical_gaussians')
                if canonical_gaussians is None:
                    print(f"    Object {object_id}: no canonical_gaussians, skipping")
                    continue

                # 转换为numpy以检查维度
                if isinstance(canonical_gaussians, torch.Tensor):
                    canonical_gaussians = canonical_gaussians.detach().cpu().numpy()

                # 检查点数：跳过点数少于5000的物体
                num_points = canonical_gaussians.shape[0]
                if num_points < 5000:
                    print(f"    Object {object_id}: only {num_points} points (< 5000), skipping")
                    continue

                print(f"    Object {object_id}: {num_points} points")

                # 获取每帧的frame_gaussians
                frame_gaussians_dict = obj_data.get('frame_gaussians', {})
                if not frame_gaussians_dict:
                    print(f"    Object {object_id}: no frame_gaussians, skipping")
                    continue

                # 检查帧数：只有一帧时无需refine，直接跳过
                available_frames = sorted(frame_gaussians_dict.keys())
                if len(available_frames) <= 1:
                    print(f"    Object {object_id}: only {len(available_frames)} frame(s), no need to refine, skipping")
                    continue

                # 构建frame_data_list
                frame_data_list = []
                for frame_idx in available_frames[:self.icp_generator.max_frames_per_object]:
                    frame_gaussian = frame_gaussians_dict[frame_idx]
                    if isinstance(frame_gaussian, torch.Tensor):
                        frame_gaussian = frame_gaussian.detach().cpu().numpy()

                    frame_data_list.append({
                        'gaussians': frame_gaussian,
                        'frame_idx': frame_idx
                    })

                print(f"    Object {object_id}: {len(frame_data_list)} frames")

                # Run ICP on this object's frames to get refined target
                # 传入pred_scale用于尺度转换
                all_aligned_points = self.icp_generator.run_icp_on_object_frames(
                    object_id, frame_data_list, pred_scale=pred_scale
                )

                # all_aligned_points是一个列表，需要合并成单个numpy array
                if all_aligned_points is None or len(all_aligned_points) == 0:
                    print(f"    Object {object_id}: ICP failed, skipping")
                    continue

                # 将列表中的所有点云合并
                merged_points = np.concatenate(all_aligned_points, axis=0)  # [total_points, 3] - in metric scale

                # 检查点数是否匹配
                if merged_points.shape[0] != canonical_gaussians.shape[0]:
                    print(f"    Object {object_id}: point count mismatch "
                          f"(merged: {merged_points.shape[0]}, canonical: {canonical_gaussians.shape[0]}), skipping")
                    continue

                # 将metric尺度的点转换回归一化尺度
                # merged_points 当前是 metric scale，需要转换回 normalized scale
                merged_points_normalized = merged_points * pred_scale
                print(f"    Converting ICP results back to normalized scale: points * {pred_scale}")

                target_gaussians = canonical_gaussians.copy()
                target_gaussians[:, :3] = merged_points_normalized

                # Save the ICP result if successful
                if target_gaussians is not None:
                    self.icp_generator.save_sample_pair(
                        object_id=object_id,
                        input_gaussians=canonical_gaussians,  # 使用canonical作为input
                        target_gaussians=target_gaussians,     # ICP refined作为target
                        pred_scale=pred_scale,
                        sample_idx=self.icp_generator.stats['total_samples_generated']
                    )
                    print(f"    ✓ Saved ICP sample for object {object_id}")

        except Exception as e:
            print(f"  ✗ ICP generation failed for sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()

    def run(self, max_samples=100, sample_interval=50):
        """
        逐样本运行数据收集

        Args:
            max_samples: 最多处理多少个样本
            sample_interval: 样本采集间隔，避免密集连续采集
        """
        print("\n" + "="*60)
        print(f"Starting data collection ({max_samples} samples, interval={sample_interval})")
        print("="*60 + "\n")

        total_samples = min(max_samples, len(self.dataset))
        total_objects = 0

        # 使用间隔采样
        sample_indices = range(0, len(self.dataset), sample_interval)[:total_samples]

        for idx, sample_idx in enumerate(tqdm(sample_indices, desc="Processing samples")):
            if idx < 1500:
                continue
            try:
                num_objects = self.process_one_sample(sample_idx)
                total_objects += num_objects

                if (idx + 1) % 10 == 0:
                    print(f"\n  Sample {idx + 1}/{len(sample_indices)} (dataset index {sample_idx}): "
                          f"Collected {num_objects} objects "
                          f"(Total: {total_objects})")

            except Exception as e:
                print(f"\n  ✗ Error in sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print("\n" + "="*60)
        print(f"✓ Data collection complete!")
        print(f"  Total samples processed: {len(sample_indices)}")
        print(f"  Total objects collected: {total_objects}")
        print(f"  Total ICP samples: {self.icp_generator.stats.get('successful_samples', 0)}")
        print("="*60 + "\n")

        return total_objects


@hydra.main(config_path="../../config/waymo", config_name="stage2_icp_collect", version_base=None)
def main(cfg: DictConfig):
    """主函数"""

    print("\nConfiguration:")
    print("-" * 60)
    print(f"Max samples: {cfg.get('icp_max_batches', 100)}")
    print(f"Sample interval: {cfg.get('sample_interval', 50)}")
    print(f"Output dir: {cfg.get('icp_data_dir', './icp_supervision_data_real')}")
    print("-" * 60 + "\n")

    # 创建收集器
    collector = SimpleStage2DataCollector(cfg)

    # 逐样本收集数据
    max_samples = cfg.get('icp_max_batches', 100)
    sample_interval = cfg.get('sample_interval', 50)
    total_objects = collector.run(max_samples=max_samples, sample_interval=sample_interval)

    print("\n" + "="*60)
    print("✓ All done!")
    print("="*60)
    print(f"\nResults:")
    print(f"  - Total objects: {total_objects}")
    print(f"  - ICP data saved to: {cfg.get('icp_data_dir', './icp_supervision_data_real')}")
    print()


if __name__ == "__main__":
    main()
