#!/usr/bin/env python3
"""
Interactive Gaussian Splatting Viewer with Time Frame Selection
基于demo_GS.py，支持当前网络输出和时间帧选择
"""

import os
import sys
import numpy as np
import torch
import time
import threading
import argparse
from copy import deepcopy

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/vggt'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from vggt.models.vggt import VGGT
from dust3r.utils.misc import tf32_off
from src.dust3r.inference import inference
from src.train import cut3r_batch_to_vggt
from src.online_dynamic_processor import OnlineDynamicProcessor
import nerfview as nerfview
import viser
from gsplat import rasterization


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Interactive Gaussian Splatting Viewer")

    # 基础参数
    parser.add_argument("--model_path", type=str,
                       default="src/checkpoints/waymo_stage1_online/fromaggregator_all_lr1e-5_procrustes_area500_velocityconstraint0.05_gtcamera_xyzgrad+fixdbscan+sky+fixepsmetric+noconf/checkpoint-epoch_0_37980.pth",
                       help="Path to Stage1 model checkpoint")
    parser.add_argument("--dataset_root", type=str,
                       default="data/waymo/train_full/",
                       help="Path to dataset root directory")
    parser.add_argument("--idx", type=int, default=0, help="Sequence index")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--num_views", type=int, default=8, help="Number of views")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--fps", type=float, default=10.0, help="Playback frame rate (frames per second)")

    # Dynamic processor参数
    parser.add_argument("--velocity_transform_mode", type=str, default="procrustes",
                       choices=["simple", "procrustes"], help="Velocity transformation mode")
    parser.add_argument("--use_gt_camera", action="store_true", help="Use GT camera parameters")

    # 聚类参数
    parser.add_argument("--velocity_threshold", type=float, default=0.1,
                       help="Velocity threshold for clustering")
    parser.add_argument("--clustering_eps", type=float, default=0.3,
                       help="DBSCAN eps parameter (meters)")
    parser.add_argument("--clustering_min_samples", type=int, default=10,
                       help="DBSCAN min_samples parameter")
    parser.add_argument("--min_object_size", type=int, default=500,
                       help="Minimum object size (points)")
    parser.add_argument("--tracking_position_threshold", type=float, default=2.0,
                       help="Position threshold for tracking")
    parser.add_argument("--tracking_velocity_threshold", type=float, default=0.2,
                       help="Velocity threshold for tracking")

    # 可视化参数
    parser.add_argument("--conf_threshold", type=float, default=0.0,
                       help="Confidence threshold for filtering gaussians")

    return parser.parse_args()


def load_model(model_path, device):
    """加载Stage1模型"""
    print(f"Loading model from: {model_path}")

    model = VGGT(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        use_sky_token=True,
        sh_degree=0
    )

    checkpoint = torch.load(model_path, map_location="cpu")
    ckpt = checkpoint.get('model', checkpoint)
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model


def load_dataset(dataset_root, num_views):
    """加载数据集"""
    from src.dust3r.datasets.waymo import Waymo_Multi

    seq_name = os.path.basename(dataset_root)
    root_dir = os.path.dirname(dataset_root)

    print(f"Loading dataset - Root: {root_dir}, Sequence: {seq_name}")

    dataset = Waymo_Multi(
        split=None,
        ROOT=root_dir,
        img_ray_mask_p=[1.0, 0.0, 0.0],
        valid_camera_id_list=["1", "2", "3"],
        resolution=[(518, 378), (518, 336), (518, 294), (518, 252), (518, 210),
                    (518, 140), (378, 518), (336, 518), (294, 518), (252, 518)],
        num_views=num_views,
        seed=42,
        n_corres=0,
        seq_aug_crop=True
    )

    return dataset


class GaussianViewer:
    """交互式Gaussian Splatting查看器，支持时间帧选择"""

    def __init__(self, scene, vggt_batch, device, port=8080, fps=10.0):
        """
        Args:
            scene: 包含static_gaussians和dynamic_objects的场景字典
            vggt_batch: 包含相机参数等信息的batch
            device: 计算设备
            port: Viser服务器端口
            fps: 播放帧率
        """
        self.scene = scene
        self.vggt_batch = vggt_batch
        self.device = device
        self.fps = fps

        # 提取场景信息
        self.static_gaussians = scene.get('static_gaussians')  # [N, 14]
        self.dynamic_objects = scene.get('dynamic_objects', [])

        # 相机参数
        self.intrinsics = vggt_batch['intrinsics'][0]  # [S, 3, 3]
        self.extrinsics = vggt_batch['extrinsics'][0]  # [S, 4, 4]
        self.num_frames = self.intrinsics.shape[0]

        print(f"Scene info:")
        print(f"  - Static gaussians: {self.static_gaussians.shape[0] if self.static_gaussians is not None else 0}")
        print(f"  - Dynamic objects: {len(self.dynamic_objects)}")
        print(f"  - Number of frames: {self.num_frames}")

        # 启动Viser服务器
        print(f"Launching Gaussian viewer on port {port}...")
        self.server = viser.ViserServer(port=port, verbose=False)
        # self.server.scene.set_up_direction("+y")

        # 添加时间帧滑块
        self.frame_slider = self.server.gui.add_slider(
            "Time Frame",
            min=0,
            max=self.num_frames - 1,
            step=1,
            initial_value=0,
        )

        # 添加播放控制按钮
        self.play_button = self.server.gui.add_button("Play")
        self.pause_button = self.server.gui.add_button("Pause")

        # 添加FPS滑块
        self.fps_slider = self.server.gui.add_slider(
            "FPS",
            min=1.0,
            max=60.0,
            step=1.0,
            initial_value=fps,
        )

        # 添加动态物体显示开关
        self.show_dynamic_checkbox = self.server.gui.add_checkbox(
            "Show Dynamic Objects",
            initial_value=True,
        )

        # 添加静态物体显示开关
        self.show_static_checkbox = self.server.gui.add_checkbox(
            "Show Static Objects",
            initial_value=True,
        )

        # 播放状态
        self.is_playing = False
        self.playback_thread = None

        # 设置按钮回调
        @self.play_button.on_click
        def _(_):
            self.start_playback()

        @self.pause_button.on_click
        def _(_):
            self.stop_playback()

        # 创建viewer
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self._viewer_render_fn,
            mode="rendering",
        )

        print("Viewer ready!")


    def _collect_gaussians_for_frame(self, frame_idx):
        """收集指定帧的所有Gaussian参数"""
        all_means = []
        all_scales = []
        all_colors = []
        all_rotations = []
        all_opacities = []

        # Static gaussians
        if self.show_static_checkbox.value and self.static_gaussians is not None and self.static_gaussians.shape[0] > 0:
            static_gaussians = self.static_gaussians  # [N, 14]
            all_means.append(static_gaussians[:, :3])
            all_scales.append(static_gaussians[:, 3:6])
            all_colors.append(static_gaussians[:, 6:9].unsqueeze(-2))
            all_rotations.append(static_gaussians[:, 9:13])
            all_opacities.append(static_gaussians[:, 13])

        # Dynamic objects
        if self.show_dynamic_checkbox.value:
            for obj_data in self.dynamic_objects:
                # 检查物体是否在当前帧存在
                if not self._object_exists_in_frame(obj_data, frame_idx):
                    continue

                # 获取物体在正规空间的Gaussian参数
                canonical_gaussians = obj_data.get('canonical_gaussians')  # [N, 14]
                if canonical_gaussians is None or canonical_gaussians.shape[0] == 0:
                    continue

                # 获取从canonical空间到当前帧的变换
                frame_transform = self._get_object_transform_to_frame(obj_data, frame_idx)
                if frame_transform is None:
                    transformed_gaussians = canonical_gaussians
                else:
                    transformed_gaussians = self._apply_transform_to_gaussians(
                        canonical_gaussians, frame_transform
                    )

                # 添加到渲染列表
                if transformed_gaussians.shape[0] > 0:
                    all_means.append(transformed_gaussians[:, :3])
                    all_scales.append(transformed_gaussians[:, 3:6])
                    all_colors.append(transformed_gaussians[:, 6:9].unsqueeze(-2))
                    all_rotations.append(transformed_gaussians[:, 9:13])
                    all_opacities.append(transformed_gaussians[:, 13])

        if len(all_means) == 0:
            return None

        # Concatenate
        means = torch.cat(all_means, dim=0)  # [N, 3]
        scales = torch.cat(all_scales, dim=0)  # [N, 3]
        colors = torch.cat(all_colors, dim=0)  # [N, 1, 3]
        rotations = torch.cat(all_rotations, dim=0)  # [N, 4]
        opacities = torch.cat(all_opacities, dim=0)  # [N]

        # Fix NaN/Inf
        means = torch.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
        scales = torch.nan_to_num(scales, nan=0.01, posinf=1.0, neginf=0.01)
        colors = torch.nan_to_num(colors, nan=0.5, posinf=1.0, neginf=0.0)
        rotations = torch.nan_to_num(rotations, nan=0.0, posinf=1.0, neginf=-1.0)
        opacities = torch.nan_to_num(opacities, nan=0.5, posinf=1.0, neginf=0.0)

        return {
            'means': means,
            'scales': scales,
            'colors': colors,
            'rotations': rotations,
            'opacities': opacities
        }


    def _object_exists_in_frame(self, obj_data, frame_idx):
        """检查动态物体是否在指定帧中存在"""
        if 'frame_transforms' in obj_data:
            frame_transforms = obj_data['frame_transforms']
            if frame_idx in frame_transforms:
                return True
        return False


    def _get_object_transform_to_frame(self, obj_data, frame_idx):
        """获取从canonical空间到指定帧的变换矩阵"""
        reference_frame = obj_data.get('reference_frame', 0)
        if frame_idx == reference_frame:
            return None

        if 'frame_transforms' in obj_data:
            frame_transforms = obj_data['frame_transforms']
            if frame_idx in frame_transforms:
                frame_to_canonical = frame_transforms[frame_idx]
                canonical_to_frame = torch.inverse(frame_to_canonical)
                return canonical_to_frame

        return None


    def _apply_transform_to_gaussians(self, gaussians, transform):
        """将变换应用到Gaussian参数"""
        # gaussians: [N, 14] - [xyz(3), scale(3), color(3), quat(4), opacity(1)]
        # transform: [4, 4] 变换矩阵

        # 检查变换矩阵
        if torch.allclose(transform, torch.zeros_like(transform), atol=1e-6):
            transform = torch.eye(4, dtype=transform.dtype, device=transform.device)

        transformed_gaussians = gaussians.clone()

        # 变换位置
        positions = gaussians[:, :3]  # [N, 3]
        positions_homo = torch.cat([positions, torch.ones(
            positions.shape[0], 1, device=positions.device)], dim=1)  # [N, 4]
        transformed_positions = torch.mm(
            transform, positions_homo.T).T[:, :3]  # [N, 3]
        transformed_gaussians[:, :3] = transformed_positions

        return transformed_gaussians


    @torch.no_grad()
    def _viewer_render_fn(
        self,
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState
    ):
        """渲染回调函数"""
        # 获取渲染参数
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height

        # 获取相机参数
        c2w = torch.from_numpy(camera_state.c2w).float().to(self.device)
        K = torch.from_numpy(camera_state.get_K([width, height])).float().to(self.device)
        w2c = c2w.inverse()

        # 获取当前帧
        frame_idx = self.frame_slider.value

        # 收集当前帧的Gaussians
        gaussians = self._collect_gaussians_for_frame(frame_idx)

        if gaussians is None:
            # 返回白色图像
            return np.ones((height, width, 3), dtype=np.float32)

        # 渲染
        try:
            render_result = rasterization(
                gaussians['means'],
                gaussians['rotations'],
                gaussians['scales'],
                gaussians['opacities'],
                gaussians['colors'],
                w2c.unsqueeze(0),
                K.unsqueeze(0),
                width,
                height,
                sh_degree=0,
                render_mode="RGB+ED",
                radius_clip=0,
                near_plane=0.0001,
                far_plane=1000.0,
                eps2d=0.3,
                backgrounds=torch.ones(1, 3, device=self.device),  # 白色背景
            )

            rendered_image = render_result[0][0, :, :, :3]  # [H, W, 3]
            rendered_image = torch.clamp(rendered_image, min=0, max=1)

            return rendered_image.cpu().numpy()

        except Exception as e:
            print(f"Error rendering frame {frame_idx}: {e}")
            return np.zeros((height, width, 3), dtype=np.float32)


    def start_playback(self):
        """开始自动播放"""
        if self.is_playing:
            return

        self.is_playing = True
        print("Starting playback...")

        def playback_loop():
            while self.is_playing:
                current_fps = self.fps_slider.value
                sleep_time = 1.0 / current_fps

                # 更新帧
                current_frame = self.frame_slider.value
                next_frame = (current_frame + 1) % self.num_frames
                self.frame_slider.value = next_frame

                # 等待
                time.sleep(sleep_time)

        self.playback_thread = threading.Thread(target=playback_loop, daemon=True)
        self.playback_thread.start()


    def stop_playback(self):
        """停止自动播放"""
        if not self.is_playing:
            return

        self.is_playing = False
        print("Stopping playback...")

        if self.playback_thread is not None:
            self.playback_thread.join(timeout=1.0)
            self.playback_thread = None


def run_inference(args):
    """运行推理并返回场景数据"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load model and dataset
    model = load_model(args.model_path, device)
    dataset = load_dataset(args.dataset_root, args.num_views)

    # Create dynamic processor
    dynamic_processor = OnlineDynamicProcessor(
        device=device,
        velocity_transform_mode=args.velocity_transform_mode,
        velocity_threshold=args.velocity_threshold,
        clustering_eps=args.clustering_eps,
        clustering_min_samples=args.clustering_min_samples,
        min_object_size=args.min_object_size,
        tracking_position_threshold=args.tracking_position_threshold,
        tracking_velocity_threshold=args.tracking_velocity_threshold
    )

    print(f"\n{'='*60}")
    print(f"Processing sequence index: {args.idx}")
    print(f"{'='*60}\n")

    # Load data
    views = dataset.__getitem__((args.idx, 2, args.num_views))

    # Run Stage1 inference
    with torch.no_grad():
        outputs, batch = inference(views, model, device)

    # 转换为vggt batch
    vggt_batch = cut3r_batch_to_vggt(views)

    # Vggt forward
    with torch.no_grad():
        preds = model(
            vggt_batch['images'],
            gt_extrinsics=vggt_batch['extrinsics'],
            gt_intrinsics=vggt_batch['intrinsics'],
            frame_sample_ratio=1.0
        )

    # 处理use_gt_camera参数
    preds_for_dynamic = preds.copy() if isinstance(preds, dict) else preds
    if args.use_gt_camera and 'pose_enc' in preds_for_dynamic:
        from vggt.utils.pose_enc import extri_intri_to_pose_encoding
        gt_extrinsics = vggt_batch['extrinsics']
        gt_intrinsics = vggt_batch['intrinsics']
        image_size_hw = vggt_batch['images'].shape[-2:]

        gt_pose_enc = extri_intri_to_pose_encoding(
            gt_extrinsics, gt_intrinsics, image_size_hw, pose_encoding_type="absT_quaR_FoV"
        )
        preds_for_dynamic['pose_enc'] = gt_pose_enc
        print(f"[INFO] Using GT camera parameters for dynamic object processing")
    else:
        print(f"[INFO] Using predicted camera parameters for dynamic object processing")

    # 创建空的辅助模型字典
    auxiliary_models = {}

    # Process dynamic objects
    dynamic_objects_data = dynamic_processor.process_dynamic_objects(
        preds_for_dynamic, vggt_batch, auxiliary_models
    )

    # Build scene
    dynamic_objects = dynamic_objects_data.get('dynamic_objects', []) if dynamic_objects_data is not None else []
    static_gaussians = dynamic_objects_data.get('static_gaussians') if dynamic_objects_data is not None else None

    scene = {
        'static_gaussians': static_gaussians,
        'dynamic_objects': dynamic_objects
    }

    return scene, vggt_batch, device


def main():
    args = parse_args()

    # import debugpy
    # debugpy.listen(5697)
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()

    with tf32_off():
        # 运行推理
        scene, vggt_batch, device = run_inference(args)

        # 创建viewer
        viewer = GaussianViewer(scene, vggt_batch, device, port=args.port, fps=args.fps)

        print(f"\n{'='*60}")
        print(f"Viewer is running on http://localhost:{args.port}")
        print(f"Press Ctrl+C to exit")
        print(f"{'='*60}\n")

        # 保持运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting...")


if __name__ == "__main__":
    main()
