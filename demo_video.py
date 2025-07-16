import os
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
from copy import deepcopy
from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/vggt'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from vggt.models.vggt import VGGT
from dust3r.utils.misc import tf32_off
from training.loss import cross_render_and_loss
# Import model and inference functions after adding the ckpt path.
from src.dust3r.inference import inference

import re
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from accelerate.logging import get_logger
printer = get_logger(__name__, log_level="DEBUG")

# Set random seed for reproducibility.
random.seed(42)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo/self/checkpoint-epoch_0_6828.pth",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--model_velocity_path",
        type=str,
        default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo/flow-samweight1/checkpoint-epoch_0_8344.pth",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="512",
        help="Shape that input images will be rescaled to; if using 224+linear model, choose 224 otherwise 512",
    )
    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=2,
        help="Visualization threshold for the point cloud viewer. Ranging from 1 to INF",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_sam_8344",
        help="value for tempfile.tempdir",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=1000,
        help="Index of the video frame to process. If using a video file, this is the frame index to start from.",
    )

    return parser.parse_args()


def depth_to_world_points(depth, intrinsic):
    """
    将深度图转换为世界坐标系下的3D点

    参数:
    depth: [N, H, W, 1] 深度图(单位为米)
    intrinsic: [1, N, 3, 3] 相机内参矩阵

    返回:
    world_points: [N, H, W, 3] 世界坐标点(x,y,z)
    """
    with tf32_off():
        N, H, W, _ = depth.shape

        # 生成像素坐标网格 (u,v,1)
        v, u = torch.meshgrid(torch.arange(H, device=depth.device),
                             torch.arange(W, device=depth.device),
                             indexing='ij')
        uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1)  # [H, W, 3]
        uv1 = uv1.unsqueeze(0).expand(N, -1, -1, -1)  # [N, H, W, 3]
        # uv1 -> float32
        uv1 = uv1.float()

        # 转换为相机坐标 (X,Y,Z)
        depth = depth.squeeze(-1)  # [N, H, W]
        intrinsic = intrinsic.squeeze(0)  # [N, 3, 3]

        # 计算相机坐标: (u,v,1) * depth / fx,fy,1
        # 需要处理批量维度
        camera_points = torch.einsum('nij,nhwj->nhwi', torch.inverse(intrinsic), uv1)  # [N, H, W, 3]
        camera_points = camera_points * depth.unsqueeze(-1)  # [N, H, W, 3]

    return camera_points



def prepare_output(preds, vggt_batch):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.

    Returns:
        tuple: (points, colors, confidence, camera parameters dictionary)
    """
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    conf = preds["depth_conf"] > 10
    interval = 2


    # metric depth 
    _, img_dict = cross_render_and_loss(conf, interval, None, None, preds["depth"].detach(), preds["gaussian_params"], preds["velocity"], preds["pose_enc"], vggt_batch["extrinsics"], vggt_batch["intrinsics"], vggt_batch["images"], vggt_batch["depths"], vggt_batch["point_masks"])
    

    return img_dict



def parse_seq_path(p):
    if os.path.isdir(p):
        img_paths = sorted(glob.glob(f"{p}/*"))
        tmpdirname = None
    else:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        frame_interval = 1
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
    return img_paths, tmpdirname



def run_inference(dataset, model, device, args):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    
    # Prepare input views.
    print("Preparing input views...")
    idx = args.idx
    num_views = 24
    views = dataset.__getitem__((idx, 2, num_views))

    

    # Run inference.
    print("Running inference...")
    start_time = time.time()
    outputs, batch = inference(views, model, device)
    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
    )

    # Process outputs for visualization.
    print("Preparing output for visualization...")
    img_dict = prepare_output(outputs, batch)

    # img_dict -> video
    img_dict = deepcopy(img_dict)
    # img的类型为tensor
    for key in img_dict:
        if isinstance(img_dict[key], torch.Tensor):
            img_dict[key] = img_dict[key].cpu().numpy()
        elif isinstance(img_dict[key], list):
            img_dict[key] = [img.cpu().numpy() if isinstance(img, torch.Tensor) else img for img in img_dict[key]]
        else:
            raise TypeError(f"Unsupported type {type(img_dict[key])} in img_dict[{key}]")
    
    def normalize_to_uint8(arr):
        arr = arr.astype(np.float32)
        # vmin, vmax = np.percentile(arr, 2), np.percentile(arr, 98)
        vmin = 0
        vmax = 2
        arr = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0, 1)
        return (arr * 255).astype(np.uint8)

    # 将两个深度图从灰度图转换为RGB图(红近, 蓝远)
    img_dict["target_depth_pred"] = np.stack([
    cv2.applyColorMap(
            normalize_to_uint8(img_dict["target_depth_pred"][i][0]), cv2.COLORMAP_JET
        ).transpose(2, 0, 1)
        for i in range(len(img_dict["target_depth_pred"]))
    ], axis=0)
    img_dict["target_depth_gt"] = np.stack([
        cv2.applyColorMap(
            normalize_to_uint8(img_dict["target_depth_gt"][i][0]), cv2.COLORMAP_JET
        ).transpose(2, 0, 1)
        for i in range(len(img_dict["target_depth_gt"]))
    ], axis=0)

    
    # 将其他的rgb图*255转换为int
    for key in ["source_rgb", "target_rgb_pred", "target_rgb_gt", "velocity"]:
        img_dict[key] = np.stack([
            (img_dict[key][i] * 255).astype(np.uint8) for i in range(len(img_dict[key]))
        ], axis=0)
    

    # 将6种img_dict拼接为video
    video_path = os.path.join(args.output_dir, str(args.idx) + "_" + views[0]['label'].split('.')[0] + ".mp4")
    with iio.get_writer(video_path, fps=10) as writer:
        max_length = max(len(img_dict["source_rgb"]), len(img_dict["target_rgb_pred"]),
                         len(img_dict["target_rgb_gt"]), len(img_dict["target_depth_pred"]),
                         len(img_dict["target_depth_gt"]), len(img_dict["velocity"]))
        for i in range(max_length):
            frame = []
            for key in img_dict:
                if i < len(img_dict[key]):
                    frame.append(img_dict[key][i])
                else:
                    frame.append(np.zeros_like(img_dict[key][0]))
            combined_frame = np.concatenate(frame, axis=1)
            writer.append_data(combined_frame.transpose(1, 2, 0))  # Transpose to HWC format for video writer
    print(f"Output video saved to {video_path}")

    # # 将source_rgb和velocity拼接为video
    # video_path = os.path.join(args.output_dir, str(args.idx) + "_" + views[0]['label'].split('.')[0] + ".mp4")
    # with iio.get_writer(video_path, fps=10) as writer:
    #     for i in range(len(img_dict["source_rgb"])):
    #         frame = np.concatenate([img_dict["source_rgb"][i], img_dict["velocity"][i]], axis=1)
    #         writer.append_data(frame.transpose(1, 2, 0))
    # print(f"Output video saved to {video_path}")


def main():

    args = parse_args()

    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    # Add the checkpoint path (required for model imports in the dust3r package).
    add_path_to_dust3r(args.model_path)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    from src.dust3r.datasets.waymo import Waymo_Multi
    dataset = Waymo_Multi(allow_repeat=False, split=None, ROOT="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train", img_ray_mask_p=[1.0, 0.0, 0.0], aug_crop=16, resolution=[(518, 378),(518, 336),(518, 294),(518, 252),(518, 210),(518, 140),(378, 518),(336, 518),(294, 518),(252, 518)], num_views=24, n_corres=0, seq_aug_crop=True)


    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    model = VGGT(img_size=518, patch_size=14, embed_dim=1024)
    ckpt = torch.load(args.model_path, map_location=device)['model']
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    del ckpt
    ckpt = torch.load(args.model_velocity_path, map_location=device)['model']
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    ckpt = {k.replace("velocity_head.", ""): v for k, v in ckpt.items()}
    model.velocity_head.load_state_dict(ckpt, strict=False)
    del ckpt


    model.eval()
    model = model.to(device)
    
    idx = 0
    while True:
        print(f"\n========== Running inference for idx={idx} ==========")
        args.idx = idx
        try:
            run_inference(dataset, model, device, args)
        except Exception as e:
            print(f"Error at idx={idx}: {e}")
        idx += 200

if __name__ == "__main__":
    main()
