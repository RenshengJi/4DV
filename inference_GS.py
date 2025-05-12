import os
import sys
import numpy as np
import torch
from torchvision.utils import save_image
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
import json

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
        default="/data/yuxue.yang/fl/zq/4DVideo/src/checkpoint-epoch_0_11934.pth",
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
        "--output_dir",
        type=str,
        default="./demo_tmp",
        help="value for tempfile.tempdir",
    )
    parser.add_argument(
        "--revisit",
        type=int,
        default=1,
        help="Number of times to revisit each view during inference.",
    )
    

    return parser.parse_args()


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


def run_inference(args):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    # Add the checkpoint path (required for model imports in the dust3r package).
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

    # Import model and inference functions after adding the ckpt path.
    from src.dust3r.inference import inference, inference_recurrent
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.utils.metrics import compute_lpips, compute_psnr, compute_ssim
    from viser_utils import PointCloudViewer

    from src.dust3r.datasets.waymo import Waymo_Multi
    dataset = Waymo_Multi(allow_repeat=True, split=None, ROOT="../data/dust3r_data/processed_waymo/", img_ray_mask_p=[1.0, 0.0, 0.0], aug_crop=16, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 208), (512, 144), (384, 512), (336, 512), (288, 512), (256, 512)], num_views=64, n_corres=0)
    # Prepare input views.
    print("Preparing input views...")
    idx = 1
    num_views = 64
    views = dataset.__getitem__((idx, 2, num_views))
    

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()

    # Run inference.
    print("Running inference...")
    start_time = time.time()
    outputs, state_args = inference(views, model, device)
    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
    )

    # Process outputs for visualization.
    print("Preparing output for visualization...")

   
    relative_path = os.path.relpath(args.model_path, "src/checkpoints")
    output_path = os.path.join(args.output_dir, relative_path.replace(".pth", ""))
    os.makedirs(output_path, exist_ok=True)
    metrics = []
    # Prepare lists to store frames for each type of view
    self_view_frames = []
    other_view_frames = []
    gt_frames = []

    for i in range(len(outputs["pred"])):
        # Convert tensors to numpy arrays and append to respective lists
        self_view_frames.append(
            (outputs["pred"][i]["render_from_self_view"][0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        other_view_frames.append(
            (outputs["pred"][i]["render_in_other_view"][0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        gt_frames.append(
            ((outputs['views'][i]['img'][0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
        )
        metrics.append(
            {
                "psnr": compute_psnr(
                    outputs['views'][i]['img'] * 0.5 + 0.5,
                    outputs["pred"][i]["render_in_other_view"],
                ).item(),
                "ssim": compute_ssim(
                    outputs['views'][i]['img'] * 0.5 + 0.5,
                    outputs["pred"][i]["render_in_other_view"],
                ).item(),
                "lpips": compute_lpips(
                    outputs['views'][i]['img'] * 0.5 + 0.5,
                    outputs["pred"][i]["render_in_other_view"],
                ).item(),
            }
        )

    # Define video output paths
    self_view_video_path = os.path.join(output_path, "self_view.mp4")
    other_view_video_path = os.path.join(output_path, "other_view.mp4")
    gt_video_path = os.path.join(output_path, "gt.mp4")

    # Save frames as videos
    def save_video(frames, video_path, fps=30):
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()

    save_video(self_view_frames, self_view_video_path)
    save_video(other_view_frames, other_view_video_path)
    save_video(gt_frames, gt_video_path)
        

    summary = {
        "average_psnr": np.mean([m["psnr"] for m in metrics]),
        "average_ssim": np.mean([m["ssim"] for m in metrics]),
        "average_lpips": np.mean([m["lpips"] for m in metrics]),
        "metrics": metrics,
    }
    with open(os.path.join(output_path, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=4)


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
