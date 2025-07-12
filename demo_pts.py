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
        default="/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo/step2(fix_mask)/checkpoint-epoch_0_42143.pth",
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
        default="./demo_tmp",
        help="value for tempfile.tempdir",
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


    # metric depth 
    pred_depth_metric = preds["depth"]
    gt_intrinsic = vggt_batch["intrinsics"]
    gt_extrinsic = vggt_batch["extrinsics"]


    # no metric 
    pred_depth = preds["depth_init"]
    pred_pose = preds["pose_enc"]
    pred_extrinsic, pred_intrinsic= pose_encoding_to_extri_intri(pred_pose, vggt_batch["images"].shape[-2:])
    pred_extrinsic = torch.cat([pred_extrinsic, torch.tensor([0, 0, 0, 1], device=pred_extrinsic.device)[None,None,None,:].repeat(1,pred_extrinsic.shape[1],1,1)], dim=-2)


    # 通过pred_extrinsic和gt_extrinsic中的T部分计算当前sence的scale
    pred_extrinsic_T = pred_extrinsic.squeeze(0)[:, :3, 3]  # [S, 3]
    gt_extrinsic_T = gt_extrinsic.squeeze(0)[:, :3, 3]  # [S, 3]
    scales = torch.norm(gt_extrinsic_T, dim=-1) / torch.norm(pred_extrinsic_T, dim=-1)  # [S]
    scale = scales[1:].mean()  # 平均缩放因子

    # 将pred_depth缩放到metric depth
    pred_depth_metric = pred_depth * scale
    

    images = vggt_batch["images"]
    velocity = preds["velocity"]
    B, S, _, image_height, image_width = images.shape
    depth = pred_depth_metric.view(pred_depth_metric.shape[0]*pred_depth_metric.shape[1], pred_depth_metric.shape[2], pred_depth_metric.shape[3], 1)
    world_points = depth_to_world_points(depth, gt_intrinsic)
    world_points = world_points.view(world_points.shape[0], world_points.shape[1]*world_points.shape[2], 3)
    extrinsic_inv = torch.linalg.inv(gt_extrinsic)
    xyz = torch.matmul(extrinsic_inv[0,:,:3,:3] , world_points.transpose(-1,-2)).transpose(-1,-2) + \
        extrinsic_inv[0,:,:3,3:4].transpose(-1,-2)
    pts3ds = xyz.reshape(xyz.shape[0], 1, image_height, image_width, 3)  # [B, H, W, 3]
    conf = preds["depth_init_conf"]


    # gt_intrinsic, gt_extrinsic -> cam_dict
    gt_intrinsic = gt_intrinsic.squeeze(0)  # [S, 3, 3]
    gt_extrinsic = gt_extrinsic.squeeze(0)  # [S, 4, 4]
    R_c2w = gt_extrinsic[:, :3, :3]  # [S, 3, 3]
    t_c2w = gt_extrinsic[:, :3, 3]  # [S, 3]
    focal = gt_intrinsic[:, 0, 0]  # [S]
    pp = gt_intrinsic[:, 0:2, 2]  # [S, 2]

    cam_dict = {
        "focal": focal.cpu().numpy(),
        "pp": pp.cpu().numpy(),
        "R": R_c2w.cpu().numpy(),
        "t": t_c2w.cpu().numpy(),
    }

    images = images.squeeze(0).permute(0, 2, 3, 1)  # [B, H, W, 3]
    conf = conf.reshape(conf.shape[1], conf.shape[0], conf.shape[2], conf.shape[3])


    pts3ds = pts3ds.reshape(pts3ds.shape[0], pts3ds.shape[1], pts3ds.shape[2]*pts3ds.shape[3], 3)  # [B, 1, H*W, 3]
    images = images.reshape(images.shape[0], images.shape[1]*images.shape[2], 3)  # [B, H*W, 3]
    conf = conf.reshape(conf.shape[0], conf.shape[1], conf.shape[2]*conf.shape[3])  # [B, 1, H*W]


    depth_gt = vggt_batch["depths"]
    depth_gt = depth_gt.reshape(depth_gt.shape[0]*depth_gt.shape[1], depth_gt.shape[2], depth_gt.shape[3], 1)
    world_points_gt = depth_to_world_points(depth_gt, gt_intrinsic)
    world_points_gt = world_points_gt.reshape(world_points_gt.shape[0], world_points_gt.shape[1]*world_points_gt.shape[2], 3)
    world_points_gt = torch.matmul(extrinsic_inv[0,:,:3,:3] , world_points_gt.transpose(-1,-2)).transpose(-1,-2) + \
        extrinsic_inv[0,:,:3,3:4].transpose(-1,-2)
    world_points_gt = world_points_gt.reshape(world_points_gt.shape[0], 1, world_points_gt.shape[1], 3)  # [B, 1, H*W, 3]


    images_gt = torch.zeros_like(images)
    images_gt[:, :, 0] = 1  # Set red channel to
    conf_gt = vggt_batch["point_masks"].permute(1, 0, 2, 3)
    conf_gt = conf_gt.reshape(conf_gt.shape[0], conf_gt.shape[1], conf_gt.shape[2]*conf_gt.shape[3]) * 100

    
    pts3ds = torch.cat([pts3ds, world_points_gt], dim=2)  # [B, 1, 2*H*W, 3]
    images = torch.cat([images, images_gt], dim=1)  # [B, 2*H*W, 3]
    conf = torch.cat([conf, conf_gt], dim=2)  # [B, 1, 2*H*W]


    pts3ds = [pts3d.cpu() for pts3d in pts3ds]
    images = [image.cpu() for image in images]
    conf = [c.cpu() for c in conf]

    

    return pts3ds, images, conf, cam_dict



# def prepare_output(outputs, outdir, revisit=1, use_pose=True):
#     """
#     Process inference outputs to generate point clouds and camera parameters for visualization.

#     Args:
#         outputs (dict): Inference outputs.
#         revisit (int): Number of revisits per view.
#         use_pose (bool): Whether to transform points using camera pose.

#     Returns:
#         tuple: (points, colors, confidence, camera parameters dictionary)
#     """
#     from src.dust3r.utils.camera import pose_encoding_to_camera
#     from src.dust3r.post_process import estimate_focal_knowing_depth
#     from src.dust3r.utils.geometry import geotrf

#     # Only keep the outputs corresponding to one full pass.
#     valid_length = len(outputs["pred"]) // revisit
#     outputs["pred"] = outputs["pred"][-valid_length:]
#     outputs["views"] = outputs["views"][-valid_length:]

#     pts3ds_self_ls = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]

#     pts3ds_other = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]
#     conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
#     conf_other = [output["conf"].cpu() for output in outputs["pred"]]
#     pts3ds_self = torch.cat(pts3ds_self_ls, 0)

#     # Recover camera poses.
#     pr_poses = [
#         pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
#         for pred in outputs["pred"]
#     ]
#     R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
#     t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

#     if use_pose:
#         transformed_pts3ds_other = []
#         for pose, pself in zip(pr_poses, pts3ds_self):
#             transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
#         pts3ds_other = transformed_pts3ds_other
#         conf_other = conf_self

#     # Estimate focal length based on depth.
#     B, H, W, _ = pts3ds_self.shape
#     pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
#     focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

#     colors = [
#         0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
#     ]

#     cam_dict = {
#         "focal": focal.cpu().numpy(),
#         "pp": pp.cpu().numpy(),
#         "R": R_c2w.cpu().numpy(),
#         "t": t_c2w.cpu().numpy(),
#     }


#     return pts3ds_other, colors, conf_other, cam_dict


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
    add_path_to_dust3r(args.model_path)

    # Import model and inference functions after adding the ckpt path.
    from src.dust3r.inference import inference, inference_recurrent
    from src.dust3r.model import ARCroco3DStereo
    from viser_utils import PointCloudViewer


    from src.dust3r.datasets.waymo import Waymo_Multi
    dataset = Waymo_Multi(allow_repeat=True, split=None, ROOT="/mnt/teams/algo-teams/yuxue.yang/4DVideo/preprocessed_dataset/waymo/train", img_ray_mask_p=[1.0, 0.0, 0.0], aug_crop=16, resolution=[(518, 378),(518, 336),(518, 294),(518, 252),(518, 210),(518, 140),(378, 518),(336, 518),(294, 518),(252, 518)], num_views=24, n_corres=0)
    # Prepare input views.
    print("Preparing input views...")
    # idx = 1000
    idx = 2000
    num_views = 24
    views = dataset.__getitem__((idx, 2, num_views))

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    model = VGGT(img_size=518, patch_size=14, embed_dim=1024)
    ckpt = torch.load(args.model_path, map_location=device)['model']
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    model = model.to(device)

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
    pts3ds, colors, conf, cam_dict = prepare_output(outputs, batch)

    # Convert tensors to numpy arrays for visualization.
    pts3ds_to_vis = [p.cpu().numpy() for p in pts3ds]
    colors_to_vis = [c.cpu().numpy() for c in colors]
    edge_colors = [None] * len(pts3ds_to_vis)

    # Create and run the point cloud viewer.
    print("Launching point cloud viewer...")
    viewer = PointCloudViewer(
        model,
        None,
        pts3ds_to_vis,
        colors_to_vis,
        conf,
        cam_dict,
        device=device,
        edge_color_list=edge_colors,
        show_camera=True,
        vis_threshold=args.vis_threshold,
        size = args.size
    )
    viewer.run()


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
