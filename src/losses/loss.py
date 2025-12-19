# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from src.utils import scene_flow_to_rgb, tf32_off, compute_lpips
from gsplat.rendering import rasterization
from models.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from losses.stage2_loss import prune_gaussians_by_voxel
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor


def parse_gaussian_params(gaussian_params, sh_degree=0):
    """
    Parse gaussian parameters based on sh_degree.

    Args:
        gaussian_params: [..., output_dim] tensor containing gaussian parameters
        sh_degree: spherical harmonics degree (0, 1, 2, 3, ...)

    Returns:
        dict with keys: 'scale', 'sh_coeffs', 'rotations', 'opacity'
        and their corresponding tensor slices
    """
    sh_dim = 3 * ((sh_degree + 1) ** 2)
    scale_start, scale_end = 3, 6
    sh_start, sh_end = 6, 6 + sh_dim
    rotation_start, rotation_end = sh_end, sh_end + 4
    opacity_idx = rotation_end

    return {
        'scale': gaussian_params[..., scale_start:scale_end],
        'sh_coeffs': gaussian_params[..., sh_start:sh_end],
        'rotations': gaussian_params[..., rotation_start:rotation_end],
        'opacity': gaussian_params[..., opacity_idx:opacity_idx+1],
        'indices': {
            'scale': (scale_start, scale_end),
            'sh': (sh_start, sh_end),
            'rotation': (rotation_start, rotation_end),
            'opacity': opacity_idx
        }
    }


def depth_to_world_points(depth, intrinsic):
    """
    Convert depth map to 3D points in camera coordinate system.

    Args:
        depth: [N, H, W, 1] - Depth map in meters
        intrinsic: [1, N, 3, 3] - Camera intrinsic matrix

    Returns:
        camera_points: [N, H, W, 3] - 3D points in camera coordinates
    """
    with tf32_off():
        N, H, W, _ = depth.shape

        v, u = torch.meshgrid(torch.arange(H, device=depth.device),
                              torch.arange(W, device=depth.device),
                              indexing='ij')
        uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1)
        uv1 = uv1.unsqueeze(0).expand(N, -1, -1, -1).float()

        depth = depth.squeeze(-1)
        intrinsic = intrinsic.squeeze(0)

        camera_points = torch.einsum(
            'nij,nhwj->nhwi', torch.inverse(intrinsic), uv1)
        camera_points = camera_points * depth.unsqueeze(-1)

    return camera_points


def velocity_loss(velocity):
    """
    Velocity regularization loss.

    Args:
        velocity: [B, S, H*W, 3] or [S, H*W, 3] - Predicted velocity field (already activated)

    Returns:
        torch.Tensor: Velocity regularization loss
    """
    velocity = velocity.reshape(-1, 3)
    velocity_loss = F.l1_loss(velocity, torch.zeros_like(velocity))
    velocity_loss = check_and_fix_inf_nan(velocity_loss, "velocity_loss")
    return velocity_loss



def gt_flow_loss_ours(velocity, vggt_batch):
    """
    Supervise velocity prediction with ground truth flow map.

    Args:
        velocity: [B, S, H*W, 3] - Predicted velocity
        vggt_batch: dict - Batch data containing GT flowmap
            - 'flowmap': [B, S, H, W, 4] - GT flowmap (first 3 dims are 3D velocity in camera coords, 4th dim is mask)
            - 'extrinsics': [B, S, 4, 4] - Camera extrinsics

    Returns:
        dict: Loss dictionary containing gt_flow_loss
    """
    with tf32_off():
        gt_flowmap = vggt_batch.get('flowmap')

        if gt_flowmap is None:
            return {
                "gt_flow_loss": torch.tensor(0.0, device=velocity.device, requires_grad=True)
            }

        gt_extrinsic = vggt_batch['extrinsics']
        B, S, H, W, flow_dim = gt_flowmap.shape

        gt_velocity_3d = gt_flowmap[..., :3]
        gt_velocity_mask = gt_flowmap[..., 3] != 0

        if gt_velocity_mask.sum() == 0:
            return {
                "gt_flow_loss": torch.tensor(0.0, device=velocity.device, requires_grad=True),
                "gt_flow_num_valid": 0
            }

        velocity = velocity.reshape(B, S, H, W, 3)

        valid_mask_expanded = gt_velocity_mask.unsqueeze(-1).expand(-1, -1, -1, -1, 3)

        pred_velocity_valid = velocity[valid_mask_expanded]
        gt_velocity_valid = gt_velocity_3d[valid_mask_expanded]

        loss = F.l1_loss(pred_velocity_valid, gt_velocity_valid)
        loss = check_and_fix_inf_nan(loss, "gt_flow_loss")

        return {
            "gt_flow_loss": loss,
            "gt_flow_num_valid": gt_velocity_mask.sum().item()
        }


def self_render_and_loss(vggt_batch, preds, sampled_frame_indices=None, sh_degree=0, enable_voxel_pruning=True, voxel_size=0.002):
    """
    Self-rendering loss: each frame renders and supervises itself with optional sky mask replacement.

    Args:
        vggt_batch: dict containing GT data including images, depths, sky_masks, point_masks
        preds: dict containing model predictions including gaussian_params, xyz_camera, extrinsics, intrinsics, sky_colors
        sampled_frame_indices: list of frame indices to render (optional, renders all frames if None)
        sh_degree: int, spherical harmonics degree
        enable_voxel_pruning: bool, whether to enable voxel pruning
        voxel_size: float, voxel size in metric scale (meters)

    Returns:
        dict: Loss dictionary
        dict: Image dictionary for visualization
    """
    with tf32_off():
        gt_rgb = vggt_batch["images"]
        gt_depths = vggt_batch["depths"]
        sky_masks = vggt_batch.get("sky_masks")
        point_masks = vggt_batch.get("point_masks")

        gaussian_params = preds["gaussian_params"]
        pred_sky_colors = preds.get("sky_colors")

        B, S, _, image_height, image_width = gt_rgb.shape

        xyz = preds['xyz_camera'][0]

        output_dim = 11 + 3 * ((sh_degree + 1) ** 2)
        gaussian_params = gaussian_params.reshape(
            1, -1, image_height * image_width, output_dim)

        parsed = parse_gaussian_params(gaussian_params[0], sh_degree)
        scale = parsed['scale']
        sh_coeffs = parsed['sh_coeffs']
        sh_coeffs_reshaped = sh_coeffs.reshape(S, image_height * image_width, (sh_degree + 1) ** 2, 3)
        rotations = parsed['rotations']
        opacity = parsed['opacity'].squeeze(-1)

        extrinsics = preds['extrinsics']
        intrinsics = preds['intrinsics']
        viewmat = extrinsics.permute(1, 0, 2, 3)
        K = intrinsics.permute(1, 0, 2, 3)

        if sampled_frame_indices is None:
            sampled_frame_indices = list(range(S))
        num_frames_to_render = len(sampled_frame_indices)

        render_colors_tensor = torch.zeros(num_frames_to_render, image_height, image_width, 4, device=gt_rgb.device, dtype=xyz.dtype)
        render_alphas_tensor = torch.zeros(num_frames_to_render, image_height, image_width, 1, device=gt_rgb.device, dtype=xyz.dtype)

        depth_scale_factor = vggt_batch.get('depth_scale_factor', None)
        if depth_scale_factor is not None and torch.is_tensor(depth_scale_factor):
            depth_scale_factor = depth_scale_factor.item()

        for render_idx, i in enumerate(sampled_frame_indices):
            mean_current = xyz[i]
            scale_current = scale[i]
            rotations_current = rotations[i]
            opacity_current = opacity[i]
            sh_coeffs_current = sh_coeffs_reshaped[i]

            if sky_masks is not None:
                current_sky_mask = sky_masks[0, i].flatten()
                non_sky_mask = ~current_sky_mask.bool()

                if non_sky_mask.sum() > 0:
                    mean_current = mean_current[non_sky_mask]
                    scale_current = scale_current[non_sky_mask]
                    rotations_current = rotations_current[non_sky_mask]
                    opacity_current = opacity_current[non_sky_mask]
                    sh_coeffs_current = sh_coeffs_current[non_sky_mask]
                else:
                    render_color = torch.zeros((image_height, image_width, 4),
                                             device=mean_current.device, dtype=mean_current.dtype)
                    render_alpha = torch.zeros((image_height, image_width, 1),
                                             device=mean_current.device, dtype=mean_current.dtype)
                    render_colors_tensor[render_idx] = render_color
                    render_alphas_tensor[render_idx] = render_alpha
                    continue

            if enable_voxel_pruning and mean_current.shape[0] > 0:
                mean_current, scale_current, rotations_current, opacity_current, sh_coeffs_current = prune_gaussians_by_voxel(
                    mean_current, scale_current, rotations_current, opacity_current, sh_coeffs_current,
                    voxel_size=voxel_size,
                    depth_scale_factor=depth_scale_factor
                )

            if mean_current.shape[0] > 0:
                render_color, render_alpha, _ = rasterization(
                    mean_current, rotations_current,
                    scale_current, opacity_current, sh_coeffs_current,
                    viewmat[i], K[i], image_width, image_height,
                    sh_degree=sh_degree, render_mode="RGB+ED",
                    radius_clip=0, near_plane=0.0001,
                    far_plane=1000.0,
                    eps2d=0.3,
                )
            else:
                render_color = torch.zeros((image_height, image_width, 4),
                                         device=gt_rgb.device, dtype=xyz.dtype)
                render_alpha = torch.zeros((image_height, image_width, 1),
                                         device=gt_rgb.device, dtype=xyz.dtype)

            render_colors_tensor[render_idx] = render_color
            if render_alpha is not None:
                render_alphas_tensor[render_idx] = render_alpha

        gt_colors_sampled = gt_rgb[0, sampled_frame_indices]
        gt_depths_sampled = gt_depths[0, sampled_frame_indices]

        pred_rgb = render_colors_tensor[..., :3].permute(0, 3, 1, 2)
        pred_rgb = torch.clamp(pred_rgb, min=0, max=1)
        pred_depth = render_colors_tensor[..., -1]

        if pred_sky_colors is not None:
            pred_rgb_with_sky = pred_rgb.clone()
            for render_idx, i in enumerate(sampled_frame_indices):
                rendered_rgb_frame = pred_rgb[render_idx]
                rendered_alpha = render_alphas_tensor[render_idx, :, :, 0]
                sky_colors_chw = pred_sky_colors[0, render_idx]
                alpha_3ch = rendered_alpha.unsqueeze(0)
                pred_rgb_with_sky[render_idx] = alpha_3ch * rendered_rgb_frame + (1 - alpha_3ch) * sky_colors_chw

            pred_rgb = pred_rgb_with_sky

        valid_depth_mask = point_masks[0, sampled_frame_indices].bool()
        depth_loss = F.l1_loss(pred_depth[valid_depth_mask], gt_depths_sampled[valid_depth_mask])
        depth_loss = check_and_fix_inf_nan(depth_loss, "self_depth_loss")

        rgb_loss = F.l1_loss(pred_rgb, gt_colors_sampled)
        rgb_loss = check_and_fix_inf_nan(rgb_loss, "self_rgb_mse")

        lpips_loss = compute_lpips(pred_rgb, gt_colors_sampled).mean()
        lpips_loss = check_and_fix_inf_nan(lpips_loss, "self_lpips_loss")

        self_loss_dict = {
            "loss_self_render_rgb": rgb_loss,
            "loss_self_render_lpips": lpips_loss,
            "loss_self_render_depth": depth_loss,
        }

        img_dict = {
            "self_rgb_pred": pred_rgb,
            "self_rgb_gt": gt_colors_sampled,
            "self_depth_pred": pred_depth.unsqueeze(1),
            "self_depth_gt": gt_depths_sampled.unsqueeze(1),
        }

        return self_loss_dict, img_dict


def sky_opacity_loss(gaussian_params, sky_masks, weight=1.0):
    """
    Encourage gaussian opacity to be 0 in sky regions.

    Args:
        gaussian_params: [B, S, H, W, 14] - Gaussian parameters
        sky_masks: [B, S, H, W] - Sky masks where 1 indicates sky regions
        weight: Loss weight

    Returns:
        dict: Loss dictionary containing sky_opacity_loss
    """
    opacity = gaussian_params[..., 13:14]
    non_sky_masks = 1.0 - sky_masks

    sky_opacity = opacity * sky_masks.unsqueeze(-1)
    non_sky_opacity = opacity * non_sky_masks.unsqueeze(-1)

    sky_loss = torch.mean(torch.abs(sky_opacity))
    non_sky_loss = torch.mean(torch.abs(non_sky_opacity - 1.0))

    total_loss = sky_loss

    return {
        "sky_opacity_loss": total_loss * weight
    }


def check_and_fix_inf_nan(loss_tensor, loss_name, hard_max=100):
    """
    Check and fix inf/nan values in loss tensor.

    Args:
        loss_tensor: Loss tensor to check
        loss_name: Name of the loss for diagnostic prints
        hard_max: Maximum allowed value

    Returns:
        torch.Tensor: Fixed loss tensor with inf/nan replaced by 0
    """
    if torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any():
        for _ in range(10):
            print(f"{loss_name} has inf or nan. Setting those values to 0.")
        loss_tensor = torch.where(
            torch.isnan(loss_tensor) | torch.isinf(loss_tensor),
            torch.tensor(0.0, device=loss_tensor.device, requires_grad=True),
            loss_tensor
        )

    loss_tensor = torch.clamp(loss_tensor, min=-hard_max, max=hard_max)

    return loss_tensor


def camera_loss(pred_pose_enc_list, batch, loss_type="l1", gamma=0.6, pose_encoding_type="absT_quaR_FoV", weight_T=1.0, weight_R=1.0, weight_fl=0.5, frame_num=-100):
    """
    Camera pose loss with multi-scale predictions.

    Args:
        pred_pose_enc_list: List of predicted pose encodings at different scales
        batch: Batch dictionary containing ground truth data
        loss_type: Type of loss ("l1", "l2", or "huber")
        gamma: Weight decay factor for multi-scale losses
        pose_encoding_type: Type of pose encoding
        weight_T: Weight for translation loss
        weight_R: Weight for rotation loss
        weight_fl: Weight for focal length loss
        frame_num: Number of frames to use for loss computation

    Returns:
        dict: Loss dictionary
    """
    mask_valid = batch['point_masks']
    batch_valid_mask = mask_valid[:, 0].sum(dim=[-1, -2]) > 100
    num_predictions = len(pred_pose_enc_list)

    gt_extrinsic = batch['extrinsics']
    gt_intrinsic = batch['intrinsics']
    image_size_hw = batch['images'].shape[-2:]

    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsic, gt_intrinsic, image_size_hw, pose_encoding_type=pose_encoding_type)

    loss_T = loss_R = loss_fl = 0

    for i in range(num_predictions):
        i_weight = gamma ** (num_predictions - i - 1)
        cur_pred_pose_enc = pred_pose_enc_list[i]

        if batch_valid_mask.sum() == 0:
            loss_T_i = (cur_pred_pose_enc * 0).mean()
            loss_R_i = (cur_pred_pose_enc * 0).mean()
            loss_fl_i = (cur_pred_pose_enc * 0).mean()
        else:
            if frame_num > 0:
                loss_T_i, loss_R_i, loss_fl_i = camera_loss_single(cur_pred_pose_enc[batch_valid_mask][:, :frame_num].clone(
                ), gt_pose_encoding[batch_valid_mask][:, :frame_num].clone(), loss_type=loss_type)
            else:
                loss_T_i, loss_R_i, loss_fl_i = camera_loss_single(cur_pred_pose_enc[batch_valid_mask].clone(
                ), gt_pose_encoding[batch_valid_mask].clone(), loss_type=loss_type)
        loss_T += loss_T_i * i_weight
        loss_R += loss_R_i * i_weight
        loss_fl += loss_fl_i * i_weight

    loss_T = loss_T / num_predictions
    loss_R = loss_R / num_predictions
    loss_fl = loss_fl / num_predictions
    loss_camera = loss_T * weight_T + loss_R * weight_R + loss_fl * weight_fl

    loss_dict = {
        "loss_camera": loss_camera,
        "loss_T": loss_T,
        "loss_R": loss_R,
        "loss_fl": loss_fl
    }

    return loss_dict


def camera_loss_single(cur_pred_pose_enc, gt_pose_encoding, loss_type="l1"):
    if loss_type == "l1":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).abs()
        loss_R = (cur_pred_pose_enc[..., 3:7] -
                  gt_pose_encoding[..., 3:7]).abs()
        loss_fl = (cur_pred_pose_enc[..., 7:] -
                   gt_pose_encoding[..., 7:]).abs()
    elif loss_type == "l2":
        loss_T = (cur_pred_pose_enc[..., :3] -
                  gt_pose_encoding[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (cur_pred_pose_enc[..., 3:7] -
                  gt_pose_encoding[..., 3:7]).norm(dim=-1)
        loss_fl = (cur_pred_pose_enc[..., 7:] -
                   gt_pose_encoding[..., 7:]).norm(dim=-1)
    elif loss_type == "huber":
        loss_T = huber_loss(
            cur_pred_pose_enc[..., :3], gt_pose_encoding[..., :3])
        loss_R = huber_loss(
            cur_pred_pose_enc[..., 3:7], gt_pose_encoding[..., 3:7])
        loss_fl = huber_loss(
            cur_pred_pose_enc[..., 7:], gt_pose_encoding[..., 7:])
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_fl = check_and_fix_inf_nan(loss_fl, "loss_fl")

    loss_T = loss_T.clamp(max=100)
    loss_T = loss_T.mean()
    loss_R = loss_R.mean()
    loss_fl = loss_fl.mean()

    return loss_T, loss_R, loss_fl


def normalize_pointcloud(pts3d, valid_mask, eps=1e-3):
    """
    Normalize point cloud by average distance.

    Args:
        pts3d: [B, S, H, W, 3] - 3D points
        valid_mask: [B, S, H, W] - Valid point mask
        eps: Small epsilon for numerical stability

    Returns:
        pts3d: Normalized 3D points
        avg_scale: Average scale factor
    """
    dist = pts3d.norm(dim=-1)

    dist_sum = (dist * valid_mask).sum(dim=[1, 2, 3])
    valid_count = valid_mask.sum(dim=[1, 2, 3])

    avg_scale = (dist_sum / (valid_count + eps)).clamp(min=eps, max=1e3)

    pts3d = pts3d / avg_scale.view(-1, 1, 1, 1, 1)
    return pts3d, avg_scale


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss computing L1 difference between adjacent pixels.

    Args:
        prediction: [B, H, W, C] - Predicted values
        target: [B, H, W, C] - Ground truth values
        mask: [B, H, W] - Valid pixel mask
        conf: [B, H, W] - Confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization

    Returns:
        Gradient loss scalar
    """
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss


def depth_loss(depth, depth_conf, batch, gamma=1.0, alpha=0.2, loss_type="conf", predict_disparity=False, affine_inv=False, gradient_loss=None, valid_range=-1, disable_conf=False, all_mean=False, **kwargs):

    gt_depth = batch['depths'].clone()
    valid_mask = batch['point_masks']

    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")

    gt_depth = gt_depth[..., None]

    if loss_type == "conf":
        conf_loss_dict = conf_loss(depth, depth_conf, gt_depth, valid_mask,
                                   batch, normalize_pred=False, normalize_gt=False,
                                   gamma=gamma, alpha=alpha, affine_inv=affine_inv, gradient_loss=gradient_loss, valid_range=valid_range, postfix="_depth", disable_conf=disable_conf, all_mean=all_mean)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    return conf_loss_dict


def point_loss(pts3d, pts3d_conf, batch, normalize_pred=False, gamma=1.0, alpha=0.2, affine_inv=False, gradient_loss=None, valid_range=-1, camera_centric_reg=-1, disable_conf=False, all_mean=False, conf_loss_type="v1", **kwargs):
    """
    3D point prediction loss with confidence weighting.

    Args:
        pts3d: [B, S, H, W, 3] - Predicted 3D points
        pts3d_conf: [B, S, H, W] - Prediction confidence
        batch: Batch dictionary containing ground truth
        normalize_pred: Whether to normalize predictions
        gamma: Confidence loss weight
        alpha: Confidence regularization weight
        affine_inv: Whether to apply affine invariant alignment
        gradient_loss: Type of gradient loss to use
        valid_range: Quantile range for valid loss filtering
        camera_centric_reg: Camera-centric regularization weight
        disable_conf: Whether to disable confidence weighting
        all_mean: Whether to average all losses together
        conf_loss_type: Type of confidence loss

    Returns:
        dict: Loss dictionary
    """
    gt_pts3d = batch['world_points']
    valid_mask = batch['point_masks']
    gt_pts3d = check_and_fix_inf_nan(gt_pts3d, "gt_pts3d")

    if conf_loss_type == "v1":
        conf_loss_fn = conf_loss
    else:
        raise ValueError(f"Invalid conf loss type: {conf_loss_type}")

    conf_loss_dict = conf_loss_fn(pts3d, pts3d_conf, gt_pts3d, valid_mask,
                                  batch, normalize_pred=normalize_pred, gamma=gamma, alpha=alpha, affine_inv=affine_inv,
                                  gradient_loss=gradient_loss, valid_range=valid_range, camera_centric_reg=camera_centric_reg, disable_conf=disable_conf, all_mean=all_mean)

    return conf_loss_dict


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by quantile threshold and clamp to maximum value.

    Args:
        loss_tensor: Loss tensor to filter
        valid_range: Quantile threshold (0 to 1)
        min_elements: Minimum elements required for filtering
        hard_max: Maximum allowed value

    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= 1000:
        return loss_tensor

    if loss_tensor.numel() > 100000000:
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[
            :1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    loss_tensor = loss_tensor.clamp(max=hard_max)

    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def conf_loss(pts3d, pts3d_conf, gt_pts3d, valid_mask,  batch, normalize_gt=True, normalize_pred=True, gamma=1.0, alpha=0.2, affine_inv=False, gradient_loss=None, valid_range=-1, camera_centric_reg=-1, disable_conf=False, all_mean=False, postfix=""):
    """
    Confidence-weighted loss for 3D point predictions.

    Args:
        pts3d: Predicted 3D points
        pts3d_conf: Prediction confidence
        gt_pts3d: Ground truth 3D points
        valid_mask: Valid point mask
        batch: Batch dictionary
        normalize_gt: Whether to normalize ground truth
        normalize_pred: Whether to normalize predictions
        gamma: Confidence loss weight
        alpha: Confidence regularization weight
        affine_inv: Whether to apply affine invariant alignment
        gradient_loss: Type of gradient loss
        valid_range: Quantile range for filtering
        camera_centric_reg: Camera-centric regularization
        disable_conf: Whether to disable confidence weighting
        all_mean: Whether to average all losses
        postfix: Suffix for loss names

    Returns:
        dict: Loss dictionary
    """
    if normalize_gt:
        gt_pts3d, gt_pts3d_scale = normalize_pointcloud(gt_pts3d, valid_mask)

    if normalize_pred:
        pts3d, pred_pts3d_scale = normalize_pointcloud(pts3d, valid_mask)

    if affine_inv:
        scale, shift = closed_form_scale_and_shift(pts3d, gt_pts3d, valid_mask)
        pts3d = pts3d * scale + shift

    loss_reg_first_frame, loss_reg_other_frames, loss_grad_first_frame, loss_grad_other_frames = reg_loss(
        pts3d, gt_pts3d, valid_mask, gradient_loss=gradient_loss)

    if disable_conf:
        conf_loss_first_frame = gamma * loss_reg_first_frame
        conf_loss_other_frames = gamma * loss_reg_other_frames
    else:
        first_frame_conf = pts3d_conf[:, 0:1, ...]
        other_frames_conf = pts3d_conf[:, 1:, ...]
        first_frame_mask = valid_mask[:, 0:1, ...]
        other_frames_mask = valid_mask[:, 1:, ...]

        conf_loss_first_frame = gamma * loss_reg_first_frame * \
            first_frame_conf[first_frame_mask] - alpha * \
            torch.log(first_frame_conf[first_frame_mask])
        conf_loss_other_frames = gamma * loss_reg_other_frames * \
            other_frames_conf[other_frames_mask] - alpha * \
            torch.log(other_frames_conf[other_frames_mask])

    if conf_loss_first_frame.numel() > 0 and conf_loss_other_frames.numel() > 0:
        if valid_range > 0:
            conf_loss_first_frame = filter_by_quantile(
                conf_loss_first_frame, valid_range)
            conf_loss_other_frames = filter_by_quantile(
                conf_loss_other_frames, valid_range)

        conf_loss_first_frame = check_and_fix_inf_nan(
            conf_loss_first_frame, f"conf_loss_first_frame{postfix}")
        conf_loss_other_frames = check_and_fix_inf_nan(
            conf_loss_other_frames, f"conf_loss_other_frames{postfix}")
    else:
        conf_loss_first_frame = pts3d * 0
        conf_loss_other_frames = pts3d * 0
        # print("No valid conf loss", batch["seq_name"])

    if all_mean and conf_loss_first_frame.numel() > 0 and conf_loss_other_frames.numel() > 0:
        all_conf_loss = torch.cat(
            [conf_loss_first_frame, conf_loss_other_frames])
        conf_loss = all_conf_loss.mean() if all_conf_loss.numel() > 0 else 0

        conf_loss_first_frame = conf_loss_first_frame.mean(
        ) if conf_loss_first_frame.numel() > 0 else 0
        conf_loss_other_frames = conf_loss_other_frames.mean(
        ) if conf_loss_other_frames.numel() > 0 else 0
    else:
        conf_loss_first_frame = conf_loss_first_frame.mean(
        ) if conf_loss_first_frame.numel() > 0 else 0
        conf_loss_other_frames = conf_loss_other_frames.mean(
        ) if conf_loss_other_frames.numel() > 0 else 0

        conf_loss = conf_loss_first_frame + conf_loss_other_frames

    reg_loss_value = (loss_reg_first_frame.mean() if loss_reg_first_frame.numel() > 0 else 0) + \
               (loss_reg_other_frames.mean() if loss_reg_other_frames.numel() > 0 else 0)

    loss_dict = {
        f"loss_conf{postfix}": conf_loss,
        f"loss_reg{postfix}": reg_loss_value,
        f"loss_reg1{postfix}": loss_reg_first_frame.detach().mean() if loss_reg_first_frame.numel() > 0 else 0,
        f"loss_reg2{postfix}": loss_reg_other_frames.detach().mean() if loss_reg_other_frames.numel() > 0 else 0,
        f"loss_conf1{postfix}": conf_loss_first_frame,
        f"loss_conf2{postfix}": conf_loss_other_frames,
    }

    if gradient_loss is not None:
        loss_grad = loss_grad_first_frame + loss_grad_other_frames
        loss_dict[f"loss_grad1{postfix}"] = loss_grad_first_frame
        loss_dict[f"loss_grad2{postfix}"] = loss_grad_other_frames
        loss_dict[f"loss_grad{postfix}"] = loss_grad

    return loss_dict


def reg_loss(pts3d, gt_pts3d, valid_mask, gradient_loss=None):

    first_frame_pts3d = pts3d[:, 0:1, ...]
    first_frame_gt_pts3d = gt_pts3d[:, 0:1, ...]
    first_frame_mask = valid_mask[:, 0:1, ...]

    other_frames_pts3d = pts3d[:, 1:, ...]
    other_frames_gt_pts3d = gt_pts3d[:, 1:, ...]
    other_frames_mask = valid_mask[:, 1:, ...]

    loss_reg_first_frame = torch.norm(
        first_frame_gt_pts3d[first_frame_mask] - first_frame_pts3d[first_frame_mask], dim=-1)
    loss_reg_other_frames = torch.norm(
        other_frames_gt_pts3d[other_frames_mask] - other_frames_pts3d[other_frames_mask], dim=-1)

    if gradient_loss == "grad":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(first_frame_pts3d.reshape(
            bb*ss, hh, ww, nc), first_frame_gt_pts3d.reshape(bb*ss, hh, ww, nc), first_frame_mask.reshape(bb*ss, hh, ww))
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(other_frames_pts3d.reshape(
            bb*ss, hh, ww, nc), other_frames_gt_pts3d.reshape(bb*ss, hh, ww, nc), other_frames_mask.reshape(bb*ss, hh, ww))
    elif gradient_loss == "grad_impl2":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(first_frame_pts3d.reshape(bb*ss, hh, ww, nc), first_frame_gt_pts3d.reshape(
            bb*ss, hh, ww, nc), first_frame_mask.reshape(bb*ss, hh, ww), gradient_loss_fn=gradient_loss_impl2)
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(other_frames_pts3d.reshape(bb*ss, hh, ww, nc), other_frames_gt_pts3d.reshape(
            bb*ss, hh, ww, nc), other_frames_mask.reshape(bb*ss, hh, ww), gradient_loss_fn=gradient_loss_impl2)
    elif gradient_loss == "normal":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(first_frame_pts3d.reshape(bb*ss, hh, ww, nc), first_frame_gt_pts3d.reshape(
            bb*ss, hh, ww, nc), first_frame_mask.reshape(bb*ss, hh, ww), gradient_loss_fn=normal_loss, scales=3)
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(other_frames_pts3d.reshape(bb*ss, hh, ww, nc), other_frames_gt_pts3d.reshape(
            bb*ss, hh, ww, nc), other_frames_mask.reshape(bb*ss, hh, ww), gradient_loss_fn=normal_loss, scales=3)
    else:
        loss_grad_first_frame = 0
        loss_grad_other_frames = 0

    loss_reg_first_frame = check_and_fix_inf_nan(
        loss_reg_first_frame, "loss_reg_first_frame")
    loss_reg_other_frames = check_and_fix_inf_nan(
        loss_reg_other_frames, "loss_reg_other_frames")

    return loss_reg_first_frame, loss_reg_other_frames, loss_grad_first_frame, loss_grad_other_frames


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None):
    """
    Normal-based loss comparing angles between predicted and ground truth normals.

    Args:
        prediction: [B, H, W, 3] - Predicted 3D points
        target: [B, H, W, 3] - Ground truth 3D points
        mask: [B, H, W] - Valid pixel mask
        cos_eps: Epsilon for numerical stability
        conf: [B, H, W] - Confidence weights (optional)

    Returns:
        Scalar loss averaged over valid regions
    """
    pred_normals, pred_valids = point_map_to_normal(
        prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids = point_map_to_normal(
        target,     mask, eps=cos_eps)

    all_valid = pred_valids & gt_valids

    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    dot = torch.sum(pred_normals * gt_normals, dim=-1)
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    loss = 1 - dot

    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            gamma = 1.0
            alpha = 0.2

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return loss.mean()
        else:
            return loss.mean()


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Compute surface normals from 3D point map using cross products.

    Args:
        point_map: [B, H, W, 3] - 3D points in 2D grid layout
        mask: [B, H, W] - Valid pixel mask

    Returns:
        normals: [4, B, H, W, 3] - Normal vectors for 4 cross-product directions
        valids: [4, B, H, W] - Corresponding valid masks
    """
    with torch.cuda.amp.autocast(enabled=False):
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1, 1, 1, 1),
                    mode='constant', value=0).permute(0, 2, 3, 1)

        center = pts[:, 1:-1, 1:-1, :]
        up = pts[:, :-2,  1:-1, :]
        left = pts[:, 1:-1, :-2, :]
        down = pts[:, 2:,   1:-1, :]
        right = pts[:, 1:-1, 2:, :]

        up_dir = up - center
        left_dir = left - center
        down_dir = down - center
        right_dir = right - center

        n1 = torch.cross(up_dir,   left_dir,  dim=-1)
        n2 = torch.cross(left_dir, down_dir,  dim=-1)
        n3 = torch.cross(down_dir, right_dir, dim=-1)
        n4 = torch.cross(right_dir, up_dir,    dim=-1)

        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:,
                                                      1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2] & padded_mask[:,
                                                     1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:,
                                                      1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:] & padded_mask[:,
                                                    1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        normals = torch.stack([n1, n2, n3, n4], dim=0)
        valids = torch.stack([v1, v2, v3, v4], dim=0)

        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss for spatial smoothness.

    Args:
        prediction: [B, H, W, C] - Predicted values
        target: [B, H, W, C] - Ground truth values
        mask: [B, H, W] - Valid pixel mask
        conf: [B, H, W] - Confidence weights (optional)
        gamma: Confidence loss weight
        alpha: Confidence regularization weight

    Returns:
        Gradient loss scalar
    """
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]
        gamma = 1.0
        alpha = 0.2

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    image_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))

    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        image_loss = torch.sum(image_loss) / divisor

    return image_loss


def gradient_loss_multi_scale(prediction, target, mask, scales=4, gradient_loss_fn=gradient_loss, conf=None):
    """
    Compute gradient loss across multiple scales
    """

    total = 0
    for scale in range(scales):
        step = pow(2, scale)

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def torch_quantile(
    input: torch.Tensor,
    q: float | torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Efficient quantile computation using torch.kthvalue.

    Better than torch.quantile for large inputs:
    - No 2**24 size limit
    - Much faster on large tensors

    Args:
        input: Input tensor
        q: Quantile value (0 to 1, scalar only)
        dim: Dimension along which to compute quantile
        keepdim: Whether to keep dimension
        interpolation: {"nearest", "lower", "higher"}
        out: Output tensor (None only)

    Returns:
        Quantile tensor
    """
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(
            f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    if out is not None:
        raise ValueError(
            f"Only None value is currently supported for out (got {out})!")

    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out


def scale_loss(predicted_scale, gt_scale):
    """
    Compute L1 loss between predicted scene scale and ground truth scale.

    Args:
        predicted_scale: [B] - Predicted scene scale from scale_head
        gt_scale: [B] or scalar - Ground truth depth_scale_factor (1 / dist_avg)

    Returns:
        dict: Loss dictionary containing scale_loss
    """
    with tf32_off():
        # Ensure gt_scale is a tensor
        if not isinstance(gt_scale, torch.Tensor):
            gt_scale = torch.tensor(gt_scale, device=predicted_scale.device, dtype=predicted_scale.dtype)

        # If gt_scale is a scalar, expand to match batch size
        if gt_scale.dim() == 0:
            gt_scale = gt_scale.expand(predicted_scale.shape[0])

        # Compute L1 loss
        loss = F.l1_loss(predicted_scale, gt_scale)
        loss = check_and_fix_inf_nan(loss, "scale_loss")

        return {
            "scale_loss": loss,
            "scale_pred_mean": predicted_scale.mean().detach(),
            "scale_gt_mean": gt_scale.mean().detach(),
        }


def segment_loss(segment_logits, segment_conf, vggt_batch, gamma=1.0, disable_conf=False, class_weights=None):
    """
    Cross-entropy loss for semantic segmentation.

    Args:
        segment_logits: [B, S, H, W, 4] - Predicted logits for 4 classes
        segment_conf: [B, S, H, W] - Confidence scores (unused, for API compatibility)
        vggt_batch: Batch dict containing 'segment_label' and 'segment_mask'
        gamma: Unused, for API compatibility
        disable_conf: Unused, for API compatibility
        class_weights: [4] - Class weights for imbalanced classes (optional)

    Returns:
        dict: Loss dictionary
    """
    with tf32_off():
        if 'segment_label' not in vggt_batch or 'segment_mask' not in vggt_batch:
            return {
                "segment_loss": torch.tensor(0.0, device=segment_logits.device),
            }

        gt_labels = vggt_batch['segment_label']
        gt_mask = vggt_batch['segment_mask']

        B, S, H, W, num_classes = segment_logits.shape
        assert num_classes == 4, f"Expected 4 classes, got {num_classes}"

        segment_logits_flat = segment_logits.reshape(B * S * H * W, num_classes)
        gt_labels_flat = gt_labels.reshape(B * S * H * W).long()
        gt_mask_flat = gt_mask.reshape(B * S * H * W)

        valid_indices = gt_mask_flat > 0.5
        if valid_indices.sum() == 0:
            return {
                "segment_loss": torch.tensor(0.0, device=segment_logits.device),
            }

        segment_logits_valid = segment_logits_flat[valid_indices]
        gt_labels_valid = gt_labels_flat[valid_indices]

        if class_weights is not None:
            class_weights = class_weights.to(segment_logits.device)
        ce_loss = F.cross_entropy(segment_logits_valid, gt_labels_valid, weight=class_weights, reduction='mean')
        ce_loss = check_and_fix_inf_nan(ce_loss, "segment_ce_loss")

        return {
            "segment_loss": ce_loss,
        }


def voxel_quantize_random_sampling(xyz, attributes_dict, sky_mask=None, voxel_size=0.05, gt_scale=1.0):
    """
    Voxel quantization with random sampling - randomly select one point per voxel.

    Args:
        xyz: [N, 3] - 3D positions in metric scale
        attributes_dict: dict of [N, ...] tensors - Attributes to aggregate
        sky_mask: [N] or None - Boolean mask for sky pixels to filter out
        voxel_size: Voxel size in metric scale (e.g., 0.05 for 5cm)
        gt_scale: GT scale factor to convert to metric scale

    Returns:
        selected_xyz: [M, 3] - Selected 3D positions
        selected_attributes: dict of [M, ...] tensors - Selected attributes
    """
    with tf32_off():
        if sky_mask is not None:
            valid_mask = ~sky_mask.bool()
            xyz_valid = xyz[valid_mask]

            if xyz_valid.shape[0] == 0:
                empty_result = {k: v[valid_mask] for k, v in attributes_dict.items()}
                return xyz_valid, empty_result
        else:
            xyz_valid = xyz
            valid_mask = None

        xyz_metric = xyz_valid / gt_scale

        voxel_indices = torch.floor(xyz_metric / voxel_size).long()

        voxel_ids = (voxel_indices[:, 0] * 73856093) ^ \
                    (voxel_indices[:, 1] * 19349663) ^ \
                    (voxel_indices[:, 2] * 83492791)

        unique_voxel_ids, inverse_indices = torch.unique(
            voxel_ids, return_inverse=True
        )

        random_values = torch.rand(len(voxel_ids), device=xyz.device)

        _, selected_indices = torch_scatter.scatter_max(
            random_values, inverse_indices
        )

        selected_xyz = xyz_valid[selected_indices]
        selected_attributes = {}
        if valid_mask is not None:
            for k, v in attributes_dict.items():
                v_valid = v[valid_mask]
                selected_attributes[k] = v_valid[selected_indices]
        else:
            for k, v in attributes_dict.items():
                selected_attributes[k] = v[selected_indices]

        return selected_xyz, selected_attributes

