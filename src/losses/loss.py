# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# DIRTY VERSION, TO BE CLEANED UP

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
        Note: confidence is returned separately by gaussian_head, not in gaussian_params
    """
    sh_dim = 3 * ((sh_degree + 1) ** 2)

    # Calculate indices
    # Channels: [xyz_offset(3), scale(3), sh_coeffs(sh_dim), rotation(4), opacity(1)]
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
        camera_points = torch.einsum(
            'nij,nhwj->nhwi', torch.inverse(intrinsic), uv1)  # [N, H, W, 3]
        camera_points = camera_points * depth.unsqueeze(-1)  # [N, H, W, 3]

    return camera_points


def velocity_loss(velocity):
    """
    独立的velocity正则化损失函数

    Args:
        velocity: [B, S, H*W, 3] 或 [S, H*W, 3] - 预测的速度场（已激活）

    Returns:
        torch.Tensor: velocity正则化损失
    """
    # velocity regularization loss
    velocity = velocity.reshape(-1, 3)  # [S, H*W, 3]
    # velocity已在模型forward中激活，这里直接使用
    velocity_loss = F.l1_loss(velocity, torch.zeros_like(velocity))
    velocity_loss = check_and_fix_inf_nan(velocity_loss, "velocity_loss")
    return velocity_loss



def gt_flow_loss_ours(velocity, vggt_batch):
    """
    使用GT flowmap对velocity head进行直接监督

    Args:
        velocity: [B, S, H*W, 3] - 预测的velocity
        vggt_batch: dict - 包含GT flowmap的batch数据
            - 'flowmap': [B, S, H, W, 4] - GT flowmap，前3维是3D velocity（在各帧相机坐标系下），第4维是类别（0表示无GT信息）
            - 'extrinsics': [B, S, 4, 4] - 相机外参（已转换为从第一帧到各帧的变换）

    Returns:
        dict: 包含gt_flow_loss的损失字典
    """
    with tf32_off():
        # 获取GT flowmap
        gt_flowmap = vggt_batch.get('flowmap')  # [B, S, H, W, 4]

        if gt_flowmap is None:
            # 如果没有GT flowmap，返回0损失
            return {
                "gt_flow_loss": torch.tensor(0.0, device=velocity.device, requires_grad=True)
            }

        # 使用GT的pose（从vggt_batch获取）
        gt_extrinsic = vggt_batch['extrinsics']  # [B, S, 4, 4] - GT extrinsics

        B, S, H, W, flow_dim = gt_flowmap.shape

        # 提取GT velocity (前3维) 和 mask (第4维)
        gt_velocity_3d = gt_flowmap[..., :3]  # [B, S, H, W, 3]
        gt_velocity_mask = gt_flowmap[..., 3] != 0  # [B, S, H, W] - 有GT velocity的区域

        # 检查是否有有效的GT velocity
        if gt_velocity_mask.sum() == 0:
            return {
                "gt_flow_loss": torch.tensor(0.0, device=velocity.device, requires_grad=True),
                "gt_flow_num_valid": 0
            }

        # 将velocity reshape为与gt_velocity相同的格式
        velocity = velocity.reshape(B, S, H, W, 3)  # [B, S, H, W, 3]


        # 只在有GT velocity的位置计算loss
        # 扩展mask以匹配velocity的维度
        valid_mask_expanded = gt_velocity_mask.unsqueeze(-1).expand(-1, -1, -1, -1, 3)  # [B, S, H, W, 3]

        # 提取有效位置的velocity
        pred_velocity_valid = velocity[valid_mask_expanded]  # [N_valid]
        gt_velocity_valid = gt_velocity_3d[valid_mask_expanded]  # [N_valid]

        # 计算L1 loss
        loss = F.l1_loss(pred_velocity_valid, gt_velocity_valid)
        loss = check_and_fix_inf_nan(loss, "gt_flow_loss")

        # 返回损失字典
        return {
            "gt_flow_loss": loss,
            "gt_flow_num_valid": gt_velocity_mask.sum().item()
        }


def self_render_and_loss(vggt_batch, preds, sampled_frame_indices=None, sh_degree=0, enable_voxel_pruning=True, voxel_size=0.002):
    """
    自渲染损失函数：每一帧自己进行渲染与监督，支持天空区域的mask替代

    Args:
        vggt_batch: dict 包含GT数据的batch，包括images, depths, sky_masks, point_masks等
        preds: dict 模型预测结果，包含gaussian_params, xyz_camera, extrinsics, intrinsics, pred_sky_colors等
        sampled_frame_indices: list 采样的帧索引 (可选，如果为None则渲染所有帧)
        sh_degree: int spherical harmonics degree
        enable_voxel_pruning: bool 是否启用voxel剪枝
        voxel_size: float voxel大小（metric尺度，单位米）

    Returns:
        dict: 损失字典
        dict: 图像字典（用于可视化）
    """
    with tf32_off():
        # 从vggt_batch中获取GT数据
        gt_rgb = vggt_batch["images"]  # [B, S, 3, H, W]
        gt_depths = vggt_batch["depths"]  # [B, S, H, W]
        sky_masks = vggt_batch.get("sky_masks")  # [B, S, H, W]
        point_masks = vggt_batch.get("point_masks")  # [B, S, H, W]

        # 从preds中获取预测数据
        gaussian_params = preds["gaussian_params"]  # [B, S, H*W, output_dim]
        pred_sky_colors = preds.get("sky_colors")  # [B, num_frames, 3, H, W]

        # 获取图像尺寸
        B, S, _, image_height, image_width = gt_rgb.shape

        # 1. 从preds中直接获取xyz
        xyz = preds['xyz_camera'][0]  # [S, H*W, 3]

        # 处理高斯参数 (已在forward中激活，直接使用)
        output_dim = 11 + 3 * ((sh_degree + 1) ** 2)
        gaussian_params = gaussian_params.reshape(
            1, -1, image_height * image_width, output_dim)  # [1, S, H*W, output_dim]

        # Parse gaussian parameters
        parsed = parse_gaussian_params(gaussian_params[0], sh_degree)
        scale = parsed['scale']  # [S, H*W, 3] (already activated)
        sh_coeffs = parsed['sh_coeffs']  # [S, H*W, sh_dim]
        sh_coeffs_reshaped = sh_coeffs.reshape(S, image_height * image_width, (sh_degree + 1) ** 2, 3)
        rotations = parsed['rotations']  # [S, H*W, 4] (already normalized)
        opacity = parsed['opacity'].squeeze(-1)  # [S, H*W] (already activated)

        # 2. 从preds中直接获取相机参数并准备渲染参数
        extrinsics = preds['extrinsics']  # [B, S, 4, 4]
        intrinsics = preds['intrinsics']  # [B, S, 3, 3]
        viewmat = extrinsics.permute(1, 0, 2, 3)  # [S, B, 4, 4]
        K = intrinsics.permute(1, 0, 2, 3)  # [S, B, 3, 3]

        # Determine which frames to render
        if sampled_frame_indices is None:
            # Render all frames if not specified
            sampled_frame_indices = list(range(S))
        num_frames_to_render = len(sampled_frame_indices)

        # Pre-allocate tensors for results (only for sampled frames)
        render_colors_tensor = torch.zeros(num_frames_to_render, image_height, image_width, 4, device=gt_rgb.device, dtype=xyz.dtype)
        render_alphas_tensor = torch.zeros(num_frames_to_render, image_height, image_width, 1, device=gt_rgb.device, dtype=xyz.dtype)

        # 获取depth_scale_factor用于voxel pruning
        depth_scale_factor = vggt_batch.get('depth_scale_factor', None)
        if depth_scale_factor is not None and torch.is_tensor(depth_scale_factor):
            depth_scale_factor = depth_scale_factor.item()

        # 对采样的帧进行自渲染
        for render_idx, i in enumerate(sampled_frame_indices):
            # 使用当前帧的3D点进行渲染（不使用velocity，因为是自己渲染自己）
            mean_current = xyz[i]  # 不使用velocity，直接使用当前帧的3D点
            scale_current = scale[i]
            rotations_current = rotations[i]
            opacity_current = opacity[i]
            sh_coeffs_current = sh_coeffs_reshaped[i]

            # 3. 天空处理：与stage2_loss.py保持一致，过滤掉天空区域的gaussian点
            if sky_masks is not None:
                # 将sky_masks转换为与gaussian点相同的格式 [H*W]
                current_sky_mask = sky_masks[0, i].flatten()  # [H*W]
                # 保留非天空区域的gaussian点（sky_mask为0的地方）
                non_sky_mask = ~current_sky_mask.bool()

                if non_sky_mask.sum() > 0:
                    # 过滤天空区域
                    mean_current = mean_current[non_sky_mask]
                    scale_current = scale_current[non_sky_mask]
                    rotations_current = rotations_current[non_sky_mask]
                    opacity_current = opacity_current[non_sky_mask]
                    sh_coeffs_current = sh_coeffs_current[non_sky_mask]
                else:
                    # 如果全是天空，创建空的渲染结果
                    render_color = torch.zeros((image_height, image_width, 4),
                                             device=mean_current.device, dtype=mean_current.dtype)
                    render_alpha = torch.zeros((image_height, image_width, 1),
                                             device=mean_current.device, dtype=mean_current.dtype)
                    render_colors_tensor[render_idx] = render_color
                    render_alphas_tensor[render_idx] = render_alpha
                    continue

            # 应用voxel pruning（如果启用且有点可以渲染）
            if enable_voxel_pruning and mean_current.shape[0] > 0:
                mean_current, scale_current, rotations_current, opacity_current, sh_coeffs_current = prune_gaussians_by_voxel(
                    mean_current, scale_current, rotations_current, opacity_current, sh_coeffs_current,
                    voxel_size=voxel_size,
                    depth_scale_factor=depth_scale_factor
                )

            # 渲染
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
                # 没有点可渲染，创建空的渲染结果
                render_color = torch.zeros((image_height, image_width, 4),
                                         device=gt_rgb.device, dtype=xyz.dtype)
                render_alpha = torch.zeros((image_height, image_width, 1),
                                         device=gt_rgb.device, dtype=xyz.dtype)

            # Write directly to pre-allocated tensor using render_idx
            render_colors_tensor[render_idx] = render_color
            if render_alpha is not None:
                render_alphas_tensor[render_idx] = render_alpha

        # Extract RGB and depth from render results (only sampled frames)
        gt_colors_sampled = gt_rgb[0, sampled_frame_indices]  # [num_frames_to_render, 3, H, W]
        gt_depths_sampled = gt_depths[0, sampled_frame_indices]  # [num_frames_to_render, H, W]

        pred_rgb = render_colors_tensor[..., :3].permute(0, 3, 1, 2)   # [num_frames_to_render, 3, H, W]
        pred_rgb = torch.clamp(pred_rgb, min=0, max=1)
        pred_depth = render_colors_tensor[..., -1]  # [num_frames_to_render, H, W]

        # 如果有天空颜色预测，使用alpha通道进行加权平均（与stage2_loss.py保持一致）
        if pred_sky_colors is not None:
            # 创建一个新的tensor来避免inplace操作
            pred_rgb_with_sky = pred_rgb.clone()
            for render_idx, i in enumerate(sampled_frame_indices):
                # 获取当前帧的渲染RGB和alpha
                rendered_rgb_frame = pred_rgb[render_idx]  # [3, H, W]

                # Extract alpha: [H, W, 1] -> [H, W]
                rendered_alpha = render_alphas_tensor[render_idx, :, :, 0]  # [H, W]

                # 获取天空颜色
                # pred_sky_colors: [B, num_frames, 3, H, W], 使用render_idx索引
                sky_colors_chw = pred_sky_colors[0, render_idx]  # [3, H, W]

                # 使用alpha进行加权平均：final = alpha * rendered + (1-alpha) * sky
                # 这与stage2_loss.py第219-222行的处理方式完全一致
                alpha_3ch = rendered_alpha.unsqueeze(0)  # [1, H, W]
                pred_rgb_with_sky[render_idx] = alpha_3ch * rendered_rgb_frame + (1 - alpha_3ch) * sky_colors_chw

            pred_rgb = pred_rgb_with_sky

        # 计算损失 (only on sampled frames)
        valid_depth_mask = point_masks[0, sampled_frame_indices].bool()  # [num_frames_to_render, H, W]
        depth_loss = F.l1_loss(pred_depth[valid_depth_mask], gt_depths_sampled[valid_depth_mask])
        depth_loss = check_and_fix_inf_nan(depth_loss, "self_depth_loss")

        rgb_loss = F.l1_loss(pred_rgb, gt_colors_sampled)
        rgb_loss = check_and_fix_inf_nan(rgb_loss, "self_rgb_mse")

        # Compute LPIPS loss
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
    Loss to encourage gaussian opacity to be 0 in sky regions and 1 in non-sky regions.

    Args:
        gaussian_params (torch.Tensor): Gaussian parameters [B, S, H, W, 14]
        sky_masks (torch.Tensor): Sky masks [B, S, H, W] where 1 indicates sky regions
        weight (float): Loss weight

    Returns:
        dict: Loss dictionary containing sky_opacity_loss
    """
    # Extract opacity from gaussian parameters
    opacity = gaussian_params[..., 13:14]

    # Create non-sky mask (inverse of sky mask)
    non_sky_masks = 1.0 - sky_masks  # [B, S, H, W]

    # Apply sky mask to get opacity values in sky regions
    sky_opacity = opacity * sky_masks.unsqueeze(-1)  # [B, S, H, W, 1]

    # Apply non-sky mask to get opacity values in non-sky regions
    non_sky_opacity = opacity * non_sky_masks.unsqueeze(-1)  # [B, S, H, W, 1]

    # Compute L1 loss to encourage sky opacity to be 0
    sky_loss = torch.mean(torch.abs(sky_opacity))

    # Compute L1 loss to encourage non-sky opacity to be 1
    non_sky_loss = torch.mean(torch.abs(non_sky_opacity - 1.0))

    # Combine both losses
    total_loss = sky_loss # + non_sky_loss

    return {
        "sky_opacity_loss": total_loss * weight
    }


def check_and_fix_inf_nan(loss_tensor, loss_name, hard_max=100):
    """
    Checks if 'loss_tensor' contains inf or nan. If it does, replace those 
    values with zero and print the name of the loss tensor.

    Args:
        loss_tensor (torch.Tensor): The loss tensor to check.
        loss_name (str): Name of the loss (for diagnostic prints).

    Returns:
        torch.Tensor: The checked and fixed loss tensor, with inf/nan replaced by 0.
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
    # Extract predicted and ground truth components
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

    loss_T = loss_T.clamp(max=100)  # TODO: remove this
    loss_T = loss_T.mean()
    loss_R = loss_R.mean()
    loss_fl = loss_fl.mean()

    return loss_T, loss_R, loss_fl


def normalize_pointcloud(pts3d, valid_mask, eps=1e-3):
    """
    pts3d: B, S, H, W, 3
    valid_mask: B, S, H, W
    """
    dist = pts3d.norm(dim=-1)

    dist_sum = (dist * valid_mask).sum(dim=[1, 2, 3])
    valid_count = valid_mask.sum(dim=[1, 2, 3])

    avg_scale = (dist_sum / (valid_count + eps)).clamp(min=eps, max=1e3)

    # avg_scale = avg_scale.view(-1, 1, 1, 1, 1)

    pts3d = pts3d / avg_scale.view(-1, 1, 1, 1, 1)
    return pts3d, avg_scale


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.

    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Sum gradients and normalize by number of valid pixels
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
    pts3d: B, S, H, W, 3
    pts3d_conf: B, S, H, W
    """
    # gt_pts3d: B, S, H, W, 3
    gt_pts3d = batch['world_points']
    # valid_mask: B, S, H, W
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
    Filters a loss tensor by keeping only values below a certain quantile threshold.
    Also clamps individual values to hard_max.

    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss

    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= 1000:
        # too small, just return
        return loss_tensor

    # Randomly sample if tensor is too large
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[
            :1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values
    loss_tensor = loss_tensor.clamp(max=hard_max)

    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def conf_loss(pts3d, pts3d_conf, gt_pts3d, valid_mask,  batch, normalize_gt=True, normalize_pred=True, gamma=1.0, alpha=0.2, affine_inv=False, gradient_loss=None, valid_range=-1, camera_centric_reg=-1, disable_conf=False, all_mean=False, postfix=""):
    # normalize
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

        # for logging only
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

    # Compute reg_loss for logging
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
        # loss_grad_first_frame and loss_grad_other_frames are already meaned
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
    Computes the normal-based loss by comparing the angle between
    predicted normals and ground-truth normals.

    prediction: (B, H, W, 3) - Predicted 3D coordinates/points
    target:     (B, H, W, 3) - Ground-truth 3D coordinates/points
    mask:       (B, H, W)    - Valid pixel mask (1 = valid, 0 = invalid)

    Returns: scalar (averaged over valid regions)
    """
    pred_normals, pred_valids = point_map_to_normal(
        prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids = point_map_to_normal(
        target,     mask, eps=cos_eps)

    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    # pred_normals and gt_normals are (4, B, H, W, 3)
    # We want to compare corresponding normals where all_valid is True
    dot = torch.sum(pred_normals * gt_normals, dim=-1)  # shape: (4, B, H, W)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot  # shape: (4, B, H, W)

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            gamma = 1.0  # hard coded
            alpha = 0.2  # hard coded

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return loss.mean()
        else:
            return loss.mean()


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    point_map: (B, H, W, 3)  - 3D points laid out in a 2D grid
    mask:      (B, H, W)     - valid pixels (bool)

    Returns:
      normals: (4, B, H, W, 3)  - normal vectors for each of the 4 cross-product directions
      valids:  (4, B, H, W)     - corresponding valid masks
    """

    with torch.cuda.amp.autocast(enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1, 1, 1, 1),
                    mode='constant', value=0).permute(0, 2, 3, 1)

        # Each pixel's neighbors
        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up = pts[:, :-2,  1:-1, :]
        left = pts[:, 1:-1, :-2, :]
        down = pts[:, 2:,   1:-1, :]
        right = pts[:, 1:-1, 2:, :]

        # Direction vectors
        up_dir = up - center
        left_dir = left - center
        down_dir = down - center
        right_dir = right - center

        # Four cross products (shape: B,H,W,3 each)
        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir, up_dir,    dim=-1)  # right x up

        # Validity for each cross-product direction
        # We require that both directions' pixels are valid
        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:,
                                                      1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2] & padded_mask[:,
                                                     1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:,
                                                      1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:] & padded_mask[:,
                                                    1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        # Stack them to shape (4,B,H,W,3), (4,B,H,W)
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize each direction's normal
        # shape is (4, B, H, W, 3), so dim=-1 is the vector dimension
        # clamp_min(eps) to avoid division by zero
        # lengths = torch.norm(normals, dim=-1, keepdim=True).clamp_min(eps)
        # normals = normals / lengths
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

        # Zero out invalid entries so they don't pollute subsequent computations
        # normals = normals * valids.unsqueeze(-1)

    return normals, valids


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    # prediction: B, H, W, C
    # target: B, H, W, C
    # mask: B, H, W

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
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(
            f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Sanitization: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Sanitization: inteporlation
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

    # Sanitization: out
    if out is not None:
        raise ValueError(
            f"Only None value is currently supported for out (got {out})!")

    # Logic
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Rectification: keepdim
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
    Compute cross-entropy loss for semantic segmentation.

    Args:
        segment_logits: [B, S, H, W, 4] - Predicted logits for 4 classes (bg, vehicle, sign, pedestrian+cyclist)
        segment_conf: [B, S, H, W] - Confidence scores for segmentation predictions (unused, kept for API compatibility)
        vggt_batch: Batch dict containing 'segment_label' [B, S, H, W] and 'segment_mask' [B, S, H, W]
        gamma: float - Unused, kept for API compatibility
        disable_conf: bool - Unused, kept for API compatibility
        class_weights: Optional[Tensor] - Class weights for imbalanced classes, shape [4]

    Returns:
        dict: Loss dictionary containing segmentation loss
    """
    with tf32_off():
        # Extract GT labels and masks from batch
        if 'segment_label' not in vggt_batch or 'segment_mask' not in vggt_batch:
            # No segmentation GT available, return zero loss
            return {
                "segment_loss": torch.tensor(0.0, device=segment_logits.device),
            }

        gt_labels = vggt_batch['segment_label']  # [B, S, H, W] with class indices (0-3)
        gt_mask = vggt_batch['segment_mask']  # [B, S, H, W] binary mask (1=valid, 0=invalid)

        B, S, H, W, num_classes = segment_logits.shape
        assert num_classes == 4, f"Expected 4 classes, got {num_classes}"

        # Reshape for loss computation
        segment_logits_flat = segment_logits.reshape(B * S * H * W, num_classes)  # [B*S*H*W, 4]
        gt_labels_flat = gt_labels.reshape(B * S * H * W).long()  # [B*S*H*W]
        gt_mask_flat = gt_mask.reshape(B * S * H * W)  # [B*S*H*W]

        # Apply mask to filter out invalid pixels
        valid_indices = gt_mask_flat > 0.5
        if valid_indices.sum() == 0:
            # No valid pixels, return zero loss
            return {
                "segment_loss": torch.tensor(0.0, device=segment_logits.device),
            }

        segment_logits_valid = segment_logits_flat[valid_indices]  # [N_valid, 4]
        gt_labels_valid = gt_labels_flat[valid_indices]  # [N_valid]

        # Compute cross-entropy loss
        if class_weights is not None:
            class_weights = class_weights.to(segment_logits.device)
        ce_loss = F.cross_entropy(segment_logits_valid, gt_labels_valid, weight=class_weights, reduction='mean')
        ce_loss = check_and_fix_inf_nan(ce_loss, "segment_ce_loss")

        return {
            "segment_loss": ce_loss,
        }


def voxel_quantize_random_sampling(xyz, attributes_dict, sky_mask=None, voxel_size=0.05, gt_scale=1.0):
    """
    体素量化with随机采样 - 每个体素随机选择一个pixel

    Args:
        xyz: [N, 3] - 3D positions in metric scale
        attributes_dict: dict of tensors with shape [N, ...] - attributes to aggregate (gaussian params, depth, etc.)
        sky_mask: [N] or None - Boolean mask, True for sky pixels (to be filtered out). If None, no sky filtering.
        voxel_size: float - Voxel size in metric scale (e.g., 0.05 for 5cm)
        gt_scale: float or tensor - GT scale factor to convert to metric scale

    Returns:
        selected_xyz: [M, 3] - Selected 3D positions
        selected_attributes: dict of tensors [M, ...] - Selected attributes
    """
    with tf32_off():
        # Filter out sky pixels (if sky_mask is provided)
        if sky_mask is not None:
            valid_mask = ~sky_mask.bool()
            xyz_valid = xyz[valid_mask]  # [N_valid, 3]

            if xyz_valid.shape[0] == 0:
                # No valid points, return empty
                empty_result = {k: v[valid_mask] for k, v in attributes_dict.items()}
                return xyz_valid, empty_result
        else:
            # No sky filtering, use all points
            xyz_valid = xyz
            valid_mask = None

        # Convert to metric scale
        xyz_metric = xyz_valid / gt_scale

        # Compute voxel indices
        voxel_indices = torch.floor(xyz_metric / voxel_size).long()  # [N_valid, 3]

        # Create unique voxel IDs using large prime numbers to avoid collisions
        voxel_ids = (voxel_indices[:, 0] * 73856093) ^ \
                    (voxel_indices[:, 1] * 19349663) ^ \
                    (voxel_indices[:, 2] * 83492791)  # [N_valid]

        # Get unique voxel IDs and inverse indices for grouping
        unique_voxel_ids, inverse_indices = torch.unique(
            voxel_ids, return_inverse=True
        )

        # For each point, generate a random value
        # Then select the point with maximum random value in each voxel group
        random_values = torch.rand(len(voxel_ids), device=xyz.device)

        # Use scatter_max to find the index of maximum random value per voxel
        _, selected_indices = torch_scatter.scatter_max(
            random_values, inverse_indices
        )

        # Select xyz and attributes
        selected_xyz = xyz_valid[selected_indices]
        selected_attributes = {}
        if valid_mask is not None:
            # Sky mask was provided, need to filter attributes first
            for k, v in attributes_dict.items():
                v_valid = v[valid_mask]
                selected_attributes[k] = v_valid[selected_indices]
        else:
            # No sky filtering, directly index attributes
            for k, v in attributes_dict.items():
                selected_attributes[k] = v[selected_indices]

        return selected_xyz, selected_attributes

