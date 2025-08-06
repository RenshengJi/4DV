# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# DIRTY VERSION, TO BE CLEANED UP

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dust3r/utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../vggt/'))

from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from dust3r.utils.metrics import compute_lpips
from gsplat.rendering import rasterization
from dust3r.utils.misc import tf32_off
from dust3r.utils.image import scene_flow_to_rgb

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


def velocity_loss(velocity):
    """
    独立的velocity正则化损失函数
    
    Args:
        velocity: [B, S, H*W, 3] 或 [S, H*W, 3] - 预测的速度场
    
    Returns:
        torch.Tensor: velocity正则化损失
    """
    # velocity regularization loss
    velocity = velocity.reshape(-1, 3)  # [S, H*W, 3]
    velocity = torch.sign(velocity) * (torch.exp(torch.abs(velocity)) - 1)
    velocity_loss = F.l1_loss(velocity, torch.zeros_like(velocity))
    velocity_loss = check_and_fix_inf_nan(velocity_loss, "velocity_loss")
    return velocity_loss


def velocity_local_to_global(velocity, extrinsic):
    """
    将速度从局部坐标系（每一帧自己的坐标系）转换到全局坐标系（第一帧的坐标系）
    
    Args:
        velocity: [N, 3] - 局部坐标系下的速度向量
        extrinsic: [B, S, 4, 4] - 相机外参矩阵，其中B=1，S是序列长度
        
    Returns:
        torch.Tensor: [N, 3] - 全局坐标系下的速度向量
    """
    with tf32_off():
        # 获取序列长度
        B, S, _, _ = extrinsic.shape
        assert B == 1, f"Expected batch size 1, got {B}"
        
        # 重塑velocity以匹配帧数
        # velocity: [N, 3] -> [S, H*W, 3] 其中 N = S * H * W
        N = velocity.shape[0]
        H_W = N // S
        velocity_reshaped = velocity.reshape(S, H_W, 3)  # [S, H*W, 3]
        
        # 获取第一帧的变换矩阵作为全局坐标系
        global_transform = extrinsic[0, 0]  # [4, 4] - 第一帧的相机到世界变换
        
        # 初始化全局速度
        global_velocity = torch.zeros_like(velocity_reshaped)
        
        # 对每一帧进行处理
        for frame_idx in range(S):
            # 获取当前帧的变换矩阵
            current_transform = extrinsic[0, frame_idx]  # [4, 4] - 当前帧的相机到世界变换
            
            # 计算从当前帧到全局坐标系的变换
            # 全局坐标系 = 第一帧坐标系
            # 当前帧到全局的变换 = 全局变换的逆 × 当前帧变换
            current_to_global = torch.matmul(torch.linalg.inv(global_transform), current_transform)
            
            # 提取旋转矩阵和平移向量
            R = current_to_global[:3, :3]  # [3, 3]
            t = current_to_global[:3, 3]   # [3]
            
            # 将局部速度转换到全局坐标系
            # 对于速度向量，只需要应用旋转变换，不需要平移
            frame_velocity = velocity_reshaped[frame_idx]  # [H*W, 3]
            global_frame_velocity = torch.matmul(frame_velocity, R.T)  # [H*W, 3]
            
            global_velocity[frame_idx] = global_frame_velocity
        
        # 重塑回原始形状
        global_velocity = global_velocity.reshape(N, 3)
        
        return global_velocity


def cross_render_and_loss(conf, interval, forward_consist_mask, backward_consist_mask, depth, gaussian_params, velocity, pose_enc, extrinsic, intrinsic, gt_rgb, gt_depth, point_masks):
    # gaussian_params: [N, 10]
    # extrinsic, intrinsic: 当前帧相机参数
    # gt_depth, gt_rgb: ground truth

    with tf32_off():

        extrinsic, intrinsic= pose_encoding_to_extri_intri(pose_enc, gt_rgb.shape[-2:])
        extrinsic = torch.cat([extrinsic, torch.tensor([0, 0, 0, 1], device=extrinsic.device)[None,None,None,:].repeat(1,extrinsic.shape[1],1,1)], dim=-2)

        # 1. 构造高斯参数
        B, S, _, image_height, image_width = gt_rgb.shape
        depth = depth.view(depth.shape[0]*depth.shape[1], depth.shape[2], depth.shape[3], 1)
        world_points = depth_to_world_points(depth, intrinsic)
        world_points = world_points.view(world_points.shape[0], world_points.shape[1]*world_points.shape[2], 3)
        extrinsic_inv = torch.linalg.inv(extrinsic)
        xyz = torch.matmul(extrinsic_inv[0,:,:3,:3] , world_points.transpose(-1,-2)).transpose(-1,-2) + \
              extrinsic_inv[0,:,:3,3:4].transpose(-1,-2)
        xyz = xyz.reshape(xyz.shape[0], image_height * image_width, 3)  # [S, H*W, 3]
        velocity = velocity.squeeze(0).reshape(-1, image_height * image_width, 3)  # [S, H*W, 3]
        velocity = torch.sign(velocity) * (torch.exp(torch.abs(velocity)) - 1)
        velocity = velocity_local_to_global(velocity.reshape(-1, 3), extrinsic_inv).reshape(S, image_height * image_width, 3)

        gaussian_params = gaussian_params.reshape(1, -1, image_height * image_width, 14)  # [S, H*W, 14]
        scale = gaussian_params[0, ..., 3:6]
        scale = (0.05 * torch.exp(scale)).clamp_max(0.3)  # [S, H*W, 3]
        color = gaussian_params[0, ..., 6:9].unsqueeze(-2)  # [S, H*W, 1, 3]
        rotations = gaussian_params[0, ..., 9:13] 
        
        # Add safety checks for rotation normalization to prevent division by zero
        rotation_norms = torch.norm(rotations, dim=-1, keepdim=True)
        # Add small epsilon to prevent division by zero
        rotation_norms = torch.clamp(rotation_norms, min=1e-8)
        rotations = rotations / rotation_norms
        
        opacity = gaussian_params[0, ..., 13:14].sigmoid().squeeze(-1)  # [S, H*W]


        viewmat = extrinsic.permute(1, 0, 2, 3) 
        K = intrinsic.permute(1, 0, 2, 3) 


        render_colors = []
        gt_colors = []
        gt_depths = []
        masks = []
        masks_depth = []
        source_rgb = []


        forward_consist_mask = torch.ones_like(conf[:, :, None, :, :]) if forward_consist_mask is None else forward_consist_mask
        backward_consist_mask = torch.ones_like(conf[:, :, None, :, :]) if backward_consist_mask is None else backward_consist_mask


        # forward
        for i in range(S - interval):
            mask = (forward_consist_mask & conf[:, :, None, :, :])[0,i].flatten() 
            if mask.sum() == 0:
                continue
            mean_moved = xyz[i] + velocity[i]
            render_color, _, _ = rasterization(
                mean_moved[mask], rotations[i][mask], scale[i][mask], opacity[i][mask], color[i][mask],
                viewmat[i+interval], K[i+interval], image_width, image_height,
                sh_degree=0, render_mode="RGB+ED",
                radius_clip=0, near_plane=0.0001,
                far_plane=1000.0,
                eps2d=0.3,
            )
            render_colors.append(render_color[0])
            gt_colors.append(gt_rgb[0,i+interval])
            source_rgb.append(gt_rgb[0,i])
            gt_depths.append(depth[i+interval].squeeze(-1))
            masks.append((backward_consist_mask & conf[:,:,None,:,:])[0,i+interval])
            masks_depth.append((backward_consist_mask.squeeze(2) & conf)[0,i+interval])


        # # backward
        # for i in range(interval, S):
        #     mask = (backward_consist_mask & conf[:, :, None, :, :])[0,i].flatten() 
        #     mean_moved = xyz[i] - velocity[i]
        #     render_color, _, _ = rasterization(
        #         mean_moved[mask], rotations[i][mask], scale[i][mask], opacity[i][mask], color[i][mask],
        #         viewmat[i-interval], K[i-interval], image_width, image_height,
        #         sh_degree=0, render_mode="RGB+ED",
        #         radius_clip=0, near_plane=0.0001,
        #         far_plane=1000.0,
        #         eps2d=0.0,
        #     )
        #     render_colors.append(render_color[0])
        #     gt_colors.append(gt_rgb[0,i-interval])
        #     source_rgb.append(gt_rgb[0,i])
        #     # gt_depths.append(gt_depth[0,i-interval])
        #     gt_depths.append(depth[i-interval].squeeze(-1))
        #     masks.append((forward_consist_mask & conf[:,:,None,:,:])[0,i-interval])
        #     # masks_depth.append((forward_consist_mask.squeeze(2) & conf)[0,i-interval] & point_masks[0,i-interval])
        #     masks_depth.append((forward_consist_mask.squeeze(2) & conf)[0,i-interval])

        if len(render_colors) == 0:
            return {
                "loss_render_rgb": torch.tensor(0.0, device=gt_rgb.device, requires_grad=True),
                "loss_render_lpips": torch.tensor(0.0, device=gt_rgb.device, requires_grad=True),
                "loss_render_depth": torch.tensor(0.0, device=gt_rgb.device, requires_grad=True),
            }, {}
        
        render_colors = torch.stack(render_colors, dim=0) 
        gt_colors = torch.stack(gt_colors, dim=0)  # [S, 3, H, W]
        source_rgb = torch.stack(source_rgb, dim=0)  # [S, 3, H, W]
        gt_depths = torch.stack(gt_depths, dim=0)  # [S, H, W]
        masks = torch.stack(masks, dim=0)  # [S, H, W]
        masks_depth = torch.stack(masks_depth, dim=0)  # [S, H, W]
        pred_rgb = render_colors[..., :3].permute(0,3,1,2)   # [S, 3, H, W]
        pred_rgb = torch.clamp(pred_rgb, min=0, max=1)
        pred_depth = render_colors[..., -1]  # [S, H, W]

        depth_loss = F.l1_loss(pred_depth[masks_depth], gt_depths[masks_depth])
        depth_loss = check_and_fix_inf_nan(depth_loss, "depth_loss")

        rgb_loss = F.l1_loss(pred_rgb[masks.repeat(1,3,1,1)], gt_colors[masks.repeat(1,3,1,1)])
        rgb_loss = check_and_fix_inf_nan(rgb_loss, "rgb_mse")

        pred_rgb_lpips = gt_colors.clone()
        pred_rgb_lpips[masks.repeat(1,3,1,1)] = pred_rgb[masks.repeat(1,3,1,1)]
        
        lpips_loss = compute_lpips(pred_rgb_lpips, gt_colors).mean()
        lpips_loss = check_and_fix_inf_nan(lpips_loss, "lpips_loss")


        gaussian_loss_dict = {
            "loss_render_rgb": rgb_loss,
            "loss_render_lpips": lpips_loss,
            "loss_render_depth": depth_loss,
        }

        velocity_img_forward = scene_flow_to_rgb(velocity.reshape(S, image_height, image_width, 3), 0.03).permute(0,3,1,2)
        velocity_img_backward = scene_flow_to_rgb(-velocity.reshape(S, image_height, image_width, 3), 0.03).permute(0,3,1,2)
        velocity_img = torch.cat([velocity_img_forward[:S-interval], velocity_img_backward[interval:]], dim=0)

        img_dict = {
            "source_rgb": source_rgb,
            "target_rgb_pred": pred_rgb,
            "target_rgb_gt": gt_colors,
            "target_depth_pred": pred_depth.unsqueeze(1),
            "target_depth_gt": gt_depths.unsqueeze(1),
            "velocity": velocity_img_forward,
        }

        return gaussian_loss_dict, img_dict
    


def flow_loss(conf, interval, forward_flow, backward_flow, forward_consist_mask, backward_consist_mask, depth, velocity, pose_enc, extrinsic, intrinsic, gt_rgb):
    # gaussian_params: [N, 10]
    # extrinsic, intrinsic: 当前帧相机参数
    # gt_depth, gt_rgb: ground truth

    with tf32_off():

        extrinsic, intrinsic= pose_encoding_to_extri_intri(pose_enc, gt_rgb.shape[-2:])
        extrinsic = torch.cat([extrinsic, torch.tensor([0, 0, 0, 1], device=extrinsic.device)[None,None,None,:].repeat(1,extrinsic.shape[1],1,1)], dim=-2)

        # 1. gaussian means
        depth = depth.view(depth.shape[0]*depth.shape[1], depth.shape[2], depth.shape[3], 1)
        world_points = depth_to_world_points(depth, intrinsic)
        world_points = world_points.view(world_points.shape[0], world_points.shape[1]*world_points.shape[2], 3)

        extrinsic_inv = torch.linalg.inv(extrinsic)
        xyz = torch.matmul(extrinsic_inv[0,:,:3,:3] , world_points.transpose(-1,-2)).transpose(-1,-2) + \
              extrinsic_inv[0,:,:3,3:4].transpose(-1,-2)
        mean = xyz.reshape(-1, 3)
        B, S, _, H, W = forward_flow.shape
        gaussian_means = mean.reshape(B, S, H, W, 3).permute(0, 1, 4, 2, 3).contiguous()

        # 2. gaussian velocity
        velocity = velocity.reshape(-1, 3)
        velocity = torch.sign(velocity) * (torch.exp(torch.abs(velocity)) - 1)
        velocity = velocity_local_to_global(velocity, extrinsic_inv)
        gaussian_fwd_vel = velocity.reshape(B, S, H, W, 3).permute(0, 1, 4, 2, 3).contiguous()
        gaussian_bwd_vel = - velocity.reshape(B, S, H, W, 3).permute(0, 1, 4, 2, 3).contiguous()

        # forward loss
        warped_means, target_means = warp_gaussian(
            forward_flow,
            forward_consist_mask & conf[:, :, None, :, :],
            gaussian_means,
            gaussian_fwd_vel,
            S, H, W,
            "forward",
            interval=interval
        )
        forward_loss = F.smooth_l1_loss(warped_means, target_means)
        forward_loss = check_and_fix_inf_nan(forward_loss, "forward_loss")

        # backward loss
        warped_means, target_means = warp_gaussian(
            backward_flow,
            backward_consist_mask & conf[:, :, None, :, :],
            gaussian_means,
            gaussian_bwd_vel,
            S, H, W,
            "backward",
            interval=interval
        )
        backward_loss = F.smooth_l1_loss(warped_means, target_means)
        backward_loss = check_and_fix_inf_nan(backward_loss, "backward_loss")

        flow_loss_dict = {
            "forward_loss": forward_loss,
            "backward_loss": backward_loss,
        }

        return flow_loss_dict
    

def warp_gaussian(flow, mask, gaussian_means, gaussian_vel, T, H, W, direction="forward",interval=1):
    if direction == "forward":
        mask[:, -interval:, :, :] = False
    elif direction == "backward":
        mask[:, 0:interval, :, :] = False
    else:
        raise ValueError("direction must be forward or backward")
    
    # # for debug : mask全为True
    # mask = torch.ones_like(mask)

    inds = torch.nonzero(mask, as_tuple=True)
    init_pos_b, init_pos_t, _, init_pos_h, init_pos_w = inds
    if len(init_pos_b) == 0:
        return torch.zeros((0, 3), device=gaussian_means.device), \
               torch.zeros((0, 3), device=gaussian_means.device)

    flow = flow[init_pos_b, init_pos_t, :, init_pos_h, init_pos_w]
    warped_pos_b = init_pos_b
    if direction == "forward":
        warped_pos_t = (init_pos_t + interval).clamp(min=0, max=T-1)
    elif direction == "backward":
        warped_pos_t = (init_pos_t - interval).clamp(min=0, max=T-1)
    else:
        raise ValueError(f"Unknown direction {direction} for warping gaussian means")
    warped_pos_h = (init_pos_h + flow[:, 1]).clamp(min=0, max=H-1).round().long()
    warped_pos_w = (init_pos_w + flow[:, 0]).clamp(min=0, max=W-1).round().long()
    warped_gaussian_means = gaussian_means[init_pos_b, init_pos_t, :, init_pos_h, init_pos_w] + \
                           gaussian_vel[init_pos_b, init_pos_t, :, init_pos_h, init_pos_w]
    target_gaussian_means = gaussian_means[warped_pos_b, warped_pos_t, :, warped_pos_h, warped_pos_w]
    return warped_gaussian_means, target_gaussian_means



def sam2_velocity_consistency_loss_impl(images, velocity, sam2_model, device):
    """
    Compute velocity consistency loss using SAM2 masks with fully vectorized operations.
    
    Args:
        images: [B, S, 3, H, W] - input images in [0, 1] range
        velocity: [B, S, H, W, 3] - predicted velocity
        sam2_model: SAM2AutomaticMaskGenerator instance
        device: torch device
    
    Returns:
        dict: loss dictionary containing sam2_velocity_consistency_loss
    """
    import numpy as np
    
    B, S, C, H, W = images.shape
    
    # Convert images to numpy for SAM2
    images = images.permute(0, 1, 3, 4, 2) # [B, S, 3, H, W] -> [B, S, H, W, 3]
    images_np = (images * 255).cpu().numpy().astype(np.uint8)  # [0, 1] -> [0, 255]
    
    # Reshape for batch processing
    images_flat = images_np.reshape(-1, H, W, 3)  # [B*S, H, W, 3]
    velocity_flat = velocity.reshape(-1, H, W, 3)  # [B*S, H, W, 3]
    
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_masks = 0
    
    # Process all images at once using batch inference
    try:
        # Batch process all images through SAM2
        # Note: SAM2AutomaticMaskGenerator doesn't support true batch processing,
        # but we can optimize by processing in chunks to reduce overhead
        batch_size = min(24, images_flat.shape[0])  # Process in chunks of 4
        all_masks = []
        
        for chunk_start in range(0, images_flat.shape[0], batch_size):
            chunk_end = min(chunk_start + batch_size, images_flat.shape[0])
            chunk_images = images_flat[chunk_start:chunk_end]
            
            # Process chunk of images
            chunk_masks = []
            for i, image in enumerate(chunk_images):
                try:
                    masks = sam2_model.generate(image)
                    chunk_masks.append(masks)
                except Exception as e:
                    print(f"Error processing SAM2 mask for image {chunk_start + i}: {e}")
                    chunk_masks.append([])
            
            all_masks.extend(chunk_masks)
        
        # Vectorized processing of all masks
        all_consistency_losses = []
        
        for batch_idx, masks in enumerate(all_masks):
            if len(masks) == 0:
                continue
                
            # Get velocity for this batch
            frame_velocity = velocity_flat[batch_idx]  # [H, W, 3]
            
            # Convert all masks to tensors at once
            mask_tensors = []
            for mask_info in masks:
                mask = mask_info['segmentation']  # [H, W] boolean array
                mask_tensor = torch.from_numpy(mask).to(device, dtype=torch.bool)
                mask_tensors.append(mask_tensor)
            
            if len(mask_tensors) == 0:
                continue
                
            # Stack all masks for vectorized processing
            stacked_masks = torch.stack(mask_tensors)  # [num_masks, H, W]
            
            # Filter masks by size
            mask_sizes = stacked_masks.sum(dim=(1, 2))  # [num_masks]
            valid_mask_indices = mask_sizes >= 10  # Filter small masks
            
            if not valid_mask_indices.any():
                continue
                
            # Get valid masks
            valid_masks = stacked_masks[valid_mask_indices]  # [valid_num_masks, H, W]
            
            # Vectorized velocity consistency computation
            # Expand velocity to match all valid masks
            velocity_expanded = frame_velocity.unsqueeze(0).expand(valid_masks.shape[0], -1, -1, -1)  # [valid_num_masks, H, W, 3]
            
            # Compute consistency for all masks at once
            batch_consistency_losses = []
            
            for i, mask in enumerate(valid_masks):
                # Get velocities for this specific mask
                mask_velocities = velocity_expanded[i][mask]  # [mask_pixels, 3]
                
                # Check for NaN or Inf values
                if torch.isnan(mask_velocities).any() or torch.isinf(mask_velocities).any():
                    continue
                
                # Compute velocity consistency within mask
                velocity_mean = mask_velocities.mean(dim=0, keepdim=True)  # [1, 3]
                
                # Check for NaN or Inf in mean
                if torch.isnan(velocity_mean).any() or torch.isinf(velocity_mean).any():
                    continue
                
                velocity_variance = ((mask_velocities - velocity_mean) ** 2).mean()
                
                # Check for NaN or Inf in variance
                if torch.isnan(velocity_variance) or torch.isinf(velocity_variance):
                    continue
                
                batch_consistency_losses.append(velocity_variance)
            
            # Add batch losses to total
            if batch_consistency_losses:
                all_consistency_losses.extend(batch_consistency_losses)
                
    except Exception as e:
        print(f"Error in vectorized SAM2 processing: {e}")
    
    # Compute final loss
    if all_consistency_losses:
        total_loss = torch.stack(all_consistency_losses).sum()
        total_masks = len(all_consistency_losses)
        avg_loss = total_loss / total_masks
        
        # Final safety check
        if torch.isnan(avg_loss) or torch.isinf(avg_loss):
            print(f"Warning: Final loss is NaN or Inf, setting to 0")
            avg_loss = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        avg_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    return {
        "sam2_velocity_consistency_loss": avg_loss,
        "sam2_total_masks": total_masks
    }


def self_render_and_loss(depth, gaussian_params, pose_enc, extrinsic, intrinsic, gt_rgb):
    """
    自渲染损失函数：每一帧自己进行渲染与监督，不使用任何mask
    
    Args:
        depth: [B, S, H, W] 深度图
        gaussian_params: [B, S, H*W, 14] 高斯参数
        velocity: [B, S, H, W, 3] 速度场
        pose_enc: [B, S, 7] 姿态编码
        extrinsic: [B, S, 4, 4] 外参矩阵
        intrinsic: [B, S, 3, 3] 内参矩阵
        gt_rgb: [B, S, 3, H, W] 真实RGB图像
        gt_depth: [B, S, H, W] 真实深度图
    
    Returns:
        dict: 损失字典
        dict: 图像字典（用于可视化）
    """
    with tf32_off():
        # 从姿态编码获取相机参数
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, gt_rgb.shape[-2:])
        extrinsic = torch.cat([extrinsic, torch.tensor([0, 0, 0, 1], device=extrinsic.device)[None,None,None,:].repeat(1,extrinsic.shape[1],1,1)], dim=-2)

        # 获取图像尺寸
        B, S, _, image_height, image_width = gt_rgb.shape
        
        # 1. 构造高斯参数
        depth = depth.view(depth.shape[0]*depth.shape[1], depth.shape[2], depth.shape[3], 1)
        world_points = depth_to_world_points(depth, intrinsic)
        world_points = world_points.view(world_points.shape[0], world_points.shape[1]*world_points.shape[2], 3)
        
        # 转换到相机坐标系
        extrinsic_inv = torch.linalg.inv(extrinsic)
        xyz = torch.matmul(extrinsic_inv[0,:,:3,:3] , world_points.transpose(-1,-2)).transpose(-1,-2) + \
              extrinsic_inv[0,:,:3,3:4].transpose(-1,-2)
        xyz = xyz.reshape(xyz.shape[0], image_height * image_width, 3)  # [S, H*W, 3]
        
        # 处理高斯参数
        gaussian_params = gaussian_params.reshape(1, -1, image_height * image_width, 14)  # [S, H*W, 14]
        scale = gaussian_params[0, ..., 3:6]
        scale = (0.05 * torch.exp(scale)).clamp_max(0.3)  # [S, H*W, 3]
        color = gaussian_params[0, ..., 6:9].unsqueeze(-2)  # [S, H*W, 1, 3]
        rotations = gaussian_params[0, ..., 9:13] 
        
        # 旋转归一化
        rotation_norms = torch.norm(rotations, dim=-1, keepdim=True)
        rotation_norms = torch.clamp(rotation_norms, min=1e-8)
        rotations = rotations / rotation_norms
        
        opacity = gaussian_params[0, ..., 13:14].sigmoid().squeeze(-1)  # [S, H*W]

        # 准备渲染参数
        viewmat = extrinsic.permute(1, 0, 2, 3) 
        K = intrinsic.permute(1, 0, 2, 3) 

        render_colors = []
        gt_colors = []
        gt_depths = []

        # 对每一帧进行自渲染
        for i in range(S):
            # 使用当前帧的3D点进行渲染（不使用velocity，因为是自己渲染自己）
            mean_current = xyz[i]  # 不使用velocity，直接使用当前帧的3D点
            
            # 渲染当前帧
            render_color, _, _ = rasterization(
                mean_current, rotations[i], scale[i], opacity[i], color[i],
                viewmat[i], K[i], image_width, image_height,
                sh_degree=0, render_mode="RGB+ED",
                radius_clip=0, near_plane=0.0001,
                far_plane=1000.0,
                eps2d=0.3,
            )
            render_colors.append(render_color[0])
            gt_colors.append(gt_rgb[0, i])
            gt_depths.append(depth[i].squeeze(-1))

        if len(render_colors) == 0:
            return {
                "loss_self_render_rgb": torch.tensor(0.0, device=gt_rgb.device, requires_grad=True),
                "loss_self_render_lpips": torch.tensor(0.0, device=gt_rgb.device, requires_grad=True),
                "loss_self_render_depth": torch.tensor(0.0, device=gt_rgb.device, requires_grad=True),
            }, {}
        
        # 堆叠结果
        render_colors = torch.stack(render_colors, dim=0) 
        gt_colors = torch.stack(gt_colors, dim=0)  # [S, 3, H, W]
        gt_depths = torch.stack(gt_depths, dim=0)  # [S, H, W]
        
        pred_rgb = render_colors[..., :3].permute(0,3,1,2)   # [S, 3, H, W]
        pred_rgb = torch.clamp(pred_rgb, min=0, max=1)
        pred_depth = render_colors[..., -1]  # [S, H, W]

        # 计算损失（不使用mask，直接计算全图损失）
        depth_loss = F.l1_loss(pred_depth, gt_depths)
        depth_loss = check_and_fix_inf_nan(depth_loss, "self_depth_loss")

        rgb_loss = F.l1_loss(pred_rgb, gt_colors)
        rgb_loss = check_and_fix_inf_nan(rgb_loss, "self_rgb_mse")
        
        lpips_loss = compute_lpips(pred_rgb, gt_colors).mean()
        lpips_loss = check_and_fix_inf_nan(lpips_loss, "self_lpips_loss")


        self_loss_dict = {
            "loss_self_render_rgb": rgb_loss,
            "loss_self_render_lpips": lpips_loss,
            "loss_self_render_depth": depth_loss,
        }

        img_dict = {
            "self_rgb_pred": pred_rgb,
            "self_rgb_gt": gt_colors,
            "self_depth_pred": pred_depth.unsqueeze(1),
            "self_depth_gt": gt_depths.unsqueeze(1),
        }

        return self_loss_dict, img_dict


def sky_opacity_loss(gaussian_params, sky_masks, weight=1.0):
    """
    Loss to encourage gaussian opacity to be 0 in sky regions and 1 in non-sky regions.
    
    Args:
        gaussian_params (torch.Tensor): Gaussian parameters [B, S, H, W, 15]
        sky_masks (torch.Tensor): Sky masks [B, S, H, W] where 1 indicates sky regions
        weight (float): Loss weight
        
    Returns:
        dict: Loss dictionary containing sky_opacity_loss
    """
    # Extract opacity from gaussian parameters
    opacity = gaussian_params[..., 13:14].sigmoid()

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
    total_loss = sky_loss + non_sky_loss
    
    return {
        "sky_opacity_loss": total_loss * weight
    }


def sky_color_loss(pred_sky_colors, gt_images, sky_masks, weight=1.0):
    """
    Loss to encourage predicted sky colors to match GT images in sky regions.
    
    Args:
        pred_sky_colors (torch.Tensor): Predicted sky colors [B, S, H, W, 3]
        gt_images (torch.Tensor): Ground truth images [B, S, 3, H, W] in range [0, 1]
        sky_masks (torch.Tensor): Sky masks [B, S, H, W] where 1 indicates sky regions
        weight (float): Loss weight
        
    Returns:
        dict: Loss dictionary containing sky_color_loss
    """
    # Convert GT images to [B, S, H, W, 3] format
    gt_images = gt_images.permute(0, 1, 3, 4, 2)  # [B, S, H, W, 3]
    
    # Apply sky mask to get GT colors in sky regions
    gt_sky_colors = gt_images * sky_masks.unsqueeze(-1)  # [B, S, H, W, 3]
    pred_sky_colors_masked = pred_sky_colors * sky_masks.unsqueeze(-1)  # [B, S, H, W, 3]
    
    # Compute L1 loss between predicted and GT sky colors
    color_loss = torch.mean(torch.abs(pred_sky_colors_masked - gt_sky_colors))
    
    # Optional: Add L2 loss for smoother gradients
    color_loss_l2 = torch.mean((pred_sky_colors_masked - gt_sky_colors) ** 2)
    
    total_loss = color_loss + 0.1 * color_loss_l2
    
    return {
        "sky_color_loss": total_loss * weight
    }



def check_and_fix_inf_nan(loss_tensor, loss_name, hard_max = 100):
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



def camera_loss(pred_pose_enc_list, batch, loss_type="l1", gamma=0.6, pose_encoding_type="absT_quaR_FoV", weight_T = 1.0, weight_R = 1.0, weight_fl = 0.5, frame_num = -100):
    # Extract predicted and ground truth components
    mask_valid = batch['point_masks']
    
    batch_valid_mask = mask_valid[:, 0].sum(dim=[-1, -2]) > 100
    num_predictions = len(pred_pose_enc_list)

    gt_extrinsic = batch['extrinsics']
    gt_intrinsic = batch['intrinsics']
    image_size_hw = batch['images'].shape[-2:]

    gt_pose_encoding = extri_intri_to_pose_encoding(gt_extrinsic, gt_intrinsic, image_size_hw, pose_encoding_type=pose_encoding_type)

    loss_T = loss_R = loss_fl = 0

    for i in range(num_predictions):
        i_weight = gamma ** (num_predictions - i - 1)

        cur_pred_pose_enc = pred_pose_enc_list[i]

        if batch_valid_mask.sum() == 0:
            loss_T_i = (cur_pred_pose_enc * 0).mean()
            loss_R_i = (cur_pred_pose_enc * 0).mean()
            loss_fl_i = (cur_pred_pose_enc * 0).mean()
        else:
            if frame_num>0:
                loss_T_i, loss_R_i, loss_fl_i = camera_loss_single(cur_pred_pose_enc[batch_valid_mask][:, :frame_num].clone(), gt_pose_encoding[batch_valid_mask][:, :frame_num].clone(), loss_type=loss_type)
            else:
                loss_T_i, loss_R_i, loss_fl_i = camera_loss_single(cur_pred_pose_enc[batch_valid_mask].clone(), gt_pose_encoding[batch_valid_mask].clone(), loss_type=loss_type)
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

    with torch.no_grad():
        # compute auc
        last_pred_pose_enc = pred_pose_enc_list[-1]

        last_pred_extrinsic, _ = pose_encoding_to_extri_intri(last_pred_pose_enc.detach(), image_size_hw, pose_encoding_type=pose_encoding_type, build_intrinsics=False)

        rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(last_pred_extrinsic.float(), gt_extrinsic.float(), gt_extrinsic.device)


        if rel_rangle_deg.numel() == 0 and rel_tangle_deg.numel() == 0:
            rel_rangle_deg = torch.FloatTensor([0]).to(gt_extrinsic.device).to(gt_extrinsic.dtype)
            rel_tangle_deg = torch.FloatTensor([0]).to(gt_extrinsic.device).to(gt_extrinsic.dtype)

        thresholds = [5, 15]
        for threshold in thresholds:
            loss_dict[f"Rac_{threshold}"] = (rel_rangle_deg < threshold).float().mean()
            loss_dict[f"Tac_{threshold}"] = (rel_tangle_deg < threshold).float().mean()

        _, normalized_histogram = calculate_auc(
            rel_rangle_deg, rel_tangle_deg, max_threshold=30, return_list=True
        )

        auc_thresholds = [30, 10, 5, 3]
        for auc_threshold in auc_thresholds:
            cur_auc = torch.cumsum(
                normalized_histogram[:auc_threshold], dim=0
            ).mean()
            loss_dict[f"Auc_{auc_threshold}"] = cur_auc

    return loss_dict, last_pred_extrinsic


def camera_loss_single(cur_pred_pose_enc, gt_pose_encoding, loss_type="l1"):
    if loss_type == "l1":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).abs()
        loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).abs()
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).abs()
    elif loss_type == "l2":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).norm(dim=-1)
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).norm(dim=-1)
    elif loss_type == "huber":
        loss_T = huber_loss(cur_pred_pose_enc[..., :3], gt_pose_encoding[..., :3])
        loss_R = huber_loss(cur_pred_pose_enc[..., 3:7], gt_pose_encoding[..., 3:7])
        loss_fl = huber_loss(cur_pred_pose_enc[..., 7:], gt_pose_encoding[..., 7:])
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_fl = check_and_fix_inf_nan(loss_fl, "loss_fl")

    loss_T = loss_T.clamp(max=100) # TODO: remove this
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

    dist_sum = (dist * valid_mask).sum(dim=[1,2,3])
    valid_count = valid_mask.sum(dim=[1,2,3])

    avg_scale = (dist_sum / (valid_count + eps)).clamp(min=eps, max=1e3)

    # avg_scale = avg_scale.view(-1, 1, 1, 1, 1)

    pts3d = pts3d / avg_scale.view(-1, 1, 1, 1, 1)
    return pts3d, avg_scale


def depth_loss(depth, depth_conf, batch, gamma=1.0, alpha=0.2, loss_type="conf", predict_disparity=False, affine_inv=False, gradient_loss= None, valid_range=-1, disable_conf=False, all_mean=False, **kwargs):

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
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
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

    loss_reg_first_frame, loss_reg_other_frames, loss_grad_first_frame, loss_grad_other_frames = reg_loss(pts3d, gt_pts3d, valid_mask, gradient_loss=gradient_loss)


    if disable_conf:
        conf_loss_first_frame = gamma * loss_reg_first_frame
        conf_loss_other_frames = gamma * loss_reg_other_frames
    else:
        first_frame_conf = pts3d_conf[:, 0:1, ...]
        other_frames_conf = pts3d_conf[:, 1:, ...]
        first_frame_mask = valid_mask[:, 0:1, ...]
        other_frames_mask = valid_mask[:, 1:, ...]

        conf_loss_first_frame = gamma * loss_reg_first_frame * first_frame_conf[first_frame_mask] - alpha * torch.log(first_frame_conf[first_frame_mask])
        conf_loss_other_frames = gamma * loss_reg_other_frames * other_frames_conf[other_frames_mask] - alpha * torch.log(other_frames_conf[other_frames_mask])


    if conf_loss_first_frame.numel() >0 and conf_loss_other_frames.numel() >0:
        if valid_range>0:
            conf_loss_first_frame = filter_by_quantile(conf_loss_first_frame, valid_range)
            conf_loss_other_frames = filter_by_quantile(conf_loss_other_frames, valid_range)

        conf_loss_first_frame = check_and_fix_inf_nan(conf_loss_first_frame, f"conf_loss_first_frame{postfix}")
        conf_loss_other_frames = check_and_fix_inf_nan(conf_loss_other_frames, f"conf_loss_other_frames{postfix}")
    else:
        conf_loss_first_frame = pts3d * 0
        conf_loss_other_frames = pts3d * 0
        print("No valid conf loss", batch["seq_name"])


    if all_mean and conf_loss_first_frame.numel() > 0 and conf_loss_other_frames.numel() > 0:
        all_conf_loss = torch.cat([conf_loss_first_frame, conf_loss_other_frames])
        conf_loss = all_conf_loss.mean() if all_conf_loss.numel() > 0 else 0

        # for logging only
        conf_loss_first_frame = conf_loss_first_frame.mean() if conf_loss_first_frame.numel() > 0 else 0
        conf_loss_other_frames = conf_loss_other_frames.mean() if conf_loss_other_frames.numel() > 0 else 0
    else:
        conf_loss_first_frame = conf_loss_first_frame.mean() if conf_loss_first_frame.numel() > 0 else 0
        conf_loss_other_frames = conf_loss_other_frames.mean() if conf_loss_other_frames.numel() > 0 else 0

        conf_loss = conf_loss_first_frame + conf_loss_other_frames


    # Verified that the loss is the same

    loss_dict = {
        f"loss_conf{postfix}": conf_loss,
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


    loss_reg_first_frame = torch.norm(first_frame_gt_pts3d[first_frame_mask] - first_frame_pts3d[first_frame_mask], dim=-1)
    loss_reg_other_frames = torch.norm(other_frames_gt_pts3d[other_frames_mask] - other_frames_pts3d[other_frames_mask], dim=-1)

    if gradient_loss == "grad":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(first_frame_pts3d.reshape(bb*ss, hh, ww, nc), first_frame_gt_pts3d.reshape(bb*ss, hh, ww, nc), first_frame_mask.reshape(bb*ss, hh, ww))
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(other_frames_pts3d.reshape(bb*ss, hh, ww, nc), other_frames_gt_pts3d.reshape(bb*ss, hh, ww, nc), other_frames_mask.reshape(bb*ss, hh, ww))
    elif gradient_loss == "grad_impl2":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(first_frame_pts3d.reshape(bb*ss, hh, ww, nc), first_frame_gt_pts3d.reshape(bb*ss, hh, ww, nc), first_frame_mask.reshape(bb*ss, hh, ww), gradient_loss_fn=gradient_loss_impl2)
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(other_frames_pts3d.reshape(bb*ss, hh, ww, nc), other_frames_gt_pts3d.reshape(bb*ss, hh, ww, nc), other_frames_mask.reshape(bb*ss, hh, ww), gradient_loss_fn=gradient_loss_impl2)
    elif gradient_loss == "normal":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(first_frame_pts3d.reshape(bb*ss, hh, ww, nc), first_frame_gt_pts3d.reshape(bb*ss, hh, ww, nc), first_frame_mask.reshape(bb*ss, hh, ww), gradient_loss_fn=normal_loss, scales=3)
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(other_frames_pts3d.reshape(bb*ss, hh, ww, nc), other_frames_gt_pts3d.reshape(bb*ss, hh, ww, nc), other_frames_mask.reshape(bb*ss, hh, ww), gradient_loss_fn=normal_loss, scales=3)
    else:
        loss_grad_first_frame = 0
        loss_grad_other_frames = 0


    loss_reg_first_frame = check_and_fix_inf_nan(loss_reg_first_frame, "loss_reg_first_frame")
    loss_reg_other_frames = check_and_fix_inf_nan(loss_reg_other_frames, "loss_reg_other_frames")

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
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = point_map_to_normal(target,     mask, eps=cos_eps)

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

            gamma = 1.0 # hard coded
            alpha = 0.2 # hard coded

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
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

        # Each pixel's neighbors
        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2 , :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        # Direction vectors
        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        # Four cross products (shape: B,H,W,3 each)
        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

        # Validity for each cross-product direction
        # We require that both directions' pixels are valid
        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        # Stack them to shape (4,B,H,W,3), (4,B,H,W)
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

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


def gradient_loss_multi_scale(prediction, target, mask, scales=4, gradient_loss_fn = gradient_loss, conf=None):
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
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

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
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

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





def vggt_distillation_loss(student_preds, teacher_preds, weight_pose=1.0, weight_depth=1.0, weight_depth_conf=1.0):
    """
    VGGT teacher-student distillation loss for pose_enc, depth, and depth_conf.
    
    Args:
        student_preds (dict): Student model predictions containing 'pose_enc', 'depth', and 'depth_conf'
        teacher_preds (dict): Teacher model predictions containing 'pose_enc', 'depth', and 'depth_conf'
        weight_pose (float): Weight for pose distillation loss
        weight_depth (float): Weight for depth distillation loss
        weight_depth_conf (float): Weight for depth confidence distillation loss
        
    Returns:
        dict: Loss dictionary containing distillation losses
    """
    loss_dict = {}
    
    # Pose encoding distillation loss
    if "pose_enc" in student_preds and "pose_enc" in teacher_preds:
        pose_loss = F.mse_loss(student_preds["pose_enc"], teacher_preds["pose_enc"])
        pose_loss = check_and_fix_inf_nan(pose_loss, "pose_distillation_loss")
        loss_dict["loss_pose_distillation"] = pose_loss * weight_pose
    else:
        loss_dict["loss_pose_distillation"] = torch.tensor(0.0, device=student_preds.get("pose_enc", torch.tensor(0.0)).device, requires_grad=True)
    
    # Depth distillation loss
    if "depth" in student_preds and "depth" in teacher_preds:
        depth_loss = F.mse_loss(student_preds["depth"], teacher_preds["depth"])
        depth_loss = check_and_fix_inf_nan(depth_loss, "depth_distillation_loss")
        loss_dict["loss_depth_distillation"] = depth_loss * weight_depth
    else:
        loss_dict["loss_depth_distillation"] = torch.tensor(0.0, device=student_preds.get("depth", torch.tensor(0.0)).device, requires_grad=True)
    
    # Depth confidence distillation loss
    if "depth_conf" in student_preds and "depth_conf" in teacher_preds:
        depth_conf_loss = F.mse_loss(student_preds["depth_conf"], teacher_preds["depth_conf"])
        depth_conf_loss = check_and_fix_inf_nan(depth_conf_loss, "depth_conf_distillation_loss")
        loss_dict["loss_depth_conf_distillation"] = depth_conf_loss * weight_depth_conf
    else:
        loss_dict["loss_depth_conf_distillation"] = torch.tensor(0.0, device=student_preds.get("depth_conf", torch.tensor(0.0)).device, requires_grad=True)
    
    return loss_dict




