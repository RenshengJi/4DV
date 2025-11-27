# --------------------------------------------------------
# training code for CUT3R
# --------------------------------------------------------
# References:
# DUSt3R: https://github.com/naver/dust3r
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math
import cv2
from collections import defaultdict
from pathlib import Path
from typing import Sized
import re

import faulthandler
faulthandler.enable()

# ===== VGGT相关导入 =====
import sys
# 添加vggt路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'vggt'))
from vggt.vggt.models.vggt import VGGT
from vggt.training.loss import camera_loss, depth_loss, point_loss, cross_render_and_loss, flow_loss, gt_flow_loss, gt_flow_loss_ours, self_render_and_loss, velocity_loss, sky_opacity_loss, sky_color_loss, vggt_distillation_loss, scale_loss, aggregator_render_loss
from vggt.utils.auxiliary import RAFTCfg, calc_flow
from vggt.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding

# ===== 在线动态处理器和Stage2 Loss =====
from online_dynamic_processor import OnlineDynamicProcessor
from vggt.training.stage2_loss import Stage2CompleteLoss


sys.path.append(os.path.join(os.path.dirname(__file__), 'dam2'))

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
import re
sys.path.append(os.path.join(os.path.dirname(__file__), "SEA-RAFT/core"))
from raft import RAFT


torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from dust3r.datasets import get_data_loader
from dust3r.utils.misc import tf32_off
import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa

import hydra
from omegaconf import OmegaConf
import logging
import pathlib
from tqdm import tqdm
import random
import builtins
import shutil

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datetime import timedelta
import torch.multiprocessing
from collections import OrderedDict

torch.multiprocessing.set_sharing_strategy("file_system")

printer = get_logger(__name__, log_level="DEBUG")

def strip_module(state_dict):
    """
    Removes the 'module.' prefix from the keys of a state_dict.
    Args:
        state_dict (dict): The original state_dict with possible 'module.' prefixes.
    Returns:
        OrderedDict: A new state_dict with 'module.' prefixes removed.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def setup_for_distributed(accelerator: Accelerator):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (accelerator.num_processes > 8)
        if accelerator.is_main_process or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def save_current_code(outdir):
    now = datetime.datetime.now()  # current date and time
    date_time = now.strftime("%m_%d-%H:%M:%S")
    src_dir = "."
    dst_dir = os.path.join(outdir, "code", "{}".format(date_time))
    shutil.copytree(
        src_dir,
        dst_dir,
        ignore=shutil.ignore_patterns(
            ".vscode*",
            "assets*",
            "example*",
            "checkpoints*",
            "OLD*",
            "logs*",
            "out*",
            "runs*",
            "*.png",
            "*.mp4",
            "*__pycache__*",
            "*.git*",
            "*.idea*",
            "*.zip",
            "*.jpg",
            "*.pt",
            "*.pth",
        ),
        dirs_exist_ok=True,
    )
    return dst_dir


def train(args):

    accelerator = Accelerator(
        gradient_accumulation_steps=args.accum_iter,
        mixed_precision="bf16",
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
            InitProcessGroupKwargs(timeout=timedelta(seconds=6000)),
        ],
    )
    device = accelerator.device

    setup_for_distributed(accelerator)

    # 初始化动态处理器和Stage2损失函数（用于aggregator_all loss）
    dynamic_processor_config = {
        'min_object_size': getattr(args, 'min_object_size', 100),
        'max_objects_per_frame': getattr(args, 'max_objects_per_frame', 10),
        'velocity_threshold': getattr(args, 'velocity_threshold', 0.1),
        'clustering_eps': getattr(args, 'clustering_eps', 0.02),
        'clustering_min_samples': getattr(args, 'clustering_min_samples', 10),
        'tracking_position_threshold': getattr(args, 'tracking_position_threshold', 2.0),
        'tracking_velocity_threshold': getattr(args, 'tracking_velocity_threshold', 0.2),
        'use_optical_flow_aggregation': getattr(args, 'enable_optical_flow_aggregation', True),
        'use_velocity_based_transform': getattr(args, 'use_velocity_based_transform', False),
        'velocity_transform_mode': getattr(args, 'velocity_transform_mode', 'simple')
    }

    # 初始化动态处理器
    dynamic_processor = OnlineDynamicProcessor(
        device=device,
        memory_efficient=getattr(args, 'stage2_memory_efficient', True),
        **dynamic_processor_config
    )

    # 初始化Stage2损失函数（用于aggregator_all）
    stage2_loss_config = {
        'rgb_weight': getattr(args, 'aggregator_all_render_rgb_weight', 1.0),
        'depth_weight': getattr(args, 'aggregator_all_render_depth_weight', 1.0),
        'lpips_weight': getattr(args, 'aggregator_all_render_lpips_weight', 0.1),
        'render_only_dynamic': getattr(args, 'stage2_render_only_dynamic', False),
        'supervise_only_dynamic': getattr(args, 'stage2_supervise_only_dynamic', False),
        'supervise_middle_frame_only': getattr(args, 'stage2_supervise_middle_frame_only', False),
    }
    stage2_criterion = Stage2CompleteLoss(render_loss_config=stage2_loss_config)
    stage2_criterion.to(device)

    printer.info("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        dst_dir = save_current_code(outdir=args.output_dir)
        printer.info(f"Saving current code to {dst_dir}")

    # auto resume
    if not args.resume:
        last_ckpt_fname = os.path.join(args.output_dir, f"checkpoint-last.pth")
        args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    printer.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))

    # fix the seed
    seed = args.seed + accelerator.state.process_index
    printer.info(
        f"Setting seed to {seed} for process {accelerator.state.process_index}"
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = args.benchmark

    # training dataset and loader
    printer.info("Building train dataset %s", args.train_dataset)
    #  dataset and loader
    data_loader_train = build_dataset(
        args.train_dataset,
        args.batch_size,
        args.num_workers,
        accelerator=accelerator,
        test=False,
        fixed_length=args.fixed_length
    )
    # printer.info("Building test dataset %s", args.test_dataset)
    # data_loader_test = {
    #     dataset.split("(")[0]: build_dataset(
    #         dataset,
    #         args.batch_size,
    #         args.num_workers,
    #         accelerator=accelerator,
    #         test=True,
    #         fixed_length=True
    #     )
    #     for dataset in args.test_dataset.split("+")
    # }


    # model
    printer.info("Loading model: %s", args.model)
    model = eval(args.model)
    model.gradient_checkpointing_enable()

    # 应用VGGT冻结策略配置
    vggt_freeze_strategy = getattr(args, 'vggt_freeze_strategy', None)
    if vggt_freeze_strategy is not None:
        printer.info(f"Applying VGGT freeze strategy: {vggt_freeze_strategy}")
        model.set_freeze(vggt_freeze_strategy)
    else:
        printer.info("No VGGT freeze strategy specified, using model defaults")

    printer.info(f"All model parameters: {sum(p.numel() for p in model.parameters())}")
    printer.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    model.to(device)

    if not args.pretrained_velocity:
        printer.info(f"Loading pretrained: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        # 删除 gaussian_head 的 output_conv2.2 参数（可能因为 sh_degree 不匹配）
        # keys_to_remove = [k for k in ckpt.keys() if 'gaussian_head.scratch.output_conv2.2' in k]
        # for key in keys_to_remove:
        #     printer.info(f"Removing {key} from checkpoint (sh_degree mismatch)")
        #     del ckpt[key]
        model.load_state_dict(ckpt, strict=False)
        del ckpt
    else:
        printer.info(f"Resume from: {args.pretrained_velocity}")
        checkpoint = torch.load(args.pretrained_velocity, map_location=device)
        ckpt = strip_module(checkpoint.get('model', checkpoint))  # Handle both wrapped and direct state_dict
        # 删除 gaussian_head 的 output_conv2.2 参数（可能因为 sh_degree 不匹配）
        # keys_to_remove = [k for k in ckpt.keys() if 'gaussian_head.scratch.output_conv2.2' in k]
        # for key in keys_to_remove:
        #     printer.info(f"Removing {key} from checkpoint (sh_degree mismatch)")
        #     del ckpt[key]
        model.load_state_dict(ckpt, strict=False)
        del ckpt, checkpoint
    

    # 加载辅助模型
    auxiliary_model_configs = getattr(args, "auxiliary_models", None)
    auxiliary_models = dict()
    if auxiliary_model_configs is not None:
        for model_name, model_config in auxiliary_model_configs.items():
            # 检查是否是DAM2模型
            if "DepthAnythingV2" in model_config:
                # 延迟导入DAM2，避免Hydra冲突
                try:
                    # 添加dam2目录到Python路径
                    dam2_path = os.path.join(os.path.dirname(__file__), 'dam2')
                    if dam2_path not in sys.path:
                        sys.path.insert(0, dam2_path)
                    from depth_anything_v2.dpt import DepthAnythingV2
                    
                    # 解析DAM2配置
                    encoder_match = re.search(r"encoder\s*=\s*\"([^\"]+)\"", model_config)
                    offload_match = re.search(r"offload\s*=\s*(\"True\"|False)\b", model_config, re.IGNORECASE)
                    
                    if encoder_match:
                        encoder = encoder_match.group(1)
                        # 直接从src目录加载模型文件
                        ckpt_path = os.path.join(os.path.dirname(__file__), f"depth_anything_v2_{encoder}.pth")
                        offload = offload_match.group(1).lower() == "true" if offload_match else False
                        
                        # 模型配置
                        model_configs = {
                            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                        }
                        
                        # 构建DAM2模型
                        dam2_model = DepthAnythingV2(**model_configs[encoder])
                        state_dict = torch.load(ckpt_path, map_location="cpu" if offload else device, weights_only=True)
                        dam2_model.load_state_dict(state_dict)
                        
                        if not offload:
                            dam2_model.to(device)
                        
                        dam2_model.eval()
                        dam2_model.requires_grad_(False)
                        
                        auxiliary_models[model_name] = dam2_model
                        printer.info(f"successfully load DAM2 model: {model_name}")
                    else:
                        printer.warning(f"Missing encoder or path for DAM2 model: {model_name}")
                except Exception as e:
                    printer.warning(f"Failed to load DAM2 model {model_name}: {e}")
                    printer.info("Skipping DAM2 model loading due to error")
            # 检查是否是VGGT teacher模型
            elif "VGGT" in model_config:
                try:
                    # 解析VGGT配置
                    path_match = re.search(r"path\s*=\s*\"([^\"]+)\"", model_config)
                    offload_match = re.search(r"offload\s*=\s*(\"True\"|False)\b", model_config, re.IGNORECASE)
                    
                    if path_match:
                        ckpt_path = path_match.group(1)
                        offload = offload_match.group(1).lower() == "true" if offload_match else False
                        
                        # 构建VGGT teacher模型
                        teacher_model = VGGT(img_size=518, patch_size=14, embed_dim=1024, use_sky_token=False)
                        
                        # 加载checkpoint
                        state_dict = torch.load(ckpt_path, map_location="cpu" if offload else device)
                        if "model" in state_dict:
                            state_dict = state_dict["model"]
                        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                        
                        missing_keys, unexpected_keys = teacher_model.load_state_dict(state_dict, strict=False)
                        printer.info(f"Unexpected key numbers in VGGT teacher {model_name}: {len(unexpected_keys)}")
                        printer.info(f"Missing key numbers in VGGT teacher {model_name}: {len(missing_keys)}")
                        
                        if not offload:
                            teacher_model.to(device)
                        
                        teacher_model.eval()
                        teacher_model.requires_grad_(False)
                        
                        auxiliary_models[model_name] = teacher_model
                        printer.info(f"successfully load VGGT teacher model: {model_name}")
                    else:
                        printer.warning(f"Missing path for VGGT teacher model: {model_name}")
                except Exception as e:
                    printer.warning(f"Failed to load VGGT teacher model {model_name}: {e}")
                    printer.info("Skipping VGGT teacher model loading due to error")
            else:
                # 原有的RAFT模型加载逻辑
                auxiliary_model = eval(model_config)
                offload_match = re.search(r"offload\s*=\s*(\"True\"|False)\b", model_config, re.IGNORECASE)
                if offload_match:
                    offload = offload_match.group(1).lower() == "true"
                else:
                    offload = False
                if not offload:
                    auxiliary_model.to(device)
                # capture the model checkpoint path from the config string
                model_ckpt_match = re.search(r"path\s*=\s*\"([^\"]+)\"", model_config)
                if model_ckpt_match:
                    model_ckpt = model_ckpt_match.group(1)
                    state_dict = torch.load(model_ckpt, map_location="cpu" if offload else device, weights_only=True)
                    missing_keys, unexpected_keys = auxiliary_model.load_state_dict(state_dict, strict=False)
                    printer.info(f"Unexpected key numbers in {model_name}: {len(unexpected_keys)}")
                    printer.info(f"Missing key numbers in {model_name}: {len(missing_keys)}")
                auxiliary_model.eval()
                auxiliary_model.requires_grad_(False)
                auxiliary_models[model_name] = auxiliary_model
                printer.info(f"successfully load auxiliary model: {model_name}")



    # # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.get_parameter_groups(model, args.weight_decay)

    # 检查是否有可训练参数
    if not param_groups:
        raise ValueError(
            "No trainable parameters found! "
            "Please check vggt_freeze_strategy is not freezing all parameters\n"
            f"Stage1 trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
    printer.info(f"Total parameter groups for optimizer: {len(param_groups)}")
    for i, group in enumerate(param_groups):
        group_name = group.get('name', f'group_{i}')
        num_params = len(group['params'])
        group_lr = group.get('lr', args.lr)
        printer.info(f"  Group {i} ({group_name}): {num_params} parameters, lr={group_lr}")

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler(accelerator=accelerator)
    accelerator.even_batches = True
    optimizer, model, data_loader_train = accelerator.prepare(
        optimizer, model, data_loader_train
    )


    def save_model(epoch, fname, best_so_far):
        # 保存主模型
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            fname=fname,
            best_so_far=best_so_far,
        )

    # 加载主模型
    best_so_far = misc.load_model(
        args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler
    )
    if best_so_far is None:
        best_so_far = float("inf")
    log_writer = (
        SummaryWriter(log_dir=args.output_dir) if accelerator.is_main_process else None
    )

    printer.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs + 1):
        if hasattr(data_loader_train, "dataset") and hasattr(data_loader_train.dataset, "set_epoch"):
            data_loader_train.dataset.set_epoch(epoch)
        if (
            hasattr(data_loader_train, "batch_sampler")
            and hasattr(data_loader_train.batch_sampler, "batch_sampler")
            and hasattr(data_loader_train.batch_sampler.batch_sampler, "set_epoch")
        ):
            data_loader_train.batch_sampler.batch_sampler.set_epoch(epoch)
        model.train()
        metric_logger = misc.MetricLogger(delimiter="	")
        header = f"Epoch: [{epoch}]"
        optimizer.zero_grad()
        for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader_train, args.print_freq, accelerator, header)):
            with accelerator.accumulate(model):
                epoch_f = epoch + data_iter_step / len(data_loader_train)
                if data_iter_step % args.accum_iter == 0:
                    misc.adjust_learning_rate(optimizer, epoch_f, args)
                vggt_batch = cut3r_batch_to_vggt(batch)

                # 初始化sky_masks
                sky_masks = None
                if "dam2" in auxiliary_models and vggt_batch.get("images") is not None:
                    sky_masks = dam2_sky_mask_generation(
                        vggt_batch["images"], 
                        auxiliary_models["dam2"],
                        device=device
                    )
                # 将sky masks添加到vggt_batch中，供后续使用
                vggt_batch["sky_masks"] = sky_masks

                # For VGGT model
                frame_sample_ratio = getattr(args, 'aggregator_frame_sample_ratio', 1.0)
                preds = model(
                    vggt_batch["images"],
                    gt_extrinsics=vggt_batch.get("extrinsics"),
                    gt_intrinsics=vggt_batch.get("intrinsics"),
                    frame_sample_ratio=frame_sample_ratio
                )

                # =============== STAGE 1 TRAINING LOGIC ===============
                loss = 0.0
                loss_dict = {}

                # 获取损失权重配置
                loss_weights = {
                    'self_render_weight': getattr(args, 'self_render_weight', 1.0),
                    'flow_loss_weight': getattr(args, 'flow_loss_weight', 1.0),
                    'conf_flow_loss_weight': getattr(args, 'conf_flow_loss_weight', 1.0),  # GT flowmap conf损失
                    'grad_flow_loss_weight': getattr(args, 'grad_flow_loss_weight', 1.0),  # GT flowmap grad损失
                    'reg_flow_loss_weight': getattr(args, 'reg_flow_loss_weight', 1.0),   # GT flowmap reg损失
                    'gt_flow_loss_ours_weight': getattr(args, 'gt_flow_loss_ours_weight', 0.0),  # GT flowmap ours损失
                    'sky_opacity_weight': getattr(args, 'sky_opacity_weight', 1.0),
                    'sky_color_weight': getattr(args, 'sky_color_weight', 1.0),
                    'velocity_reg_weight': getattr(args, 'velocity_reg_weight', 0.001),
                    'vggt_distill_weight': getattr(args, 'vggt_distill_weight', 1.0),
                    'camera_loss_weight': getattr(args, 'camera_loss_weight', 1.0),
                    'conf_depth_loss_weight': getattr(args, 'conf_depth_loss_weight', 1.0),
                    'grad_depth_loss_weight': getattr(args, 'grad_depth_loss_weight', 1.0),
                    'reg_depth_loss_weight': getattr(args, 'reg_depth_loss_weight', 1.0),
                    'scale_loss_weight': getattr(args, 'scale_loss_weight', 1.0),
                    'aggregator_render_rgb_weight': getattr(args, 'aggregator_render_rgb_weight', 1.0),
                    'aggregator_render_depth_weight': getattr(args, 'aggregator_render_depth_weight', 1.0),
                    'aggregator_render_lpips_weight': getattr(args, 'aggregator_render_lpips_weight', 0.0),
                    'aggregator_all_render_rgb_weight': getattr(args, 'aggregator_all_render_rgb_weight', 0.0),
                    'aggregator_all_render_depth_weight': getattr(args, 'aggregator_all_render_depth_weight', 0.0),
                    'aggregator_all_render_lpips_weight': getattr(args, 'aggregator_all_render_lpips_weight', 0.0),
                }

                # VGGT teacher predictions
                # 只要teacher存在就执行替换，确保其他损失计算使用稳定的teacher预测值
                teacher_preds = None
                student_preds_original = None
                if "vggt_teacher" in auxiliary_models and vggt_batch.get("images") is not None:
                    try:
                        with torch.no_grad():
                            teacher_preds = auxiliary_models["vggt_teacher"](
                                vggt_batch["images"],
                                compute_sky_color_loss=False,
                                sky_masks=vggt_batch.get("sky_masks"),
                                gt_images=vggt_batch["images"],
                            )

                        # 保存原始student预测值用于蒸馏损失
                        student_preds_original = {
                            'depth': preds["depth"].clone() if preds.get("depth") is not None else None,
                            'pose_enc': preds["pose_enc"].clone() if preds.get("pose_enc") is not None else None,
                            'depth_conf': preds["depth_conf"].clone() if preds.get("depth_conf") is not None else None
                        }

                        # 强制替换：无论蒸馏权重如何，都使用teacher预测值进行其他损失计算
                        # 这确保了即使unfreeze backbone时，其他损失仍使用稳定的预测值
                        if teacher_preds.get("depth") is not None:
                            preds["depth"] = teacher_preds["depth"]
                        if teacher_preds.get("pose_enc") is not None:
                            preds["pose_enc"] = teacher_preds["pose_enc"]
                        if teacher_preds.get("depth_conf") is not None:
                            preds["depth_conf"] = teacher_preds["depth_conf"]

                    except Exception as e:
                        print(f"Error in VGGT teacher inference: {e}")
                        teacher_preds = None

                # 计算光流（flow_loss需要）
                interval = getattr(args, 'flow_interval', 2)
                forward_flow = backward_flow = forward_consist_mask = backward_consist_mask = None

                if loss_weights['flow_loss_weight'] > 0 and "flow" in auxiliary_models:
                    try:
                        forward_flow, backward_flow, _, _, forward_consist_mask, backward_consist_mask, _, _ = calc_flow(
                            vggt_batch["images"], auxiliary_models["flow"],
                            check_consistency=True,
                            geo_thresh=auxiliary_models["flow"].args.geo_thresh,
                            photo_thresh=auxiliary_models["flow"].args.photo_thresh,
                            interval=interval,
                            return_heatmap=True
                        )
                    except Exception as e:
                        print(f"Error in optical flow computation: {e}")

                # 1. Self Render Loss (训练gaussian head)
                if loss_weights['self_render_weight'] > 0 and vggt_batch.get("images") is not None and vggt_batch.get("depths") is not None:
                    try:
                        self_loss_dict, _ = self_render_and_loss(
                            preds["depth"].detach(),
                            preds["gaussian_params"],
                            preds["pose_enc"].detach(),
                            vggt_batch["extrinsics"],
                            vggt_batch["intrinsics"],
                            vggt_batch["images"],
                            pred_sky_colors=preds.get("pred_sky_colors"),
                            sky_masks=vggt_batch.get("sky_masks")
                        )
                        # 将所有self_render损失加入到总loss中
                        self_render_rgb_loss = self_loss_dict.get("loss_self_render_rgb", 0.0)
                        self_render_lpips_loss = self_loss_dict.get("loss_self_render_lpips", 0.0)
                        self_render_depth_loss = self_loss_dict.get("loss_self_render_depth", 0.0)

                        # 使用分别的权重加入总loss
                        if loss_weights.get('self_render_rgb_weight') is not None:
                            # 使用新的分项权重
                            loss += (loss_weights.get('self_render_rgb_weight', 0.0) * self_render_rgb_loss +
                                   loss_weights.get('self_render_lpips_weight', 0.0) * self_render_lpips_loss +
                                   loss_weights.get('self_render_depth_weight', 0.0) * self_render_depth_loss)
                        else:
                            # 向后兼容：使用旧的总权重
                            total_self_render_loss = (self_render_rgb_loss +
                                                    self_render_lpips_loss +
                                                    self_render_depth_loss)
                            loss += loss_weights.get('self_render_weight', 0.0) * total_self_render_loss

                        # 将所有损失记录到loss_dict中
                        loss_dict.update(self_loss_dict)
                    except Exception as e:
                        print(f"Error in self render loss computation: {e}")

                # 2. Flow Loss (训练velocity head)
                if loss_weights['flow_loss_weight'] > 0 and forward_flow is not None and vggt_batch.get("images") is not None:
                    try:
                        conf = preds["depth_conf"] > 2
                        flow_loss_dict = flow_loss(
                            conf, interval, forward_flow, backward_flow,
                            forward_consist_mask, backward_consist_mask,
                            preds["depth"], preds["velocity"], preds["pose_enc"],
                            vggt_batch["extrinsics"], vggt_batch["intrinsics"], vggt_batch["images"],
                            vggt_batch
                        )
                        flow_loss_value = flow_loss_dict.get("forward_loss", 0.0)
                        loss += loss_weights['flow_loss_weight'] * flow_loss_value
                        loss_dict.update(flow_loss_dict)
                    except Exception as e:
                        print(f"Error in flow loss computation: {e}")

                # 3. GT Flow Loss (直接使用GT flowmap监督velocity head)
                # 返回三个损失：loss_conf_flow, loss_reg_flow, loss_grad_flow
                if (loss_weights['conf_flow_loss_weight'] > 0 or
                    loss_weights['grad_flow_loss_weight'] > 0 or
                    loss_weights['reg_flow_loss_weight'] > 0) and \
                   preds.get("velocity") is not None and \
                   preds.get("velocity_conf") is not None and \
                   vggt_batch.get("flowmap") is not None:
                    try:
                        gt_flow_loss_dict = gt_flow_loss(
                            preds["velocity"],
                            preds["velocity_conf"],
                            vggt_batch,
                            gradient_loss="grad",
                            valid_range=0.98
                        )

                        # 分别计算三个损失
                        conf_flow_loss = gt_flow_loss_dict.get("loss_conf_flow", 1.0)
                        reg_flow_loss = gt_flow_loss_dict.get("loss_reg_flow", 1.0)
                        grad_flow_loss = gt_flow_loss_dict.get("loss_grad_flow", 1.0)

                        loss += loss_weights['conf_flow_loss_weight'] * conf_flow_loss
                        loss += loss_weights['reg_flow_loss_weight'] * reg_flow_loss
                        loss += loss_weights['grad_flow_loss_weight'] * grad_flow_loss

                        loss_dict.update(gt_flow_loss_dict)
                    except Exception as e:
                        print(f"Error in GT flow loss computation: {e}")
                        import traceback
                        traceback.print_exc()

                # 3b. GT Flow Loss Ours
                if loss_weights['gt_flow_loss_ours_weight'] > 0 and \
                   preds.get("velocity") is not None and \
                   vggt_batch.get("flowmap") is not None:
                    try:
                        gt_flow_loss_ours_dict = gt_flow_loss_ours(
                            preds["velocity"],
                            vggt_batch
                        )
                        gt_flow_ours = gt_flow_loss_ours_dict.get("gt_flow_loss", 0.0)
                        loss += loss_weights['gt_flow_loss_ours_weight'] * gt_flow_ours

                        loss_dict.update({
                            'gt_flow_loss_ours': gt_flow_ours,
                            'gt_flow_ours_num_valid': gt_flow_loss_ours_dict.get('gt_flow_num_valid', 0)
                        })
                    except Exception as e:
                        print(f"Error in GT flow loss ours computation: {e}")
                        import traceback
                        traceback.print_exc()

                # 4. Sky Opacity Loss (监督gaussian head的opacity参数)
                if loss_weights['sky_opacity_weight'] > 0 and preds.get("gaussian_params") is not None and sky_masks is not None:
                    try:
                        sky_opacity_loss_dict = sky_opacity_loss(
                            preds["gaussian_params"],
                            sky_masks,
                            weight=1.0
                        )
                        sky_opacity_loss_value = sky_opacity_loss_dict.get("sky_opacity_loss", 0.0)
                        loss += loss_weights['sky_opacity_weight'] * sky_opacity_loss_value
                        loss_dict.update(sky_opacity_loss_dict)
                    except Exception as e:
                        print(f"Error in sky opacity loss computation: {e}")

                # 5. Sky Color Loss (监督sky token以及sky head学习)
                if loss_weights['sky_color_weight'] > 0 and preds.get("pred_sky_colors") is not None and sky_masks is not None:
                    try:
                        sky_color_loss_dict = sky_color_loss(
                            preds["pred_sky_colors"],
                            vggt_batch["images"],
                            sky_masks,
                            weight=1.0
                        )
                        sky_color_loss_value = sky_color_loss_dict.get("sky_color_loss", 0.0)
                        loss += loss_weights['sky_color_weight'] * sky_color_loss_value
                        loss_dict.update(sky_color_loss_dict)
                    except Exception as e:
                        print(f"Error in sky color loss computation: {e}")

                # 6. Velocity Regularization Loss (约束velocity值)
                if loss_weights['velocity_reg_weight'] > 0 and preds.get("velocity") is not None:
                    try:
                        velocity_loss_value = velocity_loss(preds["velocity"])
                        loss += loss_weights['velocity_reg_weight'] * velocity_loss_value
                        loss_dict.update({"loss_velocity": velocity_loss_value})
                    except Exception as e:
                        print(f"Error in velocity regularization loss computation: {e}")

                # 7. VGGT Distillation Loss (蒸馏损失，监督depth head、camera head等)
                if loss_weights['vggt_distill_weight'] > 0 and teacher_preds is not None and student_preds_original is not None:
                    try:
                        # 构建原始student预测字典用于蒸馏损失计算
                        student_preds_for_distill = preds.copy()
                        # 使用原始student预测值进行蒸馏
                        if student_preds_original['depth'] is not None:
                            student_preds_for_distill['depth'] = student_preds_original['depth']
                        if student_preds_original['pose_enc'] is not None:
                            student_preds_for_distill['pose_enc'] = student_preds_original['pose_enc']
                        if student_preds_original['depth_conf'] is not None:
                            student_preds_for_distill['depth_conf'] = student_preds_original['depth_conf']

                        distill_loss_dict = vggt_distillation_loss(
                            student_preds=student_preds_for_distill,
                            teacher_preds=teacher_preds
                        )
                        distill_loss_value = distill_loss_dict.get("loss_distillation", 0.0)
                        loss += loss_weights['vggt_distill_weight'] * distill_loss_value
                        loss_dict.update(distill_loss_dict)
                    except Exception as e:
                        print(f"Error in VGGT distillation loss computation: {e}")

                # 8. Camera Loss (监督camera head训练)
                if loss_weights['camera_loss_weight'] > 0 and preds.get("pose_enc") is not None and vggt_batch.get("extrinsics") is not None and vggt_batch.get("intrinsics") is not None:
                    try:
                        camera_loss_dict = camera_loss(
                            [preds["pose_enc"]],
                            vggt_batch,
                        )
                        camera_loss_value = camera_loss_dict.get("loss_camera", 0.0)
                        loss += loss_weights['camera_loss_weight'] * camera_loss_value
                        loss_dict.update(camera_loss_dict)
                    except Exception as e:
                        print(f"Error in camera loss computation: {e}")

                # 9. Depth Loss (监督depth head训练)
                if loss_weights['conf_depth_loss_weight'] > 0 and preds.get("depth") is not None and preds.get("depth_conf") is not None and vggt_batch.get("depths") is not None:
                    try:
                        depth_loss_dict = depth_loss(
                            preds["depth"],
                            preds["depth_conf"],
                            vggt_batch,
                            gradient_loss="grad",
                            valid_range=0.98
                        )
                        depth_loss_value = depth_loss_dict.get("loss_conf_depth", 0.0)
                        gradient_loss_value = depth_loss_dict.get("loss_grad_depth", 0.0)
                        reg_loss_value = depth_loss_dict.get("loss_reg_depth", 0.0)
                        loss += loss_weights['conf_depth_loss_weight'] * depth_loss_value + loss_weights['grad_depth_loss_weight'] * gradient_loss_value + loss_weights['reg_depth_loss_weight'] * reg_loss_value
                        loss_dict.update(depth_loss_dict)
                    except Exception as e:
                        print(f"Error in depth loss computation: {e}")

                # 10. Scale Loss (监督scale head训练，使用depth_scale_factor作为GT)
                if loss_weights['scale_loss_weight'] > 0 and preds.get("scale") is not None and vggt_batch.get("depth_scale_factor") is not None:
                    try:
                        scale_loss_dict = scale_loss(
                            preds["scale"],
                            vggt_batch["depth_scale_factor"]
                        )
                        scale_loss_value = scale_loss_dict.get("scale_loss", 0.0)
                        loss += loss_weights['scale_loss_weight'] * scale_loss_value
                        loss_dict.update(scale_loss_dict)
                    except Exception as e:
                        print(f"Error in scale loss computation: {e}")

                # 11. Aggregator Render Loss (监督gaussian_head，辅助监督depth和velocity)
                if (loss_weights['aggregator_render_rgb_weight'] > 0 or
                    loss_weights['aggregator_render_depth_weight'] > 0 or
                    loss_weights['aggregator_render_lpips_weight'] > 0) and \
                   preds.get("gaussian_params") is not None and \
                   preds.get("depth") is not None and \
                   preds.get("velocity") is not None and \
                   vggt_batch.get("depth_scale_factor") is not None:
                    try:
                        # Get sky masks if available
                        sky_masks = vggt_batch.get("sky_masks")  # [B, S, H, W]

                        # Get voxel size from config
                        voxel_size = getattr(args, 'aggregator_voxel_size', 0.05)

                        # Get GT scale factor
                        gt_scale_factor = vggt_batch["depth_scale_factor"]

                        # Get pre-computed sky colors and sampled frame indices from model forward
                        sky_colors = preds.get("sky_colors")  # [B, num_frames, 3, H, W] or None
                        sampled_frame_indices = preds.get("sampled_frame_indices")  # list or None

                        # Compute aggregator render loss
                        aggregator_loss_dict = aggregator_render_loss(
                            gaussian_params=preds["gaussian_params"],
                            depth=preds["depth"],
                            velocity=preds["velocity"],  # velocity不参与梯度计算
                            sky_masks=sky_masks,
                            gt_extrinsic=vggt_batch["extrinsics"],
                            gt_intrinsic=vggt_batch["intrinsics"],
                            gt_rgb=vggt_batch["images"],
                            gt_depth=vggt_batch["depths"],
                            gt_depth_mask=vggt_batch.get("point_masks"),
                            voxel_size=voxel_size,
                            gt_scale=gt_scale_factor,
                            sky_colors=sky_colors,
                            sampled_frame_indices=sampled_frame_indices,
                            use_lpips=(loss_weights['aggregator_render_lpips_weight'] > 0),
                            dynamic_threshold=getattr(args, 'aggregator_dynamic_threshold', 0.1)
                        )

                        # Add weighted losses
                        if aggregator_loss_dict.get("aggregator_render_rgb_loss") is not None:
                            loss += loss_weights['aggregator_render_rgb_weight'] * aggregator_loss_dict["aggregator_render_rgb_loss"]
                        if aggregator_loss_dict.get("aggregator_render_depth_loss") is not None:
                            loss += loss_weights['aggregator_render_depth_weight'] * aggregator_loss_dict["aggregator_render_depth_loss"]
                        if aggregator_loss_dict.get("aggregator_render_lpips_loss") is not None:
                            loss += loss_weights['aggregator_render_lpips_weight'] * aggregator_loss_dict["aggregator_render_lpips_loss"]

                        loss_dict.update(aggregator_loss_dict)
                    except Exception as e:
                        print(f"Error in aggregator render loss computation: {e}")
                        import traceback
                        traceback.print_exc()

                # 12. Aggregator_all Render Loss (代替Stage2 refine网络的渲染loss)
                # 这个loss直接在Stage1中使用动态物体处理+渲染监督,跳过Stage2的refine网络
                aggregator_all_rgb_weight = loss_weights.get('aggregator_all_render_rgb_weight', 0.0)
                aggregator_all_depth_weight = loss_weights.get('aggregator_all_render_depth_weight', 0.0)
                aggregator_all_lpips_weight = loss_weights.get('aggregator_all_render_lpips_weight', 0.0)
                aggregator_all_start_iter = getattr(args, 'aggregator_all_start_iter', 50000)

                # 计算当前iteration
                current_iteration = epoch * len(data_loader_train) + data_iter_step

                if ((aggregator_all_rgb_weight > 0 or
                    aggregator_all_depth_weight > 0 or
                    aggregator_all_lpips_weight > 0) and
                    current_iteration >= aggregator_all_start_iter):
                    try:
                        # Step 1: 处理动态物体 (使用OnlineDynamicProcessor)
                        # 使用GT camera而不是预测的pose_enc
                        preds_for_dynamic = preds.copy()
                        if 'pose_enc' in preds_for_dynamic:
                            # 将GT的extrinsics和intrinsics转换为pose_enc格式
                            gt_extrinsics = vggt_batch['extrinsics']  # [B, S, 4, 4]
                            gt_intrinsics = vggt_batch['intrinsics']  # [B, S, 3, 3]
                            image_size_hw = vggt_batch['images'].shape[-2:]

                            gt_pose_enc = extri_intri_to_pose_encoding(
                                gt_extrinsics, gt_intrinsics, image_size_hw, pose_encoding_type="absT_quaR_FoV"
                            )
                            preds_for_dynamic['pose_enc'] = gt_pose_enc
                            # preds_for_dynamic['velocity'] = preds_for_dynamic['velocity'].detach()

                        dynamic_objects_data = dynamic_processor.process_dynamic_objects(
                            preds_for_dynamic, vggt_batch, auxiliary_models
                        )

                        # 检查是否有有效的动态物体（注意：即使没有动态物体，也要计算静态物体的loss）
                        num_objects = len(dynamic_objects_data['dynamic_objects']) if dynamic_objects_data else 0
                        has_valid_dynamic = num_objects > 0

                        # 无论是否有动态物体，都需要计算loss（因为有静态物体）
                        if dynamic_objects_data is not None:
                            # Step 2: 跳过Stage2 refine网络,直接构建场景
                            # 不调用 stage2_model()，直接使用原始的canonical gaussians
                            dynamic_objects = dynamic_objects_data['dynamic_objects'] if has_valid_dynamic else []
                            static_gaussians = dynamic_objects_data.get('static_gaussians')

                            # 构建"refined"场景 (实际上是未refine的原始数据)
                            aggregator_all_scene = {
                                'static_gaussians': static_gaussians,
                                'dynamic_objects': dynamic_objects  # 可能为空列表，但静态物体仍然存在
                            }

                            # Step 3: 计算渲染loss (使用Stage2的loss函数)
                            B, S, C, H, W = vggt_batch['images'].shape
                            gt_images = vggt_batch['images']
                            gt_depths = vggt_batch.get('depths', torch.ones(B, S, H, W, device=device) * 5.0)

                            # 使用GT相机参数（从vggt_batch中获取）
                            intrinsics = vggt_batch['intrinsics']  # [B, S, 3, 3]
                            extrinsics = vggt_batch['extrinsics']  # [B, S, 4, 4]

                            sky_masks = vggt_batch.get('sky_masks', None)

                            # 获取sky_colors和sampled_frame_indices（如果有）
                            sky_colors = preds.get('sky_colors', None)  # [B, num_frames, 3, H, W]
                            sampled_frame_indices = preds.get('sampled_frame_indices', None)  # [num_frames]

                            # 使用Stage2的criterion计算loss
                            aggregator_all_loss_dict = stage2_criterion(
                                refinement_results={'refined_dynamic_objects': dynamic_objects},  # 无refine结果
                                refined_scene=aggregator_all_scene,
                                gt_images=gt_images,
                                gt_depths=gt_depths,
                                intrinsics=intrinsics,
                                extrinsics=extrinsics,
                                sky_masks=sky_masks,
                                original_dynamic_objects=dynamic_objects,
                                sky_colors=sky_colors,
                                sampled_frame_indices=sampled_frame_indices
                            )

                            # 提取各项loss并加权
                            aggregator_all_rgb_loss = aggregator_all_loss_dict.get('stage2_rgb_loss', 0.0)
                            aggregator_all_depth_loss = aggregator_all_loss_dict.get('stage2_depth_loss', 0.0)
                            aggregator_all_lpips_loss = aggregator_all_loss_dict.get('stage2_lpips_loss', 0.0)

                            # 提取可视化指标 (不参与梯度回传)
                            aggregator_all_rgb_loss_sky = aggregator_all_loss_dict.get('stage2_rgb_loss_sky', None)
                            aggregator_all_rgb_loss_nonsky = aggregator_all_loss_dict.get('stage2_rgb_loss_nonsky', None)

                            # 加入总loss
                            if aggregator_all_rgb_weight > 0:
                                loss += aggregator_all_rgb_weight * aggregator_all_rgb_loss
                            if aggregator_all_depth_weight > 0:
                                loss += aggregator_all_depth_weight * aggregator_all_depth_loss
                            if aggregator_all_lpips_weight > 0:
                                loss += aggregator_all_lpips_weight * aggregator_all_lpips_loss

                            # 记录到loss_dict (重命名为aggregator_all前缀)
                            loss_dict_update = {
                                'aggregator_all_rgb_loss': float(aggregator_all_rgb_loss) if isinstance(aggregator_all_rgb_loss, torch.Tensor) else aggregator_all_rgb_loss,
                                'aggregator_all_depth_loss': float(aggregator_all_depth_loss) if isinstance(aggregator_all_depth_loss, torch.Tensor) else aggregator_all_depth_loss,
                                'aggregator_all_lpips_loss': float(aggregator_all_lpips_loss) if isinstance(aggregator_all_lpips_loss, torch.Tensor) else aggregator_all_lpips_loss,
                                'aggregator_all_num_objects': num_objects
                            }

                            # 添加可视化指标 (如果存在)
                            if aggregator_all_rgb_loss_sky is not None:
                                loss_dict_update['aggregator_all_rgb_loss_sky'] = float(aggregator_all_rgb_loss_sky) if isinstance(aggregator_all_rgb_loss_sky, torch.Tensor) else aggregator_all_rgb_loss_sky
                            if aggregator_all_rgb_loss_nonsky is not None:
                                loss_dict_update['aggregator_all_rgb_loss_nonsky'] = float(aggregator_all_rgb_loss_nonsky) if isinstance(aggregator_all_rgb_loss_nonsky, torch.Tensor) else aggregator_all_rgb_loss_nonsky

                            loss_dict.update(loss_dict_update)
                        else:
                            # dynamic_objects_data为None，无法计算loss
                            loss_dict.update({
                                'aggregator_all_rgb_loss': 0.0,
                                'aggregator_all_depth_loss': 0.0,
                                'aggregator_all_lpips_loss': 0.0,
                                'aggregator_all_num_objects': 0
                            })

                    except Exception as e:
                        print(f"Error in aggregator_all render loss computation: {e}")
                        import traceback
                        traceback.print_exc()

                lr = optimizer.param_groups[0]["lr"]
                metric_logger.update(epoch=epoch)
                metric_logger.update(lr=lr)
                # 先记录第一阶段的损失（第二阶段损失会在后面加入）
                loss_value_stage1 = float(loss)
                metric_logger.update(loss=loss_value_stage1, **loss_dict)
                if log_writer is not None:
                    step = epoch * len(data_loader_train) + data_iter_step
                    log_writer.add_scalar("train_loss", loss_value_stage1, step)
                    log_writer.add_scalar("train_lr", lr, step)
                    for name, val in loss_dict.items():
                        if isinstance(val, torch.Tensor):
                            if val.ndim > 0:
                                continue
                        if isinstance(val, dict):
                            continue
                        log_writer.add_scalar("train_" + name, val, step)

                # =============== 梯度更新逻辑 ===============
                loss_value = float(loss)

                # 只有Stage1训练
                loss_scaler(
                    loss,
                    optimizer,
                    parameters=model.parameters(),
                    update_grad=True,
                    clip_grad=1.0,
                )
                optimizer.zero_grad()

                # 按照save_freq保存模型
                if (
                    data_iter_step % int(args.save_freq * len(data_loader_train)) == 0
                    and data_iter_step != 0
                    and data_iter_step % len(data_loader_train) != 0
                ):
                    print("saving at step", data_iter_step)
                    save_model(epoch - 1, f"epoch_{epoch}_{data_iter_step}", float("inf"))
                metric_logger.synchronize_between_processes(accelerator)
                printer.info("Averaged stats: %s", metric_logger)
                


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    printer.info("Training time {}".format(total_time_str))

    save_final_model(accelerator, args, args.epochs, model, best_so_far=best_so_far)


def save_final_model(accelerator, args, epoch, model_without_ddp, best_so_far=None):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "checkpoint-final.pth"
    to_save = {
        "args": args,
        "model": (
            model_without_ddp
            if isinstance(model_without_ddp, dict)
            else model_without_ddp.cpu().state_dict()
        ),
        "epoch": epoch,
    }
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    printer.info(f">> Saving model to {checkpoint_path} ...")
    misc.save_on_master(accelerator, to_save, checkpoint_path)


def build_dataset(dataset, batch_size, num_workers, accelerator, test=False, fixed_length=False):
    split = ["Train", "Test"][test]
    printer.info(f"Building {split} Data loader for dataset: {dataset}")
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=not (test),
        drop_last=not (test),
        accelerator=accelerator,
        fixed_length=fixed_length
    )
    return loader


def dam2_sky_mask_generation(images, dam2_model, device):
    """
    Generate sky masks using DAM2 depth predictions.
    
    Args:
        images: [B, S, 3, H, W] - input images in range [0, 1]
        dam2_model: DepthAnythingV2 model instance
        device: torch device
    
    Returns:
        torch.Tensor: [B, S, H, W] - sky masks where 1 indicates sky (depth=0) regions
    """
    B, S, C, H, W = images.shape
    sky_masks = torch.zeros(B, S, H, W, device=device)
    
    with torch.no_grad():
        for b in range(B):
            for s in range(S):
                # 获取单张图片
                img = images[b, s]  # [3, H, W]
                
                # 转换为numpy格式用于DAM2推理
                img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img_np = np.clip(img_np, 0, 255)
                
                # 使用DAM2预测深度
                depth = dam2_model.infer_image(img_np, input_size=518)
                
                # 将深度转换为tensor
                depth_tensor = torch.from_numpy(depth).to(device)
                
                # 调整深度图尺寸以匹配原图
                if depth_tensor.shape != (H, W):
                    depth_tensor = F.interpolate(
                        depth_tensor.unsqueeze(0).unsqueeze(0), 
                        size=(H, W), 
                        mode='bilinear', 
                        align_corners=True
                    ).squeeze(0).squeeze(0)
                
                # 生成sky mask：深度值接近0的区域被认为是天空
                # 使用阈值来识别深度为0的区域
                depth_threshold = 0.1  # 可以根据需要调整阈值
                sky_mask = (depth_tensor < depth_threshold).float()
                
                # 可选：使用形态学操作来平滑mask
                # 这里可以添加开运算和闭运算来改善mask质量
                
                sky_masks[b, s] = sky_mask
    
    return sky_masks


def cut3r_batch_to_vggt(views):
    # views: List[Dict], 长度为num_views
    # 目标: [1, S, 3, H, W] (B=1, S=num_views)
    imgs = [v['img'] for v in views]  # List of [B,3,H,W]
    imgs = torch.stack(imgs, dim=0)  # [S,B,3,H,W]

    vggt_batch = {
        'images': imgs * 0.5 + 0.5,  # [S,B,3,H,W], 归一化到[0,1]
        'depths': torch.stack([v['depthmap'] for v in views], dim=0) if 'depthmap' in views[0] else None,
        'intrinsics': torch.stack([v['camera_intrinsics'] for v in views], dim=0) if 'camera_intrinsics' in views[0] else None,
        'extrinsics': torch.stack([v['camera_pose'] for v in views], dim=0) if 'camera_pose' in views[0] else None,
        'point_masks': torch.stack([v['valid_mask'] for v in views], dim=0) if 'valid_mask' in views[0] else None,
        'world_points': torch.stack([v['pts3d'] for v in views], dim=0) if 'pts3d' in views[0] else None,
        'flowmap': torch.stack([torch.from_numpy(v['flowmap']).float() if isinstance(v['flowmap'], np.ndarray) else v['flowmap'].float() for v in views], dim=0) if 'flowmap' in views[0] and views[0]['flowmap'] is not None else None,
    }

    with tf32_off(), torch.amp.autocast("cuda", enabled=False):
        # 转换world points的坐标系到第一帧相机坐标系
        if vggt_batch['world_points'] is not None:
            B, S, H, W, _ = vggt_batch['world_points'].shape
            world_points = vggt_batch['world_points'].reshape(B, S, H*W, 3)
            world_points = torch.matmul(torch.linalg.inv(vggt_batch['extrinsics'][0])[:, :3, :3], world_points.transpose(-1, -2)).transpose(-1, -2) + \
                                       torch.linalg.inv(vggt_batch['extrinsics'][0])[:, :3, 3:4].transpose(-1, -2)
            vggt_batch['world_points'] = world_points.reshape(B, S, H, W, 3)

            # 处理flowmap - 应用scaling (ground mask已经在waymo.py的_get_views中应用)
            if vggt_batch['flowmap'] is not None:
                vggt_batch['flowmap'][..., :3] *=  0.1

            # 转换extrinsics的坐标系到第一帧相机坐标系
            vggt_batch['extrinsics'] = torch.matmul(
                    torch.linalg.inv(vggt_batch['extrinsics']),
                    vggt_batch['extrinsics'][0]
                )

            # 将extrinsics(中的T)以及world_points、depth进行非metric化
            world_points_flatten = vggt_batch['world_points'].reshape(-1, 3)
            world_points_mask_flatten = vggt_batch['point_masks'].reshape(-1) if vggt_batch['point_masks'] is not None else torch.ones_like(world_points_flatten[:, 0], dtype=torch.bool)
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
            # 5维: [S, B, H, W, C] -> [B, S, H, W, C]
            vggt_batch['flowmap'] = vggt_batch['flowmap'].permute(1, 0, 2, 3, 4).contiguous()
        elif vggt_batch['flowmap'].dim() == 4:
            # 4维: [S, H, W, C] -> [1, S, H, W, C] (添加batch维度)
            vggt_batch['flowmap'] = vggt_batch['flowmap'].unsqueeze(0).contiguous()
        # 其他维度保持不变

    return vggt_batch


@hydra.main(
    version_base=None,
    config_path=str(os.path.dirname(os.path.abspath(__file__))) + "/../config",
    config_name="train.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    if cfg.get("debug", False):
        cfg.num_workers = 0
        import debugpy
        debugpy.listen(5697)
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
    logdir = pathlib.Path(cfg.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    run()
