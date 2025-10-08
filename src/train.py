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
from vggt.training.loss import camera_loss, depth_loss, point_loss, cross_render_and_loss, flow_loss, gt_flow_loss, self_render_and_loss, velocity_loss, sky_opacity_loss, sky_color_loss, vggt_distillation_loss, scale_loss, aggregator_render_loss
from vggt.utils.auxiliary import RAFTCfg, calc_flow
from vggt.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding

# ===== 在线第二阶段训练器 =====
from online_stage2_trainer import OnlineStage2Trainer


sys.path.append(os.path.join(os.path.dirname(__file__), 'sam2'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'dam2'))

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
import re
sys.path.append(os.path.join(os.path.dirname(__file__), "SEA-RAFT/core"))
from raft import RAFT


torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from dust3r.model import (
    PreTrainedModel,
    ARCroco3DStereo,
    ARCroco3DStereoConfig,
    inf,
    strip_module,
)  # noqa: F401, needed when loading the model
from dust3r.datasets import get_data_loader
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch, loss_of_one_batch_tbptt  # noqa
from dust3r.viz import colorize
from dust3r.utils.render import get_render_results
from dust3r.gaussians import GaussianAdapterCfg, DecoderSplattingCUDACfg
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

torch.multiprocessing.set_sharing_strategy("file_system")

printer = get_logger(__name__, log_level="DEBUG")


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
    
    # 初始化在线第二阶段训练器
    stage2_config = {
        # 模型配置
        'input_gaussian_dim': getattr(args, 'input_gaussian_dim', 14),
        'output_gaussian_dim': getattr(args, 'output_gaussian_dim', 11),
        'gaussian_feature_dim': getattr(args, 'gaussian_feature_dim', 128),
        'gaussian_num_layers': getattr(args, 'gaussian_num_layers', 2),
        'gaussian_num_heads': getattr(args, 'gaussian_num_heads', 4),
        'gaussian_mlp_ratio': getattr(args, 'gaussian_mlp_ratio', 2.0),
        'pose_feature_dim': getattr(args, 'pose_feature_dim', 128),
        'pose_num_heads': getattr(args, 'pose_num_heads', 4),
        'pose_num_layers': getattr(args, 'pose_num_layers', 2),
        'max_points_per_object': getattr(args, 'max_points_per_object', 2048),
        'training_mode': getattr(args, 'stage2_training_mode', 'joint'),
        
        # 损失配置
        'rgb_loss_weight': getattr(args, 'stage2_rgb_loss_weight', 0.5),
        'depth_loss_weight': getattr(args, 'stage2_depth_loss_weight', 0.05),
        
        # 动态处理器配置
        'dynamic_processor': {
            'min_object_size': getattr(args, 'min_object_size', 100),
            'max_objects_per_frame': getattr(args, 'max_objects_per_frame', 10),
            'velocity_threshold': getattr(args, 'velocity_threshold', 0.1),
            'clustering_eps': getattr(args, 'clustering_eps', 0.02),
            'clustering_min_samples': getattr(args, 'clustering_min_samples', 10),
            'tracking_position_threshold': getattr(args, 'tracking_position_threshold', 2.0),
            'tracking_velocity_threshold': getattr(args, 'tracking_velocity_threshold', 0.2),
            'use_optical_flow_aggregation': getattr(args, 'enable_optical_flow_aggregation', True)
        }
    }
    
    online_stage2_trainer = OnlineStage2Trainer(
        stage2_config=stage2_config,
        device=device,
        enable_stage2=getattr(args, 'enable_stage2', True),
        stage2_start_epoch=getattr(args, 'stage2_start_epoch', 5),
        stage2_frequency=getattr(args, 'stage2_frequency', 10),
        memory_efficient=getattr(args, 'stage2_memory_efficient', True)
    ) if accelerator.is_main_process else None

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
    printer.info("Building test dataset %s", args.test_dataset)
    data_loader_test = {
        dataset.split("(")[0]: build_dataset(
            dataset,
            args.batch_size,
            args.num_workers,
            accelerator=accelerator,
            test=True,
            fixed_length=True
        )
        for dataset in args.test_dataset.split("+")
    }


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
        model.load_state_dict(ckpt, strict=False)
        del ckpt
    else:
        printer.info(f"Resume from: {args.pretrained_velocity}")
        checkpoint = torch.load(args.pretrained_velocity, map_location=device)
        ckpt = strip_module(checkpoint.get('model', checkpoint))  # Handle both wrapped and direct state_dict
        model.load_state_dict(ckpt, strict=False)
        del ckpt, checkpoint
    

    # 检查是否使用预处理的SAM掩码
    auxiliary_model_configs = getattr(args, "auxiliary_models", None)
    auxiliary_models = dict()
    if auxiliary_model_configs is not None:
        for model_name, model_config in auxiliary_model_configs.items():
            # 检查是否是SAM2模型
            if "SAM2Base" in model_config:
                # 延迟导入SAM2，避免Hydra冲突
                try:
                    from sam2.build_sam import build_sam2
                    from sam2.modeling.sam2_base import SAM2Base
                    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                    
                    # 解析SAM2配置
                    config_match = re.search(r"config_file\s*=\s*\"([^\"]+)\"", model_config)
                    path_match = re.search(r"path\s*=\s*\"([^\"]+)\"", model_config)
                    offload_match = re.search(r"offload\s*=\s*(\"True\"|False)\b", model_config, re.IGNORECASE)
                    
                    if config_match and path_match:
                        config_file = config_match.group(1)
                        ckpt_path = path_match.group(1)
                        offload = offload_match.group(1).lower() == "true" if offload_match else False
                        
                        # 构建SAM2模型
                        sam2_model = build_sam2(
                            config_file=config_file,
                            ckpt_path=ckpt_path,
                            device="cpu" if offload else device,
                            mode="eval"
                        )
                        
                        if not offload:
                            sam2_model.to(device)
                        
                        sam2_model.eval()
                        sam2_model.requires_grad_(False)
                        
                        # 创建SAM2AutomaticMaskGenerator
                        auxiliary_model = SAM2AutomaticMaskGenerator(
                            model=sam2_model,
                            points_per_side=16,
                            pred_iou_thresh=0.3,
                            stability_score_thresh=0.95,
                            mask_threshold=0.0,
                            box_nms_thresh=0.7,
                            min_mask_region_area=100,
                            output_mode="binary_mask"
                        )
                        
                        auxiliary_models[model_name] = auxiliary_model
                        printer.info(f"successfully load SAM2 automatic mask generator: {model_name}")
                    else:
                        printer.warning(f"Missing config_file or path for SAM2 model: {model_name}")
                except Exception as e:
                    printer.warning(f"Failed to load SAM2 model {model_name}: {e}")
                    printer.info("Skipping SAM2 model loading due to error")
            # 检查是否是DAM2模型
            elif "DepthAnythingV2" in model_config:
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
    
    # 添加第二阶段参数到优化器
    if online_stage2_trainer is not None and online_stage2_trainer.enable_stage2:
        stage2_params = online_stage2_trainer.get_stage2_parameters()
        if stage2_params:
            # 为第二阶段参数创建单独的参数组（使用较小的学习率）
            stage2_lr = getattr(args, 'stage2_learning_rate', args.lr * 0.1)  # 第二阶段使用更小的学习率
            # 直接创建参数组，stage2_params是参数列表
            if stage2_params:  # 确保参数列表不为空
                stage2_param_groups = [{
                    'params': stage2_params,
                    'lr': stage2_lr,
                    'weight_decay': getattr(args, 'stage2_weight_decay', args.weight_decay * 0.5),
                    'name': 'stage2_params'
                }]
                
                param_groups.extend(stage2_param_groups)
                print(f"Added {len(stage2_params)} Stage2 parameters to optimizer with lr={stage2_lr}")
    
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler(accelerator=accelerator)
    
    # 设置第二阶段优化器
    if online_stage2_trainer is not None:
        online_stage2_trainer.set_optimizer(optimizer)

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
        
        # 保存第二阶段模型状态
        if online_stage2_trainer is not None and online_stage2_trainer.enable_stage2:
            if accelerator.is_main_process:
                output_dir = Path(args.output_dir)
                if fname is None:
                    fname = str(epoch)
                stage2_checkpoint_path = output_dir / ("stage2-checkpoint-%s.pth" % fname)
                stage2_state = online_stage2_trainer.save_state_dict()
                misc.save_on_master(accelerator, stage2_state, stage2_checkpoint_path)
                print(f">> Saving Stage2 model to {stage2_checkpoint_path} ...")

    # 加载主模型
    best_so_far = misc.load_model(
        args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler
    )
    
    # 加载第二阶段模型状态
    if online_stage2_trainer is not None and online_stage2_trainer.enable_stage2 and args.resume is not None:
        # 构建stage2检查点路径
        if args.resume.endswith('.pth'):
            stage2_resume_path = args.resume.replace('checkpoint-', 'stage2-checkpoint-')
            if os.path.exists(stage2_resume_path):
                print(f">> Loading Stage2 model from {stage2_resume_path} ...")
                stage2_checkpoint = torch.load(stage2_resume_path, map_location="cpu")
                online_stage2_trainer.load_state_dict(stage2_checkpoint)
                print("Stage2 model loaded successfully")
            else:
                print(f"Stage2 checkpoint not found at {stage2_resume_path}, starting Stage2 from scratch")
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

                # For VGGT model, extract images from vggt_batch
                if vggt_batch.get("images") is not None:
                    # Get frame sample ratio from config for aggregator_render_loss
                    frame_sample_ratio = getattr(args, 'aggregator_frame_sample_ratio', 0.25)
                    preds = model(
                        vggt_batch["images"],
                        gt_extrinsics=vggt_batch.get("extrinsics"),
                        gt_intrinsics=vggt_batch.get("intrinsics"),
                        frame_sample_ratio=frame_sample_ratio
                    )
                else:
                    # Fallback for other models
                    preds = model(batch)

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
                    'sam2_velocity_weight': getattr(args, 'sam2_velocity_weight', 1.0),
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
                            sky_masks=vggt_batch.get("sky_masks"),
                            iteration=epoch * len(data_loader_train) + data_iter_step,
                            lpips_start_iter=getattr(args, 'lpips_start_iter', 5000)
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

                # 4. SAM2 Velocity Consistency Loss (辅助velocity head监督)
                if loss_weights['sam2_velocity_weight'] > 0 and vggt_batch.get("images") is not None:
                    use_preprocessed_sam = getattr(args, "use_preprocessed_sam", False)
                    try:
                        if use_preprocessed_sam and vggt_batch.get("sam_masks") is not None:
                            # 离线模式：使用预处理的SAM掩码
                            sam2_loss_dict = sam2_velocity_consistency_loss_with_preprocessed_masks(
                                vggt_batch["images"],
                                preds["velocity"],
                                vggt_batch["sam_masks"],
                                device=device
                            )
                        elif "sam2" in auxiliary_models:
                            # 在线模式：边训练边推理SAM2
                            sam2_loss_dict = sam2_velocity_consistency_loss(
                                vggt_batch["images"],
                                preds["velocity"],
                                auxiliary_models["sam2"],
                                device=device
                            )
                        else:
                            sam2_loss_dict = {}

                        sam2_loss_value = sam2_loss_dict.get("sam2_velocity_consistency_loss", 0.0)
                        if sam2_loss_value > 0:
                            loss += loss_weights['sam2_velocity_weight'] * sam2_loss_value
                            loss_dict.update(sam2_loss_dict)
                    except Exception as e:
                        print(f"Error in SAM2 velocity loss computation: {e}")

                # 5. Sky Opacity Loss (监督gaussian head的opacity参数)
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

                # 6. Sky Color Loss (监督sky token以及sky head学习)
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
                            velocity=preds["velocity"],
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
                            iteration=epoch * len(data_loader_train) + data_iter_step,
                            lpips_start_iter=getattr(args, 'lpips_start_iter', 5000),
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

                # =============== STAGE 2 TRAINING LOGIC ===============
                # 第二阶段：仅包含渲染损失，用于动态对象精细化
                stage2_loss = None
                stage2_loss_dict = {}

                if (online_stage2_trainer is not None and
                    getattr(args, 'enable_stage2', False) and
                    epoch >= getattr(args, 'stage2_start_epoch', 10)):
                    try:
                        stage2_loss, stage2_loss_dict = online_stage2_trainer.process_stage1_outputs(
                            preds=preds,
                            vggt_batch=vggt_batch,
                            auxiliary_models=auxiliary_models,
                            epoch=epoch,
                            iteration=data_iter_step
                        )

                        if stage2_loss is not None:
                            # 第二阶段只包含渲染损失，权重配置
                            stage2_weight = getattr(args, 'stage2_loss_weight', 0.1)

                            # 注意：第二阶段的损失不加入第一阶段的总损失中
                            # 因为第二阶段有独立的参数和优化器

                            # 记录第二阶段损失用于监控（所有stage2损失键名已经带有stage2_前缀）
                            stage2_loss_dict_for_log = {k: v for k, v in stage2_loss_dict.items() if k.startswith('stage2_')}

                            # 记录到metric_logger和tensorboard
                            metric_logger.update(**stage2_loss_dict_for_log)

                            if log_writer is not None:
                                step = epoch * len(data_loader_train) + data_iter_step
                                log_writer.add_scalar("stage2_total_loss", float(stage2_loss), step)
                                for name, val in stage2_loss_dict_for_log.items():
                                    if isinstance(val, torch.Tensor) and val.ndim == 0:
                                        log_writer.add_scalar("train_" + name, val, step)


                    except Exception as e:
                        printer.warning(f"Stage2 training failed at epoch {epoch}, iter {data_iter_step}: {e}")

                # =============== 梯度更新逻辑 ===============
                loss_value = float(loss)

                # 获取当前训练阶段配置
                training_stage = getattr(args, 'training_stage', 'stage1')  # 默认第一阶段

                if training_stage == 'stage1':
                    # ============ 第一阶段训练 ============
                    # 只更新第一阶段的参数（VGGT主模型）
                    loss_scaler(
                        loss,
                        optimizer,
                        parameters=model.parameters(),  # 第一阶段使用完整模型参数
                        update_grad=True,
                        clip_grad=1.0,
                    )
                    optimizer.zero_grad()

                elif training_stage == 'stage2':
                    # ============ 第二阶段训练 ============
                    # 只更新第二阶段的参数（refine模型），不更新第一阶段参数
                    if (stage2_loss is not None and
                        online_stage2_trainer is not None and
                        online_stage2_trainer.enable_stage2):
                        stage2_params = online_stage2_trainer.get_stage2_parameters()
                        if stage2_params:
                            # 第二阶段使用独立的损失和参数
                            loss_scaler(
                                stage2_loss,
                                optimizer,
                                parameters=stage2_params,  # 只更新第二阶段refine模型参数
                                update_grad=True,
                                clip_grad=1.0,
                            )
                            optimizer.zero_grad()
                        else:
                            print("Warning: Stage2 enabled but no parameters found for update")
                    else:
                        print("Warning: Stage2 training requested but stage2_loss is None")

                elif training_stage == 'joint':
                    # ============ 联合训练模式 ============
                    # 同时更新第一阶段和第二阶段参数（如果第二阶段启用）
                    # 第一阶段参数更新
                    loss_scaler(
                        loss,
                        optimizer,
                        parameters=model.parameters(),
                        update_grad=True,
                        clip_grad=1.0,
                    )
                    optimizer.zero_grad()

                    # 第二阶段参数更新（如果启用）
                    if (stage2_loss is not None and
                        online_stage2_trainer is not None and
                        online_stage2_trainer.enable_stage2):
                        stage2_params = online_stage2_trainer.get_stage2_parameters()
                        if stage2_params:
                            loss_scaler(
                                stage2_loss,
                                optimizer,
                                parameters=stage2_params,
                                update_grad=True,
                                clip_grad=1.0,
                            )
                            optimizer.zero_grad()
                else:
                    raise ValueError(f"Unknown training_stage: {training_stage}. Must be 'stage1', 'stage2', or 'joint'")


                # 更新最终的损失记录
                # Stage2模式下使用stage2_loss，否则使用stage1的loss_value
                if training_stage == 'stage2' and stage2_loss is not None:
                    final_loss_value = float(stage2_loss)
                else:
                    final_loss_value = loss_value

                metric_logger.meters["loss"].update(final_loss_value)
                if log_writer is not None:
                    step = epoch * len(data_loader_train) + data_iter_step
                    log_writer.add_scalar("train_loss_final", final_loss_value, step)

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

    save_final_model(accelerator, args, args.epochs, model, online_stage2_trainer, best_so_far=best_so_far)


def save_final_model(accelerator, args, epoch, model_without_ddp, stage2_trainer=None, best_so_far=None):
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
    
    # 保存第二阶段模型最终状态
    if stage2_trainer is not None and stage2_trainer.enable_stage2:
        if accelerator.is_main_process:
            stage2_final_path = output_dir / "stage2-checkpoint-final.pth"
            stage2_state = stage2_trainer.save_state_dict()
            misc.save_on_master(accelerator, stage2_state, stage2_final_path)
            printer.info(f">> Saving Stage2 final model to {stage2_final_path} ...")


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


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    epoch: int,
    loss_scaler,
    args,
    log_writer=None,
):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.accum_iter

    def save_model(epoch, fname, best_so_far):
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

    if log_writer is not None:
        printer.info("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(epoch)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, accelerator, header)
    ):
        with accelerator.accumulate(model):
            epoch_f = epoch + data_iter_step / len(data_loader)
            step = int(epoch_f * len(data_loader))
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                misc.adjust_learning_rate(optimizer, epoch_f, args)
            if not args.long_context:
                result = loss_of_one_batch(
                    batch,
                    model,
                    criterion,
                    accelerator,
                    symmetrize_batch=False,
                    use_amp=bool(args.amp),
                )
            else:
                result = loss_of_one_batch_tbptt(
                    batch,
                    model,
                    criterion,
                    chunk_size=4,
                    loss_scaler=loss_scaler,
                    optimizer=optimizer,
                    accelerator=accelerator,
                    symmetrize_batch=False,
                    use_amp=bool(args.amp),
                )
            loss, loss_details = result["loss"]  # criterion returns two values

            loss_value = float(loss)

            if not math.isfinite(loss_value):
                print(
                    f"Loss is {loss_value}, stopping training, loss details: {loss_details}"
                )
                sys.exit(1)
            if not result.get("already_backprop", False):
                loss_scaler(
                    loss,
                    optimizer,
                    parameters=model.parameters(),
                    update_grad=True,
                    clip_grad=1.0,
                )
                optimizer.zero_grad()

            is_metric = batch[0]["is_metric"]
            curr_num_view = len(batch)

            del loss
            tb_vis_img = (data_iter_step + 1) % accum_iter == 0 and (
                (step + 1) % (args.print_img_freq)
            ) == 0
            if not tb_vis_img:
                del batch
            else:
                torch.cuda.empty_cache()

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(epoch=epoch_f)
            metric_logger.update(lr=lr)
            metric_logger.update(step=step)

            metric_logger.update(loss=loss_value, **loss_details)

            if (data_iter_step + 1) % accum_iter == 0 and (
                (data_iter_step + 1) % (accum_iter * args.print_freq)
            ) == 0:
                loss_value_reduce = accelerator.gather(
                    torch.tensor(loss_value).to(accelerator.device)
                ).mean()  # MUST BE EXECUTED BY ALL NODES

                if log_writer is None:
                    continue
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int(epoch_f * 1000)
                log_writer.add_scalar("train_loss", loss_value_reduce, step)
                log_writer.add_scalar("train_lr", lr, step)
                log_writer.add_scalar("train_iter", epoch_1000x, step)
                for name, val in loss_details.items():
                    if isinstance(val, torch.Tensor):
                        if val.ndim > 0:
                            continue
                    if isinstance(val, dict):
                        continue
                    log_writer.add_scalar("train_" + name, val, step)

            if tb_vis_img:
                if log_writer is None:
                    continue
                # with torch.no_grad():
                #     depths_self, gt_depths_self = get_render_results(
                #         batch, result["pred"], self_view=True
                #     )
                #     depths_cross, gt_depths_cross = get_render_results(
                #         batch, result["pred"], self_view=False
                #     )
                #     for k in range(len(batch)):
                #         loss_details[f"self_pred_depth_{k+1}"] = (
                #             depths_self[k].detach().cpu()
                #         )
                #         loss_details[f"self_gt_depth_{k+1}"] = (
                #             gt_depths_self[k].detach().cpu()
                #         )
                #         loss_details[f"pred_depth_{k+1}"] = (
                #             depths_cross[k].detach().cpu()
                #         )
                #         loss_details[f"gt_depth_{k+1}"] = (
                #             gt_depths_cross[k].detach().cpu()
                #         )

                # imgs_stacked_dict = get_vis_imgs_new(
                #     loss_details, args.num_imgs_vis, curr_num_view, is_metric=is_metric
                # )
                # for name, imgs_stacked in imgs_stacked_dict.items():
                #     log_writer.add_images(
                #         "train" + "/" + name, imgs_stacked, step, dataformats="HWC"
                # )
                # del batch

        if (
            data_iter_step % int(args.save_freq * len(data_loader)) == 0
            and data_iter_step != 0
            and data_iter_step != len(data_loader) - 1
        ):
            print("saving at step", data_iter_step)
            save_model(epoch - 1, f"epoch_{epoch}_{data_iter_step}", float("inf"))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes(accelerator)
    printer.info("Averaged stats: %s", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    accelerator: Accelerator,
    device: torch.device,
    epoch: int,
    args,
    log_writer=None,
    prefix="test",
):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = "Test Epoch: [{}]".format(epoch)

    if log_writer is not None:
        printer.info("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(0)
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(0)

    for _, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, accelerator, header)
    ):
        result = loss_of_one_batch(
            batch,
            model,
            criterion,
            accelerator,
            symmetrize_batch=False,
            use_amp=bool(args.amp),
        )

        loss_value, loss_details = result["loss"]  # criterion returns two values
        metric_logger.update(loss=float(loss_value), **loss_details)

    printer.info("Averaged stats: %s", metric_logger)

    aggs = [("avg", "global_avg"), ("med", "median")]
    results = {
        f"{k}_{tag}": getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }

    if log_writer is not None:
        for name, val in results.items():
            if isinstance(val, torch.Tensor):
                if val.ndim > 0:
                    continue
            if isinstance(val, dict):
                continue
            log_writer.add_scalar(prefix + "_" + name, val, 1000 * epoch)

        depths_self, gt_depths_self = get_render_results(
            batch, result["pred"], self_view=True
        )
        depths_cross, gt_depths_cross = get_render_results(
            batch, result["pred"], self_view=False
        )
        for k in range(len(batch)):
            loss_details[f"self_pred_depth_{k+1}"] = depths_self[k].detach().cpu()
            loss_details[f"self_gt_depth_{k+1}"] = gt_depths_self[k].detach().cpu()
            loss_details[f"pred_depth_{k+1}"] = depths_cross[k].detach().cpu()
            loss_details[f"gt_depth_{k+1}"] = gt_depths_cross[k].detach().cpu()

        imgs_stacked_dict = get_vis_imgs_new(
            loss_details,
            args.num_imgs_vis,
            args.num_test_views,
            is_metric=batch[0]["is_metric"],
        )
        for name, imgs_stacked in imgs_stacked_dict.items():
            log_writer.add_images(
                prefix + "/" + name, imgs_stacked, 1000 * epoch, dataformats="HWC"
            )

    del loss_details, loss_value, batch
    torch.cuda.empty_cache()

    return results


def batch_append(original_list, new_list):
    for sublist, new_item in zip(original_list, new_list):
        sublist.append(new_item)
    return original_list


def gen_mask_indicator(img_mask_list, ray_mask_list, num_views, h, w):
    output = []
    for img_mask, ray_mask in zip(img_mask_list, ray_mask_list):
        out = torch.zeros((h, w * num_views, 3))
        for i in range(num_views):
            if img_mask[i] and not ray_mask[i]:
                offset = 0
            elif not img_mask[i] and ray_mask[i]:
                offset = 1
            else:
                offset = 0.5
            out[:, i * w : (i + 1) * w] += offset
        output.append(out)
    return output


def vis_and_cat(
    gt_imgs,
    pred_imgs,
    cross_gt_depths,
    cross_pred_depths,
    self_gt_depths,
    self_pred_depths,
    cross_conf,
    self_conf,
    ray_indicator,
    is_metric,
):
    cross_depth_gt_min = torch.quantile(cross_gt_depths, 0.01).item()
    cross_depth_gt_max = torch.quantile(cross_gt_depths, 0.99).item()
    cross_depth_pred_min = torch.quantile(cross_pred_depths, 0.01).item()
    cross_depth_pred_max = torch.quantile(cross_pred_depths, 0.99).item()
    cross_depth_min = min(cross_depth_gt_min, cross_depth_pred_min)
    cross_depth_max = max(cross_depth_gt_max, cross_depth_pred_max)

    cross_gt_depths_vis = colorize(
        cross_gt_depths,
        range=(
            (cross_depth_min, cross_depth_max)
            if is_metric
            else (cross_depth_gt_min, cross_depth_gt_max)
        ),
        append_cbar=True,
    )
    cross_pred_depths_vis = colorize(
        cross_pred_depths,
        range=(
            (cross_depth_min, cross_depth_max)
            if is_metric
            else (cross_depth_pred_min, cross_depth_pred_max)
        ),
        append_cbar=True,
    )

    self_depth_gt_min = torch.quantile(self_gt_depths, 0.01).item()
    self_depth_gt_max = torch.quantile(self_gt_depths, 0.99).item()
    self_depth_pred_min = torch.quantile(self_pred_depths, 0.01).item()
    self_depth_pred_max = torch.quantile(self_pred_depths, 0.99).item()
    self_depth_min = min(self_depth_gt_min, self_depth_pred_min)
    self_depth_max = max(self_depth_gt_max, self_depth_pred_max)

    self_gt_depths_vis = colorize(
        self_gt_depths,
        range=(
            (self_depth_min, self_depth_max)
            if is_metric
            else (self_depth_gt_min, self_depth_gt_max)
        ),
        append_cbar=True,
    )
    self_pred_depths_vis = colorize(
        self_pred_depths,
        range=(
            (self_depth_min, self_depth_max)
            if is_metric
            else (self_depth_pred_min, self_depth_pred_max)
        ),
        append_cbar=True,
    )
    if len(cross_conf) > 0:
        cross_conf_vis = colorize(cross_conf, append_cbar=True)
    if len(self_conf) > 0:
        self_conf_vis = colorize(self_conf, append_cbar=True)
    gt_imgs_vis = torch.zeros_like(cross_gt_depths_vis)
    gt_imgs_vis[: gt_imgs.shape[0], : gt_imgs.shape[1]] = gt_imgs
    pred_imgs_vis = torch.zeros_like(cross_gt_depths_vis)
    pred_imgs_vis[: pred_imgs.shape[0], : pred_imgs.shape[1]] = pred_imgs
    ray_indicator_vis = torch.cat(
        [
            ray_indicator,
            torch.zeros(
                ray_indicator.shape[0],
                cross_pred_depths_vis.shape[1] - ray_indicator.shape[1],
                3,
            ),
        ],
        dim=1,
    )
    out = torch.cat(
        [
            ray_indicator_vis,
            gt_imgs_vis,
            pred_imgs_vis,
            self_gt_depths_vis,
            self_pred_depths_vis,
            self_conf_vis,
            cross_gt_depths_vis,
            cross_pred_depths_vis,
            cross_conf_vis,
        ],
        dim=0,
    )
    return out


def get_vis_imgs_new(loss_details, num_imgs_vis, num_views, is_metric):
    ret_dict = {}
    gt_img_list = [[] for _ in range(num_imgs_vis)]
    pred_img_list = [[] for _ in range(num_imgs_vis)]

    cross_gt_depth_list = [[] for _ in range(num_imgs_vis)]
    cross_pred_depth_list = [[] for _ in range(num_imgs_vis)]

    self_gt_depth_list = [[] for _ in range(num_imgs_vis)]
    self_pred_depth_list = [[] for _ in range(num_imgs_vis)]

    cross_view_conf_list = [[] for _ in range(num_imgs_vis)]
    self_view_conf_list = [[] for _ in range(num_imgs_vis)]
    cross_view_conf_exits = False
    self_view_conf_exits = False

    img_mask_list = [[] for _ in range(num_imgs_vis)]
    ray_mask_list = [[] for _ in range(num_imgs_vis)]

    if num_views > 30:
        stride = 5
    elif num_views > 20:
        stride = 3
    elif num_views > 10:
        stride = 2
    else:
        stride = 1
    for i in range(0, num_views, stride):
        gt_imgs = 0.5 * (loss_details[f"gt_img{i+1}"] + 1)[:num_imgs_vis].detach().cpu()
        width = gt_imgs.shape[2]
        pred_imgs = (
            0.5 * (loss_details[f"pred_rgb_{i+1}"] + 1)[:num_imgs_vis].detach().cpu()
        )
        gt_img_list = batch_append(gt_img_list, gt_imgs.unbind(dim=0))
        pred_img_list = batch_append(pred_img_list, pred_imgs.unbind(dim=0))

        cross_pred_depths = (
            loss_details[f"pred_depth_{i+1}"][:num_imgs_vis].detach().cpu()
        )
        cross_gt_depths = (
            loss_details[f"gt_depth_{i+1}"]
            .to(gt_imgs.device)[:num_imgs_vis]
            .detach()
            .cpu()
        )
        cross_pred_depth_list = batch_append(
            cross_pred_depth_list, cross_pred_depths.unbind(dim=0)
        )
        cross_gt_depth_list = batch_append(
            cross_gt_depth_list, cross_gt_depths.unbind(dim=0)
        )

        self_gt_depths = (
            loss_details[f"self_gt_depth_{i+1}"][:num_imgs_vis].detach().cpu()
        )
        self_pred_depths = (
            loss_details[f"self_pred_depth_{i+1}"][:num_imgs_vis].detach().cpu()
        )
        self_gt_depth_list = batch_append(
            self_gt_depth_list, self_gt_depths.unbind(dim=0)
        )
        self_pred_depth_list = batch_append(
            self_pred_depth_list, self_pred_depths.unbind(dim=0)
        )

        if f"conf_{i+1}" in loss_details:
            cross_view_conf = loss_details[f"conf_{i+1}"][:num_imgs_vis].detach().cpu()
            cross_view_conf_list = batch_append(
                cross_view_conf_list, cross_view_conf.unbind(dim=0)
            )
            cross_view_conf_exits = True

        if f"self_conf_{i+1}" in loss_details:
            self_view_conf = (
                loss_details[f"self_conf_{i+1}"][:num_imgs_vis].detach().cpu()
            )
            self_view_conf_list = batch_append(
                self_view_conf_list, self_view_conf.unbind(dim=0)
            )
            self_view_conf_exits = True

        img_mask_list = batch_append(
            img_mask_list,
            loss_details[f"img_mask_{i+1}"][:num_imgs_vis].detach().cpu().unbind(dim=0),
        )
        ray_mask_list = batch_append(
            ray_mask_list,
            loss_details[f"ray_mask_{i+1}"][:num_imgs_vis].detach().cpu().unbind(dim=0),
        )

    # each element in the list is [H, num_views * W, (3)], the size of the list is num_imgs_vis
    gt_img_list = [torch.cat(sublist, dim=1) for sublist in gt_img_list]
    pred_img_list = [torch.cat(sublist, dim=1) for sublist in pred_img_list]
    cross_pred_depth_list = [
        torch.cat(sublist, dim=1) for sublist in cross_pred_depth_list
    ]
    cross_gt_depth_list = [torch.cat(sublist, dim=1) for sublist in cross_gt_depth_list]
    self_gt_depth_list = [torch.cat(sublist, dim=1) for sublist in self_gt_depth_list]
    self_pred_depth_list = [
        torch.cat(sublist, dim=1) for sublist in self_pred_depth_list
    ]
    cross_view_conf_list = (
        [torch.cat(sublist, dim=1) for sublist in cross_view_conf_list]
        if cross_view_conf_exits
        else []
    )
    self_view_conf_list = (
        [torch.cat(sublist, dim=1) for sublist in self_view_conf_list]
        if self_view_conf_exits
        else []
    )
    # each elment in the list is [num_views,], the size of the list is num_imgs_vis
    img_mask_list = [torch.stack(sublist, dim=0) for sublist in img_mask_list]
    ray_mask_list = [torch.stack(sublist, dim=0) for sublist in ray_mask_list]

    ray_indicator = gen_mask_indicator(
        img_mask_list, ray_mask_list, len(img_mask_list[0]), 30, width
    )

    for i in range(num_imgs_vis):
        out = vis_and_cat(
            gt_img_list[i],
            pred_img_list[i],
            cross_gt_depth_list[i],
            cross_pred_depth_list[i],
            self_gt_depth_list[i],
            self_pred_depth_list[i],
            cross_view_conf_list[i],
            self_view_conf_list[i],
            ray_indicator[i],
            is_metric[i],
        )
        ret_dict[f"imgs_{i}"] = out
    return ret_dict


def sam2_velocity_consistency_loss(images, velocity, sam2_model, device):
    """
    Compute velocity consistency loss using SAM2 masks.
    
    Args:
        images: [B, S, 3, H, W] - input images
        velocity: [B, S, H, W, 3] - predicted velocity
        sam2_model: SAM2AutomaticMaskGenerator instance
        device: torch device
    
    Returns:
        dict: loss dictionary containing sam2_velocity_consistency_loss
    """
    from vggt.training.loss import sam2_velocity_consistency_loss_impl
    
    return sam2_velocity_consistency_loss_impl(images, velocity, sam2_model, device)


def sam2_velocity_consistency_loss_with_preprocessed_masks(images, velocity, sam_masks, device):
    """
    Compute velocity consistency loss using preprocessed SAM masks.
    
    Args:
        images: [B, S, 3, H, W] - input images
        velocity: [B, S, H, W, 3] - predicted velocity
        sam_masks: [B, S, num_masks, H, W] - preprocessed SAM masks
        device: torch device
    
    Returns:
        dict: loss dictionary containing sam2_velocity_consistency_loss
    """
    from vggt.training.loss import sam2_velocity_consistency_loss_with_masks_impl
    
    return sam2_velocity_consistency_loss_with_masks_impl(images, velocity, sam_masks, device)


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

    # 处理SAM掩码（如果存在）
    if 'sam_masks' in views[0]:
        sam_masks_list = []
        for v in views:
            if v['sam_masks'] is not None:
                # 确保SAM掩码是tensor格式
                if isinstance(v['sam_masks'], np.ndarray):
                    sam_masks = torch.from_numpy(v['sam_masks']).float()
                else:
                    sam_masks = v['sam_masks'].float()
                
                # 处理形状为 (1, num_mask, h, w) 的情况
                if sam_masks.dim() == 4 and sam_masks.shape[0] == 1:
                    sam_masks = sam_masks.squeeze(0)  # 移除batch维度，变成 (num_mask, h, w)
                
                sam_masks_list.append(sam_masks)
            else:
                # 如果没有SAM掩码，创建空的tensor
                sam_masks_list.append(torch.zeros(0, v['img'].shape[1], v['img'].shape[2]))
        
        # 找到最大数量的掩码
        max_num_masks = max(masks.shape[0] for masks in sam_masks_list)
        
        # 填充到相同数量的掩码
        padded_sam_masks = []
        for masks in sam_masks_list:
            if masks.shape[0] < max_num_masks:
                # 用零填充
                padding = torch.zeros(max_num_masks - masks.shape[0], masks.shape[1], masks.shape[2]).to(masks.device)
                masks = torch.cat([masks, padding], dim=0)
            padded_sam_masks.append(masks)
        
        vggt_batch['sam_masks'] = torch.stack(padded_sam_masks, dim=0)  # [S, num_masks, H, W]

    with tf32_off(), torch.amp.autocast("cuda", enabled=False):
        # 转换world points的坐标系到第一帧相机坐标系
        if vggt_batch['world_points'] is not None:
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
    
    # 处理SAM掩码的维度转换
    if 'sam_masks' in vggt_batch:
        # 从 [S, num_masks, H, W] 转换为 [B, S, num_masks, H, W]，其中B=1
        vggt_batch['sam_masks'] = vggt_batch['sam_masks'].unsqueeze(0).contiguous()  # [B, S, num_masks, H, W]

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
        debugpy.listen(5691)
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
    logdir = pathlib.Path(cfg.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    run()
