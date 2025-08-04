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
from vggt.training.loss import camera_loss, depth_loss, point_loss, cross_render_and_loss, flow_loss, self_render_and_loss, velocity_loss, sky_opacity_loss, sky_color_loss, vggt_distillation_loss
from vggt.utils.auxiliary import RAFTCfg, calc_flow
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# ===== SAM2相关导入 =====
# 添加sam2路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'sam2'))
# 延迟导入SAM2，避免Hydra冲突
# from sam2.build_sam import build_sam2
# from sam2.modeling.sam2_base import SAM2Base

# ===== DAM2相关导入 =====
# 添加dam2路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'dam2'))
# 延迟导入DAM2，避免Hydra冲突
# from depth_anything_v2.dpt import DepthAnythingV2

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
    model.gradient_checkpointing_enable(True)
    printer.info(f"All model parameters: {sum(p.numel() for p in model.parameters())}")

    model.to(device)

    if args.pretrained and not args.resume:
        printer.info(f"Loading pretrained: {args.pretrained}")
        # ckpt = torch.load(args.pretrained, map_location=device, weights_only=True)
        ckpt = torch.load(args.pretrained, map_location=device)['model']
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt, strict=False)
        del ckpt
        

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
                        teacher_model = VGGT(img_size=518, patch_size=14, embed_dim=1024)
                        
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
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler(accelerator=accelerator)

    accelerator.even_batches = True
    optimizer, model, data_loader_train = accelerator.prepare(
        optimizer, model, data_loader_train
    )


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

                if "dam2" in auxiliary_models and vggt_batch.get("images") is not None:
                    sky_masks = dam2_sky_mask_generation(
                        vggt_batch["images"], 
                        auxiliary_models["dam2"],
                        device=device
                    )
                    # 将sky masks添加到vggt_batch中，供后续使用
                    vggt_batch["sky_masks"] = sky_masks

                preds = model(
                    vggt_batch["images"],
                    compute_sky_color_loss=True,
                    sky_masks=vggt_batch["sky_masks"],
                    gt_images=vggt_batch["images"],
                    # pose_enc=preds["pose_enc"].detach()
                )

                loss = 0.0
                loss_dict = {}
                
                # VGGT teacher distillation
                if "vggt_teacher" in auxiliary_models and vggt_batch.get("images") is not None:
                    try:
                        with torch.no_grad():
                            teacher_preds = auxiliary_models["vggt_teacher"](
                                vggt_batch["images"],
                                compute_sky_color_loss=False,
                                sky_masks=vggt_batch.get("sky_masks"),
                                gt_images=vggt_batch["images"],
                            )
                        
                        # 计算蒸馏损失
                        distillation_loss_dict = vggt_distillation_loss(
                            student_preds=preds,
                            teacher_preds=teacher_preds,
                            weight_pose=1.0,  # 可以根据需要调整权重
                            weight_depth=1.0,  # 可以根据需要调整权重
                            weight_depth_conf=1.0  # 可以根据需要调整权重
                        )
                        
                        # 将蒸馏损失添加到总损失中
                        loss += distillation_loss_dict["loss_pose_distillation"] + distillation_loss_dict["loss_depth_distillation"] + distillation_loss_dict["loss_depth_conf_distillation"]
                        loss_dict.update(distillation_loss_dict)
                        
                    except Exception as e:
                        print(f"Error in VGGT teacher distillation: {e}")
                        # 添加零损失作为fallback
                        loss_dict.update({
                            "loss_pose_distillation": torch.tensor(0.0, device=device, requires_grad=True),
                            "loss_depth_distillation": torch.tensor(0.0, device=device, requires_grad=True),
                            "loss_depth_conf_distillation": torch.tensor(0.0, device=device, requires_grad=True),
                        })
                
                

                interval = 2

                forward_flow, backward_flow, forward_consist_mask, backward_consist_mask, forward_in_bound_mask, backward_in_bound_mask = calc_flow(
                    vggt_batch["images"], auxiliary_models["flow"],
                    check_consistency=True,
                    geo_thresh=auxiliary_models["flow"].args.geo_thresh,
                    photo_thresh=auxiliary_models["flow"].args.photo_thresh,
                    interval=interval,
                )

                # camera loss
                # if vggt_batch.get("extrinsics") is not None and vggt_batch.get("intrinsics") is not None and vggt_batch.get("point masks") is not None:
                #     cam_loss, __ = camera_loss([preds["pose_enc"]], vggt_batch)
                #     loss += cam_loss["loss_camera"]
                #     loss_dict.update(cam_loss)

                # point loss
                # if vggt_batch.get("world_points") is not None and vggt_batch.get("point_masks") is not None:
                #     point_loss_dict = point_loss(preds["world_points"], preds["world_points_conf"], vggt_batch)
                #     loss += 0.00000001 * point_loss_dict.get("loss_conf", 0.0)
                #     loss_dict.update(point_loss_dict)

                # depth loss
                # if vggt_batch.get("depths") is not None and vggt_batch.get("point_masks") is not None:
                #     depth_loss_dict = depth_loss(preds["depth"], preds["depth_conf"], vggt_batch)
                #     loss += depth_loss_dict.get("loss_conf_depth", 0.0)
                #     loss_dict.update(depth_loss_dict)

                # # gaussian loss (cross)
                # if vggt_batch.get("images") is not None and vggt_batch.get("depths") is not None:
                #     conf = preds["depth_conf"] > 2
                #     try:
                #         gaussian_loss_dict, _ = cross_render_and_loss(conf, interval, forward_consist_mask, backward_consist_mask, preds["depth"].detach(), preds["gaussian_params"], preds["velocity"], preds["pose_enc"], vggt_batch["extrinsics"], vggt_batch["intrinsics"], vggt_batch["images"], vggt_batch["depths"], vggt_batch["point_masks"])
                #         loss += gaussian_loss_dict.get("loss_render_rgb", 0.0) + 0.0 * gaussian_loss_dict.get("loss_render_lpips", 0.0) + gaussian_loss_dict.get("loss_render_depth", 0.0) 
                #         loss_dict.update(gaussian_loss_dict)
                #     except Exception as e:
                #         print(f"Error in gaussian loss computation: {e}")
                #         # 添加零损失作为fallback
                #         loss_dict.update({
                #             "loss_render_rgb": torch.tensor(0.0, device=device, requires_grad=True),
                #             "loss_render_lpips": torch.tensor(0.0, device=device, requires_grad=True),
                #             "loss_render_depth": torch.tensor(0.0, device=device, requires_grad=True),
                #         })

                # self render loss (self)
                if vggt_batch.get("images") is not None and vggt_batch.get("depths") is not None:
                    try:
                        self_loss_dict, _ = self_render_and_loss(preds["depth"].detach(), preds["gaussian_params"], preds["pose_enc"], vggt_batch["extrinsics"], vggt_batch["intrinsics"], vggt_batch["images"])
                        loss += self_loss_dict.get("loss_self_render_rgb", 0.0) + 0.0 * self_loss_dict.get("loss_self_render_lpips", 0.0) + self_loss_dict.get("loss_self_render_depth", 0.0)
                        loss_dict.update(self_loss_dict)
                    except Exception as e:
                        print(f"Error in self render loss computation: {e}")
                        # 添加零损失作为fallback
                        loss_dict.update({
                            "loss_self_render_rgb": torch.tensor(0.0, device=device, requires_grad=True),
                            "loss_self_render_lpips": torch.tensor(0.0, device=device, requires_grad=True),
                            "loss_self_render_depth": torch.tensor(0.0, device=device, requires_grad=True),
                        })

                # optical flow -> velocity loss
                if vggt_batch.get("images") is not None and vggt_batch.get("depths") is not None:
                    conf = preds["depth_conf"] > 2
                    try:
                        flow_loss_dict = flow_loss(conf, interval, forward_flow, backward_flow, forward_consist_mask, backward_consist_mask, preds["depth"], preds["velocity"], preds["pose_enc"], vggt_batch["extrinsics"], vggt_batch["intrinsics"], vggt_batch["images"])
                        loss += flow_loss_dict.get("forward_loss", 0.0) # + flow_loss_dict.get("backward_loss", 0.0)
                        loss_dict.update(flow_loss_dict)
                    except Exception as e:
                        print(f"Error in flow loss computation: {e}")
                        # 添加零损失作为fallback
                        loss_dict.update({
                            "forward_loss": torch.tensor(0.0, device=device, requires_grad=True),
                            "backward_loss": torch.tensor(0.0, device=device, requires_grad=True),
                        })
                
                # velocity regularization loss
                if vggt_batch.get("images") is not None:
                    try:
                        velocity_loss_value = velocity_loss(preds["velocity"])
                        loss += 0.001 * velocity_loss_value  # 使用较小的权重
                        loss_dict.update({"loss_velocity": velocity_loss_value})
                    except Exception as e:
                        print(f"Error in velocity loss computation: {e}")
                        # 添加零损失作为fallback
                        loss_dict.update({
                            "loss_velocity": torch.tensor(0.0, device=device, requires_grad=True),
                        })
                
                # # SAM2 mask-based velocity consistency loss
                # if "sam2" in auxiliary_models and vggt_batch.get("images") is not None:
                #     try:
                #         sam2_velocity_loss_dict = sam2_velocity_consistency_loss(
                #             vggt_batch["images"], 
                #             preds["velocity"], 
                #             auxiliary_models["sam2"],
                #             device=device
                #         )
                #         loss += sam2_velocity_loss_dict.get("sam2_velocity_consistency_loss", 0.0)
                #         loss_dict.update(sam2_velocity_loss_dict)
                #     except Exception as e:
                #         print(f"Error in SAM2 loss computation: {e}")
                #         # 添加零损失作为fallback
                #         loss_dict.update({
                #             "sam2_velocity_consistency_loss": torch.tensor(0.0, device=device, requires_grad=True),
                #         })
                
                # DAM2 sky mask generation and sky loss                    
                # Sky opacity loss: 鼓励sky区域的gaussian opacity为0
                if preds.get("gaussian_params") is not None:
                    sky_opacity_loss_dict = sky_opacity_loss(
                        preds["gaussian_params"], 
                        sky_masks, 
                        weight=1.0
                    )
                    loss += sky_opacity_loss_dict.get("sky_opacity_loss", 0.0)
                    loss_dict.update(sky_opacity_loss_dict)
                    
                    # 计算sky color loss
                    sky_color_loss_dict = sky_color_loss(
                        preds["pred_sky_colors"], 
                        vggt_batch["images"], 
                        sky_masks, 
                        weight=1.0
                    )
                    loss += sky_color_loss_dict.get("sky_color_loss", 0.0)
                    loss_dict.update(sky_color_loss_dict)

            
                
                loss_value = float(loss)
                
                loss_scaler(
                    loss,
                    optimizer,
                    parameters=model.parameters(),
                    update_grad=True,
                    clip_grad=1.0,
                )
                optimizer.zero_grad()
                lr = optimizer.param_groups[0]["lr"]
                metric_logger.update(epoch=epoch)
                metric_logger.update(lr=lr)
                metric_logger.update(loss=loss_value, **loss_dict)
                if log_writer is not None:
                    step = epoch * len(data_loader_train) + data_iter_step
                    log_writer.add_scalar("train_loss", loss_value, step)
                    log_writer.add_scalar("train_lr", lr, step)
                    for name, val in loss_dict.items():
                        if isinstance(val, torch.Tensor):
                            if val.ndim > 0:
                                continue
                        if isinstance(val, dict):
                            continue
                        log_writer.add_scalar("train_" + name, val, step)

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
    }

    with tf32_off(), torch.amp.autocast("cuda", enabled=False):
        # 转换world points的坐标系到第一帧相机坐标系
        B, S, H, W, _ = vggt_batch['world_points'].shape
        world_points = vggt_batch['world_points'].reshape(B, S, H*W, 3)
        world_points = torch.matmul(torch.linalg.inv(vggt_batch['extrinsics'][0])[:, :3, :3], world_points.transpose(-1, -2)).transpose(-1, -2) + \
                                   torch.linalg.inv(vggt_batch['extrinsics'][0])[:, :3, 3:4].transpose(-1, -2)
        vggt_batch['world_points'] = world_points.reshape(B, S, H, W, 3)

        # 转换extrinsics的坐标系到第一帧相机坐标系
        vggt_batch['extrinsics'] = torch.matmul(
                torch.linalg.inv(vggt_batch['extrinsics']),
                vggt_batch['extrinsics'][0]
            )

    vggt_batch['images'] = vggt_batch['images'].permute(1, 0, 2, 3, 4).contiguous()
    vggt_batch['depths'] = vggt_batch['depths'].permute(1, 0, 2, 3).contiguous() if vggt_batch['depths'] is not None else None
    vggt_batch['intrinsics'] = vggt_batch['intrinsics'].permute(1, 0, 2, 3).contiguous() if vggt_batch['intrinsics'] is not None else None
    vggt_batch['extrinsics'] = vggt_batch['extrinsics'].permute(1, 0, 2, 3).contiguous() if vggt_batch['extrinsics'] is not None else None
    vggt_batch['point_masks'] = vggt_batch['point_masks'].permute(1, 0, 2, 3).contiguous() if vggt_batch['point_masks'] is not None else None
    vggt_batch['world_points'] = vggt_batch['world_points'].permute(1, 0, 2, 3, 4).contiguous() if vggt_batch['world_points'] is not None else None

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
        debugpy.listen(5678)
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
    logdir = pathlib.Path(cfg.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    run()
