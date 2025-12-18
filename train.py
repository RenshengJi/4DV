# --------------------------------------------------------
# training code for CUT3R
# --------------------------------------------------------
# References:
# DUSt3R: https://github.com/naver/dust3r
# --------------------------------------------------------
import builtins
import datetime
import os
import pathlib
import random
import re
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path

import faulthandler
faulthandler.enable()

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing
from torch.utils.tensorboard.writer import SummaryWriter

import hydra
from omegaconf import OmegaConf

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datetime import timedelta

# ===== 添加子模块路径 =====
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/dam2'))

# ===== VGGT相关导入 =====
from src.models import VGGT
from src.losses import (
    camera_loss, depth_loss, gt_flow_loss_ours,
    self_render_and_loss, velocity_loss, sky_opacity_loss,
    scale_loss, segment_loss, Stage2CompleteLoss
)

# ===== Dynamic Processing =====
from src.dynamic_processing import DynamicProcessor

# ===== Dataset and Utils =====
from src.dataset import get_data_loader
from src.utils import training as misc
from src.utils import NativeScalerWithGradNormCount as NativeScaler

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
torch.autograd.set_detect_anomaly(True)

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
            ".claude*",
            "data*",
            "results*",
            "configs*",
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
        'velocity_transform_mode': getattr(args, 'velocity_transform_mode', 'simple')
    }

    # 初始化动态处理器
    dynamic_processor = DynamicProcessor(
        device=device,
        velocity_threshold=dynamic_processor_config.get('velocity_threshold', 0.1),
        clustering_eps=dynamic_processor_config.get('clustering_eps', 0.02),
        clustering_min_samples=dynamic_processor_config.get('clustering_min_samples', 10),
        min_object_size=dynamic_processor_config.get('min_object_size', 100),
        tracking_position_threshold=dynamic_processor_config.get('tracking_position_threshold', 2.0),
        registration_mode=dynamic_processor_config.get('velocity_transform_mode', 'simple'),
        use_registration=dynamic_processor_config.get('use_optical_flow_aggregation', True)
    )

    # 初始化Stage2损失函数（用于aggregator_all）
    stage2_loss_config = {
        'rgb_weight': getattr(args, 'aggregator_all_render_rgb_weight', 1.0),
        'depth_weight': getattr(args, 'aggregator_all_render_depth_weight', 1.0),
        'lpips_weight': getattr(args, 'aggregator_all_render_lpips_weight', 0.1),
        'render_only_dynamic': getattr(args, 'stage2_render_only_dynamic', False),
        'enable_voxel_pruning': getattr(args, 'enable_voxel_pruning', True),
        'voxel_size': getattr(args, 'voxel_size', 0.002),
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
                    dam2_path = os.path.join(os.path.dirname(__file__), 'src/dam2')
                    if dam2_path not in sys.path:
                        sys.path.insert(0, dam2_path)
                    from depth_anything_v2.dpt import DepthAnythingV2

                    # 解析DAM2配置
                    encoder_match = re.search(r"encoder\s*=\s*\"([^\"]+)\"", model_config)
                    offload_match = re.search(r"offload\s*=\s*(\"True\"|False)\b", model_config, re.IGNORECASE)

                    if encoder_match:
                        encoder = encoder_match.group(1)
                        # 直接从src目录加载模型文件
                        ckpt_path = os.path.join(os.path.dirname(__file__), f"src/depth_anything_v2_{encoder}.pth")
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



    # # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.get_parameter_groups(model, args.weight_decay)

    # 检查是否有可训练参数
    if not param_groups:
        raise ValueError(
            "No trainable parameters found! "
            "Please check vggt_freeze_strategy is not freezing all parameters\n"
            f"Stage1 trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

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

    for epoch in range(args.start_epoch, args.epochs):
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
        for data_iter_step, vggt_batch in enumerate(metric_logger.log_every(data_loader_train, args.print_freq, accelerator, header)):
            with accelerator.accumulate(model):
                epoch_f = epoch + data_iter_step / len(data_loader_train)
                if data_iter_step % args.accum_iter == 0:
                    misc.adjust_learning_rate(optimizer, epoch_f, args)

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
                    'self_render_rgb_weight': getattr(args, 'self_render_rgb_weight', 1.0),
                    'self_render_lpips_weight': getattr(args, 'self_render_lpips_weight', 0.0),
                    'self_render_depth_weight': getattr(args, 'self_render_depth_weight', 0.0),
                    'gt_flow_loss_ours_weight': getattr(args, 'gt_flow_loss_ours_weight', 0.0),  # GT flowmap ours损失
                    'sky_opacity_weight': getattr(args, 'sky_opacity_weight', 1.0),
                    'velocity_reg_weight': getattr(args, 'velocity_reg_weight', 0.001),
                    'camera_loss_weight': getattr(args, 'camera_loss_weight', 1.0),
                    'conf_depth_loss_weight': getattr(args, 'conf_depth_loss_weight', 1.0),
                    'grad_depth_loss_weight': getattr(args, 'grad_depth_loss_weight', 1.0),
                    'reg_depth_loss_weight': getattr(args, 'reg_depth_loss_weight', 1.0),
                    'scale_loss_weight': getattr(args, 'scale_loss_weight', 1.0),
                    'segment_loss_weight': getattr(args, 'segment_loss_weight', 0.0),  # Segmentation loss
                    'aggregator_all_render_rgb_weight': getattr(args, 'aggregator_all_render_rgb_weight', 0.0),
                    'aggregator_all_render_depth_weight': getattr(args, 'aggregator_all_render_depth_weight', 0.0),
                    'aggregator_all_render_lpips_weight': getattr(args, 'aggregator_all_render_lpips_weight', 0.0),
                }

                # 计算光流（已删除flow_loss，不再需要）
                interval = getattr(args, 'flow_interval', 2)

                # Self Render Loss
                if loss_weights['self_render_weight'] > 0 and vggt_batch.get("images") is not None and vggt_batch.get("depths") is not None:
                    try:
                        self_loss_dict, _ = self_render_and_loss(
                            vggt_batch=vggt_batch,
                            preds=preds,
                            enable_voxel_pruning=getattr(args, 'enable_voxel_pruning', True),
                            voxel_size=getattr(args, 'voxel_size', 0.002)
                        )
                        # 将所有self_render损失加入到总loss中
                        self_render_rgb_loss = self_loss_dict.get("loss_self_render_rgb", 0.0)
                        self_render_lpips_loss = self_loss_dict.get("loss_self_render_lpips", 0.0)
                        self_render_depth_loss = self_loss_dict.get("loss_self_render_depth", 0.0)

                        loss += (loss_weights.get('self_render_rgb_weight', 0.0) * self_render_rgb_loss +
                                loss_weights.get('self_render_lpips_weight', 0.0) * self_render_lpips_loss +
                                loss_weights.get('self_render_depth_weight', 0.0) * self_render_depth_loss)

                        # 将所有损失记录到loss_dict中
                        loss_dict.update(self_loss_dict)
                    except Exception as e:
                        print(f"Error in self render loss computation: {e}")

                # GT Flow Loss Ours
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

                # Sky Opacity Loss (监督gaussian head的opacity参数)
                if loss_weights['sky_opacity_weight'] > 0 and preds.get("gaussian_params") is not None and vggt_batch.get("sky_masks") is not None:
                    try:
                        sky_opacity_loss_dict = sky_opacity_loss(
                            preds["gaussian_params"],
                            vggt_batch["sky_masks"],
                            weight=1.0
                        )
                        sky_opacity_loss_value = sky_opacity_loss_dict.get("sky_opacity_loss", 0.0)
                        loss += loss_weights['sky_opacity_weight'] * sky_opacity_loss_value
                        loss_dict.update(sky_opacity_loss_dict)
                    except Exception as e:
                        print(f"Error in sky opacity loss computation: {e}")

                # Velocity Regularization Loss (约束velocity值)
                if loss_weights['velocity_reg_weight'] > 0 and preds.get("velocity") is not None:
                    try:
                        velocity_loss_value = velocity_loss(preds["velocity"])
                        loss += loss_weights['velocity_reg_weight'] * velocity_loss_value
                        loss_dict.update({"loss_velocity": velocity_loss_value})
                    except Exception as e:
                        print(f"Error in velocity regularization loss computation: {e}")

                # Segmentation Loss (监督segment head训练)
                if loss_weights['segment_loss_weight'] > 0 and preds.get("segment_logits") is not None and preds.get("segment_conf") is not None:
                    try:
                        segment_loss_dict = segment_loss(
                            preds["segment_logits"],
                            preds["segment_conf"],
                            vggt_batch
                        )
                        segment_loss_value = segment_loss_dict.get("segment_loss", 0.0)
                        loss += loss_weights['segment_loss_weight'] * segment_loss_value
                        loss_dict.update(segment_loss_dict)
                    except Exception as e:
                        print(f"Error in segmentation loss computation: {e}")

                # Camera Loss (监督camera head训练)
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

                # Depth Loss (监督depth head训练)
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

                # Scale Loss (监督scale head训练，使用depth_scale_factor作为GT)
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

                # Aggregator_all Render Loss (代替Stage2 refine网络的渲染loss)
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
                        # Step 1: Process dynamic objects (using DynamicProcessor)
                        preds_for_dynamic = preds.copy()
                        if args.aggregator_all_detach_velocity and 'velocity_global' in preds_for_dynamic:
                            preds_for_dynamic['velocity_global'] = preds_for_dynamic['velocity_global'].detach()

                        # Use new unified processor
                        result = dynamic_processor.process(preds_for_dynamic, vggt_batch)

                        # Convert to legacy format for compatibility
                        dynamic_objects_data = dynamic_processor.to_legacy_format(result)

                        # 检查是否有有效的动态物体（注意：即使没有动态物体，也要计算静态物体的loss）
                        num_cars = len(dynamic_objects_data.get('dynamic_objects_cars', [])) if dynamic_objects_data else 0
                        num_people = len(dynamic_objects_data.get('dynamic_objects_people', [])) if dynamic_objects_data else 0
                        has_valid_dynamic = (num_cars + num_people) > 0

                        # 无论是否有动态物体，都需要计算loss（因为有静态物体）
                        if dynamic_objects_data is not None:
                            # Step 2: 跳过Stage2 refine网络,直接构建场景
                            # 不调用 stage2_model()，直接使用原始的canonical gaussians
                            dynamic_objects_cars = dynamic_objects_data.get('dynamic_objects_cars', []) if has_valid_dynamic else []
                            dynamic_objects_people = dynamic_objects_data.get('dynamic_objects_people', []) if has_valid_dynamic else []
                            static_gaussians = dynamic_objects_data.get('static_gaussians')

                            # 构建"refined"场景 (实际上是未refine的原始数据)
                            aggregator_all_scene = {
                                'static_gaussians': static_gaussians,
                                'dynamic_objects_cars': dynamic_objects_cars,  # 车辆（使用刚体假设聚合）
                                'dynamic_objects_people': dynamic_objects_people  # 行人（每帧单独）
                            }

                            # Step 3: 计算渲染loss (使用Stage2的loss函数)
                            B, S, C, H, W = vggt_batch['images'].shape
                            gt_images = vggt_batch['images']
                            gt_depths = vggt_batch.get('depths', torch.ones(B, S, H, W, device=device) * 5.0)

                            # 使用预测的相机参数（从preds中获取）
                            intrinsics = preds['intrinsics']  # [B, S, 3, 3]
                            extrinsics = preds['extrinsics']  # [B, S, 4, 4]

                            sky_masks = vggt_batch.get('sky_masks', None)

                            # 获取sky_colors和sampled_frame_indices（如果有）
                            sky_colors = preds.get('sky_colors', None)  # [B, num_frames, 3, H, W]
                            sampled_frame_indices = preds.get('sampled_frame_indices', None)  # [num_frames]

                            # 使用Stage2的criterion计算loss
                            aggregator_all_loss_dict = stage2_criterion(
                                refinement_results={
                                    'refined_dynamic_objects_cars': dynamic_objects_cars,
                                    'refined_dynamic_objects_people': dynamic_objects_people
                                },  # 无refine结果
                                refined_scene=aggregator_all_scene,
                                gt_images=gt_images,
                                gt_depths=gt_depths,
                                intrinsics=intrinsics,
                                extrinsics=extrinsics,
                                sky_masks=sky_masks,
                                sky_colors=sky_colors,
                                sampled_frame_indices=sampled_frame_indices,
                                depth_scale_factor=vggt_batch.get('depth_scale_factor', None),
                                camera_indices=vggt_batch.get('camera_indices', None),  # New: for multi-camera support
                                frame_indices=vggt_batch.get('frame_indices', None)     # New: for multi-camera support
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
                                'aggregator_all_num_objects': num_cars + num_people,
                                'aggregator_all_num_cars': num_cars,
                                'aggregator_all_num_people': num_people
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

    # gather the stats from all processes (应该在epoch结束后，for循环外面)
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


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="train.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    if cfg.get("debug", False):
        cfg.num_workers = 0
        import debugpy
        debugpy.listen(5698)
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
    logdir = pathlib.Path(cfg.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    run()
