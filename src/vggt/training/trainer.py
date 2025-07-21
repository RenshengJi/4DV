# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
# Plan: This file should provide a trainer class, which provides
# 1. The init of DDP
# 2. The init of optimizers, tb, timers, and so on
# 3. A basic training framework (especially for finetuning)
#       self._train_epoch_
#       self._process_batch_
#       self._step_
# ETA: hope I can finish this before June 8th
'''
import contextlib

import copy
import functools
import gc
import json
import logging
import math
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

import random
import string
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision

from einops import rearrange
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from PIL import Image

from datetime import timedelta

#
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules
class Trainer:
    """
    Trainer supporting the DDP training strategies.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        resume_checkpoint_path: Optional[str] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
    ):
        self.resume_checkpoint_path = resume_checkpoint_path

        self._setup_env_variables(env_variables)
        self._setup_timers()

        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        # self.checkpoint_conf = TrainerCheckpointConf(**checkpoint).infer_missing()

        # hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.optim_conf = optim

        self.where = 0.0
        self.seed_value = seed_value

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        assert (
            is_dist_avail_and_initialized()
        ), "Torch distributed needs to be initialized before calling the trainer."

        self._setup_components()  # Except Optimizer everything is setup here.
        self._setup_dataloaders()
        self._move_to_device()

        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.2f")

        if self.mode != "val":
            self._construct_optimizers()

        ################################
        for _ in range(10):
            print(f"Custom resume ckpt: {self.resume_checkpoint_path}")
        if self.resume_checkpoint_path is not None:
            self._load_resuming_checkpoint(self.resume_checkpoint_path)
        ################################

        self.load_checkpoint()
        self._setup_ddp_distributed_training(distributed, device)

        dist.barrier()

    def _setup_timers(self):
        """
        Initializes counters for elapsed time and eta.
        """
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0


    def _get_meters(self, phase_filters=None):
        if self.meters is None:
            return {}
        meters = {}
        for phase, phase_meters in self.meters.items():
            if phase_filters is not None and phase not in phase_filters:
                continue
            for key, key_meters in phase_meters.items():
                for name, meter in key_meters.items():
                    meters[f"{phase}_{key}/{name}"] = meter
        return meters


    def _setup_env_variables(self, env_variables_conf) -> None:
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        print(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf, distributed_conf) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        dist.init_process_group(backend=distributed_conf.backend,
                timeout=timedelta(minutes=distributed_conf.timeout_mins))

        self.rank = dist.get_rank()


    def _setup_device(self, device):
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")


    def _setup_components(self):
        logging.info("Setting up components: Model, loss, optim, meters etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}
        self.prev_epoch_data_ids = {'train': None, 'val': None}

        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        if self.optim_conf.frozen_module_names is not None:
            logging.info(
                f"[Start] Freezing 'frozen_module_names' on rank {self.distributed_rank}"
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(
                f"[Done] Freezing 'frozen_module_names' on rank {self.distributed_rank}"
            )


        model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
        model_summary(self.model, log_file=model_summary_path)
        logging.info(f"Model summary saved to {model_summary_path}")


        import pdb;pdb.set_trace()
        self.loss = None
        self.loss = instantiate(self.loss_conf, _convert_="all")
        # if self.loss_conf:
        #     self.loss = wrap_base_loss(instantiate(self.loss_conf, _convert_="all"))
            # self.loss = {
            #     key: wrap_base_loss(el)
            #     for (key, el) in instantiate(self.loss_conf, _convert_="all").items()
            # }
            # self.loss = nn.ModuleDict(self.loss)

        self.meters = {}
        self.best_meter_values = {}
        if self.meters_conf:
            self.meters = instantiate(self.meters_conf, _convert_="all")

        # Use standard Gradient Scaler for DDP
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)

        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.gradient_logger = instantiate(self.optim_conf.gradient_logger)

        logging.info("Finished setting up components: Model, loss, optim, meters etc.")


    def _setup_ddp_distributed_training(self, distributed_conf, device):
        # Simplified to only handle standard nn.Module with DDP
        assert isinstance(self.model, torch.nn.Module)

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

        if distributed_conf.comms_dtype is not None:  # noqa
            from torch.distributed.algorithms import ddp_comm_hooks

            amp_type = get_amp_type(distributed_conf.comms_dtype)
            if amp_type == torch.bfloat16:
                hook = ddp_comm_hooks.default_hooks.bf16_compress_hook
                logging.info("Enabling bfloat16 grad communication")
            else:
                hook = ddp_comm_hooks.default_hooks.fp16_compress_hook
                logging.info("Enabling fp16 grad communication")
            process_group = None
            self.model.register_comm_hook(process_group, hook)


    def _move_to_device(self):
        print(
            f"Moving components to device {self.device} and local rank {self.local_rank}."
        )
        self.model.to(self.device)

        if self.loss:
            copy_data_to_device(self.loss, self.device)
        if self.scaler:
            copy_data_to_device(self.scaler, self.device)
        for meter in self._get_meters().values():
            meter.set_sync_device(self.device)

        print(
            f"Done moving components to device {self.device} and local rank {self.local_rank}."
        )
