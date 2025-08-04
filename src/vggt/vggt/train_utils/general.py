import torch
import torch.nn as nn
import os
import math
import random
import numpy as np
from typing import Union, Optional
import logging
from iopath.common.file_io import g_pathmgr
import torch.distributed as dist
from pathlib import Path
from typing import Dict, Iterable, List

def safe_makedirs(path: str):
    if not path:
        logging.warning("safe_makedirs called with an empty path. No operation performed.")
        return False

    try:
        os.makedirs(path, exist_ok=True)
        logging.info(f"Directory '{path}' now exists.")
        return True
    except OSError as e:
        logging.error(f"Failed to create directory '{path}'. Reason: {e}")
        raise
    except Exception as e:
        # Catch any other unexpected errors.
        logging.error(f"An unexpected error occurred while creating directory '{path}'. Reason: {e}")
        raise



def set_seeds(seed_value, max_epochs, dist_rank):
    """
    Set the python random, numpy and torch seed for each gpu. Also set the CUDA
    seeds if the CUDA is available. This ensures deterministic nature of the training.
    """
    seed_value = (seed_value + dist_rank) * max_epochs
    logging.info(f"GPU SEED: {seed_value}")
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # for multi-GPU




def log_env_variables():
    env_keys = sorted(list(os.environ.keys()))
    st = ""
    for k in env_keys:
        v = os.environ[k]
        st += f"{k}={v}\n"
    logging.info("Logging ENV_VARIABLES")
    logging.info(st)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True



class AverageMeter:
    """Computes and stores the average and current value.
    Args:
        name (str): Name of the metric being tracked
        device (torch.device, optional): Device for tensor operations. Defaults to None.
        fmt (str): Format string for displaying values. Defaults to ":f"
    """

    def __init__(self, name: str, device: Optional[torch.device] = None, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.device = device
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self._allow_updates = True

    def update(self, val, n=1):
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        """String representation showing current and average values."""
        fmtstr = "{name}: {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    @property
    def value(self) -> float:
        """Get the current value."""
        return self.val

    @property
    def average(self) -> float:
        """Get the running average."""
        return self.avg

#################


_UNITS = ('', ' K', ' M', ' B', ' T')          # U+202F = thin-space for nicer look

def pretty_int(n: int) -> str:
    """Abbreviate a non-negative integer (0 → 0, 12_345 → '12.3 K')."""
    assert n >= 0, 'pretty_int() expects a non-negative int'
    if n < 1_000:
        return f'{n:,}'
    exp = int(math.log10(n) // 3)        # group of 3 digits
    exp = min(exp, len(_UNITS) - 1)      # cap at trillions
    value = n / 10 ** (3 * exp)
    return f'{value:.1f}'.rstrip('0').rstrip('.') + _UNITS[exp]


def model_summary(model: torch.nn.Module,
                  *,
                  log_file = None,
                  prefix: str = '') -> None:
    """
    Print / save a compact parameter summary.

    Args
    ----
    model      : The PyTorch nn.Module to inspect.
    log_file   : Optional path – if given, the full `str(model)` and per-parameter
                 lists are written there (three separate *.txt files).
    prefix     : Optional string printed at the beginning of every log line
                 (handy when several models share the same stdout).
    """
    if get_rank():          # only rank-0 prints
        return

    # --- counts -------------------------------------------------------------
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    frozen    = total - trainable

    print(prefix + '='*60)
    print(prefix + f'Model type : {model.__class__.__name__}')
    print(prefix + f'Total      : {pretty_int(total)} parameters')
    print(prefix + f'  trainable: {pretty_int(trainable)}')
    print(prefix + f'  frozen   : {pretty_int(frozen)}')
    print(prefix + '='*60)

    # --- optional file dump -------------------------------------------------
    if log_file is None:
        return

    log_file = Path(log_file)
    log_file.write_text(str(model))                      # full architecture

    # two extra detailed lists
    def _dump(names: Iterable[str], fname: str):
        """Write a formatted per-parameter list to *log_file.with_name(fname)*."""
        with open(log_file.with_name(fname), 'w') as f:
            for n in names:
                p = dict(model.named_parameters())[n]
                shape = str(tuple(p.shape))
                f.write(f'{n:<60s} {shape:<20} {p.numel()}\n')

    named = dict(model.named_parameters())
    _dump([n for n,p in named.items() if p.requires_grad],  'trainable.txt')
    _dump([n for n,p in named.items() if not p.requires_grad], 'frozen.txt')


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()
