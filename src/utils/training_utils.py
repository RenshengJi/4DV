"""
Training utility functions.
"""
import builtins
import datetime
import os
import shutil
from collections import OrderedDict
from accelerate import Accelerator


def strip_module(state_dict):
    """
    Remove 'module.' prefix from state_dict keys.

    Args:
        state_dict: Original state_dict with possible 'module.' prefixes

    Returns:
        OrderedDict: New state_dict with 'module.' prefixes removed
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def setup_for_distributed(accelerator: Accelerator):
    """
    Disable printing when not in master process.

    Args:
        accelerator: Accelerator instance
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (accelerator.num_processes > 8)
        if accelerator.is_main_process or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")
            builtin_print(*args, **kwargs)

    builtins.print = print


def save_current_code(outdir):
    """
    Save current codebase snapshot.

    Args:
        outdir: Output directory for code snapshot

    Returns:
        str: Path to saved code directory
    """
    now = datetime.datetime.now()
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
