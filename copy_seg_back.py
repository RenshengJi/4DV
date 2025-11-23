#!/usr/bin/env python3
"""
Script to copy all X_seg.png files from data/waymo/seg back to data/waymo/train_full
while preserving the first-level subdirectory structure.
Uses multiprocessing for faster copying.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

def copy_single_file(args):
    """
    Copy a single file to the target location.

    Args:
        args: Tuple of (seg_file_path, target_dir_path)

    Returns:
        Tuple of (success: bool, error_msg: str or None)
    """
    seg_file, target_path = args

    try:
        # Get the first-level subdirectory name
        subdir_name = seg_file.parent.name

        # Create target subdirectory
        target_subdir = target_path / subdir_name
        target_subdir.mkdir(parents=True, exist_ok=True)

        # Target file path
        target_file = target_subdir / seg_file.name

        # Copy the file
        shutil.copy2(str(seg_file), str(target_file))
        return (True, None)
    except Exception as e:
        return (False, f"Error copying {seg_file}: {e}")


def copy_seg_files(source_dir, target_dir, num_workers=None):
    """
    Copy all *_seg.png files from source_dir to target_dir,
    preserving the first-level subdirectory structure.
    Uses multiprocessing for faster copying.

    Args:
        source_dir: Source directory (e.g., data/waymo/seg)
        target_dir: Target directory (e.g., data/waymo/train_full)
        num_workers: Number of worker processes (default: cpu_count())
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Check if source directory exists
    if not source_path.exists():
        print(f"Error: Source directory {source_dir} does not exist!")
        return

    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)

    # Find all *_seg.png files
    seg_files = list(source_path.glob("*/*_seg.png"))

    if not seg_files:
        print(f"No *_seg.png files found in {source_dir}")
        return

    print(f"Found {len(seg_files)} seg files to copy")

    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), 32)  # Cap at 32 to avoid too many processes

    print(f"Using {num_workers} worker processes")

    # Prepare arguments for multiprocessing
    copy_args = [(seg_file, target_path) for seg_file in seg_files]

    # Process files in parallel
    copied_count = 0
    skipped_count = 0
    errors = []

    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better performance with progress bar
        results = list(tqdm(
            pool.imap_unordered(copy_single_file, copy_args, chunksize=100),
            total=len(seg_files),
            desc="Copying seg files"
        ))

    # Count results
    for success, error_msg in results:
        if success:
            copied_count += 1
        else:
            skipped_count += 1
            if error_msg:
                errors.append(error_msg)

    print(f"\nCompleted!")
    print(f"Copied: {copied_count} files")
    if skipped_count > 0:
        print(f"Skipped: {skipped_count} files")
        if errors:
            print("\nFirst 10 errors:")
            for error in errors[:10]:
                print(f"  {error}")

if __name__ == "__main__":
    source_dir = "data/waymo/seg"
    target_dir = "data/waymo/train_full"

    print(f"Copying *_seg.png files from {source_dir} to {target_dir}")
    print(f"Preserving first-level subdirectory structure\n")

    copy_seg_files(source_dir, target_dir)
