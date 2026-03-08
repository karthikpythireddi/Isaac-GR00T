#!/usr/bin/env python3
"""
Download RoboCasa GR1 tabletop task demo data from HuggingFace.

Downloads specific task datasets in LeRobot v2 format (parquet + mp4).

Usage:
  python scripts/download_robocasa_demos.py --task CuttingboardToBasket
  python scripts/download_robocasa_demos.py --task BottleToCabinetClose
  python scripts/download_robocasa_demos.py --all_recommended
"""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


# HuggingFace repo containing all GR1 tabletop task demos
HF_REPO = "nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim"

# Local base directory for downloaded data
BASE_DIR = Path("examples/robocasa-gr1-tabletop-tasks/gr1_finetune_data")

# Map short task names to HF subdirectory names
TASK_MAP = {
    # Tasks we recommend for preference pair collection (moderate base policy success)
    "CuttingboardToBasket": "gr1_arms_waist.CuttingboardToBasket",
    "BottleToCabinetClose": "gr1_arms_waist.BottleToCabinetClose",
    "PlateToBowl": "gr1_arms_waist.PlateToBowl",
    "TrayToPot": "gr1_arms_waist.TrayToPot",
    # Additional tasks if needed
    "CuttingboardToPan": "gr1_arms_waist.CuttingboardToPan",
    "CuttingboardToPot": "gr1_arms_waist.CuttingboardToPot",
    "PlacematToBasket": "gr1_arms_waist.PlacematToBasket",
    "PlacematToBowl": "gr1_arms_waist.PlacematToBowl",
    "TrayToPlate": "gr1_arms_waist.TrayToPlate",
    "TrayToCardboardbox": "gr1_arms_waist.TrayToCardboardbox",
}

# 4 recommended tasks for RLHF training
RECOMMENDED_TASKS = [
    "CuttingboardToBasket",  # 58.0% base success
    "BottleToCabinetClose",  # 51.5% base success
    "PlateToBowl",           # 57.0% base success
    "TrayToPot",             # 64.5% base success
]


def download_task(task_name: str, n_episodes: int = 100):
    """Download demo data for one task."""
    if task_name not in TASK_MAP:
        print(f"Unknown task: {task_name}")
        print(f"Available: {list(TASK_MAP.keys())}")
        return

    hf_subdir = TASK_MAP[task_name]
    local_dir = BASE_DIR / hf_subdir

    if local_dir.exists():
        # Check if data is already there
        parquet_dir = local_dir / "data" / "chunk-000"
        if parquet_dir.exists() and len(list(parquet_dir.glob("*.parquet"))) >= n_episodes:
            print(f"[skip] {task_name}: {local_dir} already has data")
            return

    print(f"[download] {task_name} -> {local_dir}")
    print(f"  HF repo: {HF_REPO}")
    print(f"  Subdir:  {hf_subdir}")

    # Download from HuggingFace
    # The dataset repo has subdirectories per task
    try:
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=[
                f"{hf_subdir}/meta/*",
                f"{hf_subdir}/data/chunk-000/*",
                f"{hf_subdir}/videos/chunk-000/**/*",
            ],
        )
        print(f"[done] {task_name} downloaded to {local_dir}")
    except Exception as e:
        # Try alternative: the data might be a standalone repo per task
        print(f"  [info] snapshot_download with allow_patterns failed: {e}")
        print(f"  [info] Trying direct download...")

        try:
            snapshot_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                local_dir=str(BASE_DIR),
                allow_patterns=[
                    f"{hf_subdir}/meta/*",
                    f"{hf_subdir}/data/chunk-000/*",
                    f"{hf_subdir}/videos/chunk-000/**/*",
                ],
            )
            print(f"[done] {task_name} downloaded to {local_dir}")
        except Exception as e2:
            print(f"  [error] Download failed: {e2}")
            print(f"  You may need to manually download from {HF_REPO}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="Task name to download")
    parser.add_argument("--all_recommended", action="store_true",
                        help="Download all 4 recommended tasks")
    parser.add_argument("--n_episodes", type=int, default=100,
                        help="Number of episodes to download per task")
    parser.add_argument("--list", action="store_true",
                        help="List available tasks")
    args = parser.parse_args()

    if args.list:
        print("Available tasks:")
        for name, hf_dir in TASK_MAP.items():
            marker = " [recommended]" if name in RECOMMENDED_TASKS else ""
            print(f"  {name}: {hf_dir}{marker}")
        return

    if args.all_recommended:
        for task in RECOMMENDED_TASKS:
            download_task(task, args.n_episodes)
    elif args.task:
        download_task(args.task, args.n_episodes)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
