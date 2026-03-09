#!/usr/bin/env python3
"""
make_eval_montage.py

Create side-by-side comparison figures from evaluation videos.
For each task, extracts key frames from base/DPO/RWR/PPO rollouts
and arranges them into a single montage figure for the report.

Usage:
    python scripts/make_eval_montage.py \
        --eval_dir outputs/eval \
        --output_dir outputs/figures \
        --model_order base dpo rwr ppo \
        --n_frames 6
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np
from pathlib import Path


def extract_frames_from_video(video_path: str, n_frames: int = 6) -> list:
    """Extract n_frames evenly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Warning: cannot open {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Evenly spaced frame indices
    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def pick_best_video(video_dir: str, prefer_success: bool = True) -> str | None:
    """Pick a representative video from the directory.
    Prefers successful episodes if available."""
    if not os.path.isdir(video_dir):
        return None

    videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not videos:
        return None

    if prefer_success:
        success_videos = [v for v in videos if "_s1" in os.path.basename(v)]
        if success_videos:
            return success_videos[0]

    # Fall back to first video
    return videos[0]


def add_label(frame: np.ndarray, text: str, position="top",
              font_scale=0.7, thickness=2) -> np.ndarray:
    """Add text label to a frame."""
    frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    if position == "top":
        x = (frame.shape[1] - text_size[0]) // 2
        y = text_size[1] + 10
        # Background rectangle
        cv2.rectangle(frame, (x - 5, 2), (x + text_size[0] + 5, y + 8), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)
    elif position == "bottom":
        x = (frame.shape[1] - text_size[0]) // 2
        y = frame.shape[0] - 10
        cv2.rectangle(frame, (x - 5, y - text_size[1] - 5),
                      (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)

    return frame


def make_montage_for_task(
    task_name: str,
    eval_dir: str,
    model_labels: list[str],
    n_frames: int = 6,
    frame_height: int = 180,
) -> np.ndarray | None:
    """Create a montage for one task: rows=models, cols=time steps."""
    rows = []

    for label in model_labels:
        video_dir = os.path.join(eval_dir, label, task_name, "videos")
        video_path = pick_best_video(video_dir)

        if video_path is None:
            print(f"  No video for {label}/{task_name}")
            continue

        frames = extract_frames_from_video(video_path, n_frames)
        if not frames:
            print(f"  Could not extract frames from {video_path}")
            continue

        # Resize all frames to same height
        resized = []
        for f in frames:
            h, w = f.shape[:2]
            new_w = int(w * frame_height / h)
            resized.append(cv2.resize(f, (new_w, frame_height)))

        # Ensure all frames same width (use max)
        max_w = max(f.shape[1] for f in resized)
        padded = []
        for f in resized:
            if f.shape[1] < max_w:
                pad = np.zeros((frame_height, max_w - f.shape[1], 3), dtype=np.uint8)
                f = np.concatenate([f, pad], axis=1)
            padded.append(f)

        # Add model label to first frame
        padded[0] = add_label(padded[0], label.upper(), position="top")

        # Add time step labels
        for i, f in enumerate(padded):
            t_label = f"t={i}" if i < len(padded) - 1 else "final"
            padded[i] = add_label(f, t_label, position="bottom", font_scale=0.4, thickness=1)

        # Concatenate frames horizontally with 2px border
        border = np.ones((frame_height, 2, 3), dtype=np.uint8) * 128
        row_parts = []
        for i, f in enumerate(padded):
            if i > 0:
                row_parts.append(border)
            row_parts.append(f)
        row = np.concatenate(row_parts, axis=1)
        rows.append(row)

    if not rows:
        return None

    # Ensure all rows same width
    max_w = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        padded_rows.append(r)

    # Add horizontal border between rows
    h_border = np.ones((2, max_w, 3), dtype=np.uint8) * 128
    final_parts = []
    for i, r in enumerate(padded_rows):
        if i > 0:
            final_parts.append(h_border)
        final_parts.append(r)

    return np.concatenate(final_parts, axis=0)


def load_success_rates(eval_dir: str, model_labels: list[str]) -> dict:
    """Load success rates from eval_results.json files."""
    rates = {}
    for label in model_labels:
        label_dir = os.path.join(eval_dir, label)
        if not os.path.isdir(label_dir):
            continue
        rates[label] = {}
        for task_dir in sorted(os.listdir(label_dir)):
            results_path = os.path.join(label_dir, task_dir, "eval_results.json")
            if os.path.exists(results_path):
                with open(results_path) as f:
                    data = json.load(f)
                rates[label][task_dir] = data["summary"]["success_rate"]
    return rates


def make_summary_table(rates: dict, model_labels: list[str], output_path: str):
    """Create a summary table image with success rates."""
    # Collect all tasks
    all_tasks = set()
    for label_rates in rates.values():
        all_tasks.update(label_rates.keys())
    tasks = sorted(all_tasks)

    # Shorten task names for display
    short_names = []
    for t in tasks:
        name = t.replace("_GR1ArmsAndWaistFourierHands_Env", "")
        name = name.replace("PosttrainPnPNovelFrom", "")
        name = name.replace("SplitA", "")
        short_names.append(name)

    # Build table as image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cell_h, cell_w = 40, 200
    header_w = 250
    img_h = cell_h * (len(model_labels) + 1) + 20
    img_w = header_w + cell_w * len(tasks) + cell_w  # +1 for average column

    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    # Header row
    for j, name in enumerate(short_names):
        x = header_w + j * cell_w + 10
        cv2.putText(img, name[:20], (x, 30), font, 0.4, (0, 0, 0), 1)
    cv2.putText(img, "Average", (header_w + len(tasks) * cell_w + 10, 30),
                font, 0.45, (0, 0, 0), 1)

    # Data rows
    for i, label in enumerate(model_labels):
        y = cell_h * (i + 1) + 30
        cv2.putText(img, label.upper(), (10, y), font, 0.6, (0, 0, 0), 2)

        task_rates = []
        for j, task in enumerate(tasks):
            x = header_w + j * cell_w + 10
            rate = rates.get(label, {}).get(task, None)
            if rate is not None:
                pct = f"{rate*100:.0f}%"
                color = (0, 128, 0) if rate >= 0.5 else (0, 0, 200)
                task_rates.append(rate)
            else:
                pct = "N/A"
                color = (128, 128, 128)
            cv2.putText(img, pct, (x, y), font, 0.5, color, 1)

        # Average
        if task_rates:
            avg = np.mean(task_rates)
            x = header_w + len(tasks) * cell_w + 10
            color = (0, 128, 0) if avg >= 0.5 else (0, 0, 200)
            cv2.putText(img, f"{avg*100:.1f}%", (x, y), font, 0.5, color, 2)

    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Summary table saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create evaluation montage figures")
    parser.add_argument("--eval_dir", default="outputs/eval")
    parser.add_argument("--output_dir", default="outputs/figures")
    parser.add_argument("--model_order", nargs="+", default=["base", "dpo", "rwr", "ppo"])
    parser.add_argument("--n_frames", type=int, default=6)
    parser.add_argument("--frame_height", type=int, default=180)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover tasks from eval directory
    all_tasks = set()
    for label in args.model_order:
        label_dir = os.path.join(args.eval_dir, label)
        if os.path.isdir(label_dir):
            for task_dir in os.listdir(label_dir):
                if os.path.isdir(os.path.join(label_dir, task_dir)):
                    all_tasks.add(task_dir)

    tasks = sorted(all_tasks)
    print(f"Found {len(tasks)} tasks: {tasks}")
    print(f"Models: {args.model_order}")

    # Create per-task montages
    for task in tasks:
        print(f"\nCreating montage for: {task}")
        montage = make_montage_for_task(
            task, args.eval_dir, args.model_order,
            n_frames=args.n_frames, frame_height=args.frame_height,
        )
        if montage is not None:
            short_name = task.replace("_GR1ArmsAndWaistFourierHands_Env", "")
            out_path = os.path.join(args.output_dir, f"montage_{short_name}.png")
            cv2.imwrite(out_path, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {out_path}")
        else:
            print(f"  No videos found for {task}")

    # Create combined montage (all tasks stacked)
    print("\nCreating combined montage...")
    all_montages = []
    for task in tasks:
        montage = make_montage_for_task(
            task, args.eval_dir, args.model_order,
            n_frames=args.n_frames, frame_height=args.frame_height,
        )
        if montage is not None:
            # Add task title bar
            short_name = task.replace("_GR1ArmsAndWaistFourierHands_Env", "")
            short_name = short_name.replace("PosttrainPnPNovelFrom", "")
            title_bar = np.zeros((30, montage.shape[1], 3), dtype=np.uint8)
            cv2.putText(title_bar, short_name, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            all_montages.append(title_bar)
            all_montages.append(montage)
            # Separator
            sep = np.ones((4, montage.shape[1], 3), dtype=np.uint8) * 200
            all_montages.append(sep)

    if all_montages:
        # Ensure all same width
        max_w = max(m.shape[1] for m in all_montages)
        padded = []
        for m in all_montages:
            if m.shape[1] < max_w:
                pad = np.zeros((m.shape[0], max_w - m.shape[1], 3), dtype=np.uint8)
                m = np.concatenate([m, pad], axis=1)
            padded.append(m)

        combined = np.concatenate(padded, axis=0)
        out_path = os.path.join(args.output_dir, "combined_montage.png")
        cv2.imwrite(out_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"Combined montage saved: {out_path}")

    # Create success rate summary table
    rates = load_success_rates(args.eval_dir, args.model_order)
    if rates:
        table_path = os.path.join(args.output_dir, "success_rate_table.png")
        make_summary_table(rates, args.model_order, table_path)

    print("\nDone! Figures in:", args.output_dir)


if __name__ == "__main__":
    main()
