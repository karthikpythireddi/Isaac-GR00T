#!/usr/bin/env python3
"""
make_linkedin_video.py

Creates a high-quality LinkedIn demo video from GR00T eval rollouts.
For each task, plays Base / DPO / RWR side-by-side simultaneously,
picking the best available episode (SUCCESS preferred, else first).

Usage (run on Lightning AI after eval):
    python scripts/make_linkedin_video.py \
        --eval_dir outputs/eval \
        --output_dir outputs/linkedin_videos \
        --model_order base dpo rwr

    # Single task:
    python scripts/make_linkedin_video.py \
        --eval_dir outputs/eval \
        --task PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env \
        --output_dir outputs/linkedin_videos
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np
from pathlib import Path


# ── Display config ────────────────────────────────────────────────────────────

MODEL_DISPLAY = {
    "base": "Base  GR00T N1.6",
    "dpo":  "DPO",
    "rwr":  "RLHF / RWR",
    "ppo":  "PPO",
}
MODEL_COLORS_BGR = {
    "base": (176, 114,  76),   # steel blue
    "dpo":  ( 82, 132, 221),   # orange
    "rwr":  (104, 168,  85),   # green
    "ppo":  ( 82,  78, 196),   # red
}
TASK_SHORT = {
    "PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env":
        "Bottle → Cabinet",
    "PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env":
        "Cutting Board → Basket",
    "PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env":
        "Plate → Bowl",
    "PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env":
        "Tray → Pot",
}

PANEL_W    = 480    # width per model panel
HEADER_H   = 52     # per-model label bar height
TITLE_H    = 60     # top banner height
FOOTER_H   = 36     # bottom strip height
FPS_OUT    = 30


# ── Helpers ───────────────────────────────────────────────────────────────────

def pick_video(video_dir: str) -> tuple:
    """Return (path, is_success). Prefers SUCCESS episodes."""
    videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not videos:
        return None, False
    for v in videos:
        if "SUCCESS" in os.path.basename(v):
            return v, True
    return videos[0], "SUCCESS" in os.path.basename(videos[0])


def load_success_rate(eval_dir: str, model: str, task: str) -> int:
    fp = os.path.join(eval_dir, model, task, "eval_results.json")
    if os.path.isfile(fp):
        with open(fp) as f:
            d = json.load(f)
        sr = d.get("summary", {}).get("success_rate", 0)
        return int(round(sr * 100))
    return -1


def read_video_frames(path: str, target_w: int, target_h: int):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame)
    cap.release()
    return frames


def make_header(w: int, model: str, sr: int, is_success: bool) -> np.ndarray:
    bar = np.zeros((HEADER_H, w, 3), dtype=np.uint8)
    bar[:] = MODEL_COLORS_BGR[model]
    label = MODEL_DISPLAY.get(model, model)
    sr_str = f"{sr}%" if sr >= 0 else "N/A"
    icon   = "✓" if is_success else "✗"
    text   = f"{label}   {sr_str} avg success   {icon}"

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = (w - tw) // 2
    y = (HEADER_H + th) // 2 - 2
    cv2.putText(bar, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return bar


def make_title(total_w: int, task_label: str) -> np.ndarray:
    bar = np.zeros((TITLE_H, total_w, 3), dtype=np.uint8)
    bar[:] = (25, 25, 25)

    # Main title
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    title = f"GR00T Humanoid  |  {task_label}"
    (tw, th), _ = cv2.getTextSize(title, font, scale, thick)
    x = (total_w - tw) // 2
    cv2.putText(bar, title, (x, th + 6), font, scale, (240, 240, 240), thick, cv2.LINE_AA)

    # Subtitle
    sub = "Preference Optimization for Continuous Robotic Control  |  Stanford CS234"
    sfont, sscale, sthick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    (stw, _), _ = cv2.getTextSize(sub, sfont, sscale, sthick)
    sx = (total_w - stw) // 2
    cv2.putText(bar, sub, (sx, th + 30), sfont, sscale, (160, 160, 160), sthick, cv2.LINE_AA)
    return bar


def make_footer(total_w: int) -> np.ndarray:
    bar = np.zeros((FOOTER_H, total_w, 3), dtype=np.uint8)
    bar[:] = (18, 18, 18)
    text = "Karthik Pythireddi · Taylor Tam · Jonathan Lu   |   Stanford University"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = (total_w - tw) // 2
    y = (FOOTER_H + th) // 2
    cv2.putText(bar, text, (x, y), font, scale, (140, 140, 140), thick, cv2.LINE_AA)
    return bar


# ── Main per-task video builder ───────────────────────────────────────────────

def make_task_video(eval_dir: str, task: str, models: list, output_path: str):
    print(f"\n[{task}]")
    task_label = TASK_SHORT.get(task, task[:40])

    # Load frames for each model
    model_data = {}
    frame_h = None
    for model in models:
        video_dir = os.path.join(eval_dir, model, task, "videos")
        vpath, is_success = pick_video(video_dir)
        sr = load_success_rate(eval_dir, model, task)
        if vpath is None:
            print(f"  [{model}] no video found — using blank")
            model_data[model] = {"frames": [], "sr": sr, "success": False}
            continue
        print(f"  [{model}] {os.path.basename(vpath)}  sr={sr}%  success={is_success}")

        # Probe native size from first model that has a video
        if frame_h is None:
            cap = cv2.VideoCapture(vpath)
            native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            frame_h = int(native_h * PANEL_W / native_w)

        frames = read_video_frames(vpath, PANEL_W, frame_h)
        model_data[model] = {"frames": frames, "sr": sr, "success": is_success}

    if frame_h is None:
        print("  No videos found for any model — skipping task")
        return

    n_frames  = max(len(d["frames"]) for d in model_data.values()) or 1
    total_w   = PANEL_W * len(models)
    total_h   = TITLE_H + HEADER_H + frame_h + FOOTER_H

    title_bar  = make_title(total_w, task_label)
    footer_bar = make_footer(total_w)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, FPS_OUT, (total_w, total_h))

    blank = np.zeros((frame_h, PANEL_W, 3), dtype=np.uint8)

    for fi in range(n_frames):
        panels = []
        for model in models:
            d = model_data[model]
            frames = d["frames"]
            frame  = frames[min(fi, len(frames) - 1)] if frames else blank
            header = make_header(PANEL_W, model, d["sr"], d["success"])
            panels.append(np.vstack([header, frame]))

        row  = np.hstack(panels)
        full = np.vstack([title_bar, row, footer_bar])
        writer.write(full)

    writer.release()
    print(f"  Saved → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir",    default="outputs/eval")
    parser.add_argument("--output_dir",  default="outputs/linkedin_videos")
    parser.add_argument("--model_order", nargs="+", default=["base", "dpo", "rwr"])
    parser.add_argument("--task",        default=None,
                        help="Single task name. If omitted, all 4 tasks are generated.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.task:
        tasks = [args.task]
    else:
        tasks = list(TASK_SHORT.keys())

    for task in tasks:
        safe = task.replace("_GR1ArmsAndWaistFourierHands_Env", "")
        out  = os.path.join(args.output_dir, f"groot_{safe}.mp4")
        make_task_video(args.eval_dir, task, args.model_order, out)

    print(f"\nAll done. Videos in: {args.output_dir}")


if __name__ == "__main__":
    main()
