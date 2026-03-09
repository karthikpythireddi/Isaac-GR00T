#!/usr/bin/env python3
"""
Visualize preference pairs from HDF5 file.

Creates side-by-side video grids (winner vs loser) and frame montages.

Usage:
  python scripts/visualize_preference_pairs.py \
      --hdf5_path preference_data/gr1_demo_pairs/all_4tasks_demo_preferences.hdf5 \
      --output_dir visualizations \
      --n_pairs 4 \
      --mode video   # video | frames | both
"""

import argparse
import os

import h5py
import numpy as np


def get_video_key(group):
    """Find the ego_view video key (handles different naming conventions)."""
    for k in group["obs"].keys():
        if k.startswith("video.ego_view"):
            return k
    return None


def make_frame_montage(winner_frames, loser_frames, n_cols=6):
    """Create a montage showing sampled frames from winner and loser side by side."""
    # Sample evenly spaced frames
    w_indices = np.linspace(0, len(winner_frames) - 1, n_cols, dtype=int)
    l_indices = np.linspace(0, len(loser_frames) - 1, n_cols, dtype=int)

    w_sampled = winner_frames[w_indices]
    l_sampled = loser_frames[l_indices]

    h, w_px = w_sampled.shape[1], w_sampled.shape[2]

    # Add green border to winner, red border to loser
    border = 4
    def add_border(frames, color):
        bordered = np.full(
            (len(frames), h + 2 * border, w_px + 2 * border, 3),
            color, dtype=np.uint8
        )
        bordered[:, border:-border, border:-border, :] = frames
        return bordered

    w_bordered = add_border(w_sampled, [0, 200, 0])   # green
    l_bordered = add_border(l_sampled, [200, 0, 0])    # red

    # Concatenate horizontally for each row, then stack vertically
    w_row = np.concatenate(list(w_bordered), axis=1)
    l_row = np.concatenate(list(l_bordered), axis=1)

    # Add labels
    label_h = 30
    label_strip = np.ones((label_h, w_row.shape[1], 3), dtype=np.uint8) * 40
    separator = np.ones((8, w_row.shape[1], 3), dtype=np.uint8) * 128

    montage = np.concatenate([label_strip, w_row, separator, label_strip, l_row], axis=0)
    return montage


def make_side_by_side_video(winner_frames, loser_frames, fps=10):
    """Create a side-by-side video array (winner left, loser right)."""
    max_len = max(len(winner_frames), len(loser_frames))
    h, w_px = winner_frames.shape[1], winner_frames.shape[2]

    # Pad shorter trajectory by repeating last frame
    def pad_to(frames, target_len):
        if len(frames) >= target_len:
            return frames[:target_len]
        pad = np.repeat(frames[-1:], target_len - len(frames), axis=0)
        # Dim the repeated frames to indicate episode ended
        pad = (pad * 0.5).astype(np.uint8)
        return np.concatenate([frames, pad], axis=0)

    w_padded = pad_to(winner_frames, max_len)
    l_padded = pad_to(loser_frames, max_len)

    # Add colored top bar
    bar_h = 6
    w_bar = np.zeros((max_len, bar_h, w_px, 3), dtype=np.uint8)
    w_bar[:, :, :, 1] = 200  # green
    l_bar = np.zeros((max_len, bar_h, w_px, 3), dtype=np.uint8)
    l_bar[:, :, :, 0] = 200  # red

    w_with_bar = np.concatenate([w_bar, w_padded], axis=1)
    l_with_bar = np.concatenate([l_bar, l_padded], axis=1)

    # Separator
    sep = np.ones((max_len, h + bar_h, 4, 3), dtype=np.uint8) * 128

    combined = np.concatenate([w_with_bar, sep, l_with_bar], axis=2)
    return combined


def visualize_pairs(hdf5_path, output_dir, n_pairs=4, mode="both", pair_indices=None):
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        total = int(f["metadata"].attrs["n_pairs"])
        print(f"Total pairs: {total}")

        if pair_indices is None:
            # Pick evenly spaced pairs across different tasks
            pair_indices = np.linspace(0, total - 1, n_pairs, dtype=int).tolist()

        for idx in pair_indices:
            pair_key = f"pair_{idx}"
            if pair_key not in f:
                print(f"  {pair_key} not found, skipping")
                continue

            pair = f[pair_key]
            attrs = dict(pair.attrs)
            ptype = attrs.get("preference_type", "unknown")
            w_len = attrs.get("winner_length", "?")
            l_len = attrs.get("loser_length", "?")
            w_reward = attrs.get("winner_cumulative_reward", "?")
            l_reward = attrs.get("loser_cumulative_reward", "?")

            print(f"\n  {pair_key}: type={ptype}")
            print(f"    Winner: {w_len} steps, reward={w_reward}")
            print(f"    Loser:  {l_len} steps, reward={l_reward}")

            # Get video frames
            w_vid_key = get_video_key(pair["winner"])
            l_vid_key = get_video_key(pair["loser"])
            if not w_vid_key or not l_vid_key:
                print(f"    No video data found, skipping")
                continue

            winner_frames = pair["winner"]["obs"][w_vid_key][:]
            loser_frames = pair["loser"]["obs"][l_vid_key][:]
            print(f"    Winner frames: {winner_frames.shape}")
            print(f"    Loser frames:  {loser_frames.shape}")

            # Frame montage
            if mode in ("frames", "both"):
                montage = make_frame_montage(winner_frames, loser_frames)
                from PIL import Image
                img = Image.fromarray(montage)

                # Add text labels
                try:
                    from PIL import ImageDraw, ImageFont
                    draw = ImageDraw.Draw(img)
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                    except Exception:
                        font = ImageFont.load_default()
                    draw.text((10, 5), f"WINNER (demo, {w_len} steps, r={w_reward})", fill=(0, 255, 0), font=font)
                    montage_h = montage.shape[0]
                    mid_y = winner_frames.shape[1] + 4 + 8 + 30  # after winner row + sep
                    draw.text((10, mid_y + 5), f"LOSER (rollout, {l_len} steps, r={l_reward})", fill=(255, 80, 80), font=font)
                except Exception:
                    pass

                out_path = os.path.join(output_dir, f"pair_{idx}_montage.png")
                img.save(out_path)
                print(f"    Saved: {out_path}")

            # Side-by-side video
            if mode in ("video", "both"):
                import cv2
                combined = make_side_by_side_video(winner_frames, loser_frames)
                out_path = os.path.join(output_dir, f"pair_{idx}_comparison.mp4")
                h_vid, w_vid = combined.shape[1], combined.shape[2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, 10, (w_vid, h_vid))
                for frame in combined:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                writer.release()
                print(f"    Saved: {out_path}")

    print(f"\nAll visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_path", default="preference_data/gr1_demo_pairs/all_4tasks_demo_preferences.hdf5")
    parser.add_argument("--output_dir", default="visualizations")
    parser.add_argument("--n_pairs", type=int, default=4)
    parser.add_argument("--mode", choices=["video", "frames", "both"], default="both")
    parser.add_argument("--pair_indices", type=int, nargs="+", default=None,
                        help="Specific pair indices to visualize")
    args = parser.parse_args()
    visualize_pairs(args.hdf5_path, args.output_dir, args.n_pairs, args.mode, args.pair_indices)


if __name__ == "__main__":
    main()
