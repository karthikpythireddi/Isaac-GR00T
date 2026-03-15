#!/usr/bin/env python3
"""
build_demo_preference_pairs.py

Builds preference pairs using RoboCasa expert demonstrations as WINNERS
and base GR00T N1.6 policy rollouts (failures) as LOSERS.

This gives clean, high-quality preference signal:
  - Winners: Expert demos (always successful, ground-truth actions)
  - Losers: Base policy rollouts that FAIL the task

Pipeline:
  1. Load demo episodes from LeRobot format (parquet + mp4)
  2. Convert demos to HDF5-compatible format (split state/action, decode video)
  3. Collect base policy rollouts via PolicyClient
  4. Pair each failed rollout with a random demo as (winner=demo, loser=rollout)
  5. Save as HDF5 matching existing preference data format

Prerequisites:
  1. Download demo data:
       python scripts/download_robocasa_demos.py --task CuttingboardToBasket

  2. Start GR00T inference server:
       conda activate groot
       python gr00t/eval/run_gr00t_server.py \
           --model-path nvidia/GR00T-N1.6-3B \
           --embodiment-tag GR1 \
           --use-sim-policy-wrapper --port 5555

  3. Run this script (in gr1_sim env):
       conda activate gr1_sim
       python scripts/build_demo_preference_pairs.py \
           --env_name "gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env" \
           --demo_data_dir examples/robocasa-gr1-tabletop-tasks/gr1_finetune_data/gr1_arms_waist.CuttingboardToBasket \
           --n_pairs 50 --output_dir preference_data/gr1_demo_pairs
"""

import argparse
import gc
import json
import os
import random

import cv2
import gymnasium as gym
import h5py
import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.policy.server_client import PolicyClient


# ── GR1 embodiment config (5 body parts, matches existing 400-pair data) ──

STATE_SLICES = {
    "left_arm":  (0, 7),
    "right_arm": (22, 29),
    "left_hand": (7, 13),
    "right_hand": (29, 35),
    "waist":     (41, 44),
}

ACTION_SLICES = STATE_SLICES  # same layout for actions

ACTION_HORIZON = 16  # GR00T predicts 16 future actions per query
N_ACTION_STEPS = 8   # env steps per policy query


# ── Demo loading ──────────────────────────────────────────────────────────

def load_demo_episode(data_dir: str, episode_idx: int) -> dict:
    """
    Load one demo episode from LeRobot format.

    Returns dict matching our HDF5 structure:
      obs:     {state.{key}: (T, D), video.ego_view: (T, 256, 256, 3)}
      actions: {action.{key}: (T_chunks, ACTION_HORIZON, D)}
      success: True  (demos are always successful)
      length:  T_chunks
    """
    chunk_idx = episode_idx // 1000
    parquet_path = os.path.join(
        data_dir, "data", f"chunk-{chunk_idx:03d}",
        f"episode_{episode_idx:06d}.parquet"
    )
    video_path = os.path.join(
        data_dir, "videos", f"chunk-{chunk_idx:03d}",
        "observation.images.ego_view",
        f"episode_{episode_idx:06d}.mp4"
    )

    # Read parquet
    df = pq.read_table(parquet_path).to_pandas()
    T = len(df)

    # Split flat state vector into component keys
    state_vec = np.stack(df["observation.state"].values).astype(np.float32)  # (T, 44)
    obs = {}
    for key, (start, end) in STATE_SLICES.items():
        obs[f"state.{key}"] = state_vec[:, start:end]  # (T, D)

    # Decode video frames from mp4
    frames = _decode_video(video_path, T)
    obs["video.ego_view"] = frames  # (T, 256, 256, 3) uint8

    # Split flat action vector and create action chunks
    action_vec = np.stack(df["action"].values).astype(np.float32)  # (T, 44)
    actions = {}
    for key, (start, end) in ACTION_SLICES.items():
        per_step = action_vec[:, start:end]  # (T, D)
        actions[f"action.{key}"] = _chunk_actions(per_step, ACTION_HORIZON)

    T_chunks = actions[f"action.{list(ACTION_SLICES.keys())[0]}"].shape[0]

    # Trim obs to match T_chunks (since we subsample every N_ACTION_STEPS)
    for k in obs:
        obs[k] = obs[k][:T_chunks]

    return {
        "obs": obs,
        "actions": actions,
        "success": True,
        "length": T_chunks,
        "cumulative_reward": 1.0,
    }


def _decode_video(video_path: str, expected_frames: int) -> np.ndarray:
    """Decode mp4 to numpy array (T, H, W, 3) uint8."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV reads BGR, convert to RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from {video_path}")

    arr = np.stack(frames)  # (T_video, H, W, 3)

    # Video may have different frame count than parquet rows due to fps
    # Subsample or pad to match expected_frames
    if len(arr) > expected_frames:
        arr = arr[:expected_frames]
    elif len(arr) < expected_frames:
        # Repeat last frame
        pad = np.repeat(arr[-1:], expected_frames - len(arr), axis=0)
        arr = np.concatenate([arr, pad], axis=0)

    return arr


def _chunk_actions(per_step: np.ndarray, horizon: int) -> np.ndarray:
    """
    Convert per-step actions (T, D) to action chunks (T_chunks, horizon, D).

    At each chunk timestep t (every N_ACTION_STEPS steps), take the next
    `horizon` actions. Pad with last action if near end of trajectory.
    """
    T, D = per_step.shape
    chunk_indices = list(range(0, T, N_ACTION_STEPS))
    chunks = []
    for t in chunk_indices:
        window = per_step[t:t + horizon]
        if len(window) < horizon:
            pad = np.repeat(window[-1:], horizon - len(window), axis=0)
            window = np.concatenate([window, pad], axis=0)
        chunks.append(window)
    return np.array(chunks)  # (T_chunks, horizon, D)


def get_available_episodes(data_dir: str) -> list:
    """Get list of available episode indices from episodes.jsonl."""
    episodes_path = os.path.join(data_dir, "meta", "episodes.jsonl")
    episodes = []
    with open(episodes_path) as f:
        for line in f:
            ep = json.loads(line)
            episodes.append(ep["episode_index"])
    return episodes


# ── Rollout collection (reused from collect_preferences_groot.py) ─────────

def make_env_fn(env_name: str, max_episode_steps: int):
    def _make():
        env = gym.make(env_name, enable_render=True)
        env = MultiStepWrapper(
            env,
            video_delta_indices=np.array([0]),
            state_delta_indices=np.array([0]),
            n_action_steps=N_ACTION_STEPS,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        )
        return env
    return _make


def rollout(vec_env, policy: PolicyClient, seed: int) -> dict:
    """Roll out the base GR00T policy. Returns dict with obs, actions, success, etc."""
    obs, _ = vec_env.reset(seed=[seed])
    policy.reset()

    obs_keys = [k for k in obs.keys() if not k.startswith("annotation")]
    traj_obs = {k: [] for k in obs_keys}
    traj_act = {}
    success = False
    length = 0
    cumulative_reward = 0.0

    while True:
        for k in obs_keys:
            traj_obs[k].append(obs[k][0, 0])

        policy_obs = dict(obs)
        # Remap padded video key back to base key expected by GR00T server
        # (MultiStepWrapper pads key to e.g. video.ego_view_pad_res256_freq20)
        for k in list(policy_obs.keys()):
            if k != "video.ego_view" and ("ego_view" in k or k.startswith("video.")):
                policy_obs["video.ego_view"] = policy_obs[k]
                break
        # Pad missing state keys for embodiments that expect them
        for mk, shape in [("state.left_leg", (1,1,6)),
                           ("state.right_leg", (1,1,6)),
                           ("state.neck", (1,1,3))]:
            if mk not in policy_obs:
                policy_obs[mk] = np.zeros(shape, dtype=np.float32)

        actions, _info = policy.get_action(policy_obs)

        if not traj_act:
            traj_act = {ak: [] for ak in actions}
        for ak in actions:
            traj_act[ak].append(actions[ak][0])

        obs, chunk_reward, done, _truncated, infos = vec_env.step(actions)
        length += 1
        cumulative_reward += float(np.asarray(chunk_reward).flat[0])

        ep_success = False
        if "success" in infos:
            s = np.asarray(infos["success"])
            if s.dtype == object:
                ep_success = any(bool(np.any(v)) for v in s.flat)
            else:
                ep_success = bool(s.any())
        elif "final_info" in infos and infos["final_info"] is not None:
            fi = infos["final_info"]
            if isinstance(fi, (list, tuple)) and len(fi) > 0 and fi[0] is not None:
                ep_success = bool(fi[0].get("success", False))

        if ep_success:
            success = True
            break
        if done:
            break

    for k in traj_obs:
        traj_obs[k] = np.array(traj_obs[k]) if traj_obs[k] else np.array([])
    for ak in traj_act:
        traj_act[ak] = np.array(traj_act[ak]) if traj_act[ak] else np.array([])

    return {
        "obs": traj_obs,
        "actions": traj_act,
        "success": success,
        "length": length,
        "cumulative_reward": cumulative_reward,
    }


# ── Preference pair building ─────────────────────────────────────────────

def collect_demo_vs_rollout_pairs(
    vec_env,
    policy: PolicyClient,
    demo_data_dir: str,
    n_pairs: int = 50,
    seed_offset: int = 0,
    max_attempts_factor: int = 5,
) -> list:
    """
    Collect (demo=winner, failed_rollout=loser) preference pairs.

    For each failed base policy rollout, pair it with a random expert demo.
    Successful rollouts are skipped (we only want failures as losers).
    """
    available_eps = get_available_episodes(demo_data_dir)
    print(f"  Available demo episodes: {len(available_eps)}")

    # Pre-load a batch of demos to avoid repeated I/O
    n_demos_to_load = min(len(available_eps), max(n_pairs, 20))
    demo_indices = random.sample(available_eps, n_demos_to_load)
    print(f"  Pre-loading {n_demos_to_load} demo episodes...", flush=True)

    demos = []
    for i, ep_idx in enumerate(demo_indices):
        try:
            demo = load_demo_episode(demo_data_dir, ep_idx)
            demos.append(demo)
            if (i + 1) % 10 == 0:
                print(f"    Loaded {i+1}/{n_demos_to_load} demos", flush=True)
        except Exception as e:
            print(f"    [warn] Failed to load episode {ep_idx}: {e}", flush=True)

    if len(demos) == 0:
        raise RuntimeError("No demo episodes could be loaded!")
    print(f"  Successfully loaded {len(demos)} demos", flush=True)

    pairs = []
    n_successes = 0
    n_failures = 0
    seed = seed_offset
    max_seeds = seed_offset + n_pairs * max_attempts_factor

    while len(pairs) < n_pairs and seed < max_seeds:
        print(
            f"  [rollout] seed={seed} — pair {len(pairs)+1}/{n_pairs} "
            f"(successes={n_successes}, failures={n_failures})",
            flush=True,
        )
        traj = rollout(vec_env, policy, seed=seed)
        print(
            f"    result: success={traj['success']}, "
            f"chunks={traj['length']}, reward={traj['cumulative_reward']:.3f}",
            flush=True,
        )
        seed += 1

        if traj["success"]:
            n_successes += 1
            print(f"    [skip] Rollout succeeded — not useful as loser", flush=True)
            continue

        n_failures += 1

        # Pick a random demo as the winner
        demo = random.choice(demos)
        pairs.append((demo, traj, "demo_vs_failure"))

        print(
            f"  pair {len(pairs)}/{n_pairs} | "
            f"type=demo_vs_failure | "
            f"demo_len={demo['length']} rollout_len={traj['length']}",
            flush=True,
        )

    print(
        f"\n  Collection complete: {len(pairs)} pairs "
        f"({n_failures} failures used, {n_successes} successes skipped)"
    )
    return pairs


# ── Save to HDF5 ─────────────────────────────────────────────────────────

def save_pairs_to_hdf5(pairs: list, output_path: str, task_name: str):
    """Save preference pairs to HDF5, matching existing format."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with h5py.File(output_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["n_pairs"] = len(pairs)
        meta.attrs["task_name"] = task_name
        meta.attrs["n_demo_vs_failure"] = len(pairs)
        meta.attrs["source"] = "demo_vs_base_policy_rollout"

        for i, (winner, loser, pref_type) in enumerate(pairs):
            grp = f.create_group(f"pair_{i}")
            grp.attrs["winner_length"] = winner["length"]
            grp.attrs["loser_length"] = loser["length"]
            grp.attrs["winner_success"] = winner["success"]
            grp.attrs["loser_success"] = loser["success"]
            grp.attrs["winner_cumulative_reward"] = winner["cumulative_reward"]
            grp.attrs["loser_cumulative_reward"] = loser["cumulative_reward"]
            grp.attrs["preference_type"] = pref_type

            for label, traj in [("winner", winner), ("loser", loser)]:
                t_grp = grp.create_group(label)
                act_grp = t_grp.create_group("actions")
                for ak, av in traj["actions"].items():
                    if av is not None and len(av) > 0:
                        act_grp.create_dataset(ak, data=av)

                obs_grp = t_grp.create_group("obs")
                for ok, ov in traj["obs"].items():
                    if ov is not None and len(ov) > 0:
                        obs_grp.create_dataset(ok, data=ov)

            if (i + 1) % 10 == 0:
                print(f"    Saved {i+1}/{len(pairs)} pairs...", flush=True)

    print(f"  Saved {len(pairs)} pairs -> {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build preference pairs: RoboCasa demos (winners) vs "
                    "base GR00T policy failures (losers)."
    )
    parser.add_argument("--env_name", required=True,
                        help="Gym env ID for rollout collection")
    parser.add_argument("--demo_data_dir", required=True,
                        help="Path to LeRobot-format demo data directory")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--output_dir", default="preference_data/gr1_demo_pairs")
    parser.add_argument("--n_pairs", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--seed_offset", type=int, default=0)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Demo-vs-Rollout Preference Pair Builder")
    print(f"{'='*60}")
    print(f"Task env     : {args.env_name}")
    print(f"Demo data    : {args.demo_data_dir}")
    print(f"Server       : {args.host}:{args.port}")
    print(f"n_pairs      : {args.n_pairs}")
    print(f"{'='*60}\n")

    vec_env = gym.vector.SyncVectorEnv(
        [make_env_fn(args.env_name, args.max_steps)]
    )
    policy = PolicyClient(host=args.host, port=args.port, strict=False)

    pairs = collect_demo_vs_rollout_pairs(
        vec_env, policy, args.demo_data_dir,
        n_pairs=args.n_pairs,
        seed_offset=args.seed_offset,
    )

    vec_env.close()
    gc.collect()

    if len(pairs) == 0:
        print("[warn] No pairs collected — base policy may succeed too often.")
        return

    task_name = args.env_name.split("/")[-1]
    output_path = os.path.join(args.output_dir, f"{task_name}_demo_preferences.hdf5")
    save_pairs_to_hdf5(pairs, output_path, task_name)

    print(f"\n[done] Output: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
