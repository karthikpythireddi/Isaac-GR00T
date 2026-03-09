#!/usr/bin/env python3
"""
eval_policy.py

Evaluate a GR00T policy on RoboCasa GR1 tabletop tasks via client-server.
Reports success rate, average reward, and episode lengths.

Prerequisites:
  GR00T server running:
    python gr00t/eval/run_gr00t_server.py \
        --model-path <model_path> \
        --embodiment-tag GR1 \
        --use-sim-policy-wrapper --port 5555

Usage:
  python scripts/eval_policy.py \
      --env_name "gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env" \
      --host localhost --port 5555 \
      --n_episodes 20 \
      --output_dir outputs/eval/dpo/CuttingboardToBasket
"""

import argparse
import glob
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.policy.server_client import PolicyClient


N_ACTION_STEPS = 8


def make_env_fn(env_name: str, max_episode_steps: int, video_dir: str = None):
    def _make():
        env = gym.make(env_name, enable_render=True)
        if video_dir is not None:
            from gr00t.eval.sim.wrapper.video_recording_wrapper import (
                VideoRecorder,
                VideoRecordingWrapper,
            )
            video_recorder = VideoRecorder.create_h264(fps=30, crf=22)
            env = VideoRecordingWrapper(
                env,
                video_recorder,
                video_dir=Path(video_dir),
                steps_per_render=1,
                max_episode_steps=max_episode_steps,
                overlay_text=False,
            )
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


def rename_latest_video(video_dir: str, ep: int, success: bool, known_files: set):
    """Rename the latest video file to include episode number and outcome."""
    if video_dir is None:
        return
    current_files = set(glob.glob(os.path.join(video_dir, "*.mp4")))
    new_files = current_files - known_files
    for fpath in new_files:
        basename = os.path.basename(fpath)
        status = "SUCCESS" if success else "FAIL"
        new_name = f"ep{ep:02d}_{status}_{basename}"
        new_path = os.path.join(video_dir, new_name)
        os.rename(fpath, new_path)
        known_files.add(new_path)


def evaluate(vec_env, policy: PolicyClient, n_episodes: int, max_steps: int,
             video_dir: str = None):
    """Run n_episodes and collect success/reward/length stats."""
    results = []
    known_files = set(glob.glob(os.path.join(video_dir, "*.mp4"))) if video_dir else set()

    for ep in range(n_episodes):
        seed = ep * 7 + 42  # deterministic but spread out
        obs, _ = vec_env.reset(seed=[seed])
        policy.reset()

        success = False
        length = 0
        cumulative_reward = 0.0

        while True:
            policy_obs = dict(obs)
            # Pad missing state keys
            for mk, shape in [("state.left_leg", (1,1,6)),
                               ("state.right_leg", (1,1,6)),
                               ("state.neck", (1,1,3))]:
                if mk not in policy_obs:
                    policy_obs[mk] = np.zeros(shape, dtype=np.float32)

            actions, _ = policy.get_action(policy_obs)
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

        results.append({
            "episode": ep,
            "seed": seed,
            "success": success,
            "length": length,
            "cumulative_reward": cumulative_reward,
        })

        # Rename video to include episode number and outcome
        rename_latest_video(video_dir, ep, success, known_files)

        status = "SUCCESS" if success else "FAIL"
        print(
            f"  Episode {ep+1}/{n_episodes}: {status} | "
            f"steps={length} | reward={cumulative_reward:.3f}",
            flush=True,
        )

    return results


def summarize(results: list, task_name: str) -> dict:
    """Compute summary statistics."""
    n = len(results)
    n_success = sum(1 for r in results if r["success"])
    avg_reward = np.mean([r["cumulative_reward"] for r in results])
    avg_length = np.mean([r["length"] for r in results])
    success_lengths = [r["length"] for r in results if r["success"]]
    avg_success_length = np.mean(success_lengths) if success_lengths else 0

    summary = {
        "task_name": task_name,
        "n_episodes": n,
        "n_success": n_success,
        "success_rate": n_success / n if n > 0 else 0,
        "avg_reward": float(avg_reward),
        "avg_length": float(avg_length),
        "avg_success_length": float(avg_success_length),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate GR00T policy")
    parser.add_argument("--env_name", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--output_dir", default="outputs/eval")
    parser.add_argument("--record_video", action="store_true",
                        help="Record rollout videos for each episode")
    args = parser.parse_args()

    task_name = args.env_name.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"Evaluating: {task_name}")
    print(f"Server: {args.host}:{args.port}")
    print(f"Episodes: {args.n_episodes}")
    print(f"{'='*60}\n")

    video_dir = None
    if args.record_video:
        video_dir = os.path.join(args.output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        print(f"Recording videos to: {video_dir}")

    vec_env = gym.vector.SyncVectorEnv(
        [make_env_fn(args.env_name, args.max_steps, video_dir=video_dir)]
    )
    policy = PolicyClient(host=args.host, port=args.port, strict=False)

    results = evaluate(vec_env, policy, args.n_episodes, args.max_steps,
                       video_dir=video_dir)
    vec_env.close()

    summary = summarize(results, task_name)

    print(f"\n{'='*60}")
    print(f"Results: {task_name}")
    print(f"  Success rate: {summary['n_success']}/{summary['n_episodes']} "
          f"({summary['success_rate']*100:.1f}%)")
    print(f"  Avg reward:   {summary['avg_reward']:.3f}")
    print(f"  Avg length:   {summary['avg_length']:.1f}")
    print(f"{'='*60}\n")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2)
    print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()
