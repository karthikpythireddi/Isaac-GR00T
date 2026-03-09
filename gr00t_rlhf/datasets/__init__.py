"""
Preference dataset for GR00T DPO/RWR training.

Loads (winner, loser) trajectory pairs from HDF5 files produced by
collect_preferences_groot.py and converts them to GR00T model inputs
using the model's own AutoProcessor for proper VLM processing.
"""

import random
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import VLAStepData


class GR00TPreferenceDataset(Dataset):
    """
    Loads preference pairs from an HDF5 file.

    Each __getitem__ returns:
        winner_obs      : {obs_key: np.ndarray}
        winner_actions  : {action_key: np.ndarray (T_chunks, n_action_steps, D)}
        loser_obs       : same
        loser_actions   : same
        preference_type : str
    """

    def __init__(self, hdf5_path: str, n_windows_per_pair: int = 5):
        self.hdf5_path = hdf5_path
        self.n_windows_per_pair = n_windows_per_pair
        self.f = h5py.File(hdf5_path, "r")
        self.n_pairs = int(self.f["metadata"].attrs["n_pairs"])

    def __len__(self):
        return self.n_pairs * self.n_windows_per_pair

    def _load_traj(self, traj_grp: h5py.Group) -> dict:
        actions = {k: traj_grp["actions"][k][:] for k in traj_grp["actions"]}
        obs = {k: traj_grp["obs"][k][:] for k in traj_grp["obs"]}
        return {"actions": actions, "obs": obs}

    def __getitem__(self, idx: int) -> dict:
        pair_idx = idx // self.n_windows_per_pair
        grp = self.f[f"pair_{pair_idx}"]
        winner = self._load_traj(grp["winner"])
        loser = self._load_traj(grp["loser"])
        return {
            "winner_obs": winner["obs"],
            "winner_actions": winner["actions"],
            "loser_obs": loser["obs"],
            "loser_actions": loser["actions"],
            "preference_type": str(grp.attrs["preference_type"]),
        }

    def __del__(self):
        try:
            self.f.close()
        except Exception:
            pass


def _sample_window(obs: dict, actions: dict) -> tuple[dict, dict]:
    """Pick one random timestep and return obs frame + action chunk."""
    T = min(v.shape[0] for v in obs.values())
    T_act = min(v.shape[0] for v in actions.values())
    t = random.randint(0, min(T, T_act) - 1)

    obs_step = {}
    for k, v in obs.items():
        frame = v[t]
        obs_step[k] = frame[None].astype(np.uint8) if frame.ndim == 3 else frame.astype(np.float32)

    act_step = {k: v[t].astype(np.float32) for k, v in actions.items()}
    return obs_step, act_step


def _build_vla_step(obs_step: dict, act_step: dict, state_keys: list,
                    action_keys: list, video_keys: list) -> VLAStepData:
    """Convert a sampled window into VLAStepData for the GR00T processor."""
    # Build images dict: view_name -> list[np.ndarray (H, W, 3)]
    # Use the full obs key (minus "video." prefix) as dict key so it matches
    # the processor's modality_configs (e.g. "ego_view_bg_crop_pad_res256_freq20")
    images = {}
    for k in video_keys:
        for ok, ov in obs_step.items():
            if ok == k or ok == f"video.{k}" or ok.startswith(f"video.{k}"):
                # Derive the actual view name from the obs key
                view_name = ok.removeprefix("video.") if ok.startswith("video.") else ok
                if ov.ndim == 4:
                    images[view_name] = [ov[i] for i in range(ov.shape[0])]
                else:
                    images[view_name] = [ov]
                break

    # Build states dict: state_name -> np.ndarray (T, D)
    # Processor does state[key][-1] to get reference state, so need (T, D) not (D,)
    states = {}
    for k in state_keys:
        key_with_prefix = f"state.{k}"
        val = None
        if key_with_prefix in obs_step:
            val = obs_step[key_with_prefix].astype(np.float64)
        elif k in obs_step:
            val = obs_step[k].astype(np.float64)
        if val is not None:
            val = np.atleast_1d(val)
            if val.ndim == 1:
                val = val[np.newaxis, :]  # (D,) -> (1, D)
            states[k] = val

    # Build actions dict: action_name -> np.ndarray (horizon, D)
    actions = {}
    for k in action_keys:
        key_with_prefix = f"action.{k}"
        if key_with_prefix in act_step:
            actions[k] = np.atleast_2d(act_step[key_with_prefix].astype(np.float64))
        elif k in act_step:
            actions[k] = np.atleast_2d(act_step[k].astype(np.float64))

    return VLAStepData(
        images=images,
        states=states,
        actions=actions,
        text="perform the task",
        embodiment=EmbodimentTag.GR1,
    )


def make_preference_collator(
    embodiment_tag: str,
    action_keys: list,
    state_keys: list,
    video_keys: list,
    processor=None,
):
    """
    Returns a collate_fn that converts raw preference samples into
    GR00T model input dicts using the model's AutoProcessor.

    If processor is provided, uses it for proper VLM processing.
    Otherwise falls back to simple tensor stacking (for testing only).
    """
    if processor is None:
        raise ValueError(
            "processor is required. Load it with: "
            "AutoProcessor.from_pretrained(model_path)"
        )

    # Get the data collator from the processor
    data_collator = processor.collator

    # Build mapping from bare video key (e.g. "ego_view") to the processor's
    # full modality key (e.g. "ego_view_bg_crop_pad_res256_freq20")
    proc_video_keys = processor.modality_configs[embodiment_tag]["video"].modality_keys
    video_key_map = {}
    for bare in video_keys:
        for full in proc_video_keys:
            if full == bare or full.startswith(bare):
                video_key_map[bare] = full
                break

    def _process_sample(obs_step, act_step):
        """Process a single sample through the GR00T processor."""
        vla_step = _build_vla_step(obs_step, act_step, state_keys, action_keys, video_keys)
        # Remap image keys to match processor's expected modality keys
        remapped_images = {}
        for k, v in vla_step.images.items():
            remapped_images[video_key_map.get(k, k)] = v
        vla_step = VLAStepData(
            images=remapped_images,
            states=vla_step.states,
            actions=vla_step.actions,
            text=vla_step.text,
            embodiment=vla_step.embodiment,
        )
        messages = [{"type": "episode_step", "content": vla_step}]
        processed = processor(messages)
        return processed

    def collate_fn(batch: list) -> dict:
        winner_processed, loser_processed = [], []
        for sample in batch:
            for traj_type, proc_list in [("winner", winner_processed), ("loser", loser_processed)]:
                obs_step, act_step = _sample_window(
                    sample[f"{traj_type}_obs"], sample[f"{traj_type}_actions"]
                )
                processed = _process_sample(obs_step, act_step)
                proc_list.append(processed)

        # Use the model's own collator to batch the processed samples
        winner_batch = data_collator(winner_processed)["inputs"]
        loser_batch = data_collator(loser_processed)["inputs"]

        return {"winner": winner_batch, "loser": loser_batch}

    return collate_fn
