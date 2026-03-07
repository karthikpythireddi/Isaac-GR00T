"""
Preference dataset for GR00T DPO/RWR training.

Loads (winner, loser) trajectory pairs from HDF5 files produced by
collect_preferences_groot.py and converts them to GR00T model inputs.
"""

import random
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


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


def make_preference_collator(
    embodiment_tag: str,
    action_keys: list,
    state_keys: list,
    video_keys: list,
):
    """
    Returns a collate_fn that converts raw preference samples into
    GR00T model input dicts with keys:
        video.{key}  : (B, 1, H, W, 3) uint8
        state.{key}  : (B, D) float32
        action.{key} : (B, n_action_steps, D) float32
        annotation.human.coarse_action : list[str]
        embodiment_tag : str
    """

    def _build_inputs(obs_step: dict, act_step: dict) -> dict:
        inp = {
            "embodiment_tag": embodiment_tag,
            "annotation.human.coarse_action": "perform the task",
        }
        for k in video_keys:
            if k in obs_step:
                inp[f"video.{k}"] = obs_step[k]
        for k in state_keys:
            if k in obs_step:
                inp[f"state.{k}"] = obs_step[k]
        for k in action_keys:
            if k in act_step:
                inp[f"action.{k}"] = act_step[k]
        return inp

    def _stack(inputs_list: list) -> dict:
        stacked = {}
        for key in inputs_list[0]:
            if key in ("embodiment_tag", "annotation.human.coarse_action"):
                stacked[key] = [s[key] for s in inputs_list]
            else:
                vals = [s[key] for s in inputs_list if key in s]
                if vals:
                    stacked[key] = torch.from_numpy(np.stack(vals, axis=0))
        return stacked

    def collate_fn(batch: list) -> dict:
        winner_inputs, loser_inputs = [], []
        for sample in batch:
            for traj_type, inp_list in [("winner", winner_inputs), ("loser", loser_inputs)]:
                obs_step, act_step = _sample_window(
                    sample[f"{traj_type}_obs"], sample[f"{traj_type}_actions"]
                )
                inp_list.append(_build_inputs(obs_step, act_step))
        return {"winner": _stack(winner_inputs), "loser": _stack(loser_inputs)}

    return collate_fn
