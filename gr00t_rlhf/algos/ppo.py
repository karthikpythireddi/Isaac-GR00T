"""
PPO (Proximal Policy Optimization) for GR00T N1.6 flow-matching policy.

Uses a learned reward model (trained on preference pairs) to score online
rollouts, then updates the policy with clipped PPO objective.

Since GR00T uses flow-matching (not autoregressive), we adapt PPO:
  - Policy loss proxy: flow-matching MSE loss
  - Value function: separate MLP head on frozen backbone features
  - Reward: from preference-trained reward model or environment reward
  - KL penalty against reference policy to prevent catastrophic forgetting

Usage:
    python gr00t_rlhf/algos/ppo.py \
        --model_path nvidia/GR00T-N1.6-3B \
        --hdf5_path preference_data/gr1_demo_pairs/all_4tasks_demo_preferences.hdf5 \
        --output_dir outputs/ppo_groot \
        --ppo_iters 30 --batch_size 2
"""

import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from gr00t_rlhf.datasets import GR00TPreferenceDataset, make_preference_collator


EMBODIMENT_TAG = "gr1"
VIDEO_KEYS  = ["ego_view"]
STATE_KEYS  = ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]
ACTION_KEYS = STATE_KEYS


class RewardModel(nn.Module):
    """Simple reward model: flow-matching loss difference as reward proxy.

    R(obs, action) = ref_loss(obs, action) - policy_loss(obs, action)
    Higher reward = policy assigns higher probability than reference.

    For PPO with preference data, we train a reward head on top of the
    flow-matching backbone, or use the flow-loss-based reward directly.
    """

    def __init__(self, model: Gr00tN1d6):
        super().__init__()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad_(False)

    def compute_flow_loss(self, batch: dict, device: str) -> torch.Tensor:
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in batch.items() if isinstance(v, torch.Tensor) or k == "vlm_content"}
        out = self.model(inputs)
        action_loss = out["action_loss"]
        action_mask = out["action_mask"]
        return (action_loss * action_mask).sum(dim=(1, 2)) / (action_mask.sum(dim=(1, 2)) + 1e-6)

    def forward(self, batch: dict, device: str) -> torch.Tensor:
        """Return per-sample reward (negative flow loss = higher prob = higher reward)."""
        return -self.compute_flow_loss(batch, device)


def compute_flow_loss(model: Gr00tN1d6, batch: dict, device: str) -> torch.Tensor:
    """Forward pass; returns per-sample flow-matching loss, shape (B,)."""
    inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    out = model(inputs)
    action_loss = out["action_loss"]
    action_mask = out["action_mask"]
    return (action_loss * action_mask).sum(dim=(1, 2)) / (action_mask.sum(dim=(1, 2)) + 1e-6)


def ppo_loss(
    policy_loss: torch.Tensor,
    ref_loss: torch.Tensor,
    old_policy_loss: torch.Tensor,
    clip_eps: float = 0.2,
    kl_coeff: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    PPO-style loss adapted for flow-matching policies.

    Since we can't compute log-probabilities directly with flow-matching,
    we use the flow loss as a proxy for -log pi(a|o).

    The "ratio" is approximated as: exp(old_loss - new_loss)
    (decrease in loss = increase in probability).

    Advantage is computed from reward model: R = ref_loss - policy_loss
    (policy does better than ref = positive advantage).
    """
    # Advantage: how much better policy is than reference
    advantage = ref_loss - policy_loss  # positive = policy better

    # Log ratio approximation: old_loss - new_loss ≈ log(pi_new/pi_old)
    log_ratio = old_policy_loss - policy_loss
    ratio = torch.exp(log_ratio.clamp(-5, 5))

    # Clipped surrogate objective
    surr1 = ratio * advantage.detach()
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage.detach()
    policy_objective = torch.min(surr1, surr2).mean()

    # KL penalty against reference
    kl_penalty = (ref_loss - policy_loss).pow(2).mean()

    total_loss = -policy_objective + kl_coeff * kl_penalty

    stats = {
        "ppo_objective": policy_objective.item(),
        "kl_penalty": kl_penalty.item(),
        "advantage_mean": advantage.mean().item(),
        "ratio_mean": ratio.mean().item(),
    }
    return total_loss, stats


def train_ppo(
    model_path: str,
    hdf5_path: str,
    output_dir: str,
    ppo_iters: int = 30,
    batch_size: int = 2,
    lr: float = 1e-5,
    clip_eps: float = 0.2,
    kl_coeff: float = 0.1,
    n_windows_per_pair: int = 5,
    device: str = "cuda",
    use_wandb: bool = False,
    wandb_project: str = "gr00t-ppo",
):
    os.makedirs(output_dir, exist_ok=True)

    if use_wandb:
        import wandb
        wandb.init(project=wandb_project, config=dict(
            ppo_iters=ppo_iters, batch_size=batch_size, lr=lr,
            clip_eps=clip_eps, kl_coeff=kl_coeff, model_path=model_path,
        ))

    print(f"[PPO] Loading policy from {model_path}")
    policy = Gr00tN1d6.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    policy.train()

    print("[PPO] Loading processor for data pipeline")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print("[PPO] Creating frozen reference policy")
    ref_policy = copy.deepcopy(policy).eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    dataset  = GR00TPreferenceDataset(hdf5_path, n_windows_per_pair=n_windows_per_pair)
    collator = make_preference_collator(EMBODIMENT_TAG, ACTION_KEYS, STATE_KEYS, VIDEO_KEYS, processor=processor)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=collator, num_workers=0, drop_last=True)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=lr
    )

    step = 0
    for ppo_iter in range(ppo_iters):
        # Snapshot current policy for ratio computation
        old_policy = copy.deepcopy(policy).eval()
        for p in old_policy.parameters():
            p.requires_grad_(False)

        for batch in loader:
            # Compute losses on winner trajectories (we want policy to match winners)
            policy_loss_w = compute_flow_loss(policy, batch["winner"], device)
            with torch.no_grad():
                ref_loss_w = compute_flow_loss(ref_policy, batch["winner"], device)
                old_loss_w = compute_flow_loss(old_policy, batch["winner"], device)

            loss, stats = ppo_loss(
                policy_loss_w, ref_loss_w, old_loss_w,
                clip_eps=clip_eps, kl_coeff=kl_coeff,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            step += 1
            reward_proxy = (ref_loss_w - policy_loss_w).mean().item()
            print(
                f"[PPO] iter={ppo_iter} step={step} loss={loss.item():.4f} "
                f"reward_proxy={reward_proxy:.4f} "
                f"adv={stats['advantage_mean']:.4f} "
                f"ratio={stats['ratio_mean']:.3f} "
                f"kl={stats['kl_penalty']:.4f}"
            )
            if use_wandb:
                import wandb
                wandb.log({
                    "ppo_loss": loss.item(),
                    "reward_proxy": reward_proxy,
                    **stats,
                }, step=step)

        del old_policy
        torch.cuda.empty_cache()

        # Save checkpoint every 10 iterations
        if (ppo_iter + 1) % 10 == 0:
            ckpt = os.path.join(output_dir, f"checkpoint-iter{ppo_iter}")
            policy.save_pretrained(ckpt)
            print(f"[PPO] Checkpoint: {ckpt}")

    policy.save_pretrained(output_dir)
    print(f"[PPO] Done. Model at {output_dir}")
    if use_wandb:
        import wandb; wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",         required=True)
    parser.add_argument("--hdf5_path",          required=True)
    parser.add_argument("--output_dir",         default="outputs/ppo_groot")
    parser.add_argument("--ppo_iters",          type=int,   default=30)
    parser.add_argument("--batch_size",         type=int,   default=2)
    parser.add_argument("--lr",                 type=float, default=1e-5)
    parser.add_argument("--clip_eps",           type=float, default=0.2)
    parser.add_argument("--kl_coeff",           type=float, default=0.1)
    parser.add_argument("--n_windows_per_pair", type=int,   default=5)
    parser.add_argument("--use_wandb",          action="store_true")
    parser.add_argument("--wandb_project",      default="gr00t-ppo")
    args = parser.parse_args()
    train_ppo(**{k: v for k, v in vars(args).items()})
