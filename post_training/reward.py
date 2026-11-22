"""Versioned execution + fidelity reward (`reward_v1`) and group-normalized advantages."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from post_training import REWARD_VERSION


@dataclass(frozen=True)
class RewardWeights:
    version: str = REWARD_VERSION
    w_success: float = 1.0
    w_survival: float = 0.5
    w_exec_track: float = 0.3
    w_fidelity: float = 1.0
    invalid_penalty: float = -1.0
    exec_joint_scale: float = 0.25
    exec_root_scale: float = 0.25
    fid_joint_scale: float = 0.30
    fid_body_scale: float = 0.20


def compute_sample_rewards(
    tracker_results: list[dict[str, Any]],
    pred_joint_pos: np.ndarray,
    pred_body_pos_z: np.ndarray,
    gt_joint_pos: np.ndarray,
    gt_body_pos_z: np.ndarray,
    lengths: np.ndarray,
    weights: RewardWeights = RewardWeights(),
) -> dict[str, np.ndarray]:
    """Combine frozen SONIC tracker execution feedback with reference fidelity.

    Fidelity compares generated joint angles and root-relative 30-body positions
    against the clip's Z-up reference G1 trajectory (e.g. g1_new.pth), preventing
    the policy from maximizing tracking survival by standing still.
    """
    n = len(tracker_results)
    rewards = np.empty(n, dtype=np.float32)
    is_success = np.zeros(n, dtype=np.bool_)
    is_valid = np.zeros(n, dtype=np.bool_)
    survival_ratio = np.zeros(n, dtype=np.float32)
    exec_track = np.zeros(n, dtype=np.float32)
    fidelity = np.zeros(n, dtype=np.float32)
    ref_joint_mae = np.full(n, np.nan, dtype=np.float32)
    ref_body_mpjpe = np.full(n, np.nan, dtype=np.float32)

    for i, trk in enumerate(tracker_results):
        t_len = int(lengths[i])
        p_jp = pred_joint_pos[i, :t_len]
        p_bp = pred_body_pos_z[i, :t_len]
        g_jp = gt_joint_pos[i, :t_len]
        g_bp = gt_body_pos_z[i, :t_len]

        finite_ok = (
            bool(trk.get("valid", False))
            and bool(np.isfinite(p_jp).all())
            and bool(np.isfinite(p_bp).all())
            and bool(np.isfinite(g_jp).all())
            and bool(np.isfinite(g_bp).all())
        )
        if not finite_ok:
            rewards[i] = float(weights.invalid_penalty)
            continue

        is_valid[i] = True
        succ = bool(trk["is_success"])
        is_success[i] = succ
        total_50 = max(1, int(trk["total_frames_50hz"]))
        first_fail = int(trk["first_failure_frame_50hz"])
        surv = 1.0 if succ else float(np.clip(first_fail / total_50, 0.0, 1.0))
        survival_ratio[i] = surv

        e_j = float(trk["exec_joint_mae_rad"])
        e_r = float(trk["exec_root_pos_err_m"])
        r_trk = math.exp(-e_j / weights.exec_joint_scale) * math.exp(-e_r / weights.exec_root_scale)
        exec_track[i] = r_trk

        # Root-relative body fidelity (invariant to global translation offset).
        j_mae = float(np.mean(np.abs(p_jp - g_jp)))
        p_rel = p_bp - p_bp[:, 0:1, :]
        g_rel = g_bp - g_bp[:, 0:1, :]
        b_mpjpe = float(np.mean(np.linalg.norm(p_rel - g_rel, axis=-1)))
        ref_joint_mae[i] = j_mae
        ref_body_mpjpe[i] = b_mpjpe
        r_fid = 0.5 * math.exp(-j_mae / weights.fid_joint_scale) + 0.5 * math.exp(-b_mpjpe / weights.fid_body_scale)
        fidelity[i] = r_fid

        rewards[i] = (
            weights.w_success * (1.0 if succ else 0.0)
            + weights.w_survival * surv
            + weights.w_exec_track * r_trk
            + weights.w_fidelity * r_fid
        )

    return {
        "reward": rewards,
        "is_success": is_success,
        "is_valid": is_valid,
        "survival_ratio": survival_ratio,
        "exec_track": exec_track,
        "fidelity": fidelity,
        "ref_joint_mae_rad": ref_joint_mae,
        "ref_body_mpjpe_m": ref_body_mpjpe,
    }


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    clip_value: float = 5.0,
) -> torch.Tensor:
    """Group-normalized advantages (GRPO / DDPO-Group) over (B * K,) rewards."""
    if rewards.ndim != 1 or rewards.numel() % group_size != 0:
        raise ValueError(f"Expected 1D rewards divisible by group_size={group_size}, got {tuple(rewards.shape)}")
    grouped = rewards.float().view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, unbiased=False, keepdim=True)
    adv = torch.where(std > 1e-6, (grouped - mean) / (std + 1e-4), torch.zeros_like(grouped))
    return adv.clamp(-clip_value, clip_value).reshape(-1)
