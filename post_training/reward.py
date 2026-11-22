"""Versioned execution + fidelity rewards (`reward_v1`, `reward_v2`) and group advantages."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from post_training import REWARD_VERSION, REWARD_VERSION_V2, REWARD_VERSION_V3


@dataclass(frozen=True)
class RewardWeights:
    version: str = REWARD_VERSION
    w_success: float = 1.0
    w_survival: float = 0.5
    # reward_v1 combined execution term
    w_exec_track: float = 0.3
    # reward_v2 / reward_v3 decomposed execution, seam & velocity-preservation terms
    w_exec_root: float = 1.2
    w_exec_joint: float = 0.8
    w_seam: float = 0.0
    w_fidelity: float = 1.0
    w_vel_fidelity: float = 0.0
    invalid_penalty: float = -1.0
    exec_joint_scale: float = 0.25
    exec_root_scale: float = 0.25
    seam_scale: float = 0.05
    fid_joint_scale: float = 0.30
    fid_body_scale: float = 0.20
    vel_fid_scale: float = 0.30


def make_reward_weights_v2(
    w_success: float = 1.0,
    w_survival: float = 0.5,
    w_exec_root: float = 1.2,
    w_exec_joint: float = 0.8,
    w_fidelity: float = 0.4,
    w_vel_fidelity: float = 0.3,
) -> RewardWeights:
    """Evidence-driven reward_v2: sharper root/joint execution terms + velocity preservation."""
    return RewardWeights(
        version=REWARD_VERSION_V2,
        w_success=w_success,
        w_survival=w_survival,
        w_exec_track=0.0,
        w_exec_root=w_exec_root,
        w_exec_joint=w_exec_joint,
        w_fidelity=w_fidelity,
        w_vel_fidelity=w_vel_fidelity,
        exec_joint_scale=0.15,
        exec_root_scale=0.15,
        fid_joint_scale=0.30,
        fid_body_scale=0.20,
        vel_fid_scale=0.30,
    )


def make_reward_weights_v3(
    w_success: float = 1.0,
    w_survival: float = 0.6,
    w_exec_root: float = 1.2,
    w_exec_joint: float = 0.8,
    w_seam: float = 0.8,
    w_fidelity: float = 0.4,
    w_vel_fidelity: float = 0.5,
) -> RewardWeights:
    """Long-horizon stitched reward_v3: adds explicit seam consistency + stronger velocity preservation."""
    return RewardWeights(
        version=REWARD_VERSION_V3,
        w_success=w_success,
        w_survival=w_survival,
        w_exec_track=0.0,
        w_exec_root=w_exec_root,
        w_exec_joint=w_exec_joint,
        w_seam=w_seam,
        w_fidelity=w_fidelity,
        w_vel_fidelity=w_vel_fidelity,
        exec_joint_scale=0.15,
        exec_root_scale=0.20,
        seam_scale=0.05,
        fid_joint_scale=0.30,
        fid_body_scale=0.20,
        vel_fid_scale=0.30,
    )


def compute_sample_rewards(
    tracker_results: list[dict[str, Any]],
    pred_joint_pos: np.ndarray,
    pred_body_pos_z: np.ndarray,
    gt_joint_pos: np.ndarray,
    gt_body_pos_z: np.ndarray,
    lengths: np.ndarray,
    weights: RewardWeights = RewardWeights(),
    fps: np.ndarray | float = 30.0,
    seam_overlap_err: np.ndarray | None = None,
    seam_joint_jump: np.ndarray | None = None,
    interior_joint_step: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Combine frozen SONIC tracker execution feedback with reference fidelity.

    Supports `reward_v1`, `reward_v2`, and `reward_v3` (stitched long-horizon
    reward with explicit boundary seam consistency).
    """
    if weights.version not in (REWARD_VERSION, REWARD_VERSION_V2, REWARD_VERSION_V3):
        raise ValueError(f"Unsupported reward version: {weights.version!r}")

    n = len(tracker_results)
    fps_arr = np.full(n, float(fps), dtype=np.float32) if np.ndim(fps) == 0 else np.asarray(fps, dtype=np.float32)
    rewards = np.empty(n, dtype=np.float32)
    is_success = np.zeros(n, dtype=np.bool_)
    is_valid = np.zeros(n, dtype=np.bool_)
    survival_ratio = np.zeros(n, dtype=np.float32)
    exec_track = np.zeros(n, dtype=np.float32)
    exec_root_score = np.zeros(n, dtype=np.float32)
    exec_joint_score = np.zeros(n, dtype=np.float32)
    exec_joint_mae = np.full(n, np.nan, dtype=np.float32)
    exec_root_err = np.full(n, np.nan, dtype=np.float32)
    fidelity = np.zeros(n, dtype=np.float32)
    vel_fidelity = np.zeros(n, dtype=np.float32)
    seam_score = np.ones(n, dtype=np.float32)
    mean_joint_vel = np.full(n, np.nan, dtype=np.float32)
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
        exec_joint_mae[i] = e_j
        exec_root_err[i] = e_r
        r_ej = math.exp(-e_j / weights.exec_joint_scale)
        r_er = math.exp(-e_r / weights.exec_root_scale)
        exec_joint_score[i] = r_ej
        exec_root_score[i] = r_er
        r_trk = r_ej * r_er
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

        # Joint angular speed fidelity (prevents sluggish / over-smoothed motion).
        if t_len >= 2:
            fps_i = float(fps_arr[i])
            p_speed = np.abs(np.diff(p_jp, axis=0)) * fps_i
            g_speed = np.abs(np.diff(g_jp, axis=0)) * fps_i
            mean_joint_vel[i] = float(np.mean(p_speed))
            vel_mae = float(np.mean(np.abs(p_speed - g_speed)))
            r_vfid = math.exp(-vel_mae / weights.vel_fid_scale)
        else:
            mean_joint_vel[i] = 0.0
            r_vfid = 1.0
        vel_fidelity[i] = r_vfid

        if weights.version == REWARD_VERSION:
            rewards[i] = (
                weights.w_success * (1.0 if succ else 0.0)
                + weights.w_survival * surv
                + weights.w_exec_track * r_trk
                + weights.w_fidelity * r_fid
            )
        elif weights.version == REWARD_VERSION_V2:
            rewards[i] = (
                weights.w_success * (1.0 if succ else 0.0)
                + weights.w_survival * surv
                + weights.w_exec_root * r_er
                + weights.w_exec_joint * r_ej
                + weights.w_fidelity * r_fid
                + weights.w_vel_fidelity * r_vfid
            )
        else:
            if seam_overlap_err is not None and seam_joint_jump is not None and interior_joint_step is not None:
                ov_err = max(0.0, float(seam_overlap_err[i]))
                excess = max(0.0, float(seam_joint_jump[i]) - float(interior_joint_step[i]))
                s_err = 0.5 * ov_err + 0.5 * excess
                r_seam = math.exp(-s_err / weights.seam_scale)
            else:
                r_seam = 1.0
            seam_score[i] = r_seam
            rewards[i] = (
                weights.w_success * (1.0 if succ else 0.0)
                + weights.w_survival * surv
                + weights.w_exec_root * r_er
                + weights.w_exec_joint * r_ej
                + weights.w_seam * r_seam
                + weights.w_fidelity * r_fid
                + weights.w_vel_fidelity * r_vfid
            )

    return {
        "reward": rewards,
        "is_success": is_success,
        "is_valid": is_valid,
        "survival_ratio": survival_ratio,
        "exec_track": exec_track,
        "exec_root_score": exec_root_score,
        "exec_joint_score": exec_joint_score,
        "exec_joint_mae_rad": exec_joint_mae,
        "exec_root_pos_err_m": exec_root_err,
        "seam_score": seam_score,
        "fidelity": fidelity,
        "vel_fidelity": vel_fidelity,
        "mean_joint_vel_rad_s": mean_joint_vel,
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
