"""Exact rigid multi-window stitching & multi-frame overlap blending in Y-up space."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from motion_lib.motion_utils.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from post_training.policy import build_ddim_schedule

G1_LOCAL_VEL_START = 186  # [0:174] body_pose_r6d, [174:180] cam orient, [180:186] gv orient, [186:189] local_transl_vel


@torch.no_grad()
def sample_ddim_synchronized(
    pipeline: Any,
    denoiser: torch.nn.Module,
    f_cond: torch.Tensor,
    length: torch.Tensor,
    x_init: torch.Tensor,
    chunks_meta: list[tuple[int, int]],
    total_len: int,
    ddim_steps: int = 50,
    eta: float = 0.0,
    all_valid_length: bool = False,
    use_bf16: bool = True,
    sync_mode: str = "none",
    sync_weight_mode: str = "cosine",
    inpaint_resamples: int = 1,
    resample_seed: int = 2026,
) -> torch.Tensor:
    """DDIM sampling across overlapping chunks with optional overlap-region synchronization.

    Empirically verified 189-D invariance across independently canonicalized windows:
      - [0:174]   body_pose_r6d:    window-invariant (0.00e+00)
      - [174:180] global_orient_c:  window-invariant (per-frame camera frame, ~1e-6)
      - [180:186] global_orient_gv: window-invariant (per-frame GV_t frame, ~3e-6)
      - [186:189] local_transl_vel: window-invariant on frames 0..ov-2 (body frame, ~3e-5);
        frame ov-1 (last valid frame of earlier chunk) is last-frame padded during encoding,
        so synchronization leaves [186:189] on frame ov-1 controlled by the later chunk.

    Args:
        sync_mode:
            - `"none"`: independent batched DDIM across chunks (baseline).
            - `"multidiffusion"`: batched bidirectional joint denoising — shares initial
              noise on overlapping valid frames and fuses `pred_x0` across overlapping
              windows at every DDIM step with cosine (or uniform) interior weights.
            - `"inpainting"`: sequential causal DDIM prefix inpainting (chunk 0 -> N-1)
              constraining each chunk's overlap prefix to the previous chunk's clean `x0`
              (with optional RePaint-style resampling via `inpaint_resamples > 1`).
    """
    if sync_mode not in ("none", "multidiffusion", "inpainting"):
        raise ValueError(f"Unsupported sync_mode {sync_mode!r}; expected 'none', 'multidiffusion', or 'inpainting'")
    if sync_weight_mode not in ("cosine", "uniform"):
        raise ValueError(f"Unsupported sync_weight_mode {sync_weight_mode!r}")
    if inpaint_resamples < 1:
        raise ValueError(f"inpaint_resamples must be >= 1, got {inpaint_resamples}")

    device = f_cond.device
    n_chunks, window, x_dim = x_init.shape
    schedule = build_ddim_schedule(int(pipeline.train_diffusion.num_timesteps), int(ddim_steps))
    alphas_full = np.asarray(pipeline.train_diffusion.alphas_cumprod, dtype=np.float64)
    ab = np.array([alphas_full[t] for t in schedule], dtype=np.float64)
    ab_prev = np.append(1.0, ab[:-1])

    sigmas = (
        float(eta)
        * np.sqrt(np.maximum((1.0 - ab_prev) / (1.0 - ab), 0.0))
        * np.sqrt(np.maximum(1.0 - ab / ab_prev, 0.0))
    )
    sigmas[0] = 0.0
    sqrt_ab = np.sqrt(ab)
    sqrt_1m_ab = np.sqrt(1.0 - ab)
    sqrt_ab_prev = np.sqrt(ab_prev)
    dir_coeff = np.sqrt(np.maximum(1.0 - ab_prev - sigmas**2, 0.0))
    ts_orig = torch.tensor(schedule, dtype=torch.long, device=device)

    denoiser.eval()

    if sync_mode == "none" or n_chunks == 1:
        x = x_init.clone().float()
        for step_idx in reversed(range(ddim_steps)):
            ts = ts_orig[step_idx].expand(n_chunks)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16 and device.type == "cuda"):
                out = denoiser(
                    f_cond=f_cond,
                    length=length,
                    xt=x,
                    timesteps=ts,
                    all_valid_length=all_valid_length,
                )
            pred_x0 = out["pred_x_start"].float()
            eps = (x - float(sqrt_ab[step_idx]) * pred_x0) / max(float(sqrt_1m_ab[step_idx]), 1e-8)
            mean = float(sqrt_ab_prev[step_idx]) * pred_x0 + float(dir_coeff[step_idx]) * eps
            sigma_k = float(sigmas[step_idx])
            x = mean + sigma_k * torch.randn_like(x) if (step_idx > 0 and sigma_k > 0.0) else mean
        return x

    if sync_mode == "multidiffusion":
        # 1. Share i.i.d. N(0, I) initial noise across overlapping valid frames.
        x_canvas_init = torch.zeros(total_len, x_dim, device=device, dtype=torch.float32)
        prev_end = 0
        for ci, (start, actual_len) in enumerate(chunks_meta):
            end = start + actual_len
            if ci == 0:
                x_canvas_init[start:end] = x_init[ci, :actual_len].float()
            else:
                ov = prev_end - start
                x_canvas_init[prev_end:end] = x_init[ci, ov:actual_len].float()
            prev_end = end

        x = x_init.clone().float()
        for ci, (start, actual_len) in enumerate(chunks_meta):
            x[ci, :actual_len] = x_canvas_init[start : start + actual_len]

        # 2. Precompute per-chunk fusion weights (separate for [0:186] and [186:189]).
        w_pose = torch.zeros(n_chunks, window, device=device, dtype=torch.float32)
        w_vel = torch.zeros(n_chunks, window, device=device, dtype=torch.float32)
        for ci, (start, actual_len) in enumerate(chunks_meta):
            w_pose[ci, :actual_len] = 1.0
            w_vel[ci, :actual_len] = 1.0

        prev_end = chunks_meta[0][0] + chunks_meta[0][1]
        for ci in range(1, n_chunks):
            start, actual_len = chunks_meta[ci]
            prev_start, prev_len = chunks_meta[ci - 1]
            ov = prev_end - start
            if ov > 0:
                if sync_weight_mode == "cosine":
                    idx = torch.arange(1, ov + 1, device=device, dtype=torch.float32)
                    alpha = 0.5 * (1.0 - torch.cos(math.pi * idx / float(ov + 1)))
                else:
                    alpha = torch.full((ov,), 0.5, device=device, dtype=torch.float32)
                w_pose[ci - 1, prev_len - ov : prev_len] = 1.0 - alpha
                w_pose[ci, :ov] = alpha
                w_vel[ci - 1, prev_len - ov : prev_len] = 1.0 - alpha
                w_vel[ci, :ov] = alpha
                # Exclude earlier chunk's last padded velocity frame (prev_len - 1).
                w_vel[ci - 1, prev_len - 1] = 0.0
                w_vel[ci, ov - 1] = 1.0
            prev_end = start + actual_len

        denom_pose = torch.zeros(total_len, 1, device=device, dtype=torch.float32)
        denom_vel = torch.zeros(total_len, 1, device=device, dtype=torch.float32)
        for ci, (start, actual_len) in enumerate(chunks_meta):
            denom_pose[start : start + actual_len, 0] += w_pose[ci, :actual_len]
            denom_vel[start : start + actual_len, 0] += w_vel[ci, :actual_len]
        denom_pose = denom_pose.clamp(min=1e-8)
        denom_vel = denom_vel.clamp(min=1e-8)

        # 3. Synchronized batched DDIM denoising loop.
        vel_s = min(G1_LOCAL_VEL_START, x_dim)
        for step_idx in reversed(range(ddim_steps)):
            ts = ts_orig[step_idx].expand(n_chunks)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16 and device.type == "cuda"):
                out = denoiser(
                    f_cond=f_cond,
                    length=length,
                    xt=x,
                    timesteps=ts,
                    all_valid_length=all_valid_length,
                )
            pred_x0 = out["pred_x_start"].float()

            canvas_pose = torch.zeros(total_len, vel_s, device=device, dtype=torch.float32)
            canvas_vel = torch.zeros(total_len, x_dim - vel_s, device=device, dtype=torch.float32)
            for ci, (start, actual_len) in enumerate(chunks_meta):
                end = start + actual_len
                canvas_pose[start:end] += w_pose[ci, :actual_len, None] * pred_x0[ci, :actual_len, :vel_s]
                if x_dim > vel_s:
                    canvas_vel[start:end] += w_vel[ci, :actual_len, None] * pred_x0[ci, :actual_len, vel_s:]
            canvas_pose = canvas_pose / denom_pose
            if x_dim > vel_s:
                canvas_vel = canvas_vel / denom_vel

            for ci, (start, actual_len) in enumerate(chunks_meta):
                end = start + actual_len
                pred_x0[ci, :actual_len, :vel_s] = canvas_pose[start:end]
                if x_dim > vel_s:
                    pred_x0[ci, :actual_len, vel_s:] = canvas_vel[start:end]

            eps = (x - float(sqrt_ab[step_idx]) * pred_x0) / max(float(sqrt_1m_ab[step_idx]), 1e-8)
            mean = float(sqrt_ab_prev[step_idx]) * pred_x0 + float(dir_coeff[step_idx]) * eps
            sigma_k = float(sigmas[step_idx])
            if step_idx > 0 and sigma_k > 0.0:
                noise_canvas = torch.randn(total_len, x_dim, device=device, dtype=torch.float32)
                noise = torch.randn_like(x)
                for ci, (start, actual_len) in enumerate(chunks_meta):
                    noise[ci, :actual_len] = noise_canvas[start : start + actual_len]
                x = mean + sigma_k * noise
            else:
                x = mean
        return x

    # sync_mode == "inpainting": sequential causal DDIM prefix inpainting (chunk 0 -> N-1)
    vel_s = min(G1_LOCAL_VEL_START, x_dim)
    x0_out = torch.zeros_like(x_init, dtype=torch.float32)
    res_gen = torch.Generator(device=device).manual_seed(int(resample_seed))
    prev_end = 0

    for ci, (start, actual_len) in enumerate(chunks_meta):
        f_c = f_cond[ci : ci + 1]
        l_c = length[ci : ci + 1]
        all_valid_c = bool(actual_len == window)
        x_c = x_init[ci : ci + 1].clone().float()

        if ci == 0:
            for step_idx in reversed(range(ddim_steps)):
                ts = ts_orig[step_idx].expand(1)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16 and device.type == "cuda"):
                    out = denoiser(f_cond=f_c, length=l_c, xt=x_c, timesteps=ts, all_valid_length=all_valid_c)
                pred_x0 = out["pred_x_start"].float()
                eps = (x_c - float(sqrt_ab[step_idx]) * pred_x0) / max(float(sqrt_1m_ab[step_idx]), 1e-8)
                mean = float(sqrt_ab_prev[step_idx]) * pred_x0 + float(dir_coeff[step_idx]) * eps
                sigma_k = float(sigmas[step_idx])
                x_c = mean + sigma_k * torch.randn_like(x_c) if (step_idx > 0 and sigma_k > 0.0) else mean
            x0_out[ci] = x_c[0]
            prev_end = start + actual_len
            continue

        prev_len = chunks_meta[ci - 1][1]
        ov = prev_end - start
        x0_known_full = torch.zeros(1, window, x_dim, device=device, dtype=torch.float32)
        x0_known_full[0, :ov] = x0_out[ci - 1, prev_len - ov : prev_len]

        mask = torch.zeros(1, window, x_dim, device=device, dtype=torch.float32)
        if ov > 1:
            mask[0, : ov - 1, :] = 1.0
        mask[0, ov - 1, :vel_s] = 1.0  # Leave local_transl_vel free at prev's padded end frame.

        eps_known = x_init[ci : ci + 1].clone().float()

        for step_idx in reversed(range(ddim_steps)):
            ts = ts_orig[step_idx].expand(1)
            n_res = inpaint_resamples if step_idx > 0 else 1
            for r_idx in range(n_res):
                x_known_t = float(sqrt_ab[step_idx]) * x0_known_full + float(sqrt_1m_ab[step_idx]) * eps_known
                x_in = mask * x_known_t + (1.0 - mask) * x_c

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16 and device.type == "cuda"):
                    out = denoiser(f_cond=f_c, length=l_c, xt=x_in, timesteps=ts, all_valid_length=all_valid_c)
                pred_x0 = out["pred_x_start"].float()
                pred_x0_sync = mask * x0_known_full + (1.0 - mask) * pred_x0

                eps = (x_in - float(sqrt_ab[step_idx]) * pred_x0_sync) / max(float(sqrt_1m_ab[step_idx]), 1e-8)
                mean = float(sqrt_ab_prev[step_idx]) * pred_x0_sync + float(dir_coeff[step_idx]) * eps
                sigma_k = float(sigmas[step_idx])
                x_prev = mean + sigma_k * torch.randn_like(x_c) if (step_idx > 0 and sigma_k > 0.0) else mean

                if step_idx > 0 and r_idx < n_res - 1:
                    ratio_ab = float(ab[step_idx] / max(ab_prev[step_idx], 1e-12))
                    z = torch.randn(x_c.shape, device=device, dtype=torch.float32, generator=res_gen)
                    x_c = math.sqrt(ratio_ab) * x_prev + math.sqrt(max(1.0 - ratio_ab, 0.0)) * z
                    # Keep eps_known Markov-consistent with the exact jump-back sample q(x_t | x_{t-1}).
                    eps_known = (x_c - float(sqrt_ab[step_idx]) * x0_known_full) / max(float(sqrt_1m_ab[step_idx]), 1e-8)
                else:
                    x_c = x_prev

        x0_out[ci] = (mask * x0_known_full + (1.0 - mask) * x_c)[0]
        prev_end = start + actual_len

    return x0_out


def chunk_windows(total_len: int, window: int = 120, overlap: int = 1) -> list[tuple[int, int]]:
    """Partition `total_len` frames into overlapping windows `(start, actual_len)`.

    Guarantees `overlap == 1` reproduces `tools/infer_video.py` chunk starts exactly,
    and for any `1 <= overlap < window`, every chunk after chunk 0 has `actual_len >= overlap + 1`.
    """
    if overlap < 1 or overlap >= window:
        raise ValueError(f"overlap must satisfy 1 <= overlap < window ({window}), got {overlap}")
    if total_len <= window:
        return [(0, total_len)]
    stride = window - overlap
    starts = [0]
    while starts[-1] + window < total_len:
        starts.append(starts[-1] + stride)
    return [(s, min(window, total_len - s)) for s in starts]


def slerp_quaternions(q0: torch.Tensor, q1: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Shortest-arc geodesic SLERP between unit wxyz quaternions `q0` and `q1` (..., 4)."""
    q0_n = F.normalize(q0, dim=-1)
    q1_n = F.normalize(q1, dim=-1)
    dot = (q0_n * q1_n).sum(dim=-1, keepdim=True)
    # Enforce shortest arc on S^3 (q and -q represent the same rotation).
    q1_n = torch.where(dot < 0.0, -q1_n, q1_n)
    dot = dot.abs().clamp(0.0, 1.0)

    a = alpha.unsqueeze(-1) if alpha.ndim == q0.ndim - 1 else alpha
    theta = torch.acos(dot.clamp(max=1.0 - 1e-7))
    sin_theta = torch.sin(theta)
    w0 = torch.sin((1.0 - a) * theta) / sin_theta.clamp(min=1e-7)
    w1 = torch.sin(a * theta) / sin_theta.clamp(min=1e-7)
    slerp_out = w0 * q0_n + w1 * q1_n
    lerp_out = F.normalize((1.0 - a) * q0_n + a * q1_n, dim=-1)
    return F.normalize(torch.where(dot > 0.9995, lerp_out, slerp_out), dim=-1)


def _compute_overlap_alignment_yup(
    prev_rp: torch.Tensor,
    prev_rq: torch.Tensor,
    curr_rp: torch.Tensor,
    curr_rq: torch.Tensor,
    align_mode: str = "so3",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute rigid `(delta_R, t_prev, t_curr)` aligning `curr` overlap to `prev` overlap in Y-up.

    Args:
        prev_rp, curr_rp: `(O, 3)` overlap root positions.
        prev_rq, curr_rq: `(O, 4)` overlap wxyz root quaternions.
        align_mode: `"so3"` (full 3D Wahba/Kabsch projection onto SO(3)) or
            `"yaw"` (gravity-preserving Y-axis yaw rotation only).
    """
    if align_mode not in ("so3", "yaw"):
        raise ValueError(f"Unsupported align_mode {align_mode!r}; expected 'so3' or 'yaw'")
    ov = prev_rp.shape[0]
    device = prev_rp.device
    dtype = prev_rp.dtype

    if ov == 1 and align_mode == "so3":
        R_prev = quaternion_to_matrix(prev_rq[0])
        R_curr = quaternion_to_matrix(curr_rq[0])
        delta_R = R_prev @ R_curr.transpose(-1, -2)
        return delta_R, prev_rp[0], curr_rp[0]

    # Interior-peaked weights suppress noisy boundary frames of both windows.
    idx = torch.arange(1, ov + 1, device=device, dtype=dtype)
    w = torch.sin(math.pi * idx / float(ov + 1))
    w = w / w.sum()

    R_prev = quaternion_to_matrix(prev_rq)  # (O, 3, 3)
    R_curr = quaternion_to_matrix(curr_rq)  # (O, 3, 3)
    M = torch.einsum("o,oik,ojk->ij", w, R_prev, R_curr)  # sum_o w_o R_prev[o] R_curr[o]^T

    if align_mode == "yaw":
        # Exact optimal Y-axis rotation maximizing tr(R_y(theta)^T M).
        yaw = torch.atan2(M[0, 2] - M[2, 0], M[0, 0] + M[2, 2])
        c, s = torch.cos(yaw), torch.sin(yaw)
        zero = torch.zeros_like(c)
        one = torch.ones_like(c)
        delta_R = torch.stack(
            [
                torch.stack([c, zero, s]),
                torch.stack([zero, one, zero]),
                torch.stack([-s, zero, c]),
            ],
            dim=0,
        )
    else:
        U, _, Vt = torch.linalg.svd(M)
        det = torch.det(U @ Vt)
        diag = torch.ones(3, device=device, dtype=dtype)
        diag[2] = torch.where(det < 0.0, -1.0, 1.0)
        delta_R = U @ torch.diag(diag) @ Vt

    t_prev = (w[:, None] * prev_rp).sum(dim=0)
    t_curr = (w[:, None] * curr_rp).sum(dim=0)
    return delta_R, t_prev, t_curr


@torch.no_grad()
def stitch_sequence_yup(
    jp_c: torch.Tensor,
    rp_c: torch.Tensor,
    rq_c: torch.Tensor,
    bp_c: torch.Tensor,
    chunks_meta: list[tuple[int, int]],
    total_len: int,
    align_mode: str = "so3",
    blend_mode: str = "cosine",
    fk_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, torch.Tensor | list[int] | list[tuple[int, int]]]:
    """Stitch arbitrary-length decoded Y-up chunks with single- or multi-frame overlap blending.

    When every overlap has length 1 and `align_mode == "so3"`, this matches
    `tools/infer_video.py::align_window_to_previous` exactly. For multi-frame
    overlaps (`O > 1`), aligns over the overlap interior, blends `joint_pos` and
    `root_pos` with `blend_mode` (`"cosine"` or `"linear"`), SLERPs `root_quat`,
    and recomputes `body_pos` via `fk_fn` when provided so links stay kinematically
    exact.
    """
    if blend_mode not in ("cosine", "linear"):
        raise ValueError(f"Unsupported blend_mode {blend_mode!r}; expected 'cosine' or 'linear'")

    device = jp_c.device
    dtype = jp_c.dtype
    jp_out = torch.zeros(total_len, jp_c.shape[-1], device=device, dtype=dtype)
    rp_out = torch.zeros(total_len, 3, device=device, dtype=dtype)
    rq_out = torch.zeros(total_len, 4, device=device, dtype=dtype)
    bp_out = torch.zeros(total_len, bp_c.shape[2], 3, device=device, dtype=dtype)

    prev_end = 0
    seam_indices: list[int] = []
    overlap_spans: list[tuple[int, int]] = []
    used_multi_blend = False

    for ci, (start, actual_len) in enumerate(chunks_meta):
        end = start + actual_len
        p_jp = jp_c[ci, :actual_len]
        p_rp = rp_c[ci, :actual_len]
        p_rq = rq_c[ci, :actual_len]
        p_bp = bp_c[ci, :actual_len]

        if ci == 0:
            jp_out[start:end] = p_jp
            rp_out[start:end] = p_rp
            rq_out[start:end] = p_rq
            bp_out[start:end] = p_bp
            prev_end = end
            continue

        ov = prev_end - start
        if ov < 1 or ov >= actual_len:
            raise ValueError(
                f"Invalid overlap ov={ov} for chunk {ci} (start={start}, actual_len={actual_len}, prev_end={prev_end})"
            )

        delta_R, t_prev, t_curr = _compute_overlap_alignment_yup(
            rp_out[start:prev_end],
            rq_out[start:prev_end],
            p_rp[:ov],
            p_rq[:ov],
            align_mode=align_mode,
        )
        aligned_rp = (p_rp - t_curr) @ delta_R.transpose(-1, -2) + t_prev
        aligned_bp = (p_bp - t_curr) @ delta_R.transpose(-1, -2) + t_prev
        aligned_rq = matrix_to_quaternion(delta_R.unsqueeze(0) @ quaternion_to_matrix(p_rq))

        overlap_spans.append((start, prev_end))
        if ov == 1:
            seam_indices.append(start)
            jp_out[start + 1 : end] = p_jp[1:]
            rp_out[start + 1 : end] = aligned_rp[1:]
            rq_out[start + 1 : end] = aligned_rq[1:]
            bp_out[start + 1 : end] = aligned_bp[1:]
        else:
            used_multi_blend = True
            seam_indices.append(start + ov // 2)
            idx = torch.arange(1, ov + 1, device=device, dtype=dtype)
            if blend_mode == "cosine":
                alpha = 0.5 * (1.0 - torch.cos(math.pi * idx / float(ov + 1)))
            else:
                alpha = idx / float(ov + 1)

            jp_out[start:prev_end] = (1.0 - alpha[:, None]) * jp_out[start:prev_end] + alpha[:, None] * p_jp[:ov]
            rp_out[start:prev_end] = (1.0 - alpha[:, None]) * rp_out[start:prev_end] + alpha[:, None] * aligned_rp[:ov]
            rq_out[start:prev_end] = slerp_quaternions(rq_out[start:prev_end], aligned_rq[:ov], alpha)
            if fk_fn is None:
                bp_out[start:prev_end] = (
                    (1.0 - alpha[:, None, None]) * bp_out[start:prev_end] + alpha[:, None, None] * aligned_bp[:ov]
                )

            jp_out[prev_end:end] = p_jp[ov:]
            rp_out[prev_end:end] = aligned_rp[ov:]
            rq_out[prev_end:end] = aligned_rq[ov:]
            bp_out[prev_end:end] = aligned_bp[ov:]

        prev_end = end

    if used_multi_blend and fk_fn is not None:
        bp_out = fk_fn(jp_out, rp_out, rq_out)

    return {
        "joint_pos": jp_out,
        "root_pos_w": rp_out,
        "root_quat_w": rq_out,
        "body_pos_w": bp_out,
        "seam_indices": seam_indices,
        "overlap_spans": overlap_spans,
    }


@torch.no_grad()
def stitch_two_chunks_yup(
    jp0: torch.Tensor,
    rp0: torch.Tensor,
    rq0: torch.Tensor,
    bp0: torch.Tensor,
    jp1: torch.Tensor,
    rp1: torch.Tensor,
    rq1: torch.Tensor,
    bp1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Batched exact rigid alignment matching `tools/infer_video.py::align_window_to_previous`.

    Args:
        jp0, jp1: (N, W, 29) G1 joint angles for chunk 0 (`[0:W]`) and chunk 1 (`[W-1:2W-1]`).
        rp0, rp1: (N, W, 3) Y-up root positions.
        rq0, rq1: (N, W, 4) Y-up wxyz root quaternions.
        bp0, bp1: (N, W, J, 3) Y-up body link positions.

    Returns:
        Stitched `(jp, rp, rq, bp)` of length `2*W - 1` plus `seam_info` dict with
        `overlap_err`, `seam_jump`, and `interior_step` (each `(N,)` in radians).
    """
    prev_pos = rp0[:, -1]       # (N, 3)
    prev_quat = rq0[:, -1]      # (N, 4)
    curr_origin = rp1[:, 0]     # (N, 3)
    curr_quat = rq1[:, 0]       # (N, 4)

    R_prev = quaternion_to_matrix(prev_quat)
    R_curr = quaternion_to_matrix(curr_quat)
    delta_R = R_prev @ R_curr.transpose(-1, -2)  # (N, 3, 3)

    aligned_rp1 = torch.einsum("nlj,nij->nli", rp1 - curr_origin[:, None, :], delta_R) + prev_pos[:, None, :]
    aligned_bp1 = (
        torch.einsum("nlkj,nij->nlki", bp1 - curr_origin[:, None, None, :], delta_R)
        + prev_pos[:, None, None, :]
    )
    aligned_rq1 = matrix_to_quaternion(delta_R[:, None] @ quaternion_to_matrix(rq1))

    jp_stitched = torch.cat([jp0, jp1[:, 1:]], dim=1)          # (N, 2W-1, 29)
    rp_stitched = torch.cat([rp0, aligned_rp1[:, 1:]], dim=1)  # (N, 2W-1, 3)
    rq_stitched = torch.cat([rq0, aligned_rq1[:, 1:]], dim=1)  # (N, 2W-1, 4)
    bp_stitched = torch.cat([bp0, aligned_bp1[:, 1:]], dim=1)  # (N, 2W-1, J, 3)

    # Exact boundary discontinuity metrics (radians):
    overlap_err = (jp1[:, 0] - jp0[:, -1]).abs().mean(dim=-1)  # (N,) same video frame W-1
    seam_jump = (jp1[:, 1] - jp0[:, -1]).abs().mean(dim=-1)    # (N,) stitched step W-1 -> W
    all_steps = (jp_stitched[:, 1:] - jp_stitched[:, :-1]).abs().mean(dim=-1)  # (N, 2W-2)
    seam_idx = jp0.shape[1] - 1
    interior_steps = torch.cat([all_steps[:, :seam_idx], all_steps[:, seam_idx + 1 :]], dim=1)
    interior_step = interior_steps.mean(dim=-1)

    seam_info = {
        "overlap_err": overlap_err,
        "seam_jump": seam_jump,
        "interior_step": interior_step,
    }
    return jp_stitched, rp_stitched, rq_stitched, bp_stitched, seam_info
