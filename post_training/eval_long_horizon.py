#!/usr/bin/env python3
"""Long-horizon multi-window stitched Base vs PPO vs GT evaluation.

Uses the exact 1-frame overlap windowing and rigid endpoint alignment from
`tools/infer_video.py::align_window_to_previous` on full-length test sequences
(`>= min_frames`, e.g. 240-1901 frames in BEDLAM2 and H36M) to measure:
  - Long-horizon Protocol v4 success rate & survival ratio vs GT ceiling
  - Execution tracking error (`exec_root_pos_err_m`, `exec_joint_mae_rad`)
  - Window stitch-seam joint discontinuity vs interior frame-to-frame step
  - Failure timing relative to window stitch boundaries
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hmr4d.model.offline.utils.obs_joints import select_coco17_no_nose_ears
from hmr4d.utils.geo.hmr_cam import normalize_kp2d
from post_training.dataset import VIDEO_SOURCE_SPECS
from post_training.eval_compare import _exact_mcnemar_p, build_condition
from post_training.policy import G1_PRED_X_DIM, YUP_TO_ZUP_M
from post_training.stitching import chunk_windows, sample_ddim_synchronized, stitch_sequence_yup
from post_training.tracker_pool import PersistentTrackerPool
from tools.inference_utils import load_pipeline


def _pad_window(x: torch.Tensor, pad_len: int) -> torch.Tensor:
    if pad_len <= 0:
        return x
    return torch.cat([x, x[-1:].expand(pad_len, *x.shape[1:])], dim=0)


def _chunk_windows(total_len: int, window: int = 120, overlap: int = 1) -> list[tuple[int, int]]:
    return chunk_windows(total_len, window=window, overlap=overlap)


@torch.no_grad()
def _decode_chunks_yup(
    pipeline: Any,
    x_0: torch.Tensor,
    cam_angvel: torch.Tensor,
    length: torch.Tensor,
    root_pos_0_yup: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode batched chunks in internal Y-up space before rigid stitching."""
    from hmr4d.model.offline.pipeline.offline_pipeline import get_smpl_params_w_Rt_v2
    from hmr4d.utils.geo.hmr_global import rollout_local_transl_vel
    from pytorch3d.transforms import (
        axis_angle_to_matrix,
        matrix_to_axis_angle,
        matrix_to_quaternion,
        rotation_6d_to_matrix,
    )

    decode_dict = pipeline.endecoder.decode_g1_new(x_0.float())
    go_c_aa = matrix_to_axis_angle(decode_dict["global_orient_rotmat"].float())
    go_gv_aa = matrix_to_axis_angle(rotation_6d_to_matrix(decode_dict["global_orient_gv_r6d"].float()))
    local_transl_vel = decode_dict["local_transl_vel"].float()
    params_w = get_smpl_params_w_Rt_v2(
        global_orient_gv=go_gv_aa,
        local_transl_vel=local_transl_vel,
        global_orient_c=go_c_aa,
        cam_angvel=cam_angvel.float(),
        length=length,
    )
    pred_root_aa_w = params_w["global_orient"]
    pred_root_R_w = axis_angle_to_matrix(pred_root_aa_w)
    pred_root_quat_yup = matrix_to_quaternion(pred_root_R_w)
    pred_root_pos_yup = rollout_local_transl_vel(local_transl_vel, pred_root_aa_w, root_pos_0_yup.float())
    R_bp = decode_dict["body_pose_rotmat"].float()
    pred_dof_byd = torch.atan2(R_bp[..., 0, 2], R_bp[..., 0, 0])
    pred_body_yup, _ = pipeline.endecoder.fk_g1(
        pred_dof_byd, pred_root_pos_yup, pred_root_quat_yup, world_up="y"
    )
    return pred_dof_byd, pred_root_pos_yup, pred_root_quat_yup, pred_body_yup


@torch.no_grad()
def infer_stitched_sequence(
    pipeline: Any,
    denoiser: torch.nn.Module,
    sample: dict[str, Any],
    x_inits: torch.Tensor,
    chunks_meta: list[tuple[int, int]],
    device: torch.device,
    ddim_steps: int,
    ddim_eta: float,
    use_bf16: bool,
    window: int = 120,
    align_mode: str = "so3",
    blend_mode: str = "cosine",
    sync_mode: str = "none",
    sync_weight_mode: str = "cosine",
    inpaint_resamples: int = 1,
    resample_seed: int = 2026,
) -> dict[str, np.ndarray]:
    """Run batched chunk DDIM (optionally overlap-synchronized) + rigid stitching -> Z-up numpy."""
    from motion_lib.motion_utils.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix

    total_len = int(sample["length"])
    obs_full = select_coco17_no_nose_ears(normalize_kp2d(sample["kp2d"], sample["bbx_xys"])).float()
    valid_full = torch.as_tensor(sample["mask"]["valid"], dtype=torch.bool)
    obs_full[~valid_full] = 0.0
    bbx_full = sample["bbx_xys"].float()
    K_full = sample["K_fullimg"].float()
    angvel_full = sample["cam_angvel"].float()
    imgfeat_full = sample["f_imgseq"].float()
    imgfeat_mask_scalar = bool(sample["mask"]["f_imgseq"])
    root0_yup = sample["g1_target"]["g1_body_pos_w"][0:1, 0, :].float()

    obs_b, bbx_b, K_b, ang_b, feat_b, len_b = [], [], [], [], [], []
    for start, actual_len in chunks_meta:
        end = start + actual_len
        pad_len = window - actual_len
        obs_b.append(_pad_window(obs_full[start:end], pad_len))
        bbx_b.append(_pad_window(bbx_full[start:end], pad_len))
        K_b.append(_pad_window(K_full[start:end], pad_len))
        ang_b.append(_pad_window(angvel_full[start:end], pad_len))
        feat_b.append(_pad_window(imgfeat_full[start:end], pad_len))
        len_b.append(actual_len)

    n_chunks = len(chunks_meta)
    batch_gpu = {
        "obs": torch.stack(obs_b, dim=0).to(device),
        "bbx_xys": torch.stack(bbx_b, dim=0).to(device),
        "K_fullimg": torch.stack(K_b, dim=0).to(device),
        "cam_angvel": torch.stack(ang_b, dim=0).to(device),
        "f_imgseq": torch.stack(feat_b, dim=0).to(device),
        "f_imgseq_mask": torch.full((n_chunks,), imgfeat_mask_scalar, dtype=torch.bool, device=device),
        "length": torch.tensor(len_b, dtype=torch.long, device=device),
    }
    f_cond, length_t, all_valid = build_condition(pipeline, batch_gpu, use_bf16=use_bf16)
    x0_chunks = sample_ddim_synchronized(
        pipeline=pipeline,
        denoiser=denoiser,
        f_cond=f_cond,
        length=length_t,
        x_init=x_inits,
        chunks_meta=chunks_meta,
        total_len=total_len,
        ddim_steps=ddim_steps,
        eta=ddim_eta,
        all_valid_length=all_valid,
        use_bf16=use_bf16,
        sync_mode=sync_mode,
        sync_weight_mode=sync_weight_mode,
        inpaint_resamples=inpaint_resamples,
        resample_seed=resample_seed,
    )
    root0_chunks = root0_yup.to(device).unsqueeze(0).expand(n_chunks, 1, 3)
    jp_c, rp_c, rq_c, bp_c = _decode_chunks_yup(
        pipeline, x0_chunks, batch_gpu["cam_angvel"], length_t, root0_chunks
    )

    # Measure raw inter-window joint discrepancy on overlap frames BEFORE post-hoc blending.
    pre_ov_errs: list[float] = []
    prev_end_chk = chunks_meta[0][0] + chunks_meta[0][1]
    for ci in range(1, n_chunks):
        s_c, al_c = chunks_meta[ci]
        p_len = chunks_meta[ci - 1][1]
        ov_c = prev_end_chk - s_c
        if ov_c > 0:
            diff_ov = (jp_c[ci, :ov_c] - jp_c[ci - 1, p_len - ov_c : p_len]).abs().mean().item()
            pre_ov_errs.append(float(diff_ov))
        prev_end_chk = s_c + al_c
    pre_blend_overlap_err = float(np.mean(pre_ov_errs)) if pre_ov_errs else 0.0

    def _fk_g1_single(jp: torch.Tensor, rp: torch.Tensor, rq: torch.Tensor) -> torch.Tensor:
        bp, _ = pipeline.endecoder.fk_g1(jp.unsqueeze(0), rp.unsqueeze(0), rq.unsqueeze(0), world_up="y")
        return bp[0]

    stitched = stitch_sequence_yup(
        jp_c=jp_c,
        rp_c=rp_c,
        rq_c=rq_c,
        bp_c=bp_c,
        chunks_meta=chunks_meta,
        total_len=total_len,
        align_mode=align_mode,
        blend_mode=blend_mode,
        fk_fn=_fk_g1_single,
    )
    jp_out = stitched["joint_pos"]
    rp_out = stitched["root_pos_w"]
    rq_out = stitched["root_quat_w"]
    bp_out = stitched["body_pos_w"]
    seam_indices = stitched["seam_indices"]
    overlap_spans = stitched["overlap_spans"]

    M_pos = torch.tensor(YUP_TO_ZUP_M, device=device, dtype=torch.float32)
    rp_z = rp_out @ M_pos
    bp_z = bp_out @ M_pos
    rq_z = F.normalize(matrix_to_quaternion(M_pos.T @ quaternion_to_matrix(rq_out)), dim=-1)

    jp_np = jp_out.cpu().numpy().astype(np.float32)
    diffs = np.mean(np.abs(np.diff(jp_np, axis=0)), axis=-1)  # (T-1,)
    seam_jumps: list[float] = []
    overlap_mean_steps: list[float] = []
    interior_mask = np.ones(len(diffs), dtype=bool)
    for s, e in overlap_spans:
        if e - s == 1:
            if 0 <= s < len(diffs):
                seam_jumps.append(float(diffs[s]))
                overlap_mean_steps.append(float(diffs[s]))
                interior_mask[s] = False
        else:
            lo = max(0, s - 1)
            hi = min(len(diffs), e)
            if hi > lo:
                zone = diffs[lo:hi]
                seam_jumps.append(float(np.max(zone)))
                overlap_mean_steps.append(float(np.mean(zone)))
                interior_mask[lo:hi] = False
    interior_step = float(np.mean(diffs[interior_mask])) if interior_mask.any() else 0.0
    seam_jump = float(np.mean(seam_jumps)) if seam_jumps else 0.0
    overlap_mean_step = float(np.mean(overlap_mean_steps)) if overlap_mean_steps else 0.0

    return {
        "joint_pos": jp_np,
        "root_pos_w": rp_z.cpu().numpy().astype(np.float32),
        "root_quat_w": rq_z.cpu().numpy().astype(np.float32),
        "body_pos_w": bp_z.cpu().numpy().astype(np.float32),
        "seam_indices": seam_indices,
        "overlap_spans": overlap_spans,
        "seam_joint_jump_rad": seam_jump,
        "overlap_mean_step_rad": overlap_mean_step,
        "pre_blend_overlap_err_rad": pre_blend_overlap_err,
        "interior_joint_step_rad": interior_step,
    }


def _extract_gt_zup(sample: dict[str, Any]) -> dict[str, np.ndarray]:
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

    L = int(sample["length"])
    g1 = sample["g1_target"]
    M_pos = torch.tensor(YUP_TO_ZUP_M, dtype=torch.float32)
    jp = g1["g1_joint_pos"][:L].float().numpy().astype(np.float32)
    bp_z = (g1["g1_body_pos_w"][:L].float() @ M_pos).numpy().astype(np.float32)
    rp_z = bp_z[:, 0, :].copy()
    rq_y = g1["g1_body_quat_w"][:L, 0, :].float()
    rq_z = F.normalize(matrix_to_quaternion(M_pos.T @ quaternion_to_matrix(rq_y)), dim=-1).numpy().astype(np.float32)
    return {"joint_pos": jp, "root_pos_w": rp_z, "root_quat_w": rq_z, "body_pos_w": bp_z}


def _seam_distance_50hz(
    fail_50: int,
    seam_indices: list[int],
    fps: float,
    control_hz: float = 50.0,
    overlap_spans: list[tuple[int, int]] | None = None,
) -> int | None:
    if fail_50 < 0:
        return None
    if overlap_spans:
        dists = []
        for s, e in overlap_spans:
            s50 = int(round(s * control_hz / fps))
            e50 = int(round(max(s, e - 1) * control_hz / fps))
            if s50 <= fail_50 <= e50:
                dists.append(0)
            else:
                dists.append(min(abs(fail_50 - s50), abs(fail_50 - e50)))
        return int(min(dists))
    if not seam_indices:
        return None
    seams_50 = [int(round(s * control_hz / fps)) for s in seam_indices]
    return int(min(abs(fail_50 - s50) for s50 in seams_50))


def _summarize(recs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(recs)
    if n == 0:
        return {"count": 0}
    out: dict[str, Any] = {"count": n}
    for tag in ("gt", "base", "ppo"):
        if tag == "ppo" and recs[0].get("ppo") is None:
            continue
        succ = [1.0 if r[tag]["is_success"] else 0.0 for r in recs]
        surv = [r[tag]["survival_ratio"] for r in recs]
        rerr = [r[tag]["exec_root_pos_err_m"] for r in recs]
        jmae = [r[tag]["exec_joint_mae_rad"] for r in recs]
        gt_pass = [r for r in recs if r["gt"]["is_success"]]
        near_seam = [
            r for r in recs
            if (not r[tag]["is_success"])
            and r[tag].get("fail_dist_to_seam_50hz") is not None
            and r[tag]["fail_dist_to_seam_50hz"] <= 25
        ]
        fails = [r for r in recs if not r[tag]["is_success"]]
        entry: dict[str, Any] = {
            "success_rate": float(np.mean(succ)),
            "fail_count": len(fails),
            "survival_ratio": float(np.mean(surv)),
            "exec_root_pos_err_m": float(np.nanmean(rerr)),
            "exec_joint_mae_rad": float(np.nanmean(jmae)),
            "fails_within_0p5s_of_seam": len(near_seam),
        }
        if gt_pass and tag != "gt":
            entry["success_given_gt_pass"] = float(np.mean([1.0 if r[tag]["is_success"] else 0.0 for r in gt_pass]))
        if tag != "gt":
            entry["ref_joint_mae_rad"] = float(np.nanmean([r[tag]["ref_joint_mae_rad"] for r in recs]))
            entry["ref_body_mpjpe_m"] = float(np.nanmean([r[tag]["ref_body_mpjpe_m"] for r in recs]))
            entry["seam_joint_jump_rad"] = float(np.mean([r[tag]["seam_joint_jump_rad"] for r in recs]))
            entry["overlap_mean_step_rad"] = float(np.mean([r[tag].get("overlap_mean_step_rad", r[tag]["seam_joint_jump_rad"]) for r in recs]))
            entry["pre_blend_overlap_err_rad"] = float(np.mean([r[tag].get("pre_blend_overlap_err_rad", 0.0) for r in recs]))
            entry["interior_joint_step_rad"] = float(np.mean([r[tag]["interior_joint_step_rad"] for r in recs]))
            entry["mean_num_chunks"] = float(np.mean([r["num_chunks"] for r in recs]))
        out[tag] = entry

    if recs[0].get("ppo") is not None:
        bp = sum(1 for r in recs if r["base"]["is_success"] and r["ppo"]["is_success"])
        pf = sum(1 for r in recs if (not r["base"]["is_success"]) and r["ppo"]["is_success"])
        pr = sum(1 for r in recs if r["base"]["is_success"] and (not r["ppo"]["is_success"]))
        bf = sum(1 for r in recs if (not r["base"]["is_success"]) and (not r["ppo"]["is_success"]))
        out["paired_base_vs_ppo"] = {
            "both_pass": bp,
            "ppo_fixed_base_fail": pf,
            "ppo_regressed_base_pass": pr,
            "both_fail": bf,
            "mcnemar_p_value": _exact_mcnemar_p(pr, pf),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", default=["bedlam2", "h36m"])
    parser.add_argument("--split", type=str, default="test", choices=("val", "test"))
    parser.add_argument("--g1-pth-name", type=str, default="g1_new.pth")
    parser.add_argument("--base-ckpt", type=Path, default=REPO_ROOT / "vith_offline_g1.ckpt")
    parser.add_argument("--ppo-ckpt", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=240)
    parser.add_argument("--max-seqs-per-source", type=int, default=0)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--overlap", type=int, default=1, help="Number of overlap frames between adjacent 120f windows (1=hard stitch, >1=blended)")
    parser.add_argument("--align-mode", type=str, default="so3", choices=("so3", "yaw"))
    parser.add_argument("--blend-mode", type=str, default="cosine", choices=("cosine", "linear"))
    parser.add_argument("--sync-mode", type=str, default="none", choices=("none", "multidiffusion", "inpainting"))
    parser.add_argument("--sync-weight-mode", type=str, default="cosine", choices=("cosine", "uniform"))
    parser.add_argument("--inpaint-resamples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tracker-python", type=Path, default=Path(os.environ.get("TRACKER_PYTHON_BIN", sys.executable)))
    parser.add_argument("--policy-onnx", type=Path, default=REPO_ROOT / "inputs/checkpoints/g1_tracker/model_step_020000_g1.onnx")
    parser.add_argument("--xml-path", type=Path, default=REPO_ROOT / "g1_tracker/assets/g1/rollout/scene_29dof_mujoco.xml")
    parser.add_argument("--grounding-xml", type=Path, default=REPO_ROOT / "g1_tracker/assets/g1/grounding/mjcf/g1_29dof_rev_1_0.xml")
    parser.add_argument("--tracker-workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda"
    torch.set_float32_matmul_precision("high")

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = ["network.latent_dim=1024", "network.num_layers=16", "network.num_heads=16"]
    base_pipe = load_pipeline(str(args.base_ckpt), exp="offline/g1_training_set_exp1", device=str(device), overrides=overrides).eval()
    ppo_pipe = (
        load_pipeline(str(args.ppo_ckpt), exp="offline/g1_training_set_exp1", device=str(device), overrides=overrides).eval()
        if args.ppo_ckpt is not None
        else None
    )

    root = Path(args.dataset_root).expanduser().resolve()
    all_records: list[dict[str, Any]] = []

    with PersistentTrackerPool(
        tracker_python=args.tracker_python,
        policy_onnx=args.policy_onnx,
        xml_path=args.xml_path,
        grounding_xml=args.grounding_xml,
        num_workers=args.tracker_workers,
    ) as pool:
        for src in args.sources:
            seq_jobs: list[tuple[str, Any, int]] = []
            for spec in VIDEO_SOURCE_SPECS[src]:
                data_dir = root / spec["subdir"]
                kwargs: dict[str, Any] = {
                    "data_dir": str(data_dir),
                    "smplx_pth_name": "motion.pth",
                    "g1_pth_name": args.g1_pth_name,
                    "bbox_pth_paths": [str(data_dir / p) for p in spec["bbox"]],
                    "vitpose_pth_paths": [str(data_dir / p) for p in spec["kp2d"]],
                    "imgfeat_pth_paths": [str(data_dir / p) for p in spec["imgfeat"]],
                    "motion_frames": 120,
                    "min_motion_frames": int(args.min_frames),
                    "g1_input_is_yup": False,
                    "dof_pos_order": "mjc",
                    "canonicalize_first_frame": True,
                    "floor_adjust": True,
                    "yaw_only_canon": True,
                    "split": args.split,
                    "full_sequence": True,
                    "split_seed": 42,
                    "split_ratios": (0.9, 0.05, 0.05),
                }
                kwargs.update(spec.get("extra", {}))
                ds = spec["cls"](**kwargs)
                for i in range(len(ds)):
                    seq_jobs.append((spec["subdir"], ds, i))

            if args.max_seqs_per_source > 0 and len(seq_jobs) > args.max_seqs_per_source:
                rng = np.random.default_rng(args.seed)
                pick = np.sort(rng.choice(len(seq_jobs), size=args.max_seqs_per_source, replace=False))
                seq_jobs = [seq_jobs[int(i)] for i in pick]

            print(f"[LongEval:{src}] {len(seq_jobs)} full sequences (>= {args.min_frames} frames)", flush=True)

            # Process in batches of 16 sequences so tracker pool stays 100% saturated!
            for b_start in range(0, len(seq_jobs), 16):
                chunk_jobs = seq_jobs[b_start : b_start + 16]
                metas: list[dict[str, Any]] = []
                gt_list, base_list, ppo_list = [], [], []
                for j_off, (subdir, ds, idx) in enumerate(chunk_jobs):
                    sample = ds[idx]
                    L = int(sample["length"])
                    fps = float(sample["fps"])
                    chunks_meta = _chunk_windows(L, window=120, overlap=args.overlap)
                    seq_seed = args.seed + (b_start + j_off) * 97
                    gen = torch.Generator(device=device).manual_seed(seq_seed)
                    x_inits = torch.randn((len(chunks_meta), 120, G1_PRED_X_DIM), device=device, dtype=torch.float32, generator=gen)

                    gt_z = _extract_gt_zup(sample)
                    base_z = infer_stitched_sequence(
                        base_pipe, base_pipe.denoiser3d, sample, x_inits, chunks_meta, device,
                        args.ddim_steps, args.ddim_eta, use_bf16,
                        align_mode=args.align_mode, blend_mode=args.blend_mode,
                        sync_mode=args.sync_mode, sync_weight_mode=args.sync_weight_mode,
                        inpaint_resamples=args.inpaint_resamples, resample_seed=seq_seed + 50000,
                    )
                    ppo_z = (
                        infer_stitched_sequence(
                            ppo_pipe, ppo_pipe.denoiser3d, sample, x_inits, chunks_meta, device,
                            args.ddim_steps, args.ddim_eta, use_bf16,
                            align_mode=args.align_mode, blend_mode=args.blend_mode,
                            sync_mode=args.sync_mode, sync_weight_mode=args.sync_weight_mode,
                            inpaint_resamples=args.inpaint_resamples, resample_seed=seq_seed + 50000,
                        )
                        if ppo_pipe is not None
                        else None
                    )
                    gt_list.append(gt_z)
                    base_list.append(base_z)
                    ppo_list.append(ppo_z)
                    metas.append({
                        "source": src,
                        "part_subdir": subdir,
                        "clip_key": str(sample["meta"]["key"]),
                        "length": L,
                        "fps": fps,
                        "duration_s": round(L / max(fps, 1e-3), 2),
                        "num_chunks": len(chunks_meta),
                        "seam_indices": base_z["seam_indices"],
                        "overlap_spans": base_z["overlap_spans"],
                    })

                # Submit all variable-length sequences in parallel using padded batch arrays.
                def eval_group(items: list[dict[str, np.ndarray]]) -> list[dict[str, Any]]:
                    max_l = max(x["joint_pos"].shape[0] for x in items)
                    S = len(items)
                    jp = np.zeros((S, max_l, 29), dtype=np.float32)
                    rp = np.zeros((S, max_l, 3), dtype=np.float32)
                    rq = np.zeros((S, max_l, 4), dtype=np.float32)
                    bp = np.zeros((S, max_l, 30, 3), dtype=np.float32)
                    lens = np.zeros(S, dtype=np.int64)
                    fps_arr = np.array([m["fps"] for m in metas], dtype=np.float32)
                    for i, it in enumerate(items):
                        l_i = it["joint_pos"].shape[0]
                        jp[i, :l_i] = it["joint_pos"]
                        rp[i, :l_i] = it["root_pos_w"]
                        rq[i, :l_i] = it["root_quat_w"]
                        bp[i, :l_i] = it["body_pos_w"]
                        lens[i] = l_i
                    return pool.evaluate_batch(jp, rp, rq, bp, lens, fps=fps_arr)

                trk_gt = eval_group(gt_list)
                trk_base = eval_group(base_list)
                trk_ppo = eval_group(ppo_list) if ppo_pipe is not None else None

                for i, m in enumerate(metas):
                    def pack(trk_r: dict[str, Any], pred_d: dict[str, np.ndarray], gt_d: dict[str, np.ndarray] | None) -> dict[str, Any]:
                        tot50 = max(1, int(trk_r["total_frames_50hz"]))
                        succ = bool(trk_r["is_success"])
                        fail = int(trk_r["first_failure_frame_50hz"])
                        d_out: dict[str, Any] = {
                            "valid": bool(trk_r["valid"]),
                            "is_success": succ,
                            "first_failure_frame_50hz": fail,
                            "total_frames_50hz": tot50,
                            "survival_ratio": 1.0 if succ else float(np.clip(fail / tot50, 0.0, 1.0)),
                            "exec_joint_mae_rad": float(trk_r["exec_joint_mae_rad"]),
                            "exec_root_pos_err_m": float(trk_r["exec_root_pos_err_m"]),
                            "fail_dist_to_seam_50hz": _seam_distance_50hz(
                                fail, m["seam_indices"], m["fps"], overlap_spans=m.get("overlap_spans")
                            ),
                        }
                        if gt_d is not None:
                            jmae = float(np.mean(np.abs(pred_d["joint_pos"] - gt_d["joint_pos"])))
                            p_rel = pred_d["body_pos_w"] - pred_d["body_pos_w"][:, 0:1, :]
                            g_rel = gt_d["body_pos_w"] - gt_d["body_pos_w"][:, 0:1, :]
                            mpjpe = float(np.mean(np.linalg.norm(p_rel - g_rel, axis=-1)))
                            d_out["ref_joint_mae_rad"] = jmae
                            d_out["ref_body_mpjpe_m"] = mpjpe
                            d_out["seam_joint_jump_rad"] = pred_d["seam_joint_jump_rad"]
                            d_out["overlap_mean_step_rad"] = pred_d["overlap_mean_step_rad"]
                            d_out["pre_blend_overlap_err_rad"] = pred_d["pre_blend_overlap_err_rad"]
                            d_out["interior_joint_step_rad"] = pred_d["interior_joint_step_rad"]
                        return d_out

                    rec = {
                        **m,
                        "gt": pack(trk_gt[i], gt_list[i], None),
                        "base": pack(trk_base[i], base_list[i], gt_list[i]),
                        "ppo": pack(trk_ppo[i], ppo_list[i], gt_list[i]) if trk_ppo is not None else None,
                    }
                    all_records.append(rec)

                print(f"[LongEval:{src}] {min(b_start+16, len(seq_jobs))}/{len(seq_jobs)} elapsed={time.perf_counter()-t0:.1f}s", flush=True)

    # Stratify by length bucket as well as source!
    len_buckets = {
        "240_to_359f (8-12s)": [r for r in all_records if 240 <= r["length"] < 360],
        "360_to_599f (12-20s)": [r for r in all_records if 360 <= r["length"] < 600],
        "600_plus_f (20-76s)": [r for r in all_records if r["length"] >= 600],
    }
    summary = {
        "config": {
            "split": args.split,
            "sources": args.sources,
            "min_frames": args.min_frames,
            "ddim_steps": args.ddim_steps,
            "ddim_eta": args.ddim_eta,
            "overlap": args.overlap,
            "align_mode": args.align_mode,
            "blend_mode": args.blend_mode,
            "sync_mode": args.sync_mode,
            "sync_weight_mode": args.sync_weight_mode,
            "inpaint_resamples": args.inpaint_resamples,
            "base_ckpt": str(args.base_ckpt.resolve()),
            "ppo_ckpt": str(args.ppo_ckpt.resolve()) if args.ppo_ckpt else None,
            "elapsed_s": round(time.perf_counter() - t0, 2),
        },
        "overall": _summarize(all_records),
        "by_source": {s: _summarize([r for r in all_records if r["source"] == s]) for s in args.sources},
        "by_length": {k: _summarize(v) for k, v in len_buckets.items() if v},
    }
    (out_dir / "sequences.jsonl").write_text("\n".join(json.dumps(r) for r in all_records) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== LONG-HORIZON STITCHED SUMMARY ===")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
