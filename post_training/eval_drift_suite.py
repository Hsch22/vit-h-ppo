#!/usr/bin/env python3
"""Dual-protocol kinematic + physical long-horizon drift evaluation suite.

Evaluates ViT-H base and candidate PPO checkpoints (plus optional 189-D subspace
surgery probes) under identical initial noise and MultiDiffusion stitching,
reporting both:
  1. Kinematic reference quality (mirroring EMDB/RICH origin_metrics):
     PA-MPJPE, WA-MPJPE, W-MPJPE, local-MPJPE, RTE, Jitter, kinematic FS,
     plus vertical drift diagnostics Zmax and FootZmax.
  2. Physical MuJoCo execution (mirroring R11_core_metrics / Paper Table 2):
     250-frame segmented tracking success (Seg250-Succ), full-sequence
     tracking success (Long-Succ), physical Foot slip (cm/s), executed
     Long MPJPE (mm), root Z error, and exact failure-mode breakdown
     (anchor_height / ee_height / anchor_ori).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_training.dataset import VIDEO_SOURCE_SPECS
from post_training.eval_compare import _exact_mcnemar_p, build_condition
from post_training.eval_long_horizon import (
    _chunk_windows,
    _decode_chunks_yup,
    _extract_gt_zup,
    _pad_window,
    normalize_kp2d,
    select_coco17_no_nose_ears,
)
from post_training.policy import G1_PRED_X_DIM, YUP_TO_ZUP_M
from post_training.stitching import sample_ddim_synchronized, stitch_sequence_yup
from post_training.tracker_pool import PersistentTrackerPool
from tools.inference_utils import load_pipeline

FOOT_INDICES = (18, 19)  # left_ankle_roll_link, right_ankle_roll_link


def _kabsch_align_3d(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigidly align (N, 3) src points to dst points via Kabsch SVD."""
    src_mean = src.mean(axis=0, keepdims=True)
    dst_mean = dst.mean(axis=0, keepdims=True)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = src_c.T @ dst_c
    u, _, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    return (src_c @ rot.T) + dst_mean


def _align_first_frame_yaw_and_pos(pred_bp: np.ndarray, gt_bp: np.ndarray) -> np.ndarray:
    """Align frame-0 heading (yaw in XY plane) and root XYZ to GT frame 0."""
    p0 = pred_bp[0] - pred_bp[0, 0:1]
    g0 = gt_bp[0] - gt_bp[0, 0:1]
    c = float(np.sum(p0[:, 0] * g0[:, 0] + p0[:, 1] * g0[:, 1]))
    s = float(np.sum(p0[:, 0] * g0[:, 1] - p0[:, 1] * g0[:, 0]))
    ang = float(np.arctan2(s, c))
    ca, sa = np.cos(ang), np.sin(ang)
    rot = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (pred_bp - pred_bp[0:1, 0:1]) @ rot.T + gt_bp[0:1, 0:1]


def compute_kinematic_metrics(
    pred_bp_z: np.ndarray,
    gt_bp_z: np.ndarray,
    fps: float,
) -> dict[str, float]:
    """Compute EMDB/RICH-style kinematic quality + vertical Z-drift metrics."""
    L = int(pred_bp_z.shape[0])
    bp = _align_first_frame_yaw_and_pos(pred_bp_z.astype(np.float32), gt_bp_z.astype(np.float32))
    gt_bp = gt_bp_z.astype(np.float32)
    rp = bp[:, 0]
    gt_rp = gt_bp[:, 0]

    # 1. local-MPJPE (root-relative, mm)
    local_mpjpe = float(np.mean(np.linalg.norm((bp - rp[:, None]) - (gt_bp - gt_rp[:, None]), axis=-1)) * 1000.0)

    # 2. PA-MPJPE (frame-wise rigid Procrustes, mm)
    pa_errs = []
    stride = 1 if L <= 600 else 2
    for t in range(0, L, stride):
        aligned_t = _kabsch_align_3d(bp[t], gt_bp[t])
        pa_errs.append(float(np.mean(np.linalg.norm(aligned_t - gt_bp[t], axis=-1)) * 1000.0))
    pa_mpjpe = float(np.mean(pa_errs))

    # 3. WA-MPJPE (100-frame chunk rigid-aligned, mm)
    wa_errs = []
    for s0 in range(0, L, 100):
        e0 = min(L, s0 + 100)
        if e0 - s0 < 10:
            continue
        pc = bp[s0:e0].reshape(-1, 3)
        gc = gt_bp[s0:e0].reshape(-1, 3)
        aligned_c = _kabsch_align_3d(pc, gc)
        wa_errs.append(float(np.mean(np.linalg.norm(aligned_c - gc, axis=-1)) * 1000.0))
    wa_mpjpe = float(np.mean(wa_errs)) if wa_errs else float("nan")

    # 4. W-MPJPE (first-frame yaw+pos aligned world MPJPE, mm)
    w_mpjpe = float(np.mean(np.linalg.norm(bp - gt_bp, axis=-1)) * 1000.0)

    # 5. RTE (% of GT root path length)
    gt_path = float(np.sum(np.linalg.norm(np.diff(gt_rp, axis=0), axis=-1)))
    rte_pct = float(np.mean(np.linalg.norm(rp - gt_rp, axis=-1)) / max(gt_path, 1.0) * 100.0)

    # 6. Jitter (WHAM jerk / 10)
    if L >= 4:
        acc3 = np.diff(bp, n=3, axis=0) * (float(fps) ** 3)
        jitter = float(np.mean(np.linalg.norm(acc3, axis=-1)) / 10.0)
    else:
        jitter = 0.0

    # 7. Kinematic FS (foot horizontal displacement mm/frame on stance frames)
    f_pos = bp[:, list(FOOT_INDICES), :]
    f_vel_xy_mm = np.linalg.norm(np.diff(f_pos[:, :, :2], axis=0), axis=-1) * 1000.0
    f_h = f_pos[:-1, :, 2]
    contact = f_h < (float(np.percentile(f_h, 15)) + 0.03)
    fs_mm = float(np.mean(f_vel_xy_mm[contact])) if bool(contact.any()) else 0.0

    # 8. Vertical drift diagnostics (mm)
    z_diff_mm = (rp[:, 2] - gt_rp[:, 2]) * 1000.0
    z_abs_mm = float(np.mean(np.abs(z_diff_mm)))
    z_max_mm = float(np.max(np.abs(z_diff_mm)))
    foot_z_max_mm = float(np.max(np.abs(bp[:, :, 2].min(axis=1) - gt_bp[:, :, 2].min(axis=1))) * 1000.0)

    return {
        "pa_mpjpe_mm": pa_mpjpe,
        "wa_mpjpe_mm": wa_mpjpe,
        "w_mpjpe_mm": w_mpjpe,
        "local_mpjpe_mm": local_mpjpe,
        "rte_pct": rte_pct,
        "jitter": jitter,
        "fs_mm": fs_mm,
        "z_abs_mm": z_abs_mm,
        "z_max_mm": z_max_mm,
        "foot_z_max_mm": foot_z_max_mm,
    }


def _decode_and_stitch_x0(
    pipeline: Any,
    x0_chunks: torch.Tensor,
    cam_angvel: torch.Tensor,
    length_t: torch.Tensor,
    chunks_meta: list[tuple[int, int]],
    total_len: int,
    root0_yup: torch.Tensor,
    device: torch.device,
    align_mode: str = "yaw",
    blend_mode: str = "cosine",
) -> dict[str, np.ndarray]:
    from motion_lib.motion_utils.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix

    n_chunks = len(chunks_meta)
    root0_c = root0_yup.to(device=device, dtype=torch.float32).unsqueeze(0).expand(n_chunks, 1, 3)
    jp_c, rp_c, rq_c, bp_c = _decode_chunks_yup(pipeline, x0_chunks, cam_angvel, length_t, root0_c)

    def fk_fn(jp: torch.Tensor, rp: torch.Tensor, rq: torch.Tensor) -> torch.Tensor:
        bp, _ = pipeline.endecoder.fk_g1(jp.unsqueeze(0), rp.unsqueeze(0), rq.unsqueeze(0), world_up="y")
        return bp[0]

    st = stitch_sequence_yup(
        jp_c=jp_c,
        rp_c=rp_c,
        rq_c=rq_c,
        bp_c=bp_c,
        chunks_meta=chunks_meta,
        total_len=total_len,
        align_mode=align_mode,
        blend_mode=blend_mode,
        fk_fn=fk_fn,
    )
    M_pos = torch.tensor(YUP_TO_ZUP_M, device=device, dtype=torch.float32)
    rp_z = st["root_pos_w"] @ M_pos
    bp_z = st["body_pos_w"] @ M_pos
    rq_z = matrix_to_quaternion(M_pos.transpose(-1, -2) @ quaternion_to_matrix(st["root_quat_w"]))
    rq_z = torch.nn.functional.normalize(rq_z, dim=-1)
    return {
        "joint_pos": st["joint_pos"].cpu().numpy().astype(np.float32),
        "root_pos_w": rp_z.cpu().numpy().astype(np.float32),
        "root_quat_w": rq_z.cpu().numpy().astype(np.float32),
        "body_pos_w": bp_z.cpu().numpy().astype(np.float32),
    }


def _slice_segments(pred: dict[str, np.ndarray], seg_len: int = 250) -> list[dict[str, np.ndarray]]:
    """Split a full sequence into non-overlapping seg_len-frame segments (mirroring Paper Table 2)."""
    L = int(pred["joint_pos"].shape[0])
    segs: list[dict[str, np.ndarray]] = []
    if L <= seg_len:
        return [pred]
    for s0 in range(0, L, seg_len):
        e0 = min(L, s0 + seg_len)
        if e0 - s0 < 60 and segs:
            continue
        segs.append({k: v[s0:e0] for k, v in pred.items()})
    return segs


def _eval_batch_in_tracker(
    pool: PersistentTrackerPool,
    items: list[tuple[dict[str, np.ndarray], float]],
) -> list[dict[str, Any]]:
    if not items:
        return []
    max_t = max(int(d["joint_pos"].shape[0]) for d, _ in items)
    n = len(items)
    jp = np.zeros((n, max_t, 29), dtype=np.float32)
    rp = np.zeros((n, max_t, 3), dtype=np.float32)
    rq = np.zeros((n, max_t, 4), dtype=np.float32)
    rq[..., 0] = 1.0
    bp = np.zeros((n, max_t, 30, 3), dtype=np.float32)
    lens = np.zeros((n,), dtype=np.int64)
    fps_arr = np.zeros((n,), dtype=np.float32)
    for i, (d, fps_i) in enumerate(items):
        L = int(d["joint_pos"].shape[0])
        jp[i, :L] = d["joint_pos"]
        rp[i, :L] = d["root_pos_w"]
        rq[i, :L] = d["root_quat_w"]
        bp[i, :L] = d["body_pos_w"]
        lens[i] = L
        fps_arr[i] = float(fps_i)
    return pool.evaluate_batch(jp, rp, rq, bp, lens, fps=fps_arr)


def _summarize_model(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    n = len(records)
    seg_total = sum(int(r["seg250_total"]) for r in records)
    seg_pass = sum(int(r["seg250_pass"]) for r in records)
    long_pass = sum(1 for r in records if r["long_is_success"])

    def _mean(key: str) -> float:
        vals = [float(r[key]) for r in records if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "count": n,
        "seg250_total": seg_total,
        "seg250_pass": seg_pass,
        "seg250_succ_pct": float(100.0 * seg_pass / max(1, seg_total)),
        "long_pass": long_pass,
        "long_succ_pct": float(100.0 * long_pass / n),
        "survival_ratio": _mean("long_survival_ratio"),
        "exec_foot_slip_cm_s": _mean("exec_foot_slip_cm_s"),
        "exec_root_pos_err_m": _mean("exec_root_pos_err_m"),
        "exec_root_z_err_m": _mean("exec_root_z_err_m"),
        "exec_body_mpjpe_mm": _mean("exec_body_mpjpe_mm"),
        "exec_joint_mae_rad": _mean("exec_joint_mae_rad"),
        "n_fail_anchor_height": sum(1 for r in records if r["failed_anchor_height"]),
        "n_fail_ee_height": sum(1 for r in records if r["failed_end_effector_height"]),
        "n_fail_anchor_ori": sum(1 for r in records if r["failed_anchor_orientation"]),
        "pa_mpjpe_mm": _mean("pa_mpjpe_mm"),
        "wa_mpjpe_mm": _mean("wa_mpjpe_mm"),
        "w_mpjpe_mm": _mean("w_mpjpe_mm"),
        "local_mpjpe_mm": _mean("local_mpjpe_mm"),
        "rte_pct": _mean("rte_pct"),
        "jitter": _mean("jitter"),
        "fs_mm": _mean("fs_mm"),
        "z_abs_mm": _mean("z_abs_mm"),
        "z_max_mm": _mean("z_max_mm"),
        "foot_z_max_mm": _mean("foot_z_max_mm"),
    }


def _paired_summary(base_recs: list[dict[str, Any]], cand_recs: list[dict[str, Any]]) -> dict[str, Any]:
    l_fixed = sum(1 for b, c in zip(base_recs, cand_recs) if (not b["long_is_success"]) and c["long_is_success"])
    l_reg = sum(1 for b, c in zip(base_recs, cand_recs) if b["long_is_success"] and (not c["long_is_success"]))
    s_fixed = sum(
        1
        for b, c in zip(base_recs, cand_recs)
        for bs, cs in zip(b["seg250_flags"], c["seg250_flags"])
        if (not bs) and cs
    )
    s_reg = sum(
        1
        for b, c in zip(base_recs, cand_recs)
        for bs, cs in zip(b["seg250_flags"], c["seg250_flags"])
        if bs and (not cs)
    )
    return {
        "long_fixed": l_fixed,
        "long_regressed": l_reg,
        "long_mcnemar_p": _exact_mcnemar_p(l_fixed, l_reg),
        "seg250_fixed": s_fixed,
        "seg250_regressed": s_reg,
        "seg250_mcnemar_p": _exact_mcnemar_p(s_fixed, s_reg),
        "delta_z_max_mm": float(np.mean([c["z_max_mm"] - b["z_max_mm"] for b, c in zip(base_recs, cand_recs)])),
        "delta_w_mpjpe_mm": float(np.mean([c["w_mpjpe_mm"] - b["w_mpjpe_mm"] for b, c in zip(base_recs, cand_recs)])),
        "delta_rte_pct": float(np.mean([c["rte_pct"] - b["rte_pct"] for b, c in zip(base_recs, cand_recs)])),
        "delta_exec_foot_slip_cm_s": float(
            np.mean([c["exec_foot_slip_cm_s"] - b["exec_foot_slip_cm_s"] for b, c in zip(base_recs, cand_recs)])
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--split", type=str, default="val", choices=("val", "test"))
    p.add_argument("--sources", nargs="+", default=["h36m", "bedlam2"])
    p.add_argument("--g1-pth-name", type=str, default="g1_new.pth")
    p.add_argument("--min-frames", type=int, default=240)
    p.add_argument("--max-bedlam2", type=int, default=0, help="Cap bedlam2 count (0 = all; h36m is always kept full)")
    p.add_argument("--base-ckpt", type=Path, required=True)
    p.add_argument(
        "--cand",
        action="append",
        default=[],
        help="Candidate spec `name=path`. May be passed multiple times.",
    )
    p.add_argument(
        "--surgery-from",
        type=str,
        default=None,
        help="Optional candidate name to also evaluate with Base[180:189] subspace surgery (`<name>_base180_189`).",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--overlap", type=int, default=16)
    p.add_argument("--align-mode", type=str, default="yaw", choices=("yaw", "so3"))
    p.add_argument("--blend-mode", type=str, default="cosine", choices=("cosine", "linear"))
    p.add_argument("--sync-mode", type=str, default="multidiffusion", choices=("none", "multidiffusion"))
    p.add_argument("--seg-len", type=int, default=250)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--tracker-python", type=Path, required=True)
    p.add_argument("--policy-onnx", type=Path, default=REPO_ROOT / "inputs/checkpoints/g1_tracker/model_step_020000_g1.onnx")
    p.add_argument("--xml-path", type=Path, default=REPO_ROOT / "g1_tracker/assets/g1/rollout/scene_29dof_mujoco.xml")
    p.add_argument("--grounding-xml", type=Path, default=REPO_ROOT / "g1_tracker/assets/g1/grounding/mjcf/g1_29dof_rev_1_0.xml")
    p.add_argument("--tracker-workers", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cand_specs: list[tuple[str, Path]] = []
    for raw in args.cand:
        if "=" not in raw:
            raise ValueError(f"--cand must be `name=path`, got {raw!r}")
        name, path_str = raw.split("=", 1)
        cand_specs.append((name.strip(), Path(path_str.strip())))

    # Load pipelines
    pipes: dict[str, Any] = {}
    pipes["base"] = load_pipeline(str(args.base_ckpt), exp="offline/g1_training_set_exp1", device=str(device))
    pipes["base"].eval()
    for name, cpath in cand_specs:
        pipes[name] = load_pipeline(str(cpath), exp="offline/g1_training_set_exp1", device=str(device))
        pipes[name].eval()

    model_names = ["base"] + [n for n, _ in cand_specs]
    surgery_name: str | None = None
    if args.surgery_from is not None:
        if args.surgery_from not in pipes:
            raise ValueError(f"--surgery-from={args.surgery_from!r} not in candidates {list(pipes)}")
        surgery_name = f"{args.surgery_from}_base180_189"
        model_names.append(surgery_name)

    # Load datasets
    jobs: list[tuple[str, Any, int]] = []
    for src in args.sources:
        src_jobs: list[tuple[str, Any, int]] = []
        for spec in VIDEO_SOURCE_SPECS[src]:
            data_dir = args.dataset_root / spec["subdir"]
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
                src_jobs.append((src, ds, i))
        if src == "bedlam2" and args.max_bedlam2 > 0 and len(src_jobs) > args.max_bedlam2:
            # Always keep known regression keywords + deterministic uniform subset
            must_keep = []
            rest = []
            for j_i, (s_n, ds, idx) in enumerate(src_jobs):
                key_s = str(ds.idx2meta[idx]) if hasattr(ds, "idx2meta") else ""
                if any(tok in key_s for tok in ("yogastudio", "row_2976", "row_298", "row_826")):
                    must_keep.append((s_n, ds, idx))
                else:
                    rest.append((s_n, ds, idx))
            need = max(0, args.max_bedlam2 - len(must_keep))
            rng = np.random.default_rng(args.seed)
            pick = np.sort(rng.choice(len(rest), size=min(need, len(rest)), replace=False))
            src_jobs = must_keep + [rest[int(k)] for k in pick]
        print(f"[DriftSuite:{args.split}:{src}] selected {len(src_jobs)} sequences (>= {args.min_frames}f)", flush=True)
        jobs.extend(src_jobs)

    records_by_model: dict[str, list[dict[str, Any]]] = {m: [] for m in model_names}
    records_by_src_model: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {m: [] for m in model_names})
    seq_log_path = args.output_dir / "sequences.jsonl"

    with PersistentTrackerPool(
        tracker_python=args.tracker_python,
        policy_onnx=args.policy_onnx,
        xml_path=args.xml_path,
        grounding_xml=args.grounding_xml,
        num_workers=args.tracker_workers,
    ) as pool, open(seq_log_path, "w") as seq_f:
        for b0 in range(0, len(jobs), 8):
            chunk = jobs[b0 : b0 + 8]
            batch_preds: list[dict[str, dict[str, np.ndarray]]] = []
            batch_metas: list[dict[str, Any]] = []
            batch_gts: list[dict[str, np.ndarray]] = []

            for j_off, (src, ds, idx) in enumerate(chunk):
                sample = ds[idx]
                L = int(sample["length"])
                fps = float(sample["fps"])
                clip_key = str(ds.idx2meta[idx].get("key", idx)) if hasattr(ds, "idx2meta") and isinstance(ds.idx2meta[idx], dict) else str(idx)
                chunks_meta = _chunk_windows(L, window=120, overlap=args.overlap)

                obs_full = select_coco17_no_nose_ears(normalize_kp2d(sample["kp2d"], sample["bbx_xys"])).float()
                valid_full = torch.as_tensor(sample["mask"]["valid"], dtype=torch.bool)
                obs_full[~valid_full] = 0.0
                obs_b, bbx_b, K_b, ang_b, feat_b, len_b = [], [], [], [], [], []
                for start, alen in chunks_meta:
                    end = start + alen
                    pl = 120 - alen
                    obs_b.append(_pad_window(obs_full[start:end], pl))
                    bbx_b.append(_pad_window(sample["bbx_xys"].float()[start:end], pl))
                    K_b.append(_pad_window(sample["K_fullimg"].float()[start:end], pl))
                    ang_b.append(_pad_window(sample["cam_angvel"].float()[start:end], pl))
                    feat_b.append(_pad_window(sample["f_imgseq"].float()[start:end], pl))
                    len_b.append(alen)
                batch_gpu = {
                    "obs": torch.stack(obs_b).to(device),
                    "bbx_xys": torch.stack(bbx_b).to(device),
                    "K_fullimg": torch.stack(K_b).to(device),
                    "cam_angvel": torch.stack(ang_b).to(device),
                    "f_imgseq": torch.stack(feat_b).to(device),
                    "f_imgseq_mask": torch.full((len(chunks_meta),), bool(sample["mask"]["f_imgseq"]), dtype=torch.bool, device=device),
                    "length": torch.tensor(len_b, dtype=torch.long, device=device),
                }
                f_cond, length_t, all_valid = build_condition(pipes["base"], batch_gpu, use_bf16=True)
                seq_seed = args.seed + (b0 + j_off) * 97
                gen = torch.Generator(device=device).manual_seed(seq_seed)
                x_inits = torch.randn((len(chunks_meta), 120, G1_PRED_X_DIM), device=device, dtype=torch.float32, generator=gen)

                x0_map: dict[str, torch.Tensor] = {}
                for nm, pipe in pipes.items():
                    x0_map[nm] = sample_ddim_synchronized(
                        pipe,
                        pipe.denoiser3d,
                        f_cond,
                        length_t,
                        x_inits,
                        chunks_meta,
                        L,
                        args.ddim_steps,
                        0.0,
                        all_valid,
                        True,
                        args.sync_mode,
                        "cosine" if args.blend_mode == "cosine" else "uniform",
                    )
                if surgery_name is not None:
                    x0_map[surgery_name] = torch.cat(
                        [x0_map[args.surgery_from][..., :180], x0_map["base"][..., 180:189]],
                        dim=-1,
                    )

                root0_yup = sample["g1_target"]["g1_body_pos_w"][0:1, 0, :].float()
                preds_one: dict[str, dict[str, np.ndarray]] = {}
                for nm in model_names:
                    preds_one[nm] = _decode_and_stitch_x0(
                        pipes["base"],
                        x0_map[nm],
                        batch_gpu["cam_angvel"],
                        length_t,
                        chunks_meta,
                        L,
                        root0_yup,
                        device,
                        align_mode=args.align_mode,
                        blend_mode=args.blend_mode,
                    )
                batch_preds.append(preds_one)
                batch_gts.append(_extract_gt_zup(sample))
                batch_metas.append({"source": src, "clip_key": clip_key, "length": L, "fps": fps})

            # Submit all full sequences + all 250f segments for this batch to parallel MuJoCo pool
            full_jobs: list[tuple[dict[str, np.ndarray], float]] = []
            seg_jobs: list[tuple[dict[str, np.ndarray], float]] = []
            seg_slices: list[tuple[int, int]] = []
            for s_i, (preds_one, meta) in enumerate(zip(batch_preds, batch_metas)):
                fps = meta["fps"]
                for nm in model_names:
                    full_jobs.append((preds_one[nm], fps))
                    segs = _slice_segments(preds_one[nm], seg_len=args.seg_len)
                    st_idx = len(seg_jobs)
                    for sg in segs:
                        seg_jobs.append((sg, fps))
                    seg_slices.append((st_idx, len(seg_jobs)))

            full_res = _eval_batch_in_tracker(pool, full_jobs)
            seg_res = _eval_batch_in_tracker(pool, seg_jobs)

            ptr = 0
            for s_i, (preds_one, gt_d, meta) in enumerate(zip(batch_preds, batch_gts, batch_metas)):
                seq_out = dict(meta)
                for nm in model_names:
                    fr = full_res[ptr]
                    s_start, s_end = seg_slices[ptr]
                    sr_list = seg_res[s_start:s_end]
                    ptr += 1

                    kin = compute_kinematic_metrics(preds_one[nm]["body_pos_w"], gt_d["body_pos_w"], meta["fps"])
                    total50 = max(1, int(fr["total_frames_50hz"]))
                    ff = int(fr["first_failure_frame_50hz"])
                    surv = 1.0 if fr["is_success"] else float(np.clip(ff / total50, 0.0, 1.0))
                    seg_flags = [bool(x["is_success"]) for x in sr_list]
                    rec = {
                        "long_is_success": bool(fr["is_success"]),
                        "long_survival_ratio": surv,
                        "seg250_total": len(seg_flags),
                        "seg250_pass": sum(1 for f in seg_flags if f),
                        "seg250_flags": seg_flags,
                        "exec_foot_slip_cm_s": float(fr.get("exec_foot_slip_cm_s", float("nan"))),
                        "exec_root_pos_err_m": float(fr.get("exec_root_pos_err_m", float("nan"))),
                        "exec_root_z_err_m": float(fr.get("exec_root_z_err_m", float("nan"))),
                        "exec_body_mpjpe_mm": float(fr.get("exec_body_mpjpe_m", float("nan"))) * 1000.0,
                        "exec_joint_mae_rad": float(fr.get("exec_joint_mae_rad", float("nan"))),
                        "failed_anchor_height": bool(fr.get("failed_anchor_height", False)),
                        "failed_end_effector_height": bool(fr.get("failed_end_effector_height", False)),
                        "failed_anchor_orientation": bool(fr.get("failed_anchor_orientation", False)),
                        **kin,
                    }
                    records_by_model[nm].append(rec)
                    records_by_src_model[meta["source"]][nm].append(rec)
                    seq_out[nm] = rec
                seq_f.write(json.dumps(seq_out) + "\n")
                seq_f.flush()
            print(f"[DriftSuite] completed {min(b0 + len(chunk), len(jobs))}/{len(jobs)} seqs ({time.perf_counter()-t0:.1f}s)", flush=True)

    summary: dict[str, Any] = {
        "config": {
            "split": args.split,
            "sources": args.sources,
            "min_frames": args.min_frames,
            "max_bedlam2": args.max_bedlam2,
            "overlap": args.overlap,
            "align_mode": args.align_mode,
            "blend_mode": args.blend_mode,
            "sync_mode": args.sync_mode,
            "seg_len": args.seg_len,
            "seed": args.seed,
            "elapsed_s": time.perf_counter() - t0,
        },
        "overall": {nm: _summarize_model(records_by_model[nm]) for nm in model_names},
        "paired_vs_base_overall": {
            nm: _paired_summary(records_by_model["base"], records_by_model[nm])
            for nm in model_names
            if nm != "base"
        },
        "by_source": {},
    }
    for src, smap in records_by_src_model.items():
        summary["by_source"][src] = {
            "models": {nm: _summarize_model(smap[nm]) for nm in model_names},
            "paired_vs_base": {
                nm: _paired_summary(smap["base"], smap[nm])
                for nm in model_names
                if nm != "base"
            },
        }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Pretty console table
    def _print_table(title: str, mdict: dict[str, dict[str, Any]], pdict: dict[str, dict[str, Any]]) -> None:
        print(f"\n================ {title} ================")
        hdr = f"{'Model':22s} | {'Seg250%':7s} | {'Long%':7s} | {'Slip':6s} | {'L-MPJPE':7s} | {'PA':5s} | {'W':6s} | {'RTE%':5s} | {'Zmax':6s} | {'Fail(H/EE/O)':12s}"
        print(hdr)
        print("-" * len(hdr))
        for nm in model_names:
            r = mdict[nm]
            fails = f"{r['n_fail_anchor_height']}/{r['n_fail_ee_height']}/{r['n_fail_anchor_ori']}"
            print(
                f"{nm:22s} | {r['seg250_succ_pct']:6.2f}% | {r['long_succ_pct']:6.2f}% | "
                f"{r['exec_foot_slip_cm_s']:6.2f} | {r['exec_body_mpjpe_mm']:7.1f} | "
                f"{r['pa_mpjpe_mm']:5.2f} | {r['w_mpjpe_mm']:6.1f} | {r['rte_pct']:5.2f} | "
                f"{r['z_max_mm']:6.1f} | {fails:12s}"
            )
        for nm, pd in pdict.items():
            print(
                f"  vs base [{nm}]: Seg250 +{pd['seg250_fixed']}/-{pd['seg250_regressed']} (p={pd['seg250_mcnemar_p']:.3g}) | "
                f"Long +{pd['long_fixed']}/-{pd['long_regressed']} (p={pd['long_mcnemar_p']:.3g}) | "
                f"ΔZmax={pd['delta_z_max_mm']:+.2f}mm ΔW={pd['delta_w_mpjpe_mm']:+.2f}mm ΔRTE={pd['delta_rte_pct']:+.2f}% ΔSlip={pd['delta_exec_foot_slip_cm_s']:+.3f}"
            )

    _print_table("OVERALL", summary["overall"], summary["paired_vs_base_overall"])
    for src, blk in summary["by_source"].items():
        _print_table(f"SOURCE: {src.upper()}", blk["models"], blk["paired_vs_base"])


if __name__ == "__main__":
    main()
