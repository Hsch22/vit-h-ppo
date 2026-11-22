#!/usr/bin/env python3
"""Paired Base vs PPO evaluation on held-out video-conditioned G1 clips.

Evaluates a base checkpoint and one or more PPO checkpoints on identical clips,
identical initial Gaussian noise x_T, and the strictly frozen SONIC/MuJoCo
Protocol v4 tracker. Reports per-source and overall metrics, paired win/loss
counts (including exact McNemar p-value), motion-amplitude diagnostics, and
hard-clip (base-failure / high-error) breakdowns.
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
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_training.dataset import build_video_ppo_dataset
from post_training.policy import G1_PRED_X_DIM, build_ddim_schedule, decode_x0_to_g1_zup
from post_training.reward import RewardWeights, compute_sample_rewards
from post_training.tracker_pool import PersistentTrackerPool
from tools.inference_utils import inspect_checkpoint_imgfeat_encoder, load_pipeline


def _exact_mcnemar_p(b_wins: int, p_wins: int) -> float:
    """Exact two-sided binomial McNemar p-value for paired binary outcomes."""
    n = b_wins + p_wins
    if n == 0:
        return 1.0
    k = min(b_wins, p_wins)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return float(min(1.0, 2.0 * tail))


@torch.no_grad()
def sample_ddim_paired(
    pipeline: Any,
    denoiser: torch.nn.Module,
    f_cond: torch.Tensor,
    length: torch.Tensor,
    x_init: torch.Tensor,
    ddim_steps: int,
    eta: float,
    all_valid_length: bool,
    use_bf16: bool,
    step_noises: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Run DDIM sampling (supports eta=0.0 deterministic release mode and eta>0)."""
    device = x_init.device
    base_diff = pipeline.train_diffusion
    schedule = build_ddim_schedule(int(base_diff.num_timesteps), int(ddim_steps))
    alphas_full = np.asarray(base_diff.alphas_cumprod, dtype=np.float64)
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
    x = x_init.clone().float()
    bsz = x.shape[0]
    denoiser.eval()

    for rev_i, step_idx in enumerate(reversed(range(ddim_steps))):
        ts = ts_orig[step_idx].expand(bsz)
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
        if step_idx > 0 and sigma_k > 0.0:
            noise = step_noises[rev_i] if step_noises is not None else torch.randn_like(x)
            x = mean + sigma_k * noise
        else:
            x = mean
    return x


@torch.no_grad()
def build_condition(pipeline: Any, batch_gpu: dict[str, Any], use_bf16: bool) -> tuple[torch.Tensor, torch.Tensor, bool]:
    from hmr4d.utils.geo.hmr_cam import compute_bbox_info_bedlam

    cliff_cam = compute_bbox_info_bedlam(batch_gpu["bbx_xys"], batch_gpu["K_fullimg"])
    f_cam_angvel = batch_gpu.get("cam_angvel")
    if f_cam_angvel is not None and getattr(pipeline.args, "normalize_cam_angvel", False):
        f_cam_angvel = (f_cam_angvel - pipeline.cam_angvel_mean) / pipeline.cam_angvel_std.clamp(min=1e-8)
    f_imgseq = batch_gpu.get("f_imgseq")
    f_imgseq_mask = batch_gpu.get("f_imgseq_mask")
    device = cliff_cam.device
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16 and device.type == "cuda"):
        f_cond = pipeline.cond_embedder(
            obs=batch_gpu["obs"],
            f_cliffcam=cliff_cam,
            f_cam_angvel=f_cam_angvel,
            f_imgseq=f_imgseq,
            f_imgseq_mask=f_imgseq_mask,
            drop_obs=None,
        )
    length = batch_gpu["length"].long()
    all_valid = bool((length == int(f_cond.shape[1])).all().item())
    return f_cond.float(), length, all_valid


def _motion_kinematics(joint_pos: np.ndarray, root_pos_w: np.ndarray, valid_len: int, fps: float) -> tuple[float, float]:
    """Return mean joint speed (rad/s) and root XY path length (m) to diagnose motion shrinkage."""
    jp = joint_pos[:valid_len]
    rp = root_pos_w[:valid_len]
    if valid_len < 2:
        return 0.0, 0.0
    j_vel = float(np.mean(np.abs(np.diff(jp, axis=0))) * fps)
    r_path = float(np.sum(np.linalg.norm(np.diff(rp[:, :2], axis=0), axis=-1)))
    return j_vel, r_path


def _select_indices(total: int, max_clips: int, seed: int) -> list[int]:
    if max_clips <= 0 or max_clips >= total:
        return list(range(total))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(total, size=max_clips, replace=False))
    return [int(i) for i in idx]


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"count": 0}

    def mean_key(prefix: str, key: str) -> float:
        vals = [r[prefix][key] for r in records if r[prefix].get("valid", False) and math.isfinite(r[prefix][key])]
        return float(np.mean(vals)) if vals else float("nan")

    def rate_key(prefix: str, key: str) -> float:
        return float(np.mean([1.0 if r[prefix][key] else 0.0 for r in records]))

    both_pass = sum(1 for r in records if r["base"]["is_success"] and r["ppo"]["is_success"])
    base_fail_ppo_pass = sum(1 for r in records if (not r["base"]["is_success"]) and r["ppo"]["is_success"])
    base_pass_ppo_fail = sum(1 for r in records if r["base"]["is_success"] and (not r["ppo"]["is_success"]))
    both_fail = sum(1 for r in records if (not r["base"]["is_success"]) and (not r["ppo"]["is_success"]))

    summary: dict[str, Any] = {
        "count": n,
        "paired_outcomes": {
            "both_pass": both_pass,
            "ppo_fixed_base_fail": base_fail_ppo_pass,
            "ppo_regressed_base_pass": base_pass_ppo_fail,
            "both_fail": both_fail,
            "mcnemar_p_value": _exact_mcnemar_p(base_pass_ppo_fail, base_fail_ppo_pass),
        },
        "base": {
            "success_rate": rate_key("base", "is_success"),
            "invalid_rate": 1.0 - rate_key("base", "valid"),
            "survival_ratio": mean_key("base", "survival_ratio"),
            "exec_track": mean_key("base", "exec_track"),
            "exec_joint_mae_rad": mean_key("base", "exec_joint_mae_rad"),
            "exec_root_pos_err_m": mean_key("base", "exec_root_pos_err_m"),
            "fidelity": mean_key("base", "fidelity"),
            "ref_joint_mae_rad": mean_key("base", "ref_joint_mae_rad"),
            "ref_body_mpjpe_m": mean_key("base", "ref_body_mpjpe_m"),
            "reward": mean_key("base", "reward"),
            "mean_joint_vel_rad_s": mean_key("base", "mean_joint_vel_rad_s"),
            "root_xy_path_m": mean_key("base", "root_xy_path_m"),
        },
        "ppo": {
            "success_rate": rate_key("ppo", "is_success"),
            "invalid_rate": 1.0 - rate_key("ppo", "valid"),
            "survival_ratio": mean_key("ppo", "survival_ratio"),
            "exec_track": mean_key("ppo", "exec_track"),
            "exec_joint_mae_rad": mean_key("ppo", "exec_joint_mae_rad"),
            "exec_root_pos_err_m": mean_key("ppo", "exec_root_pos_err_m"),
            "fidelity": mean_key("ppo", "fidelity"),
            "ref_joint_mae_rad": mean_key("ppo", "ref_joint_mae_rad"),
            "ref_body_mpjpe_m": mean_key("ppo", "ref_body_mpjpe_m"),
            "reward": mean_key("ppo", "reward"),
            "mean_joint_vel_rad_s": mean_key("ppo", "mean_joint_vel_rad_s"),
            "root_xy_path_m": mean_key("ppo", "root_xy_path_m"),
        },
        "gt_kinematics": {
            "mean_joint_vel_rad_s": float(np.mean([r["gt_joint_vel_rad_s"] for r in records])),
            "root_xy_path_m": float(np.mean([r["gt_root_xy_path_m"] for r in records])),
        },
    }
    summary["delta_ppo_minus_base"] = {
        k: float(summary["ppo"][k] - summary["base"][k]) for k in summary["base"]
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", default=["h36m", "bedlam", "bedlam2"])
    parser.add_argument("--split", type=str, default="val", choices=("val", "test"))
    parser.add_argument("--g1-pth-name", type=str, default="g1_new.pth")
    parser.add_argument("--base-ckpt", type=Path, default=REPO_ROOT / "vith_offline_g1.ckpt")
    parser.add_argument("--ppo-ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-clips-per-source", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-bf16", action="store_true")

    parser.add_argument(
        "--tracker-python",
        type=Path,
        default=Path(os.environ.get("TRACKER_PYTHON_BIN", sys.executable)),
    )
    parser.add_argument(
        "--policy-onnx",
        type=Path,
        default=REPO_ROOT / "inputs/checkpoints/g1_tracker/model_step_020000_g1.onnx",
    )
    parser.add_argument(
        "--xml-path",
        type=Path,
        default=REPO_ROOT / "g1_tracker/assets/g1/rollout/scene_29dof_mujoco.xml",
    )
    parser.add_argument(
        "--grounding-xml",
        type=Path,
        default=REPO_ROOT / "g1_tracker/assets/g1/grounding/mjcf/g1_29dof_rev_1_0.xml",
    )
    parser.add_argument("--tracker-workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_bf16 = (not args.no_bf16) and (device.type == "cuda")
    torch.set_float32_matmul_precision("high")

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify encoder metadata and load both pipelines strictly.
    base_enc, base_prov = inspect_checkpoint_imgfeat_encoder(str(args.base_ckpt))
    ppo_enc, ppo_prov = inspect_checkpoint_imgfeat_encoder(str(args.ppo_ckpt))
    print(f"[Eval] base={args.base_ckpt} (enc={base_enc}, {base_prov})", flush=True)
    print(f"[Eval] ppo ={args.ppo_ckpt} (enc={ppo_enc}, {ppo_prov})", flush=True)
    print(f"[Eval] split={args.split}, sources={args.sources}, max_per_source={args.max_clips_per_source}, ddim_steps={args.ddim_steps}, eta={args.ddim_eta}", flush=True)

    overrides = ["network.latent_dim=1024", "network.num_layers=16", "network.num_heads=16"]
    base_pipe = load_pipeline(str(args.base_ckpt), exp="offline/g1_training_set_exp1", device=str(device), overrides=overrides).eval()
    ppo_pipe = load_pipeline(str(args.ppo_ckpt), exp="offline/g1_training_set_exp1", device=str(device), overrides=overrides).eval()

    # Quantify actual weight shift between base and PPO before running eval.
    w_diffs = [
        (p1.float() - p2.float()).abs().max().item()
        for p1, p2 in zip(base_pipe.denoiser3d.parameters(), ppo_pipe.denoiser3d.parameters())
    ]
    w_l2 = math.sqrt(
        sum(
            float(((p1.float() - p2.float()) ** 2).sum().item())
            for p1, p2 in zip(base_pipe.denoiser3d.parameters(), ppo_pipe.denoiser3d.parameters())
        )
    )
    print(f"[Eval] denoiser3d weight delta: max_abs={max(w_diffs):.4e}, l2={w_l2:.4e}", flush=True)

    reward_weights = RewardWeights()
    all_records: list[dict[str, Any]] = []
    by_source: dict[str, list[dict[str, Any]]] = {}

    with PersistentTrackerPool(
        tracker_python=args.tracker_python,
        policy_onnx=args.policy_onnx,
        xml_path=args.xml_path,
        grounding_xml=args.grounding_xml,
        num_workers=args.tracker_workers,
    ) as tracker_pool:
        print(f"[Eval] PersistentTrackerPool ready: {args.tracker_workers} workers (protocol v{tracker_pool.protocol_version})", flush=True)

        for src_i, source in enumerate(args.sources):
            ds_full = build_video_ppo_dataset(
                dataset_root=args.dataset_root,
                sources=[source],
                split=args.split,
                g1_pth_name=args.g1_pth_name,
                split_seed=42,
            )
            chosen = _select_indices(len(ds_full), args.max_clips_per_source, seed=args.seed + src_i * 101)
            ds = Subset(ds_full, chosen)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))
            print(f"[Eval:{source}] evaluating {len(chosen)}/{len(ds_full)} clips...", flush=True)

            src_records: list[dict[str, Any]] = []
            gen = torch.Generator(device=device).manual_seed(args.seed + src_i * 1009)
            clip_offset = 0

            for batch in loader:
                batch_gpu = {
                    k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                f_cond, length, all_valid = build_condition(base_pipe, batch_gpu, use_bf16=use_bf16)
                bsz, seq_len, _ = f_cond.shape

                # Shared initial noise (and shared per-step noise if eta > 0) for exact paired comparison.
                x_init = torch.randn((bsz, seq_len, G1_PRED_X_DIM), device=device, dtype=torch.float32, generator=gen)
                step_noises = (
                    [
                        torch.randn((bsz, seq_len, G1_PRED_X_DIM), device=device, dtype=torch.float32, generator=gen)
                        for _ in range(args.ddim_steps)
                    ]
                    if args.ddim_eta > 0.0
                    else None
                )

                x0_base = sample_ddim_paired(
                    base_pipe, base_pipe.denoiser3d, f_cond, length, x_init, args.ddim_steps, args.ddim_eta, all_valid, use_bf16, step_noises
                )
                x0_ppo = sample_ddim_paired(
                    ppo_pipe, ppo_pipe.denoiser3d, f_cond, length, x_init, args.ddim_steps, args.ddim_eta, all_valid, use_bf16, step_noises
                )

                dec_base = decode_x0_to_g1_zup(base_pipe, x0_base, batch_gpu["cam_angvel"], length, batch_gpu["root_pos_0_yup"])
                dec_ppo = decode_x0_to_g1_zup(ppo_pipe, x0_ppo, batch_gpu["cam_angvel"], length, batch_gpu["root_pos_0_yup"])

                lens_np = length.cpu().numpy()
                fps_np = batch["fps"].numpy()
                gt_jp = batch["gt_joint_pos"].numpy()
                gt_bp = batch["gt_body_pos_z"].numpy()

                eval_pair: dict[str, tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
                for tag, dec in (("base", dec_base), ("ppo", dec_ppo)):
                    jp = dec["joint_pos"].cpu().numpy()
                    rp = dec["root_pos_w"].cpu().numpy()
                    rq = dec["root_quat_w"].cpu().numpy()
                    bp = dec["body_pos_w"].cpu().numpy()
                    trk = tracker_pool.evaluate_batch(jp, rp, rq, bp, lens_np, fps=fps_np)
                    sc = compute_sample_rewards(trk, jp, bp, gt_jp, gt_bp, lens_np, reward_weights)
                    eval_pair[tag] = (trk, sc, {"jp": jp, "rp": rp})

                for i in range(bsz):
                    vlen = int(lens_np[i])
                    fps_i = float(fps_np[i])
                    gt_jvel, gt_rpath = _motion_kinematics(gt_jp[i], gt_bp[i, :, 0, :], vlen, fps_i)
                    rec: dict[str, Any] = {
                        "source": source,
                        "dataset_idx": chosen[clip_offset + i],
                        "length": vlen,
                        "fps": fps_i,
                        "gt_joint_vel_rad_s": gt_jvel,
                        "gt_root_xy_path_m": gt_rpath,
                    }
                    for tag in ("base", "ppo"):
                        trk_i = eval_pair[tag][0][i]
                        sc_i = eval_pair[tag][1]
                        arr_i = eval_pair[tag][2]
                        jvel, rpath = _motion_kinematics(arr_i["jp"][i], arr_i["rp"][i], vlen, fps_i)
                        rec[tag] = {
                            "valid": bool(sc_i["is_valid"][i]),
                            "is_success": bool(sc_i["is_success"][i]),
                            "first_failure_frame_50hz": int(trk_i["first_failure_frame_50hz"]),
                            "total_frames_50hz": int(trk_i["total_frames_50hz"]),
                            "survival_ratio": float(sc_i["survival_ratio"][i]),
                            "exec_track": float(sc_i["exec_track"][i]),
                            "exec_joint_mae_rad": float(trk_i["exec_joint_mae_rad"]),
                            "exec_root_pos_err_m": float(trk_i["exec_root_pos_err_m"]),
                            "fidelity": float(sc_i["fidelity"][i]),
                            "ref_joint_mae_rad": float(sc_i["ref_joint_mae_rad"][i]),
                            "ref_body_mpjpe_m": float(sc_i["ref_body_mpjpe_m"][i]),
                            "reward": float(sc_i["reward"][i]),
                            "grounding_offset_m": float(trk_i["grounding_offset_m"]),
                            "initial_joint_clip_count": int(trk_i["initial_joint_clip_count"]),
                            "mean_joint_vel_rad_s": jvel,
                            "root_xy_path_m": rpath,
                        }
                    src_records.append(rec)
                    all_records.append(rec)
                clip_offset += bsz

            by_source[source] = src_records
            s_agg = _aggregate(src_records)
            print(
                f"[Eval:{source} DONE] n={s_agg['count']} | "
                f"succ base={s_agg['base']['success_rate']:.4f} ppo={s_agg['ppo']['success_rate']:.4f} "
                f"(fixed={s_agg['paired_outcomes']['ppo_fixed_base_fail']} regressed={s_agg['paired_outcomes']['ppo_regressed_base_pass']} p={s_agg['paired_outcomes']['mcnemar_p_value']:.4f}) | "
                f"exec_rerr base={s_agg['base']['exec_root_pos_err_m']:.4f} ppo={s_agg['ppo']['exec_root_pos_err_m']:.4f} | "
                f"ref_jmae base={s_agg['base']['ref_joint_mae_rad']:.4f} ppo={s_agg['ppo']['ref_joint_mae_rad']:.4f}",
                flush=True,
            )

    # Hard subsets: (1) all clips where base failed; (2) top-25% hardest by base exec_root_pos_err_m.
    base_fail_records = [r for r in all_records if not r["base"]["is_success"]]
    valid_base_errs = [r["base"]["exec_root_pos_err_m"] for r in all_records if math.isfinite(r["base"]["exec_root_pos_err_m"])]
    q75_err = float(np.quantile(valid_base_errs, 0.75)) if valid_base_errs else float("inf")
    hard_q75_records = [
        r for r in all_records
        if (not r["base"]["is_success"]) or (math.isfinite(r["base"]["exec_root_pos_err_m"]) and r["base"]["exec_root_pos_err_m"] >= q75_err)
    ]

    summary = {
        "config": {
            "base_ckpt": str(args.base_ckpt.resolve()),
            "ppo_ckpt": str(args.ppo_ckpt.resolve()),
            "split": args.split,
            "sources": args.sources,
            "g1_pth_name": args.g1_pth_name,
            "max_clips_per_source": args.max_clips_per_source,
            "ddim_steps": args.ddim_steps,
            "ddim_eta": args.ddim_eta,
            "seed": args.seed,
            "elapsed_s": round(time.perf_counter() - t_start, 2),
        },
        "weight_delta": {"denoiser3d_max_abs": max(w_diffs), "denoiser3d_l2": w_l2},
        "overall": _aggregate(all_records),
        "by_source": {s: _aggregate(recs) for s, recs in by_source.items()},
        "hard_subsets": {
            "base_failed_clips": _aggregate(base_fail_records),
            "top25pct_base_root_err_threshold_m": q75_err,
            "top25pct_hard_clips": _aggregate(hard_q75_records),
        },
    }

    (out_dir / "per_clip.jsonl").write_text("\n".join(json.dumps(r) for r in all_records) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== OVERALL SUMMARY ===")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
