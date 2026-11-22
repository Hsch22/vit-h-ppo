#!/usr/bin/env python3
"""Multi-GPU Base + GT trackability scanner and hard-clip miner for ViT-H PPO.

Scans video-conditioned G1 clips (`h36m`, `bedlam`, `bedlam2`) with deterministic
non-overlapping 120-frame windows, runs the frozen Base model (`DDIM-50 eta=0.0`
by default) and ground-truth `g1_new.pth` through the frozen SONIC/MuJoCo
Protocol v4 tracker, and partitions clips into auditable buckets:
  - bucket_a_fixable_fail : GT passes Protocol v4, Base fails
  - bucket_b_high_exec_err: Both pass, Base execution error in top-20% (or above threshold)
  - bucket_c_easy_pass    : Both pass, low execution error
  - untrackable_gt        : GT itself fails Protocol v4 (filtered out of hard mining)
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
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hmr4d.model.offline.utils.obs_joints import select_coco17_no_nose_ears
from hmr4d.utils.geo.hmr_cam import normalize_kp2d
from post_training.dataset import VIDEO_SOURCE_SPECS
from post_training.eval_compare import _motion_kinematics, build_condition, sample_ddim_paired
from post_training.policy import G1_PRED_X_DIM, YUP_TO_ZUP_M, decode_x0_to_g1_zup
from post_training.reward import RewardWeights, compute_sample_rewards
from post_training.tracker_pool import PersistentTrackerPool
from tools.inference_utils import load_pipeline


class _MiningClipWrapper(Dataset):
    """Wrap deterministic clip windows with both Base inputs and GT Z-up tracker inputs."""

    def __init__(self, inner: Dataset, source: str, part_subdir: str) -> None:
        self.inner = inner
        self.source = source
        self.part_subdir = part_subdir
        self._m_pos = torch.tensor(YUP_TO_ZUP_M, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

        sample = self.inner[idx]
        obs = select_coco17_no_nose_ears(normalize_kp2d(sample["kp2d"], sample["bbx_xys"])).float()
        valid = torch.as_tensor(sample["mask"]["valid"], dtype=torch.bool)
        obs[~valid] = 0.0

        g1 = sample["g1_target"]
        gt_body_yup = g1["g1_body_pos_w"].float()
        gt_body_zup = gt_body_yup @ self._m_pos
        gt_root_pos_zup = gt_body_zup[:, 0, :].clone()
        gt_root_quat_yup = g1["g1_body_quat_w"][:, 0, :].float()
        gt_root_R_yup = quaternion_to_matrix(gt_root_quat_yup)
        gt_root_quat_zup = F.normalize(matrix_to_quaternion(self._m_pos.T @ gt_root_R_yup), dim=-1)
        root_pos_0_yup = gt_body_yup[0:1, 0, :].clone()

        meta = self.inner.idx2meta[idx]
        return {
            "source": self.source,
            "part_subdir": self.part_subdir,
            "local_idx": int(idx),
            "clip_key": str(meta["key"]),
            "seg_id": int(meta.get("seg_id", 0)),
            "usable_len": int(meta.get("usable_len", sample["length"])),
            "length": torch.as_tensor(sample["length"], dtype=torch.long),
            "obs": obs,
            "bbx_xys": sample["bbx_xys"].float(),
            "K_fullimg": sample["K_fullimg"].float(),
            "cam_angvel": sample["cam_angvel"].float(),
            "f_imgseq": sample["f_imgseq"].float(),
            "f_imgseq_mask": torch.as_tensor(sample["mask"]["f_imgseq"], dtype=torch.bool),
            "root_pos_0_yup": root_pos_0_yup,
            "gt_joint_pos": g1["g1_joint_pos"].float(),
            "gt_root_pos_z": gt_root_pos_zup,
            "gt_root_quat_z": gt_root_quat_zup,
            "gt_body_pos_z": gt_body_zup,
            "fps": sample["fps"].float(),
        }


def build_deterministic_mining_dataset(
    dataset_root: Path | str,
    sources: list[str],
    target_split: str,
    g1_pth_name: str = "g1_new.pth",
    motion_frames: int = 120,
    split_seed: int = 42,
    split_ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
) -> ConcatDataset:
    """Build deterministic non-overlapping 120-frame windows for any split (`train`/`val`/`test`).

    Passes `split='val'` and `subset_split=target_split` so `Bedlam2G1Dataset`
    uses deterministic segment slicing (`seg_id * motion_frames`) even when
    mining `train` keys.
    """
    root = Path(dataset_root).expanduser().resolve()
    parts: list[Dataset] = []
    for source in sources:
        if source not in VIDEO_SOURCE_SPECS:
            raise ValueError(f"Unsupported source {source!r}")
        for spec in VIDEO_SOURCE_SPECS[source]:
            data_dir = root / spec["subdir"]
            g1_file = data_dir / g1_pth_name
            if not g1_file.is_file():
                raise FileNotFoundError(f"Missing required G1 file: {g1_file}")
            kwargs: dict[str, Any] = {
                "data_dir": str(data_dir),
                "smplx_pth_name": "motion.pth",
                "g1_pth_name": g1_pth_name,
                "bbox_pth_paths": [str(data_dir / p) for p in spec["bbox"]],
                "vitpose_pth_paths": [str(data_dir / p) for p in spec["kp2d"]],
                "imgfeat_pth_paths": [str(data_dir / p) for p in spec["imgfeat"]],
                "motion_frames": int(motion_frames),
                "min_motion_frames": 60,
                "g1_input_is_yup": False,
                "dof_pos_order": "mjc",
                "canonicalize_first_frame": True,
                "floor_adjust": True,
                "yaw_only_canon": True,
                "split": "val",
                "subset_split": target_split,
                "split_seed": int(split_seed),
                "split_ratios": split_ratios,
            }
            kwargs.update(spec.get("extra", {}))
            parts.append(_MiningClipWrapper(spec["cls"](**kwargs), source=source, part_subdir=spec["subdir"]))
    return ConcatDataset(parts)


def _init_dist() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, local_rank, world_size
    return 0, 0, 1


def _summarize_records(
    records: list[dict[str, Any]],
    root_err_quantile: float = 0.80,
    joint_mae_quantile: float = 0.80,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assign buckets and compute auditable overall & per-source statistics."""
    # Compute hard execution-error thresholds over Base-passed, GT-passable (or unchecked) clips.
    pass_root_errs = [
        r["base"]["exec_root_pos_err_m"]
        for r in records
        if r["base"]["is_success"] and math.isfinite(r["base"]["exec_root_pos_err_m"])
        and (r["gt"] is None or r["gt"]["is_success"])
    ]
    pass_joint_maes = [
        r["base"]["exec_joint_mae_rad"]
        for r in records
        if r["base"]["is_success"] and math.isfinite(r["base"]["exec_joint_mae_rad"])
        and (r["gt"] is None or r["gt"]["is_success"])
    ]
    root_thr = float(np.quantile(pass_root_errs, root_err_quantile)) if pass_root_errs else float("inf")
    jmae_thr = float(np.quantile(pass_joint_maes, joint_mae_quantile)) if pass_joint_maes else float("inf")

    buckets: dict[str, list[dict[str, Any]]] = {
        "bucket_a_fixable_fail": [],
        "bucket_b_high_exec_err": [],
        "bucket_c_easy_pass": [],
        "untrackable_gt": [],
    }

    for r in records:
        gt_known = r["gt"] is not None
        gt_ok = True if not gt_known else bool(r["gt"]["is_success"])
        base_ok = bool(r["base"]["is_success"])
        if gt_known and not gt_ok:
            bname = "untrackable_gt"
        elif not base_ok:
            bname = "bucket_a_fixable_fail"
        elif (
            r["base"]["exec_root_pos_err_m"] >= root_thr
            or r["base"]["exec_joint_mae_rad"] >= jmae_thr
        ):
            bname = "bucket_b_high_exec_err"
        else:
            bname = "bucket_c_easy_pass"
        r["bucket"] = bname
        buckets[bname].append(r)

    def stats_for(subset: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(subset)
        if n == 0:
            return {"count": 0}
        gt_eval = [x for x in subset if x["gt"] is not None]
        b_succ = float(np.mean([1.0 if x["base"]["is_success"] else 0.0 for x in subset]))
        b_surv = float(np.mean([x["base"]["survival_ratio"] for x in subset]))
        b_rerr = float(np.nanmean([x["base"]["exec_root_pos_err_m"] for x in subset]))
        b_jmae = float(np.nanmean([x["base"]["exec_joint_mae_rad"] for x in subset]))
        b_rjmae = float(np.nanmean([x["base"]["ref_joint_mae_rad"] for x in subset]))
        b_mpjpe = float(np.nanmean([x["base"]["ref_body_mpjpe_m"] for x in subset]))
        b_fid = float(np.nanmean([x["base"]["fidelity"] for x in subset]))
        gt_jvel = float(np.mean([x["gt_joint_vel_rad_s"] for x in subset]))
        gt_rpath = float(np.mean([x["gt_root_xy_path_m"] for x in subset]))
        out: dict[str, Any] = {
            "count": n,
            "base_success_rate": b_succ,
            "base_fail_count": int(sum(1 for x in subset if not x["base"]["is_success"])),
            "base_survival_ratio": b_surv,
            "base_exec_root_pos_err_m": b_rerr,
            "base_exec_joint_mae_rad": b_jmae,
            "base_ref_joint_mae_rad": b_rjmae,
            "base_ref_body_mpjpe_m": b_mpjpe,
            "base_fidelity": b_fid,
            "gt_joint_vel_rad_s": gt_jvel,
            "gt_root_xy_path_m": gt_rpath,
            "bucket_counts": {k: int(sum(1 for x in subset if x["bucket"] == k)) for k in buckets},
        }
        if gt_eval:
            out["gt_evaluated_count"] = len(gt_eval)
            out["gt_success_rate"] = float(np.mean([1.0 if x["gt"]["is_success"] else 0.0 for x in gt_eval]))
            out["gt_exec_root_pos_err_m"] = float(np.nanmean([x["gt"]["exec_root_pos_err_m"] for x in gt_eval]))
            out["gt_exec_joint_mae_rad"] = float(np.nanmean([x["gt"]["exec_joint_mae_rad"] for x in gt_eval]))
            gt_pass_subset = [x for x in gt_eval if x["gt"]["is_success"]]
            if gt_pass_subset:
                out["base_success_given_gt_pass"] = float(
                    np.mean([1.0 if x["base"]["is_success"] else 0.0 for x in gt_pass_subset])
                )
        return out

    sources = sorted(set(r["source"] for r in records))
    summary = {
        "thresholds": {
            "root_err_quantile": root_err_quantile,
            "joint_mae_quantile": joint_mae_quantile,
            "exec_root_pos_err_m_thr": root_thr,
            "exec_joint_mae_rad_thr": jmae_thr,
        },
        "overall": stats_for(records),
        "by_source": {s: stats_for([r for r in records if r["source"] == s]) for s in sources},
        "by_bucket": {k: stats_for(v) for k, v in buckets.items()},
    }

    hard_index = {
        "thresholds": summary["thresholds"],
        "bucket_counts": {k: len(v) for k, v in buckets.items()},
        "buckets": {
            k: [
                {
                    "global_idx": r["global_idx"],
                    "source": r["source"],
                    "part_subdir": r["part_subdir"],
                    "local_idx": r["local_idx"],
                    "clip_key": r["clip_key"],
                    "seg_id": r["seg_id"],
                    "length": r["length"],
                    "fps": r["fps"],
                    "base_is_success": r["base"]["is_success"],
                    "base_exec_root_pos_err_m": r["base"]["exec_root_pos_err_m"],
                    "base_exec_joint_mae_rad": r["base"]["exec_joint_mae_rad"],
                    "gt_is_success": None if r["gt"] is None else r["gt"]["is_success"],
                }
                for r in v
            ]
            for k, v in buckets.items()
        },
    }
    return summary, hard_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", default=["h36m", "bedlam", "bedlam2"])
    parser.add_argument("--split", type=str, required=True, choices=("train", "val", "test"))
    parser.add_argument("--g1-pth-name", type=str, default="g1_new.pth")
    parser.add_argument("--base-ckpt", type=Path, default=REPO_ROOT / "vith_offline_g1.ckpt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-clips", type=int, default=0, help="0 = scan all clips in split")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gt-mode", type=str, default="all", choices=("all", "hard", "none"))
    parser.add_argument("--gt-hard-root-thr", type=float, default=0.11)
    parser.add_argument("--gt-hard-jmae-thr", type=float, default=0.11)
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
    rank, local_rank, world_size = _init_dist()
    is_main = rank == 0
    t_start = time.perf_counter()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    use_bf16 = (not args.no_bf16) and (device.type == "cuda")
    torch.set_float32_matmul_precision("high")

    out_dir = args.output_dir.expanduser().resolve()
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    ds_full = build_deterministic_mining_dataset(
        dataset_root=args.dataset_root,
        sources=args.sources,
        target_split=args.split,
        g1_pth_name=args.g1_pth_name,
        split_seed=42,
    )
    total_avail = len(ds_full)
    if args.max_clips > 0 and args.max_clips < total_avail:
        rng = np.random.default_rng(args.seed)
        selected_global = np.sort(rng.choice(total_avail, size=args.max_clips, replace=False)).tolist()
    else:
        selected_global = list(range(total_avail))

    # Static strided shard across ranks so every clip is processed exactly once.
    rank_indices = selected_global[rank::world_size]
    if is_main:
        print(
            f"[Mine] split={args.split} sources={args.sources} total={len(selected_global)}/{total_avail} "
            f"world_size={world_size} gt_mode={args.gt_mode} ddim={args.ddim_steps} eta={args.ddim_eta}",
            flush=True,
        )

    overrides = ["network.latent_dim=1024", "network.num_layers=16", "network.num_heads=16"]
    base_pipe = load_pipeline(str(args.base_ckpt), exp="offline/g1_training_set_exp1", device=str(device), overrides=overrides).eval()

    loader = DataLoader(
        Subset(ds_full, rank_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    reward_weights = RewardWeights()
    shard_path = out_dir / f"shard_rank{rank:02d}.jsonl"
    done = 0

    with PersistentTrackerPool(
        tracker_python=args.tracker_python,
        policy_onnx=args.policy_onnx,
        xml_path=args.xml_path,
        grounding_xml=args.grounding_xml,
        num_workers=args.tracker_workers,
    ) as tracker_pool, shard_path.open("w", encoding="utf-8") as fout:
        for batch in loader:
            batch_gpu = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            f_cond, length, all_valid = build_condition(base_pipe, batch_gpu, use_bf16=use_bf16)
            bsz, seq_len, _ = f_cond.shape

            # Deterministic per-clip initial Gaussian noise keyed by global_idx.
            batch_globals = rank_indices[done : done + bsz]
            x_inits = []
            for gidx in batch_globals:
                g = torch.Generator(device=device).manual_seed(args.seed + int(gidx) * 17)
                x_inits.append(torch.randn((1, seq_len, G1_PRED_X_DIM), device=device, dtype=torch.float32, generator=g))
            x_init = torch.cat(x_inits, dim=0)

            x0_base = sample_ddim_paired(
                base_pipe,
                base_pipe.denoiser3d,
                f_cond,
                length,
                x_init,
                args.ddim_steps,
                args.ddim_eta,
                all_valid,
                use_bf16,
            )
            dec_base = decode_x0_to_g1_zup(
                base_pipe, x0_base, batch_gpu["cam_angvel"], length, batch_gpu["root_pos_0_yup"]
            )

            lens_np = length.cpu().numpy()
            fps_np = batch["fps"].numpy()
            jp = dec_base["joint_pos"].cpu().numpy()
            rp = dec_base["root_pos_w"].cpu().numpy()
            rq = dec_base["root_quat_w"].cpu().numpy()
            bp = dec_base["body_pos_w"].cpu().numpy()
            gt_jp = batch["gt_joint_pos"].numpy()
            gt_rp = batch["gt_root_pos_z"].numpy()
            gt_rq = batch["gt_root_quat_z"].numpy()
            gt_bp = batch["gt_body_pos_z"].numpy()

            base_trk = tracker_pool.evaluate_batch(jp, rp, rq, bp, lens_np, fps=fps_np)
            base_sc = compute_sample_rewards(base_trk, jp, bp, gt_jp, gt_bp, lens_np, reward_weights)

            # Decide which clips in this batch get GT trackability verification.
            gt_indices: list[int] = []
            if args.gt_mode == "all":
                gt_indices = list(range(bsz))
            elif args.gt_mode == "hard":
                for i in range(bsz):
                    rerr = float(base_trk[i]["exec_root_pos_err_m"])
                    jmae = float(base_trk[i]["exec_joint_mae_rad"])
                    if (
                        (not bool(base_sc["is_success"][i]))
                        or (not math.isfinite(rerr))
                        or rerr >= args.gt_hard_root_thr
                        or jmae >= args.gt_hard_jmae_thr
                    ):
                        gt_indices.append(i)

            gt_res_map: dict[int, dict[str, Any]] = {}
            if gt_indices:
                idx_arr = np.array(gt_indices, dtype=np.int64)
                gt_trk = tracker_pool.evaluate_batch(
                    gt_jp[idx_arr],
                    gt_rp[idx_arr],
                    gt_rq[idx_arr],
                    gt_bp[idx_arr],
                    lens_np[idx_arr],
                    fps=fps_np[idx_arr],
                )
                for pos, orig_i in enumerate(gt_indices):
                    gt_res_map[orig_i] = gt_trk[pos]

            for i in range(bsz):
                vlen = int(lens_np[i])
                fps_i = float(fps_np[i])
                gt_jvel, gt_rpath = _motion_kinematics(gt_jp[i], gt_rp[i], vlen, fps_i)
                b_jvel, b_rpath = _motion_kinematics(jp[i], rp[i], vlen, fps_i)
                gt_info = None
                if i in gt_res_map:
                    g_t = gt_res_map[i]
                    g_tot = max(1, int(g_t["total_frames_50hz"]))
                    g_succ = bool(g_t["is_success"])
                    g_fail = int(g_t["first_failure_frame_50hz"])
                    gt_info = {
                        "valid": bool(g_t["valid"]),
                        "is_success": g_succ,
                        "first_failure_frame_50hz": g_fail,
                        "total_frames_50hz": g_tot,
                        "survival_ratio": 1.0 if g_succ else float(np.clip(g_fail / g_tot, 0.0, 1.0)),
                        "exec_joint_mae_rad": float(g_t["exec_joint_mae_rad"]),
                        "exec_root_pos_err_m": float(g_t["exec_root_pos_err_m"]),
                    }
                rec = {
                    "global_idx": int(batch_globals[i]),
                    "source": str(batch["source"][i]),
                    "part_subdir": str(batch["part_subdir"][i]),
                    "local_idx": int(batch["local_idx"][i]),
                    "clip_key": str(batch["clip_key"][i]),
                    "seg_id": int(batch["seg_id"][i]),
                    "usable_len": int(batch["usable_len"][i]),
                    "length": vlen,
                    "fps": fps_i,
                    "gt_joint_vel_rad_s": gt_jvel,
                    "gt_root_xy_path_m": gt_rpath,
                    "base": {
                        "valid": bool(base_sc["is_valid"][i]),
                        "is_success": bool(base_sc["is_success"][i]),
                        "first_failure_frame_50hz": int(base_trk[i]["first_failure_frame_50hz"]),
                        "total_frames_50hz": int(base_trk[i]["total_frames_50hz"]),
                        "survival_ratio": float(base_sc["survival_ratio"][i]),
                        "exec_track": float(base_sc["exec_track"][i]),
                        "exec_joint_mae_rad": float(base_trk[i]["exec_joint_mae_rad"]),
                        "exec_root_pos_err_m": float(base_trk[i]["exec_root_pos_err_m"]),
                        "fidelity": float(base_sc["fidelity"][i]),
                        "ref_joint_mae_rad": float(base_sc["ref_joint_mae_rad"][i]),
                        "ref_body_mpjpe_m": float(base_sc["ref_body_mpjpe_m"][i]),
                        "reward": float(base_sc["reward"][i]),
                        "mean_joint_vel_rad_s": b_jvel,
                        "root_xy_path_m": b_rpath,
                    },
                    "gt": gt_info,
                }
                fout.write(json.dumps(rec) + "\n")
            done += bsz
            if is_main and (done % (args.batch_size * 10) == 0 or done == len(rank_indices)):
                elapsed = time.perf_counter() - t_start
                est_total = done * world_size
                print(
                    f"[Mine:{args.split}] ~{min(est_total, len(selected_global))}/{len(selected_global)} "
                    f"elapsed={elapsed:.1f}s ({est_total/max(elapsed,1e-3):.1f} clips/s)",
                    flush=True,
                )

    if world_size > 1:
        dist.barrier()

    if is_main:
        all_records: list[dict[str, Any]] = []
        for r in range(world_size):
            sp = out_dir / f"shard_rank{r:02d}.jsonl"
            for line in sp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    all_records.append(json.loads(line))
            sp.unlink()
        all_records.sort(key=lambda x: x["global_idx"])
        summary, hard_index = _summarize_records(all_records)
        summary["config"] = {
            "split": args.split,
            "sources": args.sources,
            "g1_pth_name": args.g1_pth_name,
            "base_ckpt": str(args.base_ckpt.resolve()),
            "ddim_steps": args.ddim_steps,
            "ddim_eta": args.ddim_eta,
            "gt_mode": args.gt_mode,
            "world_size": world_size,
            "total_clips": len(all_records),
            "elapsed_s": round(time.perf_counter() - t_start, 2),
        }
        (out_dir / "clips.jsonl").write_text("\n".join(json.dumps(x) for x in all_records) + "\n")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (out_dir / "hard_index.json").write_text(json.dumps(hard_index, indent=2) + "\n")
        print(f"\n=== MINING SUMMARY ({args.split}) ===")
        print(json.dumps(summary, indent=2), flush=True)

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
