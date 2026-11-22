#!/usr/bin/env python3
"""Multi-window stitched long-horizon Diffusion-PPO v3 trainer (`ddpo_stitched_v3`).

Rolls out two overlapping 120-frame windows (`[0:120]` and `[119:239]`), stitches
them into a continuous 239-frame G1 trajectory with the exact rigid endpoint
alignment from `tools/infer_video.py`, evaluates the full 239-frame trajectory in
the frozen SONIC/MuJoCo Protocol v4 tracker, and optimizes joint two-window
likelihoods with explicit seam-consistency + execution + velocity-preservation
rewards (`reward_v3`) against the frozen base pretrained denoiser.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_training import POLICY_VERSION_STITCHED, REWARD_VERSION_V3
from post_training.dataset_stitched import (
    STITCHED_TOTAL_LEN,
    STITCHED_WINDOW,
    StitchedKeyWeightedDistributedSampler,
    build_stitched_ppo_dataset,
    load_key_buckets_from_mining_clips,
)
from post_training.eval_compare import sample_ddim_paired
from post_training.eval_long_horizon import _decode_chunks_yup
from post_training.policy import G1_PRED_X_DIM, StochasticDDIMPolicy, YUP_TO_ZUP_M
from post_training.reward import (
    RewardWeights,
    compute_group_advantages,
    compute_sample_rewards,
    make_reward_weights_v3,
)
from post_training.stitching import stitch_two_chunks_yup
from post_training.tracker_pool import PersistentTrackerPool
from tools.inference_utils import load_pipeline


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _init_distributed() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, local_rank, world_size
    return 0, 0, 1


def _all_reduce_mean(value: float, device: torch.device, world_size: int) -> float:
    if world_size == 1:
        return float(value)
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float((t / world_size).item())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def decode_and_stitch_zup(
    pipeline: Any,
    x0_both: torch.Tensor,
    ang_c0: torch.Tensor,
    ang_c1: torch.Tensor,
    root_pos_0_yup: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Decode stacked [c0, c1] (2N, 120, 189) -> stitched 239f Z-up G1 motion + seam metrics."""
    from motion_lib.motion_utils.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix

    n_tot = x0_both.shape[0]
    assert n_tot % 2 == 0
    n = n_tot // 2
    device = x0_both.device
    ang_both = torch.cat([ang_c0, ang_c1], dim=0)
    len_both = torch.full((n_tot,), STITCHED_WINDOW, dtype=torch.long, device=device)
    root0_both = torch.cat([root_pos_0_yup, root_pos_0_yup], dim=0)

    jp_b, rp_b, rq_b, bp_b = _decode_chunks_yup(pipeline, x0_both, ang_both, len_both, root0_both)
    jp_y, rp_y, rq_y, bp_y, seam_info = stitch_two_chunks_yup(
        jp_b[:n], rp_b[:n], rq_b[:n], bp_b[:n],
        jp_b[n:], rp_b[n:], rq_b[n:], bp_b[n:],
    )

    M_pos = torch.tensor(YUP_TO_ZUP_M, device=device, dtype=torch.float32)
    rp_z = rp_y @ M_pos
    bp_z = bp_y @ M_pos
    rq_z = F.normalize(matrix_to_quaternion(M_pos.T @ quaternion_to_matrix(rq_y)), dim=-1)
    return {
        "joint_pos": jp_y,
        "root_pos_w": rp_z,
        "root_quat_w": rq_z,
        "body_pos_w": bp_z,
        "overlap_err": seam_info["overlap_err"],
        "seam_jump": seam_info["seam_jump"],
        "interior_step": seam_info["interior_step"],
    }


def _pack_two_chunk_cond(
    policy: StochasticDDIMPolicy, pipeline: Any, batch_gpu: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build stacked [c0, c1] condition tensors -> (2B, 120, 1024), (2B,)."""
    bsz = batch_gpu["obs_c0"].shape[0]
    device = batch_gpu["obs_c0"].device
    b_both = {
        "obs": torch.cat([batch_gpu["obs_c0"], batch_gpu["obs_c1"]], dim=0),
        "bbx_xys": torch.cat([batch_gpu["bbx_c0"], batch_gpu["bbx_c1"]], dim=0),
        "K_fullimg": torch.cat([batch_gpu["K_c0"], batch_gpu["K_c1"]], dim=0),
        "cam_angvel": torch.cat([batch_gpu["ang_c0"], batch_gpu["ang_c1"]], dim=0),
        "f_imgseq": torch.cat([batch_gpu["feat_c0"], batch_gpu["feat_c1"]], dim=0),
        "mask": {"f_imgseq": torch.cat([batch_gpu["f_imgseq_mask"], batch_gpu["f_imgseq_mask"]], dim=0)},
        "length": torch.full((2 * bsz,), STITCHED_WINDOW, dtype=torch.long, device=device),
    }
    f_cond, length, _ = policy.build_condition(pipeline, b_both)
    return f_cond, length


def _save_checkpoint(
    path: Path,
    latest_link: Path,
    step: int,
    pipeline: Any,
    denoiser_module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    provenance: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.denoiser3d.load_state_dict(denoiser_module.state_dict())
    full_state = {f"pipeline.{k}": v.detach().cpu() for k, v in pipeline.state_dict().items()}
    payload = {
        "ppo_step": int(step),
        "policy_version": POLICY_VERSION_STITCHED,
        "reward_version": REWARD_VERSION_V3,
        "imgfeat_encoder": "sam3d",
        "model_state_dict": full_state,
        "denoiser_state_dict": {k: v.detach().cpu() for k, v in denoiser_module.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(),
        },
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "provenance": provenance,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    # Symlink latest.pt instead of duplicating a 3.2 GB file on shared storage.
    tmp_link = latest_link.with_suffix(".pt.linktmp")
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(path.name)
    os.replace(tmp_link, latest_link)


@torch.no_grad()
def run_stitched_validation(
    pipeline: Any,
    denoiser: torch.nn.Module,
    policy: StochasticDDIMPolicy,
    val_loader: DataLoader,
    tracker_pool: PersistentTrackerPool,
    reward_weights: RewardWeights,
    device: torch.device,
    seed: int,
    eval_ddim_steps: int = 50,
    eval_ddim_eta: float = 0.0,
) -> dict[str, float]:
    """Evaluate deployment DDIM-50 eta=0.0 on held-out 239-frame stitched pairs."""
    denoiser.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    acc: dict[str, list[float]] = {
        "valid": [], "succ": [], "surv": [], "rerr": [], "jmae": [],
        "seam_jump": [], "overlap_err": [], "seam_score": [],
        "fid": [], "vfid": [], "jvel": [], "ref_jmae": [], "ref_mpjpe": [], "rew": [],
    }
    for batch in val_loader:
        batch_gpu = {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        f_cond, length = _pack_two_chunk_cond(policy, pipeline, batch_gpu)
        n2, seq_len, _ = f_cond.shape
        x_init = torch.randn((n2, seq_len, G1_PRED_X_DIM), device=device, dtype=torch.float32, generator=gen)
        step_noises = (
            [torch.randn((n2, seq_len, G1_PRED_X_DIM), device=device, dtype=torch.float32, generator=gen) for _ in range(eval_ddim_steps)]
            if eval_ddim_eta > 0.0
            else None
        )
        x0_both = sample_ddim_paired(
            pipeline=pipeline,
            denoiser=denoiser,
            f_cond=f_cond,
            length=length,
            x_init=x_init,
            ddim_steps=eval_ddim_steps,
            eta=eval_ddim_eta,
            all_valid_length=True,
            use_bf16=policy.use_bf16,
            step_noises=step_noises,
        )
        stitched = decode_and_stitch_zup(
            pipeline, x0_both, batch_gpu["ang_c0"], batch_gpu["ang_c1"], batch_gpu["root_pos_0_yup"]
        )
        bsz = batch_gpu["obs_c0"].shape[0]
        jp = stitched["joint_pos"].cpu().numpy()
        rp = stitched["root_pos_w"].cpu().numpy()
        rq = stitched["root_quat_w"].cpu().numpy()
        bp = stitched["body_pos_w"].cpu().numpy()
        ov_err = stitched["overlap_err"].cpu().numpy()
        s_jump = stitched["seam_jump"].cpu().numpy()
        i_step = stitched["interior_step"].cpu().numpy()
        lens = np.full(bsz, STITCHED_TOTAL_LEN, dtype=np.int64)
        fps_np = batch["fps"].numpy()

        trk_res = tracker_pool.evaluate_batch(jp, rp, rq, bp, lens, fps=fps_np)
        scored = compute_sample_rewards(
            tracker_results=trk_res,
            pred_joint_pos=jp,
            pred_body_pos_z=bp,
            gt_joint_pos=batch["gt_joint_pos"].numpy(),
            gt_body_pos_z=batch["gt_body_pos_z"].numpy(),
            lengths=lens,
            weights=reward_weights,
            fps=fps_np,
            seam_overlap_err=ov_err,
            seam_joint_jump=s_jump,
            interior_joint_step=i_step,
        )
        acc["valid"].extend(scored["is_valid"].tolist())
        acc["succ"].extend(scored["is_success"].tolist())
        acc["surv"].extend(scored["survival_ratio"].tolist())
        acc["rerr"].extend(scored["exec_root_pos_err_m"].tolist())
        acc["jmae"].extend(scored["exec_joint_mae_rad"].tolist())
        acc["seam_jump"].extend(s_jump.tolist())
        acc["overlap_err"].extend(ov_err.tolist())
        acc["seam_score"].extend(scored["seam_score"].tolist())
        acc["fid"].extend(scored["fidelity"].tolist())
        acc["vfid"].extend(scored["vel_fidelity"].tolist())
        acc["jvel"].extend(scored["mean_joint_vel_rad_s"].tolist())
        acc["ref_jmae"].extend(scored["ref_joint_mae_rad"].tolist())
        acc["ref_mpjpe"].extend(scored["ref_body_mpjpe_m"].tolist())
        acc["rew"].extend(scored["reward"].tolist())

    valid_arr = np.asarray(acc["valid"], dtype=np.bool_)
    def _vm(k: str) -> float:
        a = np.asarray(acc[k], dtype=np.float64)
        return float(np.nanmean(a[valid_arr])) if valid_arr.any() else 0.0

    n = max(len(acc["succ"]), 1)
    return {
        "val/success_rate": float(np.sum(acc["succ"])) / n,
        "val/survival_ratio": float(np.sum(acc["surv"])) / n,
        "val/exec_root_pos_err_m": _vm("rerr"),
        "val/exec_joint_mae_rad": _vm("jmae"),
        "val/seam_joint_jump_rad": _vm("seam_jump"),
        "val/seam_overlap_err_rad": _vm("overlap_err"),
        "val/seam_score": float(np.sum(acc["seam_score"])) / n,
        "val/fidelity": float(np.sum(acc["fid"])) / n,
        "val/vel_fidelity": float(np.sum(acc["vfid"])) / n,
        "val/mean_joint_vel_rad_s": _vm("jvel"),
        "val/ref_joint_mae_rad": _vm("ref_jmae"),
        "val/ref_body_mpjpe_m": _vm("ref_mpjpe"),
        "val/invalid_rate": 1.0 - (float(valid_arr.sum()) / n),
        "val/reward_mean": float(np.sum(acc["rew"])) / n,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", default=["bedlam2", "h36m", "bedlam"])
    parser.add_argument("--g1-pth-name", type=str, default="g1_new.pth")
    parser.add_argument("--base-ckpt", type=Path, default=REPO_ROOT / "vith_offline_g1.ckpt")
    parser.add_argument("--init-ckpt", type=Path, default=None, help="Optional warm-start weights (KL stays anchored to --base-ckpt)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--hard-mining-clips", type=Path, default=None, help="Optional train clips.jsonl to exclude untrackable_gt & upweight hard sequences")
    parser.add_argument("--hard-weight-mult", type=float, default=2.5)

    # Frozen tracker boundary
    parser.add_argument("--tracker-python", type=Path, default=Path(os.environ.get("TRACKER_PYTHON_BIN", sys.executable)))
    parser.add_argument("--policy-onnx", type=Path, default=REPO_ROOT / "inputs/checkpoints/g1_tracker/model_step_020000_g1.onnx")
    parser.add_argument("--xml-path", type=Path, default=REPO_ROOT / "g1_tracker/assets/g1/rollout/scene_29dof_mujoco.xml")
    parser.add_argument("--grounding-xml", type=Path, default=REPO_ROOT / "g1_tracker/assets/g1/grounding/mjcf/g1_29dof_rev_1_0.xml")
    parser.add_argument("--tracker-workers", type=int, default=8)

    # Stitched PPO v3 hyperparameters
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--ddim-eta", type=float, default=0.7)
    parser.add_argument("--prompts-per-rank", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--update-steps-per-epoch", type=int, default=10)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.015)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min-lr", type=float, default=4e-6)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--val-max-clips", type=int, default=64)
    parser.add_argument("--eval-ddim-steps", type=int, default=50)
    parser.add_argument("--eval-ddim-eta", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    # Reward v3 weights
    parser.add_argument("--w-success", type=float, default=1.0)
    parser.add_argument("--w-survival", type=float, default=0.6)
    parser.add_argument("--w-exec-root", type=float, default=1.2)
    parser.add_argument("--w-exec-joint", type=float, default=0.8)
    parser.add_argument("--w-seam", type=float, default=0.8)
    parser.add_argument("--w-fidelity", type=float, default=0.4)
    parser.add_argument("--w-vel-fidelity", type=float, default=0.5)

    # Hopper acceleration
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--compile-rollout", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead", choices=("default", "reduce-overhead"))
    parser.add_argument("--num-data-workers", type=int, default=2)

    # SwanLab monitoring
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--swanlab-project", type=str, default="vit-h-ppo")
    parser.add_argument("--swanlab-exp-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = _init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    _seed_everything(args.seed + rank)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = args.output_dir.expanduser().resolve()
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load frozen base pipeline strictly; optionally warm-start denoiser3d from --init-ckpt.
    overrides = ["network.latent_dim=1024", "network.num_layers=16", "network.num_heads=16"]
    pipeline = load_pipeline(str(args.base_ckpt), exp="offline/g1_training_set_exp1", device=str(device), overrides=overrides)
    frozen_aux_prefixes = ("pred_cam_head.", "static_conf_head.")
    denoiser_trainable_names = {
        name for name, p in pipeline.denoiser3d.named_parameters()
        if p.requires_grad and not name.startswith(frozen_aux_prefixes)
    }
    for param in pipeline.parameters():
        param.requires_grad_(False)
    for name, param in pipeline.denoiser3d.named_parameters():
        if name in denoiser_trainable_names:
            param.requires_grad_(True)
    pipeline.eval()

    # Keep base_denoiser strictly pinned to original pretrained base_ckpt!
    base_denoiser = copy.deepcopy(pipeline.denoiser3d).to(device).eval()
    for param in base_denoiser.parameters():
        param.requires_grad_(False)

    if args.init_ckpt is not None and args.resume is None:
        init_payload = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        if "denoiser_state_dict" in init_payload:
            pipeline.denoiser3d.load_state_dict(init_payload["denoiser_state_dict"], strict=True)
        else:
            init_pipe = load_pipeline(str(args.init_ckpt), exp="offline/g1_training_set_exp1", device="cpu", overrides=overrides)
            pipeline.denoiser3d.load_state_dict(init_pipe.denoiser3d.state_dict(), strict=True)
            del init_pipe
        if is_main:
            print(f"[PPOv3] Warm-started denoiser3d from {args.init_ckpt} (KL anchored to {args.base_ckpt})", flush=True)

    denoiser_raw = pipeline.denoiser3d.to(device).eval()
    trainable_params = [p for p in denoiser_raw.parameters() if p.requires_grad]
    for param in trainable_params:
        canon_stride = torch.empty_like(param.data, memory_format=torch.contiguous_format).stride()
        if param.data.stride() != canon_stride:
            param.data = param.data.clone(memory_format=torch.contiguous_format)

    denoiser_train: torch.nn.Module = (
        DDP(denoiser_raw, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if world_size > 1
        else denoiser_raw
    )
    denoiser_rollout: torch.nn.Module = (
        torch.compile(denoiser_raw, mode=args.compile_mode)
        if args.compile_rollout and device.type == "cuda"
        else denoiser_raw
    )
    base_denoiser_rollout: torch.nn.Module = (
        torch.compile(base_denoiser, mode=args.compile_mode)
        if args.compile_rollout and device.type == "cuda"
        else base_denoiser
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=(device.type == "cuda"),
    )

    start_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        denoiser_raw.load_state_dict(ckpt["denoiser_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt["ppo_step"])
        if is_main:
            print(f"[PPOv3] Resumed from {args.resume} at step {start_step}", flush=True)

    policy = StochasticDDIMPolicy(
        pipeline=pipeline,
        ddim_steps=args.ddim_steps,
        eta=args.ddim_eta,
        use_bf16=(not args.no_bf16),
        device=device,
    )
    reward_weights = make_reward_weights_v3(
        w_success=args.w_success,
        w_survival=args.w_survival,
        w_exec_root=args.w_exec_root,
        w_exec_joint=args.w_exec_joint,
        w_seam=args.w_seam,
        w_fidelity=args.w_fidelity,
        w_vel_fidelity=args.w_vel_fidelity,
    )

    # 2. Build 239-frame stitched train & val datasets.
    train_ds, train_meta_keys = build_stitched_ppo_dataset(
        dataset_root=args.dataset_root,
        sources=args.sources,
        split="train",
        g1_pth_name=args.g1_pth_name,
        split_seed=args.seed,
    )
    untrackable_keys: set[tuple[str, str]] = set()
    hard_ab_keys: set[tuple[str, str]] = set()
    if args.hard_mining_clips is not None:
        untrackable_keys, hard_ab_keys = load_key_buckets_from_mining_clips(args.hard_mining_clips)
    train_sampler = StitchedKeyWeightedDistributedSampler(
        meta_keys=train_meta_keys,
        untrackable_keys=untrackable_keys,
        hard_ab_keys=hard_ab_keys,
        num_replicas=world_size,
        rank=rank,
        hard_weight_mult=args.hard_weight_mult,
        num_samples_per_epoch=max(len(train_ds), args.max_steps * args.prompts_per_rank * world_size),
        seed=args.seed,
    )
    if is_main:
        print(f"[StitchedSampler] {train_sampler.stats} hard_mult={args.hard_weight_mult}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.prompts_per_rank,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_data_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = None
    val_clips = 0
    if is_main:
        val_ds_full, _ = build_stitched_ppo_dataset(
            dataset_root=args.dataset_root,
            sources=args.sources,
            split="val",
            g1_pth_name=args.g1_pth_name,
            split_seed=args.seed,
        )
        if args.val_max_clips > 0 and len(val_ds_full) > args.val_max_clips:
            rng = np.random.default_rng(args.seed + 77)
            pick = np.sort(rng.choice(len(val_ds_full), size=args.val_max_clips, replace=False)).tolist()
            val_ds = Subset(val_ds_full, pick)
        else:
            val_ds = val_ds_full
        val_clips = len(val_ds)
        val_loader = DataLoader(
            val_ds,
            batch_size=8,
            shuffle=False,
            num_workers=min(2, args.num_data_workers),
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    provenance = {
        "git_commit": _git_commit(),
        "policy_version": POLICY_VERSION_STITCHED,
        "reward_version": REWARD_VERSION_V3,
        "base_ckpt": str(args.base_ckpt.resolve()),
        "base_ckpt_sha256": _sha256_file(args.base_ckpt.resolve()),
        "init_ckpt": str(args.init_ckpt.resolve()) if args.init_ckpt else None,
        "tracker_onnx": str(args.policy_onnx.resolve()),
        "tracker_onnx_sha256": _sha256_file(args.policy_onnx.resolve()),
        "sources": list(args.sources),
        "g1_pth_name": args.g1_pth_name,
        "world_size": world_size,
        "train_pairs": len(train_ds),
        "val_pairs": val_clips,
    }
    if is_main:
        (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(
            f"[PPOv3] Ready: world_size={world_size}, train_pairs={len(train_ds)}, val_pairs={val_clips}, "
            f"prompts/rank={args.prompts_per_rank}, group_size={args.group_size}, "
            f"global_stitched_rollouts/step={world_size * args.prompts_per_rank * args.group_size}",
            flush=True,
        )

    swan_run = None
    if is_main and args.swanlab:
        import swanlab

        api_key = os.environ.get("SWANLAB_API_KEY")
        if api_key:
            swanlab.login(api_key=api_key, save=False)
        exp_name = args.swanlab_exp_name or f"vith-ppo-v3-stitched-{time.strftime('%m%d-%H%M')}"
        swan_run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=exp_name,
            logdir=str(output_dir / "swanlog"),
            config={
                **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "policy_version": POLICY_VERSION_STITCHED,
                "reward_version": REWARD_VERSION_V3,
                "world_size": world_size,
                "train_pairs": len(train_ds),
                "val_pairs": val_clips,
            },
        )
        print(f"[SwanLab] Initialized project={args.swanlab_project} exp={exp_name}", flush=True)

    tracker_pool = PersistentTrackerPool(
        tracker_python=args.tracker_python,
        policy_onnx=args.policy_onnx,
        xml_path=args.xml_path,
        grounding_xml=args.grounding_xml,
        num_workers=args.tracker_workers,
    )
    if is_main:
        print(f"[TrackerPool] Started {args.tracker_workers} persistent workers/rank (protocol v{tracker_pool.protocol_version})", flush=True)

    metrics_jsonl = (output_dir / "metrics.jsonl").open("a", encoding="utf-8") if is_main else None
    try:
        if is_main and val_loader is not None:
            val_metrics = run_stitched_validation(
                pipeline=pipeline,
                denoiser=denoiser_raw,
                policy=policy,
                val_loader=val_loader,
                tracker_pool=tracker_pool,
                reward_weights=reward_weights,
                device=device,
                seed=args.seed + 10000,
                eval_ddim_steps=args.eval_ddim_steps,
                eval_ddim_eta=args.eval_ddim_eta,
            )
            print(f"[VAL step={start_step}] {json.dumps(val_metrics)}", flush=True)
            if swan_run is not None:
                swan_run.log(val_metrics, step=start_step)
            if metrics_jsonl is not None:
                metrics_jsonl.write(json.dumps({"step": start_step, **val_metrics}) + "\n")
                metrics_jsonl.flush()
        if world_size > 1:
            dist.barrier()

        epoch_idx = 0
        loader_iter = iter(train_loader)
        replay_verified = False

        for step in range(start_step + 1, args.max_steps + 1):
            t_step_start = time.perf_counter()
            try:
                batch = next(loader_iter)
            except StopIteration:
                epoch_idx += 1
                train_sampler.set_epoch(epoch_idx)
                loader_iter = iter(train_loader)
                batch = next(loader_iter)

            batch_gpu = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            # --- A. Joint Two-Window Stochastic DDIM Rollout + Exact Stitching ---
            t0 = time.perf_counter()
            f_cond_both, len_both = _pack_two_chunk_cond(policy, pipeline, batch_gpu)
            rollout_gen = torch.Generator(device=device).manual_seed(args.seed + step * 997 + rank * 31)
            rollout = policy.rollout_group(
                denoiser=denoiser_rollout,
                base_denoiser=base_denoiser_rollout,
                f_cond=f_cond_both,
                length=len_both,
                group_size=args.group_size,
                all_valid_length=True,
                generator=rollout_gen,
            )
            f_cond_rep = f_cond_both.repeat_interleave(args.group_size, dim=0)
            len_rep = len_both.repeat_interleave(args.group_size, dim=0)
            n_stitched = args.prompts_per_rank * args.group_size
            old_logp_joint = 0.5 * (rollout.old_log_probs[:, :n_stitched] + rollout.old_log_probs[:, n_stitched:])

            ang_c0_rep = batch_gpu["ang_c0"].repeat_interleave(args.group_size, dim=0)
            ang_c1_rep = batch_gpu["ang_c1"].repeat_interleave(args.group_size, dim=0)
            root0_rep = batch_gpu["root_pos_0_yup"].repeat_interleave(args.group_size, dim=0)
            stitched = decode_and_stitch_zup(pipeline, rollout.x_0, ang_c0_rep, ang_c1_rep, root0_rep)
            torch.cuda.synchronize()
            rollout_s = time.perf_counter() - t0

            # --- B. Mandatory Joint Replay Self-Check ---
            if not replay_verified:
                with torch.no_grad():
                    chk_lp_both, chk_kl_both = policy.replay_step(
                        denoiser=denoiser_raw,
                        rollout=rollout,
                        stoch_idx=0,
                        f_cond_rep=f_cond_rep,
                        length_rep=len_rep,
                        all_valid_length=True,
                    )
                    chk_lp_joint = 0.5 * (chk_lp_both[:n_stitched] + chk_lp_both[n_stitched:])
                    chk_kl_joint = 0.5 * (chk_kl_both[:n_stitched] + chk_kl_both[n_stitched:])
                    max_err = float((chk_lp_joint - old_logp_joint[0]).abs().max().item())
                    ratio_dev = float((torch.exp(chk_lp_joint - old_logp_joint[0]) - 1.0).abs().max().item())
                if is_main:
                    print(
                        f"[PPOv3 REPLAY CHECK] max|new_logp - old_logp|={max_err:.2e}, "
                        f"max|ratio - 1|={ratio_dev:.2e}, init_kl={float(chk_kl_joint.mean().item()):.2e}",
                        flush=True,
                    )
                if max_err > 5e-3 or ratio_dev > 5e-3:
                    raise RuntimeError(f"Joint stitched replay failed: logp_err={max_err}, ratio_dev={ratio_dev}")
                replay_verified = True

            # --- C. Parallel Frozen SONIC/MuJoCo 239-Frame Tracker Feedback + reward_v3 ---
            t1 = time.perf_counter()
            jp_np = stitched["joint_pos"].cpu().numpy()
            rp_np = stitched["root_pos_w"].cpu().numpy()
            rq_np = stitched["root_quat_w"].cpu().numpy()
            bp_np = stitched["body_pos_w"].cpu().numpy()
            ov_np = stitched["overlap_err"].cpu().numpy()
            sj_np = stitched["seam_jump"].cpu().numpy()
            is_np = stitched["interior_step"].cpu().numpy()
            lens_np = np.full(n_stitched, STITCHED_TOTAL_LEN, dtype=np.int64)
            fps_rep = np.repeat(batch["fps"].numpy(), args.group_size, axis=0)
            gt_jp_rep = np.repeat(batch["gt_joint_pos"].numpy(), args.group_size, axis=0)
            gt_bp_rep = np.repeat(batch["gt_body_pos_z"].numpy(), args.group_size, axis=0)

            trk_res = tracker_pool.evaluate_batch(jp_np, rp_np, rq_np, bp_np, lens_np, fps=fps_rep)
            scored = compute_sample_rewards(
                tracker_results=trk_res,
                pred_joint_pos=jp_np,
                pred_body_pos_z=bp_np,
                gt_joint_pos=gt_jp_rep,
                gt_body_pos_z=gt_bp_rep,
                lengths=lens_np,
                weights=reward_weights,
                fps=fps_rep,
                seam_overlap_err=ov_np,
                seam_joint_jump=sj_np,
                interior_joint_step=is_np,
            )
            tracker_s = time.perf_counter() - t1

            rewards_t = torch.from_numpy(scored["reward"]).to(device=device, dtype=torch.float32)
            advantages = compute_group_advantages(rewards_t, group_size=args.group_size)

            # Warmup + cosine LR schedule
            if args.min_lr > 0.0 or args.warmup_steps > 0:
                if args.warmup_steps > 0 and step <= args.warmup_steps:
                    cur_lr = args.lr * (step / max(args.warmup_steps, 1))
                else:
                    prog = min(max((step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1), 0.0), 1.0)
                    cur_lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * prog))
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr
            else:
                cur_lr = args.lr

            # --- D. Differentiable Joint Two-Window PPO Update ---
            t2 = time.perf_counter()
            denoiser_raw.eval()
            num_stoch = len(rollout.t_indices)
            steps_per_epoch = min(args.update_steps_per_epoch, num_stoch)
            ppo_losses, kl_vals, ratios_mean, clip_fracs, grad_norms = [], [], [], [], []

            for _ in range(args.ppo_epochs):
                perm = torch.randperm(num_stoch).tolist()[:steps_per_epoch]
                optimizer.zero_grad(set_to_none=True)
                for idx_pos, stoch_idx in enumerate(perm):
                    sync_ctx = (
                        denoiser_train.no_sync()
                        if (world_size > 1 and idx_pos < len(perm) - 1)
                        else torch.enable_grad()
                    )
                    with sync_ctx:
                        new_lp_both, kl_both = policy.replay_step(
                            denoiser=denoiser_train,
                            rollout=rollout,
                            stoch_idx=stoch_idx,
                            f_cond_rep=f_cond_rep,
                            length_rep=len_rep,
                            all_valid_length=True,
                        )
                        new_lp_joint = 0.5 * (new_lp_both[:n_stitched] + new_lp_both[n_stitched:])
                        kl_joint = 0.5 * (kl_both[:n_stitched] + kl_both[n_stitched:])
                        log_ratio = (new_lp_joint - old_logp_joint[stoch_idx]).clamp(-10.0, 10.0)
                        ratio = torch.exp(log_ratio)
                        surr1 = ratio * advantages
                        surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * advantages
                        pg_loss = -torch.minimum(surr1, surr2).mean()
                        kl_mean = kl_joint.mean()
                        loss = (pg_loss + args.kl_coef * kl_mean) / len(perm)
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Non-finite PPOv3 loss at step {step}: {loss.item()}")
                        loss.backward()

                        ppo_losses.append(float(pg_loss.item()))
                        kl_vals.append(float(kl_mean.item()))
                        ratios_mean.append(float(ratio.mean().item()))
                        clip_fracs.append(float(((ratio - 1.0).abs() > args.clip_eps).float().mean().item()))

                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip).item())
                if not math.isfinite(grad_norm):
                    raise RuntimeError(f"Non-finite grad norm at step {step}: {grad_norm}")
                optimizer.step()
                grad_norms.append(grad_norm)

            torch.cuda.synchronize()
            update_s = time.perf_counter() - t2
            total_s = time.perf_counter() - t_step_start

            valid_mask = scored["is_valid"]
            def _vavg(arr: np.ndarray) -> float:
                return float(np.nanmean(arr[valid_mask])) if valid_mask.any() else 0.0

            step_metrics = {
                "train/lr": float(cur_lr),
                "train/reward_mean": _all_reduce_mean(float(scored["reward"].mean()), device, world_size),
                "train/reward_std": _all_reduce_mean(float(scored["reward"].std()), device, world_size),
                "train/success_rate": _all_reduce_mean(float(scored["is_success"].mean()), device, world_size),
                "train/survival_ratio": _all_reduce_mean(float(scored["survival_ratio"].mean()), device, world_size),
                "train/exec_root_pos_err_m": _all_reduce_mean(_vavg(scored["exec_root_pos_err_m"]), device, world_size),
                "train/exec_joint_mae_rad": _all_reduce_mean(_vavg(scored["exec_joint_mae_rad"]), device, world_size),
                "train/seam_joint_jump_rad": _all_reduce_mean(_vavg(sj_np), device, world_size),
                "train/seam_overlap_err_rad": _all_reduce_mean(_vavg(ov_np), device, world_size),
                "train/seam_score": _all_reduce_mean(float(scored["seam_score"].mean()), device, world_size),
                "train/fidelity": _all_reduce_mean(float(scored["fidelity"].mean()), device, world_size),
                "train/vel_fidelity": _all_reduce_mean(float(scored["vel_fidelity"].mean()), device, world_size),
                "train/mean_joint_vel_rad_s": _all_reduce_mean(_vavg(scored["mean_joint_vel_rad_s"]), device, world_size),
                "train/ref_joint_mae_rad": _all_reduce_mean(_vavg(scored["ref_joint_mae_rad"]), device, world_size),
                "train/pg_loss": _all_reduce_mean(float(np.mean(ppo_losses)), device, world_size),
                "train/kl_div": _all_reduce_mean(float(np.mean(kl_vals)), device, world_size),
                "train/ratio_mean": _all_reduce_mean(float(np.mean(ratios_mean)), device, world_size),
                "train/clip_frac": _all_reduce_mean(float(np.mean(clip_fracs)), device, world_size),
                "train/grad_norm": _all_reduce_mean(float(np.mean(grad_norms)), device, world_size),
                "perf/rollout_s": rollout_s,
                "perf/tracker_s": tracker_s,
                "perf/update_s": update_s,
                "perf/step_s": total_s,
            }

            if is_main:
                print(
                    f"[PPOv3 step={step:04d}/{args.max_steps}] "
                    f"rew={step_metrics['train/reward_mean']:.3f} "
                    f"succ={step_metrics['train/success_rate']:.3f} "
                    f"rerr={step_metrics['train/exec_root_pos_err_m']:.4f} "
                    f"jmae={step_metrics['train/exec_joint_mae_rad']:.4f} "
                    f"seam={step_metrics['train/seam_joint_jump_rad']:.4f} "
                    f"ov={step_metrics['train/seam_overlap_err_rad']:.4f} "
                    f"vfid={step_metrics['train/vel_fidelity']:.3f} "
                    f"kl={step_metrics['train/kl_div']:.3f} "
                    f"clip={step_metrics['train/clip_frac']:.3f} | "
                    f"roll={rollout_s:.2f}s trk={tracker_s:.2f}s upd={update_s:.2f}s",
                    flush=True,
                )
                if swan_run is not None:
                    swan_run.log(step_metrics, step=step)
                if metrics_jsonl is not None:
                    metrics_jsonl.write(json.dumps({"step": step, **step_metrics}) + "\n")
                    metrics_jsonl.flush()

            if args.eval_every > 0 and step % args.eval_every == 0:
                if is_main and val_loader is not None:
                    val_metrics = run_stitched_validation(
                        pipeline=pipeline,
                        denoiser=denoiser_raw,
                        policy=policy,
                        val_loader=val_loader,
                        tracker_pool=tracker_pool,
                        reward_weights=reward_weights,
                        device=device,
                        seed=args.seed + 10000,
                        eval_ddim_steps=args.eval_ddim_steps,
                        eval_ddim_eta=args.eval_ddim_eta,
                    )
                    print(f"[VAL step={step}] {json.dumps(val_metrics)}", flush=True)
                    if swan_run is not None:
                        swan_run.log(val_metrics, step=step)
                    if metrics_jsonl is not None:
                        metrics_jsonl.write(json.dumps({"step": step, **val_metrics}) + "\n")
                        metrics_jsonl.flush()
                if world_size > 1:
                    dist.barrier()

            if is_main and (step % args.save_every == 0 or step == args.max_steps):
                ckpt_path = output_dir / "checkpoints" / f"ppo_step_{step:05d}.pt"
                latest_path = output_dir / "checkpoints" / "latest.pt"
                _save_checkpoint(ckpt_path, latest_path, step, pipeline, denoiser_raw, optimizer, args, provenance)
                print(f"[CKPT] Saved {ckpt_path}", flush=True)
            if world_size > 1 and (step % args.save_every == 0 or step == args.max_steps):
                dist.barrier()
    finally:
        tracker_pool.close()
        if metrics_jsonl is not None:
            metrics_jsonl.close()
        if swan_run is not None:
            swan_run.finish()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
