#!/usr/bin/env python3
"""Diffusion-PPO (DDPO / group-normalized PPO) trainer for ViT-H on Hopper GPUs.

Supports single-GPU verification and multi-GPU DDP via `torchrun`, Hopper BF16
+ SDPA + torch.compile rollout acceleration, persistent cross-environment
SONIC/MuJoCo tracker workers, checkpoint save/resume, and SwanLab live tracking.
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_training import POLICY_VERSION, REWARD_VERSION, REWARD_VERSION_V2
from post_training.dataset import (
    HardClipDistributedSampler,
    build_hard_val_subset,
    build_video_ppo_dataset,
)
from post_training.eval_compare import sample_ddim_paired
from post_training.policy import G1_PRED_X_DIM, StochasticDDIMPolicy, decode_x0_to_g1_zup
from post_training.reward import (
    RewardWeights,
    compute_group_advantages,
    compute_sample_rewards,
    make_reward_weights_v2,
)
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


def _save_checkpoint(
    path: Path,
    step: int,
    pipeline: Any,
    denoiser_module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    provenance: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Build a full pipeline state_dict directly compatible with tools/infer_video.py & load_pipeline().
    pipeline.denoiser3d.load_state_dict(denoiser_module.state_dict())
    full_state = {f"pipeline.{k}": v.detach().cpu() for k, v in pipeline.state_dict().items()}
    payload = {
        "ppo_step": int(step),
        "policy_version": POLICY_VERSION,
        "reward_version": str(getattr(args, "reward_version", REWARD_VERSION)),
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
        "config": vars(args),
        "provenance": provenance,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def run_validation(
    pipeline: Any,
    denoiser: torch.nn.Module,
    base_denoiser: torch.nn.Module,
    policy: StochasticDDIMPolicy,
    val_loader: DataLoader,
    tracker_pool: PersistentTrackerPool,
    reward_weights: RewardWeights,
    device: torch.device,
    max_batches: int,
    seed: int,
    eval_ddim_steps: int = 50,
    eval_ddim_eta: float = 0.0,
    val_bucket_a_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate current policy on held-out validation clips (deployment DDIM-50 eta=0.0 by default)."""
    denoiser.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    all_valid_flags: list[bool] = []
    all_succ: list[bool] = []
    all_surv: list[float] = []
    all_exec_trk: list[float] = []
    all_exec_rerr: list[float] = []
    all_exec_jmae: list[float] = []
    all_fid: list[float] = []
    all_vfid: list[float] = []
    all_jvel: list[float] = []
    all_ref_jmae: list[float] = []
    all_ref_mpjpe: list[float] = []
    all_rew: list[float] = []

    for batch_idx, batch in enumerate(val_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch_gpu = {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        batch_gpu["mask"] = {"f_imgseq": batch_gpu["f_imgseq_mask"]}
        f_cond, length, all_valid = policy.build_condition(pipeline, batch_gpu)
        bsz, seq_len, _ = f_cond.shape
        x_init = torch.randn((bsz, seq_len, G1_PRED_X_DIM), device=device, dtype=torch.float32, generator=gen)
        step_noises = (
            [torch.randn((bsz, seq_len, G1_PRED_X_DIM), device=device, dtype=torch.float32, generator=gen) for _ in range(eval_ddim_steps)]
            if eval_ddim_eta > 0.0
            else None
        )
        x_0 = sample_ddim_paired(
            pipeline=pipeline,
            denoiser=denoiser,
            f_cond=f_cond,
            length=length,
            x_init=x_init,
            ddim_steps=eval_ddim_steps,
            eta=eval_ddim_eta,
            all_valid_length=all_valid,
            use_bf16=policy.use_bf16,
            step_noises=step_noises,
        )
        decoded = decode_x0_to_g1_zup(
            pipeline=pipeline,
            x_0=x_0,
            cam_angvel=batch_gpu["cam_angvel"],
            length=length,
            root_pos_0_yup=batch_gpu["root_pos_0_yup"],
        )
        jp = decoded["joint_pos"].cpu().numpy()
        rp = decoded["root_pos_w"].cpu().numpy()
        rq = decoded["root_quat_w"].cpu().numpy()
        bp = decoded["body_pos_w"].cpu().numpy()
        lens = length.cpu().numpy()
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
        )
        all_valid_flags.extend(scored["is_valid"].tolist())
        all_succ.extend(scored["is_success"].tolist())
        all_surv.extend(scored["survival_ratio"].tolist())
        all_exec_trk.extend(scored["exec_track"].tolist())
        all_exec_rerr.extend(scored["exec_root_pos_err_m"].tolist())
        all_exec_jmae.extend(scored["exec_joint_mae_rad"].tolist())
        all_fid.extend(scored["fidelity"].tolist())
        all_vfid.extend(scored["vel_fidelity"].tolist())
        all_jvel.extend(scored["mean_joint_vel_rad_s"].tolist())
        all_ref_jmae.extend(scored["ref_joint_mae_rad"].tolist())
        all_ref_mpjpe.extend(scored["ref_body_mpjpe_m"].tolist())
        all_rew.extend(scored["reward"].tolist())

    n = max(len(all_succ), 1)
    valid_arr = np.asarray(all_valid_flags, dtype=np.bool_)
    def _vmean(arr_list: list[float]) -> float:
        a = np.asarray(arr_list, dtype=np.float64)
        return float(np.nanmean(a[valid_arr])) if valid_arr.any() else 0.0

    out = {
        "val/success_rate": float(np.sum(all_succ)) / n,
        "val/survival_ratio": float(np.sum(all_surv)) / n,
        "val/exec_track": float(np.sum(all_exec_trk)) / n,
        "val/exec_root_pos_err_m": _vmean(all_exec_rerr),
        "val/exec_joint_mae_rad": _vmean(all_exec_jmae),
        "val/fidelity": float(np.sum(all_fid)) / n,
        "val/vel_fidelity": float(np.sum(all_vfid)) / n,
        "val/mean_joint_vel_rad_s": _vmean(all_jvel),
        "val/ref_joint_mae_rad": _vmean(all_ref_jmae),
        "val/ref_body_mpjpe_m": _vmean(all_ref_mpjpe),
        "val/invalid_rate": 1.0 - (float(valid_arr.sum()) / n),
        "val/reward_mean": float(np.sum(all_rew)) / n,
    }
    if val_bucket_a_mask is not None and len(all_succ) > 0:
        m = val_bucket_a_mask[: len(all_succ)]
        if m.any():
            out["val_bucket_a/success_rate"] = float(np.mean(np.asarray(all_succ, dtype=np.float64)[m]))
            out["val_bucket_a/survival_ratio"] = float(np.mean(np.asarray(all_surv, dtype=np.float64)[m]))
            out["val_bucket_a/exec_root_pos_err_m"] = float(np.nanmean(np.asarray(all_exec_rerr, dtype=np.float64)[m]))
        mb = ~m
        if mb.any():
            out["val_bucket_b/success_rate"] = float(np.mean(np.asarray(all_succ, dtype=np.float64)[mb]))
            out["val_bucket_b/exec_root_pos_err_m"] = float(np.nanmean(np.asarray(all_exec_rerr, dtype=np.float64)[mb]))
            out["val_bucket_b/exec_joint_mae_rad"] = float(np.nanmean(np.asarray(all_exec_jmae, dtype=np.float64)[mb]))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", default=["h36m", "bedlam", "bedlam2"])
    parser.add_argument("--g1-pth-name", type=str, default="g1_new.pth")
    parser.add_argument("--base-ckpt", type=Path, default=REPO_ROOT / "vith_offline_g1.ckpt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)

    # Fixed tracker boundary (kept strictly frozen for base-vs-PPO comparisons)
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
    parser.add_argument("--tracker-workers", type=int, default=8)

    # Diffusion-PPO hyperparameters
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--ddim-eta", type=float, default=0.7)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--prompts-per-rank", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--update-steps-per-epoch", type=int, default=10)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.04)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-ddim-steps", type=int, default=50)
    parser.add_argument("--eval-ddim-eta", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # Hard-clip weighted sampling & hard validation index (from mine_hard_clips.py)
    parser.add_argument("--hard-index", type=Path, default=None)
    parser.add_argument("--share-a", type=float, default=0.20)
    parser.add_argument("--share-b", type=float, default=0.55)
    parser.add_argument("--share-c", type=float, default=0.25)
    parser.add_argument("--val-hard-index", type=Path, default=None)
    parser.add_argument("--val-max-bucket-b", type=int, default=107)

    # Reward weights (`reward_v1` or `reward_v2`)
    parser.add_argument(
        "--reward-version",
        type=str,
        default=REWARD_VERSION,
        choices=(REWARD_VERSION, REWARD_VERSION_V2),
    )
    parser.add_argument("--w-success", type=float, default=1.0)
    parser.add_argument("--w-survival", type=float, default=0.5)
    parser.add_argument("--w-exec-track", type=float, default=0.3)
    parser.add_argument("--w-exec-root", type=float, default=1.2)
    parser.add_argument("--w-exec-joint", type=float, default=0.8)
    parser.add_argument("--w-fidelity", type=float, default=1.0)
    parser.add_argument("--w-vel-fidelity", type=float, default=0.3)

    # Hopper H100 optimizations
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--compile-rollout", action="store_true")
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=("default", "reduce-overhead"),
    )
    parser.add_argument("--num-data-workers", type=int, default=4)

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

    # 1. Load pretrained ViT-H base pipeline strictly & keep a frozen base reference denoiser.
    pipeline = load_pipeline(
        str(args.base_ckpt),
        exp="offline/g1_training_set_exp1",
        device=str(device),
        overrides=["network.latent_dim=1024", "network.num_layers=16", "network.num_heads=16"],
    )
    # Preserve denoiser3d's own trainable mask (keeping fixed sinusoidal
    # sequence_pos_encoder.pe and non-policy auxiliary heads pred_cam_head /
    # static_conf_head frozen so every DDP parameter participates in pred_x_start).
    frozen_aux_prefixes = ("pred_cam_head.", "static_conf_head.")
    denoiser_trainable_names = {
        name
        for name, p in pipeline.denoiser3d.named_parameters()
        if p.requires_grad and not name.startswith(frozen_aux_prefixes)
    }
    for param in pipeline.parameters():
        param.requires_grad_(False)
    for name, param in pipeline.denoiser3d.named_parameters():
        if name in denoiser_trainable_names:
            param.requires_grad_(True)
    pipeline.eval()

    base_denoiser = copy.deepcopy(pipeline.denoiser3d).to(device).eval()
    for param in base_denoiser.parameters():
        param.requires_grad_(False)

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
            print(f"[PPO] Resumed from {args.resume} at step {start_step}", flush=True)

    policy = StochasticDDIMPolicy(
        pipeline=pipeline,
        ddim_steps=args.ddim_steps,
        eta=args.ddim_eta,
        use_bf16=(not args.no_bf16),
        device=device,
    )
    if args.reward_version == REWARD_VERSION_V2:
        reward_weights = make_reward_weights_v2(
            w_success=args.w_success,
            w_survival=args.w_survival,
            w_exec_root=args.w_exec_root,
            w_exec_joint=args.w_exec_joint,
            w_fidelity=args.w_fidelity,
            w_vel_fidelity=args.w_vel_fidelity,
        )
    else:
        reward_weights = RewardWeights(
            version=REWARD_VERSION,
            w_success=args.w_success,
            w_survival=args.w_survival,
            w_exec_track=args.w_exec_track,
            w_fidelity=args.w_fidelity,
        )

    # 2. Build video-conditioned train & held-out val splits.
    use_hard_mining = args.hard_index is not None
    train_ds = build_video_ppo_dataset(
        dataset_root=args.dataset_root,
        sources=args.sources,
        split="train",
        g1_pth_name=args.g1_pth_name,
        split_seed=args.seed,
        deterministic_windows=use_hard_mining,
    )
    val_ds_full = (
        build_video_ppo_dataset(
            dataset_root=args.dataset_root,
            sources=args.sources,
            split="val",
            g1_pth_name=args.g1_pth_name,
            split_seed=args.seed,
        )
        if is_main
        else None
    )
    val_bucket_a_mask: np.ndarray | None = None
    if val_ds_full is not None and args.val_hard_index is not None:
        val_ds, val_bucket_a_mask = build_hard_val_subset(
            val_ds_full, args.val_hard_index, max_bucket_b=args.val_max_bucket_b, seed=args.seed
        )
    else:
        val_ds = val_ds_full

    if use_hard_mining:
        train_sampler: Any = HardClipDistributedSampler(
            hard_index_path=args.hard_index,
            dataset_len=len(train_ds),
            num_replicas=world_size,
            rank=rank,
            share_a=args.share_a,
            share_b=args.share_b,
            share_c=args.share_c,
            num_samples_per_epoch=max(len(train_ds), args.max_steps * args.prompts_per_rank * world_size),
            seed=args.seed,
        )
        if is_main:
            print(
                f"[HardSampler] buckets={train_sampler.bucket_counts} "
                f"excluded_untrackable_gt={train_sampler.excluded_untrackable} "
                f"shares=({args.share_a:.2f}, {args.share_b:.2f}, {args.share_c:.2f})",
                flush=True,
            )
    else:
        train_sampler = (
            DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
            if world_size > 1
            else None
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.prompts_per_rank,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_data_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_batch_size = 16 if args.val_hard_index is not None else args.prompts_per_rank
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=min(2, args.num_data_workers),
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        if val_ds is not None
        else None
    )
    val_clips = len(val_ds) if val_ds is not None else 0

    provenance = {
        "git_commit": _git_commit(),
        "base_ckpt": str(args.base_ckpt.resolve()),
        "base_ckpt_sha256": _sha256_file(args.base_ckpt.resolve()),
        "tracker_onnx": str(args.policy_onnx.resolve()),
        "tracker_onnx_sha256": _sha256_file(args.policy_onnx.resolve()),
        "xml_path": str(args.xml_path.resolve()),
        "grounding_xml": str(args.grounding_xml.resolve()),
        "sources": list(args.sources),
        "g1_pth_name": args.g1_pth_name,
        "world_size": world_size,
        "train_clips": len(train_ds),
        "val_clips": val_clips,
    }
    if is_main:
        (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(
            f"[PPO] Ready: world_size={world_size}, train_clips={len(train_ds)}, val_clips={val_clips}, "
            f"prompts/rank={args.prompts_per_rank}, group_size={args.group_size}, "
            f"ddim_steps={args.ddim_steps}, eta={args.ddim_eta}",
            flush=True,
        )

    # 3. Initialize SwanLab on rank 0 (API key from environment only; never logged or committed).
    swan_run = None
    if is_main and args.swanlab:
        import swanlab

        api_key = os.environ.get("SWANLAB_API_KEY")
        if api_key:
            swanlab.login(api_key=api_key, save=False)
        exp_name = args.swanlab_exp_name or f"vith-ppo-{'_'.join(args.sources)}-{time.strftime('%m%d-%H%M')}"
        swan_run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=exp_name,
            logdir=str(output_dir / "swanlog"),
            config={
                **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "policy_version": POLICY_VERSION,
                "reward_version": reward_weights.version,
                "world_size": world_size,
                "train_clips": len(train_ds),
                "val_clips": val_clips,
            },
        )
        print(f"[SwanLab] Initialized project={args.swanlab_project} exp={exp_name}", flush=True)

    # 4. Launch persistent cross-env SONIC/MuJoCo tracker worker pool.
    tracker_pool = PersistentTrackerPool(
        tracker_python=args.tracker_python,
        policy_onnx=args.policy_onnx,
        xml_path=args.xml_path,
        grounding_xml=args.grounding_xml,
        num_workers=args.tracker_workers,
    )
    if is_main:
        print(
            f"[TrackerPool] Started {args.tracker_workers} persistent workers/rank "
            f"(protocol v{tracker_pool.protocol_version})",
            flush=True,
        )

    metrics_jsonl = (output_dir / "metrics.jsonl").open("a", encoding="utf-8") if is_main else None
    try:
        # Baseline held-out evaluation at start_step (rank 0) so SwanLab has base reference point!
        if is_main and (args.eval_batches > 0 or args.val_hard_index is not None):
            val_max_b = 0 if args.val_hard_index is not None else args.eval_batches
            val_metrics = run_validation(
                pipeline=pipeline,
                denoiser=denoiser_raw,
                base_denoiser=base_denoiser,
                policy=policy,
                val_loader=val_loader,
                tracker_pool=tracker_pool,
                reward_weights=reward_weights,
                device=device,
                max_batches=val_max_b,
                seed=args.seed + 10000,
                eval_ddim_steps=args.eval_ddim_steps,
                eval_ddim_eta=args.eval_ddim_eta,
                val_bucket_a_mask=val_bucket_a_mask,
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
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch_idx)
                loader_iter = iter(train_loader)
                batch = next(loader_iter)

            batch_gpu = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            batch_gpu["mask"] = {"f_imgseq": batch_gpu["f_imgseq_mask"]}

            # --- A. GPU Stochastic DDIM Rollout ---
            t0 = time.perf_counter()
            f_cond, length, all_valid = policy.build_condition(pipeline, batch_gpu)
            rollout_gen = torch.Generator(device=device).manual_seed(args.seed + step * 997 + rank * 31)
            rollout = policy.rollout_group(
                denoiser=denoiser_rollout,
                base_denoiser=base_denoiser_rollout,
                f_cond=f_cond,
                length=length,
                group_size=args.group_size,
                all_valid_length=all_valid,
                generator=rollout_gen,
            )
            f_cond_rep = f_cond.repeat_interleave(args.group_size, dim=0)
            length_rep = length.repeat_interleave(args.group_size, dim=0)
            cam_angvel_rep = batch_gpu["cam_angvel"].repeat_interleave(args.group_size, dim=0)
            root_pos_0_rep = batch_gpu["root_pos_0_yup"].repeat_interleave(args.group_size, dim=0)
            decoded = decode_x0_to_g1_zup(
                pipeline=pipeline,
                x_0=rollout.x_0,
                cam_angvel=cam_angvel_rep,
                length=length_rep,
                root_pos_0_yup=root_pos_0_rep,
            )
            torch.cuda.synchronize()
            rollout_s = time.perf_counter() - t0

            # --- B. Replay Self-Check (Mandatory once before first update!) ---
            if not replay_verified:
                with torch.no_grad():
                    check_logp, check_kl = policy.replay_step(
                        denoiser=denoiser_raw,
                        rollout=rollout,
                        stoch_idx=0,
                        f_cond_rep=f_cond_rep,
                        length_rep=length_rep,
                        all_valid_length=all_valid,
                    )
                    max_logp_err = float((check_logp - rollout.old_log_probs[0]).abs().max().item())
                    ratio_dev = float((torch.exp(check_logp - rollout.old_log_probs[0]) - 1.0).abs().max().item())
                if is_main:
                    print(
                        f"[PPO REPLAY CHECK] max|new_logp - old_logp|={max_logp_err:.2e}, "
                        f"max|ratio - 1|={ratio_dev:.2e}, init_kl={float(check_kl.mean().item()):.2e}",
                        flush=True,
                    )
                if max_logp_err > 5e-3 or ratio_dev > 5e-3:
                    raise RuntimeError(
                        f"Old-policy likelihood replay failed: logp_err={max_logp_err}, ratio_dev={ratio_dev}"
                    )
                replay_verified = True

            # --- C. Parallel Frozen SONIC/MuJoCo Tracker Feedback + Reward ---
            t1 = time.perf_counter()
            jp_np = decoded["joint_pos"].cpu().numpy()
            rp_np = decoded["root_pos_w"].cpu().numpy()
            rq_np = decoded["root_quat_w"].cpu().numpy()
            bp_np = decoded["body_pos_w"].cpu().numpy()
            lens_np = length_rep.cpu().numpy()
            fps_rep = np.repeat(batch["fps"].numpy(), args.group_size, axis=0)
            trk_res = tracker_pool.evaluate_batch(jp_np, rp_np, rq_np, bp_np, lens_np, fps=fps_rep)
            gt_jp_rep = np.repeat(batch["gt_joint_pos"].numpy(), args.group_size, axis=0)
            gt_bp_rep = np.repeat(batch["gt_body_pos_z"].numpy(), args.group_size, axis=0)
            scored = compute_sample_rewards(
                tracker_results=trk_res,
                pred_joint_pos=jp_np,
                pred_body_pos_z=bp_np,
                gt_joint_pos=gt_jp_rep,
                gt_body_pos_z=gt_bp_rep,
                lengths=lens_np,
                weights=reward_weights,
                fps=fps_rep,
            )
            tracker_s = time.perf_counter() - t1

            rewards_t = torch.from_numpy(scored["reward"]).to(device=device, dtype=torch.float32)
            advantages = compute_group_advantages(rewards_t, group_size=args.group_size)

            # Optional linear warmup + cosine LR decay.
            if args.min_lr > 0.0 or args.warmup_steps > 0:
                if args.warmup_steps > 0 and step <= args.warmup_steps:
                    cur_lr = args.lr * (step / max(args.warmup_steps, 1))
                else:
                    prog = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
                    prog = min(max(prog, 0.0), 1.0)
                    cur_lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * prog))
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr
            else:
                cur_lr = args.lr

            # --- D. Differentiable PPO Clipped Surrogate + Analytic KL Updates ---
            t2 = time.perf_counter()
            denoiser_raw.eval()  # Keep dropout deterministic so rollout & replay share exact density!
            num_stoch = len(rollout.t_indices)
            steps_per_epoch = min(args.update_steps_per_epoch, num_stoch)
            ppo_losses, kl_vals, ratios_mean, clip_fracs, grad_norms = [], [], [], [], []

            for _ in range(args.ppo_epochs):
                perm = torch.randperm(num_stoch).tolist()[:steps_per_epoch]
                optimizer.zero_grad(set_to_none=True)
                for idx_pos, stoch_idx in enumerate(perm):
                    # Under DDP, synchronize gradients only on the final accumulated diffusion step.
                    sync_ctx = (
                        denoiser_train.no_sync()
                        if (world_size > 1 and idx_pos < len(perm) - 1)
                        else torch.enable_grad()
                    )
                    with sync_ctx:
                        new_logp, kl = policy.replay_step(
                            denoiser=denoiser_train,
                            rollout=rollout,
                            stoch_idx=stoch_idx,
                            f_cond_rep=f_cond_rep,
                            length_rep=length_rep,
                            all_valid_length=all_valid,
                        )
                        old_logp = rollout.old_log_probs[stoch_idx]
                        log_ratio = (new_logp - old_logp).clamp(-10.0, 10.0)
                        ratio = torch.exp(log_ratio)
                        surr1 = ratio * advantages
                        surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * advantages
                        pg_loss = -torch.minimum(surr1, surr2).mean()
                        kl_mean = kl.mean()
                        loss = (pg_loss + args.kl_coef * kl_mean) / len(perm)
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Non-finite PPO loss at step {step}: {loss.item()}")
                        loss.backward()

                        ppo_losses.append(float(pg_loss.item()))
                        kl_vals.append(float(kl_mean.item()))
                        ratios_mean.append(float(ratio.mean().item()))
                        clip_fracs.append(float(((ratio - 1.0).abs() > args.clip_eps).float().mean().item()))

                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip).item())
                if not math.isfinite(grad_norm):
                    raise RuntimeError(f"Non-finite gradient norm at step {step}: {grad_norm}")
                optimizer.step()
                grad_norms.append(grad_norm)

            torch.cuda.synchronize()
            update_s = time.perf_counter() - t2
            total_s = time.perf_counter() - t_step_start

            # --- E. Aggregate Metrics Across Ranks & Log to SwanLab ---
            valid_mask = scored["is_valid"]
            def _valid_avg(arr: np.ndarray) -> float:
                return float(np.nanmean(arr[valid_mask])) if valid_mask.any() else 0.0

            step_metrics = {
                "train/lr": float(cur_lr),
                "train/reward_mean": _all_reduce_mean(float(scored["reward"].mean()), device, world_size),
                "train/reward_std": _all_reduce_mean(float(scored["reward"].std()), device, world_size),
                "train/success_rate": _all_reduce_mean(float(scored["is_success"].mean()), device, world_size),
                "train/survival_ratio": _all_reduce_mean(float(scored["survival_ratio"].mean()), device, world_size),
                "train/exec_track": _all_reduce_mean(float(scored["exec_track"].mean()), device, world_size),
                "train/exec_root_pos_err_m": _all_reduce_mean(_valid_avg(scored["exec_root_pos_err_m"]), device, world_size),
                "train/exec_joint_mae_rad": _all_reduce_mean(_valid_avg(scored["exec_joint_mae_rad"]), device, world_size),
                "train/fidelity": _all_reduce_mean(float(scored["fidelity"].mean()), device, world_size),
                "train/vel_fidelity": _all_reduce_mean(float(scored["vel_fidelity"].mean()), device, world_size),
                "train/mean_joint_vel_rad_s": _all_reduce_mean(_valid_avg(scored["mean_joint_vel_rad_s"]), device, world_size),
                "train/ref_joint_mae_rad": _all_reduce_mean(_valid_avg(scored["ref_joint_mae_rad"]), device, world_size),
                "train/ref_body_mpjpe_m": _all_reduce_mean(_valid_avg(scored["ref_body_mpjpe_m"]), device, world_size),
                "train/invalid_rate": _all_reduce_mean(1.0 - float(valid_mask.mean()), device, world_size),
                "train/pg_loss": _all_reduce_mean(float(np.mean(ppo_losses)), device, world_size),
                "train/kl_div": _all_reduce_mean(float(np.mean(kl_vals)), device, world_size),
                "train/ratio_mean": _all_reduce_mean(float(np.mean(ratios_mean)), device, world_size),
                "train/clip_frac": _all_reduce_mean(float(np.mean(clip_fracs)), device, world_size),
                "train/grad_norm": _all_reduce_mean(float(np.mean(grad_norms)), device, world_size),
                "perf/rollout_s": rollout_s,
                "perf/tracker_s": tracker_s,
                "perf/update_s": update_s,
                "perf/step_s": total_s,
                "perf/rollouts_per_s": (args.prompts_per_rank * args.group_size * world_size) / max(total_s, 1e-6),
            }

            if is_main:
                print(
                    f"[PPO step={step:04d}/{args.max_steps}] "
                    f"rew={step_metrics['train/reward_mean']:.3f} "
                    f"succ={step_metrics['train/success_rate']:.3f} "
                    f"rerr={step_metrics['train/exec_root_pos_err_m']:.4f} "
                    f"jmae={step_metrics['train/exec_joint_mae_rad']:.4f} "
                    f"fid={step_metrics['train/fidelity']:.3f} "
                    f"vfid={step_metrics['train/vel_fidelity']:.3f} "
                    f"kl={step_metrics['train/kl_div']:.4f} "
                    f"clip={step_metrics['train/clip_frac']:.3f} "
                    f"gnorm={step_metrics['train/grad_norm']:.3f} | "
                    f"roll={rollout_s:.2f}s trk={tracker_s:.2f}s upd={update_s:.2f}s",
                    flush=True,
                )
                if swan_run is not None:
                    swan_run.log(step_metrics, step=step)
                if metrics_jsonl is not None:
                    metrics_jsonl.write(json.dumps({"step": step, **step_metrics}) + "\n")
                    metrics_jsonl.flush()

            # Periodic held-out evaluation
            if args.eval_every > 0 and step % args.eval_every == 0:
                if is_main and (args.eval_batches > 0 or args.val_hard_index is not None):
                    val_max_b = 0 if args.val_hard_index is not None else args.eval_batches
                    val_metrics = run_validation(
                        pipeline=pipeline,
                        denoiser=denoiser_raw,
                        base_denoiser=base_denoiser,
                        policy=policy,
                        val_loader=val_loader,
                        tracker_pool=tracker_pool,
                        reward_weights=reward_weights,
                        device=device,
                        max_batches=val_max_b,
                        seed=args.seed + 10000,
                        eval_ddim_steps=args.eval_ddim_steps,
                        eval_ddim_eta=args.eval_ddim_eta,
                        val_bucket_a_mask=val_bucket_a_mask,
                    )
                    print(f"[VAL step={step}] {json.dumps(val_metrics)}", flush=True)
                    if swan_run is not None:
                        swan_run.log(val_metrics, step=step)
                    if metrics_jsonl is not None:
                        metrics_jsonl.write(json.dumps({"step": step, **val_metrics}) + "\n")
                        metrics_jsonl.flush()
                if world_size > 1:
                    dist.barrier()

            # Periodic checkpointing
            if is_main and (step % args.save_every == 0 or step == args.max_steps):
                ckpt_path = output_dir / "checkpoints" / f"ppo_step_{step:05d}.pt"
                latest_path = output_dir / "checkpoints" / "latest.pt"
                _save_checkpoint(ckpt_path, step, pipeline, denoiser_raw, optimizer, args, provenance)
                _save_checkpoint(latest_path, step, pipeline, denoiser_raw, optimizer, args, provenance)
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
