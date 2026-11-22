# Validation record

Recorded on 2026-11-16, on macOS arm64 with isolated Python 3.10.20
environments. This record describes repository setup checks, not a full
reproduction of ViT-H or a PPO experiment.

## Verified

| Check | Observed result |
| --- | --- |
| Source import | All 322 regular archive files preserved at tag `base/vith-release-2482c5c`; provenance recorded in `SOURCE_IMPORT.md`. |
| Python and shell syntax | Python source parsing and shell syntax checks passed. |
| Inference utility regressions | All 33 existing tests passed using `requirements-cpu-tests.txt`. |
| Tracker success regressions | All 3 existing tests passed. |
| Dependency consistency | `uv pip check` passed in both isolated CPU-test and tracker environments. |
| Checkpoint manifest | All 6 entries passed manifest validation; immutable Hugging Face revision, byte sizes and SHA-256 identities recorded. |
| Actual checkpoint download | Tracker ONNX downloaded, 58,053,634 bytes, SHA-256 verified against the manifest. |
| Downloader integrity failures | Corrupt existing file rejected without modification; corrupt downloaded fixture rejected without publication and its temporary file removed. |
| Real tracker execution | ONNX Runtime and MuJoCo completed the synthetic standing check described below. |
| Diff hygiene | Whitespace checks passed; environments, downloaded weights and generated outputs excluded from Git. |

The inference regression command was:

```bash
TMPDIR=/private/tmp .venv-cpu/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

On this Mac, the multiprocessing tests required permission to create shared
memory outside the execution sandbox. `TMPDIR=/private/tmp` avoided the
macOS `/var` versus `/private/var` symlink distinction in an existing path
assertion. No production logic or test assertions were changed to bypass
these conditions. The suite also exercises an explicitly requested encoder
override and prints a warning; that is not evidence that HMR2 is a valid
replacement for this checkpoint's SAM3D conditioning.

Tracker success tests used:

```bash
PYTHONPATH=g1_tracker .venv-tracker/bin/python -m unittest discover -s g1_tracker/tests -p 'test_*.py' -v
```

## Synthetic tracker execution

```bash
.venv-tracker/bin/python scripts/smoke_tracker.py \
  --output-dir outputs/setup_tracker_smoke
```

The input was a synthetic standing reference, not a model-generated motion
or a video. It contained 61 frames at 30 FPS (two seconds including both
endpoints). The run used the pinned tracker ONNX and bundled G1 assets.
The saved result was reloaded and checked for expected shapes and finite
values. Its summary was:

```json
{
  "input_kind": "synthetic_standing_not_video",
  "frames": 61,
  "fps": 30,
  "tracker_protocol_version": 4,
  "is_success": [true],
  "first_failure_frame_50hz": [-1],
  "initial_joint_clip_count": [0],
  "output_arrays_finite": true
}
```

This verifies tracker checkpoint loading, simulation, and output handling
for one simple reference. It does not measure motion coverage, robustness,
video conditioning, or improvement from post-training. Generated files
remain local under ignored `outputs/`; reruns need a new output directory.

## Not yet verified or implemented

- The full Linux/CUDA base inference environment has not been installed or
  exercised here. Only the tracker checkpoint was downloaded; the base,
  detector, pose and optional checkpoint files were not downloaded or loaded.
- Fast-SAM/SAM3D integration and its externally supplied weights and body
  assets still require a compatible shared runtime and a real-video test.
- No real video has been processed through base inference and then tracking
  as part of this repository setup. No historical benchmark is reproduced.
- PPO policy density, value function, rollout collection, reward, KL control
  and training have not been implemented. Their required decisions are
  documented in `PPO_INTEGRATION.md`.
- GPU memory use, throughput, multi-GPU training and convergence are untested.

The added GitHub Actions workflow runs CPU regressions and manifest
validation; it does not download checkpoints or perform GPU inference.
Consult the actual Actions run for its status rather than interpreting
the existence of the workflow as a successful remote check.

Setup changes add documentation, pinned checkpoint management, a synthetic
tracker smoke command and CPU CI. The main requirements now explicitly
include the directly imported `easydict` and `yacs` packages. Model math,
checkpoint loading semantics, tracker policy, and evaluation thresholds
remain those of the imported release.

## Linux H100 end-to-end validation (2026-11-22)

Recorded on Linux x86_64 with an NVIDIA H100 80GB HBM3 (`sm_90`) GPU using
isolated Python 3.10.22 environments (`PYTHONNOUSERSITE=1`).

### Combined Fast-SAM-3D-Body + ViT-H environment finding

Upstream `facebookresearch/dinov3` (loaded by `Fast-SAM-3D-Body` via
`torch.hub`) imports `from torch.amp import custom_fwd, custom_bwd`, which
was added in PyTorch 2.4 and fails under the release `torch==2.3.0+cu121` pin.
Applying [requirements-sam3d.txt](../requirements-sam3d.txt)
(`torch==2.4.1+cu121`, `torchvision==0.19.1+cu121`, `triton==3.0.0`,
`nvidia-cudnn-cu12==9.1.0.70`, prebuilt `pytorch3d==0.7.8`, `numpy==1.23.5`,
and the `sam_3d_body` runtime packages) resolved the import cleanly with
`pip check` passing in both `vith-base` and `vith-tracker`.

### Verified on H100

| Check | Observed result |
| --- | --- |
| Preflight (`base` & `tracker`) | `scripts/preflight.py --mode base --require-gpu --verify-hash` and `--mode tracker --verify-hash` both passed with `Isolated user-site: True`. |
| Checkpoints & external assets | `base` (3.564 GB) and `tracker` (0.058 GB) profiles passed SHA-256 verification; `dpvo.pth` verified; external `sam-3d-body-dinov3/model.ckpt` (SHA-256 `b5a2f9d3...`), `mhr_model.pt` (Meta MHR v1.0.1, SHA-256 `352e271a...`), and `SMPLX_NEUTRAL.npz` (SMPL-X v1.1, SHA-256 `37602144...`) verified. |
| Base + infra unit tests | 40/40 passed (`unittest discover -s tests`). |
| Tracker unit tests & parallel equivalence | 5/5 passed including `num_workers=1` vs `num_workers=2` bit-exact equality; 8-sample 61-frame benchmark ran in 2.22s (`num_workers=8`) vs 4.66s (`num_workers=1`, 2.10x speedup, `np.array_equal` True across all fields). |
| ViT-H 50-step DDIM GPU forward | Strict `load_pipeline` loaded in 5.62s; 120-frame 50-step DDIM forward ran in 0.77s with 0.91 GiB peak VRAM and finite outputs. |
| Real-video end-to-end inference + tracking | 300-frame (10.0s @ 30 FPS, 960x540, 3 chunks) real human dancing video completed `tools/infer_video.py` (YOLO + ViTPose + Fast-SAM3D + SimpleVO + ViT-H) in 53s and `g1_tracker/tools/track_g1.py` in 4s. All reference and tracked arrays were finite; tracker Protocol v4 reported `is_success=[true]`, `first_failure_frame_50hz=[-1]`, `grounding_offset_m=[0.00566]`, `initial_joint_clip_count=[0]`. |

### Hopper (`sm_90`) bottleneck profiling & optimization

Profiling `RelativeTransformer` (`L=120, D=1024, 16 layers, 16 heads`) on H100
via `torch.profiler` showed baseline FP32 spent ~78% of CUDA time in SIMT/Ampere
FP32 GEMMs, while naive BF16 autocast regressed (`82.82 ms` vs `67.63 ms` TF32)
due to `@autocast(enabled=False)` FP32 casts inside `rotary_embedding.py` and
manual `einops.einsum` attention in `encoder_rope.py`. Replacing attention with
`F.scaled_dot_product_attention`, pre-caching non-persistent RoPE `cos`/`sin`
tables, adding mask-free `all_valid_length` FlashAttention dispatch, and
enabling Hopper BF16 + `torch.compile` yielded:

| Workload (`B=32, L=120` on H100 80GB) | FP32 baseline | SDPA + CachedRoPE + BF16 | + `torch.compile` | Speedup |
| --- | --- | --- | --- | --- |
| 50-step DDIM rollout | 2193.3 ms | 566.3 ms (3.87x) | **240.6 ms** (`reduce-overhead`) | **9.11x** |
| Training fwd + bwd | 130.28 ms | 50.68 ms (2.57x) | **37.99 ms** (`default`) | **3.43x** |

Numerical parity vs the original manual `einsum` attention was verified at
`max_abs = 2.82e-04`, `max_rel = 2.59e-06` (and `< 1e-5` in FP32 unit tests).

For tracker feedback during PPO, `PersistentTrackerPool` eliminates per-step
subprocess startup and disk I/O while preserving exact Protocol v4 parity
(`off_diff = 7.61e-11` m on the 300-frame dancing reference):

| `PersistentTrackerPool` workers (16 x 120f batch) | Init | Batch latency | Throughput |
| --- | --- | --- | --- |
| 4 workers | 1.61 s | 3.734 s | 4.3 traj/s |
| 8 workers | 1.07 s | 2.267 s | 7.1 traj/s |
| 16 workers | 1.10 s | 1.328 s | 12.0 traj/s |

### 2-GPU Diffusion-PPO end-to-end smoke verification (`h36m`, `g1_new.pth`)

Verified with `torchrun --standalone --nproc_per_node=2 post_training/train_ppo.py`
(`prompts_per_rank=4`, `group_size=4`, `ddim_steps=20`, `ddim_eta=0.7`,
`ppo_epochs=2`, `update_steps_per_epoch=10`, `--compile-rollout --compile-mode reduce-overhead`,
`--tracker-workers 8`, and live SwanLab logging):

| Check | Observed result |
| --- | --- |
| Unit tests (`tests/`) | 46/46 passed (including 6 new PPO policy/reward/SDPA tests in `tests/test_ppo_components.py`). |
| Baseline held-out validation (`step=0`) | `val/success_rate=0.875`, `val/survival_ratio=0.9817`, `val/exec_track=0.4040`, `val/fidelity=0.2625`, `val/invalid_rate=0.0`. |
| Old-policy likelihood replay self-check | `max|new_logp - old_logp| = 5.01e-06`, `max|ratio - 1| = 5.01e-06`, `init_kl = 6.13e-08` (pre-resume) and `6.32e-06` (post-resume). |
| Steady-state 2-GPU PPO step timing | `roll=0.17s` (compiled BF16 CUDA-graph rollout), `trk=2.49s` (persistent tracker pool), `upd=1.40s` (DDP + fused `AdamW`), finite losses/gradients (`ratio≈0.989`, `clip≈0.02`, `kl≈0.034`, `gnorm≈1.46–3.98`). |
| Checkpoint save/resume + strict `load_pipeline` | Resumed from `ppo_step_00003.pt` through step 5; `inspect_checkpoint_imgfeat_encoder` returned `'sam3d'`; `load_pipeline` strict load succeeded; parameter deltas confirmed `denoiser3d=9.374e-05` (>0) while `fixed_pe=0.0`, `aux_head=0.0`, and `cond_embedder=0.0`. |

### 500-step 8-GPU DDP PPO pilot & paired Base-vs-PPO test evaluation (`g1_new.pth`)

An 8-GPU DDP pilot run (`max_steps=500`, `prompts_per_rank=4`, `group_size=4`,
`128` trajectories/step, `ddim_steps=20`, `ddim_eta=0.7`, `lr=1e-5`,
`kl_coef=0.04`, `w=(1.0, 0.5, 0.3, 1.0)`) trained over `h36m + bedlam + bedlam2`
(`51,485` train clips, `2,883` val clips) completed cleanly (all 500 steps,
`denoiser3d` relative L2 weight delta `2.972 / 2733.09 = 0.109%`).

Paired Base (`vith_offline_g1.ckpt`) vs PPO (`ppo_step_00500.pt`) evaluation
via `post_training/eval_compare.py` on **640 untouched `test` clips**
(`140 h36m` + `250 bedlam` + `250 bedlam2`, identical initial noise `x_T`,
released 50-step deterministic DDIM `eta=0.0`, frozen Protocol v4 tracker):

| Metric (`n=640` paired `test` clips, DDIM-50 `eta=0.0`) | Base | PPO-500 | Delta (`PPO - Base`) | Paired significance |
| --- | ---: | ---: | ---: | --- |
| `success_rate` (Protocol v4) | 0.9859 (631/640) | 0.9859 (631/640) | `0.0000` (0 fixed / 0 regressed) | McNemar `p=1.0` |
| `survival_ratio` | 0.99185 | 0.99193 | `+0.00008` | — |
| `exec_joint_mae_rad` (lower better) | 0.10239 | **0.10173** | **`-0.00066`** (`-0.64%`) | `t = -6.11` (61.9% better) |
| `exec_root_pos_err_m` (lower better) | 0.10484 | 0.10588 | `+0.00104` | `t = +1.52` (not sig.) |
| `ref_joint_mae_rad` vs `g1_new.pth` | 0.13977 | **0.13929** | **`-0.00048`** | `t = -5.78` (61.4% better) |
| `ref_body_mpjpe_m` vs `g1_new.pth` | 0.23428 | 0.23422 | `-0.00005` | `t = -1.53` (not sig.) |
| `fidelity` | 0.51002 | **0.51057** | **`+0.00055`** | `t = +5.74` (60.3% better) |
| `reward_v1` | 2.13406 | 2.13439 | `+0.00033` | `t = +1.26` (not sig.) |
| `mean_joint_vel_rad_s` (GT: `0.4302`) | 0.38515 | 0.38274 | `-0.00240` (`-0.62%`) | `t = -14.55` |
| `root_xy_path_m` (GT: `0.9800`) | 0.71253 | 0.71547 | `+0.00294` | `t = +6.73` |

Additional diagnostic findings:
- **Train-mode sampler check (`bedlam`, `n=250`, DDIM-20 `eta=0.7`)**: Base
  scored `0.9640` (`241/250`), PPO-500 scored `0.9560` (`239/250`, `0` fixed /
  `2` regressed, McNemar `p=0.50`), `exec_root_pos_err_m` `0.1169 -> 0.1156 m`,
  confirming the small effect size is not merely a train-vs-deploy sampler gap.
- **GT trackability audit on all 9 Base-failed clips**: 6 of the 9 clips also
  failed Protocol v4 when tracking ground-truth `g1_new.pth` directly; only 3
  of 640 (`0.47%`) were GT-trackable clips where Base failed.
- **Real-video end-to-end check** (300-frame dancing clip, `tools/infer_video.py`
  + `track_g1.py`, seed 42): both passed (`is_success=True`), with PPO-500
  changing joint angles (`max_abs=0.1305 rad`, `mean_abs=0.0115 rad`) and
  reducing `exec_root_err` (`0.2057 -> 0.1977 m`) and `exec_j_mae`
  (`0.1699 -> 0.1681 rad`) while keeping `root_xy_path` identical (`2.491 m`).

### Remaining open research questions

- Uniform in-domain 120-frame sampling saturates near 98.6% Base success and
  gave `fidelity` 3.3x more weight than `exec_track`, yielding only a `0.109%`
  weight shift in 500 steps. Next-round PPO requires hard-clip / OOD /
  longer-horizon mining, higher execution-tracking weight, and larger effective
  update budgets verified against held-out hard benchmarks and real videos.
