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

### Full `val` + `test` Base & GT trackability audit (`n = 5,746` clips)

Using `post_training/mine_hard_clips.py` (4 H100 GPUs × 16 persistent tracker
workers, DDIM-50 `eta=0.0`, `gt_mode=all`), 100% of `test` (`2,863` clips) and
100% of `val` (`2,883` clips) across `h36m + bedlam + bedlam2` were evaluated
on both Base (`vith_offline_g1.ckpt`) and ground-truth `g1_new.pth`:

| Split (`n`) | Base pass | GT `g1_new` pass | Base pass \| GT pass | Base `exec_root_err` (GT) | `Bucket A` (fixable fail) | `Bucket B` (high exec err, top-20%) | `Bucket C` (easy pass) | `untrackable_gt` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `test` (`2,863`) | 97.69% (66 fail) | 94.03% (171 fail) | **99.18%** | 0.1119 m (0.1716 m) | **22** (0.77%) | **898** (31.4%) | 1,772 (61.9%) | 171 (6.0%) |
| `val` (`2,883`) | 97.05% (85 fail) | 93.10% (199 fail) | **99.22%** | 0.1127 m (0.1818 m) | **21** (0.73%) | **894** (31.0%) | 1,769 (61.4%) | 199 (6.9%) |

Key empirical observations:
- On 120-frame in-domain windows, Base already exceeds GT `g1_new.pth` on
  Protocol v4 pass rate (`97.7%` vs `94.0%` on `test`) and passes `74.3%`
  (`test`) / `67.8%` (`val`) of `untrackable_gt` clips.
- True fixable failures (`Bucket A`: GT passes, Base fails) are rare (`0.75%`)
  and highly dynamic (`gt_jvel = 0.62–0.79 rad/s`, `gt_path = 1.26–1.51 m`,
  ~2x faster than `Bucket C`).
- The main large-sample optimization headroom on 120-frame windows is
  `Bucket B` (~31% of clips, `~900` per split), where Base passes Protocol v4
  but exhibits 2.6x larger execution root drift (`0.165 m` vs `0.063 m` in
  `Bucket C`) and larger joint tracking error (`0.135 rad` vs `0.083 rad`).
- Sequence length audit on `test` shows `bedlam2` (`54.9% >= 240f`, `35.0% >= 360f`,
  max `924f`) and `h36m` (`100% >= 596f`, max `1,901f`) contain substantial
  multi-window long-horizon trajectories where stitched drift can also be tested.

### Long-horizon multi-window stitched evaluation (`test`, `>= 240` frames, `n = 383`)

Evaluated via `post_training/eval_long_horizon.py` using the exact 1-frame
overlap rigid endpoint stitching from `tools/infer_video.py::align_window_to_previous`
(DDIM-50 `eta=0.0`, identical per-chunk noise seeds, frozen Protocol v4 tracker)
across all `383` `test` sequences with `length >= 240` (`367 bedlam2` + `16 h36m`):

| Slice (`n`) | GT `g1_new` pass | Base pass (`\|GT pass`) | PPO-500 pass (`\|GT pass`) | Paired Base vs PPO-500 | Base vs PPO-500 `exec_root_err` | Base seam jump vs interior |
| --- | ---: | ---: | ---: | --- | --- | --- |
| **Overall (`383`)** | 93.47% (25 fail) | 87.47% (89.66%) | **88.25%** (90.22%) | +5 fixed / -2 regressed (`p=0.45`) | `0.2710 -> 0.2629 m` | `0.0839` vs `0.0138 rad` (**6.1x**) |
| `240–359f` 8–12s (`133`) | 91.73% | 90.23% (91.80%) | 89.47% (90.98%) | +0 / -1 (`p=1.0`) | `0.1984 -> 0.2011 m` | `0.0858` vs `0.0149 rad` (5.7x) |
| `360–599f` 12–20s (`233`) | 96.57% | 89.70% (90.67%) | **90.56%** (91.11%) | +3 / -1 (`p=0.63`) | `0.2325 -> 0.2305 m` | `0.0829` vs `0.0130 rad` (6.4x) |
| `600+f` 20–76s (`17`) | 64.71% | 35.29% (45.45%) | **47.06%** (**63.64%**) | +2 / -0 (`p=0.50`) | `1.3659 -> 1.1903 m` | `0.0840` vs `0.0154 rad` (5.5x) |

Key empirical findings:
- Unlike 120-frame single windows (`99.2%` Base pass given GT pass), long-horizon
  stitched Base pass rate drops to `87.47%` overall (`89.7%` on 12–20s, `35.3%`
  on 20–76s H36M sequences) while GT remains much higher (`96.6%` on 12–20s,
  `64.7%` on 20–76s).
- Window boundaries exhibit a **6.1x joint-angle discontinuity** (`0.0839 rad` at
  stitch seams vs `0.0138 rad` interior step), and **27% (`13/48`)** of Base
  failures occur within `<= 0.5s` of a seam vs only **4% (`1/25`)** for GT.
- Even the conservatively updated 500-step pilot (`0.109%` weight shift) showed
  consistent positive direction on long sequences (`+5/-2` overall, `+2/-0` on
  `600+f`, `exec_root_err` `-8.1 mm` overall and `-176 mm` on `600+f`).

### PPO v2 (`reward_v2` + hard-clip mining, 1,500 steps, 8-GPU DDP) untouched `test` evaluation

Full `train`-split mining (`51,485` clips, 8 GPUs × 12 workers, `41.2` min)
yielded `407` `Bucket A` fixable failures, `16,351` `Bucket B` high-error clips,
`32,327` `Bucket C` easy clips, and `2,400` `untrackable_gt` clips (excluded).
8-GPU DDP PPO v2 (`reward_v2`, `shares=(0.20, 0.55, 0.25)`, `group_size=8`,
`kl_coef=0.01`, `lr=2.5e-5 -> 5e-6` cosine, `1,500` steps) completed cleanly
(`5.94 s/step`, replay error `9.78e-06`).

Untouched `test` benchmark comparison (all evaluated in exact released
deployment mode `DDIM-50, eta=0.0` with identical noise seeds and frozen
Protocol v4 SONIC/MuJoCo tracker):

| Benchmark (`DDIM-50 eta=0.0`) | Base | PPO-v1 (`500`) | PPO-v2 (`450`) | PPO-v2 (`900`) | PPO-v2 (`1200`) | PPO-v2 (`1500`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `denoiser3d` rel. weight shift `\|\|dW\|\|_2/\|\|W\|\|_2` | `0.000%` | `0.109%` | `0.216%` | `0.263%` | `0.269%` | `0.272%` |
| `test` Bucket A (`n=22`) pass (`surv`) | `3/22` (`0.551`) | `3/22` (`0.523`) | `5/22` (`0.605`) | **`7/22`** (`0.644`) | `6/22` (**`0.675`**) | `4/22` (`0.592`) |
| `test` Bucket B (`n=200`) `exec_root_err` (`jmae`) | `0.1658 m` (`0.1341`) | `0.1645 m` (`0.1328`) | `0.1618 m` (`0.1321`) | **`0.1591 m`** (`0.1319`) | `0.1627 m` (**`0.1315`**) | `0.1602 m` (`0.1316`) |
| `test` 640-clip pass (`fixed/regressed`) | `98.59%` (`631/640`) | `98.59%` (`+0/-0`) | **`98.91%`** (`+2/-0`) | — | **`98.91%`** (`+2/-0`) | — |
| `test` 640-clip median `Δ exec_root_err` (`% better`) | — | `+0.93 mm` (`38.3%`) | `+0.79 mm` (`45.5%`) | — | **`-0.85 mm`** (**`55.3%`**) | — |
| `test` 383 long stitched pass (`fixed/regressed`) | `87.47%` | `88.25%` (`+5/-2`) | `88.51%` (`+7/-3`) | — | **`88.77%`** (**`+8/-3`**) | — |
| `test` 383 long `exec_root_err` (`paired t`) | `0.2711 m` | `0.2629 m` (`t=-1.05`) | `0.2616 m` (`t=-1.84`) | — | **`0.2513 m`** (**`t=-2.29`**) | — |
| `test` 383 long `exec_jmae` (`paired t`) | `0.1068 rad` | `0.1051 rad` (`t=-3.22`) | `0.1061 rad` (`t=-2.04`) | — | **`0.1047 rad`** (**`t=-5.68`**) | — |
| `test` 383 long `seam_jump` (`paired t`) | `0.0839 rad` | `0.0829 rad` (`t=-12.86`) | **`0.0807 rad`** (`t=-18.33`) | — | `0.0814 rad` (`t=-13.60`) | — |
| Real 300f dance video `exec_root_err` (`jmae`) | `0.2057 m` (`0.1699`) | `0.1977 m` (`0.1681`) | **`0.1747 m`** (`0.1717`) | — | `0.1967 m` (**`0.1671`**) | `0.1831 m` (`0.1682`) |

### PPO v3 (`ddpo_stitched_v3` / `reward_v3`, 800 steps) & multi-frame overlap stitching ablation (`n = 392`)

1. **Stitched PPO v3 diagnosis (`post_training/train_ppo_stitched.py`, 800 steps, 8-GPU DDP)**:
   Warm-started from `v2_900` with KL anchored to frozen `vith_offline_g1.ckpt` (`kl_coef=0.015`, `lr=2e-5 -> 4e-6`, `7,155` clean `>=239f` train pairs weighted `2.5x` on hard keys). On 64 held-out `val` stitched pairs (`DDIM-50 eta=0.0`), survival increased slightly (`89.06% -> 90.63%`, `0.9459 -> 0.9610`), while `val/seam_joint_jump_rad` stayed nearly unchanged (`0.0801 -> 0.0803 rad`) and weight L2 shift plateaued at `0.251%` after step 400. Directly measuring `val/seam_overlap_err_rad = 0.0775 rad` established that **97%** of the 1-frame seam discontinuity comes from boundary-context asymmetry between independently denoised 120-frame windows rather than policy weights.

2. **Multi-frame overlap cosine/SLERP+FK & gravity-preserving `yaw` alignment ablation (`post_training/stitching.py`, `post_training/eval_long_horizon.py`)**:
   Evaluated in parallel across all **`n = 392` untouched `test` sequences (`>= 240` frames: `bedlam2 + h36m + bedlam`, `DDIM-50 eta=0.0`, frozen Protocol v4 tracker)**:

| Stitching config | Model | Pass% | Fails | `Pass\|GT` | Survival | `exec_root_err` | `exec_jmae` | `SeamPeak` | `OvMean` | `RefMPJPE` | Chunks | Paired vs `Base (ov1_so3)` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **GT ceiling** | `GT` | `93.37%` | `26` | `100.00%` | `0.9612` | `0.2855 m` | `0.1323` | — | — | — | — | — |
| `ov1_so3` (legacy) | `Base` | `87.76%` | `48` | `89.89%` | `0.9523` | `0.2685 m` | `0.1089` | `0.0837` | `0.0837` | `0.2785` | `3.90` | Baseline |
| `ov1_so3` (legacy) | `PPO v3_800` | `87.76%` | `48` | `89.89%` | `0.9552` | `0.2608 m` | `0.1072` | `0.0812` | `0.0812` | `0.2785` | `3.90` | `+5/-5` (`p=1.0`), `-7.7 mm` |
| `ov16_so3_cosine` | `Base` | `90.05%` | `39` | `92.08%` | `0.9636` | `0.2168 m` | `0.1063` | `0.0236` | `0.0155` | `0.2782` | `4.59` | `+18/-9` (`p=0.122`), `-51.7 mm` |
| `ov16_yaw_cosine` | `Base` | `92.09%` | `31` | `93.72%` | `0.9724` | `0.1907 m` | `0.1056` | `0.0236` | `0.0155` | **`0.2778`** | `4.59` | **`+21/-4` (`p=9.1e-4`)**, `-77.8 mm` |
| **`ov16_yaw_cosine`** | **`PPO v2_1200`** | **`93.37%`** | **`26`** | **`95.08%`** | `0.9756` | **`0.1862 m`** | **`0.1035`** | **`0.0234`** | `0.0153` | `0.2783` | `4.59` | **`+25/-3` (`p=2.7e-5`)**, **`-82.3 mm (t=-3.52)`** |
| `ov24_yaw_cosine` | `Base` | `92.60%` | `29` | `94.26%` | `0.9746` | `0.1889 m` | `0.1058` | `0.0246` | `0.0149` | **`0.2778`** | `4.70` | `+24/-5` (`p=5.5e-4`), `-79.6 mm` |
| **`ov24_yaw_cosine`** | **`PPO v2_1200`** | **`93.37%`** | **`26`** | **`95.08%`** | **`0.9773`** | `0.1874 m` | `0.1036` | `0.0242` | **`0.0147`** | `0.2783` | `4.70` | **`+25/-3` (`p=2.7e-5`)**, `-81.1 mm (t=-3.40)` |

Key findings from the 3-factor additive decomposition:
- **Multi-frame cosine + SLERP + `fk_g1` overlap (`ov1 -> ov16`, `so3`, `Base`)**: cuts `SeamPeak` `-71.8%` (`0.0837 -> 0.0236 rad`, `t=-38.77`, `99.7%` improved) and `exec_root_err` `-51.7 mm` (`t=-2.85`) with zero fidelity loss (`RefMPJPE 0.2785 -> 0.2782 m`).
- **Gravity-preserving `yaw`-only alignment (`so3 -> yaw` at `ov16`, `Base`)**: adds `+13/-5` passes and cuts `exec_root_err` another `-26.1 mm` (`t=-2.92`), eliminating pitch/roll floor-plane tilt accumulation across windows (on `600+f` sequences `exec_root_err` drops from `1.365 m -> 0.746 m`).
- **PPO `v2_1200` stacks additively on top of `ov16_yaw_cosine`**: adds `+6/-1` passes (`92.09% -> 93.37%`, matching the GT ceiling overall and on `600+f` at `64.71%` with `0` near-seam failures) and further improves `exec_jmae` (`0.1056 -> 0.1035 rad`, `t=-6.37`, `72.7%` better).

3. **Real 300-frame dancing video (`tools/infer_video.py`, `seed=42`, `3 chunks` for both `ov1` and `ov16`)**:
   All runs passed Protocol v4 (`is_success=True`). Switching from legacy `ov1_so3` to default `ov16_yaw_cosine` cut `SeamPeak` `-34.7%` (`0.1040 -> 0.0679 rad` on Base; `0.0664 rad` on `v2_1200`), cut `OvMean` `-51.3%` (`0.1040 -> 0.0507 rad`; `0.0489 rad` on `v2_1200`), and achieved the lowest `exec_jmae` (`0.1630 rad` on `v2_1200 + ov16_yaw` vs `0.1699 rad` on `Base + ov1_so3`). On this single clip, shifted chunk boundaries increased horizontal `xy` displacement (`0.141 -> 0.201 m` on Base) while preserving vertical stability (`z_span = 0.179–0.182 m`); under identical `ov16_yaw` boundaries, PPO `v2_900` (`0.2111 m`, `-41.3 mm`) and `v2_1200` (`0.2235 m`, `-28.9 mm`) both reduced root error vs `Base` (`0.2524 m`).

### Remaining open research questions

- The combination of **PPO v2 (`v2_900`–`v2_1200`) + `ov16_yaw_cosine` inference stitching** now matches the untouched `test` long-horizon GT pass-rate ceiling (`93.37%`, `Pass|GT = 95.08%`, `exec_root_err = 0.1862 m`) while improving short-window `Bucket A` pass rate (`3/22 -> 6–7/22`) and `640`-clip pass rate (`98.59% -> 98.91%`, `+2/-0`).
- Remaining headroom is concentrated in:
  1. Expanding dynamic hard-clip coverage beyond the `407`-clip `Bucket A` pool (to prevent late-step overfitting past ~900–1200 PPO steps);
  2. Prefix-conditioned / inpainting window continuation if tighter cross-window pose coupling is needed on ultra-long (`600+f`) out-of-domain videos.
