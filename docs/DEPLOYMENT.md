# Deployment

Run commands from the repository root. This repository includes the released
offline base model, the Diffusion-PPO post-training and MultiDiffusion stitching
pipeline (`post_training/`), and the fixed G1 tracker (`g1_tracker/`). See
[VALIDATION.md](VALIDATION.md) for benchmark results and remaining limits.

## Supported base-model platform

The release targets **Linux x86_64, Python 3.10, CUDA 12.1, and PyTorch 2.3.0**.
Its requirements include a Linux-specific PyTorch3D wheel. Full video
preprocessing also selects `cuda:0` explicitly. macOS and CPU-only machines
can run selected utility tests, but are not supported for the complete
video-to-motion command. Passing `--device cpu` does not move preprocessing
to CPU.

Create the base environment:

```bash
conda create -n vith-base python=3.10 -y
conda activate vith-base
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
```

Keep the tracker in a separate environment. The release base requirements
pin `imageio==2.34.1`, while the tracker pins `imageio==2.37.0`; installing
both requirements files into one environment changes the base environment.

## Released checkpoints

The checkpoint helper uses the repository's pinned Hugging Face revision
and file checksums. Weights are downloaded into ignored local paths, not
committed to Git.

```bash
python scripts/checkpoints.py list --profile base
python scripts/checkpoints.py download --profile base
python scripts/checkpoints.py verify --profile base
python scripts/checkpoints.py download --profile tracker
python scripts/checkpoints.py verify --profile tracker
```

If direct access to `https://huggingface.co` is restricted, pass `--hf-endpoint`
or set `HF_ENDPOINT` (for example `HF_ENDPOINT=https://hf-mirror.com`). Every
downloaded file is still verified against the manifest's pinned commit, byte
size, and SHA-256 digest before publication.

The `optional` profile is for optional assets such as DPVO and HMR2; `all`
includes every manifest asset. Downloading a profile does not install the
corresponding software or the external assets below. The released base uses
SAM3D image features; HMR2 is not a substitute for a missing SAM3D setup.

## PPO post-trained checkpoints

Post-trained Diffusion-PPO inference checkpoints (`float32`, weights-only,
`imgfeat_encoder="sam3d"`) are hosted on ModelScope at
[Jasonhsc/vit-h-ppo](https://www.modelscope.cn/models/Jasonhsc/vit-h-ppo)
under `checkpoints/` with full SHA-256 digests in `manifest.json`:

```bash
pip install modelscope
modelscope download --model Jasonhsc/vit-h-ppo \
  checkpoints/hard_ppo_model_soup_50_50_step1200.pt \
  --local_dir ./inputs/checkpoints/vith_ppo
```

See the root [README.md](../README.md) for the full checkpoint table and
recommended inference flags.

## External assets and the SAM3D integration prerequisite

These files are **not** supplied by `eerrr1/ViT-H-checkpoints`:

| Required resource | Expected location |
| --- | --- |
| Fast-SAM-3D-Body source and dependencies | `third-party/Fast-SAM-3D-Body/` |
| SAM3D network weights | `third-party/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/model.ckpt` |
| Matching SAM3D configuration | `third-party/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/model_config.yaml` |
| MHR body model | `third-party/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt` |
| Neutral SMPL-X model used when constructing the base decoder | `inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz` |

Obtain SAM3D resources through
[Fast-SAM-3D-Body](https://github.com/yangtiming/Fast-SAM-3D-Body) and the
body models through their official providers:
[SMPL-X](https://smpl-x.is.tue.mpg.de/) and, for paths that require it,
[SMPL](https://smpl.is.tue.mpg.de/). Preserve their access and usage terms;
do not add these assets to Git.

**Combined Fast-SAM/base Python environment:**
The current extractor imports `sam_3d_body` in the same Python process as
ViT-H. Upstream `facebookresearch/dinov3` (loaded by `Fast-SAM-3D-Body` via
`torch.hub`) imports `from torch.amp import custom_fwd, custom_bwd`, which was
added in PyTorch 2.4 and fails under the release `torch==2.3.0+cu121` pin.
Apply the validated [requirements-sam3d.txt](../requirements-sam3d.txt)
overlay (`torch==2.4.1+cu121`, `torchvision==0.19.1+cu121`, `triton==3.0.0`,
`nvidia-cudnn-cu12==9.1.0.70`, prebuilt `pytorch3d==0.7.8`, and the
`sam_3d_body` runtime packages) on top of `requirements.txt` (using
`--no-build-isolation` for `chumpy` and `cython_bbox`). Keep `vith-tracker`
separate. This repository does not provide an unverified cross-environment
launcher or silently switch encoders.

DPVO is optional and additionally needs its external source/build. Follow
[INFERENCE.md](INFERENCE.md) if explicitly using `--use-dpvo`. Otherwise
the existing default uses SimpleVO. Use `--static-cam` only for a camera
known to be static.

## Video to G1 reference motion

After validating the combined runtime and supplying the external assets:

```bash
conda activate vith-base
python tools/infer_video.py \
  --video /path/to/input.mp4 \
  --ckpt vith_offline_g1.ckpt \
  --sam3d-root third-party/Fast-SAM-3D-Body \
  --sam3d-ckpt-path third-party/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/model.ckpt \
  --sam3d-mhr-path third-party/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --output-root outputs/preprocess/run_001 \
  --output outputs/run_001/reference.npz
```

The command resamples input video timestamps to 30 FPS. Keep distinct
output/cache directories for concurrent jobs. Cache identity, camera
conventions, and output fields are documented in
[INFERENCE.md](INFERENCE.md). A generated NPZ is a reference motion, not
proof that the robot completed it.

## Fixed G1 tracker

The tracker consumes the reference NPZ through a file interface and can
run independently of the base environment:

```bash
conda create -n vith-tracker python=3.10 -y
conda activate vith-tracker
python -m pip install -r g1_tracker/requirements-tracker.txt
python -m pip check

python g1_tracker/tools/track_g1.py \
  --input outputs/run_001/reference.npz \
  --output outputs/run_001/tracked.npz \
  --input-fps 30 \
  --policy-onnx inputs/checkpoints/g1_tracker/model_step_020000_g1.onnx \
  --xml-path g1_tracker/assets/g1/rollout/scene_29dof_mujoco.xml \
  --grounding-xml g1_tracker/assets/g1/grounding/mjcf/g1_29dof_rev_1_0.xml
```

For batched reference NPZ files (such as PPO rollout batches or dataset
evaluation), pass `--num-workers N` (default `1`) to evaluate independent
samples across parallel CPU worker processes while keeping `--ort-threads 1`
per worker and preserving identical Protocol v4 outputs.

The two XML paths have different roles and are not interchangeable.
Keep the policy, XML assets, and protocol fixed when comparing the base
with future PPO checkpoints. See [the tracker README](../g1_tracker/README.md)
for NPZ conventions, failure criteria, and optional rendering. Inspect
`is_success` and `first_failure_frame_50hz`; a completed process alone
does not mean successful tracking.

## Verification commands

Before launching inference or training on a shared GPU node, run the preflight
check to verify CUDA/GPU visibility, filesystem free space (including `TMPDIR`
and cache directories), user-site isolation (`PYTHONNOUSERSITE=1`), release
checkpoints, external SAM3D/MHR/SMPL-X assets, and `sam_3d_body` importability:

```bash
PYTHONNOUSERSITE=1 python scripts/preflight.py --mode base --require-gpu --verify-hash
PYTHONNOUSERSITE=1 python scripts/preflight.py --mode tracker --verify-hash
```

For training runs via `scripts/train.sh`, throughput and scratch settings can
be configured via environment variables without modifying defaults:
`TRAIN_PRECISION` (default `32`, e.g. `bf16-mixed` on Hopper/Blackwell GPUs),
`VITH_MATMUL_PRECISION` (`highest`, `high`, or `medium`), `BATCH_SIZE`,
`NUM_WORKERS`, `VITH_TMPDIR`, and `VITH_G1_PTH_NAME` (default `g1.pth`, e.g.
`g1_new.pth` when evaluating alternative retargeted G1 trajectories without
overwriting the original `g1.pth` files).

For CPU regressions, use an isolated Python 3.10 environment rather than
installing the CUDA requirements on a laptop:

```bash
python3.10 -m venv .venv-cpu
.venv-cpu/bin/python -m pip install -r requirements-cpu-tests.txt
.venv-cpu/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Alternatively, in a base environment with the required test dependencies:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

In the tracker environment:

```bash
PYTHONPATH=g1_tracker python -m unittest discover -s g1_tracker/tests -p 'test_*.py' -v
```

After downloading the tracker profile, exercise the real ONNX/MuJoCo runtime
on a synthetic standing reference in the tracker environment:

```bash
python scripts/smoke_tracker.py --output-dir outputs/tracker_smoke_001
```

The output directory must be new. Inspect its `summary.json`, including
`is_success` and the first failure frame. The script checks saved array
integrity and reports tracking success separately.

The unit tests do not require model checkpoints; the synthetic tracker
smoke requires the tracker checkpoint. CLI `--help`, utility tests,
and even a tracker rollout on a synthetic reference do not validate
video preprocessing, base checkpoint inference, or PPO training. Record
those milestones separately in [VALIDATION.md](VALIDATION.md).

The inherited `scripts/check_release.sh` is an upstream public-release
cleanup/check script. It uses GNU `grep -P` and deletes Python caches;
do not treat its result on macOS as a complete deployment check.
