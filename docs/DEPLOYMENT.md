# Deployment

Run commands from the repository root. This checkout contains the released
offline base model and a fixed G1 tracker. It does not yet contain a PPO
post-training implementation. See [VALIDATION.md](VALIDATION.md) for checks
actually performed; installation instructions are not evidence of an
end-to-end video run.

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

The `optional` profile is for optional assets such as DPVO and HMR2; `all`
includes every manifest asset. Downloading a profile does not install the
corresponding software or the external assets below. The released base uses
SAM3D image features; HMR2 is not a substitute for a missing SAM3D setup.

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

**The combined Fast-SAM/base Python environment has not been validated.**
The current extractor imports `sam_3d_body` in the same Python process as
ViT-H. Creating a separate Fast-SAM conda environment and activating it
does not establish a bridge to `vith-base`. Before running full inference,
validate a compatible shared runtime with the required SAM3D imports,
weights, and base dependencies. Record the installed versions and any
required integration changes. This repository does not provide an
unverified cross-environment launcher or silently switch encoders.

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

The two XML paths have different roles and are not interchangeable.
Keep the policy, XML assets, and protocol fixed when comparing the base
with future PPO checkpoints. See [the tracker README](../g1_tracker/README.md)
for NPZ conventions, failure criteria, and optional rendering. Inspect
`is_success` and `first_failure_frame_50hz`; a completed process alone
does not mean successful tracking.

## Verification commands

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
