# ViT-H PPO Post-Training & Long-Horizon Stitching

This repository builds on the released ViT-H offline video-to-G1 motion generator (`base/vith-release-2482c5c`) with **Diffusion-PPO post-training** (using frozen SONIC ONNX + MuJoCo execution feedback) and **long-horizon synchronized stitching** (16-frame overlap, gravity-preserving yaw cosine alignment, and MultiDiffusion joint denoising). The original upstream implementation and attribution are preserved below.

## Current Status & Key Results

- **Post-training (`post_training/`):** Implemented 8-GPU DDP Diffusion-PPO (`ddpo_single_window` / `ddpo_stitched_window`) with `combined_track`, `decomposed_vel`, `decomposed_vel_root_anchored`, and `stitched_seam` rewards, root-subspace KL anchoring on `[180:189]` (`--kl-root-weight`), hard-example mining, and dual-protocol kinematic + physical evaluation (`post_training/eval_drift_suite.py`).
- **Default inference (`tools/infer_video.py`):** Defaults to `--overlap 16 --align-mode yaw --blend-mode cosine --sync-mode multidiffusion` (`+0.9%` runtime overhead vs. un-synchronized overlap, zero extra forward passes per step).
- **Validated gains (`DDIM-50, eta=0.0` + MultiDiffusion 16-frame yaw stitching):**
  - **50/50 Model Soup (`hard_ppo_model_soup_50_50_step1200.pt`, linear average of `decomposed_step1200` + `subspace_kl_step1200`, delta cosine `+0.4861`):** Best Pareto balance between dynamic recovery and root-drift control — beats both parents on `val` Hard Bucket A (**8/21 = 38.1%**, **89.06%** overall vs. uniform-KL `7/21` / Subspace-KL `6/21` / Base `6/21`), achieves the lowest `L-MPJPE` (**284.3 mm** vs. Base `297.4` / uniform-KL `287.4` / Subspace-KL `291.1`) and `92.41%` `Seg250` (`+1/-0`) on `val` 96 long sequences, while cutting `W-MPJPE` / `RTE` / `Zmax` drift by **52% / 63% / 71%** vs. uniform-KL (`BEDLAM2 Zmax 77.1 mm < Base 78.2 mm`).
  - **Root-Subspace KL `6×` (`hard_ppo_subspace_kl_step1200.pt`, `[180:189]`):** **92.80%** Long (`+4/-0` vs. Base `89.60%`, zero regressions) & **97.79%** `Seg250` on held-out `test` 125; **100.00% (16/16)** `H36M` Long on `val` 96; on external EMDB/RICH eliminates **75%–114%** of `W-MPJPE`/`RTE` drift, improves `PA-MPJPE` over Base (`33.524`/`33.544`), recovers RICH `Long Success` to **86.11%** (`31/36`), and achieves the lowest RICH `P95 GRF` (`1.388 BW`) of all 7 compared methods.
  - **Uniform-KL `Hard-PPO step 1200` (`hard_ppo_decomposed_step1200.pt`):** **94.13%** on full `test` 392 long seqs (exceeding GT ceiling 93.37%), **90.27%** (`[87.28%, 93.02%]`) on `val` 401 × 3 seeds, **24/24 (100%)** on 8 real-world dance videos × 3 seeds, and strongest external `Push Success` (**91.04%** EMDB / **100.0%** RICH).

## PPO Post-Trained Checkpoints (ModelScope)

Large model weights (`~899 MB` each) exceed GitHub's 100 MB Git limit and free Git LFS quota, and GitHub Releases flattens directory paths. All post-trained checkpoints are synced with clean, readable paths on ModelScope:

- **ModelScope repository:** [https://www.modelscope.cn/models/Jasonhsc/vit-h-ppo](https://www.modelscope.cn/models/Jasonhsc/vit-h-ppo)
- **Format:** Weights-only `float32` inference checkpoints (`model_state_dict`, `imgfeat_encoder="sam3d"`, sanitized metadata; optimizer state stripped, bit-for-bit identical inference weights). Full SHA-256 digests are in `manifest.json` in the ModelScope repo.

```text
Jasonhsc/vit-h-ppo/
├── README.md
├── manifest.json
└── checkpoints/
    ├── hard_ppo_model_soup_50_50_step1200.pt  # Recommended default (50/50 Model Soup of uniform-KL + Subspace-KL 6x)
    ├── hard_ppo_subspace_kl_step1200.pt       # Subspace-KL 6x [180:189] (lowest drift & RICH P95 GRF)
    ├── hard_ppo_decomposed_step1200.pt        # Uniform-KL Hard-PPO step 1200 (strongest Push Success)
    ├── hard_ppo_decomposed_step0900.pt        # Short-window / hard-case (Bucket A)
    ├── pilot_uniform_step0500.pt              # Ablation: uniform-sampling PPO (500 steps)
    └── stitched_seam_step0800.pt              # Ablation: dual-window stitched-seam PPO (+800 steps)
```

| Checkpoint Path | Method | Step | Size | SHA-256 (prefix) | Recommended For |
| --- | --- | ---: | ---: | --- | --- |
| **`checkpoints/hard_ppo_model_soup_50_50_step1200.pt`** | 50/50 Model Soup (`decomposed_1200` + `subspace_kl_1200`) | 1200 | 898.85 MB | `1c2e2ca4b4149739` | **Default** — best Pareto balance (`val` Bucket A `8/21`, `L-MPJPE 284.3 mm`, `Seg250 92.41%`, 52%–71% lower drift vs uniform-KL) |
| `checkpoints/hard_ppo_subspace_kl_step1200.pt` | Hard-PPO + Subspace-KL `6×` `[180:189]` | 1200 | 898.86 MB | `864329b22dfb2aa8` | Strictest global-drift & GRF control (`test` 125 `92.80% +4/-0`, EMDB/RICH drift -75% to -114%, RICH P95 GRF `1.388 BW`) |
| `checkpoints/hard_ppo_decomposed_step1200.pt` | Hard-PPO (`decomposed_vel`, uniform KL) | 1200 | 898.86 MB | `c49cd815796bafd4` | Maximum external push robustness (`94.13%` test 392, `91.04%` EMDB / `100%` RICH Push) |
| `checkpoints/hard_ppo_decomposed_step0900.pt` | Hard-PPO (`decomposed_vel`, uniform KL) | 900 | 898.86 MB | `a0c5e44cfa16d177` | Short-window / high-dynamic hard cases (`Bucket A 7/22`, `Bucket B 0.1591 m`) |
| `checkpoints/pilot_uniform_step0500.pt` | Pilot Uniform PPO (`combined_track`) | 500 | 898.86 MB | `3f27f94248e26157` | Uniform-sampling ablation |
| `checkpoints/stitched_seam_step0800.pt` | Stitched-Seam PPO (`stitched_seam`) | 800 | 898.86 MB | `e5f48331c8e5c6f7` | Dual-window seam-reward ablation |

Download the recommended checkpoint (or all checkpoints) into the gitignored `inputs/checkpoints/vith_ppo/` directory:

```bash
pip install modelscope

# Download only the recommended Model Soup checkpoint
modelscope download --model Jasonhsc/vit-h-ppo \
  checkpoints/hard_ppo_model_soup_50_50_step1200.pt \
  --local_dir ./inputs/checkpoints/vith_ppo

# Or download the full checkpoint repo via Python SDK
python -c "from modelscope import snapshot_download; print(snapshot_download('Jasonhsc/vit-h-ppo', local_dir='./inputs/checkpoints/vith_ppo'))"
```

Run long-horizon inference with the downloaded PPO checkpoint (MultiDiffusion 16-frame yaw stitching and `sam3d` encoder auto-detection are enabled by default; add `--static-cam` for fixed-tripod videos):

```bash
python tools/infer_video.py \
  --video /path/to/input.mp4 \
  --ckpt inputs/checkpoints/vith_ppo/checkpoints/hard_ppo_model_soup_50_50_step1200.pt \
  --sam3d-root third-party/Fast-SAM-3D-Body \
  --sam3d-ckpt-path third-party/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/model.ckpt \
  --sam3d-mhr-path third-party/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --output outputs/inference/input_g1.npz
```

## Documentation & Workspace Guide

- [Deployment and external assets](docs/DEPLOYMENT.md)
- [Pinned base/tracker checkpoint download and verification](config/checkpoints.json):
  `python scripts/checkpoints.py list`
- [Architecture, PPO training, and MultiDiffusion stitching design](docs/PPO_INTEGRATION.md)
- [Four-person workflow and handoffs](docs/TEAM_WORKFLOW.md)
- [Validation benchmarks, ablations, and remaining limits](docs/VALIDATION.md)
- [Source import provenance](docs/SOURCE_IMPORT.md)

The unchanged upstream import is tagged `base/vith-release-2482c5c`. Keep checkpoints,
datasets, secrets and generated rollouts outside Git.

```bash
git clone https://github.com/Hsch22/vit-h-ppo.git
cd vit-h-ppo
```

Follow [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) to set up environments and download the upstream base/preprocessing assets (`vith_offline_g1.ckpt`, YOLO, ViTPose, SAM3D, and the SONIC ONNX tracker).

---

## Original upstream README

<div align="center">
<h1 style="border-bottom: none; margin-bottom: 0px ">ViT-H: Generative Video-To-Humanoid Control with Motion Tracking Feedback</h1>

[**Zimo He**](https://mileret.github.io/)<sup>&ast;</sup> · [**Tianyi Bian**](https://pku.ai/author/tianyi-bian/)<sup>&ast;</sup> · [**Zizhong Yang**](https://pku.ai/author/zizhong-yang/)<sup>&ast;</sup> · [**Hongjie Li**](https://awfuact.github.io/)<sup>&ast;&dagger;</sup> · [**Nan Jiang**](https://jnnan.github.io/) · [**Jieming Cui**](https://jiemingcui.github.io/) · [**Baoxiong Jia**](https://buzz-beater.github.io/) · [**Yizhou Wang**](https://cfcs.pku.edu.cn/wangyizhou/)<sup>&ddagger;</sup> · [**Siyuan Huang**](https://siyuanhuang.com/)<sup>&ddagger;</sup> · [**Yixin Zhu**](https://yzhu.io/)<sup>&ddagger;</sup> 
<br>
&ast;Equal Contribution&emsp;&dagger;Project Lead&emsp;&ddagger;Corresponding Authors&emsp;

<a href='https://arxiv.org/abs/2610.32966'><img src='https://img.shields.io/badge/arXiv-ViT--H-red' alt='arXiv'></a>
<a href='https://awfuact.github.io/vit-h/'><img src='https://img.shields.io/badge/Project_Page-ViT--H-green' alt='Project Page'></a>
<a href='https://youtu.be/j3q7f4-EUpk'><img src='https://img.shields.io/badge/Video-ViT--H-yellow' alt='Video'></a>
</div>

![alt text](assets/teaser.png)

This is the official implementation of <i>ViT-H: Generative Video-To-Humanoid Control with Motion Tracking Feedback.</i>

## Release Plan

- [x] Release inference code for offline video-to-humanoid control by November 1, 2026.
- [x] Release pre-training code for offline video-to-humanoid control by November 1, 2026.
- [ ] Release deployment code for humanoid video teleoperation by December 1, 2026.
- [ ] Release pre-training code for humanoid video teleoperation by December 1, 2026.
- [ ] Release post-training code by December 1, 2026.

## Installation

The offline release targets Linux, Python 3.10, CUDA 12.1, and PyTorch 2.3.

```bash
conda create -n vith python=3.10 -y
conda activate vith
pip install -r requirements.txt
pip install -e .
```

SAM3D preprocessing and inference require
[Fast-SAM-3D-Body](https://github.com/yangtiming/Fast-SAM-3D-Body). Clone it
into `third-party/Fast-SAM-3D-Body` and follow its environment setup guide:

```bash
mkdir -p third-party
git clone https://github.com/yangtiming/Fast-SAM-3D-Body.git \
  third-party/Fast-SAM-3D-Body
bash third-party/Fast-SAM-3D-Body/setup_env.sh
conda activate fast_sam_3d_body
```

The SAM3D weights are distributed by the SAM3D project rather than in the
ViT-H checkpoint repository.

DPVO camera estimation is optional. To enable `--use-dpvo`, install
[DPVO](https://github.com/princeton-vl/DPVO) under `third-party/DPVO`:

```bash
git clone --recursive https://github.com/princeton-vl/DPVO.git \
  third-party/DPVO
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip \
  -O /tmp/eigen-3.4.0.zip
unzip /tmp/eigen-3.4.0.zip -d third-party/DPVO/thirdparty
pip install ./third-party/DPVO
```

## Pretrained Models

Create the checkpoint root:

```bash
mkdir -p inputs/checkpoints
```

Download SMPL and SMPL-X from their official websites after registration:
[SMPL](https://smpl.is.tue.mpg.de/) and
[SMPL-X](https://smpl-x.is.tue.mpg.de/). Place the files as follows:

```text
inputs/checkpoints/
└── body_models/
    ├── smpl/
    │   └── SMPL_{GENDER}.pkl
    └── smplx/
        └── SMPLX_{GENDER}.npz
```

The [ViT-H checkpoint repository](https://huggingface.co/datasets/eerrr1/ViT-H-checkpoints)
contains the released offline model, YOLO, ViTPose, HMR2, the optional DPVO
weights, and the G1 tracker policy. Download the inference files into
the project root:

```bash
hf download eerrr1/ViT-H-checkpoints \
  vith_offline_g1.ckpt \
  inputs/checkpoints/dpvo/dpvo.pth \
  inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt \
  inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth \
  inputs/checkpoints/yolo/yolov8x.pt \
  --repo-type dataset \
  --local-dir .
```

For G1 policy tracking, download the ONNX policy separately so that it stays in
the ignored checkpoint directory:

```bash
hf download eerrr1/ViT-H-checkpoints \
  model_step_020000_g1.onnx \
  --repo-type dataset \
  --local-dir inputs/checkpoints/g1_tracker
```

The downloaded files follow this layout:

```text
vith_offline_g1.ckpt
inputs/checkpoints/
├── dpvo/
│   └── dpvo.pth
├── hmr2/
│   └── epoch=10-step=25000.ckpt
├── vitpose/
│   └── vitpose-h-multi-coco.pth
└── yolo/
    └── yolov8x.pt
```

The released configuration uses SAM3D image features. The HMR2 checkpoint is
included as an optional compatible image encoder. The SAM3D network and MHR
files are obtained through Fast-SAM-3D-Body and stay inside
`third-party/Fast-SAM-3D-Body/checkpoints/`.

## Dataset

We provide preprocessed data for training and evaluation. Due to licensing
restrictions, we do not redistribute the original datasets. Please obtain the
original data, including annotations and videos, directly from the respective
dataset websites. By downloading our preprocessed data, you agree to comply with
the terms of use of the corresponding datasets and to use the data for research
purposes only.

The released full experiment uses AMASS, BEDLAM, BEDLAM2, Human3.6M, and
BONES-SEED. Preprocessed data for the first four sources is distributed as four
compressed archives: `amass`, `bedlam`, `bedlam2`, and `h36m`. Download the
[ViT-H training dataset](https://huggingface.co/datasets/eerrr1/ViT-H) and
extract all archives into one directory.

```bash
hf download eerrr1/ViT-H \
  --repo-type dataset \
  --local-dir /path/to/training_set

cd /path/to/training_set
for archive in amass bedlam bedlam2 h36m; do
  tar --use-compress-program=unzstd -xf "${archive}.tar.zst"
done

export VITH_DATASET_ROOT=/path/to/training_set
```

The extracted layout is documented in [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md).
Dataset components and body-model assets may have terms separate from this
repository's code license.

BONES-SEED is available separately from its
[official Hugging Face repository](https://huggingface.co/datasets/bones-studio/seed).
Accept its gated-data license on Hugging Face, then download the proportional
SOMA motions and the matching G1 trajectories:

```bash
hf auth login
hf download bones-studio/seed \
  g1.tar.gz \
  soma_proportional.tar.gz \
  --repo-type dataset \
  --local-dir /path/to/bones-seed

cd /path/to/bones-seed
tar -xzf g1.tar.gz
tar -xzf soma_proportional.tar.gz
```

Convert the official 120 FPS files with the released BONES-SEED preprocessing
pipeline. It extracts the 28 proportional SOMA joints, converts the provided G1
CSV trajectories through SONIC MotionLib/FK, applies the same motion-selection
rules to both modalities, pairs them at 30 FPS, and writes session shards plus
the training index. The G1 CSV files are already retargeted by BONES-SEED; this
pipeline does not retarget the human BVH motions to G1.

```bash
cd /path/to/ViT-H-release
conda create -n bones-seed-preprocess python=3.10 -y
conda activate bones-seed-preprocess
pip install -r tools/bones_seed_preprocessing/requirements.txt

bash tools/bones_seed_preprocessing/run_preprocessing.sh \
  /path/to/bones-seed \
  /path/to/training_set/bones_seed \
  16
```

The training files are written to
`/path/to/training_set/bones_seed/paired/`. The full conversion is CPU- and
storage-intensive and must use a fresh output directory. See
[tools/bones_seed_preprocessing/README.md](tools/bones_seed_preprocessing/README.md)
for the intermediate layout, output schema, filtering rules, and provenance.

## Training

The released experiment uses a 16-layer transformer with latent dimension
1024, 16 attention heads, 120-frame windows, and mixed-source batches of 32.

```bash
export VITH_DATASET_ROOT=/path/to/training_set
export NUM_DEVICES=8
export WANDB_MODE=offline
bash scripts/train.sh
```

The default training configuration uses all five sources and therefore checks
for the converted BONES-SEED directory. The complete source and converted
layouts are documented in [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md).

For Slurm clusters, adjust the site-specific directives in
`slurm/train.sbatch`, export `VITH_DATASET_ROOT`, and submit:

```bash
sbatch slurm/train.sbatch
```

## Offline Inference

Model checkpoints are not stored in this repository. Pass the checkpoint path
explicitly. The released model was trained with SAM3D image features, so
inference requires both the SAM3D network checkpoint and its MHR body model:

```bash
python tools/infer_video.py \
  --video /path/to/input.mp4 \
  --ckpt /path/to/vith_offline_g1.ckpt \
  --sam3d-ckpt-path third-party/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/model.ckpt \
  --sam3d-mhr-path third-party/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --sam3d-root third-party/Fast-SAM-3D-Body \
  --output outputs/inference/input_g1.npz
```

For the documented `vith_offline_g1.ckpt`, the command selects SAM3D
automatically and refuses a conflicting encoder by default. Input videos are
timestamp-resampled to 30 FPS while preserving their duration. Preprocessing
caches are validated by hashes of the source video, effective settings, and model assets;
use `--force-recompute` to rebuild all managed artifacts or
`--force-recompute-imgfeat` to rebuild only image features. A user-provided
`preprocess/camera_K.pt` is validated and is never deleted by either option.
The selected SAM3D `model_config.yaml` and effective `IMG_SIZE`/`LAYER_DTYPE`
are part of that identity. Concurrent jobs using the same video-stem cache
directory are rejected by a process lock; use distinct `--output-root` and
`--output` paths for simultaneous jobs. See [cache behavior and regression
tests](docs/INFERENCE.md#timeline-and-cache-behavior) for details.

Video preprocessing additionally requires the external detector, pose,
body-model, and camera-estimation assets listed in
[docs/INFERENCE.md](docs/INFERENCE.md). Use `--static-cam` only for videos with
a known static camera. Otherwise SimpleVO estimates camera motion by default;
add `--use-dpvo` to use the installed DPVO package and `dpvo.pth` instead.

Generated G1 motions can optionally be tracked and evaluated in MuJoCo with the [G1 motion tracker](g1_tracker/README.md), whose entry point is `python g1_tracker/tools/track_g1.py`.

## Release Contents

- `hmr4d/`: datasets, model, preprocessing, and shared utilities.
- `motion_lib/`: G1 kinematics and rotation utilities.
- `scripts/train.sh`: reproducible offline training entry point.
- `tools/bones_seed_preprocessing/`: BONES-SEED-to-training-data preprocessing.
- `tools/infer_video.py`: end-to-end video inference entry point.
- `experiments/g1_training_set_exp1/`: resolved experiment configuration and checksums.
- `docs/`: dataset and inference documentation.

## Citation

If you find ViT-H helpful, please consider citing:
```bibtex
@article{he2026vith,
  title={ViT-H: Generative Video-To-Humanoid Control with Motion Tracking Feedback},
  author={He, Zimo and Bian, Tianyi and Yang, Zizhong and Li, Hongjie and Jiang, Nan and Cui, Jieming and Jia, Baoxiong and Wang, Yizhou and Huang, Siyuan and Zhu, Yixin},
  journal={arXiv preprint arXiv:2610.32966},
  year={2026}
}
```

## Acknowledgements

We thank the authors of [GVHMR](https://github.com/zju3dv/GVHMR), [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body), [Fast-SAM-3D-Body](https://github.com/yangtiming/Fast-SAM-3D-Body), [YOLOv8](https://github.com/ultralytics/ultralytics), [ViTPose](https://github.com/ViTAE-Transformer/ViTPose), [DPVO](https://github.com/princeton-vl/DPVO), [HMR 2.0 / 4D-Humans](https://github.com/shubham-goel/4D-Humans), [GMR](https://github.com/YanjieZe/GMR), [GEM](https://research.nvidia.com/labs/dair/gem/), [SONIC](https://nvlabs.github.io/GEAR-SONIC/), [HoloMotion](https://github.com/HorizonRobotics/HoloMotion), [ScaleBFM](https://github.com/zengweishuai/ScaleBFM), and [Humanoid-GPT](https://github.com/GalaxyGeneralRobotics/Humanoid-GPT) for their excellent prior work and publicly available code and models that supported this project.
