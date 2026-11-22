# Repository structure

```text
ViT-H-release/
|-- README.md
|-- LICENSE
|-- STRUCTURE.md
|-- PROVENANCE.md
|-- requirements.txt
|-- setup.py
|-- pyproject.toml
|-- .env.example
|-- .gitignore
|-- .gitattributes
|-- canonicalize_motion.py
|-- roma.py
|-- assets/
|   `-- teaser.png
|-- data/
|   |-- g1_training_set_exp1_stats.npz
|   |-- g1_training_set_exp1_pred_cam_stats.npz
|   `-- bones_seed_g1_soma_pair_index.json
|-- docs/
|   |-- DATA_LAYOUT.md
|   |-- DEPLOYMENT.md
|   |-- INFERENCE.md
|   |-- PPO_INTEGRATION.md
|   `-- VALIDATION.md
|-- g1_tracker/
|   |-- vith_tracker/
|   |-- tools/track_g1.py
|   `-- tests/
|-- post_training/
|   |-- policy.py
|   |-- reward.py
|   |-- stitching.py
|   |-- dataset.py
|   |-- dataset_stitched.py
|   |-- tracker_pool.py
|   |-- tracker_worker.py
|   |-- train_ppo.py
|   |-- train_ppo_stitched.py
|   |-- mine_hard_clips.py
|   |-- eval_compare.py
|   `-- eval_long_horizon.py
|-- scripts/
|   |-- checkpoints.py
|   |-- preflight.py
|   |-- train.sh
|   |-- train_ppo.sh
|   |-- stage_training_set.sh
|   `-- check_release.sh
|-- tests/
|   |-- test_infra_tools.py
|   |-- test_ppo_components.py
|   |-- test_preprocess_review_followup.py
|   |-- test_release_inference_fixes.py
|   `-- test_video_timeline_files.py
|-- experiments/
|   `-- g1_training_set_exp1/
|       |-- hydra_config.sanitized.yaml
|       |-- hydra_overrides.yaml
|       `-- SHA256SUMS
|-- hmr4d/
|   |-- configs/
|   |-- datamodule/
|   |-- dataset/
|   |-- model/
|   |-- network/
|   `-- utils/
|-- inputs/
|   `-- README.md
|-- motion_lib/
|-- slurm/
|   `-- train.sbatch
|-- tools/
|   |-- train.py
|   |-- bones_seed_preprocessing/
|   |   |-- README.md
|   |   |-- requirements.txt
|   |   |-- run_preprocessing.sh
|   |   |-- SOURCE_MANIFEST.json
|   |   |-- GR00T-WholeBodyControl/
|   |   `-- src/
|   |-- infer_video.py
|   |-- inference_utils.py
|   `-- video_preprocess.py
`-- unitree_description/
    `-- mjcf/
        `-- g1.xml
```

## Responsibilities

- `hmr4d/`: model architecture, training datasets, shared utilities,
  and Hydra configuration required by the released experiment and inference.
- `hmr4d/configs/exp/offline/`: the single released experiment
  configuration; its direct inheritance fragments are under
  `hmr4d/configs/base/`.
- `tools/train.py`: Lightning/Hydra training entry point.
- `tools/bones_seed_preprocessing/`: converts official proportional SOMA BVH
  and matched G1 CSV trajectories through the released SONIC preprocessing and
  FK path, then writes paired BONES-SEED training shards and their index.
- `tools/infer_video.py`: video-to-G1 offline inference entry point.
- `tools/inference_utils.py`: checkpoint loading and output-angle utilities.
- `tools/video_preprocess.py`: tracking, pose, feature, and camera preprocessing.
- `data/`: small metadata required by this experiment; no raw datasets.
- `experiments/`: sanitized configuration emitted by the training run.
- `motion_lib/` and `unitree_description/`: G1 forward kinematics.
- `post_training/`: flat single-level Diffusion-PPO package (stochastic DDIM
  policy & replay, execution/fidelity rewards, multi-frame overlap stitching,
  persistent cross-env SONIC/MuJoCo tracker pool, hard-clip mining, DDP
  trainers, and paired short/long-horizon evaluators).
- `g1_tracker/`: frozen SONIC ONNX + MuJoCo Protocol v4 tracker runtime and tests.
- `scripts/` and `slurm/`: preflight, checkpoint verification, data staging,
  and supervised/PPO launch scripts.
- `tests/`: CPU-runnable regression suite covering inference, preprocessing,
  checkpoint validation, PPO math, and multi-frame stitching.
- `inputs/`: instructions for separately licensed preprocessing assets.

## Deliberately excluded

- all training checkpoints;
- raw training datasets and private records;
- experiment outputs and W&B run data;
- Blender binaries/assets and rendering-only scripts;
- batch orchestration and site-specific job scripts;
- diagnostic, benchmark, migration, and historical backup files;
- Python bytecode, caches, editor settings, and local environment files.
