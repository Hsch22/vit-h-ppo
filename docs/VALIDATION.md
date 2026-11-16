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
