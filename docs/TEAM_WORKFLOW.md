# Team workflow

The project will post-train the released video-to-G1 model with PPO. Keep the
tracking policy fixed when comparing the base model and post-trained models.
This document assigns work and proposes handoff records; it does not provide a
PPO implementation, manifest loader, or experiment-record writer.

## Four parallel roles

| Role | Owns | First handoff | Ongoing work |
| --- | --- | --- | --- |
| Data curation | Source inventory, selection reasons, deduplication, splits, preprocessing | A small usable batch and a fixed development/test inventory | Expand data; then help PPO with loading, caches, throughput, and failed-job recovery |
| PPO training | Policy sampling and probability semantics, rollout collection, updates, checkpoint save/resume | A minimal sample-to-rollout-to-reward-to-update experiment | Scale only after the update and resume path are verified |
| Metrics | Base evaluation, tracker protocol, metric definitions, independent comparisons | Per-clip base results and failure records from the first batch | Work with training on reward definitions; keep final evaluation distinct from training reward |
| Report | Research question, experiment registry, comparison tables, case selection, limitations | An experiment table with links to source artifacts | Record decisions and evidence from day one; summarize only verified results |

Do not wait for all data to be curated before starting training integration.
Metrics and report work also start with the first batch. After that handoff,
the data owner can help the training owner without changing policy semantics,
reward definitions, or the frozen test set independently.

## Suggested edit boundaries

- Data: existing `tools/video_preprocess.py`, `tools/preprocess_cache.py`, and
  dataset code only when a confirmed data requirement needs a change. Keep raw
  videos, features, and generated motions outside Git.
- PPO: put new post-training code in a dedicated directory agreed by the team
  (for example, `post_training/`, not yet implemented). Keep changes to
  `hmr4d/model/offline/` small and review them with the metrics owner.
- Metrics: maintain evaluation code separately from the PPO update loop. Changes
  to `g1_tracker/vith_tracker/success.py`, grounding, resampling, robot XML, or
  the tracker policy change the experimental setup and require a new protocol
  record and a rerun of the corresponding base comparison.
- Report: use `docs/` for written conclusions and small result summaries. Link
  to external checkpoints and run artifacts instead of committing them.

Use one feature branch per task and small pull requests. Examples are
`sicheng/data-curation`, `sicheng/ppo-integration`, `sicheng/evaluation`, and
`sicheng/report`. Each teammate should replace `sicheng/` with their own name
prefix. Coordinate changes to shared model files before editing them.

## Existing motion handoff

`tools/infer_video.py` writes a reference-motion NPZ. The tracker contract is
documented in [g1_tracker/README.md](../g1_tracker/README.md):

| Field | Single-sample shape | Convention |
| --- | --- | --- |
| `joint_pos` | `(T, 29)` | IsaacLab G1 joint order, radians |
| `root_pos_w` | `(T, 3)` | Z-up world coordinates, metres |
| `root_quat_w` | `(T, 4)` | World orientation, `wxyz` |
| `body_pos_w` | `(T, 30, 3)` | Z-up world coordinates, metres |
| `fps` | scalar | Source motion sample rate |

The tracker requires all four motion arrays. The inference exporter writes
`body_pos_w` only when the pipeline returns it, so check its presence before
handoff. Batched tracker inputs add a leading sample dimension.

`g1_tracker/tools/track_g1.py` executes at 50 Hz by default, evaluates success
at the control rate, and resamples executed motion back to source timestamps.
It writes executed-motion arrays plus `is_success`,
`first_failure_frame_50hz`, grounding and initialization diagnostics, and
`tracker_protocol_version` (currently `4`). Keep reference and executed NPZ
files separate. A successful export is not evidence of successful tracking.

The existing success criterion is not the full set of metrics in earlier
spreadsheet reports. Push tests, long-sequence tests, energy/contact metrics,
and PPO reward terms need explicit definitions and implementation before they
can be reported as reproduced results.

## Proposed records, not implemented interfaces

The following are collaboration requirements for future tooling. Existing
inference and tracker commands do not read a clip manifest or automatically
write a complete run record.

**Clip manifest:** unique `clip_id`; dataset and source-video/session identity;
source location and content hash; crop interval; train/dev/test split; camera
mode; preprocessing identity; selection decision and reason. Split at source
video/session level before producing windows. Overlapping clips, duplicate
exports, and derived windows of one source must stay in the same split. Keep
the test inventory out of reward tuning and data-selection iterations.

**Run record:** `run_id`; code commit; model checkpoint hash and base checkpoint
hash; clip-manifest version/hash; sampling seeds and effective configuration;
preprocessing configuration and model-asset hashes; tracker ONNX and both robot
XML hashes; tracker protocol version; reward definition/version; paths to
reference, executed motion, per-clip metrics, and logs. Record the preprocessing
cache identity from the actual run rather than assuming a shared directory is
equivalent. Follow [INFERENCE.md](INFERENCE.md) for cache behavior.

**Per-clip results:** retain successes, failures, and unavailable results with
reasons. Report the denominator and do not discard failed clips from aggregate
metrics. Keep training reward values separate from independently computed
evaluation metrics.

## Milestones and shared decisions

1. **Base smoke run:** one real video produces a reference NPZ and a tracker
   NPZ. Check schema, finite values, frame count, and recorded success/failure.
2. **Frozen base comparison:** evaluate the first development batch and reserve
   an independent test inventory. Freeze the tracker, preprocessing settings,
   and metric definitions used for base/PPO comparisons.
3. **PPO integration:** training and metrics owners agree on stochastic policy
   semantics, log-probabilities, rewards, and credit assignment. Verify a small
   update and checkpoint reload before a long training run. Do not substitute
   an unspecified reward or likelihood when an interface is missing.
4. **Paired evaluation:** compare base and post-trained checkpoints on the same
   held-out inputs with the same tracker, protocol, preprocessing assets, and
   documented seed schedule. Measure fidelity as well as tracking success;
   easier motions must not silently count as better imitation.
5. **Scale and report:** expand training data after the small run is usable.
   Report per-clip changes, failure cases, and compute cost alongside aggregates.

Any change to splits, sampling, rewards, or evaluation definitions must be
recorded before interpreting a new result. The report owner maintains the
decision log; the relevant implementation owner remains responsible for the
code and validation evidence.
