# PPO integration plan

## Scope and current status

The project will post-train the pretrained video-to-G1 motion generator with
PPO using feedback from a fixed motion tracker. This is a development plan,
not a runnable PPO implementation. The imported release provides supervised
diffusion training, inference, and a MuJoCo tracker. It does not provide a PPO
trainer, PPO rollout buffer, policy likelihood adapter, critic, or reward
definition. The upstream release checklist also marks post-training code as
unreleased.

The immutable import is tagged `base/vith-release-2482c5c`. Keep a frozen copy
of the base generator for evaluation and for any explicitly selected
regularization objective. The low-level SONIC policy is not the model being
post-trained.

## Verified base model

| Component | Released behavior | Owning source |
| --- | --- | --- |
| Diffusion | 1,000 training steps; cosine noise schedule; predicts clean motion `x0`; fixed variance | [Factory](../hmr4d/utils/diffusion/model_util.py#L5), [base configuration](../hmr4d/configs/base/g1_offline_model.yaml#L46) |
| Generator | Transformer with 16 layers, hidden width 1,024 and 16 attention heads; 120-frame training windows | [Resolved experiment](../experiments/g1_training_set_exp1/hydra_config.sanitized.yaml#L95) |
| Video condition | 14 normalized 2D joints, bounding box and intrinsics, camera angular velocity, and 1,024-D SAM3D image features | [Input construction](../tools/infer_video.py#L156), [condition assembly](../hmr4d/model/offline/pipeline/offline_pipeline.py#L233) |
| Released inference | 50-step DDIM, `eta=0.0`, classifier-free guidance scale `1.0`; fresh initial Gaussian noise | [Resolved sampling configuration](../experiments/g1_training_set_exp1/hydra_config.sanitized.yaml#L510), [sampling entry](../hmr4d/model/offline/pipeline/offline_pipeline.py#L624) |
| Supervised objective | Random diffusion timestep, noisy motion input, clean-motion MSE and configured geometry losses | [Training path](../hmr4d/model/offline/pipeline/offline_pipeline.py#L273), [motion loss](../hmr4d/model/offline/pipeline/offline_pipeline.py#L395) |

The 189-D normalized motion representation is:

- 174 values for 29 scalar joint angles encoded as 6D rotations;
- 6 values for root orientation in the camera frame;
- 6 values for root orientation in the gravity-view frame;
- 3 values for root displacement increments in the body frame.

Use the executable [encoder](../hmr4d/model/offline/utils/endecoder.py#L409)
and [decoder](../hmr4d/model/offline/utils/endecoder.py#L465) as the authority.
The field called `local_transl_vel` is a displacement increment in this
representation; do not silently interpret it as metres per second. World
orientation and position are recovered before forward kinematics in the
[pipeline](../hmr4d/model/offline/pipeline/offline_pipeline.py#L332).

Inference uses internal Y-up coordinates and exports Z-up motion. The
exported tracker fields are `joint_pos`, `root_pos_w`, `root_quat_w` (`wxyz`),
and `body_pos_w`. Preserve the released joint/body order, coordinate
conversion, normalization statistics, and frame rate. See
[export conversion](../tools/infer_video.py#L322) and
[tracker input schema](../g1_tracker/README.md#input).

Long-video inference currently overlaps adjacent windows by one frame and
rigidly aligns each later window to the preceding endpoint. This is part of
the baseline, not an implicit autoregressive PPO design. See
[window construction](../tools/infer_video.py#L177) and
[window alignment](../tools/infer_video.py#L189).

## Checkpoint and feature compatibility

Use [load_pipeline](../tools/inference_utils.py#L184) for baseline inference.
It composes the experiment configuration, accepts `state_dict` or
`model_state_dict`, validates all persistent state keys and shapes, and loads
strictly. Its behavior differs from the Lightning training helper's
non-strict loading. Successful file download alone does not establish model
compatibility.

The released checkpoint requires SAM3D image features. HMR2 features having
the same width does not make them interchangeable. Preserve the
[encoder compatibility check](../tools/inference_utils.py#L70). Store the
encoder identity in new checkpoint metadata so renamed PPO checkpoints do
not depend on the known-release filename rule. A comment mentioning DPO
snapshot loading does not imply that a DPO or PPO trainer is present.

## PPO decisions required before implementation

Record the selected design and its mathematical definition before adding a
training entry point. Unsupported or undefined states must fail explicitly.

1. **Policy and episode.** Define the policy state, action, transition, and
   episode boundary. In particular, specify whether PPO acts on diffusion
   denoising transitions and how a generated motion receives its execution
   feedback. Robot simulation steps and diffusion steps are different axes.
   Choose a training unit, such as one motion window, explicitly; do not
   inherit a long-video stitching rule without checking its consequences.

2. **Stochastic sampler and likelihood.** The released DDIM sampler has
   `sigma` proportional to `eta`. With `eta=0`, a denoising transition is
   deterministic conditional on its input, even though the initial noise is
   random. It cannot be assigned an ordinary nondegenerate Gaussian
   transition likelihood. Merely changing `eta` is also insufficient: the
   final transition explicitly suppresses noise. Define the stochastic
   transition distribution and treatment of deterministic steps before
   computing PPO ratios. See the
   [DDIM transition](../hmr4d/utils/diffusion/gaussian_diffusion.py#L822).

3. **Likelihood replay.** Specify which transitions, timesteps, conditions,
   valid-frame masks, actions, and old log probabilities are retained.
   Rollout and update must evaluate the same transition distribution and
   action, including CFG, variance, normalization, and dropout behavior.
   Document sums versus means over motion dimensions and timesteps. The
   existing [sampling loop](../hmr4d/utils/diffusion/gaussian_diffusion.py#L1179)
   runs under `no_grad`; PPO needs a separate, differentiable likelihood
   recomputation path. An unchanged policy must reproduce its stored log
   probabilities and yield probability ratios near one.

4. **Value and advantage.** Define the critic input and target, reward
   timing, return calculation, termination versus truncation, and advantage
   estimation. There is no existing value head. If the selected estimator
   omits a learned critic, document that choice rather than leaving a
   nominal PPO value component unimplemented.

5. **Reward and fidelity.** Version the reward terms, units, weights, and
   failure handling. Tracking success alone does not measure fidelity to
   the input video: a generator could produce an easy, nearly stationary
   reference. Select and verify a fidelity signal available for the chosen
   data. Keep held-out evaluation separate from reward tuning. Non-finite
   motions, simulator errors, and missing feedback must never become
   successful or silently dropped samples.

6. **KL and update scope.** State exactly which generator parameters are
   trainable and which are frozen. Define the distribution, direction,
   estimator, and reduction of any KL penalty; a motion-space distance is
   not automatically a KL divergence. Keep the frozen base policy distinct
   from the old rollout policy. Record clipping, learning rate, optimizer,
   update epochs, and reward scaling in the run configuration.

7. **Reproducibility.** A training checkpoint must identify the base
   checkpoint, feature encoder, source commit, data split, sampler, reward,
   tracker protocol, normalization statistics, optimizer, training step,
   and random states needed for the promised resume behavior. Validate
   save/load and resume rather than only producing a weight file.

## Fixed execution-feedback boundary

The included tracker uses a frozen SONIC ONNX policy with a 1,570-D input and
29-D output through ONNX Runtime's CPU execution provider. Its default
control rate is 50 Hz and the MuJoCo simulation timestep is 0.001 seconds
(1 kHz). Do not substitute the older presentation's 200 Hz simulation
setting. See [policy and runner](../g1_tracker/vith_tracker/tracker.py#L510).

The released [track_file](../g1_tracker/vith_tracker/tracker.py#L818) performs
initial support grounding, clips the initial joint pose, runs tracking, and
resamples the recorded trajectory. Batched input is currently processed
serially by sample. This API is a useful initial feedback boundary, but it
does not provide a vectorized PPO environment or differentiable physics.

Freeze and identify the ONNX policy, both robot XML files and assets, PD
control parameters, initial-state rules, frame-rate conversion, and success
protocol. Initial support grounding changes the reference used for success
evaluation; compare geometry in the intended frame instead of ignoring that
offset.

Protocol v4 reports a failure if pelvis height error exceeds 0.25 m,
squared pelvis orientation error exceeds 1.0 radian squared, or ankle/wrist
world-Z error exceeds 0.25 m. It exports `is_success` and
`first_failure_frame_50hz` along with grounding and initialization metadata.
See [success implementation](../g1_tracker/vith_tracker/success.py#L17) and
[output fields](../g1_tracker/README.md#output).

These fields are observations, not an established PPO reward. They also do
not establish equivalence to previously reported Push Success or Long
Success protocols; those require their own documented experiments.

## Integration milestones

| Milestone | Required evidence | Primary owners |
| --- | --- | --- |
| Reproduce the base | One real input video produces a valid motion and tracker output using pinned assets; preserve the commands, configuration and per-clip outputs | Training + metrics |
| Freeze the data and evaluation interface | A small development set and separate held-out split; stable clip IDs, feature metadata and failure records | Data + metrics |
| Resolve the PPO design | Written decisions for all items above, with the sampler and probability model checked together | Training, reviewed by metrics |
| Verify one update | Old-policy likelihood replay, finite losses and gradients, an intended parameter change, frozen tracker/base checks, and checkpoint reload | Training |
| Run a small experiment | Base versus PPO on the same held-out clips and seeds; report fidelity, execution outcomes, failures and compute cost | Metrics + training |
| Scale and report | Larger runs only after the small experiment is interpretable; report measured improvements and limitations without treating setup completion as success | All four roles |

The data owner should deliver a small validated batch early, then expand
coverage and assist with rollout jobs. The metrics owner should establish
the protocol before PPO begins. The report owner can maintain the
claim-to-experiment record from the first baseline run rather than waiting
for final results.
