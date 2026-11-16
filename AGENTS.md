# Team repository instructions

- Preserve upstream attribution, licenses, and the baseline tag.
- Inspect Git state before and after work; preserve unrelated changes.
- Resolve model, tensor, checkpoint and algorithm semantics from current code.
  State unresolved choices. Do not implement speculative fallbacks.
- The post-training target is the video-conditioned motion generator. Keep
  SONIC weights, robot XMLs, grounding and evaluation protocol fixed for
  base-versus-PPO comparisons unless the experiment explicitly changes them.
- Read `docs/PPO_INTEGRATION.md` before implementing PPO. Do not treat the
  deterministic DDIM path as a Gaussian policy with a valid density.
- Never commit checkpoints, raw data, credentials, or generated rollouts.
- Do not equate CPU tests or synthetic tracker checks with real-video GPU
  inference, PPO training, or reproduction of historical benchmark scores.
- Use `[type] Concise imperative English summary` commit messages.
- Check `docs/TEAM_WORKFLOW.md` before changing shared data or reward interfaces.
