"""Unit tests for PPO policy/reward math and Hopper SDPA/RoPE optimizations."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn as nn

from post_training.policy import StochasticDDIMPolicy, build_ddim_schedule
from post_training.reward import (
    RewardWeights,
    compute_group_advantages,
    compute_sample_rewards,
    make_reward_weights_v2,
    make_reward_weights_v3,
)
from post_training.stitching import stitch_two_chunks_yup

try:
    from hmr4d.network.base_arch.transformer.encoder_rope import RoPEAttention

    HAS_ROPE_DEPS = True
except ImportError:
    RoPEAttention = None
    HAS_ROPE_DEPS = False


class _TinyDenoiser(nn.Module):
    """Minimal differentiable denoiser matching RelativeTransformer call signature."""

    def __init__(self, x_dim: int = 189, cond_dim: int = 16) -> None:
        super().__init__()
        self.proj_x = nn.Linear(x_dim, x_dim)
        self.proj_c = nn.Linear(cond_dim, x_dim)
        self.proj_t = nn.Linear(1, x_dim)
        nn.init.normal_(self.proj_x.weight, std=0.02)
        nn.init.zeros_(self.proj_x.bias)
        nn.init.normal_(self.proj_c.weight, std=0.02)
        nn.init.zeros_(self.proj_c.bias)
        nn.init.zeros_(self.proj_t.weight)
        nn.init.zeros_(self.proj_t.bias)

    def forward(self, f_cond, length, xt, timesteps, all_valid_length=False):
        t_feat = (timesteps.float() / 1000.0).view(-1, 1, 1)
        pred = self.proj_x(xt) + self.proj_c(f_cond) + self.proj_t(t_feat)
        return {"pred_x_start": pred}


def _make_fake_pipeline(num_timesteps: int = 1000) -> SimpleNamespace:
    betas = np.linspace(1e-4, 2e-2, num_timesteps, dtype=np.float64)
    alphas_cumprod = np.cumprod(1.0 - betas)
    return SimpleNamespace(
        train_diffusion=SimpleNamespace(
            num_timesteps=num_timesteps,
            alphas_cumprod=alphas_cumprod,
        )
    )


class TestPPOComponents(unittest.TestCase):
    def test_ddim_schedule_validation(self) -> None:
        sched = build_ddim_schedule(1000, 20)
        self.assertEqual(len(sched), 20)
        self.assertEqual(sched[0], 0)
        self.assertEqual(sched[-1], 999)
        self.assertEqual(sched, sorted(set(sched)))
        with self.assertRaises(ValueError):
            build_ddim_schedule(1000, 1)
        with self.assertRaises(ValueError):
            build_ddim_schedule(1000, 1001)

    def test_policy_rejects_non_positive_eta(self) -> None:
        pipe = _make_fake_pipeline()
        with self.assertRaises(ValueError):
            StochasticDDIMPolicy(pipe, ddim_steps=10, eta=0.0, use_bf16=False, device="cpu")
        with self.assertRaises(ValueError):
            StochasticDDIMPolicy(pipe, ddim_steps=10, eta=-0.1, use_bf16=False, device="cpu")
        with self.assertRaises(ValueError):
            StochasticDDIMPolicy(pipe, ddim_steps=10, eta=1.2, use_bf16=False, device="cpu")

    def test_rollout_replay_exact_ratio_and_gradients(self) -> None:
        torch.manual_seed(0)
        pipe = _make_fake_pipeline(100)
        policy = StochasticDDIMPolicy(pipe, ddim_steps=6, eta=0.7, use_bf16=False, device="cpu")
        self.assertEqual(float(policy.sigmas[0].item()), 0.0)
        self.assertTrue(bool((policy.sigmas[1:] > 0.0).all().item()))

        denoiser = _TinyDenoiser(x_dim=189, cond_dim=16)
        base_denoiser = _TinyDenoiser(x_dim=189, cond_dim=16)
        base_denoiser.load_state_dict(denoiser.state_dict())
        for p in base_denoiser.parameters():
            p.requires_grad_(False)

        f_cond = torch.randn(2, 10, 16)
        length = torch.tensor([10, 7], dtype=torch.long)
        gen = torch.Generator(device="cpu").manual_seed(123)
        rollout = policy.rollout_group(
            denoiser=denoiser,
            base_denoiser=base_denoiser,
            f_cond=f_cond,
            length=length,
            group_size=3,
            all_valid_length=False,
            generator=gen,
        )
        # Final deterministic step t == 0 is excluded -> ddim_steps - 1 stochastic steps.
        self.assertEqual(len(rollout.t_indices), 5)
        self.assertTrue(all(t > 0 for t in rollout.t_indices))
        self.assertEqual(tuple(rollout.x_0.shape), (6, 10, 189))

        f_cond_rep = f_cond.repeat_interleave(3, dim=0)
        length_rep = length.repeat_interleave(3, dim=0)

        # Replay before any update must match old_log_probs and produce zero KL.
        for s_idx in range(len(rollout.t_indices)):
            new_logp, kl = policy.replay_step(
                denoiser=denoiser,
                rollout=rollout,
                stoch_idx=s_idx,
                f_cond_rep=f_cond_rep,
                length_rep=length_rep,
                all_valid_length=False,
            )
            max_err = float((new_logp - rollout.old_log_probs[s_idx]).abs().max().item())
            self.assertLess(max_err, 1e-6)
            self.assertLess(float(kl.abs().max().item()), 1e-7)

        # Verify differentiable backward pass updates denoiser only.
        adv = torch.tensor([1.0, -0.5, -0.5, 0.8, -0.8, 0.0])
        new_logp, kl = policy.replay_step(denoiser, rollout, 0, f_cond_rep, length_rep)
        ratio = torch.exp(new_logp - rollout.old_log_probs[0])
        loss = -(ratio * adv).mean() + 0.04 * kl.mean()
        loss.backward()
        grad_norms = [float(p.grad.norm().item()) for p in denoiser.parameters() if p.grad is not None]
        self.assertTrue(all(np.isfinite(g) for g in grad_norms))
        self.assertGreater(sum(grad_norms), 0.0)
        for p in base_denoiser.parameters():
            self.assertIsNone(p.grad)

    def test_masked_log_prob_and_kl_ignore_padding(self) -> None:
        mean_a = torch.zeros(2, 6, 8)
        mean_b = torch.zeros(2, 6, 8)
        x_prev = torch.randn(2, 6, 8)
        mask = torch.tensor([[True, True, True, False, False, False], [True, True, True, True, False, False]])
        std = torch.tensor(0.5)

        lp1 = StochasticDDIMPolicy.gaussian_log_prob(x_prev, mean_a, std, mask)
        kl1 = StochasticDDIMPolicy.analytic_gaussian_kl(mean_a, mean_b, std, mask)

        # Corrupt padded frames heavily; masked outputs must remain identical.
        x_prev_mod = x_prev.clone()
        mean_b_mod = mean_b.clone()
        x_prev_mod[0, 3:] = 999.0
        mean_b_mod[1, 4:] = -500.0
        lp2 = StochasticDDIMPolicy.gaussian_log_prob(x_prev_mod, mean_a, std, mask)
        kl2 = StochasticDDIMPolicy.analytic_gaussian_kl(mean_a, mean_b_mod, std, mask)

        self.assertTrue(torch.allclose(lp1, lp2, atol=1e-6))
        self.assertTrue(torch.allclose(kl1, kl2, atol=1e-6))

    def test_rewards_and_group_advantages(self) -> None:
        rng = np.random.default_rng(7)
        gt_jp = rng.normal(scale=0.2, size=(4, 12, 29)).astype(np.float32)
        gt_bp = rng.normal(scale=0.5, size=(4, 12, 30, 3)).astype(np.float32)

        pred_jp = gt_jp.copy()
        pred_bp = gt_bp.copy()
        # Sample 0: exact match + constant global shift (root-relative fidelity should be 1.0)
        pred_bp[0] += np.array([1.5, -2.0, 0.3], dtype=np.float32)
        # Sample 1: noisy match + failed half-way
        pred_jp[1] += 0.4
        pred_bp[1] += 0.3
        # Sample 2: invalid tracker
        # Sample 3: NaN prediction
        pred_jp[3, 0, 0] = np.nan

        trk = [
            {"valid": True, "is_success": True, "first_failure_frame_50hz": -1, "total_frames_50hz": 20, "exec_joint_mae_rad": 0.0, "exec_root_pos_err_m": 0.0},
            {"valid": True, "is_success": False, "first_failure_frame_50hz": 10, "total_frames_50hz": 20, "exec_joint_mae_rad": 0.25, "exec_root_pos_err_m": 0.25},
            {"valid": False, "is_success": False, "first_failure_frame_50hz": 0, "total_frames_50hz": 20, "exec_joint_mae_rad": float("nan"), "exec_root_pos_err_m": float("nan")},
            {"valid": True, "is_success": True, "first_failure_frame_50hz": -1, "total_frames_50hz": 20, "exec_joint_mae_rad": 0.0, "exec_root_pos_err_m": 0.0},
        ]
        weights = RewardWeights()
        out = compute_sample_rewards(trk, pred_jp, pred_bp, gt_jp, gt_bp, np.array([12, 12, 12, 12]), weights)
        self.assertAlmostEqual(float(out["fidelity"][0]), 1.0, places=5)
        self.assertAlmostEqual(float(out["reward"][0]), 2.8, places=5)  # 1.0 + 0.5 + 0.3 + 1.0
        self.assertLess(float(out["reward"][1]), float(out["reward"][0]))
        self.assertFalse(bool(out["is_valid"][2]))
        self.assertFalse(bool(out["is_valid"][3]))
        self.assertEqual(float(out["reward"][2]), weights.invalid_penalty)
        self.assertEqual(float(out["reward"][3]), weights.invalid_penalty)

        # Group advantages: group 0 has variance, group 1 has zero variance.
        adv = compute_group_advantages(torch.tensor([2.0, 1.0, 0.5, 0.5]), group_size=2)
        self.assertGreater(float(adv[0]), 0.0)
        self.assertLess(float(adv[1]), 0.0)
        self.assertEqual(float(adv[2]), 0.0)
        self.assertEqual(float(adv[3]), 0.0)

    @unittest.skipUnless(HAS_ROPE_DEPS, "einops/timm required for RoPEAttention test")
    def test_rope_sdpa_matches_manual_attention(self) -> None:
        torch.manual_seed(42)
        dim, num_heads, seq_len, bsz = 64, 4, 24, 2
        attn = RoPEAttention(embed_dim=dim, num_heads=num_heads, dropout=0.0).eval()
        x = torch.randn(bsz, seq_len, dim)
        # True means masked out in RoPEAttention
        key_padding_mask = torch.zeros(bsz, seq_len, dtype=torch.bool)
        key_padding_mask[1, 18:] = True

        with torch.no_grad():
            y_sdpa_unmasked = attn(x, attn_mask=None, key_padding_mask=None)
            y_sdpa_masked = attn(x, attn_mask=None, key_padding_mask=key_padding_mask)

            # Reference manual attention
            q = attn.query(x).reshape(bsz, seq_len, num_heads, dim // num_heads).transpose(1, 2)
            k = attn.key(x).reshape(bsz, seq_len, num_heads, dim // num_heads).transpose(1, 2)
            v = attn.value(x).reshape(bsz, seq_len, num_heads, dim // num_heads).transpose(1, 2)
            q = attn.rope.rotate_queries_or_keys(q)
            k = attn.rope.rotate_queries_or_keys(k)
            scale = (dim // num_heads) ** -0.5
            logits = torch.matmul(q, k.transpose(-2, -1)) * scale
            y_ref_unmasked = attn.proj(torch.matmul(torch.softmax(logits, dim=-1), v).transpose(1, 2).reshape(bsz, seq_len, dim))

            logits_masked = logits.masked_fill(key_padding_mask.reshape(bsz, 1, 1, seq_len), float("-inf"))
            y_ref_masked = attn.proj(torch.matmul(torch.softmax(logits_masked, dim=-1), v).transpose(1, 2).reshape(bsz, seq_len, dim))

        self.assertLess(float((y_sdpa_unmasked - y_ref_unmasked).abs().max().item()), 1e-5)
        self.assertLess(float((y_sdpa_masked - y_ref_masked).abs().max().item()), 1e-5)

    def test_reward_v2_root_exec_and_velocity_preservation(self) -> None:
        t_len = 20
        t = np.linspace(0.0, 2.0, t_len, dtype=np.float32)[:, None]
        gt_jp = np.repeat(np.sin(3.0 * t), 29, axis=1)
        gt_bp = np.zeros((t_len, 30, 3), dtype=np.float32)

        # Sample 0: matches GT velocity and has low root error (0.05m)
        # Sample 1: matches GT velocity but high root error (0.25m) -> much lower reward_v2
        # Sample 2: sluggish (0.2x amplitude -> much lower joint speed) with same low root error -> lower vel_fidelity
        pred_jp = np.stack([gt_jp, gt_jp, 0.2 * gt_jp], axis=0)
        pred_bp = np.stack([gt_bp, gt_bp, gt_bp], axis=0)
        gt_jp_b = np.stack([gt_jp, gt_jp, gt_jp], axis=0)
        gt_bp_b = np.stack([gt_bp, gt_bp, gt_bp], axis=0)

        trk = [
            {"valid": True, "is_success": True, "first_failure_frame_50hz": -1, "total_frames_50hz": 33, "exec_joint_mae_rad": 0.08, "exec_root_pos_err_m": 0.05},
            {"valid": True, "is_success": True, "first_failure_frame_50hz": -1, "total_frames_50hz": 33, "exec_joint_mae_rad": 0.08, "exec_root_pos_err_m": 0.25},
            {"valid": True, "is_success": True, "first_failure_frame_50hz": -1, "total_frames_50hz": 33, "exec_joint_mae_rad": 0.08, "exec_root_pos_err_m": 0.05},
        ]
        w2 = make_reward_weights_v2()
        out = compute_sample_rewards(trk, pred_jp, pred_bp, gt_jp_b, gt_bp_b, np.array([t_len] * 3), w2, fps=30.0)
        # Lower root error must give substantially higher reward (> +0.5 gap under w_exec_root=1.2, scale=0.15)
        self.assertGreater(float(out["reward"][0] - out["reward"][1]), 0.5)
        # Sluggish sample must have strictly lower vel_fidelity and strictly lower reward than sample 0
        self.assertLess(float(out["vel_fidelity"][2]), float(out["vel_fidelity"][0]) - 0.5)
        self.assertLess(float(out["reward"][2]), float(out["reward"][0]) - 0.2)

    def test_hard_clip_distributed_sampler_excludes_untrackable_gt(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        try:
            from post_training.dataset import HardClipDistributedSampler
        except ImportError:
            self.skipTest("dataset deps unavailable in minimal CPU test env")

        payload = {
            "buckets": {
                "bucket_a_fixable_fail": [{"global_idx": i} for i in range(0, 10)],
                "bucket_b_high_exec_err": [{"global_idx": i} for i in range(10, 110)],
                "bucket_c_easy_pass": [{"global_idx": i} for i in range(110, 310)],
                "untrackable_gt": [{"global_idx": i} for i in range(310, 360)],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            idx_path = Path(tmp) / "hard_index.json"
            idx_path.write_text(json.dumps(payload))
            sampler = HardClipDistributedSampler(
                idx_path,
                dataset_len=360,
                num_replicas=2,
                rank=0,
                share_a=0.20,
                share_b=0.55,
                share_c=0.25,
                num_samples_per_epoch=20000,
                seed=7,
            )
            draws = list(sampler)
            self.assertEqual(len(draws), 10000)
            self.assertEqual(sum(1 for d in draws if d >= 310), 0)
            frac_a = sum(1 for d in draws if 0 <= d < 10) / len(draws)
            frac_b = sum(1 for d in draws if 10 <= d < 110) / len(draws)
            frac_c = sum(1 for d in draws if 110 <= d < 310) / len(draws)
            self.assertAlmostEqual(frac_a, 0.20, delta=0.02)
            self.assertAlmostEqual(frac_b, 0.55, delta=0.02)
            self.assertAlmostEqual(frac_c, 0.25, delta=0.02)

    def test_reward_v3_seam_consistency_penalty(self) -> None:
        t_len = 24
        gt_jp = np.zeros((3, t_len, 29), dtype=np.float32)
        gt_bp = np.zeros((3, t_len, 30, 3), dtype=np.float32)
        trk = [
            {"valid": True, "is_success": True, "first_failure_frame_50hz": -1, "total_frames_50hz": 40, "exec_joint_mae_rad": 0.08, "exec_root_pos_err_m": 0.10}
            for _ in range(3)
        ]
        # Sample 0: zero overlap err & seam_jump <= interior_step -> seam_score == 1.0
        # Sample 1: moderate seam jump & overlap err -> intermediate seam_score
        # Sample 2: 6x stitch-seam jump (0.084 rad vs 0.014 interior) -> sharp penalty
        ov = np.array([0.0, 0.03, 0.09], dtype=np.float32)
        sj = np.array([0.010, 0.044, 0.084], dtype=np.float32)
        istep = np.array([0.014, 0.014, 0.014], dtype=np.float32)

        w3 = make_reward_weights_v3()
        out = compute_sample_rewards(
            trk, gt_jp, gt_bp, gt_jp, gt_bp, np.array([t_len] * 3), w3, fps=30.0,
            seam_overlap_err=ov, seam_joint_jump=sj, interior_joint_step=istep,
        )
        self.assertAlmostEqual(float(out["seam_score"][0]), 1.0, places=5)
        self.assertGreater(float(out["seam_score"][0]), float(out["seam_score"][1]))
        self.assertGreater(float(out["seam_score"][1]), float(out["seam_score"][2]))
        self.assertGreater(float(out["reward"][0] - out["reward"][2]), 0.5)

    def test_stitched_key_weighted_sampler(self) -> None:
        try:
            from post_training.dataset_stitched import StitchedKeyWeightedDistributedSampler
        except ImportError:
            self.skipTest("dataset_stitched deps unavailable in minimal CPU test env")

        meta_keys = (
            [("bedlam2", f"hard_{i}") for i in range(10)]
            + [("bedlam2", f"easy_{i}") for i in range(10)]
            + [("h36m", f"bad_{i}") for i in range(5)]
        )
        untrackable = {("h36m", f"bad_{i}") for i in range(5)}
        hard_ab = {("bedlam2", f"hard_{i}") for i in range(10)}
        sampler = StitchedKeyWeightedDistributedSampler(
            meta_keys=meta_keys,
            untrackable_keys=untrackable,
            hard_ab_keys=hard_ab,
            num_replicas=2,
            rank=0,
            hard_weight_mult=3.0,
            num_samples_per_epoch=20000,
            seed=11,
        )
        self.assertEqual(sampler.stats, {"hard_pairs": 10, "easy_pairs": 10, "excluded_untrackable": 5})
        draws = list(sampler)
        self.assertEqual(len(draws), 10000)
        # Zero untrackable draws (indices 20..24)
        self.assertEqual(sum(1 for d in draws if d >= 20), 0)
        # Expected hard fraction = (10*3) / (10*3 + 10*1) = 0.75
        frac_hard = sum(1 for d in draws if d < 10) / len(draws)
        self.assertAlmostEqual(frac_hard, 0.75, delta=0.02)

    def test_batched_stitch_matches_infer_video_align(self) -> None:
        from tools.infer_video import align_window_to_previous

        torch.manual_seed(17)
        n, w = 4, 12
        jp0 = torch.randn(n, w, 29)
        jp1 = torch.randn(n, w, 29)
        rp0 = torch.randn(n, w, 3)
        rp1 = torch.randn(n, w, 3)
        rq0 = torch.nn.functional.normalize(torch.randn(n, w, 4), dim=-1)
        rq1 = torch.nn.functional.normalize(torch.randn(n, w, 4), dim=-1)
        bp0 = torch.randn(n, w, 30, 3)
        bp1 = torch.randn(n, w, 30, 3)

        jp_s, rp_s, rq_s, bp_s, seam = stitch_two_chunks_yup(jp0, rp0, rq0, bp0, jp1, rp1, rq1, bp1)
        self.assertEqual(tuple(jp_s.shape), (n, 2 * w - 1, 29))
        self.assertEqual(tuple(rp_s.shape), (n, 2 * w - 1, 3))
        self.assertEqual(tuple(rq_s.shape), (n, 2 * w - 1, 4))
        self.assertEqual(tuple(bp_s.shape), (n, 2 * w - 1, 30, 3))

        for i in range(n):
            ref_rp1, ref_rq1, ref_bp1 = align_window_to_previous(
                rp1[i], rq1[i], bp1[i], rp0[i, -1], rq0[i, -1]
            )
            self.assertLess(float((rp_s[i, w:] - ref_rp1[1:]).abs().max().item()), 1e-5)
            self.assertLess(float((bp_s[i, w:] - ref_bp1[1:]).abs().max().item()), 1e-5)
            # Quaternions q and -q represent identical rotation
            dot = (rq_s[i, w:] * ref_rq1[1:]).sum(dim=-1).abs()
            self.assertTrue(bool((dot > 1.0 - 1e-5).all().item()))
            # Verify exact seam metrics
            self.assertAlmostEqual(
                float(seam["overlap_err"][i].item()),
                float((jp1[i, 0] - jp0[i, -1]).abs().mean().item()),
                places=6,
            )
            self.assertAlmostEqual(
                float(seam["seam_jump"][i].item()),
                float((jp_s[i, w] - jp_s[i, w - 1]).abs().mean().item()),
                places=6,
            )


if __name__ == "__main__":
    unittest.main()
