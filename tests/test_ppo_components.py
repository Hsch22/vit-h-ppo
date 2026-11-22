"""Unit tests for PPO policy/reward math and Hopper SDPA/RoPE optimizations."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn as nn

from post_training.policy import StochasticDDIMPolicy, build_ddim_schedule
from post_training.reward import RewardWeights, compute_group_advantages, compute_sample_rewards

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


if __name__ == "__main__":
    unittest.main()
