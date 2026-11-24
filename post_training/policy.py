"""Stochastic DDIM policy, differentiable likelihood replay, analytic KL, and G1 Z-up decoding.

Implements the explicit stochastic policy semantics required by docs/PPO_INTEGRATION.md:
- Uses DDIM with explicit eta in (0, 1] over S denoising steps.
- Only transitions with sigma_k > 0 (t_k > 0) contribute to policy log-probability and KL.
- The final deterministic projection step (t_k == 0, sigma == 0) is excluded from density terms.
- Log-probabilities and analytic KL are averaged over valid (frame, 189) dimensions so
  PPO importance ratios stay numerically well-conditioned.
- Rollout and replay share the exact same step_distribution() math with dropout disabled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

G1_PRED_X_DIM = 189
G1_ROOT_SUBSPACE_START = 180  # [180:186] global_orient_gv (6D) + [186:189] local_transl_vel (3D)

# Exact Y-up -> G1 Z-up matrix from tools/infer_video.py lines 323-332.
YUP_TO_ZUP_M = (
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
)


def build_ddim_schedule(original_num_steps: int, ddim_steps: int) -> list[int]:
    """Build strictly increasing original diffusion timestep indices matching SpacedDiffusion."""
    if ddim_steps < 2 or ddim_steps > original_num_steps:
        raise ValueError(f"ddim_steps must be in [2, {original_num_steps}], got {ddim_steps}")
    frac_stride = (original_num_steps - 1) / (ddim_steps - 1)
    steps = [round(frac_stride * i) for i in range(ddim_steps)]
    if len(set(steps)) != ddim_steps or steps[0] != 0:
        raise ValueError(f"Invalid DDIM schedule for steps={ddim_steps}: {steps[:5]}...{steps[-5:]}")
    return steps


@dataclass
class DiffusionRolloutBatch:
    """Stored stochastic transitions for differentiable PPO likelihood replay."""

    x_t: torch.Tensor              # (N_stoch, B_tot, L, 189) float32 on CPU/GPU
    x_prev: torch.Tensor           # (N_stoch, B_tot, L, 189) float32
    t_indices: list[int]           # length N_stoch, index into schedule (descending, > 0)
    old_log_probs: torch.Tensor    # (N_stoch, B_tot) float32
    base_means: torch.Tensor       # (N_stoch, B_tot, L, 189) float32 (from frozen base policy)
    x_0: torch.Tensor              # (B_tot, L, 189) float32 terminal generated motion
    valid_mask: torch.Tensor       # (B_tot, L) bool


class StochasticDDIMPolicy:
    """Stochastic DDIM transition model over a pretrained ViT-H denoiser."""

    def __init__(
        self,
        pipeline: Any,
        ddim_steps: int = 20,
        eta: float = 0.7,
        use_bf16: bool = True,
        device: torch.device | str = "cuda",
        kl_root_weight: float = 1.0,
    ) -> None:
        if not (0.0 < float(eta) <= 1.0):
            raise ValueError(f"PPO stochastic DDIM requires 0 < eta <= 1.0, got {eta}")
        if float(kl_root_weight) <= 0.0:
            raise ValueError(f"kl_root_weight must be > 0, got {kl_root_weight}")
        self.device = torch.device(device)
        self.eta = float(eta)
        self.ddim_steps = int(ddim_steps)
        self.use_bf16 = bool(use_bf16) and self.device.type == "cuda"
        self.kl_root_weight = float(kl_root_weight)
        dim_w = torch.ones((G1_PRED_X_DIM,), dtype=torch.float32, device=self.device)
        if self.kl_root_weight != 1.0:
            dim_w[G1_ROOT_SUBSPACE_START:] = self.kl_root_weight
            dim_w = dim_w / dim_w.mean()
        self.kl_dim_weights: torch.Tensor | None = (
            dim_w.view(1, 1, G1_PRED_X_DIM) if self.kl_root_weight != 1.0 else None
        )

        base_diff = pipeline.train_diffusion
        orig_steps = int(base_diff.num_timesteps)
        self.schedule = build_ddim_schedule(orig_steps, self.ddim_steps)

        alphas_full = np.asarray(base_diff.alphas_cumprod, dtype=np.float64)
        ab = np.array([alphas_full[t] for t in self.schedule], dtype=np.float64)
        ab_prev = np.append(1.0, ab[:-1])

        # Precompute exact Equation 12 DDIM coefficients in float64 -> float32 tensors on device.
        self.timesteps_orig = torch.tensor(self.schedule, dtype=torch.long, device=self.device)
        self.alphas_cumprod = torch.tensor(ab, dtype=torch.float32, device=self.device)
        self.alphas_cumprod_prev = torch.tensor(ab_prev, dtype=torch.float32, device=self.device)

        sigma = (
            self.eta
            * np.sqrt(np.maximum((1.0 - ab_prev) / (1.0 - ab), 0.0))
            * np.sqrt(np.maximum(1.0 - ab / ab_prev, 0.0))
        )
        sigma[0] = 0.0  # t == 0 is deterministic (matches gaussian_diffusion.py line 840)
        self.sigmas = torch.tensor(sigma, dtype=torch.float32, device=self.device)
        self.sqrt_ab = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_ab = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_ab_prev = torch.sqrt(self.alphas_cumprod_prev)
        self.dir_coeff = torch.sqrt(torch.clamp(1.0 - self.alphas_cumprod_prev - self.sigmas**2, min=0.0))

        # Check that all steps k >= 1 have strictly positive variance.
        if bool((self.sigmas[1:] <= 1e-7).any().item()):
            raise RuntimeError(f"Degenerate DDIM sigma encountered: {self.sigmas.tolist()}")

    def _ddim_mean_std(
        self,
        x_t: torch.Tensor,
        step_idx: int,
        pred_x0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute exact DDIM transition mean and scalar std in float32."""
        x_t_f = x_t.float()
        pred_x0_f = pred_x0.float()
        sqrt_ab = self.sqrt_ab[step_idx]
        sqrt_1m_ab = self.sqrt_one_minus_ab[step_idx]
        eps = (x_t_f - sqrt_ab * pred_x0_f) / sqrt_1m_ab.clamp(min=1e-8)
        mean = self.sqrt_ab_prev[step_idx] * pred_x0_f + self.dir_coeff[step_idx] * eps
        std = self.sigmas[step_idx]
        return mean, std

    def predict_x0(
        self,
        denoiser: torch.nn.Module,
        x_t: torch.Tensor,
        step_idx: int,
        f_cond: torch.Tensor,
        length: torch.Tensor,
        all_valid_length: bool = False,
    ) -> torch.Tensor:
        bsz = x_t.shape[0]
        ts = self.timesteps_orig[step_idx].expand(bsz)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            out = denoiser(
                f_cond=f_cond,
                length=length,
                xt=x_t,
                timesteps=ts,
                all_valid_length=all_valid_length,
            )
        return out["pred_x_start"].float()

    @staticmethod
    def gaussian_log_prob(
        x_prev: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-element mean Gaussian log-prob over valid (frame, 189) dimensions -> (B,)."""
        var = (std * std).clamp(min=1e-12)
        log_2pi_var = math.log(2.0 * math.pi) + torch.log(var)
        elem_logp = -0.5 * (((x_prev.float() - mean.float()) ** 2) / var + log_2pi_var)
        mask_f = valid_mask.unsqueeze(-1).to(dtype=elem_logp.dtype)  # (B, L, 1)
        denom = (mask_f.sum(dim=(1, 2)) * elem_logp.shape[-1]).clamp(min=1.0)
        return (elem_logp * mask_f).sum(dim=(1, 2)) / denom

    @staticmethod
    def analytic_gaussian_kl(
        mean_policy: torch.Tensor,
        mean_base: torch.Tensor,
        std: torch.Tensor,
        valid_mask: torch.Tensor,
        dim_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Exact closed-form KL(N(mu_p, s^2) || N(mu_b, s^2)) averaged over valid elements -> (B,).

        When ``dim_weights`` (1, 1, D) with mean 1.0 is supplied, up-weights the coupled
        root trajectory subspace ``[180:189]`` without altering total KL scale or PPO log_probs.
        """
        var = (std * std).clamp(min=1e-12)
        elem_kl = 0.5 * ((mean_policy.float() - mean_base.float()) ** 2) / var
        if dim_weights is not None:
            elem_kl = elem_kl * dim_weights.to(device=elem_kl.device, dtype=elem_kl.dtype)
        mask_f = valid_mask.unsqueeze(-1).to(dtype=elem_kl.dtype)
        denom = (mask_f.sum(dim=(1, 2)) * elem_kl.shape[-1]).clamp(min=1.0)
        return (elem_kl * mask_f).sum(dim=(1, 2)) / denom

    @torch.no_grad()
    def build_condition(self, pipeline: Any, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Compute frozen ConditionEmbedder features once for the batch."""
        from hmr4d.utils.geo.hmr_cam import compute_bbox_info_bedlam

        cliff_cam = compute_bbox_info_bedlam(batch["bbx_xys"], batch["K_fullimg"])
        f_cam_angvel = batch.get("cam_angvel")
        if f_cam_angvel is not None and getattr(pipeline.args, "normalize_cam_angvel", False):
            f_cam_angvel = (f_cam_angvel - pipeline.cam_angvel_mean) / pipeline.cam_angvel_std.clamp(min=1e-8)
        f_imgseq = batch.get("f_imgseq")
        f_imgseq_mask = None
        if f_imgseq is not None and "mask" in batch and isinstance(batch["mask"], dict):
            f_imgseq_mask = batch["mask"].get("f_imgseq")

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            f_cond = pipeline.cond_embedder(
                obs=batch["obs"],
                f_cliffcam=cliff_cam,
                f_cam_angvel=f_cam_angvel,
                f_imgseq=f_imgseq,
                f_imgseq_mask=f_imgseq_mask,
                drop_obs=None,
            )
        length = batch["length"].long()
        seq_len = int(f_cond.shape[1])
        all_valid = bool((length == seq_len).all().item())
        return f_cond.float(), length, all_valid

    @torch.no_grad()
    def rollout_group(
        self,
        denoiser: torch.nn.Module,
        base_denoiser: torch.nn.Module,
        f_cond: torch.Tensor,
        length: torch.Tensor,
        group_size: int,
        all_valid_length: bool = False,
        generator: torch.Generator | None = None,
    ) -> DiffusionRolloutBatch:
        """Sample group_size stochastic DDIM trajectories per prompt and record transitions."""
        denoiser.eval()
        base_denoiser.eval()

        bsz, seq_len, _ = f_cond.shape
        f_cond_rep = f_cond.repeat_interleave(group_size, dim=0)
        length_rep = length.repeat_interleave(group_size, dim=0)
        total_b = bsz * group_size
        valid_mask = torch.arange(seq_len, device=length_rep.device).unsqueeze(0) < length_rep.unsqueeze(1)

        x = torch.randn(
            (total_b, seq_len, G1_PRED_X_DIM),
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        )

        x_t_list: list[torch.Tensor] = []
        x_prev_list: list[torch.Tensor] = []
        t_idx_list: list[int] = []
        old_logp_list: list[torch.Tensor] = []
        base_mean_list: list[torch.Tensor] = []

        for step_idx in reversed(range(self.ddim_steps)):
            if self.device.type == "cuda" and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            pred_x0 = self.predict_x0(
                denoiser, x, step_idx, f_cond_rep, length_rep, all_valid_length=all_valid_length
            )
            mean, std = self._ddim_mean_std(x, step_idx, pred_x0)

            if step_idx > 0:
                noise = torch.randn(
                    x.shape, device=self.device, dtype=torch.float32, generator=generator
                )
                x_prev = mean + std * noise
                old_logp = self.gaussian_log_prob(x_prev, mean, std, valid_mask)

                if self.device.type == "cuda" and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                base_pred_x0 = self.predict_x0(
                    base_denoiser, x, step_idx, f_cond_rep, length_rep, all_valid_length=all_valid_length
                )
                base_mean, _ = self._ddim_mean_std(x, step_idx, base_pred_x0)

                x_t_list.append(x.detach())
                x_prev_list.append(x_prev.detach())
                t_idx_list.append(step_idx)
                old_logp_list.append(old_logp.detach())
                base_mean_list.append(base_mean.detach())
                x = x_prev
            else:
                # Final deterministic step (t == 0, sigma == 0): excluded from stochastic density.
                x = mean

        return DiffusionRolloutBatch(
            x_t=torch.stack(x_t_list, dim=0),
            x_prev=torch.stack(x_prev_list, dim=0),
            t_indices=t_idx_list,
            old_log_probs=torch.stack(old_logp_list, dim=0),
            base_means=torch.stack(base_mean_list, dim=0),
            x_0=x.detach(),
            valid_mask=valid_mask,
        )

    def replay_step(
        self,
        denoiser: torch.nn.Module,
        rollout: DiffusionRolloutBatch,
        stoch_idx: int,
        f_cond_rep: torch.Tensor,
        length_rep: torch.Tensor,
        all_valid_length: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiably recompute log_prob and exact KL for one stored stochastic step."""
        step_idx = rollout.t_indices[stoch_idx]
        x_t = rollout.x_t[stoch_idx]
        x_prev = rollout.x_prev[stoch_idx]
        base_mean = rollout.base_means[stoch_idx]

        pred_x0 = self.predict_x0(
            denoiser, x_t, step_idx, f_cond_rep, length_rep, all_valid_length=all_valid_length
        )
        mean, std = self._ddim_mean_std(x_t, step_idx, pred_x0)
        new_logp = self.gaussian_log_prob(x_prev, mean, std, rollout.valid_mask)
        kl = self.analytic_gaussian_kl(
            mean, base_mean, std, rollout.valid_mask, dim_weights=self.kl_dim_weights
        )
        return new_logp, kl


@torch.no_grad()
def decode_x0_to_g1_zup(
    pipeline: Any,
    x_0: torch.Tensor,
    cam_angvel: torch.Tensor,
    length: torch.Tensor,
    root_pos_0_yup: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Decode normalized x_0 (B, L, 189) to Z-up G1 tracker fields matching tools/infer_video.py."""
    from hmr4d.model.offline.pipeline.offline_pipeline import get_smpl_params_w_Rt_v2
    from hmr4d.utils.geo.hmr_global import rollout_local_transl_vel
    from pytorch3d.transforms import (
        axis_angle_to_matrix,
        matrix_to_axis_angle,
        matrix_to_quaternion,
        rotation_6d_to_matrix,
    )

    device = x_0.device
    bsz = x_0.shape[0]
    decode_dict = pipeline.endecoder.decode_g1_new(x_0.float())

    if root_pos_0_yup is None:
        pelvis_h = float(getattr(getattr(pipeline.args, "robot", None), "default_pelvis_height", 0.793))
        root_pos_0_yup = torch.tensor([[0.0, pelvis_h, 0.0]], device=device, dtype=torch.float32).expand(bsz, 1, 3)
    else:
        root_pos_0_yup = root_pos_0_yup.to(device=device, dtype=torch.float32)

    go_c_rotmat = decode_dict["global_orient_rotmat"].float()
    local_transl_vel = decode_dict["local_transl_vel"].float()
    go_c_aa = matrix_to_axis_angle(go_c_rotmat)
    go_gv_aa = matrix_to_axis_angle(rotation_6d_to_matrix(decode_dict["global_orient_gv_r6d"].float()))
    params_w = get_smpl_params_w_Rt_v2(
        global_orient_gv=go_gv_aa,
        local_transl_vel=local_transl_vel,
        global_orient_c=go_c_aa,
        cam_angvel=cam_angvel.to(device=device, dtype=torch.float32),
        length=length.to(device=device),
    )
    pred_root_aa_w = params_w["global_orient"]
    pred_root_R_w = axis_angle_to_matrix(pred_root_aa_w)
    pred_root_quat_yup = matrix_to_quaternion(pred_root_R_w)
    pred_root_pos_yup = rollout_local_transl_vel(local_transl_vel, pred_root_aa_w, root_pos_0_yup)

    R_bp = decode_dict["body_pose_rotmat"].float()
    pred_dof_byd = torch.atan2(R_bp[..., 0, 2], R_bp[..., 0, 0])
    pred_body_yup, _ = pipeline.endecoder.fk_g1(
        pred_dof_byd,
        pred_root_pos_yup,
        pred_root_quat_yup,
        world_up="y",
    )

    # Convert internal Y-up -> G1 Z-up world (identical to tools/infer_video.py lines 323-332).
    M_pos = torch.tensor(YUP_TO_ZUP_M, device=device, dtype=torch.float32)
    root_pos_z = pred_root_pos_yup @ M_pos
    body_pos_z = pred_body_yup @ M_pos
    root_quat_z = matrix_to_quaternion(M_pos.transpose(-1, -2) @ pred_root_R_w)
    root_quat_z = F.normalize(root_quat_z, dim=-1)
    return {
        "joint_pos": pred_dof_byd,      # (B, L, 29) IsaacLab/BYD order
        "root_pos_w": root_pos_z,       # (B, L, 3) Z-up metres
        "root_quat_w": root_quat_z,     # (B, L, 4) wxyz
        "body_pos_w": body_pos_z,       # (B, L, 30, 3) Z-up metres
    }
