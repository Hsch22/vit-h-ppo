"""Exact batched two-window rigid stitching matching `tools/infer_video.py`."""

from __future__ import annotations

import torch

from motion_lib.motion_utils.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)


@torch.no_grad()
def stitch_two_chunks_yup(
    jp0: torch.Tensor,
    rp0: torch.Tensor,
    rq0: torch.Tensor,
    bp0: torch.Tensor,
    jp1: torch.Tensor,
    rp1: torch.Tensor,
    rq1: torch.Tensor,
    bp1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Batched exact rigid alignment matching `tools/infer_video.py::align_window_to_previous`.

    Args:
        jp0, jp1: (N, W, 29) G1 joint angles for chunk 0 (`[0:W]`) and chunk 1 (`[W-1:2W-1]`).
        rp0, rp1: (N, W, 3) Y-up root positions.
        rq0, rq1: (N, W, 4) Y-up wxyz root quaternions.
        bp0, bp1: (N, W, J, 3) Y-up body link positions.

    Returns:
        Stitched `(jp, rp, rq, bp)` of length `2*W - 1` plus `seam_info` dict with
        `overlap_err`, `seam_jump`, and `interior_step` (each `(N,)` in radians).
    """
    prev_pos = rp0[:, -1]       # (N, 3)
    prev_quat = rq0[:, -1]      # (N, 4)
    curr_origin = rp1[:, 0]     # (N, 3)
    curr_quat = rq1[:, 0]       # (N, 4)

    R_prev = quaternion_to_matrix(prev_quat)
    R_curr = quaternion_to_matrix(curr_quat)
    delta_R = R_prev @ R_curr.transpose(-1, -2)  # (N, 3, 3)

    aligned_rp1 = torch.einsum("nlj,nij->nli", rp1 - curr_origin[:, None, :], delta_R) + prev_pos[:, None, :]
    aligned_bp1 = (
        torch.einsum("nlkj,nij->nlki", bp1 - curr_origin[:, None, None, :], delta_R)
        + prev_pos[:, None, None, :]
    )
    aligned_rq1 = matrix_to_quaternion(delta_R[:, None] @ quaternion_to_matrix(rq1))

    jp_stitched = torch.cat([jp0, jp1[:, 1:]], dim=1)          # (N, 2W-1, 29)
    rp_stitched = torch.cat([rp0, aligned_rp1[:, 1:]], dim=1)  # (N, 2W-1, 3)
    rq_stitched = torch.cat([rq0, aligned_rq1[:, 1:]], dim=1)  # (N, 2W-1, 4)
    bp_stitched = torch.cat([bp0, aligned_bp1[:, 1:]], dim=1)  # (N, 2W-1, J, 3)

    # Exact boundary discontinuity metrics (radians):
    overlap_err = (jp1[:, 0] - jp0[:, -1]).abs().mean(dim=-1)  # (N,) same video frame W-1
    seam_jump = (jp1[:, 1] - jp0[:, -1]).abs().mean(dim=-1)    # (N,) stitched step W-1 -> W
    all_steps = (jp_stitched[:, 1:] - jp_stitched[:, :-1]).abs().mean(dim=-1)  # (N, 2W-2)
    seam_idx = jp0.shape[1] - 1
    interior_steps = torch.cat([all_steps[:, :seam_idx], all_steps[:, seam_idx + 1 :]], dim=1)
    interior_step = interior_steps.mean(dim=-1)

    seam_info = {
        "overlap_err": overlap_err,
        "seam_jump": seam_jump,
        "interior_step": interior_step,
    }
    return jp_stitched, rp_stitched, rq_stitched, bp_stitched, seam_info
