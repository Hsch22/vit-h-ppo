#!/usr/bin/env python3
"""Persistent SONIC/MuJoCo tracker worker process (runs inside vith-tracker env).

Loads DeployPolicy, DeployRolloutRunner, and InitialSupportGrounder ONCE at
startup, then processes length-prefixed binary pickle requests over stdin/stdout
with zero disk I/O and zero per-step process spawn overhead.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import struct
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "g1_tracker"))

from vith_tracker.success import evaluate_sonic_standardized_frame
from vith_tracker.tracker import (
    BODY_NAMES,
    DeployPolicy,
    DeployRolloutRunner,
    END_EFFECTOR_INDICES,
    FOOT_BODY_NAMES,
    InitialSupportGrounder,
    TRACKER_PROTOCOL_VERSION,
    resample_motion,
    resample_rollout,
)

FOOT_BODY_INDICES = tuple(BODY_NAMES.index(name) for name in FOOT_BODY_NAMES)
FOOT_CONTACT_HEIGHT_THRESHOLD_M = 0.08


def _compute_exec_foot_slip_cm_s(
    body_pos_50hz: np.ndarray,
    control_hz: float,
    end_frame: int | None = None,
    height_threshold_m: float = FOOT_CONTACT_HEIGHT_THRESHOLD_M,
) -> float:
    """Mean horizontal ankle speed (cm/s) over grounded stance frames at 50 Hz."""
    bp = body_pos_50hz if end_frame is None else body_pos_50hz[: max(2, int(end_frame))]
    if bp.shape[0] < 2:
        return 0.0
    feet = bp[:, list(FOOT_BODY_INDICES), :]  # (T, 2, 3)
    in_contact = (feet[:-1, :, 2] < height_threshold_m) & (feet[1:, :, 2] < height_threshold_m)
    if not bool(in_contact.any()):
        return 0.0
    vel_xy_cm_s = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=-1) * float(control_hz) * 100.0
    return float(np.mean(vel_xy_cm_s[in_contact]))


def _read_exact(stream, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = stream.read(size - len(buf))
        if not chunk:
            return b""
        buf.extend(chunk)
    return bytes(buf)


def _send_msg(stream, obj: object) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("!I", len(payload)))
    stream.write(payload)
    stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-onnx", type=Path, required=True)
    parser.add_argument("--xml-path", type=Path, required=True)
    parser.add_argument("--grounding-xml", type=Path, required=True)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--ort-threads", type=int, default=1)
    args = parser.parse_args()

    policy = DeployPolicy(args.policy_onnx.resolve(), args.ort_threads)
    runner = DeployRolloutRunner(args.xml_path.resolve(), policy, args.control_hz)
    grounder = InitialSupportGrounder(args.grounding_xml.resolve())

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    # Signal readiness to parent pool.
    _send_msg(stdout, {"ready": True, "tracker_protocol_version": int(TRACKER_PROTOCOL_VERSION)})

    try:
        while True:
            header = _read_exact(stdin, 4)
            if not header:
                break
            length = struct.unpack("!I", header)[0]
            if length == 0:
                break
            raw = _read_exact(stdin, length)
            if len(raw) != length:
                break
            req = pickle.loads(raw)
            fps = float(req["fps"])
            values = {
                key: np.asarray(req[key], dtype=np.float32)[None]
                for key in ("joint_pos", "root_pos_w", "root_quat_w", "body_pos_w")
            }
            frames = int(values["joint_pos"].shape[1])
            try:
                if frames < 3 or any(not np.isfinite(v).all() for v in values.values()):
                    raise ValueError("Non-finite or too-short motion reference")
                reference = resample_motion(values, fps, args.control_hz, 0)
                reference, offset = grounder.ground(reference)
                initial_joint_pos, clip_delta = grounder.clip_initial_joint_pos(reference.joint_pos[0])
                rollout_50, is_success, first_failure = runner.rollout(reference, initial_joint_pos)
                sampled = resample_rollout(rollout_50, frames, args.control_hz, fps)

                exec_jp = sampled["joint_pos"][0]
                exec_rp = sampled["root_pos_w"][0]
                exec_bp = sampled["body_pos_w"][0]
                ref_jp = values["joint_pos"][0]
                ref_rp_grounded = values["root_pos_w"][0].copy()
                ref_rp_grounded[:, 2] += float(offset)
                ref_bp_grounded = values["body_pos_w"][0].copy()
                ref_bp_grounded[:, :, 2] += float(offset)

                finite_out = bool(np.isfinite(exec_jp).all() and np.isfinite(exec_rp).all() and np.isfinite(exec_bp).all())
                if not finite_out:
                    raise RuntimeError("Non-finite tracker rollout output")
                total_50hz = int(reference.joint_pos.shape[0])

                # Exact failure decomposition at first_failure (when failed) and physical foot slip.
                failed_anchor_height = False
                failed_anchor_orientation = False
                failed_end_effector_height = False
                fail_anchor_height_err_m = 0.0
                fail_ee_height_err_m = 0.0
                fail_anchor_ori_err_rad = 0.0
                if first_failure is not None and 0 <= int(first_failure) < total_50hz:
                    ff = int(first_failure)
                    f_res = evaluate_sonic_standardized_frame(
                        reference_anchor_position_w=reference.body_pos[ff, 0],
                        robot_anchor_position_w=rollout_50["body_pos_w"][ff, 0],
                        reference_anchor_quaternion_wxyz=reference.root_quat[ff],
                        robot_anchor_quaternion_wxyz=rollout_50["root_quat_w"][ff],
                        reference_end_effector_positions_w=reference.body_pos[ff, list(END_EFFECTOR_INDICES)],
                        robot_end_effector_positions_w=rollout_50["body_pos_w"][ff, list(END_EFFECTOR_INDICES)],
                    )
                    failed_anchor_height = bool(f_res.failed_anchor_height)
                    failed_anchor_orientation = bool(f_res.failed_anchor_orientation)
                    failed_end_effector_height = bool(f_res.failed_end_effector_height)
                    fail_anchor_height_err_m = float(f_res.anchor_height_error_m)
                    fail_ee_height_err_m = float(np.max(f_res.end_effector_height_errors_m))
                    fail_anchor_ori_err_rad = float(f_res.anchor_orientation_error_rad)

                slip_end = None if first_failure is None else int(first_failure) + 1
                exec_foot_slip = _compute_exec_foot_slip_cm_s(
                    rollout_50["body_pos_w"], args.control_hz, end_frame=slip_end
                )
                exec_root_z_err = float(np.mean(np.abs(exec_rp[:, 2] - ref_rp_grounded[:, 2])))
                exec_body_mpjpe = float(np.mean(np.linalg.norm(exec_bp - ref_bp_grounded, axis=-1)))

                resp = {
                    "valid": True,
                    "is_success": bool(is_success),
                    "first_failure_frame_50hz": -1 if first_failure is None else int(first_failure),
                    "total_frames_50hz": total_50hz,
                    "grounding_offset_m": float(offset),
                    "initial_joint_clip_count": int((np.abs(clip_delta) > 1e-8).sum()),
                    "exec_joint_mae_rad": float(np.mean(np.abs(exec_jp - ref_jp))),
                    "exec_root_pos_err_m": float(np.mean(np.linalg.norm(exec_rp - ref_rp_grounded, axis=-1))),
                    "exec_root_z_err_m": exec_root_z_err,
                    "exec_body_mpjpe_m": exec_body_mpjpe,
                    "exec_foot_slip_cm_s": exec_foot_slip,
                    "failed_anchor_height": failed_anchor_height,
                    "failed_anchor_orientation": failed_anchor_orientation,
                    "failed_end_effector_height": failed_end_effector_height,
                    "fail_anchor_height_err_m": fail_anchor_height_err_m,
                    "fail_ee_height_err_m": fail_ee_height_err_m,
                    "fail_anchor_ori_err_rad": fail_anchor_ori_err_rad,
                    "tracker_protocol_version": int(TRACKER_PROTOCOL_VERSION),
                }
            except Exception as exc:
                resp = {
                    "valid": False,
                    "error": str(exc),
                    "is_success": False,
                    "first_failure_frame_50hz": 0,
                    "total_frames_50hz": max(1, int(round((frames - 1) * args.control_hz / max(fps, 1e-3))) + 1),
                    "grounding_offset_m": 0.0,
                    "initial_joint_clip_count": 0,
                    "exec_joint_mae_rad": float("nan"),
                    "exec_root_pos_err_m": float("nan"),
                    "exec_root_z_err_m": float("nan"),
                    "exec_body_mpjpe_m": float("nan"),
                    "exec_foot_slip_cm_s": float("nan"),
                    "failed_anchor_height": False,
                    "failed_anchor_orientation": False,
                    "failed_end_effector_height": False,
                    "fail_anchor_height_err_m": float("nan"),
                    "fail_ee_height_err_m": float("nan"),
                    "fail_anchor_ori_err_rad": float("nan"),
                    "tracker_protocol_version": int(TRACKER_PROTOCOL_VERSION),
                }
            _send_msg(stdout, resp)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
