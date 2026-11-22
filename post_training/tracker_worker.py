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

from vith_tracker.tracker import (
    DeployPolicy,
    DeployRolloutRunner,
    InitialSupportGrounder,
    TRACKER_PROTOCOL_VERSION,
    resample_motion,
    resample_rollout,
)


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
                ref_jp = values["joint_pos"][0]
                ref_rp_grounded = values["root_pos_w"][0].copy()
                ref_rp_grounded[:, 2] += float(offset)

                finite_out = bool(np.isfinite(exec_jp).all() and np.isfinite(exec_rp).all())
                if not finite_out:
                    raise RuntimeError("Non-finite tracker rollout output")
                total_50hz = int(reference.joint_pos.shape[0])
                resp = {
                    "valid": True,
                    "is_success": bool(is_success),
                    "first_failure_frame_50hz": -1 if first_failure is None else int(first_failure),
                    "total_frames_50hz": total_50hz,
                    "grounding_offset_m": float(offset),
                    "initial_joint_clip_count": int((np.abs(clip_delta) > 1e-8).sum()),
                    "exec_joint_mae_rad": float(np.mean(np.abs(exec_jp - ref_jp))),
                    "exec_root_pos_err_m": float(np.mean(np.linalg.norm(exec_rp - ref_rp_grounded, axis=-1))),
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
                    "tracker_protocol_version": int(TRACKER_PROTOCOL_VERSION),
                }
            _send_msg(stdout, resp)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
