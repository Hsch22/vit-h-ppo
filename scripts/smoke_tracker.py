#!/usr/bin/env python3
"""Exercise the real tracker on a synthetic two-second standing reference.

This checks tracker installation and output integrity, not video inference,
motion diversity, benchmark performance, or PPO training.
"""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "g1_tracker"))
from vith_tracker.runtime import DEFAULT_ANGLES, MUJOCO_TO_ISAACLAB
from vith_tracker.tracker import InitialSupportGrounder, track_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    # A fresh directory prevents overwriting either a real experiment or a
    # previous smoke result.
    output.mkdir(parents=True, exist_ok=False)
    grounding = ROOT / "g1_tracker/assets/g1/grounding/mjcf/g1_29dof_rev_1_0.xml"
    grounder = InitialSupportGrounder(grounding)
    grounder.data.qpos[:] = grounder.model.qpos0
    grounder.data.qpos[7:] = DEFAULT_ANGLES
    mujoco.mj_forward(grounder.model, grounder.data)
    frames = 61  # endpoints t=0 through t=2 seconds at 30 FPS
    repeated = lambda value: np.repeat(np.asarray(value)[None], frames, axis=0).astype(np.float32)
    reference = output / "synthetic_reference.npz"
    np.savez_compressed(
        reference,
        joint_pos=repeated(DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB]),
        root_pos_w=repeated(grounder.data.qpos[:3]),
        root_quat_w=repeated(grounder.data.qpos[3:7]),
        body_pos_w=repeated(grounder.data.xpos[grounder.body_ids]),
        fps=np.asarray(30.0),
    )
    destination = output / "tracked.npz"
    track_file(
        input_path=reference,
        output_path=destination,
        policy_onnx=ROOT / "inputs/checkpoints/g1_tracker/model_step_020000_g1.onnx",
        xml_path=ROOT / "g1_tracker/assets/g1/rollout/scene_29dof_mujoco.xml",
        grounding_xml=grounding,
        input_fps=30.0,
    )
    expected = {
        "joint_pos": (1, frames, 29),
        "root_pos_w": (1, frames, 3),
        "root_quat_w": (1, frames, 4),
        "body_pos_w": (1, frames, 30, 3),
    }
    with np.load(destination, allow_pickle=False) as result:
        for key, shape in expected.items():
            if result[key].shape != shape or not np.isfinite(result[key]).all():
                raise RuntimeError(f"Invalid saved tracker output: {key}")
        if int(result["tracker_protocol_version"]) != 4:
            raise RuntimeError("Unexpected tracker protocol")
        summary = {
            "input_kind": "synthetic_standing_not_video",
            "frames": frames,
            "fps": 30,
            "tracker_protocol_version": 4,
            "is_success": result["is_success"].tolist(),
            "first_failure_frame_50hz": result["first_failure_frame_50hz"].tolist(),
            "initial_joint_clip_count": result["initial_joint_clip_count"].tolist(),
            "output_arrays_finite": True,
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
