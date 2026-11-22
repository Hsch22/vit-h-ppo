from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import mujoco
    import onnxruntime  # noqa: F401
    from vith_tracker.runtime import DEFAULT_ANGLES, MUJOCO_TO_ISAACLAB
    from vith_tracker.tracker import InitialSupportGrounder, track_file

    HAS_TRACKER_RUNTIME = True
except ImportError:
    HAS_TRACKER_RUNTIME = False


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_ONNX = REPO_ROOT / "inputs/checkpoints/g1_tracker/model_step_020000_g1.onnx"
ROLLOUT_XML = REPO_ROOT / "g1_tracker/assets/g1/rollout/scene_29dof_mujoco.xml"
GROUNDING_XML = REPO_ROOT / "g1_tracker/assets/g1/grounding/mjcf/g1_29dof_rev_1_0.xml"


@unittest.skipUnless(HAS_TRACKER_RUNTIME, "Tracker runtime (mujoco/onnxruntime) not installed")
class TestParallelTracker(unittest.TestCase):
    def test_invalid_num_workers_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dummy = Path(tmp) / "missing.npz"
            with self.assertRaises(ValueError):
                track_file(
                    input_path=dummy,
                    output_path=Path(tmp) / "out.npz",
                    policy_onnx=dummy,
                    xml_path=dummy,
                    grounding_xml=dummy,
                    num_workers=0,
                )

    @unittest.skipUnless(POLICY_ONNX.is_file(), "Tracker ONNX checkpoint not downloaded")
    def test_parallel_matches_serial_bit_for_bit(self):
        grounder = InitialSupportGrounder(GROUNDING_XML)
        grounder.data.qpos[:] = grounder.model.qpos0
        grounder.data.qpos[7:] = DEFAULT_ANGLES
        mujoco.mj_forward(grounder.model, grounder.data)

        frames = 15
        samples = 2
        joint_base = DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB].astype(np.float32)
        joint_pos = np.stack(
            [
                np.repeat(joint_base[None], frames, axis=0),
                np.repeat((joint_base + 0.01)[None], frames, axis=0),
            ],
            axis=0,
        )
        root_pos = np.repeat(grounder.data.qpos[:3][None, None], frames, axis=1)
        root_pos = np.repeat(root_pos, samples, axis=0).astype(np.float32)
        root_quat = np.repeat(grounder.data.qpos[3:7][None, None], frames, axis=1)
        root_quat = np.repeat(root_quat, samples, axis=0).astype(np.float32)
        body_pos = np.repeat(grounder.data.xpos[grounder.body_ids][None, None], frames, axis=1)
        body_pos = np.repeat(body_pos, samples, axis=0).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inp = tmp_path / "batch_ref.npz"
            np.savez_compressed(
                inp,
                joint_pos=joint_pos,
                root_pos_w=root_pos,
                root_quat_w=root_quat,
                body_pos_w=body_pos,
                fps=np.asarray(30.0),
            )
            serial = track_file(
                input_path=inp,
                output_path=tmp_path / "serial.npz",
                policy_onnx=POLICY_ONNX,
                xml_path=ROLLOUT_XML,
                grounding_xml=GROUNDING_XML,
                input_fps=30.0,
                num_workers=1,
            )
            parallel = track_file(
                input_path=inp,
                output_path=tmp_path / "parallel.npz",
                policy_onnx=POLICY_ONNX,
                xml_path=ROLLOUT_XML,
                grounding_xml=GROUNDING_XML,
                input_fps=30.0,
                num_workers=2,
            )
            self.assertEqual(set(serial.keys()), set(parallel.keys()))
            for key in serial:
                np.testing.assert_array_equal(serial[key], parallel[key], err_msg=f"Mismatch in {key}")


if __name__ == "__main__":
    unittest.main()
