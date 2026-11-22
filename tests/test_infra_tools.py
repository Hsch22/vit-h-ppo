import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import checkpoints, preflight


class TestCheckpointEndpoint(unittest.TestCase):
    def test_default_hf_endpoint(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_ENDPOINT", None)
            self.assertEqual(checkpoints.resolve_hf_endpoint(), "https://huggingface.co")

    def test_env_and_explicit_hf_endpoint(self):
        with mock.patch.dict(os.environ, {"HF_ENDPOINT": "https://hf-mirror.com/"}, clear=False):
            self.assertEqual(checkpoints.resolve_hf_endpoint(), "https://hf-mirror.com")
            self.assertEqual(
                checkpoints.resolve_hf_endpoint("https://mirror.example.org///"),
                "https://mirror.example.org",
            )

    def test_invalid_hf_endpoint_rejected(self):
        for bad in ("", "ftp://example.com", "hf-mirror.com", "https://"):
            with self.assertRaises(ValueError):
                checkpoints.resolve_hf_endpoint(bad)

    def test_invalid_retries_rejected(self):
        manifest = checkpoints.load_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                checkpoints.download(Path(tmp), manifest, manifest["artifacts"][0], retries=0)


class TestPreflight(unittest.TestCase):
    def test_preflight_reports_missing_checkpoints_in_empty_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = preflight.run_preflight(
                root=Path(tmp),
                mode="tracker",
                require_gpu=False,
                verify_hash=False,
                min_free_gb=0.0,
                output_root=Path(tmp),
            )
            self.assertFalse(report["ok"])
            self.assertTrue(any("Missing checkpoint (tracker)" in err for err in report["errors"]))

    def test_preflight_catches_low_disk_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = preflight.run_preflight(
                root=Path(tmp),
                mode="tracker",
                require_gpu=False,
                verify_hash=False,
                min_free_gb=1e9,
                output_root=Path(tmp),
            )
            self.assertFalse(report["ok"])
            self.assertTrue(any("Low free space" in err for err in report["errors"]))


class TestTrainingSetG1FilenameConfig(unittest.TestCase):
    def test_vith_g1_pth_name_env_override(self):
        try:
            from omegaconf import OmegaConf
        except ImportError:
            self.skipTest("omegaconf not installed")
        cfg_path = Path(__file__).resolve().parents[1] / "hmr4d/configs/train_datasets/training_set.yaml"
        with mock.patch.dict(os.environ, {"VITH_DATASET_ROOT": "/data/vith"}, clear=False):
            os.environ.pop("VITH_G1_PTH_NAME", None)
            cfg = OmegaConf.load(cfg_path)
            self.assertEqual(cfg.amass_00.g1_pth_path, "/data/vith/amass/part_00/g1.pth")
            self.assertEqual(cfg.bedlam2_00.g1_pth_name, "g1.pth")
            self.assertEqual(cfg.h36m_01.g1_pth_name, "g1.pth")

        with mock.patch.dict(
            os.environ,
            {"VITH_DATASET_ROOT": "/data/vith", "VITH_G1_PTH_NAME": "g1_new.pth"},
            clear=False,
        ):
            cfg = OmegaConf.load(cfg_path)
            self.assertEqual(cfg.amass_00.g1_pth_path, "/data/vith/amass/part_00/g1_new.pth")
            self.assertEqual(cfg.amass_01.g1_pth_path, "/data/vith/amass/part_01/g1_new.pth")
            self.assertEqual(cfg.bedlam2_00.g1_pth_name, "g1_new.pth")
            self.assertEqual(cfg.h36m_01.g1_pth_name, "g1_new.pth")


if __name__ == "__main__":
    unittest.main()
