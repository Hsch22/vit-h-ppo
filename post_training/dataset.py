"""Video-conditioned PPO dataset loader (strict source-level train/val split)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, Dataset

from hmr4d.dataset.bedlam.bedlam import Bedlam2G1Dataset, BedlamG1Dataset
from hmr4d.dataset.h36m.h36m import H36mG1Dataset
from hmr4d.model.offline.utils.obs_joints import select_coco17_no_nose_ears
from hmr4d.utils.geo.hmr_cam import normalize_kp2d
from post_training.policy import YUP_TO_ZUP_M


VIDEO_SOURCE_SPECS: dict[str, list[dict[str, Any]]] = {
    "h36m": [
        {
            "cls": H36mG1Dataset,
            "subdir": "h36m/part_00",
            "bbox": ["bbox.pth"],
            "kp2d": ["keypoints.pth"],
            "imgfeat": ["image_features.pth"],
        },
        {
            "cls": H36mG1Dataset,
            "subdir": "h36m/part_01",
            "bbox": ["bbox.pth"],
            "kp2d": ["keypoints.pth"],
            "imgfeat": ["image_features.pth"],
        },
    ],
    "bedlam": [
        {
            "cls": BedlamG1Dataset,
            "subdir": "bedlam/part_00",
            "bbox": ["bbox_00.pth", "bbox_01.pth"],
            "kp2d": ["keypoints_00.pth", "keypoints_01.pth"],
            "imgfeat": ["image_features_00.pth", "image_features_01.pth"],
            "extra": {"camera_source": "bedlam_smpl"},
        },
        {
            "cls": BedlamG1Dataset,
            "subdir": "bedlam/part_01",
            "bbox": ["bbox.pth"],
            "kp2d": ["keypoints.pth"],
            "imgfeat": ["image_features.pth"],
            "extra": {"camera_source": "bedlam_smpl"},
        },
    ],
    "bedlam2": [
        {
            "cls": Bedlam2G1Dataset,
            "subdir": "bedlam2/part_00",
            "bbox": ["bbox_00.pth", "bbox_01.pth"],
            "kp2d": ["keypoints_00.pth", "keypoints_01.pth"],
            "imgfeat": ["image_features_00.pth", "image_features_01.pth"],
        },
        {
            "cls": Bedlam2G1Dataset,
            "subdir": "bedlam2/part_01",
            "bbox": ["bbox.pth"],
            "kp2d": ["keypoints.pth"],
            "imgfeat": ["image_features.pth"],
        },
        {
            "cls": Bedlam2G1Dataset,
            "subdir": "bedlam2/part_02",
            "bbox": ["bbox.pth"],
            "kp2d": ["keypoints.pth"],
            "imgfeat": ["image_features.pth"],
        },
    ],
}


class PPOVithClipWrapper(Dataset):
    """Wrap a video-conditioned G1 dataset into compact PPO rollout tensors."""

    def __init__(self, inner: Dataset) -> None:
        self.inner = inner
        self._m_pos = torch.tensor(YUP_TO_ZUP_M, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.inner[idx]
        obs = select_coco17_no_nose_ears(normalize_kp2d(sample["kp2d"], sample["bbx_xys"])).float()
        valid = torch.as_tensor(sample["mask"]["valid"], dtype=torch.bool)
        obs[~valid] = 0.0
        g1 = sample["g1_target"]
        gt_body_yup = g1["g1_body_pos_w"].float()
        gt_body_zup = gt_body_yup @ self._m_pos
        root_pos_0_yup = gt_body_yup[0:1, 0, :].clone()
        return {
            "length": torch.as_tensor(sample["length"], dtype=torch.long),
            "obs": obs,
            "bbx_xys": sample["bbx_xys"].float(),
            "K_fullimg": sample["K_fullimg"].float(),
            "cam_angvel": sample["cam_angvel"].float(),
            "f_imgseq": sample["f_imgseq"].float(),
            "f_imgseq_mask": torch.as_tensor(sample["mask"]["f_imgseq"], dtype=torch.bool),
            "root_pos_0_yup": root_pos_0_yup,
            "gt_joint_pos": g1["g1_joint_pos"].float(),
            "gt_body_pos_z": gt_body_zup,
            "fps": sample["fps"].float(),
        }


def build_video_ppo_dataset(
    dataset_root: Path | str,
    sources: list[str],
    split: str,
    g1_pth_name: str = "g1_new.pth",
    motion_frames: int = 120,
    split_seed: int = 42,
    split_ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
) -> Dataset:
    """Build ConcatDataset of video-conditioned G1 sources with strict file validation."""
    root = Path(dataset_root).expanduser().resolve()
    parts: list[Dataset] = []
    for source in sources:
        if source not in VIDEO_SOURCE_SPECS:
            raise ValueError(
                f"Unsupported video PPO source {source!r}; choose from {sorted(VIDEO_SOURCE_SPECS)}"
            )
        for spec in VIDEO_SOURCE_SPECS[source]:
            data_dir = root / spec["subdir"]
            g1_file = data_dir / g1_pth_name
            if not g1_file.is_file():
                raise FileNotFoundError(
                    f"Missing required G1 trajectory file: {g1_file} "
                    f"(source={source}, g1_pth_name={g1_pth_name!r})"
                )
            kwargs: dict[str, Any] = {
                "data_dir": str(data_dir),
                "smplx_pth_name": "motion.pth",
                "g1_pth_name": g1_pth_name,
                "bbox_pth_paths": [str(data_dir / p) for p in spec["bbox"]],
                "vitpose_pth_paths": [str(data_dir / p) for p in spec["kp2d"]],
                "imgfeat_pth_paths": [str(data_dir / p) for p in spec["imgfeat"]],
                "motion_frames": int(motion_frames),
                "min_motion_frames": 60,
                "g1_input_is_yup": False,
                "dof_pos_order": "mjc",
                "canonicalize_first_frame": True,
                "floor_adjust": True,
                "yaw_only_canon": True,
                "split": split,
                "split_seed": int(split_seed),
                "split_ratios": split_ratios,
            }
            kwargs.update(spec.get("extra", {}))
            parts.append(PPOVithClipWrapper(spec["cls"](**kwargs)))
    if not parts:
        raise ValueError("No dataset parts loaded for PPO")
    return ConcatDataset(parts)
