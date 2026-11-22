"""Video-conditioned PPO dataset loader (strict source-level train/val split)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler, Subset

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
    deterministic_windows: bool = False,
) -> Dataset:
    """Build ConcatDataset of video-conditioned G1 sources with strict file validation.

    When `deterministic_windows=True`, passes `split='val'` and `subset_split=split`
    so segment indexing matches `post_training/mine_hard_clips.py` (`global_idx`).
    """
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
                "split": "val" if deterministic_windows else split,
                "subset_split": split if deterministic_windows else None,
                "split_seed": int(split_seed),
                "split_ratios": split_ratios,
            }
            kwargs.update(spec.get("extra", {}))
            parts.append(PPOVithClipWrapper(spec["cls"](**kwargs)))
    if not parts:
        raise ValueError("No dataset parts loaded for PPO")
    return ConcatDataset(parts)


class HardClipDistributedSampler(Sampler[int]):
    """Distributed bucket-weighted sampler driven by `mine_hard_clips.py` `hard_index.json`.

    Strictly excludes `untrackable_gt` clips (weight = 0) and mixes Bucket A
    (fixable Base failures), Bucket B (high execution error), and Bucket C
    (easy pass) according to explicit target probabilities.
    """

    def __init__(
        self,
        hard_index_path: Path | str,
        dataset_len: int,
        num_replicas: int = 1,
        rank: int = 0,
        share_a: float = 0.20,
        share_b: float = 0.55,
        share_c: float = 0.25,
        num_samples_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        if share_a < 0 or share_b < 0 or share_c < 0 or (share_a + share_b + share_c) <= 0:
            raise ValueError(f"Invalid bucket shares: {(share_a, share_b, share_c)}")
        data = json.loads(Path(hard_index_path).read_text(encoding="utf-8"))
        buckets = data["buckets"]

        bucket_specs = [
            ("bucket_a_fixable_fail", float(share_a)),
            ("bucket_b_high_exec_err", float(share_b)),
            ("bucket_c_easy_pass", float(share_c)),
        ]
        indices: list[int] = []
        weights: list[float] = []
        self.bucket_counts: dict[str, int] = {}
        for name, share in bucket_specs:
            entries = buckets.get(name, [])
            valid_ids = [int(e["global_idx"]) for e in entries if 0 <= int(e["global_idx"]) < dataset_len]
            self.bucket_counts[name] = len(valid_ids)
            if valid_ids and share > 0.0:
                per_item = share / len(valid_ids)
                indices.extend(valid_ids)
                weights.extend([per_item] * len(valid_ids))
        self.excluded_untrackable = len(buckets.get("untrackable_gt", []))
        if not indices:
            raise ValueError(f"No valid training clips found in hard index {hard_index_path}")

        w_arr = np.asarray(weights, dtype=np.float64)
        self._indices = np.asarray(indices, dtype=np.int64)
        self._probs = w_arr / w_arr.sum()
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        total_epoch = int(num_samples_per_epoch) if num_samples_per_epoch is not None else len(indices)
        # Round up to exact multiple of num_replicas.
        self.num_samples = int(np.ceil(total_epoch / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch * 7919)
        drawn = rng.choice(self._indices, size=self.total_size, replace=True, p=self._probs)
        rank_slice = drawn[self.rank :: self.num_replicas]
        return iter(int(i) for i in rank_slice)

    def __len__(self) -> int:
        return self.num_samples


def build_hard_val_subset(
    val_ds: Dataset,
    val_hard_index_path: Path | str,
    max_bucket_b: int = 107,
    seed: int = 42,
) -> tuple[Subset, np.ndarray]:
    """Select 100% of val Bucket A + stratified val Bucket B clips for hard evaluation."""
    data = json.loads(Path(val_hard_index_path).read_text(encoding="utf-8"))
    buckets = data["buckets"]
    a_ids = [int(e["global_idx"]) for e in buckets.get("bucket_a_fixable_fail", [])]
    b_ids = [int(e["global_idx"]) for e in buckets.get("bucket_b_high_exec_err", [])]
    if max_bucket_b > 0 and len(b_ids) > max_bucket_b:
        rng = np.random.default_rng(seed)
        pick = np.sort(rng.choice(len(b_ids), size=max_bucket_b, replace=False))
        b_ids = [b_ids[int(i)] for i in pick]
    combined = a_ids + b_ids
    is_bucket_a = np.array([True] * len(a_ids) + [False] * len(b_ids), dtype=np.bool_)
    return Subset(val_ds, combined), is_bucket_a
