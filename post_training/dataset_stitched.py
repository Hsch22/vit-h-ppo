"""Multi-window (2 x 120f, 1-frame overlap = 239f) dataset & weighted sampler for stitched PPO."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler

from hmr4d.model.offline.utils.obs_joints import select_coco17_no_nose_ears
from hmr4d.utils.geo.hmr_cam import normalize_kp2d
from post_training.dataset import VIDEO_SOURCE_SPECS
from post_training.policy import YUP_TO_ZUP_M


STITCHED_WINDOW = 120
STITCHED_TOTAL_LEN = 2 * STITCHED_WINDOW - 1  # 239 frames (1-frame overlap at index 119)


class StitchedTwoWindowWrapper(Dataset):
    """Split a 239-frame continuous clip into two 120-frame overlapping windows + 239f GT."""

    def __init__(self, inner: Dataset, source: str, window: int = STITCHED_WINDOW) -> None:
        self.inner = inner
        self.source = source
        self.window = int(window)
        self.total_len = 2 * self.window - 1
        self._m_pos = torch.tensor(YUP_TO_ZUP_M, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.inner)

    def clip_key_at(self, idx: int) -> str:
        return str(self.inner.idx2meta[idx]["key"])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.inner[idx]
        w = self.window
        t_tot = self.total_len
        obs = select_coco17_no_nose_ears(normalize_kp2d(sample["kp2d"], sample["bbx_xys"])).float()
        valid = torch.as_tensor(sample["mask"]["valid"], dtype=torch.bool)
        obs[~valid] = 0.0

        bbx = sample["bbx_xys"].float()
        K = sample["K_fullimg"].float()
        ang = sample["cam_angvel"].float()
        feat = sample["f_imgseq"].float()
        feat_mask = torch.as_tensor(sample["mask"]["f_imgseq"], dtype=torch.bool)

        g1 = sample["g1_target"]
        gt_body_yup = g1["g1_body_pos_w"][:t_tot].float()
        gt_body_zup = gt_body_yup @ self._m_pos
        root_pos_0_yup = gt_body_yup[0:1, 0, :].clone()

        s0 = slice(0, w)
        s1 = slice(w - 1, t_tot)  # exact 1-frame overlap at frame w - 1 (119)
        return {
            "obs_c0": obs[s0],
            "obs_c1": obs[s1],
            "bbx_c0": bbx[s0],
            "bbx_c1": bbx[s1],
            "K_c0": K[s0],
            "K_c1": K[s1],
            "ang_c0": ang[s0],
            "ang_c1": ang[s1],
            "feat_c0": feat[s0],
            "feat_c1": feat[s1],
            "f_imgseq_mask": feat_mask,
            "root_pos_0_yup": root_pos_0_yup,
            "gt_joint_pos": g1["g1_joint_pos"][:t_tot].float(),
            "gt_body_pos_z": gt_body_zup,
            "fps": sample["fps"].float(),
        }


def load_key_buckets_from_mining_clips(
    clips_jsonl_path: Path | str,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Return (untrackable_keys, hard_ab_keys) as sets of (source, clip_key)."""
    key_buckets: dict[tuple[str, str], set[str]] = defaultdict(set)
    with Path(clips_jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            key_buckets[(str(r["source"]), str(r["clip_key"]))].add(str(r["bucket"]))
    untrackable: set[tuple[str, str]] = set()
    hard_ab: set[tuple[str, str]] = set()
    for sk, bset in key_buckets.items():
        if "untrackable_gt" in bset:
            untrackable.add(sk)
        elif "bucket_a_fixable_fail" in bset or "bucket_b_high_exec_err" in bset:
            hard_ab.add(sk)
    return untrackable, hard_ab


def build_stitched_ppo_dataset(
    dataset_root: Path | str,
    sources: list[str],
    split: str,
    g1_pth_name: str = "g1_new.pth",
    split_seed: int = 42,
    split_ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
) -> tuple[ConcatDataset, list[tuple[str, str]]]:
    """Build ConcatDataset of 239-frame 2-window pairs plus parallel (source, clip_key) metadata."""
    root = Path(dataset_root).expanduser().resolve()
    parts: list[StitchedTwoWindowWrapper] = []
    meta_keys: list[tuple[str, str]] = []
    for source in sources:
        if source not in VIDEO_SOURCE_SPECS:
            raise ValueError(f"Unsupported video PPO source {source!r}")
        for spec in VIDEO_SOURCE_SPECS[source]:
            data_dir = root / spec["subdir"]
            g1_file = data_dir / g1_pth_name
            if not g1_file.is_file():
                raise FileNotFoundError(f"Missing required G1 file: {g1_file}")
            kwargs: dict[str, Any] = {
                "data_dir": str(data_dir),
                "smplx_pth_name": "motion.pth",
                "g1_pth_name": g1_pth_name,
                "bbox_pth_paths": [str(data_dir / p) for p in spec["bbox"]],
                "vitpose_pth_paths": [str(data_dir / p) for p in spec["kp2d"]],
                "imgfeat_pth_paths": [str(data_dir / p) for p in spec["imgfeat"]],
                "motion_frames": STITCHED_TOTAL_LEN,
                "min_motion_frames": STITCHED_TOTAL_LEN,
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
            inner = spec["cls"](**kwargs)
            if len(inner) == 0:
                continue
            wrapped = StitchedTwoWindowWrapper(inner, source=source)
            parts.append(wrapped)
            for i in range(len(wrapped)):
                meta_keys.append((source, wrapped.clip_key_at(i)))
    if not parts:
        raise ValueError(f"No >= {STITCHED_TOTAL_LEN}-frame sequences found for sources={sources} split={split}")
    return ConcatDataset(parts), meta_keys


class StitchedKeyWeightedDistributedSampler(Sampler[int]):
    """Exclude `untrackable_gt` keys and upweight hard (Bucket A/B) long sequences across DDP ranks."""

    def __init__(
        self,
        meta_keys: list[tuple[str, str]],
        untrackable_keys: set[tuple[str, str]],
        hard_ab_keys: set[tuple[str, str]],
        num_replicas: int = 1,
        rank: int = 0,
        hard_weight_mult: float = 2.5,
        num_samples_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        indices: list[int] = []
        weights: list[float] = []
        n_hard = 0
        n_easy = 0
        n_excl = 0
        for idx, sk in enumerate(meta_keys):
            if sk in untrackable_keys:
                n_excl += 1
                continue
            indices.append(idx)
            if sk in hard_ab_keys:
                weights.append(float(hard_weight_mult))
                n_hard += 1
            else:
                weights.append(1.0)
                n_easy += 1
        if not indices:
            raise ValueError("All stitched sequences were excluded by untrackable_gt filter")
        w_arr = np.asarray(weights, dtype=np.float64)
        self._indices = np.asarray(indices, dtype=np.int64)
        self._probs = w_arr / w_arr.sum()
        self.stats = {"hard_pairs": n_hard, "easy_pairs": n_easy, "excluded_untrackable": n_excl}
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        total_epoch = int(num_samples_per_epoch) if num_samples_per_epoch is not None else len(indices)
        self.num_samples = int(np.ceil(total_epoch / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch * 7919)
        drawn = rng.choice(self._indices, size=self.total_size, replace=True, p=self._probs)
        return iter(int(i) for i in drawn[self.rank :: self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples
