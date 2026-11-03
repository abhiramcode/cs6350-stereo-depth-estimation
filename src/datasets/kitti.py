"""KITTI 2015 stereo dataset loader.

Expected directory layout (from ``data_scene_flow.zip``)::

    data/kitti2015/
    └── training/
        ├── image_2/         # left rectified colour images  (000000_10.png … 000199_10.png)
        ├── image_3/         # right rectified colour images
        ├── disp_occ_0/      # 16-bit PNG occluded disparities
        ├── obj_map/          # semantic object-class PNGs
        └── calib_cam_to_cam/ # per-scene calibration files

    data/kitti2015/
    └── testing/
        ├── image_2/
        ├── image_3/
        └── calib_cam_to_cam/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

from src.datasets.io_utils import (
    parse_kitti_calib,
    read_image_rgb,
    read_kitti_disp_png,
    read_kitti_obj_map,
)
from src.datasets.schema import Calibration, StereoSample

logger = logging.getLogger(__name__)

# Deterministic 80/20 split of 200 training scenes.
# val_tune = first 40 scenes (indices 0–39)
# eval_test = remaining 160 scenes (indices 40–199)
_TOTAL_TRAIN_SCENES = 200
_VAL_TUNE_COUNT = 40  # 20 %


def _scene_indices(split: str) -> List[int]:
    """Return the list of scene indices for the requested split."""
    if split == "val_tune":
        return list(range(0, _VAL_TUNE_COUNT))
    elif split == "eval_test":
        return list(range(_VAL_TUNE_COUNT, _TOTAL_TRAIN_SCENES))
    elif split == "train_all":
        return list(range(_TOTAL_TRAIN_SCENES))
    elif split == "test_visual":
        # test split — no GT available
        return list(range(_TOTAL_TRAIN_SCENES))
    else:
        raise ValueError(
            f"Unknown KITTI split '{split}'. "
            f"Use 'val_tune', 'eval_test', 'train_all', or 'test_visual'."
        )


class KITTI2015Dataset:
    """Iterator over KITTI 2015 stereo pairs.

    Parameters:
        root:    Path to ``data/kitti2015/`` (must contain ``training/``).
        split:   One of ``"val_tune"``, ``"eval_test"``, ``"train_all"``,
                 or ``"test_visual"``.
        indices: Optional explicit list of scene indices to load (overrides split).
    """

    DATASET_NAME = "kitti2015"

    def __init__(
        self,
        root: str | Path,
        split: str = "eval_test",
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split

        if split == "test_visual":
            self._base = self.root / "testing"
        else:
            self._base = self.root / "training"

        if not self._base.is_dir():
            raise FileNotFoundError(
                f"KITTI base directory not found: {self._base}"
            )

        self._indices = list(indices) if indices is not None else _scene_indices(split)
        self._has_gt = split != "test_visual"

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[StereoSample]:
        for idx in self._indices:
            yield self._load_scene(idx)

    def __getitem__(self, i: int) -> StereoSample:
        return self._load_scene(self._indices[i])

    # ------------------------------------------------------------------ #
    #  Internal loading
    # ------------------------------------------------------------------ #
    def _load_scene(self, idx: int) -> StereoSample:
        tag = f"{idx:06d}_10"

        left_path = self._base / "image_2" / f"{tag}.png"
        right_path = self._base / "image_3" / f"{tag}.png"

        left_img = read_image_rgb(left_path)
        right_img = read_image_rgb(right_path)
        H, W = left_img.shape[:2]

        # Ground-truth disparity
        gt_disp: Optional[__import__("numpy").ndarray] = None
        valid: Optional[__import__("numpy").ndarray] = None
        if self._has_gt:
            disp_path = self._base / "disp_occ_0" / f"{tag}.png"
            if disp_path.is_file():
                gt_disp, valid = read_kitti_disp_png(disp_path)

        # Foreground mask (obj_map)
        fg_mask = None
        fg_prov: str = "unavailable"
        if self._has_gt:
            obj_path = self._base / "obj_map" / f"{tag}.png"
            fg_mask = read_kitti_obj_map(obj_path)
            if fg_mask is not None:
                fg_prov = "semantic_gt"

        # Calibration
        calib: Optional[Calibration] = None
        calib_path = self._base / "calib_cam_to_cam" / f"{idx:06d}.txt"
        if calib_path.is_file():
            try:
                f_px, baseline_m = parse_kitti_calib(calib_path)
                calib = Calibration(f_px=f_px, baseline_m=baseline_m)
            except Exception as exc:
                logger.warning("Failed to parse KITTI calib %s: %s", calib_path, exc)

        return StereoSample(
            dataset_name=self.DATASET_NAME,
            split=self.split,
            scene_id=f"{idx:06d}",
            left_img=left_img,
            right_img=right_img,
            left_img_path=str(left_path),
            right_img_path=str(right_path),
            img_size=(H, W),
            gt_disparity=gt_disp,
            valid_mask=valid,
            fg_mask=fg_mask,
            fg_mask_provenance=fg_prov,
            calib=calib,
        )
