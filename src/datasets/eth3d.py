"""ETH3D Low-Res Two-View stereo dataset loader.

Supports two directory layouts:

**Flat layout** (actual download structure)::

    data/eth3d/
    ├── delivery_area_1l/
    │   ├── im0.png
    │   ├── im1.png
    │   ├── disp0GT.pfm       # present only in training scenes
    │   ├── mask0nocc.png      # optional
    │   ├── calib.txt          # Middlebury-style calibration
    │   ├── cameras.txt        # COLMAP intrinsics
    │   └── images.txt         # COLMAP extrinsics
    └── ...

**Nested layout** (RAFT-Stereo convention)::

    data/eth3d/
    ├── two_view_training/<scene>/im0.png, im1.png
    ├── two_view_training_gt/<scene>/disp0GT.pfm
    └── two_view_test/<scene>/im0.png, im1.png

Training scenes have ``disp0GT.pfm``; test scenes do not.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

from src.datasets.fg_masking import geometric_proxy_mask, pretrained_seg_mask
from src.datasets.io_utils import (
    parse_eth3d_calib,
    parse_eth3d_calib_txt,
    read_eth3d_disp_pfm,
    read_image_rgb,
)
from src.datasets.schema import Calibration, StereoSample

logger = logging.getLogger(__name__)

# Deterministic 80/20 split of training scenes (sorted alphabetically).
# val_tune = first 20%, eval_test = remaining 80%.
_VAL_TUNE_FRAC = 0.2


def _discover_scenes_flat(root: Path) -> dict[str, dict]:
    """Discover scenes in a flat layout (all scene dirs under root)."""
    scenes: dict[str, dict] = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d / "im0.png").is_file():
            continue
        has_gt = (d / "disp0GT.pfm").is_file()
        scenes[d.name] = {
            "img_dir": d,
            "gt_dir": d if has_gt else None,
            "has_gt": has_gt,
        }
    return scenes


def _discover_scenes_nested(root: Path) -> dict[str, dict]:
    """Discover scenes in a nested layout (two_view_training/ etc.)."""
    scenes: dict[str, dict] = {}
    train_img = root / "two_view_training"
    train_gt = root / "two_view_training_gt"
    test_img = root / "two_view_test"

    for base in (train_img, test_img):
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir() or not (d / "im0.png").is_file():
                continue
            gt_dir = train_gt / d.name if (train_gt / d.name / "disp0GT.pfm").is_file() else None
            scenes[d.name] = {
                "img_dir": d,
                "gt_dir": gt_dir,
                "has_gt": gt_dir is not None,
            }
    return scenes


class ETH3DDataset:
    """Iterator over ETH3D Low-Res Two-View stereo pairs.

    Automatically detects flat vs nested directory layout.

    Parameters:
        root:    Path to ``data/eth3d/``.
        split:   One of ``"val_tune"``, ``"eval_test"``, ``"train_all"``,
                 or ``"test_visual"``.
        scenes:  Optional explicit list of scene names (overrides split).
    """

    DATASET_NAME = "eth3d"

    def __init__(
        self,
        root: str | Path,
        split: str = "eval_test",
        scenes: Optional[Sequence[str]] = None,
        fg_strategy: str = "unavailable",
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.fg_strategy = fg_strategy

        # Auto-detect layout
        if (self.root / "two_view_training").is_dir():
            all_info = _discover_scenes_nested(self.root)
        else:
            all_info = _discover_scenes_flat(self.root)

        # Separate training (has GT) from test (no GT)
        train_names = sorted(k for k, v in all_info.items() if v["has_gt"])
        test_names = sorted(k for k, v in all_info.items() if not v["has_gt"])

        self._scene_info = all_info

        if scenes is not None:
            self._scenes = list(scenes)
        else:
            val_count = max(1, int(len(train_names) * _VAL_TUNE_FRAC))
            if split == "val_tune":
                self._scenes = train_names[:val_count]
            elif split == "eval_test":
                self._scenes = train_names[val_count:]
            elif split == "train_all":
                self._scenes = train_names
            elif split == "test_visual":
                self._scenes = test_names
            else:
                raise ValueError(
                    f"Unknown ETH3D split '{split}'. "
                    f"Use 'val_tune', 'eval_test', 'train_all', or 'test_visual'."
                )

        logger.info(
            "ETH3D split='%s': %d scenes (of %d train + %d test)",
            split, len(self._scenes), len(train_names), len(test_names),
        )

    def __len__(self) -> int:
        return len(self._scenes)

    def __iter__(self) -> Iterator[StereoSample]:
        for scene in self._scenes:
            yield self._load_scene(scene)

    def __getitem__(self, i: int) -> StereoSample:
        return self._load_scene(self._scenes[i])

    @property
    def scene_names(self) -> List[str]:
        return list(self._scenes)

    # ------------------------------------------------------------------ #
    #  Internal loading
    # ------------------------------------------------------------------ #
    def _load_scene(self, scene_name: str) -> StereoSample:
        info = self._scene_info[scene_name]
        img_dir: Path = info["img_dir"]
        gt_dir: Optional[Path] = info["gt_dir"]

        left_path = img_dir / "im0.png"
        right_path = img_dir / "im1.png"

        left_img = read_image_rgb(left_path)
        right_img = read_image_rgb(right_path)
        H, W = left_img.shape[:2]

        # Ground-truth disparity
        gt_disp = None
        valid = None
        if gt_dir is not None:
            pfm_path = gt_dir / "disp0GT.pfm"
            if pfm_path.is_file():
                gt_disp, valid = read_eth3d_disp_pfm(pfm_path)

        # Foreground mask handling based on configured strategy
        fg_mask = None
        fg_prov = "unavailable"
        if self.fg_strategy == "pretrained_seg":
            fg_mask, fg_prov = pretrained_seg_mask(left_img)
        elif self.fg_strategy == "geometric_proxy" and gt_disp is not None and valid is not None:
            fg_mask, fg_prov = geometric_proxy_mask(gt_disp, valid)

        # Calibration — prefer calib.txt (simpler), fallback to COLMAP
        calib: Optional[Calibration] = None
        ndisp: Optional[int] = None

        # Try Middlebury-style calib.txt first (in scene dir or root)
        for calib_dir in (img_dir, self.root):
            calib_txt_path = calib_dir / "calib.txt"
            if calib_txt_path.is_file():
                try:
                    f_px, baseline_m, doffs, ndisp = parse_eth3d_calib_txt(calib_txt_path)
                    calib = Calibration(f_px=f_px, baseline_m=baseline_m)
                    break
                except Exception as exc:
                    logger.warning("Failed to parse calib.txt %s: %s", calib_txt_path, exc)

        # Fallback: COLMAP cameras.txt + images.txt
        if calib is None:
            colmap_candidates = [
                (img_dir / "cameras.txt", img_dir / "images.txt"),
                (self.root / "calibration" / scene_name / "cameras.txt",
                 self.root / "calibration" / scene_name / "images.txt"),
            ]
            for cam_path, img_path in colmap_candidates:
                if cam_path.is_file() and img_path.is_file():
                    try:
                        f_px, baseline_m = parse_eth3d_calib(cam_path, img_path)
                        calib = Calibration(f_px=f_px, baseline_m=baseline_m)
                        break
                    except Exception as exc:
                        logger.warning("Failed to parse COLMAP calib %s: %s", cam_path, exc)

        if calib is None:
            logger.info("No calibration found for ETH3D scene '%s'", scene_name)

        return StereoSample(
            dataset_name=self.DATASET_NAME,
            split=self.split,
            scene_id=scene_name,
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
