"""Evaluation-day loader for arbitrary (unlabeled) stereo image pairs.

Supports any directory containing left/right image pairs with optional
calibration.  When ground truth or calibration are absent the corresponding
fields are honestly set to ``None``.

Expected layout (flexible)::

    data/custom_eval/
    ├── left/
    │   ├── 001.png
    │   └── 002.jpg
    ├── right/
    │   ├── 001.png
    │   └── 002.jpg
    └── calib.yaml          # optional — f_px and baseline_m

Or flat layout::

    data/custom_eval/
    ├── left_001.png   / right_001.png
    └── left_002.jpg   / right_002.jpg
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import yaml
import numpy as np

from src.datasets.io_utils import read_image_rgb
from src.datasets.schema import Calibration, StereoSample

logger = logging.getLogger(__name__)

# Supported image extensions
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _find_pairs_subdir(root: Path) -> List[Tuple[str, Path, Path]]:
    """Find pairs from ``left/`` and ``right/`` sub-directories."""
    left_dir = root / "left"
    right_dir = root / "right"
    if not (left_dir.is_dir() and right_dir.is_dir()):
        return []

    left_files = {
        f.stem: f
        for f in sorted(left_dir.iterdir())
        if f.suffix.lower() in _IMG_EXTS
    }
    right_files = {
        f.stem: f
        for f in sorted(right_dir.iterdir())
        if f.suffix.lower() in _IMG_EXTS
    }

    pairs = []
    for stem in sorted(left_files.keys() & right_files.keys()):
        pairs.append((stem, left_files[stem], right_files[stem]))
    return pairs


def _find_pairs_flat(root: Path) -> List[Tuple[str, Path, Path]]:
    """Find pairs from flat naming: ``left_<id>.*`` / ``right_<id>.*``."""
    lefts: Dict[str, Path] = {}
    rights: Dict[str, Path] = {}
    for f in sorted(root.iterdir()):
        if not f.is_file() or f.suffix.lower() not in _IMG_EXTS:
            continue
        name = f.stem.lower()
        if name.startswith("left_"):
            lefts[name[5:]] = f
        elif name.startswith("right_"):
            rights[name[6:]] = f

    pairs = []
    for key in sorted(lefts.keys() & rights.keys()):
        pairs.append((key, lefts[key], rights[key]))
    return pairs


def _load_optional_calib(root: Path) -> Optional[Calibration]:
    """Load calibration from ``calib.yaml`` if present."""
    calib_path = root / "calib.yaml"
    if not calib_path.is_file():
        return None
    try:
        with open(calib_path, "r") as f:
            cfg = yaml.safe_load(f)
        f_px = float(cfg["f_px"])
        baseline_m = float(cfg["baseline_m"])
        return Calibration(f_px=f_px, baseline_m=baseline_m)
    except Exception as exc:
        logger.warning("Could not parse %s: %s", calib_path, exc)
        return None


class CustomPairDataset:
    """Iterator over arbitrary stereo image pairs for evaluation-day inference.

    Parameters:
        root:   Path to the custom dataset directory.
        pairs:  Optional explicit list of ``(scene_id, left_path, right_path)``.
    """

    DATASET_NAME = "custom"

    def __init__(
        self,
        root: str | Path,
        pairs: Optional[Sequence[Tuple[str, str | Path, str | Path]]] = None,
    ) -> None:
        self.root = Path(root)

        if pairs is not None:
            self._pairs = [
                (sid, Path(lp), Path(rp)) for sid, lp, rp in pairs
            ]
        else:
            self._pairs = _find_pairs_subdir(self.root)
            if not self._pairs:
                self._pairs = _find_pairs_flat(self.root)

        if not self._pairs:
            logger.warning("No stereo pairs found in %s", self.root)

        self._calib = _load_optional_calib(self.root)

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self) -> Iterator[StereoSample]:
        for scene_id, left_path, right_path in self._pairs:
            yield self._load_pair(scene_id, left_path, right_path)

    def __getitem__(self, i: int) -> StereoSample:
        sid, lp, rp = self._pairs[i]
        return self._load_pair(sid, lp, rp)

    def _load_pair(
        self, scene_id: str, left_path: Path, right_path: Path
    ) -> StereoSample:
        left_img = read_image_rgb(left_path)
        right_img = read_image_rgb(right_path)
        H, W = left_img.shape[:2]

        return StereoSample(
            dataset_name=self.DATASET_NAME,
            split="custom",
            scene_id=scene_id,
            left_img=left_img,
            right_img=right_img,
            left_img_path=str(left_path),
            right_img_path=str(right_path),
            img_size=(H, W),
            gt_disparity=None,
            valid_mask=None,
            fg_mask=None,
            fg_mask_provenance="unavailable",
            calib=self._calib,
        )
