"""Tests for dataset loaders and StereoSample schema.

Covers:
- StereoSample field types and constraints
- KITTI2015Dataset iteration, split sizes, and metadata
- ETH3DDataset iteration, split sizes, and metadata
- CustomPairDataset with and without calibration
- Missing-label / missing-calib graceful fallback
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.fixtures.generate_fixtures import (
    create_custom_fixtures,
    create_eth3d_fixtures,
    create_kitti_fixtures,
)
from src.datasets.schema import Calibration, StereoSample
from src.datasets.kitti import KITTI2015Dataset
from src.datasets.eth3d import ETH3DDataset
from src.datasets.custom_pair import CustomPairDataset
from src.datasets.fg_masking import geometric_proxy_mask


@pytest.fixture(scope="module")
def kitti_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("kitti_ds")
    create_kitti_fixtures(root, num_scenes=3)
    return root


@pytest.fixture(scope="module")
def eth3d_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("eth3d_ds")
    create_eth3d_fixtures(root, num_scenes=3)
    return root


@pytest.fixture(scope="module")
def custom_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("custom_ds")
    create_custom_fixtures(root, num_pairs=2)
    return root


# ===================================================================== #
#  StereoSample schema
# ===================================================================== #

class TestStereoSample:
    def test_schema_fields(self):
        img = np.zeros((10, 20, 3), dtype=np.uint8)
        sample = StereoSample(
            dataset_name="test",
            split="eval_test",
            scene_id="000",
            left_img=img,
            right_img=img,
            left_img_path="/fake/left.png",
            right_img_path="/fake/right.png",
            img_size=(10, 20),
        )
        assert sample.dataset_name == "test"
        assert sample.gt_disparity is None
        assert sample.valid_mask is None
        assert sample.fg_mask is None
        assert sample.fg_mask_provenance == "unavailable"
        assert sample.calib is None
        assert not sample.has_gt
        assert not sample.has_calib

    def test_with_gt(self):
        img = np.zeros((10, 20, 3), dtype=np.uint8)
        disp = np.ones((10, 20), dtype=np.float32)
        valid = np.ones((10, 20), dtype=bool)
        sample = StereoSample(
            dataset_name="test",
            split="eval_test",
            scene_id="000",
            left_img=img,
            right_img=img,
            left_img_path="/fake/left.png",
            right_img_path="/fake/right.png",
            img_size=(10, 20),
            gt_disparity=disp,
            valid_mask=valid,
            calib=Calibration(f_px=500.0, baseline_m=0.12),
        )
        assert sample.has_gt
        assert sample.has_calib

    def test_summary_string(self):
        img = np.zeros((10, 20, 3), dtype=np.uint8)
        sample = StereoSample(
            dataset_name="kitti2015",
            split="eval_test",
            scene_id="000042",
            left_img=img,
            right_img=img,
            left_img_path="/fake/left.png",
            right_img_path="/fake/right.png",
            img_size=(10, 20),
        )
        s = sample.summary()
        assert "kitti2015" in s
        assert "000042" in s
        assert "no-GT" in s


# ===================================================================== #
#  KITTI dataset loader
# ===================================================================== #

class TestKITTI2015Dataset:
    def test_load_all(self, kitti_root: Path):
        # Fixture only has 3 scenes; pass explicit indices (train_all assumes 200)
        ds = KITTI2015Dataset(kitti_root, split="train_all", indices=[0, 1, 2])
        assert len(ds) == 3
        samples = list(ds)
        assert len(samples) == 3

    def test_sample_fields(self, kitti_root: Path):
        ds = KITTI2015Dataset(kitti_root, split="train_all", indices=[0])
        sample = ds[0]

        assert isinstance(sample, StereoSample)
        assert sample.dataset_name == "kitti2015"
        assert sample.left_img.shape == (48, 64, 3)
        assert sample.right_img.shape == (48, 64, 3)
        assert sample.img_size == (48, 64)
        assert sample.has_gt
        assert sample.gt_disparity.dtype == np.float32
        assert sample.valid_mask.dtype == bool
        assert sample.fg_mask is not None
        assert sample.fg_mask_provenance == "semantic_gt"
        assert sample.has_calib
        assert sample.calib.f_px == pytest.approx(721.5377, abs=0.01)

    def test_dimension_alignment(self, kitti_root: Path):
        """All arrays (img, disp, mask) must have same H×W."""
        ds = KITTI2015Dataset(kitti_root, split="train_all", indices=[0])
        s = ds[0]
        H, W = s.img_size
        assert s.left_img.shape[:2] == (H, W)
        assert s.right_img.shape[:2] == (H, W)
        assert s.gt_disparity.shape == (H, W)
        assert s.valid_mask.shape == (H, W)
        assert s.fg_mask.shape == (H, W)

    def test_explicit_indices(self, kitti_root: Path):
        ds = KITTI2015Dataset(kitti_root, split="train_all", indices=[1])
        assert len(ds) == 1
        assert ds[0].scene_id == "000001"


# ===================================================================== #
#  ETH3D dataset loader
# ===================================================================== #

class TestETH3DDataset:
    def test_load_all(self, eth3d_root: Path):
        ds = ETH3DDataset(eth3d_root, split="train_all")
        assert len(ds) == 3
        samples = list(ds)
        assert len(samples) == 3

    def test_sample_fields(self, eth3d_root: Path):
        ds = ETH3DDataset(eth3d_root, split="train_all")
        sample = ds[0]

        assert isinstance(sample, StereoSample)
        assert sample.dataset_name == "eth3d"
        assert sample.has_gt
        assert sample.gt_disparity.dtype == np.float32
        assert sample.valid_mask.dtype == bool
        assert sample.fg_mask is None
        assert sample.fg_mask_provenance == "unavailable"
        assert sample.has_calib

    def test_no_fg_mask(self, eth3d_root: Path):
        """ETH3D must not fabricate semantic fg masks."""
        ds = ETH3DDataset(eth3d_root, split="train_all")
        for sample in ds:
            assert sample.fg_mask is None
            assert sample.fg_mask_provenance == "unavailable"


# ===================================================================== #
#  Custom pair dataset
# ===================================================================== #

class TestCustomPairDataset:
    def test_load_pairs(self, custom_root: Path):
        ds = CustomPairDataset(custom_root)
        assert len(ds) == 2
        sample = ds[0]
        assert sample.dataset_name == "custom"
        assert sample.split == "custom"
        assert not sample.has_gt
        assert sample.calib is None

    def test_with_calib(self, custom_root: Path):
        # Write a calib.yaml
        calib_path = custom_root / "calib.yaml"
        calib_path.write_text(yaml.dump({"f_px": 600.0, "baseline_m": 0.1}))

        ds = CustomPairDataset(custom_root)
        sample = ds[0]
        assert sample.has_calib
        assert sample.calib.f_px == pytest.approx(600.0)
        assert sample.calib.baseline_m == pytest.approx(0.1)

    def test_empty_dir(self, tmp_path: Path):
        ds = CustomPairDataset(tmp_path)
        assert len(ds) == 0


# ===================================================================== #
#  Foreground masking
# ===================================================================== #

class TestGeometricProxyMask:
    def test_provenance_is_geometric(self):
        disp = np.array([1.0, 5.0, 10.0, 20.0], dtype=np.float32)
        valid = np.ones(4, dtype=bool)
        mask, prov = geometric_proxy_mask(disp, valid)
        assert prov == "geometric_proxy"

    def test_above_median(self):
        disp = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        valid = np.ones(5, dtype=bool)
        mask, _ = geometric_proxy_mask(disp, valid, quantile=0.5)
        # median = 3.0 → fg = disp > 3.0 → indices 3, 4
        assert mask[3] and mask[4]
        assert not mask[0] and not mask[1]

    def test_empty_valid(self):
        disp = np.array([1.0, 2.0], dtype=np.float32)
        valid = np.zeros(2, dtype=bool)
        mask, prov = geometric_proxy_mask(disp, valid)
        assert not np.any(mask)
        assert prov == "geometric_proxy"
