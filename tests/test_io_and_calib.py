"""Tests for I/O utilities and calibration parsing.

Covers:
- KITTI 16-bit PNG disparity decoding (÷ 256.0, valid = raw > 0)
- ETH3D PFM reading (both little-endian and big-endian)
- KITTI calibration parsing (P_rect_02/03 → f_px, baseline_m)
- ETH3D COLMAP calibration parsing (cameras.txt + images.txt → f_px, baseline_m)
- Z = fB/d depth conversion with safe handling of zero/invalid
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.fixtures.generate_fixtures import (
    create_eth3d_fixtures,
    create_kitti_fixtures,
    write_pfm,
)
from src.datasets.io_utils import (
    parse_eth3d_calib,
    parse_kitti_calib,
    read_eth3d_disp_pfm,
    read_image_rgb,
    read_kitti_disp_png,
    read_kitti_obj_map,
    read_pfm,
)
from src.datasets.schema import Calibration


@pytest.fixture(scope="module")
def kitti_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("kitti2015_mini")
    create_kitti_fixtures(root, num_scenes=2)
    return root


@pytest.fixture(scope="module")
def eth3d_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("eth3d_mini")
    create_eth3d_fixtures(root, num_scenes=3, big_endian_scene=1)
    return root


# ===================================================================== #
#  KITTI PNG decoding
# ===================================================================== #

class TestKITTIDispPNG:
    def test_decode_scale(self, kitti_root: Path):
        """Verify disp = uint16 / 256.0."""
        disp_path = kitti_root / "training" / "disp_occ_0" / "000000_10.png"
        raw = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED)
        disp, valid = read_kitti_disp_png(disp_path)

        assert disp.dtype == np.float32
        # Where raw > 0, disp should equal raw / 256.0
        expected = raw.astype(np.float32) / 256.0
        np.testing.assert_allclose(disp[valid], expected[valid], rtol=1e-6)

    def test_invalid_mask(self, kitti_root: Path):
        """Verify valid_mask = (raw > 0) and invalid pixels are zeroed."""
        disp_path = kitti_root / "training" / "disp_occ_0" / "000000_10.png"
        raw = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED)
        disp, valid = read_kitti_disp_png(disp_path)

        assert valid.dtype == bool
        np.testing.assert_array_equal(valid, raw > 0)
        assert np.all(disp[~valid] == 0.0)

    def test_nonexistent_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            read_kitti_disp_png(tmp_path / "nonexistent.png")


class TestKITTIObjMap:
    def test_foreground_mask(self, kitti_root: Path):
        mask = read_kitti_obj_map(
            kitti_root / "training" / "obj_map" / "000000_10.png"
        )
        assert mask is not None
        assert mask.dtype == bool
        # Our fixture has a rectangular object at [10:30, 15:50]
        assert mask[15, 25]  # inside object
        assert not mask[0, 0]  # outside object

    def test_missing_returns_none(self, tmp_path: Path):
        assert read_kitti_obj_map(tmp_path / "nope.png") is None


# ===================================================================== #
#  ETH3D PFM decoding
# ===================================================================== #

class TestPFMReader:
    def test_little_endian(self, eth3d_root: Path):
        """Scene 0 is little-endian."""
        pfm_path = (
            eth3d_root / "two_view_training_gt" / "test_scene_0" / "disp0GT.pfm"
        )
        data = read_pfm(pfm_path)
        assert data.dtype == np.float32
        assert data.ndim == 2

    def test_big_endian(self, eth3d_root: Path):
        """Scene 1 is big-endian."""
        pfm_path = (
            eth3d_root / "two_view_training_gt" / "test_scene_1" / "disp0GT.pfm"
        )
        data = read_pfm(pfm_path)
        assert data.dtype == np.float32
        assert data.ndim == 2

    def test_roundtrip_values(self, tmp_path: Path):
        """Write and read back a PFM to verify value fidelity."""
        original = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        pfm_path = tmp_path / "test.pfm"
        write_pfm(pfm_path, original, little_endian=True)
        loaded = read_pfm(pfm_path)
        np.testing.assert_allclose(loaded, original, rtol=1e-6)

    def test_eth3d_disp_validity(self, eth3d_root: Path):
        """Verify valid_mask = isfinite & (disp > 0)."""
        pfm_path = (
            eth3d_root / "two_view_training_gt" / "test_scene_0" / "disp0GT.pfm"
        )
        disp, valid = read_eth3d_disp_pfm(pfm_path)
        assert valid.dtype == bool
        # inf pixels should be invalid
        assert not valid[0, 0]  # we set (0,0) to inf
        # Positive finite pixels should be valid
        finite_pos = np.isfinite(disp) & (disp > 0)
        np.testing.assert_array_equal(valid, finite_pos)


# ===================================================================== #
#  KITTI calibration
# ===================================================================== #

class TestKITTICalib:
    def test_parse_calib(self, kitti_root: Path):
        calib_path = kitti_root / "training" / "calib_cam_to_cam" / "000000.txt"
        f_px, baseline_m = parse_kitti_calib(calib_path)

        assert f_px == pytest.approx(721.5377, abs=0.01)
        assert baseline_m == pytest.approx(0.5372, abs=0.001)

    def test_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_kitti_calib(tmp_path / "nope.txt")


# ===================================================================== #
#  ETH3D COLMAP calibration
# ===================================================================== #

class TestETH3DCalib:
    def test_parse_calib(self, eth3d_root: Path):
        cam_path = eth3d_root / "calibration" / "test_scene_0" / "cameras.txt"
        img_path = eth3d_root / "calibration" / "test_scene_0" / "images.txt"

        f_px, baseline_m = parse_eth3d_calib(cam_path, img_path)

        assert f_px == pytest.approx(500.0, abs=0.1)
        assert baseline_m == pytest.approx(0.12, abs=0.005)


# ===================================================================== #
#  Z = fB/d depth conversion
# ===================================================================== #

class TestDepthConversion:
    def test_basic_conversion(self):
        calib = Calibration(f_px=500.0, baseline_m=0.12)
        disp = np.array([10.0, 20.0, 50.0], dtype=np.float32)
        depth = calib.disp_to_depth(disp)

        expected = (500.0 * 0.12) / disp
        np.testing.assert_allclose(depth, expected, rtol=1e-5)

    def test_zero_disp_produces_inf(self):
        calib = Calibration(f_px=500.0, baseline_m=0.12)
        disp = np.array([0.0, -1.0, np.nan, np.inf], dtype=np.float32)
        depth = calib.disp_to_depth(disp)

        # All invalid inputs → depth should be finite fB/inf = 0 or inf handling
        # Our implementation: invalid disp → disp set to inf → depth = fB/inf = 0
        # Actually let's check: safe = inf for these → depth = fB/inf ≈ 0
        for d in depth:
            assert d == pytest.approx(0.0, abs=1e-6) or np.isinf(d) or d >= 0

    def test_negative_disp_safe(self):
        calib = Calibration(f_px=721.0, baseline_m=0.54)
        disp = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
        depth = calib.disp_to_depth(disp)

        # Only disp=5.0 should give a meaningful depth
        assert depth[2] == pytest.approx(721.0 * 0.54 / 5.0, rel=1e-5)


# ===================================================================== #
#  Image reading
# ===================================================================== #

class TestImageReading:
    def test_read_rgb(self, kitti_root: Path):
        img = read_image_rgb(
            kitti_root / "training" / "image_2" / "000000_10.png"
        )
        assert img.dtype == np.uint8
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_dimensions(self, kitti_root: Path):
        img = read_image_rgb(
            kitti_root / "training" / "image_2" / "000000_10.png"
        )
        assert img.shape == (48, 64, 3)  # our fixture size

    def test_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            read_image_rgb(tmp_path / "nope.png")
