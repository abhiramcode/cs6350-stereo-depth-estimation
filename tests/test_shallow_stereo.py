"""Unit tests for shallow stereo matchers (StereoBM and StereoSGBM)."""

from pathlib import Path
import numpy as np
import pytest

from src.datasets.schema import Calibration
from src.stereo import (
    MatcherConfig,
    StereoBMConfig,
    StereoBMMatcher,
    StereoSGBMConfig,
    StereoSGBMMatcher,
    create_matcher,
    load_shallow_configs,
)


def _make_synthetic_stereo_pair(
    height: int = 120,
    width: int = 160,
    shift: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic stereo pair with random texture and a shifted central patch.

    Left image has a textured square in the center. Right image has the same square
    shifted left by `shift` pixels (producing true disparity = `shift`).
    """
    rng = np.random.RandomState(42)
    # Background texture
    base = rng.randint(40, 220, size=(height, width), dtype=np.uint8)

    left = base.copy()
    right = base.copy()

    # Create a distinctive textured foreground square
    y1, y2 = 30, 90
    x1, x2 = 50, 110
    patch = rng.randint(10, 250, size=(y2 - y1, x2 - x1), dtype=np.uint8)

    # Place patch on left
    left[y1:y2, x1:x2] = patch

    # In right view, the foreground object is shifted left by `shift` pixels
    rx1 = x1 - shift
    rx2 = x2 - shift
    right[y1:y2, rx1:rx2] = patch

    # Convert to 3-channel RGB
    left_rgb = np.repeat(left[:, :, np.newaxis], 3, axis=2)
    right_rgb = np.repeat(right[:, :, np.newaxis], 3, axis=2)

    return left_rgb, right_rgb


class TestStereoBMMatcher:
    """Tests for OpenCV StereoBM wrapper."""

    def test_synthetic_pair_disparity(self):
        shift = 16
        left, right = _make_synthetic_stereo_pair(shift=shift)
        cfg = StereoBMConfig(numDisparities=32, blockSize=15, uniquenessRatio=5)
        matcher = StereoBMMatcher(cfg)
        result = matcher.predict(left, right)

        assert result.pred_disp.shape == (120, 160)
        assert result.valid_pred.dtype == bool
        assert result.elapsed_s > 0
        assert result.total_elapsed_s >= result.elapsed_s

        # In the central region of the foreground patch, disparity should be close to `shift`
        patch_disp = result.pred_disp[40:80, 60:90]
        patch_valid = result.valid_pred[40:80, 60:90]
        if np.any(patch_valid):
            median_disp = np.median(patch_disp[patch_valid])
            assert abs(median_disp - shift) <= 2.0

    def test_invalid_pixels_are_non_positive(self):
        # Uniform zero image produces no matches
        zero_left = np.zeros((64, 64, 3), dtype=np.uint8)
        zero_right = np.zeros((64, 64, 3), dtype=np.uint8)
        matcher = StereoBMMatcher(StereoBMConfig(numDisparities=16, blockSize=7))
        result = matcher.predict(zero_left, zero_right)

        # Pixels where no match could be found should be <= 0 and marked invalid
        assert np.all(~result.valid_pred | (result.pred_disp > 0))
        assert result.density_pct >= 0.0

    def test_depth_conversion_with_calibration(self):
        left, right = _make_synthetic_stereo_pair(shift=16)
        calib = Calibration(f_px=500.0, baseline_m=0.5)
        matcher = StereoBMMatcher(StereoBMConfig(numDisparities=32, blockSize=15))
        result = matcher.predict(left, right, calib=calib)

        assert result.pred_depth is not None
        assert result.pred_depth.shape == (120, 160)
        # For valid pixels: Z = f * B / d = 500 * 0.5 / d = 250 / d
        valid = result.valid_pred
        if np.any(valid):
            expected = 250.0 / result.pred_disp[valid]
            np.testing.assert_allclose(result.pred_depth[valid], expected, rtol=1e-5)

    def test_no_calibration_leaves_depth_none(self):
        left, right = _make_synthetic_stereo_pair(shift=16)
        matcher = StereoBMMatcher(StereoBMConfig(numDisparities=32, blockSize=15))
        result = matcher.predict(left, right, calib=None)
        assert result.pred_depth is None


class TestStereoSGBMMatcher:
    """Tests for OpenCV StereoSGBM wrapper."""

    def test_synthetic_pair_disparity(self):
        shift = 16
        left, right = _make_synthetic_stereo_pair(shift=shift)
        cfg = StereoSGBMConfig(numDisparities=32, blockSize=5, uniquenessRatio=5)
        matcher = StereoSGBMMatcher(cfg)
        result = matcher.predict(left, right)

        assert result.pred_disp.shape == (120, 160)
        assert result.valid_pred.dtype == bool
        assert result.elapsed_s > 0

        patch_disp = result.pred_disp[40:80, 60:90]
        patch_valid = result.valid_pred[40:80, 60:90]
        if np.any(patch_valid):
            median_disp = np.median(patch_disp[patch_valid])
            assert abs(median_disp - shift) <= 2.0

    def test_grayscale_and_single_channel_inputs(self):
        left, right = _make_synthetic_stereo_pair(shift=16)
        left_2d = left[:, :, 0]
        right_2d = right[:, :, 0]

        matcher = StereoSGBMMatcher(StereoSGBMConfig(numDisparities=32, blockSize=5))
        # 2D input
        r1 = matcher.predict(left_2d, right_2d)
        assert r1.pred_disp.shape == (120, 160)

        # 3D single channel input
        r2 = matcher.predict(left_2d[:, :, None], right_2d[:, :, None])
        assert r2.pred_disp.shape == (120, 160)


class TestConfigAndFactory:
    """Tests for MatcherConfig, serialization, and factory creation."""

    def test_bm_config_serialization(self):
        cfg = StereoBMConfig(numDisparities=64, blockSize=21, uniquenessRatio=12)
        d = cfg.to_dict()
        assert d["numDisparities"] == 64
        assert d["blockSize"] == 21
        assert d["method_name"] == "StereoBM"

        cfg2 = StereoBMConfig.from_dict(d)
        assert cfg2.numDisparities == 64
        assert cfg2.blockSize == 21

    def test_sgbm_config_serialization(self):
        cfg = StereoSGBMConfig(numDisparities=128, blockSize=7, mode=2)
        d = cfg.to_dict()
        assert d["numDisparities"] == 128
        assert d["blockSize"] == 7
        assert d["mode"] == 2

        cfg2 = StereoSGBMConfig.from_dict(d)
        assert cfg2.numDisparities == 128
        assert cfg2.blockSize == 7

    def test_factory_string_names(self):
        m1 = create_matcher("StereoBM", numDisparities=64)
        assert isinstance(m1, StereoBMMatcher)
        assert m1.config.numDisparities == 64

        m2 = create_matcher("StereoSGBM", blockSize=7)
        assert isinstance(m2, StereoSGBMMatcher)
        assert m2.config.blockSize == 7

        m3 = create_matcher("StereoSGBM_WLS")
        assert isinstance(m3, StereoSGBMMatcher)

    def test_factory_with_config_object(self):
        cfg = StereoBMConfig(numDisparities=32)
        m = create_matcher(cfg, blockSize=11)
        assert isinstance(m, StereoBMMatcher)
        assert m.config.numDisparities == 32
        assert m.config.blockSize == 11

    def test_factory_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            create_matcher("non_existent_matcher")

    def test_load_shallow_configs_yaml(self):
        yaml_path = Path("configs/shallow_methods.yaml")
        assert yaml_path.is_file()
        configs = load_shallow_configs(str(yaml_path))

        assert "StereoBM" in configs
        assert "StereoSGBM" in configs
        assert "StereoSGBM_WLS" in configs

        assert isinstance(configs["StereoBM"], StereoBMConfig)
        assert isinstance(configs["StereoSGBM"], StereoSGBMConfig)
        assert isinstance(configs["StereoSGBM_WLS"], StereoSGBMConfig)
        assert configs["StereoSGBM_WLS"].use_wls_filter is True
