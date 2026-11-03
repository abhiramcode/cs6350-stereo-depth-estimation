"""Unit and integration tests for HITNet deep stereo matcher."""

from pathlib import Path
import numpy as np
import pytest

from src.datasets.eth3d import ETH3DDataset
from src.datasets.schema import Calibration
from src.evaluation.metrics import compute_disparity_metrics
from src.stereo import (
    HitnetConfig,
    HitnetMatcher,
    create_matcher,
    load_shallow_configs,
)


def _make_synthetic_stereo_pair(
    height: int = 120,
    width: int = 160,
    shift: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(42)
    base = rng.randint(40, 220, size=(height, width), dtype=np.uint8)

    left = base.copy()
    right = base.copy()

    # Textured patch
    y1, y2 = 30, 90
    x1, x2 = 50, 110
    patch = rng.randint(10, 250, size=(y2 - y1, x2 - x1), dtype=np.uint8)
    left[y1:y2, x1:x2] = patch

    rx1 = x1 - shift
    rx2 = x2 - shift
    right[y1:y2, rx1:rx2] = patch

    left_rgb = np.repeat(left[:, :, np.newaxis], 3, axis=2)
    right_rgb = np.repeat(right[:, :, np.newaxis], 3, axis=2)
    return left_rgb, right_rgb


class TestHitnetConfig:
    def test_config_serialization(self):
        cfg = HitnetConfig(model_variant="eth3d", resolution="640x480", device="cpu")
        d = cfg.to_dict()
        assert d["model_variant"] == "eth3d"
        assert d["resolution"] == "640x480"
        assert d["device"] == "cpu"
        assert d["method_name"] == "HITNet"

        cfg2 = HitnetConfig.from_dict(d)
        assert cfg2.model_variant == "eth3d"
        assert cfg2.resolution == "640x480"

    def test_missing_checkpoint_raises_when_no_auto_download(self):
        cfg = HitnetConfig(
            checkpoint_path="non_existent/path/model.onnx",
            auto_download=False,
        )
        with pytest.raises(FileNotFoundError, match="not found"):
            HitnetMatcher(cfg)


class TestHitnetMatcher:
    def test_matcher_instantiation_via_factory(self):
        matcher = create_matcher("hitnet", model_variant="eth3d")
        assert isinstance(matcher, HitnetMatcher)
        assert matcher.config.model_variant == "eth3d"

    def test_synthetic_pair_inference(self):
        left, right = _make_synthetic_stereo_pair(height=120, width=160, shift=16)
        calib = Calibration(f_px=500.0, baseline_m=0.5)

        matcher = HitnetMatcher(HitnetConfig(model_variant="eth3d", device="cpu"))
        result = matcher.predict(left, right, calib=calib)

        # 1. Shape preservation
        assert result.pred_disp.shape == (120, 160)
        assert result.valid_pred.shape == (120, 160)
        assert result.pred_depth is not None
        assert result.pred_depth.shape == (120, 160)

        # 2. Timing isolation
        assert result.elapsed_s > 0
        assert result.total_elapsed_s >= result.elapsed_s
        assert result.density_pct > 50.0

        # 3. Output range
        assert np.all(result.pred_disp[result.valid_pred] > 0)
        assert result.hw_info is not None

    def test_real_eth3d_sample_inference(self):
        eth3d_dir = Path("data/eth3d")
        if not eth3d_dir.is_dir():
            pytest.skip("ETH3D dataset directory not found")

        ds = ETH3DDataset(eth3d_dir, split="val_tune")
        samples = list(ds)
        if not samples:
            pytest.skip("No samples in ETH3D val_tune")

        sample = samples[0]
        matcher = HitnetMatcher(HitnetConfig(model_variant="eth3d", device="auto"))
        result = matcher.predict(sample.left, sample.right, calib=sample.calib)

        # Shape restoration to original image size
        H, W = sample.img_size
        assert result.pred_disp.shape == (H, W)
        assert result.valid_pred.shape == (H, W)
        assert result.density_pct > 80.0

        # Metrics against GT
        metrics = compute_disparity_metrics(
            pred_disp=result.pred_disp,
            gt_disp=sample.gt_disp,
            gt_valid_mask=sample.valid_mask,
            dataset=sample.dataset,
        )
        assert metrics["density_pct"] > 80.0
        # HITNet on ETH3D typically achieves EPE < 1.0 px
        assert metrics["epe"] < 2.0
