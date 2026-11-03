"""Unit and integration tests for Dusk Till Dawn (DTD) stereo matching adapter."""

from pathlib import Path
import numpy as np
import pytest

from src.datasets.eth3d import ETH3DDataset
from src.datasets.schema import Calibration
from src.evaluation.metrics import compute_disparity_metrics
from src.stereo.deep_dtd import DTDConfig, DTDMatcher
from src.stereo.shallow_bm_sgbm import create_matcher, load_shallow_configs


def _make_synthetic_stereo_pair(
    height: int = 120,
    width: int = 160,
    shift: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a simple synthetic RGB stereo pair with constant shift."""
    np.random.seed(42)
    left = np.zeros((height, width, 3), dtype=np.uint8)
    left[20:100, 30:130] = np.random.randint(50, 250, (80, 100, 3), dtype=np.uint8)

    right = np.zeros_like(left)
    if shift < 30:
        right[20:100, (30 - shift) : (130 - shift)] = left[20:100, 30:130]
    return left, right


class TestDuskTillDawn:
    """Test suite for Dusk Till Dawn (DTD) PyTorch adapter."""

    def test_config_defaults_and_serialization(self):
        cfg = DTDConfig(
            resolution="320x192",
            interpolation="linear",
            device="cpu",
        )
        assert cfg.method_name == "DuskTillDawn"
        assert cfg.resolution == "320x192"
        assert cfg.device == "cpu"

        d = cfg.to_dict()
        cfg_restored = DTDConfig.from_dict(d)
        assert cfg_restored.resolution == "320x192"
        assert cfg_restored.device == "cpu"

    def test_invalid_resolution_raises_value_error(self):
        # 100 is not divisible by 64
        cfg = DTDConfig(resolution="100x100", device="cpu")
        with pytest.raises(ValueError, match="integer multiples of 64"):
            DTDMatcher(cfg)

    def test_missing_checkpoint_raises_error(self, tmp_path):
        fake_path = tmp_path / "non_existent_dtd.pth"
        cfg = DTDConfig(
            checkpoint_path=str(fake_path),
            auto_download=False,
            device="cpu",
        )
        with pytest.raises(FileNotFoundError, match="DTD checkpoint missing"):
            DTDMatcher(cfg)

    def test_create_matcher_factory(self):
        # By string identifier
        matcher = create_matcher("dtd", device="cpu")
        assert isinstance(matcher, DTDMatcher)
        assert matcher.config.resolution == "320x192"

        # By alias
        matcher_alias = create_matcher("dusktilldawn", device="cpu")
        assert isinstance(matcher_alias, DTDMatcher)

        # By config instance
        cfg = DTDConfig(resolution="320x192", device="cpu")
        matcher_cfg = create_matcher(cfg)
        assert isinstance(matcher_cfg, DTDMatcher)

    def test_load_shallow_configs_parses_dtd(self, tmp_path):
        cfg_file = tmp_path / "test_dtd.yaml"
        cfg_file.write_text(
            """
DuskTillDawn_Test:
  method_name: DuskTillDawn_Test
  resolution: 320x192
  device: cpu
  auto_download: true
""",
            encoding="utf-8",
        )
        configs = load_shallow_configs(str(cfg_file))
        assert "DuskTillDawn_Test" in configs
        assert isinstance(configs["DuskTillDawn_Test"], DTDConfig)
        assert configs["DuskTillDawn_Test"].resolution == "320x192"

    def test_synthetic_pair_inference(self):
        left, right = _make_synthetic_stereo_pair(height=120, width=160, shift=12)
        calib = Calibration(f_px=500.0, baseline_m=0.5)

        matcher = DTDMatcher(DTDConfig(resolution="320x192", device="cpu"))
        result = matcher.predict(left, right, calib=calib)

        # 1. Shape preservation and restoration to input image resolution
        assert result.pred_disp.shape == (120, 160)
        assert result.valid_pred.shape == (120, 160)
        assert result.pred_depth is not None
        assert result.pred_depth.shape == (120, 160)

        # 2. Timing isolation
        assert result.elapsed_s > 0
        assert result.total_elapsed_s >= result.elapsed_s
        assert result.density_pct > 50.0

        # 3. Hardware info
        assert result.hw_info is not None
        assert "backend" in result.hw_info
        assert result.hw_info["backend"] == "pytorch"

    def test_real_eth3d_sample_inference(self):
        eth3d_dir = Path("data/eth3d")
        if not eth3d_dir.is_dir():
            pytest.skip("ETH3D dataset directory not found")

        ds = ETH3DDataset(eth3d_dir, split="val_tune")
        samples = list(ds)
        if not samples:
            pytest.skip("No samples in ETH3D val_tune")

        sample = samples[0]
        matcher = DTDMatcher(DTDConfig(resolution="320x192", device="cpu"))
        result = matcher.predict(sample.left, sample.right, calib=sample.calib)

        # 1. Shape restoration to original image size
        H, W = sample.img_size
        assert result.pred_disp.shape == (H, W)

        # 2. Evaluation against ground truth
        metrics = compute_disparity_metrics(result.pred_disp, sample.gt_disp, sample.valid_mask)
        assert metrics["density_pct"] > 95.0
        assert np.isfinite(metrics["epe"])
