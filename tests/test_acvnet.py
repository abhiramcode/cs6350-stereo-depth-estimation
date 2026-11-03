"""Unit and integration tests for Fast-ACVNet stereo matching adapter."""

from pathlib import Path
import numpy as np
import pytest

from src.datasets.eth3d import ETH3DDataset
from src.datasets.schema import Calibration
from src.evaluation.metrics import compute_disparity_metrics
from src.stereo.deep_acvnet import ACVNetConfig, ACVNetMatcher
from src.stereo.shallow_bm_sgbm import create_matcher, load_shallow_configs


def _make_synthetic_stereo_pair(
    height: int = 120,
    width: int = 160,
    shift: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a simple synthetic RGB stereo pair with constant shift."""
    np.random.seed(42)
    left = np.zeros((height, width, 3), dtype=np.uint8)
    # Add a bright textured square in the center
    left[30:90, 40:120] = np.random.randint(50, 250, (60, 80, 3), dtype=np.uint8)

    right = np.zeros_like(left)
    # Shift horizontal coordinates to the left
    if shift < 40:
        right[30:90, (40 - shift) : (120 - shift)] = left[30:90, 40:120]
    return left, right


class TestFastACVNet:
    """Test suite for Fast-ACVNet ONNX adapter."""

    def test_config_defaults_and_serialization(self):
        cfg = ACVNetConfig(
            model_variant="generalization",
            resolution="640x480",
            interpolation="linear",
            device="cpu",
        )
        assert cfg.method_name == "Fast-ACVNet"
        assert cfg.model_variant == "generalization"
        assert cfg.resolution == "640x480"
        assert cfg.device == "cpu"

        d = cfg.to_dict()
        cfg_restored = ACVNetConfig.from_dict(d)
        assert cfg_restored.model_variant == "generalization"
        assert cfg_restored.resolution == "640x480"
        assert cfg_restored.device == "cpu"

    def test_missing_checkpoint_raises_error(self, tmp_path):
        fake_path = tmp_path / "non_existent_acvnet.onnx"
        cfg = ACVNetConfig(
            checkpoint_path=str(fake_path),
            auto_download=False,
        )
        with pytest.raises(FileNotFoundError, match="Fast-ACVNet checkpoint missing"):
            ACVNetMatcher(cfg)

    def test_create_matcher_factory(self):
        # By string identifier
        matcher = create_matcher("fast_acvnet", device="cpu")
        assert isinstance(matcher, ACVNetMatcher)
        assert matcher.config.model_variant == "generalization"

        # By specific kitti variant
        matcher_kitti = create_matcher("acvnet_kitti", device="cpu")
        assert isinstance(matcher_kitti, ACVNetMatcher)
        assert matcher_kitti.config.model_variant == "kitti_2015"

        # By config instance
        cfg = ACVNetConfig(model_variant="generalization", device="cpu")
        matcher_cfg = create_matcher(cfg)
        assert isinstance(matcher_cfg, ACVNetMatcher)

    def test_load_shallow_configs_parses_acvnet(self, tmp_path):
        cfg_file = tmp_path / "test_deep.yaml"
        cfg_file.write_text(
            """
Fast_ACVNet_Gen:
  method_name: Fast_ACVNet_Gen
  model_variant: generalization
  resolution: 640x480
  device: cpu
  auto_download: true
""",
            encoding="utf-8",
        )
        configs = load_shallow_configs(str(cfg_file))
        assert "Fast_ACVNet_Gen" in configs
        assert isinstance(configs["Fast_ACVNet_Gen"], ACVNetConfig)
        assert configs["Fast_ACVNet_Gen"].model_variant == "generalization"

    def test_synthetic_pair_inference(self):
        left, right = _make_synthetic_stereo_pair(height=120, width=160, shift=12)
        calib = Calibration(f_px=500.0, baseline_m=0.5)

        matcher = ACVNetMatcher(ACVNetConfig(model_variant="generalization", device="cpu"))
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

        # 3. Output range & hardware info
        assert np.all(result.pred_disp[result.valid_pred] > 0)
        assert result.hw_info is not None
        assert "device" in result.hw_info
        assert "onnx_providers" in result.hw_info

    def test_real_eth3d_sample_inference(self):
        eth3d_dir = Path("data/eth3d")
        if not eth3d_dir.is_dir():
            pytest.skip("ETH3D dataset directory not found")

        ds = ETH3DDataset(eth3d_dir, split="val_tune")
        samples = list(ds)
        if not samples:
            pytest.skip("No samples in ETH3D val_tune")

        sample = samples[0]
        matcher = ACVNetMatcher(ACVNetConfig(model_variant="generalization", device="auto"))
        result = matcher.predict(sample.left, sample.right, calib=sample.calib)

        # 1. Shape restoration to original image size
        H, W = sample.img_size
        assert result.pred_disp.shape == (H, W)

        # 2. Evaluation against ground truth: density and EPE check
        metrics = compute_disparity_metrics(result.pred_disp, sample.gt_disp, sample.valid_mask)
        assert metrics["density_pct"] > 95.0
        # Fast-ACVNet Generalization achieves EPE < 0.5 px on delivery_area_1l
        assert metrics["epe"] < 0.8
        assert metrics["bad_2_0"] < 5.0
