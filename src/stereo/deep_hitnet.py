"""Headless inference adapter for Google HITNet stereo matching model.

HITNet (Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching,
CVPR 2021) operates via ONNX Runtime without requiring heavy training dependencies.
Official checkpoints are pre-trained on ETH3D (grayscale), Middlebury (RGB), and SceneFlow (RGB).

Coordinate System & Resolution Restoration:
    Because disparity represents horizontal pixel displacement ($d = x_L - x_R$), resizing
    an image by a horizontal factor $s_x = W_{orig} / W_{model}$ scales the true physical
    disparity linearly by $s_x$.
    This adapter resizes inputs to the model's native resolution $(W_{model}, H_{model})$ via
    area interpolation, runs ONNX inference, resizes the output disparity map back to the
    original image dimensions $(W_{orig}, H_{orig})$ via nearest-neighbor interpolation, and
    scales the disparity values by $s_x$.

Reference:
    - Paper: https://arxiv.org/abs/2007.12140 (CVPR 2021)
    - Source: https://github.com/nburrus/stereodemo (method_hitnet.py)
    - ONNX models: PINTO0309 model zoo (Model 142)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

import cv2
import numpy as np

from src.stereo.base import MatcherConfig, StereoMatcher

logger = logging.getLogger(__name__)

# Verified official release URLs from nburrus/stereodemo
HITNET_URLS: Dict[str, str] = {
    "hitnet_eth3d_120x160.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_eth3d_120x160.onnx",
    "hitnet_eth3d_240x320.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_eth3d_240x320.onnx",
    "hitnet_eth3d_480x640.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_eth3d_480x640.onnx",
    "hitnet_eth3d_720x1280.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_eth3d_720x1280.onnx",
    "hitnet_middlebury_120x160.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_middlebury_120x160.onnx",
    "hitnet_middlebury_240x320.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_middlebury_240x320.onnx",
    "hitnet_middlebury_480x640.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_middlebury_480x640.onnx",
    "hitnet_middlebury_720x1280.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_middlebury_720x1280.onnx",
    "hitnet_sceneflow_120x160.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_sceneflow_120x160.onnx",
    "hitnet_sceneflow_240x320.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_sceneflow_240x320.onnx",
    "hitnet_sceneflow_480x640.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_sceneflow_480x640.onnx",
    "hitnet_sceneflow_720x1280.onnx": "https://github.com/nburrus/stereodemo/releases/download/v0.1-hitnet/hitnet_sceneflow_720x1280.onnx",
}


@dataclass
class HitnetConfig(MatcherConfig):
    """Configuration for HITNet ONNX inference.

    Attributes:
        method_name: Identifier for logging/benchmarks (e.g. "HITNet").
        model_variant: "eth3d" (grayscale), "sceneflow" (RGB), or "middlebury" (RGB).
        resolution: "640x480", "1280x720", "320x240", or "160x120" (Width x Height).
        checkpoint_path: Explicit path to .onnx file. If None, resolves to checkpoints/hitnet/.
        device: "auto", "cuda", or "cpu".
        auto_download: Whether to fetch checkpoint from official URL if missing.
    """

    method_name: str = "HITNet"
    model_variant: str = "eth3d"
    resolution: str = "640x480"
    checkpoint_path: Optional[str] = None
    device: str = "auto"
    auto_download: bool = True


class HitnetMatcher(StereoMatcher):
    """Headless ONNX Runtime adapter for HITNet stereo matching."""

    def __init__(self, config: Optional[HitnetConfig] = None) -> None:
        config = config or HitnetConfig()
        super().__init__(config)
        self.config: HitnetConfig = config

        self._model_path = self._resolve_model_path()
        self._ensure_checkpoint()

        self._session, self._active_providers = self._init_session()

        # Parse model I/O signatures
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

        # Expected input shape: [1, C, H, W]
        in_shape = inputs[0].shape
        self._expected_channels = in_shape[1]
        self._model_height = in_shape[2]
        self._model_width = in_shape[3]

        # Declare required color mode: 2 channels = grayscale, 6 channels = RGB
        self.input_color = "gray" if self._expected_channels == 2 else "rgb"

        logger.info(
            "Loaded HITNet model from %s [input: %s (%dx%d, C=%d), providers: %s]",
            self._model_path.name,
            self._input_name,
            self._model_width,
            self._model_height,
            self._expected_channels,
            self._active_providers,
        )

    def _resolve_model_path(self) -> Path:
        if self.config.checkpoint_path:
            return Path(self.config.checkpoint_path)

        cols, rows = self.config.resolution.lower().split("x")
        filename = f"hitnet_{self.config.model_variant.lower()}_{rows}x{cols}.onnx"
        return Path("checkpoints/hitnet") / filename

    def _ensure_checkpoint(self) -> None:
        if self._model_path.is_file():
            return

        filename = self._model_path.name
        if filename not in HITNET_URLS:
            raise FileNotFoundError(
                f"HITNet checkpoint not found at {self._model_path} and '{filename}' "
                f"is not a known downloadable model. Available: {list(HITNET_URLS.keys())}"
            )

        if not self.config.auto_download:
            raise FileNotFoundError(
                f"HITNet checkpoint missing at {self._model_path}. "
                f"Enable auto_download=True or download from {HITNET_URLS[filename]}"
            )

        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        url = HITNET_URLS[filename]
        logger.info("Downloading HITNet checkpoint from %s to %s...", url, self._model_path)
        urllib.request.urlretrieve(url, self._model_path)

        if not self._model_path.is_file() or self._model_path.stat().st_size < 100_000:
            raise RuntimeError(f"Failed to download valid HITNet checkpoint to {self._model_path}")

        logger.info("Downloaded %s (%.1f MB)", filename, self._model_path.stat().st_size / 1e6)

    def _init_session(self):
        import onnxruntime as ort

        device = self.config.device.lower()
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:  # "auto"
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(str(self._model_path), sess_options=opts, providers=providers)
        active_providers = session.get_providers()
        return session, active_providers

    def get_hw_info(self) -> Dict[str, Any]:
        info = super().get_hw_info()
        info["device"] = self.config.device
        info["onnx_providers"] = self._active_providers
        return info

    def _compute_disparity(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        """Run HITNet inference and return disparity in original image pixel units.

        Args:
            left:  ``(H, W) uint8`` grayscale or ``(H, W, 3) uint8`` RGB.
            right: ``(H, W) uint8`` grayscale or ``(H, W, 3) uint8`` RGB.

        Returns:
            ``(H, W) float32`` disparity map in original pixel coordinates.
        """
        orig_h, orig_w = left.shape[:2]

        # 1. Resize to native model resolution
        target_size = (self._model_width, self._model_height)
        left_resized = cv2.resize(left, target_size, interpolation=cv2.INTER_AREA)
        right_resized = cv2.resize(right, target_size, interpolation=cv2.INTER_AREA)

        # 2. Add channel dimension if 2D grayscale
        if left_resized.ndim == 2:
            left_resized = left_resized[:, :, np.newaxis]
            right_resized = right_resized[:, :, np.newaxis]

        # 3. Concatenate along channel axis: [I_L, I_R] normalized to [0, 1]
        combined = np.concatenate((left_resized, right_resized), axis=-1).astype(np.float32) / 255.0

        # 4. Transpose to [1, C, H, W]
        tensor = np.transpose(combined, (2, 0, 1))[np.newaxis, ...]

        # 5. ONNX Inference
        outputs = self._session.run([self._output_name], {self._input_name: tensor})
        raw_disp = outputs[0][0].squeeze(-1)  # (H_model, W_model)

        # 6. Resolution restoration & coordinate system preservation
        # Disparity is horizontal pixel displacement: d_orig = d_model * (W_orig / W_model)
        x_scale = float(orig_w) / float(self._model_width)
        disp_resized = cv2.resize(raw_disp, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        disp_orig = disp_resized * np.float32(x_scale)

        # 7. Mask invalid / negative values
        disp_orig[disp_orig <= 0] = 0.0
        disp_orig[~np.isfinite(disp_orig)] = 0.0

        return disp_orig.astype(np.float32)
