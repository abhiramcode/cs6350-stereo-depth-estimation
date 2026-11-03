"""Base interfaces for stereo matchers.

Every matcher (shallow or deep) must subclass ``StereoMatcher`` and
implement ``predict(left, right) -> StereoResult``.

``StereoResult`` bundles the predicted disparity, a validity mask,
optional metric depth, per-image timing, and the exact configuration
used — everything needed for fair benchmarking.
"""

from __future__ import annotations

import platform
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.datasets.schema import Calibration


# ===================================================================== #
#  Configuration base
# ===================================================================== #

@dataclass
class MatcherConfig:
    """Base configuration.  Subclass per-method."""

    method_name: str = "base"

    def to_dict(self) -> Dict[str, Any]:
        """Serialise all fields to a plain dict (for logging / YAML)."""
        import dataclasses
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MatcherConfig":
        """Instantiate config from dictionary, ignoring extra keys."""
        import dataclasses
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


# ===================================================================== #
#  Result container
# ===================================================================== #

@dataclass
class StereoResult:
    """Output of a single stereo inference call.

    Attributes:
        pred_disp:     ``float32 (H, W)`` predicted disparity in **pixels**.
                       Invalid pixels are ``<= 0``.
        valid_pred:    ``bool (H, W)`` mask of valid predictions.
        pred_depth:    ``float32 (H, W)`` metric depth in metres, or ``None``
                       if calibration was unavailable.
        img_size:      ``(H, W)`` of the input images.
        elapsed_s:     Pure inference wall-clock time in seconds (excludes
                       pre-/post-processing).
        total_elapsed_s: End-to-end time including grayscale conversion,
                       disparity extraction, etc.
        config:        The exact ``MatcherConfig`` used for this prediction.
        density_pct:   Percentage of pixels with valid disparity.
        hw_info:       CPU / device metadata dict.
    """

    pred_disp: np.ndarray                   # float32 (H, W), pixels
    valid_pred: np.ndarray                  # bool (H, W)
    pred_depth: Optional[np.ndarray]        # float32 (H, W) or None
    img_size: Tuple[int, int]               # (H, W)
    elapsed_s: float                        # inference only (seconds)
    total_elapsed_s: float                  # end-to-end (seconds)
    config: MatcherConfig
    density_pct: float = 0.0
    hw_info: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        H, W = self.img_size
        return (
            f"{self.config.method_name}  {W}×{H}  "
            f"infer={self.elapsed_s*1000:.1f}ms  "
            f"total={self.total_elapsed_s*1000:.1f}ms  "
            f"density={self.density_pct:.1f}%  "
            f"depth={'yes' if self.pred_depth is not None else 'no'}"
        )


# ===================================================================== #
#  Abstract matcher
# ===================================================================== #

def _cpu_info() -> Dict[str, Any]:
    return {
        "processor": platform.processor() or platform.machine(),
        "system": platform.system(),
        "python": platform.python_version(),
    }


class StereoMatcher(ABC):
    """Unified interface for all stereo matching methods.

    Subclasses implement ``_compute_disparity`` which receives
    preprocessed images and returns raw disparity.

    The public ``predict`` method handles:
    - Preprocessing (grayscale conversion if ``input_color == 'gray'``)
    - Timing (inference-only and end-to-end)
    - Fixed-point → float conversion
    - Valid-pixel masking
    - Optional depth conversion via calibration
    """

    input_color: str = "gray"

    def __init__(self, config: MatcherConfig) -> None:
        self.config = config

    @abstractmethod
    def _compute_disparity(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        """Run the raw stereo algorithm on preprocessed images.

        Returns raw output in pixel units (or fixed-point for OpenCV methods).
        """
        ...

    def get_hw_info(self) -> Dict[str, Any]:
        """Return execution hardware info (can be overridden by subclasses)."""
        return _cpu_info()

    def predict(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        calib: Optional[Calibration] = None,
    ) -> StereoResult:
        """Run end-to-end inference.

        Args:
            left_rgb:  ``(H, W, 3) uint8`` left image.
            right_rgb: ``(H, W, 3) uint8`` right image.
            calib:     Optional calibration for depth conversion.

        Returns:
            ``StereoResult`` with disparity, timing, and metadata.
        """
        import cv2

        t0 = time.perf_counter()
        H, W = left_rgb.shape[:2]

        if getattr(self, "input_color", "gray") == "rgb":
            left_in = left_rgb
            right_in = right_rgb
        else:
            # Preprocessing: convert to grayscale uint8
            def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
                if img.ndim == 2:
                    gray = img
                elif img.ndim == 3 and img.shape[2] == 1:
                    gray = img[:, :, 0]
                elif img.ndim == 3 and img.shape[2] == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                elif img.ndim == 3 and img.shape[2] == 4:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                else:
                    raise ValueError(f"Unsupported image shape: {img.shape}")
                if gray.dtype != np.uint8:
                    gray = gray.astype(np.uint8)
                return gray

            left_in = _to_gray_uint8(left_rgb)
            right_in = _to_gray_uint8(right_rgb)

        # Core inference (timed separately)
        t_infer_start = time.perf_counter()
        raw_disp = self._compute_disparity(left_in, right_in)
        t_infer_end = time.perf_counter()

        # Post-process: convert to float32 pixel disparity
        pred_disp = self._raw_to_float_disp(raw_disp)

        # Validity mask
        valid_pred = np.isfinite(pred_disp) & (pred_disp > 0)

        # Optional depth conversion
        pred_depth = None
        if calib is not None:
            pred_depth = calib.disp_to_depth(pred_disp)

        t_end = time.perf_counter()

        n_valid = int(np.sum(valid_pred))
        n_total = H * W
        density = 100.0 * n_valid / n_total if n_total > 0 else 0.0

        return StereoResult(
            pred_disp=pred_disp,
            valid_pred=valid_pred,
            pred_depth=pred_depth,
            img_size=(H, W),
            elapsed_s=t_infer_end - t_infer_start,
            total_elapsed_s=t_end - t0,
            config=self.config,
            density_pct=density,
            hw_info=self.get_hw_info(),
        )

    def _raw_to_float_disp(self, raw: np.ndarray) -> np.ndarray:
        """Convert raw algorithm output to float32 pixel disparity.

        Override in subclasses that produce fixed-point output.
        """
        return raw.astype(np.float32)
