"""OpenCV StereoBM and StereoSGBM shallow stereo matchers.

Both methods operate on rectified grayscale image pairs and produce
disparity maps in OpenCV's 16-bit fixed-point format (raw / 16.0 = pixels).

The ``StereoSGBM`` matcher optionally applies a **WLS (Weighted Least
Squares) disparity filter** via ``cv2.ximgproc`` for hole-filling and
edge-preserving smoothing.

Configuration is fully externalised — no interactive trackbars.

Reference:
- OpenCV tutorial: https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html
- LearnOpenCV: https://learnopencv.com/depth-perception-using-stereo-camera-python-c/
- REFERENCES.md: Stage 2 agent adaptation instructions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from src.stereo.base import MatcherConfig, StereoMatcher


# ===================================================================== #
#  Configuration dataclasses
# ===================================================================== #

@dataclass
class StereoBMConfig(MatcherConfig):
    """Configuration for OpenCV ``StereoBM``.

    All parameters map directly to ``cv2.StereoBM_create`` and setters.
    ``numDisparities`` must be divisible by 16.
    ``blockSize`` must be odd and >= 5.
    """

    method_name: str = "StereoBM"

    # Core matching
    numDisparities: int = 128       # max disparity range (multiple of 16)
    blockSize: int = 15             # SAD window size (odd, >= 5)

    # Post-filtering
    preFilterType: int = 1          # 0=PREFILTER_NORMALIZED_RESPONSE, 1=PREFILTER_XSOBEL
    preFilterSize: int = 9          # must be odd, 5..255
    preFilterCap: int = 31          # 1..63

    # Validation
    textureThreshold: int = 10      # minimum texture for valid block
    uniquenessRatio: int = 15       # margin (%) over second-best match
    minDisparity: int = 0           # minimum disparity value

    # Speckle filtering
    speckleWindowSize: int = 100    # max speckle blob size
    speckleRange: int = 32          # max disparity jump within speckle


@dataclass
class StereoSGBMConfig(MatcherConfig):
    """Configuration for OpenCV ``StereoSGBM``.

    ``mode`` selects the semi-global path count:
    - ``cv2.STEREO_SGBM_MODE_SGBM``      = 5 paths
    - ``cv2.STEREO_SGBM_MODE_HH``        = 8 paths (slower, more accurate)
    - ``cv2.STEREO_SGBM_MODE_SGBM_3WAY`` = 3 paths (fastest)
    """

    method_name: str = "StereoSGBM"

    # Core matching
    numDisparities: int = 128
    blockSize: int = 5              # SADWindowSize (odd, 1..11 typical)
    minDisparity: int = 0

    # Penalties (P1 < P2 required)
    P1: Optional[int] = None        # computed from blockSize if None
    P2: Optional[int] = None

    # Validation
    uniquenessRatio: int = 10
    disp12MaxDiff: int = 1          # left-right consistency (-1 = off)

    # Speckle
    speckleWindowSize: int = 100
    speckleRange: int = 32

    # Mode
    mode: int = 3   # cv2.STEREO_SGBM_MODE_SGBM_3WAY

    # WLS filter (requires cv2.ximgproc)
    use_wls_filter: bool = False
    wls_lambda: float = 8000.0
    wls_sigma: float = 1.5

    preFilterCap: int = 63


# ===================================================================== #
#  StereoBM Matcher
# ===================================================================== #

class StereoBMMatcher(StereoMatcher):
    """OpenCV Block Matching (BM) stereo matcher.

    This is the simplest and fastest shallow baseline.
    Operates on grayscale images only.
    """

    def __init__(self, config: Optional[StereoBMConfig] = None) -> None:
        config = config or StereoBMConfig()
        super().__init__(config)
        self._cfg: StereoBMConfig = config
        self._matcher = self._build_matcher()

    def _build_matcher(self) -> cv2.StereoBM:
        cfg = self._cfg
        bm = cv2.StereoBM.create(
            numDisparities=cfg.numDisparities,
            blockSize=cfg.blockSize,
        )
        bm.setPreFilterType(cfg.preFilterType)
        bm.setPreFilterSize(cfg.preFilterSize)
        bm.setPreFilterCap(cfg.preFilterCap)
        bm.setTextureThreshold(cfg.textureThreshold)
        bm.setUniquenessRatio(cfg.uniquenessRatio)
        bm.setMinDisparity(cfg.minDisparity)
        bm.setSpeckleWindowSize(cfg.speckleWindowSize)
        bm.setSpeckleRange(cfg.speckleRange)
        return bm

    def _compute_disparity(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
    ) -> np.ndarray:
        """Compute raw fixed-point disparity (int16, ×16)."""
        return self._matcher.compute(left_gray, right_gray)

    def _raw_to_float_disp(self, raw: np.ndarray) -> np.ndarray:
        """Convert OpenCV 16-bit fixed-point to float32 pixel disparity.

        OpenCV stores disparity × 16 in int16.  Division by 16.0 gives
        the true pixel disparity.  Negative values → invalid.
        """
        disp = raw.astype(np.float32) / 16.0
        disp[disp < 0] = 0.0  # mark invalid
        return disp


# ===================================================================== #
#  StereoSGBM Matcher
# ===================================================================== #

class StereoSGBMMatcher(StereoMatcher):
    """OpenCV Semi-Global Block Matching (SGBM) stereo matcher.

    Optionally applies a WLS (Weighted Least Squares) disparity filter
    for edge-preserving smoothing when ``use_wls_filter=True``.
    """

    def __init__(self, config: Optional[StereoSGBMConfig] = None) -> None:
        config = config or StereoSGBMConfig()
        super().__init__(config)
        self._cfg: StereoSGBMConfig = config
        self._left_matcher = self._build_matcher()
        self._right_matcher: Optional[cv2.StereoMatcher] = None
        self._wls_filter = None

        if config.use_wls_filter:
            self._setup_wls()

    def _build_matcher(self) -> cv2.StereoSGBM:
        cfg = self._cfg
        cn = 1  # grayscale channels

        # Auto-compute penalties if not set
        p1 = cfg.P1 if cfg.P1 is not None else 8 * cn * cfg.blockSize * cfg.blockSize
        p2 = cfg.P2 if cfg.P2 is not None else 32 * cn * cfg.blockSize * cfg.blockSize

        sgbm = cv2.StereoSGBM.create(
            minDisparity=cfg.minDisparity,
            numDisparities=cfg.numDisparities,
            blockSize=cfg.blockSize,
            P1=p1,
            P2=p2,
            disp12MaxDiff=cfg.disp12MaxDiff,
            uniquenessRatio=cfg.uniquenessRatio,
            speckleWindowSize=cfg.speckleWindowSize,
            speckleRange=cfg.speckleRange,
            preFilterCap=cfg.preFilterCap,
            mode=cfg.mode,
        )
        return sgbm

    def _setup_wls(self) -> None:
        """Set up the WLS disparity filter (requires cv2.ximgproc)."""
        try:
            self._right_matcher = cv2.ximgproc.createRightMatcher(self._left_matcher)
            self._wls_filter = cv2.ximgproc.createDisparityWLSFilter(
                matcher_left=self._left_matcher
            )
            self._wls_filter.setLambda(self._cfg.wls_lambda)
            self._wls_filter.setSigmaColor(self._cfg.wls_sigma)
        except AttributeError:
            import logging
            logging.getLogger(__name__).warning(
                "cv2.ximgproc not available — WLS filter disabled. "
                "Install opencv-contrib-python for WLS support."
            )
            self._cfg.use_wls_filter = False
            self._right_matcher = None
            self._wls_filter = None

    def _compute_disparity(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
    ) -> np.ndarray:
        """Compute raw fixed-point disparity, optionally WLS-filtered."""
        left_disp = self._left_matcher.compute(left_gray, right_gray)

        if self._wls_filter is not None and self._right_matcher is not None:
            right_disp = self._right_matcher.compute(right_gray, left_gray)
            filtered = self._wls_filter.filter(
                disparity_map_left=left_disp,
                left_view=left_gray,
                disparity_map_right=right_disp,
            )
            return filtered

        return left_disp

    def _raw_to_float_disp(self, raw: np.ndarray) -> np.ndarray:
        """Convert OpenCV 16-bit fixed-point to float32 pixel disparity."""
        disp = raw.astype(np.float32) / 16.0
        disp[disp < 0] = 0.0
        return disp


# ===================================================================== #
#  Factory & YAML Config Loader
# ===================================================================== #

def create_matcher(
    method_or_config: str | MatcherConfig,
    **overrides,
) -> StereoMatcher:
    """Create a stereo matcher by name or config object with optional overrides.

    Args:
        method_or_config: One of ``"StereoBM"``, ``"StereoSGBM"``,
            ``"StereoSGBM_WLS"``, or an instance of ``MatcherConfig``.
        **overrides: Keyword arguments passed to the config dataclass.

    Returns:
        An instantiated ``StereoMatcher``.
    """
    if isinstance(method_or_config, StereoBMConfig):
        if overrides:
            for k, v in overrides.items():
                if hasattr(method_or_config, k):
                    setattr(method_or_config, k, v)
        return StereoBMMatcher(method_or_config)

    if isinstance(method_or_config, StereoSGBMConfig):
        if overrides:
            for k, v in overrides.items():
                if hasattr(method_or_config, k):
                    setattr(method_or_config, k, v)
        return StereoSGBMMatcher(method_or_config)

    # Check for HITNet
    from src.stereo.deep_hitnet import HitnetConfig, HitnetMatcher
    if isinstance(method_or_config, HitnetConfig):
        if overrides:
            for k, v in overrides.items():
                if hasattr(method_or_config, k):
                    setattr(method_or_config, k, v)
        return HitnetMatcher(method_or_config)

    # Check for Fast-ACVNet
    from src.stereo.deep_acvnet import ACVNetConfig, ACVNetMatcher
    if isinstance(method_or_config, ACVNetConfig):
        if overrides:
            for k, v in overrides.items():
                if hasattr(method_or_config, k):
                    setattr(method_or_config, k, v)
        return ACVNetMatcher(method_or_config)

    # Check for Dusk Till Dawn (DTD)
    from src.stereo.deep_dtd import DTDConfig, DTDMatcher
    if isinstance(method_or_config, DTDConfig):
        if overrides:
            for k, v in overrides.items():
                if hasattr(method_or_config, k):
                    setattr(method_or_config, k, v)
        return DTDMatcher(method_or_config)

    method = str(method_or_config).strip().lower()
    if method in ("stereobm", "bm", "stereo_vision", "stereovision"):
        cfg = StereoBMConfig.from_dict(overrides)
        return StereoBMMatcher(cfg)
    elif method in ("stereosgbm", "sgbm"):
        cfg = StereoSGBMConfig.from_dict(overrides)
        return StereoSGBMMatcher(cfg)
    elif method in ("stereosgbm_wls", "sgbm_wls"):
        overrides.setdefault("use_wls_filter", True)
        cfg = StereoSGBMConfig.from_dict(overrides)
        return StereoSGBMMatcher(cfg)
    elif method in ("hitnet", "hitnet_eth3d", "hitnet_sceneflow", "hitnet_middlebury"):
        if "eth3d" in method:
            overrides.setdefault("model_variant", "eth3d")
        elif "sceneflow" in method:
            overrides.setdefault("model_variant", "sceneflow")
        elif "middlebury" in method:
            overrides.setdefault("model_variant", "middlebury")
        cfg = HitnetConfig.from_dict(overrides)
        return HitnetMatcher(cfg)
    elif "acvnet" in method or "fast_acvnet" in method or "fast-acvnet" in method:
        if "kitti" in method:
            overrides.setdefault("model_variant", "kitti_2015")
        elif "sceneflow" in method:
            overrides.setdefault("model_variant", "sceneflow")
        else:
            overrides.setdefault("model_variant", "generalization")
        cfg = ACVNetConfig.from_dict(overrides)
        return ACVNetMatcher(cfg)
    elif "dtd" in method or "dusktilldawn" in method or "dusk_till_dawn" in method or "dusk" in method:
        cfg = DTDConfig.from_dict(overrides)
        return DTDMatcher(cfg)
    else:
        raise ValueError(
            f"Unknown method '{method_or_config}'. "
            "Supported: 'StereoBM', 'StereoSGBM', 'StereoSGBM_WLS', 'HITNet', 'Fast-ACVNet', 'DuskTillDawn'."
        )


def load_shallow_configs(config_path: str) -> dict[str, MatcherConfig]:
    """Load matcher configs from a YAML file.

    Returns:
        Dictionary mapping method names to their respective MatcherConfig.
    """
    import yaml
    from pathlib import Path

    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    configs = {}
    for name, params in data.items():
        if not isinstance(params, dict):
            continue
        nl = name.lower()
        if "bm" in nl and "sgbm" not in nl:
            cfg = StereoBMConfig.from_dict(params)
            cfg.method_name = name
            configs[name] = cfg
        elif "sgbm" in nl:
            cfg = StereoSGBMConfig.from_dict(params)
            cfg.method_name = name
            configs[name] = cfg
        elif "hitnet" in nl:
            from src.stereo.deep_hitnet import HitnetConfig
            cfg = HitnetConfig.from_dict(params)
            cfg.method_name = name
            configs[name] = cfg
        elif "acvnet" in nl or "fast_acvnet" in nl:
            from src.stereo.deep_acvnet import ACVNetConfig
            cfg = ACVNetConfig.from_dict(params)
            cfg.method_name = name
            configs[name] = cfg
        elif "dtd" in nl or "dusktilldawn" in nl or "dusk" in nl:
            from src.stereo.deep_dtd import DTDConfig
            cfg = DTDConfig.from_dict(params)
            cfg.method_name = name
            configs[name] = cfg
    return configs
