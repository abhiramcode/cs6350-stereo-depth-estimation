"""Stereo matching implementations."""

from src.stereo.base import MatcherConfig, StereoMatcher, StereoResult
from src.stereo.deep_acvnet import ACVNetConfig, ACVNetMatcher
from src.stereo.deep_dtd import DTDConfig, DTDMatcher
from src.stereo.deep_hitnet import HitnetConfig, HitnetMatcher
from src.stereo.shallow_bm_sgbm import (
    StereoBMConfig,
    StereoBMMatcher,
    StereoSGBMConfig,
    StereoSGBMMatcher,
    create_matcher,
    load_shallow_configs,
)

__all__ = [
    "StereoMatcher",
    "StereoResult",
    "MatcherConfig",
    "StereoBMConfig",
    "StereoSGBMConfig",
    "StereoBMMatcher",
    "StereoSGBMMatcher",
    "HitnetConfig",
    "HitnetMatcher",
    "ACVNetConfig",
    "ACVNetMatcher",
    "DTDConfig",
    "DTDMatcher",
    "create_matcher",
    "load_shallow_configs",
]
