"""Unified dataset loaders for KITTI 2015, ETH3D, and custom stereo pairs."""

from src.datasets.schema import Calibration, StereoSample
from src.datasets.kitti import KITTI2015Dataset
from src.datasets.eth3d import ETH3DDataset
from src.datasets.custom_pair import CustomPairDataset

__all__ = [
    "Calibration",
    "StereoSample",
    "KITTI2015Dataset",
    "ETH3DDataset",
    "CustomPairDataset",
]
