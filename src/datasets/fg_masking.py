"""Foreground masking utilities.

Provides a unified interface for generating foreground (object-of-interest)
masks with explicit provenance tracking.

Strategies:
1. ``semantic_gt``       — Ground-truth object labels (e.g. KITTI ``obj_map``).
2. ``pretrained_seg``    — Pre-trained instance/semantic segmentation model.
3. ``geometric_proxy``   — Median-disparity thresholding (NOT semantic!).
4. ``unavailable``       — No mask can be generated.

**Design decision:** Stage 1 exposes the interface and geometric proxy.
The pre-trained segmentation wrapper (Mask R-CNN / DeepLabV3) will be
integrated in Stage 4/5 when evaluation metrics require it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.datasets.schema import FgMaskProvenance


def geometric_proxy_mask(
    gt_disparity: np.ndarray,
    valid_mask: np.ndarray,
    quantile: float = 0.5,
) -> Tuple[np.ndarray, FgMaskProvenance]:
    """Generate a foreground mask by thresholding disparity above the median.

    Pixels with disparity above the median of *valid* disparities are
    considered "foreground" — i.e. closer to the camera than the scene
    median.

    .. warning::
        This is a **geometric heuristic**, not a semantic object mask.
        It will include ground plane and walls that happen to be close.
        The provenance is always ``"geometric_proxy"``.

    Args:
        gt_disparity: Float32 disparity map (pixels).
        valid_mask:   Boolean validity mask.
        quantile:     Quantile threshold (default 0.5 = median).

    Returns:
        (fg_mask, provenance): Boolean mask and provenance tag.
    """
    valid_disps = gt_disparity[valid_mask]
    if valid_disps.size == 0:
        return np.zeros_like(valid_mask), "geometric_proxy"

    threshold = np.quantile(valid_disps, quantile)
    fg_mask = valid_mask & (gt_disparity > threshold)
    return fg_mask, "geometric_proxy"


def pretrained_seg_mask(
    image: np.ndarray,
    model_name: str = "maskrcnn_resnet50_fpn",
    score_threshold: float = 0.5,
) -> Tuple[Optional[np.ndarray], FgMaskProvenance]:
    """Generate a foreground mask using a pre-trained segmentation model.

    .. note::
        This is a **placeholder** for Stage 4/5.  Currently returns
        ``(None, "unavailable")`` to signal that the capability is not
        yet integrated.

    Args:
        image:           (H, W, 3) uint8 RGB image.
        model_name:      Torchvision detection model name.
        score_threshold: Minimum detection confidence.

    Returns:
        (fg_mask, provenance): ``None`` and ``"unavailable"`` until
        the segmentation model is integrated.
    """
    # TODO(Stage 4/5): Import torchvision, run inference, merge instance masks.
    return None, "unavailable"
