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

import logging
from typing import Optional, Sequence, Tuple

import numpy as np

from src.datasets.schema import FgMaskProvenance

logger = logging.getLogger(__name__)

# Global cache for the segmentation model to avoid reloading weights per frame
_CACHED_SEG_MODEL = None
_CACHED_SEG_DEVICE = None


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


def _get_seg_model(model_name: str = "maskrcnn_resnet50_fpn", device: Optional[str] = None):
    """Get or instantiate cached torchvision instance segmentation model."""
    global _CACHED_SEG_MODEL, _CACHED_SEG_DEVICE
    import torch
    import torchvision

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if _CACHED_SEG_MODEL is not None and _CACHED_SEG_DEVICE == target_device:
        return _CACHED_SEG_MODEL, target_device

    try:
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_Weights,
            maskrcnn_resnet50_fpn,
        )

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn(weights=weights)
        model.to(target_device)
        model.eval()
        _CACHED_SEG_MODEL = model
        _CACHED_SEG_DEVICE = target_device
        logger.info("Loaded pre-trained %s on %s for foreground segmentation", model_name, target_device)
        return _CACHED_SEG_MODEL, target_device
    except Exception as exc:
        logger.error("Failed to load pre-trained segmentation model %s: %s", model_name, exc)
        return None, target_device


def pretrained_seg_mask(
    image: np.ndarray,
    model_name: str = "maskrcnn_resnet50_fpn",
    score_threshold: float = 0.5,
    category_filter: Optional[Sequence[int]] = None,
    device: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], FgMaskProvenance]:
    """Generate a foreground mask using a pre-trained instance segmentation model.

    Detects object instances using Mask R-CNN and creates a merged binary
    foreground mask with provenance ``"pretrained_seg"``.

    Args:
        image:           (H, W, 3) uint8 RGB image.
        model_name:      Torchvision detection model name (default: maskrcnn_resnet50_fpn).
        score_threshold: Minimum detection confidence (default: 0.5).
        category_filter: Optional list of COCO class IDs to restrict to.
        device:          Compute device ('cuda', 'cpu', or None for auto).

    Returns:
        (fg_mask, provenance): Boolean mask (H, W) and 'pretrained_seg',
        or (None, 'unavailable') if model loading fails.
    """
    if image is None or image.ndim != 3:
        return None, "unavailable"

    H, W = image.shape[:2]
    model, target_device = _get_seg_model(model_name=model_name, device=device)
    if model is None:
        return None, "unavailable"

    try:
        import torch

        # Prepare image tensor (C, H, W) normalized to [0, 1]
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(target_device)

        with torch.no_grad():
            preds = model([img_tensor])[0]

        scores = preds["scores"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()
        masks = preds["masks"].cpu().numpy()  # (N, 1, H, W)

        keep = scores >= score_threshold
        if category_filter is not None:
            keep = keep & np.isin(labels, category_filter)

        if not np.any(keep):
            # Valid inference ran, but no foreground instances met threshold
            return np.zeros((H, W), dtype=bool), "pretrained_seg"

        # Combine instance masks: instance is foreground if mask probability > 0.5
        selected_masks = masks[keep, 0, :, :] > 0.5  # (K, H, W) bool
        merged_mask = np.any(selected_masks, axis=0)

        return merged_mask, "pretrained_seg"
    except Exception as exc:
        logger.warning("Error running pretrained segmentation model: %s", exc)
        return None, "unavailable"
