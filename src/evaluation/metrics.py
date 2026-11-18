"""Standardized disparity and metric-depth evaluation metrics.

Metrics implemented:
- Disparity (pixels):
    * EPE (End-Point Error): mean absolute difference in pixels.
    * Bad-1.0 / Bad-2.0 / Bad-3.0: percentage of valid pixels with absolute error > threshold.
    * D1-all (KITTI metric): percentage of valid pixels with error > 3 px AND > 5% of GT disparity.
    * Density (%): percentage of GT valid pixels that have a valid prediction (pred > 0).
- Metric Depth (metres, where Z = f * B / d):
    * RMSE: root mean square error in metres.
    * AbsRel: absolute relative error mean(|Z_pred - Z_gt| / Z_gt).
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np


def compute_disparity_metrics(
    pred_disp: np.ndarray,
    gt_disp: np.ndarray,
    gt_valid_mask: np.ndarray,
    dataset: str = "eth3d",
) -> Dict[str, float]:
    """Compute disparity metrics over mutually valid and GT-valid pixels.

    Provides a fair comparison between sparse shallow methods (which discard
    difficult pixels) and dense deep methods (which predict all pixels).

    Args:
        pred_disp:      ``(H, W) float32`` predicted disparity in pixels.
        gt_disp:        ``(H, W) float32`` ground-truth disparity in pixels.
        gt_valid_mask:  ``(H, W) bool`` mask of valid GT pixels.
        dataset:        ``"eth3d"`` or ``"kitti"`` to tailor key metrics.

    Returns:
        Dict of metrics:
        - density_pct: % of GT valid pixels where pred_disp is valid (> 0).
        - n_gt: Count of valid GT pixels.
        - n_mutual: Count of mutually valid pixels.

        Mutually valid metrics (computed only where pred > 0 and GT is valid):
        - epe / mutual_epe: End-Point Error in pixels.
        - bad_1_0 / mutual_bad_1_0: % error > 1.0 px.
        - bad_2_0 / mutual_bad_2_0: % error > 2.0 px.
        - bad_3_0 / mutual_bad_3_0: % error > 3.0 px.
        - d1_all / mutual_d1_all: % error > 3.0 px and > 5% (KITTI criterion).

        All-valid GT metrics (official benchmark standard: missing predictions count as outliers):
        - all_bad_1_0: % of GT pixels where error > 1.0 px OR prediction is missing.
        - all_bad_2_0: % of GT pixels where error > 2.0 px OR prediction is missing.
        - all_bad_3_0: % of GT pixels where error > 3.0 px OR prediction is missing.
        - all_d1: % of GT pixels where error > 3.0 px & > 5% OR prediction is missing.
        - all_epe: EPE across all GT pixels (missing predictions penalized with 0-disparity error).
    """
    n_gt = int(np.sum(gt_valid_mask))
    if n_gt == 0:
        return {
            "density_pct": 0.0,
            "n_gt": 0,
            "n_mutual": 0,
            "epe": float("nan"),
            "mutual_epe": float("nan"),
            "bad_1_0": float("nan"),
            "mutual_bad_1_0": float("nan"),
            "bad_2_0": float("nan"),
            "mutual_bad_2_0": float("nan"),
            "bad_3_0": float("nan"),
            "mutual_bad_3_0": float("nan"),
            "d1_all": float("nan"),
            "mutual_d1_all": float("nan"),
            "all_bad_1_0": float("nan"),
            "all_bad_2_0": float("nan"),
            "all_bad_3_0": float("nan"),
            "all_d1": float("nan"),
            "all_epe": float("nan"),
        }

    pred_valid = np.isfinite(pred_disp) & (pred_disp > 0)
    mutual_mask = gt_valid_mask & pred_valid
    n_mutual = int(np.sum(mutual_mask))
    n_invalid_pred = n_gt - n_mutual

    density_pct = 100.0 * n_mutual / n_gt

    if n_mutual == 0:
        # No mutual pixels: all GT pixels are outliers
        all_epe = float(np.mean(gt_disp[gt_valid_mask]))
        return {
            "density_pct": density_pct,
            "n_gt": n_gt,
            "n_mutual": 0,
            "epe": float("nan"),
            "mutual_epe": float("nan"),
            "bad_1_0": 100.0,
            "mutual_bad_1_0": 100.0,
            "bad_2_0": 100.0,
            "mutual_bad_2_0": 100.0,
            "bad_3_0": 100.0,
            "mutual_bad_3_0": 100.0,
            "d1_all": 100.0,
            "mutual_d1_all": 100.0,
            "all_bad_1_0": 100.0,
            "all_bad_2_0": 100.0,
            "all_bad_3_0": 100.0,
            "all_d1": 100.0,
            "all_epe": all_epe,
        }

    diff = np.abs(pred_disp[mutual_mask] - gt_disp[mutual_mask])
    gt_vals = gt_disp[mutual_mask]

    # Mutually valid metrics
    epe = float(np.mean(diff))
    n_bad_1_0 = int(np.sum(diff > 1.0))
    n_bad_2_0 = int(np.sum(diff > 2.0))
    n_bad_3_0 = int(np.sum(diff > 3.0))

    rel_diff = diff / np.maximum(gt_vals, 1e-6)
    n_d1 = int(np.sum((diff > 3.0) & (rel_diff > 0.05)))

    bad_1_0 = 100.0 * n_bad_1_0 / n_mutual
    bad_2_0 = 100.0 * n_bad_2_0 / n_mutual
    bad_3_0 = 100.0 * n_bad_3_0 / n_mutual
    d1_all = 100.0 * n_d1 / n_mutual

    # All-valid GT metrics (missing predictions count as outliers)
    all_bad_1_0 = 100.0 * (n_bad_1_0 + n_invalid_pred) / n_gt
    all_bad_2_0 = 100.0 * (n_bad_2_0 + n_invalid_pred) / n_gt
    all_bad_3_0 = 100.0 * (n_bad_3_0 + n_invalid_pred) / n_gt
    all_d1 = 100.0 * (n_d1 + n_invalid_pred) / n_gt

    # All-valid EPE: 0-disparity penalty for missing predictions
    missing_gt_sum = float(np.sum(gt_disp[gt_valid_mask & ~pred_valid]))
    all_epe = float((np.sum(diff) + missing_gt_sum) / n_gt)

    return {
        "density_pct": density_pct,
        "n_gt": n_gt,
        "n_mutual": n_mutual,
        "epe": epe,
        "mutual_epe": epe,
        "bad_1_0": bad_1_0,
        "mutual_bad_1_0": bad_1_0,
        "bad_2_0": bad_2_0,
        "mutual_bad_2_0": bad_2_0,
        "bad_3_0": bad_3_0,
        "mutual_bad_3_0": bad_3_0,
        "d1_all": d1_all,
        "mutual_d1_all": d1_all,
        "all_bad_1_0": all_bad_1_0,
        "all_bad_2_0": all_bad_2_0,
        "all_bad_3_0": all_bad_3_0,
        "all_d1": all_d1,
        "all_epe": all_epe,
    }


def compute_depth_metrics(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    gt_valid_mask: np.ndarray,
    min_depth: float = 0.5,
    max_depth: float = 80.0,
) -> Dict[str, float]:
    """Compute metric depth metrics within bounded valid depth range.

    Args:
        pred_depth:     ``(H, W) float32`` predicted depth in metres.
        gt_depth:       ``(H, W) float32`` ground-truth depth in metres.
        gt_valid_mask:  ``(H, W) bool`` mask of valid GT depth.
        min_depth:      Minimum depth in metres to avoid zero-division / infinity.
        max_depth:      Maximum depth in metres to bound background error.

    Returns:
        Dict with 'rmse' (m) and 'abs_rel' (ratio).
    """
    valid_range = (
        gt_valid_mask
        & np.isfinite(gt_depth)
        & (gt_depth >= min_depth)
        & (gt_depth <= max_depth)
    )

    n_valid_range = int(np.sum(valid_range))
    pred_valid = np.isfinite(pred_depth) & (pred_depth > 0)
    eval_mask = valid_range & pred_valid
    n_eval = int(np.sum(eval_mask))

    density_pct = 100.0 * n_eval / n_valid_range if n_valid_range > 0 else 0.0

    if n_eval == 0:
        return {
            "rmse": float("nan"),
            "abs_rel": float("nan"),
            "eval_pixels": 0,
            "valid_range_pixels": n_valid_range,
            "density_pct": density_pct,
        }

    p = pred_depth[eval_mask]
    g = gt_depth[eval_mask]

    diff = np.abs(p - g)
    rmse = float(np.sqrt(np.mean(diff**2)))
    abs_rel = float(np.mean(diff / g))

    return {
        "rmse": rmse,
        "abs_rel": abs_rel,
        "eval_pixels": n_eval,
        "valid_range_pixels": n_valid_range,
        "density_pct": density_pct,
    }
