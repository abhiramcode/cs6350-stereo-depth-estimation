"""Visual grid rendering functions for stereo depth estimation.

Generates 3-column assignment grids [Estimated Depth Map, Left View, Right View],
diagnostic error heatmap grids, and multi-method comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def colorize_disparity(
    disp: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    cmap_name: str = "plasma",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    invalid_color: Tuple[int, int, int] = (20, 20, 25),
) -> np.ndarray:
    """Convert a float disparity map into an RGB uint8 image with masked invalids.

    Args:
        disp: Disparity map (H, W).
        valid_mask: Optional boolean mask (H, W) where True indicates valid pixels.
            If None, considers disp > 0 and isfinite as valid.
        cmap_name: Matplotlib colormap name (default "plasma").
        vmin: Minimum disparity for normalization. If None, uses 1st percentile of valid.
        vmax: Maximum disparity for normalization. If None, uses 99th percentile of valid.
        invalid_color: RGB uint8 color for invalid pixels (default dark charcoal).

    Returns:
        RGB uint8 array of shape (H, W, 3).
    """
    if valid_mask is None:
        valid_mask = np.isfinite(disp) & (disp > 0)
    else:
        valid_mask = valid_mask & np.isfinite(disp) & (disp > 0)

    h, w = disp.shape[:2]
    rgb = np.full((h, w, 3), invalid_color, dtype=np.uint8)

    if not np.any(valid_mask):
        return rgb

    valid_vals = disp[valid_mask]
    v_min = vmin if vmin is not None else float(np.percentile(valid_vals, 1))
    v_max = vmax if vmax is not None else float(np.percentile(valid_vals, 99))
    if v_max <= v_min:
        v_max = v_min + 1.0

    norm_vals = np.clip((valid_vals - v_min) / (v_max - v_min), 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    colored_valid = (cmap(norm_vals)[:, :3] * 255.0).astype(np.uint8)
    rgb[valid_mask] = colored_valid
    return rgb


def colorize_depth(
    depth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    cmap_name: str = "turbo",
    min_depth: float = 0.5,
    max_depth: float = 80.0,
    invalid_color: Tuple[int, int, int] = (20, 20, 25),
) -> np.ndarray:
    """Convert a metric depth map (meters) into an RGB uint8 image.

    Args:
        depth: Metric depth map in meters (H, W).
        valid_mask: Optional boolean mask. If None, considers depth in [min_depth, max_depth].
        cmap_name: Colormap name (default "turbo").
        min_depth: Minimum valid depth in meters (default 0.5m).
        max_depth: Maximum valid depth in meters (default 80.0m).
        invalid_color: RGB uint8 color for invalid/out-of-range pixels.

    Returns:
        RGB uint8 array of shape (H, W, 3).
    """
    in_range = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    if valid_mask is None:
        mask = in_range
    else:
        mask = valid_mask & in_range

    h, w = depth.shape[:2]
    rgb = np.full((h, w, 3), invalid_color, dtype=np.uint8)

    if not np.any(mask):
        return rgb

    vals = depth[mask]
    norm_vals = np.clip((vals - min_depth) / (max_depth - min_depth), 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    colored = (cmap(norm_vals)[:, :3] * 255.0).astype(np.uint8)
    rgb[mask] = colored
    return rgb


def compute_error_heatmap(
    pred_disp: np.ndarray,
    gt_disp: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute absolute disparity error and binary outlier mask.

    Args:
        pred_disp: Predicted disparity (H, W).
        gt_disp: Ground-truth disparity (H, W).
        valid_mask: Ground-truth valid mask (H, W).
        threshold: Outlier error threshold in pixels (default 3.0 px for KITTI).

    Returns:
        Tuple of:
            - abs_error: float32 absolute error (H, W), with NaN for non-evaluated pixels.
            - outlier_mask: boolean mask (H, W) where True = outlier error (> threshold).
            - colored_error: RGB uint8 visualization of error using magma/hot colormap.
    """
    h, w = pred_disp.shape[:2]
    mutual_valid = valid_mask & np.isfinite(pred_disp) & (pred_disp > 0) & (gt_disp > 0)

    abs_error = np.full((h, w), np.nan, dtype=np.float32)
    abs_error[mutual_valid] = np.abs(pred_disp[mutual_valid] - gt_disp[mutual_valid])

    # Outliers on mutual pixels
    outlier_mask = np.zeros((h, w), dtype=bool)
    outlier_mask[mutual_valid] = abs_error[mutual_valid] > threshold

    # Colored error map
    colored_error = np.zeros((h, w, 3), dtype=np.uint8)
    if np.any(mutual_valid):
        # Clip error visualization to 0..10 px for high dynamic range
        vis_err = np.clip(abs_error[mutual_valid], 0.0, 10.0) / 10.0
        cmap = plt.get_cmap("magma")
        colored_error[mutual_valid] = (cmap(vis_err)[:, :3] * 255.0).astype(np.uint8)

    # Missing prediction pixels that are valid in GT are rendered in purple/blue
    missing_mask = valid_mask & (~mutual_valid)
    colored_error[missing_mask] = np.array([120, 30, 160], dtype=np.uint8)

    return abs_error, outlier_mask, colored_error


def render_3column_grid(
    depth_or_disp_img: np.ndarray,
    left_img: np.ndarray,
    right_img: np.ndarray,
    title: str = "",
    column_labels: Tuple[str, str, str] = ("Estimated Depth Map", "Left View", "Right View"),
    out_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Render the official 3-column assignment visual grid.

    Replicates Section 4 ("Expected Input and Output") of the problem statement:
    Column 1: Estimated Depth Map
    Column 2: Left View
    Column 3: Right View

    Args:
        depth_or_disp_img: RGB uint8 (H, W, 3) or 2D disparity/depth map.
        left_img: Left camera image RGB uint8 (H, W, 3).
        right_img: Right camera image RGB uint8 (H, W, 3).
        title: Super-title for the figure.
        column_labels: Labels for each column.
        out_path: Optional file path to save the generated figure.
        dpi: Output resolution.

    Returns:
        The generated matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)

    # 1. Column 1: Estimated Depth Map
    if depth_or_disp_img.ndim == 2:
        vis_col1 = colorize_disparity(depth_or_disp_img)
    else:
        vis_col1 = depth_or_disp_img
    axes[0, 0].imshow(vis_col1)
    axes[0, 0].set_title(column_labels[0], fontsize=13, fontweight="bold", pad=8)
    axes[0, 0].axis("off")

    # 2. Column 2: Left View
    axes[0, 1].imshow(left_img)
    axes[0, 1].set_title(column_labels[1], fontsize=13, fontweight="bold", pad=8)
    axes[0, 1].axis("off")

    # 3. Column 3: Right View
    axes[0, 2].imshow(right_img)
    axes[0, 2].set_title(column_labels[2], fontsize=13, fontweight="bold", pad=8)
    axes[0, 2].axis("off")

    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    plt.tight_layout()

    if out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


def render_diagnostic_grid(
    left_img: np.ndarray,
    gt_disp: Optional[np.ndarray],
    pred_disp: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    pred_depth: Optional[np.ndarray] = None,
    scene_id: str = "",
    method_name: str = "",
    metrics_summary: str = "",
    failure_tag: str = "",
    out_path: Optional[Union[str, Path]] = None,
    dpi: int = 140,
) -> plt.Figure:
    """Render a comprehensive diagnostic grid with error heatmap and metrics.

    Layout:
    [ Left View | Ground-Truth Disparity | Predicted Disparity | Error Heatmap ]
    (Optionally includes 5th column for Metric Depth if available).

    Args:
        left_img: Left image RGB uint8.
        gt_disp: Ground-truth disparity array or None.
        pred_disp: Predicted disparity array.
        valid_mask: GT validity mask.
        pred_depth: Optional metric depth map.
        scene_id: Name/ID of scene.
        method_name: Name of evaluated matcher.
        metrics_summary: Short string of metrics (e.g. "EPE: 1.2px | D1: 5.4%").
        failure_tag: Failure mode tag (e.g. "specular_reflection", "textureless").
        out_path: Optional output path.
        dpi: Output DPI.

    Returns:
        The matplotlib Figure.
    """
    has_gt = gt_disp is not None and valid_mask is not None and np.any(valid_mask)
    has_depth = pred_depth is not None and np.any(np.isfinite(pred_depth) & (pred_depth > 0))

    n_cols = 4 if not has_depth else 5
    fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 3.8), squeeze=False)

    # 1. Left View
    axes[0, 0].imshow(left_img)
    tag_str = f" [{failure_tag.upper()}]" if failure_tag else ""
    axes[0, 0].set_title(f"Left View: {scene_id}{tag_str}", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # 2. GT Disparity
    if has_gt:
        gt_vis = np.where(valid_mask, gt_disp, np.nan)
        im1 = axes[0, 1].imshow(gt_vis, cmap="plasma")
        axes[0, 1].set_title("Ground-Truth Disparity", fontsize=11, fontweight="bold")
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Disparity (px)")
    else:
        axes[0, 1].text(0.5, 0.5, "GT Unavailable", ha="center", va="center", transform=axes[0, 1].transAxes)
        axes[0, 1].set_title("Ground Truth", fontsize=11)
    axes[0, 1].axis("off")

    # 3. Predicted Disparity
    pred_vis = np.where(pred_disp > 0, pred_disp, np.nan)
    im2 = axes[0, 2].imshow(pred_vis, cmap="plasma")
    axes[0, 2].set_title(f"{method_name} Disparity", fontsize=11, fontweight="bold")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04, label="Disparity (px)")
    axes[0, 2].axis("off")

    # 4. Error Heatmap
    if has_gt:
        _, _, error_rgb = compute_error_heatmap(pred_disp, gt_disp, valid_mask)
        axes[0, 3].imshow(error_rgb)
        axes[0, 3].set_title("Error Heatmap (|pred-gt|)", fontsize=11, fontweight="bold")
    else:
        axes[0, 3].text(0.5, 0.5, "No GT for Error", ha="center", va="center", transform=axes[0, 3].transAxes)
        axes[0, 3].set_title("Error Heatmap", fontsize=11)
    axes[0, 3].axis("off")

    # 5. Metric Depth (if available)
    if has_depth:
        depth_vis = np.where(
            np.isfinite(pred_depth) & (pred_depth > 0) & (pred_depth <= 80.0),
            pred_depth,
            np.nan,
        )
        im4 = axes[0, 4].imshow(depth_vis, cmap="turbo")
        axes[0, 4].set_title(f"{method_name} Depth (m)", fontsize=11, fontweight="bold")
        plt.colorbar(im4, ax=axes[0, 4], fraction=0.046, pad=0.04, label="Depth (m)")
        axes[0, 4].axis("off")

    if metrics_summary:
        fig.suptitle(f"{scene_id} — {method_name} | {metrics_summary}", fontsize=12, y=0.98)

    plt.tight_layout()

    if out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


def render_multimethod_grid(
    left_img: np.ndarray,
    gt_disp: Optional[np.ndarray],
    valid_mask: Optional[np.ndarray],
    methods_dict: dict[str, np.ndarray],
    scene_id: str = "",
    failure_title: str = "",
    out_path: Optional[Union[str, Path]] = None,
    dpi: int = 130,
) -> plt.Figure:
    """Render a cross-method comparison grid on a single scene.

    Compares Ground Truth and multiple methods (e.g. StereoBM, StereoSGBM,
    Fast-ACVNet+, HITNet) side by side.

    Args:
        left_img: Left image RGB uint8.
        gt_disp: Ground-truth disparity or None.
        valid_mask: Ground-truth valid mask or None.
        methods_dict: Mapping of method_name -> predicted disparity array (H, W).
        scene_id: Scene ID.
        failure_title: Sub-title describing the failure case mechanism.
        out_path: Optional output path.
        dpi: Output DPI.

    Returns:
        The matplotlib Figure.
    """
    total_panels = 1 + (1 if gt_disp is not None else 0) + len(methods_dict)
    n_cols = min(total_panels, 4)
    n_rows = (total_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows), squeeze=False)
    flat_axes = axes.flatten()

    # Panel 0: Left View
    flat_axes[0].imshow(left_img)
    flat_axes[0].set_title(f"Left View: {scene_id}", fontsize=10, fontweight="bold")
    flat_axes[0].axis("off")

    idx = 1
    # Panel 1: Ground Truth (if available)
    if gt_disp is not None and valid_mask is not None:
        gt_vis = np.where(valid_mask, gt_disp, np.nan)
        im = flat_axes[idx].imshow(gt_vis, cmap="plasma")
        flat_axes[idx].set_title("Ground-Truth Disparity", fontsize=10, fontweight="bold")
        flat_axes[idx].axis("off")
        idx += 1

    # Method Panels
    for m_name, pred in methods_dict.items():
        if idx >= len(flat_axes):
            break
        pred_vis = np.where(pred > 0, pred, np.nan)
        density = float(np.mean(pred > 0) * 100.0)
        im = flat_axes[idx].imshow(pred_vis, cmap="plasma")
        flat_axes[idx].set_title(f"{m_name} ({density:.0f}% dens)", fontsize=10, fontweight="bold")
        flat_axes[idx].axis("off")
        idx += 1

    # Turn off any remaining unused axes
    for j in range(idx, len(flat_axes)):
        flat_axes[j].axis("off")

    if failure_title:
        fig.suptitle(f"{scene_id}: {failure_title}", fontsize=12, fontweight="bold", y=0.99)

    plt.tight_layout()

    if out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig