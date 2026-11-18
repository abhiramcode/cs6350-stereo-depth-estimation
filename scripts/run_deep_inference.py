#!/usr/bin/env python3
"""Run deep learning stereo matching inference on datasets or custom pairs.

Supports:
- Google HITNet (CVPR 2021) ONNX models (ETH3D, SceneFlow)
- Fast-ACVNet+ (TPAMI 2023 / CVPR 2022) ONNX models (Generalization, KITTI 2015)
- Dusk Till Dawn (ICRA 2024) PyTorch models (DINO ViT-S/8 + UniMatch)

Logs pure inference latency, end-to-end latency, prediction density,
and quantitative metrics against ground truth (when available).

Usage:
    uv run python scripts/run_deep_inference.py --dataset eth3d --root data/eth3d --split val_tune --method HITNet_ETH3D
    uv run python scripts/run_deep_inference.py --dataset eth3d --root data/eth3d --split val_tune --method all --save-viz
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.datasets import (
    Calibration,
    CustomPairDataset,
    ETH3DDataset,
    KITTI2015Dataset,
    StereoSample,
)
from src.evaluation.metrics import compute_depth_metrics, compute_disparity_metrics
from src.stereo import (
    MatcherConfig,
    StereoMatcher,
    StereoResult,
    create_matcher,
    load_shallow_configs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deep_inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deep learning stereo inference benchmark")
    parser.add_argument("--dataset", type=str, default="eth3d", choices=["eth3d", "kitti", "custom"],
                        help="Dataset name")
    parser.add_argument("--root", type=str, default="data/eth3d",
                        help="Path to dataset root")
    parser.add_argument("--split", type=str, default="val_tune",
                        help="Dataset split (e.g., val_tune, eval_test, train_all, test_visual)")
    parser.add_argument("--method", type=str, default="all",
                        help="Deep method to run (e.g. HITNet_ETH3D, Fast_ACVNet_Generalization, DuskTillDawn_Fast, or 'all')")
    parser.add_argument("--config", type=str, default="configs/deep_methods.yaml",
                        help="Path to deep methods YAML config")
    parser.add_argument("--output-dir", type=str, default="outputs/deep_inference",
                        help="Directory to save logs and outputs")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process (for quick test)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup iterations on the first sample")
    parser.add_argument("--save-maps", action="store_true",
                        help="Save disparity and depth .npy arrays")
    parser.add_argument("--save-viz", action="store_true",
                        help="Save side-by-side visual comparison PNGs")
    parser.add_argument("--f-px", type=float, default=None,
                        help="Focal length in pixels (for custom pairs without calibration metadata)")
    parser.add_argument("--baseline-m", type=float, default=None,
                        help="Stereo baseline in metres (for custom pairs without calibration metadata)")
    parser.add_argument("--fg-strategy", type=str, default="unavailable",
                        choices=["unavailable", "pretrained_seg", "geometric_proxy"],
                        help="Foreground extraction strategy (e.g. pretrained_seg using Mask R-CNN)")
    return parser.parse_args()


def load_dataset(dataset_name: str, root: str, split: str, f_px: Optional[float] = None, baseline_m: Optional[float] = None, fg_strategy: str = "unavailable"):
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")

    if dataset_name == "eth3d":
        return ETH3DDataset(root=root_path, split=split, fg_strategy=fg_strategy)
    elif dataset_name == "kitti":
        return KITTI2015Dataset(root=root_path, split=split)
    elif dataset_name == "custom":
        calib = None
        if f_px is not None and baseline_m is not None:
            calib = Calibration(f_px=f_px, baseline_m=baseline_m)
        return CustomPairDataset(root=root_path, calib=calib, fg_strategy=fg_strategy)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def save_visualization(
    out_path: Path,
    sample: StereoSample,
    result: StereoResult,
) -> None:
    """Save a side-by-side comparison plot: Left image, predicted disparity, predicted depth."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    has_depth = result.pred_depth is not None
    n_cols = 3 if has_depth else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), squeeze=False)

    # 1. Left View
    axes[0, 0].imshow(sample.left)
    axes[0, 0].set_title(f"Left View: {sample.scene_id}")
    axes[0, 0].axis("off")

    # 2. Predicted Disparity
    disp_vis = result.pred_disp.copy()
    disp_vis[~result.valid_pred] = np.nan
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="black")
    im1 = axes[0, 1].imshow(disp_vis, cmap=cmap)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Disparity (px)")
    axes[0, 1].set_title(
        f"Pred Disp ({result.config.method_name})\n"
        f"Density: {result.density_pct:.1f}% | Infer: {result.elapsed_s*1000:.1f}ms"
    )
    axes[0, 1].axis("off")

    # 3. Predicted Depth (if available)
    if has_depth:
        depth_vis = result.pred_depth.copy()
        valid_depth = result.valid_pred & np.isfinite(depth_vis) & (depth_vis > 0)
        vmax = np.percentile(depth_vis[valid_depth], 95) if np.any(valid_depth) else 20.0
        depth_vis[~valid_depth] = np.nan
        cmap_depth = plt.get_cmap("magma_r").copy()
        cmap_depth.set_bad(color="black")
        im2 = axes[0, 2].imshow(depth_vis, cmap=cmap_depth, vmin=0, vmax=vmax)
        fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04, label="Depth (m)")
        axes[0, 2].set_title(f"Pred Depth (m, max={vmax:.1f}m)")
        axes[0, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_inference_for_matcher(
    matcher: StereoMatcher,
    dataset,
    args: argparse.Namespace,
    method_name: str,
) -> List[Dict[str, Any]]:
    logger.info("Running deep inference for method: %s", method_name)
    samples = list(dataset)
    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    if not samples:
        logger.warning("No samples found in dataset split '%s'", args.split)
        return []

    # Warm-up on the first sample
    if args.warmup > 0:
        first = samples[0]
        logger.info("Warming up (%d runs) on first sample %s...", args.warmup, first.scene_id)
        for _ in range(args.warmup):
            matcher.predict(first.left, first.right, calib=first.calib)

    records: List[Dict[str, Any]] = []

    out_dir = Path(args.output_dir) / method_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        # Run inference
        result = matcher.predict(sample.left, sample.right, calib=sample.calib)

        record: Dict[str, Any] = {
            "scene_id": sample.scene_id,
            "dataset": sample.dataset,
            "method": method_name,
            "height": sample.img_size[0],
            "width": sample.img_size[1],
            "infer_time_ms": round(result.elapsed_s * 1000.0, 2),
            "total_time_ms": round(result.total_elapsed_s * 1000.0, 2),
            "fps": round(1.0 / result.elapsed_s, 2) if result.elapsed_s > 0 else 0.0,
            "density_pct": round(result.density_pct, 2),
            "has_calib": sample.has_calib,
            "has_gt": sample.has_gt,
        }

        # Quantitative evaluation if ground truth is available
        if sample.has_gt and sample.gt_disp is not None:
            # Overall disparity metrics
            disp_metrics = compute_disparity_metrics(
                pred_disp=result.pred_disp,
                gt_disp=sample.gt_disp,
                gt_valid_mask=sample.valid_mask,
                dataset=sample.dataset,
            )
            for k, v in disp_metrics.items():
                record[f"disp_{k}"] = round(v, 4) if np.isfinite(v) else None

            # Foreground-specific metrics if fg_mask is present
            if sample.fg_mask is not None and np.any(sample.fg_mask & sample.valid_mask):
                fg_mask_valid = sample.valid_mask & sample.fg_mask
                fg_metrics = compute_disparity_metrics(
                    pred_disp=result.pred_disp,
                    gt_disp=sample.gt_disp,
                    gt_valid_mask=fg_mask_valid,
                    dataset=sample.dataset,
                )
                record["fg_provenance"] = sample.fg_mask_provenance
                for k, v in fg_metrics.items():
                    record[f"fg_disp_{k}"] = round(v, 4) if np.isfinite(v) else None

            # Metric depth metrics if calib is available
            if sample.has_calib and sample.calib is not None and result.pred_depth is not None:
                gt_depth = sample.calib.disp_to_depth(sample.gt_disp)
                depth_metrics = compute_depth_metrics(
                    pred_depth=result.pred_depth,
                    gt_depth=gt_depth,
                    gt_valid_mask=sample.valid_mask,
                )
                for k, v in depth_metrics.items():
                    record[f"depth_{k}"] = round(v, 4) if np.isfinite(v) else None

        records.append(record)

        # Optional map saving
        if args.save_maps:
            maps_dir = out_dir / "maps"
            maps_dir.mkdir(parents=True, exist_ok=True)
            np.save(maps_dir / f"{sample.scene_id}_disp.npy", result.pred_disp)
            if result.pred_depth is not None:
                np.save(maps_dir / f"{sample.scene_id}_depth.npy", result.pred_depth)

        # Optional visualization saving
        if args.save_viz:
            viz_path = out_dir / "visualizations" / f"{sample.scene_id}_{method_name}.png"
            save_visualization(viz_path, sample, result)

        logger.info(
            "[%d/%d] %s (%s): infer=%.1fms, density=%.1f%%%s",
            i + 1,
            len(samples),
            sample.scene_id,
            method_name,
            record["infer_time_ms"],
            record["density_pct"],
            f", epe={record.get('disp_epe')}" if "disp_epe" in record else "",
        )

    return records


def print_summary_table(all_records: List[Dict[str, Any]]) -> None:
    """Print an aggregate summary table per method."""
    from collections import defaultdict
    by_method = defaultdict(list)
    for r in all_records:
        by_method[r["method"]].append(r)

    print("\n" + "=" * 90)
    print(f"{'Method':<28} | {'Samples':<7} | {'Infer (ms)':<10} | {'FPS':<6} | {'Density %':<9} | {'EPE (px)':<8} | {'Bad-1.0 %':<9} | {'Bad-2.0 %':<9}")
    print("-" * 90)

    for method, recs in by_method.items():
        n = len(recs)
        avg_infer = np.mean([r["infer_time_ms"] for r in recs])
        avg_fps = np.mean([r["fps"] for r in recs])
        avg_density = np.mean([r["density_pct"] for r in recs])

        epes = [r["disp_epe"] for r in recs if r.get("disp_epe") is not None]
        bad1s = [r["disp_bad_1_0"] for r in recs if r.get("disp_bad_1_0") is not None]
        bad2s = [r["disp_bad_2_0"] for r in recs if r.get("disp_bad_2_0") is not None]

        epe_str = f"{np.mean(epes):.2f}" if epes else "N/A"
        bad1_str = f"{np.mean(bad1s):.1f}%" if bad1s else "N/A"
        bad2_str = f"{np.mean(bad2s):.1f}%" if bad2s else "N/A"

        print(f"{method:<28} | {n:<7} | {avg_infer:<10.1f} | {avg_fps:<6.1f} | {avg_density:<9.1f} | {epe_str:<8} | {bad1_str:<9} | {bad2_str:<9}")
    print("=" * 90 + "\n")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        args.dataset,
        args.root,
        args.split,
        f_px=args.f_px,
        baseline_m=args.baseline_m,
        fg_strategy=args.fg_strategy,
    )
    configs = load_shallow_configs(args.config)

    methods_to_run = []
    if args.method == "all":
        methods_to_run = list(configs.keys())
    elif "," in args.method:
        methods_to_run = [m.strip() for m in args.method.split(",") if m.strip()]
    elif args.method in configs:
        methods_to_run = [args.method]
    else:
        methods_to_run = [args.method]

    all_records: List[Dict[str, Any]] = []

    for m_name in methods_to_run:
        if m_name in configs:
            cfg = configs[m_name]
            matcher = create_matcher(cfg)
        else:
            matcher = create_matcher(m_name)

        recs = run_inference_for_matcher(matcher, dataset, args, m_name)
        all_records.extend(recs)

    if not all_records:
        logger.error("No inference records produced.")
        return 1

    # Print summary
    print_summary_table(all_records)

    # Save CSV (merge with existing records if present so prior runs are not lost)
    csv_path = out_dir / "deep_inference_results.csv"
    existing_records: List[Dict[str, Any]] = []
    if csv_path.is_file():
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Don't duplicate if same scene_id and method
                    if not any(r["scene_id"] == row["scene_id"] and r["method"] == row["method"] for r in all_records):
                        existing_records.append(row)
        except Exception as e:
            logger.warning("Could not read existing %s: %s", csv_path, e)

    combined_records = existing_records + all_records

    keys = list(combined_records[0].keys())
    all_keys = set()
    for r in combined_records:
        all_keys.update(r.keys())
    fieldnames = [k for k in keys if k in all_keys] + [k for k in sorted(all_keys) if k not in keys]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_records)
    logger.info("Saved per-frame results to %s", csv_path)

    # Save JSON
    json_path = out_dir / "deep_inference_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)
    logger.info("Saved full results JSON to %s", json_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
