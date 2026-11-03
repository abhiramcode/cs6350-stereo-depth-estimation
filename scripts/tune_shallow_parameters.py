#!/usr/bin/env python3
"""Hyperparameter tuning for shallow stereo matchers (StereoBM and StereoSGBM).

Systematically evaluates grid parameters on the `val_tune` split of the dataset,
ranks parameter combinations by disparity accuracy (EPE, Bad-1.0/2.0) and valid density,
and outputs an updated YAML configuration `configs/shallow_methods_tuned.yaml`.

Usage:
    uv run python scripts/tune_shallow_parameters.py --dataset eth3d --root data/eth3d
    uv run python scripts/tune_shallow_parameters.py --dataset eth3d --root data/eth3d --method StereoBM --quick
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.datasets import ETH3DDataset, KITTI2015Dataset
from src.evaluation.metrics import compute_disparity_metrics
from src.stereo import (
    StereoBMConfig,
    StereoBMMatcher,
    StereoSGBMConfig,
    StereoSGBMMatcher,
    create_matcher,
    load_shallow_configs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tune_shallow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune shallow stereo hyperparameters on val_tune split")
    parser.add_argument("--dataset", type=str, default="eth3d", choices=["eth3d", "kitti"],
                        help="Dataset name to tune on")
    parser.add_argument("--root", type=str, default="data/eth3d",
                        help="Path to dataset root")
    parser.add_argument("--split", type=str, default="val_tune",
                        help="Dataset split to tune on")
    parser.add_argument("--method", type=str, default="all", choices=["all", "StereoBM", "StereoSGBM"],
                        help="Method to tune")
    parser.add_argument("--max-val-scenes", type=int, default=None,
                        help="Limit number of validation scenes (default: all in split)")
    parser.add_argument("--output-config", type=str, default="configs/shallow_methods_tuned.yaml",
                        help="Path to output tuned YAML config")
    parser.add_argument("--quick", action="store_true",
                        help="Use a smaller grid for rapid tuning")
    return parser.parse_args()


def load_val_dataset(dataset_name: str, root: str, split: str, max_scenes: Optional[int] = None):
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")

    if dataset_name == "eth3d":
        ds = ETH3DDataset(root=root_path, split=split)
    elif dataset_name == "kitti":
        ds = KITTI2015Dataset(root=root_path, split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    samples = [s for s in ds if s.has_gt]
    if max_scenes is not None:
        samples = samples[:max_scenes]
    return samples


def evaluate_config(matcher, samples) -> Dict[str, float]:
    """Evaluate a matcher on all provided samples and return mean metrics."""
    epes, bad1s, bad2s, densities, latencies = [], [], [], [], []

    for s in samples:
        res = matcher.predict(s.left, s.right, calib=s.calib)
        latencies.append(res.elapsed_s * 1000.0)

        metrics = compute_disparity_metrics(
            pred_disp=res.pred_disp,
            gt_disp=s.gt_disp,
            gt_valid_mask=s.valid_mask,
            dataset=s.dataset,
        )

        densities.append(metrics["density_pct"])
        if np.isfinite(metrics["epe"]):
            epes.append(metrics["epe"])
        if np.isfinite(metrics["bad_1_0"]):
            bad1s.append(metrics["bad_1_0"])
        if np.isfinite(metrics["bad_2_0"]):
            bad2s.append(metrics["bad_2_0"])

    mean_epe = float(np.mean(epes)) if epes else float("inf")
    mean_bad1 = float(np.mean(bad1s)) if bad1s else 100.0
    mean_bad2 = float(np.mean(bad2s)) if bad2s else 100.0
    mean_density = float(np.mean(densities)) if densities else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0

    # Composite penalty: prioritize low Bad-1.0 while penalizing low density
    # Lower is better
    score = mean_bad1 + 0.3 * (100.0 - mean_density)

    return {
        "epe": round(mean_epe, 3),
        "bad_1_0": round(mean_bad1, 2),
        "bad_2_0": round(mean_bad2, 2),
        "density_pct": round(mean_density, 2),
        "latency_ms": round(mean_lat, 2),
        "composite_score": round(score, 3),
    }


def tune_stereobm(samples, quick: bool = False) -> Tuple[StereoBMConfig, Dict[str, Any]]:
    logger.info("--- Tuning StereoBM ---")
    if quick:
        num_disp_opts = [64, 128]
        block_size_opts = [9, 15]
        uniqueness_opts = [5, 15]
        texture_opts = [10]
    else:
        num_disp_opts = [64, 128, 192]
        block_size_opts = [9, 15, 21]
        uniqueness_opts = [5, 10, 15]
        texture_opts = [5, 10, 20]

    grid = list(itertools.product(num_disp_opts, block_size_opts, uniqueness_opts, texture_opts))
    logger.info("Evaluating %d StereoBM parameter combinations on %d scenes...", len(grid), len(samples))

    best_score = float("inf")
    best_cfg: Optional[StereoBMConfig] = None
    best_metrics: Dict[str, Any] = {}

    for nd, bs, uq, tex in grid:
        cfg = StereoBMConfig(
            numDisparities=nd,
            blockSize=bs,
            uniquenessRatio=uq,
            textureThreshold=tex,
        )
        matcher = StereoBMMatcher(cfg)
        metrics = evaluate_config(matcher, samples)

        if metrics["composite_score"] < best_score:
            best_score = metrics["composite_score"]
            best_cfg = cfg
            best_metrics = metrics
            logger.info(
                "  New best: numDisp=%d, bs=%d, uq=%d, tex=%d -> Bad-1=%.1f%%, Dens=%.1f%%, EPE=%.2fpx (Score: %.2f)",
                nd, bs, uq, tex, metrics["bad_1_0"], metrics["density_pct"], metrics["epe"], best_score,
            )

    return best_cfg, best_metrics


def tune_stereosgbm(samples, quick: bool = False) -> Tuple[StereoSGBMConfig, Dict[str, Any]]:
    logger.info("--- Tuning StereoSGBM ---")
    if quick:
        num_disp_opts = [64, 128]
        block_size_opts = [3, 5]
        uniqueness_opts = [5, 10]
        mode_opts = [3]  # MODE_SGBM_3WAY
    else:
        num_disp_opts = [64, 128, 192]
        block_size_opts = [3, 5, 7]
        uniqueness_opts = [5, 10, 15]
        mode_opts = [2, 3]  # MODE_HH (2) and MODE_SGBM_3WAY (3)

    grid = list(itertools.product(num_disp_opts, block_size_opts, uniqueness_opts, mode_opts))
    logger.info("Evaluating %d StereoSGBM parameter combinations on %d scenes...", len(grid), len(samples))

    best_score = float("inf")
    best_cfg: Optional[StereoSGBMConfig] = None
    best_metrics: Dict[str, Any] = {}

    for nd, bs, uq, mode in grid:
        cfg = StereoSGBMConfig(
            numDisparities=nd,
            blockSize=bs,
            uniquenessRatio=uq,
            mode=mode,
        )
        matcher = StereoSGBMMatcher(cfg)
        metrics = evaluate_config(matcher, samples)

        if metrics["composite_score"] < best_score:
            best_score = metrics["composite_score"]
            best_cfg = cfg
            best_metrics = metrics
            logger.info(
                "  New best: numDisp=%d, bs=%d, uq=%d, mode=%d -> Bad-1=%.1f%%, Dens=%.1f%%, EPE=%.2fpx (Score: %.2f)",
                nd, bs, uq, mode, metrics["bad_1_0"], metrics["density_pct"], metrics["epe"], best_score,
            )

    return best_cfg, best_metrics


def main() -> int:
    args = parse_args()
    samples = load_val_dataset(args.dataset, args.root, args.split, args.max_val_scenes)
    if not samples:
        logger.error("No valid labeled scenes found in %s/%s", args.dataset, args.split)
        return 1

    logger.info("Tuning on %d validation scenes from %s [%s]", len(samples), args.dataset, args.split)

    # Load baseline default configs for comparison
    default_configs = load_shallow_configs("configs/shallow_methods.yaml")
    tuned_dict = {}

    print("\n" + "=" * 90)
    print("STARTING HYPERPARAMETER TUNING")
    print("=" * 90)

    # 1. Tune StereoBM
    if args.method in ("all", "StereoBM"):
        def_bm = default_configs.get("StereoBM", StereoBMConfig())
        def_bm_metrics = evaluate_config(StereoBMMatcher(def_bm), samples)
        logger.info("StereoBM Default Metrics: %s", def_bm_metrics)

        best_bm_cfg, best_bm_metrics = tune_stereobm(samples, quick=args.quick)
        tuned_dict["StereoBM"] = best_bm_cfg.to_dict()

        print("\n[StereoBM Comparison]")
        print(f"  Default: numDisp={def_bm.numDisparities}, bs={def_bm.blockSize}, uq={def_bm.uniquenessRatio} "
              f"-> EPE={def_bm_metrics['epe']}px, Bad-1.0={def_bm_metrics['bad_1_0']}%, Density={def_bm_metrics['density_pct']}%, Infer={def_bm_metrics['latency_ms']}ms")
        print(f"  Tuned:   numDisp={best_bm_cfg.numDisparities}, bs={best_bm_cfg.blockSize}, uq={best_bm_cfg.uniquenessRatio} "
              f"-> EPE={best_bm_metrics['epe']}px, Bad-1.0={best_bm_metrics['bad_1_0']}%, Density={best_bm_metrics['density_pct']}%, Infer={best_bm_metrics['latency_ms']}ms")

    # 2. Tune StereoSGBM
    if args.method in ("all", "StereoSGBM"):
        def_sgbm = default_configs.get("StereoSGBM", StereoSGBMConfig())
        def_sgbm_metrics = evaluate_config(StereoSGBMMatcher(def_sgbm), samples)
        logger.info("StereoSGBM Default Metrics: %s", def_sgbm_metrics)

        best_sgbm_cfg, best_sgbm_metrics = tune_stereosgbm(samples, quick=args.quick)
        tuned_dict["StereoSGBM"] = best_sgbm_cfg.to_dict()

        # Also add a WLS version of the tuned config
        wls_dict = best_sgbm_cfg.to_dict()
        wls_dict["use_wls_filter"] = True
        wls_dict["method_name"] = "StereoSGBM_WLS"
        tuned_dict["StereoSGBM_WLS"] = wls_dict

        print("\n[StereoSGBM Comparison]")
        print(f"  Default: numDisp={def_sgbm.numDisparities}, bs={def_sgbm.blockSize}, uq={def_sgbm.uniquenessRatio}, mode={def_sgbm.mode} "
              f"-> EPE={def_sgbm_metrics['epe']}px, Bad-1.0={def_sgbm_metrics['bad_1_0']}%, Density={def_sgbm_metrics['density_pct']}%, Infer={def_sgbm_metrics['latency_ms']}ms")
        print(f"  Tuned:   numDisp={best_sgbm_cfg.numDisparities}, bs={best_sgbm_cfg.blockSize}, uq={best_sgbm_cfg.uniquenessRatio}, mode={best_sgbm_cfg.mode} "
              f"-> EPE={best_sgbm_metrics['epe']}px, Bad-1.0={best_sgbm_metrics['bad_1_0']}%, Density={best_sgbm_metrics['density_pct']}%, Infer={best_sgbm_metrics['latency_ms']}ms")

    # Save tuned YAML
    out_path = Path(args.output_config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(tuned_dict, f, sort_keys=False)
    logger.info("Saved tuned configurations to %s", out_path)

    print("\n" + "=" * 90)
    print(f"TUNING COMPLETE — Config saved to {out_path}")
    print("=" * 90 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
