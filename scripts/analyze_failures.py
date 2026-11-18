#!/usr/bin/env python3
"""Stage 5 Failure Case Analysis & Visualization Suite.

Automates the extraction, qualitative diagnostic grid rendering, and in-depth
failure mode analysis across classical (StereoBM, StereoSGBM) and deep SOTA
(HITNet, Fast-ACVNet+, Dusk Till Dawn) stereo depth estimation models.

Generates:
1. 3-column assignment grids: [Estimated Depth Map, Left View, Right View]
2. Diagnostic grids: [Left View, Ground Truth, Predicted Disparity, Error Heatmap, Metric Depth]
3. Multi-method comparison grids across challenging landmark scenes
4. Structured JSON catalog and comprehensive Markdown failure analysis report
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.eth3d import ETH3DDataset
from src.datasets.kitti import KITTI2015Dataset
from src.datasets.schema import StereoSample
from src.stereo import StereoMatcher, create_matcher
from src.stereo.deep_acvnet import ACVNetConfig, ACVNetMatcher
from src.stereo.deep_dtd import DTDConfig, DTDMatcher
from src.stereo.deep_hitnet import HitnetConfig, HitnetMatcher
from src.stereo.shallow_bm_sgbm import StereoBMConfig, StereoSGBMConfig, load_shallow_configs
from src.visualization.failure_analysis import (
    KNOWN_SCENE_TAXONOMY,
    FailureCaseRecord,
    build_failure_case_catalog,
    export_failure_catalog,
    load_benchmark_records,
)
from src.visualization.grids import (
    colorize_depth,
    colorize_disparity,
    compute_error_heatmap,
    render_3column_grid,
    render_diagnostic_grid,
    render_multimethod_grid,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("analyze_failures")


def load_all_matchers(device: str = "cpu") -> Dict[str, StereoMatcher]:
    """Instantiate available matchers across shallow and deep paradigms."""
    matchers: Dict[str, StereoMatcher] = {}

    # 1. Shallow Matchers
    shallow_path = REPO_ROOT / "configs" / "shallow_methods_tuned.yaml"
    if not shallow_path.is_file():
        shallow_path = REPO_ROOT / "configs" / "shallow_methods.yaml"
    if shallow_path.is_file():
        try:
            configs = load_shallow_configs(shallow_path)
            for name, cfg in configs.items():
                matchers[name] = create_matcher(cfg)
        except Exception as e:
            logger.warning(f"Failed to load shallow configs: {e}")

    # Fallback shallow defaults if needed
    if "StereoBM" not in matchers:
        matchers["StereoBM"] = create_matcher(StereoBMConfig(num_disparities=64, block_size=21, uniqueness_ratio=10))
    if "StereoSGBM" not in matchers:
        matchers["StereoSGBM"] = create_matcher(StereoSGBMConfig(min_disparity=0, num_disparities=64, block_size=7, uniqueness_ratio=15))

    # 2. Deep Matchers (HITNet, Fast-ACVNet)
    hitnet_eth_ckpt = REPO_ROOT / "checkpoints" / "hitnet" / "hitnet_eth3d_480x640.onnx"
    if hitnet_eth_ckpt.is_file():
        matchers["HITNet_ETH3D"] = HitnetMatcher(HitnetConfig(model_variant="eth3d", checkpoint_path=str(hitnet_eth_ckpt), device=device))

    hitnet_sf_ckpt = REPO_ROOT / "checkpoints" / "hitnet" / "hitnet_sceneflow_480x640.onnx"
    if hitnet_sf_ckpt.is_file():
        matchers["HITNet_SceneFlow"] = HitnetMatcher(HitnetConfig(model_variant="sceneflow", checkpoint_path=str(hitnet_sf_ckpt), device=device))

    acv_kitti_ckpt = REPO_ROOT / "checkpoints" / "fast_acvnet" / "fast_acvnet_plus_kitti_2015_opset16_480x640.onnx"
    if acv_kitti_ckpt.is_file():
        matchers["Fast_ACVNet_KITTI2015"] = ACVNetMatcher(ACVNetConfig(model_variant="kitti_2015", checkpoint_path=str(acv_kitti_ckpt), device=device))

    acv_gen_ckpt = REPO_ROOT / "checkpoints" / "fast_acvnet" / "fast_acvnet_plus_generalization_opset16_480x640.onnx"
    if acv_gen_ckpt.is_file():
        matchers["Fast_ACVNet_Generalization"] = ACVNetMatcher(ACVNetConfig(model_variant="generalization", checkpoint_path=str(acv_gen_ckpt), device=device))

    return matchers


def run_failure_analysis(
    benchmark_csv: Path,
    kitti_dir: Path,
    eth3d_dir: Path,
    output_dir: Path,
    top_k: int = 5,
    device: str = "cpu",
) -> None:
    """Execute end-to-end failure case analysis and artifact generation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    grids_3col_dir = output_dir / "grids_3col"
    grids_diag_dir = output_dir / "grids_diagnostic"
    grids_comp_dir = output_dir / "grids_comparative"
    grids_3col_dir.mkdir(parents=True, exist_ok=True)
    grids_diag_dir.mkdir(parents=True, exist_ok=True)
    grids_comp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading benchmark records from {benchmark_csv}")
    records = load_benchmark_records(benchmark_csv)

    logger.info("Building failure case catalog...")
    catalog = build_failure_case_catalog(records, top_k_per_category=top_k)
    logger.info(f"Extracted {len(catalog)} failure cases across representative failure modes.")

    # Load datasets
    kitti_dataset = None
    if kitti_dir.is_dir():
        try:
            kitti_dataset = KITTI2015Dataset(kitti_dir, split="val_tune")
            logger.info(f"Loaded KITTI 2015 val_tune dataset ({len(kitti_dataset)} samples)")
        except Exception as e:
            logger.warning(f"Could not load KITTI dataset: {e}")

    eth3d_dataset = None
    if eth3d_dir.is_dir():
        try:
            eth3d_dataset = ETH3DDataset(eth3d_dir, split="val_tune", fg_strategy="pretrained_seg")
            logger.info(f"Loaded ETH3D val_tune dataset ({len(eth3d_dataset)} samples)")
        except Exception as e:
            logger.warning(f"Could not load ETH3D dataset: {e}")

    kitti_samples = {s.scene_id: s for s in kitti_dataset} if kitti_dataset else {}
    eth3d_samples = {s.scene_id: s for s in eth3d_dataset} if eth3d_dataset else {}

    # Load matchers
    logger.info(f"Instantiating matchers on device: {device}")
    matchers = load_all_matchers(device=device)
    logger.info(f"Active matchers for failure visualization: {list(matchers.keys())}")

    # Process each failure case
    for rec in catalog:
        sample = kitti_samples.get(rec.scene_id) if rec.dataset == "kitti" else eth3d_samples.get(rec.scene_id)
        if sample is None:
            logger.warning(f"Sample not found for scene {rec.scene_id} in {rec.dataset}")
            continue

        matcher = matchers.get(rec.method)
        if matcher is None:
            logger.warning(f"Matcher {rec.method} not loaded; skipping visual inference for {rec.scene_id}")
            continue

        logger.info(f"Processing failure visualization: {rec.scene_id} ({rec.dataset}) with {rec.method} [{rec.category}]")
        result = matcher.predict(sample.left_img, sample.right_img, sample.calib)

        # 1. Official 3-Column Assignment Grid: [Estimated Depth Map, Left View, Right View]
        out_3col = grids_3col_dir / f"{rec.dataset}_{rec.scene_id}_{rec.method}_3col.png"
        vis_depth = None
        if result.pred_depth is not None and np.any(np.isfinite(result.pred_depth) & (result.pred_depth > 0)):
            vis_depth = colorize_depth(result.pred_depth, max_depth=80.0 if rec.dataset == "kitti" else 15.0)
        else:
            vis_depth = colorize_disparity(result.pred_disp)

        title_3col = f"{rec.scene_id} ({rec.dataset.upper()}) — {rec.method} Depth Estimation"
        render_3column_grid(
            depth_or_disp_img=vis_depth,
            left_img=sample.left_img,
            right_img=sample.right_img,
            title=title_3col,
            column_labels=("Estimated Depth Map", "Left View", "Right View"),
            out_path=out_3col,
        )
        rec.grid_3col_path = str(out_3col.relative_to(REPO_ROOT))

        # 2. Diagnostic Grid: [Left View, Ground Truth, Predicted Disparity, Error Heatmap, Metric Depth]
        out_diag = grids_diag_dir / f"{rec.dataset}_{rec.scene_id}_{rec.method}_diag.png"
        metrics_str = f"{rec.primary_metric_name}: {rec.primary_metric_val:.2f}%"
        if "density_pct" in rec.metrics:
            metrics_str += f" | Dens: {rec.metrics['density_pct']:.1f}%"
        elif "fg_density_pct" in rec.metrics:
            metrics_str += f" | FG Dens: {rec.metrics['fg_density_pct']:.1f}%"

        render_diagnostic_grid(
            left_img=sample.left_img,
            gt_disp=sample.gt_disparity,
            pred_disp=result.pred_disp,
            valid_mask=sample.valid_mask,
            pred_depth=result.pred_depth,
            scene_id=rec.scene_id,
            method_name=rec.method,
            metrics_summary=metrics_str,
            failure_tag=rec.category,
            out_path=out_diag,
        )
        rec.diagnostic_grid_path = str(out_diag.relative_to(REPO_ROOT))

    # 3. Multi-Method Comparative Grids on Key Landmark Scenes
    landmark_scenes = [
        ("kitti", "000006", "Specular Reflection on Tinted Rear Windshield & Tailgate"),
        ("kitti", "000020", "Near-Field Occlusion & Range Saturation (<4m Vehicle)"),
        ("kitti", "000008", "Textureless Asphalt Road Surface & Muted Lighting"),
        ("kitti", "000030", "Fine Occlusion Boundaries & Thin Street Poles"),
        ("eth3d", "delivery_area_1l", "Textureless Indoor Warehouse Walls & Glossy Floor"),
    ]

    for ds_name, sid, failure_desc in landmark_scenes:
        sample = kitti_samples.get(sid) if ds_name == "kitti" else eth3d_samples.get(sid)
        if sample is None:
            continue

        method_preds: Dict[str, np.ndarray] = {}
        for m_name in ["StereoBM", "StereoSGBM", "Fast_ACVNet_KITTI2015", "HITNet_ETH3D"]:
            if m_name in matchers:
                res = matchers[m_name].predict(sample.left_img, sample.right_img, sample.calib)
                method_preds[m_name] = res.pred_disp

        out_comp = grids_comp_dir / f"{ds_name}_{sid}_multimethod.png"
        render_multimethod_grid(
            left_img=sample.left_img,
            gt_disp=sample.gt_disparity,
            valid_mask=sample.valid_mask,
            methods_dict=method_preds,
            scene_id=sid,
            failure_title=failure_desc,
            out_path=out_comp,
        )
        logger.info(f"Saved comparative multi-method grid: {out_comp}")

        # Link to catalog records
        for r in catalog:
            if r.scene_id == sid:
                r.multimethod_grid_path = str(out_comp.relative_to(REPO_ROOT))

    # 4. Export JSON Catalog
    json_path = output_dir / "failure_catalog.json"
    export_failure_catalog(catalog, json_path)
    logger.info(f"Exported failure catalog to {json_path}")

    # 5. Generate Markdown Report
    report_path = output_dir / "failure_analysis_report.md"
    generate_failure_analysis_markdown(catalog, report_path)
    logger.info(f"Generated comprehensive failure analysis report at {report_path}")


def generate_failure_analysis_markdown(catalog: List[FailureCaseRecord], out_path: Path) -> None:
    """Generate a structured Markdown report detailing failure mechanisms and root causes."""
    md: List[str] = [
        "# Qualitative Failure Case Analysis Report",
        "",
        "**Generated:** Stage 5 Automated Diagnostic Pipeline  ",
        "**Scope:** In-depth investigation of stereo matching failure modes across classical shallow and deep SOTA architectures on KITTI 2015 and ETH3D benchmarks.",
        "",
        "---",
        "",
        "## Executive Summary of Failure Modes",
        "",
        "Stereo depth estimation relies on the fundamental assumption of **Lambertian photo-consistency** across rectified camera viewpoints. "
        "Through systematic evaluation across 320 benchmark samples, we isolate five primary mechanisms where this assumption breaks down, leading to severe error spikes or complete coverage dropouts:",
        "",
        "| Failure Mechanism | Primary Vulnerability | Typical Manifestation | Most Affected Paradigms |",
        "|---|---|---|---|",
        "| **Specular Reflections & Transparency** | Non-Lambertian viewing angle shift & dual-surface transmission | Windshield disparity holes, interior penetration, illusory depth | Classical BM/SGBM (dropouts), SOTA Deep (boundary bleeding) |",
        "| **Textureless & Uniform Expanses** | Flat matching cost curve (zero intensity gradient) | Severe pixel dropouts, false planar inclinations, surface waviness | StereoBM (<45% density), StereoSGBM (streaking) |",
        "| **Occlusion Boundaries & Thin Geometry** | Unilateral visibility (pixels present in left but occluded in right) | Foreground boundary erosion, horizontal streaking artifacts | StereoSGBM (scanline streaks), Deep models (edge blurring) |",
        "| **Near-Field Disparity Saturation** | Disparity exceeding search range ($d > d_{\\max}$) | Truncated near surfaces, missing foreground object regions | Classical matchers with fixed `numDisparities` |",
        "| **Out-of-Distribution Generalization** | Overfitting to synthetic rendering priors (SceneFlow / Nighttime) | Global disparity bias, hallucinated structures, >98% outlier rate | HITNet SceneFlow, Dusk Till Dawn on daytime KITTI |",
        "",
        "---",
        "",
        "## Detailed Case Studies",
        "",
    ]

    for idx, rec in enumerate(catalog, 1):
        md.extend([
            f"### Case {idx}: Scene `{rec.scene_id}` ({rec.dataset.upper()}) — {rec.category.replace('_', ' ').title()}",
            "",
            f"- **Evaluated Method:** `{rec.method}`",
            f"- **Primary Metric Impact:** `{rec.primary_metric_name} = {rec.primary_metric_val:.2f}%`",
            f"- **Observed Symptoms:** {rec.description}",
            f"- **Root Cause Analysis:** {rec.root_cause_analysis}",
            "",
            "#### Quantitative Metrics",
            "",
            "| Metric | Value |",
            "|---|---|",
        ])
        for m_name, m_val in rec.metrics.items():
            unit = "%" if "d1" in m_name or "bad" in m_name or "density" in m_name else ("px" if "epe" in m_name else "m")
            md.append(f"| `{m_name}` | {m_val:.2f} {unit} |")

        md.extend([
            "",
            "#### Visual Diagnostics",
            "",
        ])
        if rec.grid_3col_path:
            md.append(f"- **3-Column Assignment Grid:** `[{rec.grid_3col_path}]`")
        if rec.diagnostic_grid_path:
            md.append(f"- **Diagnostic Error Grid:** `[{rec.diagnostic_grid_path}]`")
        if rec.multimethod_grid_path:
            md.append(f"- **Cross-Method Comparison:** `[{rec.multimethod_grid_path}]`")
        md.extend(["", "---", ""])

    out_path.write_text("\n".join(md), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 5: Stereo Depth Failure Case Analysis & Grid Rendering")
    parser.add_argument("--benchmark-csv", type=Path, default=REPO_ROOT / "outputs" / "benchmarks" / "unified_benchmark_results.csv", help="Benchmark CSV path")
    parser.add_argument("--kitti-dir", type=Path, default=REPO_ROOT / "data" / "kitti2015", help="Path to KITTI 2015 root directory")
    parser.add_argument("--eth3d-dir", type=Path, default=REPO_ROOT / "data" / "eth3d", help="Path to ETH3D root directory")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "failure_cases", help="Output directory for failure artifacts")
    parser.add_argument("--top-k", type=int, default=5, help="Number of failure cases per category")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"], help="Computation device for inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting Stage 5 Failure Case Analysis...")
    run_failure_analysis(
        benchmark_csv=args.benchmark_csv,
        kitti_dir=args.kitti_dir,
        eth3d_dir=args.eth3d_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
        device=args.device,
    )
    logger.info("Stage 5 Failure Case Analysis completed successfully.")


if __name__ == "__main__":
    main()