"""Failure case diagnostic and categorization engine for stereo depth estimation.

Analyzes quantitative benchmark results from Stage 4, identifies extreme error
spikes and sparsity dropouts, maps them to physical/algorithmic root causes
(specularities, textureless regions, occlusions, domain transfer collapse),
and generates structured diagnostic reports.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FailureCaseRecord:
    """Detailed record of an isolated failure case."""

    scene_id: str
    dataset: str
    split: str
    method: str
    category: str
    primary_metric_name: str
    primary_metric_val: float
    description: str
    root_cause_analysis: str
    metrics: Dict[str, float] = field(default_factory=dict)
    grid_3col_path: Optional[str] = None
    diagnostic_grid_path: Optional[str] = None
    multimethod_grid_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Predefined domain knowledge mapping known challenging scene characteristics
KNOWN_SCENE_TAXONOMY: Dict[str, Dict[str, Any]] = {
    # KITTI 2015 Scenes
    "000006": {
        "category": "specular_reflection",
        "description": "Large foreground vehicle with highly reflective tinted rear windshield and glossy metallic tailgate.",
        "root_cause": (
            "Violates Lambertian photo-consistency assumption. Specular highlights shift across stereo viewpoints, "
            "causing local block matchers (StereoBM/SGBM) to fail uniqueness checks, while deep networks exhibit disparity "
            "bleeding across the transparent glass boundary."
        ),
    },
    "000020": {
        "category": "near_field_saturation",
        "description": "Massive near-field vehicle occupying >47,000 pixels at close range (<4 meters).",
        "root_cause": (
            "Near-field disparity exceeds typical classical search limits or causes severe occlusion on the left edge. "
            "StereoBM drops to 36.7% foreground density, resulting in an 80.4% all-valid D1 error, while Fast-ACVNet+ maintains "
            "complete 100% coverage with 2.8% D1 error."
        ),
    },
    "000008": {
        "category": "textureless_road_and_overcast",
        "description": "Expansive uniform asphalt roadway with overcast sky and muted shadow boundaries.",
        "root_cause": (
            "Zero intensity gradient on uniform asphalt produces a flat matching cost curve. Classical matchers reject "
            "most road pixels, while deep feature extractors without in-domain supervision hallucinate ripple artifacts."
        ),
    },
    "000030": {
        "category": "occlusion_and_thin_structures",
        "description": "Narrow street scene with distant cyclists, street poles, and roadside fences abutting parked vehicles.",
        "root_cause": (
            "Fine-scale geometry (bicycle spokes, slender poles) is smaller than the receptive field of matching windows "
            "and deep feature strides (e.g. patch size 8 or downsampling factor 4), leading to edge erosion and occlusion streaks."
        ),
    },
    "000004": {
        "category": "harsh_lighting_and_shadow",
        "description": "High dynamic range sunlight transition cast across tree foliage and road tarmac.",
        "root_cause": (
            "Deep shadow boundaries create artificial edges that dominate Census/SAD cost computation, misleading "
            "scanline optimization in StereoSGBM."
        ),
    },
    # ETH3D Scenes
    "delivery_area_1l": {
        "category": "textureless_indoor_walls_and_gloss",
        "description": "Large indoor warehouse with featureless white walls, smooth glossy concrete floor, and ceiling lamps.",
        "root_cause": (
            "Severe lack of texture across wide wall expanses creates matching ambiguity across tens of candidate pixels. "
            "StereoBM drops below 55% coverage. HITNet resolves planar surfaces via its learned planar tiles, outperforming SGBM."
        ),
    },
    "delivery_area_1s": {
        "category": "repetitive_patterns_and_occlusions",
        "description": "Delivery storage shelving with repeated metal grates and occluded pallet corners.",
        "root_cause": (
            "Periodic metal grids create periodic cost minima (aliasing), causing classical block matchers to match wrong "
            "grate cycles, resulting in Bad-1.0 error spikes."
        ),
    },
}


def load_benchmark_records(csv_path: Union[str, Path]) -> List[Dict[str, str]]:
    """Load per-sample benchmark results from CSV."""
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(f"Benchmark results CSV not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def extract_top_failures(
    records: List[Dict[str, str]],
    dataset: Optional[str] = None,
    method: Optional[str] = None,
    metric_key: str = "all_all_d1",
    top_k: int = 5,
    descending: bool = True,
) -> List[Dict[str, str]]:
    """Extract top-k records ranked by a specified evaluation metric."""
    filtered = []
    for r in records:
        if dataset and r.get("dataset") != dataset:
            continue
        if method and r.get("method") != method:
            continue
        if metric_key in r and r[metric_key] != "":
            try:
                val = float(r[metric_key])
                filtered.append((val, r))
            except ValueError:
                continue

    filtered.sort(key=lambda x: x[0], reverse=descending)
    return [item[1] for item in filtered[:top_k]]


def build_failure_case_catalog(
    records: List[Dict[str, str]],
    top_k_per_category: int = 3,
) -> List[FailureCaseRecord]:
    """Build a comprehensive, classified catalog of representative failure cases.

    Covers:
    1. Specular / reflective surfaces & transparency
    2. Textureless uniform expanses (road, walls)
    3. Occlusion boundaries and thin structures
    4. Classical near-field saturation / sparsity dropouts
    5. Cross-domain generalization collapse
    """
    catalog: List[FailureCaseRecord] = []
    seen_scenes = set()

    # 1. Classical Sparsity & Near-field Failure: StereoBM on KITTI
    bm_kitti = extract_top_failures(records, dataset="kitti", method="StereoBM", metric_key="fg_all_d1", top_k=top_k_per_category)
    for r in bm_kitti:
        sid = r["scene_id"]
        meta = KNOWN_SCENE_TAXONOMY.get(sid, {})
        cat = meta.get("category", "classical_sparsity_dropout")
        desc = meta.get("description", f"KITTI 2015 scene {sid} with high foreground matching difficulty.")
        rc = meta.get(
            "root_cause",
            "Classical block matcher fails uniqueness threshold in low-texture or occluded foreground regions, dropping valid pixels.",
        )
        catalog.append(
            FailureCaseRecord(
                scene_id=sid,
                dataset="kitti",
                split=r.get("split", "val_tune"),
                method="StereoBM",
                category=cat,
                primary_metric_name="fg_all_d1",
                primary_metric_val=float(r["fg_all_d1"]),
                description=desc,
                root_cause_analysis=rc,
                metrics={
                    "fg_all_d1": float(r.get("fg_all_d1", 0)),
                    "fg_mutual_d1": float(r.get("fg_mutual_d1_all", 0)),
                    "fg_density_pct": float(r.get("fg_density_pct", 0)),
                    "all_all_d1": float(r.get("all_all_d1", 0)),
                    "fg_depth_rmse": float(r.get("fg_depth_rmse", 0)),
                },
            )
        )
        seen_scenes.add((sid, "StereoBM"))

    # 2. StereoSGBM Streaking & Shadow Failure: StereoSGBM on KITTI
    sgbm_kitti = extract_top_failures(records, dataset="kitti", method="StereoSGBM", metric_key="all_all_d1", top_k=top_k_per_category)
    for r in sgbm_kitti:
        sid = r["scene_id"]
        if (sid, "StereoSGBM") in seen_scenes:
            continue
        meta = KNOWN_SCENE_TAXONOMY.get(sid, {})
        cat = meta.get("category", "semi_global_streaking_and_shadow")
        desc = meta.get("description", f"KITTI scene {sid} exhibiting scanline optimization artifacts.")
        rc = meta.get(
            "root_cause",
            "Dynamic programming along 8/5 paths suffers from path-cost accumulation asymmetry, producing horizontal streaks across shadow transitions.",
        )
        catalog.append(
            FailureCaseRecord(
                scene_id=sid,
                dataset="kitti",
                split=r.get("split", "val_tune"),
                method="StereoSGBM",
                category=cat,
                primary_metric_name="all_all_d1",
                primary_metric_val=float(r["all_all_d1"]),
                description=desc,
                root_cause_analysis=rc,
                metrics={
                    "all_all_d1": float(r.get("all_all_d1", 0)),
                    "all_mutual_d1": float(r.get("all_mutual_d1_all", 0)),
                    "density_pct": float(r.get("density_pct", 0)),
                    "fg_all_d1": float(r.get("fg_all_d1", 0)),
                    "depth_rmse": float(r.get("depth_rmse", 0)),
                },
            )
        )
        seen_scenes.add((sid, "StereoSGBM"))

    # 3. Deep Learning SOTA Outlier on KITTI: Fast-ACVNet-KITTI2015
    acv_kitti = extract_top_failures(records, dataset="kitti", method="Fast_ACVNet_KITTI2015", metric_key="all_d1_all", top_k=top_k_per_category)
    for r in acv_kitti:
        sid = r["scene_id"]
        if (sid, "Fast_ACVNet_KITTI2015") in seen_scenes:
            continue
        meta = KNOWN_SCENE_TAXONOMY.get(sid, {})
        cat = meta.get("category", "deep_feature_ambiguity")
        desc = meta.get("description", f"KITTI scene {sid} with complex roadway geometry and distant structures.")
        rc = meta.get(
            "root_cause",
            "Attention Concatenation Volume suppresses cost noise but smooths high-frequency disparity discontinuities at distant road boundaries.",
        )
        catalog.append(
            FailureCaseRecord(
                scene_id=sid,
                dataset="kitti",
                split=r.get("split", "val_tune"),
                method="Fast_ACVNet_KITTI2015",
                category=cat,
                primary_metric_name="all_d1_all",
                primary_metric_val=float(r["all_d1_all"]),
                description=desc,
                root_cause_analysis=rc,
                metrics={
                    "all_d1_all": float(r.get("all_d1_all", 0)),
                    "fg_d1_all": float(r.get("fg_d1_all", 0)),
                    "all_epe": float(r.get("all_epe", 0)),
                    "depth_rmse": float(r.get("depth_rmse", 0)),
                },
            )
        )
        seen_scenes.add((sid, "Fast_ACVNet_KITTI2015"))

    # 4. Cross-Domain Generalization Collapse: HITNet SceneFlow on KITTI
    hit_sf = [r for r in records if r["method"] == "HITNet_SceneFlow" and r["dataset"] == "kitti"]
    if hit_sf:
        hit_sf.sort(key=lambda r: float(r["all_d1_all"]), reverse=True)
        r = hit_sf[0]
        catalog.append(
            FailureCaseRecord(
                scene_id=r["scene_id"],
                dataset="kitti",
                split=r.get("split", "val_tune"),
                method="HITNet_SceneFlow",
                category="out_of_distribution_generalization_collapse",
                primary_metric_name="all_d1_all",
                primary_metric_val=float(r["all_d1_all"]),
                description="Synthetic SceneFlow weights deployed zero-shot onto real-world automotive camera geometry.",
                root_cause_analysis=(
                    "Catastrophic domain shift: HITNet's learned tile representations overfit to synthetic rendering statistics, "
                    "failing completely on natural outdoor automotive illumination and focal length differences (>99% D1 error)."
                ),
                metrics={
                    "all_d1_all": float(r.get("all_d1_all", 0)),
                    "all_epe": float(r.get("all_epe", 0)),
                    "depth_rmse": float(r.get("depth_rmse", 0)),
                },
            )
        )

    # 5. Indoor Low-Texture & Planar Geometry: ETH3D Delivery Area
    eth_records = [r for r in records if r["dataset"] == "eth3d" and r["method"] == "StereoBM"]
    if eth_records:
        eth_records.sort(key=lambda r: float(r["all_all_bad_1_0"]), reverse=True)
        r = eth_records[0]
        sid = r["scene_id"]
        meta = KNOWN_SCENE_TAXONOMY.get(sid, {})
        catalog.append(
            FailureCaseRecord(
                scene_id=sid,
                dataset="eth3d",
                split=r.get("split", "val_tune"),
                method="StereoBM",
                category="textureless_indoor_walls_and_gloss",
                primary_metric_name="all_all_bad_1_0",
                primary_metric_val=float(r["all_all_bad_1_0"]),
                description=meta.get("description", f"ETH3D scene {sid} with indoor warehouse expanses."),
                root_cause_analysis=meta.get(
                    "root_cause",
                    "Featureless walls cause SAD cost curve flattening, forcing block matcher to reject over 40% of the image.",
                ),
                metrics={
                    "all_all_bad_1_0": float(r.get("all_all_bad_1_0", 0)),
                    "all_mutual_bad_1_0": float(r.get("all_mutual_bad_1_0", 0)),
                    "density_pct": float(r.get("density_pct", 0)),
                    "fg_depth_rmse": float(r.get("fg_depth_rmse", 0)),
                },
            )
        )

    return catalog


def export_failure_catalog(catalog: List[FailureCaseRecord], json_path: Union[str, Path]) -> None:
    """Save failure catalog to JSON format."""
    p = Path(json_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump([rec.to_dict() for rec in catalog], f, indent=2)

