#!/usr/bin/env python3
"""Stage 1 smoke test — verify dataset loaders produce correct StereoSample objects.

Usage:
    # With synthetic fixtures (no real data needed):
    uv run python scripts/smoke_test_stage1.py --synthetic

    # With real ETH3D data:
    uv run python scripts/smoke_test_stage1.py --dataset eth3d --root data/eth3d

    # With real KITTI data:
    uv run python scripts/smoke_test_stage1.py --dataset kitti --root data/kitti2015

    # With custom stereo pairs:
    uv run python scripts/smoke_test_stage1.py --dataset custom --root data/custom_eval
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.datasets.schema import StereoSample


def _validate_sample(sample: StereoSample, label: str) -> bool:
    """Validate a single sample and print its summary."""
    ok = True
    errors = []

    # Type checks
    if sample.left_img.dtype != np.uint8:
        errors.append(f"left_img dtype={sample.left_img.dtype}, expected uint8")
    if sample.right_img.dtype != np.uint8:
        errors.append(f"right_img dtype={sample.right_img.dtype}, expected uint8")
    if sample.left_img.ndim != 3 or sample.left_img.shape[2] != 3:
        errors.append(f"left_img shape={sample.left_img.shape}, expected (H,W,3)")

    # Dimension alignment
    H, W = sample.img_size
    if sample.left_img.shape[:2] != (H, W):
        errors.append(f"left_img HxW mismatch: {sample.left_img.shape[:2]} vs {(H,W)}")
    if sample.gt_disparity is not None:
        if sample.gt_disparity.shape != (H, W):
            errors.append(f"gt_disparity shape mismatch: {sample.gt_disparity.shape}")
        if sample.gt_disparity.dtype != np.float32:
            errors.append(f"gt_disparity dtype={sample.gt_disparity.dtype}")
    if sample.valid_mask is not None:
        if sample.valid_mask.shape != (H, W):
            errors.append(f"valid_mask shape mismatch")
        if sample.valid_mask.dtype != bool:
            errors.append(f"valid_mask dtype={sample.valid_mask.dtype}")
    if sample.fg_mask is not None:
        if sample.fg_mask.shape != (H, W):
            errors.append(f"fg_mask shape mismatch")

    # Provenance
    valid_provs = {"semantic_gt", "pretrained_seg", "geometric_proxy", "unavailable"}
    if sample.fg_mask_provenance not in valid_provs:
        errors.append(f"invalid fg_mask_provenance: {sample.fg_mask_provenance}")

    # Depth conversion
    if sample.has_calib and sample.has_gt:
        depth = sample.calib.disp_to_depth(sample.gt_disparity)
        if depth.dtype != np.float32:
            errors.append(f"depth dtype={depth.dtype}")

    if errors:
        print(f"  ✗ {label}: {'; '.join(errors)}")
        ok = False
    else:
        print(f"  ✓ {sample.summary()}")

    return ok


def smoke_synthetic() -> bool:
    """Run smoke test on synthetic fixtures."""
    from tests.fixtures.generate_fixtures import (
        create_custom_fixtures,
        create_eth3d_fixtures,
        create_kitti_fixtures,
    )
    from src.datasets.kitti import KITTI2015Dataset
    from src.datasets.eth3d import ETH3DDataset
    from src.datasets.custom_pair import CustomPairDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        all_ok = True

        # KITTI
        print("\n── KITTI 2015 (synthetic) ──")
        kitti_root = tmp / "kitti"
        create_kitti_fixtures(kitti_root, num_scenes=3)
        ds = KITTI2015Dataset(kitti_root, split="train_all", indices=[0, 1, 2])
        for sample in ds:
            if not _validate_sample(sample, f"KITTI {sample.scene_id}"):
                all_ok = False

        # ETH3D
        print("\n── ETH3D (synthetic) ──")
        eth3d_root = tmp / "eth3d"
        create_eth3d_fixtures(eth3d_root, num_scenes=3)
        ds = ETH3DDataset(eth3d_root, split="train_all")
        for sample in ds:
            if not _validate_sample(sample, f"ETH3D {sample.scene_id}"):
                all_ok = False

        # Custom
        print("\n── Custom Pairs (synthetic) ──")
        custom_root = tmp / "custom"
        create_custom_fixtures(custom_root, num_pairs=2)
        ds = CustomPairDataset(custom_root)
        for sample in ds:
            if not _validate_sample(sample, f"Custom {sample.scene_id}"):
                all_ok = False

        return all_ok


def smoke_real(dataset: str, root: str) -> bool:
    """Run smoke test on real data."""
    from src.datasets.kitti import KITTI2015Dataset
    from src.datasets.eth3d import ETH3DDataset
    from src.datasets.custom_pair import CustomPairDataset

    all_ok = True

    if dataset == "kitti":
        print(f"\n── KITTI 2015 ({root}) ──")
        for split in ("val_tune", "eval_test"):
            ds = KITTI2015Dataset(root, split=split)
            print(f"  Split '{split}': {len(ds)} scenes")
            if len(ds) > 0:
                sample = ds[0]
                if not _validate_sample(sample, f"{split}[0]"):
                    all_ok = False

    elif dataset == "eth3d":
        print(f"\n── ETH3D ({root}) ──")
        for split in ("val_tune", "eval_test"):
            try:
                ds = ETH3DDataset(root, split=split)
                print(f"  Split '{split}': {len(ds)} scenes")
                if len(ds) > 0:
                    sample = ds[0]
                    if not _validate_sample(sample, f"{split}[0]"):
                        all_ok = False
            except FileNotFoundError as e:
                print(f"  Split '{split}': {e}")

    elif dataset == "custom":
        print(f"\n── Custom ({root}) ──")
        ds = CustomPairDataset(root)
        print(f"  Found {len(ds)} pairs")
        for sample in ds:
            if not _validate_sample(sample, f"custom {sample.scene_id}"):
                all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Stage 1 dataset smoke test")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run on synthetic fixtures (no real data needed)")
    parser.add_argument("--dataset", choices=["kitti", "eth3d", "custom"],
                        help="Dataset to test with real data")
    parser.add_argument("--root", type=str, help="Root path to dataset")

    args = parser.parse_args()

    if args.synthetic or (not args.dataset):
        print("=" * 60)
        print("Stage 1 Smoke Test — Synthetic Fixtures")
        print("=" * 60)
        ok = smoke_synthetic()
    else:
        if not args.root:
            parser.error("--root is required when --dataset is specified")
        print("=" * 60)
        print(f"Stage 1 Smoke Test — {args.dataset.upper()}")
        print("=" * 60)
        ok = smoke_real(args.dataset, args.root)

    print()
    if ok:
        print("✅ All smoke tests PASSED")
    else:
        print("❌ Some smoke tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
