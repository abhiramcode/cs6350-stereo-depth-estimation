"""Unit and integration tests for visualization package and failure diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from src.visualization.failure_analysis import (
    FailureCaseRecord,
    build_failure_case_catalog,
    export_failure_catalog,
    extract_top_failures,
)
from src.visualization.grids import (
    colorize_depth,
    colorize_disparity,
    compute_error_heatmap,
    render_3column_grid,
    render_diagnostic_grid,
    render_multimethod_grid,
)


class TestColorization:
    """Tests for disparity and depth colormapping."""

    def test_colorize_disparity_shape_and_dtype(self):
        disp = np.array([[10.0, 20.0], [0.0, -1.0]], dtype=np.float32)
        valid = np.array([[True, True], [False, False]])
        rgb = colorize_disparity(disp, valid_mask=valid, cmap_name="plasma")

        assert rgb.shape == (2, 2, 3)
        assert rgb.dtype == np.uint8
        # Invalid pixels should receive dark invalid color (20, 20, 25)
        assert tuple(rgb[1, 0]) == (20, 20, 25)
        assert tuple(rgb[1, 1]) == (20, 20, 25)
        # Valid pixels should not be dark background
        assert tuple(rgb[0, 0]) != (20, 20, 25)

    def test_colorize_disparity_all_invalid(self):
        disp = np.zeros((10, 10), dtype=np.float32)
        rgb = colorize_disparity(disp)
        assert rgb.shape == (10, 10, 3)
        assert np.all(rgb == np.array([20, 20, 25], dtype=np.uint8))

    def test_colorize_depth(self):
        depth = np.array([[2.5, 10.0], [0.0, 100.0]], dtype=np.float32)
        rgb = colorize_depth(depth, min_depth=0.5, max_depth=80.0)

        assert rgb.shape == (2, 2, 3)
        assert rgb.dtype == np.uint8
        # 0.0 and 100.0 are out of bounds and should be invalid
        assert tuple(rgb[1, 0]) == (20, 20, 25)
        assert tuple(rgb[1, 1]) == (20, 20, 25)
        # 2.5 and 10.0 are in bounds
        assert tuple(rgb[0, 0]) != (20, 20, 25)


class TestErrorHeatmap:
    """Tests for disparity error calculation and heatmaps."""

    def test_compute_error_heatmap_known_values(self):
        gt = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        pred = np.array([[10.5, 25.0], [0.0, 42.0]], dtype=np.float32)
        valid = np.ones((2, 2), dtype=bool)

        abs_err, outliers, err_rgb = compute_error_heatmap(pred, gt, valid, threshold=3.0)

        # Pixel (0, 0): error = 0.5 (not outlier)
        assert np.isclose(abs_err[0, 0], 0.5)
        assert not outliers[0, 0]

        # Pixel (0, 1): error = 5.0 (outlier)
        assert np.isclose(abs_err[0, 1], 5.0)
        assert outliers[0, 1]

        # Pixel (1, 0): pred=0 so mutual invalid
        assert np.isnan(abs_err[1, 0])
        assert not outliers[1, 0]
        # Missing prediction on valid GT is colored purple
        assert tuple(err_rgb[1, 0]) == (120, 30, 160)


class TestGridRendering:
    """Tests for grid rendering functions."""

    def test_render_3column_grid(self, tmp_path: Path):
        depth_vis = np.zeros((100, 150, 3), dtype=np.uint8)
        left = np.full((100, 150, 3), 128, dtype=np.uint8)
        right = np.full((100, 150, 3), 130, dtype=np.uint8)

        out_file = tmp_path / "test_3col.png"
        fig = render_3column_grid(
            depth_or_disp_img=depth_vis,
            left_img=left,
            right_img=right,
            title="Test 3-Col",
            out_path=out_file,
        )
        assert out_file.is_file()
        assert out_file.stat().st_size > 0

    def test_render_diagnostic_grid(self, tmp_path: Path):
        left = np.full((100, 150, 3), 100, dtype=np.uint8)
        gt_disp = np.full((100, 150), 20.0, dtype=np.float32)
        pred_disp = np.full((100, 150), 21.0, dtype=np.float32)
        valid = np.ones((100, 150), dtype=bool)
        pred_depth = np.full((100, 150), 5.0, dtype=np.float32)

        out_file = tmp_path / "test_diag.png"
        fig = render_diagnostic_grid(
            left_img=left,
            gt_disp=gt_disp,
            pred_disp=pred_disp,
            valid_mask=valid,
            pred_depth=pred_depth,
            scene_id="synth_01",
            method_name="TestMatcher",
            metrics_summary="EPE: 1.0px",
            failure_tag="synthetic",
            out_path=out_file,
        )
        assert out_file.is_file()
        assert out_file.stat().st_size > 0

    def test_render_multimethod_grid(self, tmp_path: Path):
        left = np.full((50, 50, 3), 100, dtype=np.uint8)
        gt_disp = np.full((50, 50), 10.0, dtype=np.float32)
        valid = np.ones((50, 50), dtype=bool)
        methods = {
            "MethodA": np.full((50, 50), 9.0, dtype=np.float32),
            "MethodB": np.full((50, 50), 12.0, dtype=np.float32),
        }

        out_file = tmp_path / "test_multi.png"
        fig = render_multimethod_grid(
            left_img=left,
            gt_disp=gt_disp,
            valid_mask=valid,
            methods_dict=methods,
            scene_id="synth_multi",
            out_path=out_file,
        )
        assert out_file.is_file()
        assert out_file.stat().st_size > 0


class TestFailureAnalysisCatalog:
    """Tests for failure case extraction and cataloging."""

    def test_extract_top_failures(self):
        records = [
            {"dataset": "kitti", "method": "StereoBM", "scene_id": "01", "all_all_d1": "15.0"},
            {"dataset": "kitti", "method": "StereoBM", "scene_id": "02", "all_all_d1": "45.0"},
            {"dataset": "kitti", "method": "StereoBM", "scene_id": "03", "all_all_d1": "30.0"},
            {"dataset": "eth3d", "method": "StereoBM", "scene_id": "04", "all_all_d1": "50.0"},
        ]
        top = extract_top_failures(records, dataset="kitti", method="StereoBM", metric_key="all_all_d1", top_k=2)
        assert len(top) == 2
        assert top[0]["scene_id"] == "02"
        assert top[1]["scene_id"] == "03"

    def test_build_and_export_catalog(self, tmp_path: Path):
        records = [
            {
                "dataset": "kitti",
                "method": "StereoBM",
                "split": "val_tune",
                "scene_id": "000006",
                "fg_all_d1": "75.5",
                "fg_mutual_d1_all": "23.9",
                "fg_density_pct": "32.2",
                "all_all_d1": "57.3",
            },
            {
                "dataset": "kitti",
                "method": "Fast_ACVNet_KITTI2015",
                "split": "val_tune",
                "scene_id": "000008",
                "all_d1_all": "10.15",
                "fg_d1_all": "2.26",
                "all_epe": "1.04",
                "depth_rmse": "7.20",
            },
        ]
        catalog = build_failure_case_catalog(records, top_k_per_category=2)
        assert len(catalog) >= 2

        json_out = tmp_path / "catalog.json"
        export_failure_catalog(catalog, json_out)
        assert json_out.is_file()

        with open(json_out, "r") as f:
            data = json.load(f)
        assert len(data) == len(catalog)
        assert data[0]["scene_id"] == "000006"

