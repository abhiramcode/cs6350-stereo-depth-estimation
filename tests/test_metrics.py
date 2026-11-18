"""Unit tests for disparity and depth metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import compute_depth_metrics, compute_disparity_metrics


def test_disparity_metrics_perfect():
    gt = np.full((10, 10), 15.0, dtype=np.float32)
    pred = np.full((10, 10), 15.0, dtype=np.float32)
    mask = np.ones((10, 10), dtype=bool)

    m = compute_disparity_metrics(pred, gt, mask)
    assert m["density_pct"] == 100.0
    assert m["epe"] == 0.0
    assert m["bad_1_0"] == 0.0
    assert m["bad_2_0"] == 0.0
    assert m["bad_3_0"] == 0.0
    assert m["d1_all"] == 0.0


def test_disparity_metrics_known_error():
    gt = np.full((10, 10), 20.0, dtype=np.float32)
    pred = np.full((10, 10), 22.5, dtype=np.float32)  # diff = 2.5
    mask = np.ones((10, 10), dtype=bool)

    m = compute_disparity_metrics(pred, gt, mask)
    assert abs(m["epe"] - 2.5) < 1e-5
    assert m["bad_1_0"] == 100.0
    assert m["bad_2_0"] == 100.0
    assert m["bad_3_0"] == 0.0  # 2.5 <= 3.0


def test_disparity_density_and_invalids():
    gt = np.full((10, 10), 20.0, dtype=np.float32)
    pred = np.full((10, 10), 20.0, dtype=np.float32)
    pred[:5, :] = -1.0  # half invalid
    mask = np.ones((10, 10), dtype=bool)

    m = compute_disparity_metrics(pred, gt, mask)
    assert m["density_pct"] == 50.0
    assert m["n_gt"] == 100
    assert m["n_mutual"] == 50
    # Mutually valid pixels have 0 error
    assert m["epe"] == 0.0
    assert m["bad_1_0"] == 0.0
    assert m["d1_all"] == 0.0
    # All-valid metrics penalize the 50 invalid pixels as outliers
    assert m["all_bad_1_0"] == 50.0
    assert m["all_bad_2_0"] == 50.0
    assert m["all_bad_3_0"] == 50.0
    assert m["all_d1"] == 50.0
    # 50 valid pixels have 0 error; 50 invalid pixels have 20.0 error -> mean = 10.0
    assert abs(m["all_epe"] - 10.0) < 1e-5


def test_disparity_metrics_all_invalid():
    gt = np.full((5, 5), 10.0, dtype=np.float32)
    pred = np.full((5, 5), -1.0, dtype=np.float32)
    mask = np.ones((5, 5), dtype=bool)

    m = compute_disparity_metrics(pred, gt, mask)
    assert m["density_pct"] == 0.0
    assert m["n_gt"] == 25
    assert m["n_mutual"] == 0
    assert np.isnan(m["epe"])
    assert m["all_bad_1_0"] == 100.0
    assert m["all_bad_2_0"] == 100.0
    assert m["all_bad_3_0"] == 100.0
    assert m["all_d1"] == 100.0
    assert abs(m["all_epe"] - 10.0) < 1e-5


def test_depth_metrics_known_values():
    gt = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    pred = np.array([[11.0, 22.0], [33.0, 44.0]], dtype=np.float32)  # diff = 10%
    mask = np.ones((2, 2), dtype=bool)

    m = compute_depth_metrics(pred, gt, mask, min_depth=1.0, max_depth=50.0)
    assert abs(m["abs_rel"] - 0.1) < 1e-5
    expected_rmse = np.sqrt(np.mean([1.0**2, 2.0**2, 3.0**2, 4.0**2]))
    assert abs(m["rmse"] - expected_rmse) < 1e-5
