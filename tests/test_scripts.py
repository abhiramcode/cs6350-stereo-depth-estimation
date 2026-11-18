"""Integration tests for shallow scripts (inference and tuning)."""

import subprocess
import sys
from pathlib import Path
import pytest


def test_shallow_inference_help():
    result = subprocess.run(
        [sys.executable, "scripts/run_shallow_inference.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--dataset" in result.stdout
    assert "--method" in result.stdout
    assert "--save-viz" in result.stdout


def test_tune_shallow_help():
    result = subprocess.run(
        [sys.executable, "scripts/tune_shallow_parameters.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--dataset" in result.stdout
    assert "--method" in result.stdout
    assert "--quick" in result.stdout


def test_deep_inference_help():
    result = subprocess.run(
        [sys.executable, "scripts/run_deep_inference.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--dataset" in result.stdout
    assert "--method" in result.stdout
    assert "--config" in result.stdout
    assert "--save-viz" in result.stdout


def test_benchmark_unified_help():
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_unified.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--dataset" in result.stdout
    assert "--methods" in result.stdout
    assert "--eth3d-fg-strategy" in result.stdout
    assert "--min-depth" in result.stdout


def test_analyze_failures_help():
    result = subprocess.run(
        [sys.executable, "scripts/analyze_failures.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--benchmark-csv" in result.stdout
    assert "--kitti-dir" in result.stdout
    assert "--eth3d-dir" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--top-k" in result.stdout


