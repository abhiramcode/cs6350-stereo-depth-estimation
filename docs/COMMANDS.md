# Project Command Handbook

This handbook is a comprehensive, chronological reference for all commands and CLI tools available in this repository. It serves as a practical cheatsheet for new users and developers as the project evolves across all stages.

---

## Table of Contents

1. [Quick Start (TL;DR)](#1-quick-start-tldr)
2. [Environment & Dependencies](#2-environment--dependencies)
3. [Dataset Acquisition & Directory Setup](#3-dataset-acquisition--directory-setup)
4. [Stage 1: Dataset Verification & Smoke Tests](#4-stage-1-dataset-verification--smoke-tests)
5. [Stage 2: Shallow Stereo Parameter Tuning](#5-stage-2-shallow-stereo-parameter-tuning)
6. [Stage 2: Shallow Stereo Inference & Benchmarking](#6-stage-2-shallow-stereo-inference--benchmarking)
7. [Stage 3: Deep Stereo Inference & Benchmarking](#7-stage-3-deep-stereo-inference--benchmarking)
8. [Stage 4: Unified Benchmarking Suite](#8-stage-4-unified-benchmarking-suite)
9. [Stage 5: Qualitative Failure Case Analysis & Diagnostics](#9-stage-5-qualitative-failure-case-analysis--diagnostics)
10. [Unit & Integration Testing (Pytest)](#10-unit--integration-testing-pytest)
11. [Directory & Artifact Reference](#11-directory--artifact-reference)

---

## 1. Quick Start (TL;DR)

If you have cloned the repository and want to run the verified pipeline right away:

```bash
# 1. Install dependencies into virtual environment
uv sync --extra dev

# 2. Download ETH3D Low-Res Two-View dataset (~500 MB)
bash scripts/download_datasets.sh eth3d

# 3. Verify dataset parsing & calibration
uv run python scripts/smoke_test_stage1.py --dataset eth3d --root data/eth3d

# 4. Run tuned shallow stereo inference with visualizations
uv run python scripts/run_shallow_inference.py \
    --dataset eth3d \
    --root data/eth3d \
    --split val_tune \
    --config configs/shallow_methods_tuned.yaml \
    --save-viz
```

---

## 2. Environment & Dependencies

This project uses `uv` for reproducible environment and dependency management.

| Command | Explanation |
|---|---|
| `uv sync` | Synchronize and install base dependencies from `pyproject.toml`. |
| `uv sync --extra dev` | Install base packages plus developer/testing tools (`pytest`, `pytest-cov`). |
| `uv sync --extra dev --extra gpu` | Install development and GPU dependencies (PyTorch, Torchvision). |
| `uv run python <script.py>` | Execute any Python script directly inside the project virtual environment. |

---

## 3. Dataset Acquisition & Directory Setup

Dataset downloader helper: [`scripts/download_datasets.sh`](file:///home/rasas/Documents/Abhiram-Files/CS6350/CS6350-Stereo-Depth-Estimation/cs6350-stereo-depth-estimation/scripts/download_datasets.sh)

| Command | Explanation |
|---|---|
| `bash scripts/download_datasets.sh eth3d` | Automatically downloads and extracts the **ETH3D Low-Res Two-View** stereo benchmark (~500 MB) into `data/eth3d/` (27 training scenes with GT + 20 test scenes). Requires `p7zip` / `7z`. |
| `bash scripts/download_datasets.sh kitti` | Prepares `data/kitti2015/` directory and extracts `data_scene_flow.zip` once downloaded manually from the KITTI benchmark website (~2 GB download, ~6 GB extracted). |
| `bash scripts/download_datasets.sh custom` | Creates template directories `data/custom_eval/left/` and `data/custom_eval/right/` with instructions for drop-in assessment pairs. |
| `bash scripts/download_datasets.sh all` | Executes setup for ETH3D, KITTI, and custom folders in sequence. |

---

## 4. Stage 1: Dataset Verification & Smoke Tests

Verification script: [`scripts/smoke_test_stage1.py`](file:///home/rasas/Documents/Abhiram-Files/CS6350/CS6350-Stereo-Depth-Estimation/cs6350-stereo-depth-estimation/scripts/smoke_test_stage1.py)

Validates that dataset loaders correctly parse left/right image pairs, ground truth disparity scales, validity masks, foreground object masks, and camera calibration parameters ($f_{\text{px}}, B_{\text{m}}$).

### Synthetic In-Memory Mode (No download required)
```bash
uv run python scripts/smoke_test_stage1.py --synthetic
```
*Creates synthetic in-memory fixtures to test the unified `StereoSample` schema, dimension alignment, and calibration depth conversion without requiring any local dataset files.*

### Real Dataset Verification
```bash
# Verify ETH3D dataset loading and calibration (f≈542px, B≈60mm)
uv run python scripts/smoke_test_stage1.py --dataset eth3d --root data/eth3d

# Verify KITTI 2015 dataset loading (16-bit PNG ÷ 256.0, P_rect calib, semantic obj_map)
uv run python scripts/smoke_test_stage1.py --dataset kitti --root data/kitti2015

# Verify unlabeled custom pairs (graceful fallback when GT/calib are absent)
uv run python scripts/smoke_test_stage1.py --dataset custom --root data/custom_eval
```

---

## 5. Stage 2: Shallow Stereo Parameter Tuning

Tuning script: [`scripts/tune_shallow_parameters.py`](file:///home/rasas/Documents/Abhiram-Files/CS6350/CS6350-Stereo-Depth-Estimation/cs6350-stereo-depth-estimation/scripts/tune_shallow_parameters.py)

Performs a grid search over key hyperparameters on the `val_tune` split, scoring configurations by disparity error (EPE, Bad-1.0/2.0) and valid density. Outputs an updated configuration file.

### Comprehensive Grid Search (Default)
```bash
uv run python scripts/tune_shallow_parameters.py --dataset eth3d --root data/eth3d --split val_tune
```
*Tunes both `StereoBM` and `StereoSGBM` on all scenes in `val_tune`, prints a before/after comparison table, and writes optimal parameters to `configs/shallow_methods_tuned.yaml`.*

### Rapid Grid Search (`--quick`)
```bash
uv run python scripts/tune_shallow_parameters.py --dataset eth3d --root data/eth3d --split val_tune --quick
```
*Uses an abbreviated search grid for quick sanity-checking and verification.*

### Method-Specific Tuning
```bash
# Tune only OpenCV StereoBM (numDisparities, blockSize, uniquenessRatio, textureThreshold)
uv run python scripts/tune_shallow_parameters.py --dataset eth3d --root data/eth3d --method StereoBM

# Tune only OpenCV StereoSGBM (numDisparities, blockSize, uniquenessRatio, mode HH vs 3WAY)
uv run python scripts/tune_shallow_parameters.py --dataset eth3d --root data/eth3d --method StereoSGBM
```

### Tuning Flags & Options
| Flag | Default | Description |
|---|---|---|
| `--dataset` | `eth3d` | Dataset to tune on (`eth3d` or `kitti`). |
| `--root` | `data/eth3d` | Root path of the dataset. |
| `--split` | `val_tune` | Split partition for tuning (e.g., `val_tune`). |
| `--method` | `all` | Method to tune: `all`, `StereoBM`, or `StereoSGBM`. |
| `--max-val-scenes` | `None` (all) | Limit tuning to the first $N$ validation scenes. |
| `--output-config` | `configs/shallow_methods_tuned.yaml` | Output YAML path for discovered optimal parameters. |
| `--quick` | `False` | Run with reduced parameter grid. |

---

## 6. Stage 2: Shallow Stereo Inference & Benchmarking

Inference script: [`scripts/run_shallow_inference.py`](file:///home/rasas/Documents/Abhiram-Files/CS6350/CS6350-Stereo-Depth-Estimation/cs6350-stereo-depth-estimation/scripts/run_shallow_inference.py)

Runs batch or single-pair inference across shallow methods. Isolates pure algorithm inference time from pre/post-processing, calculates prediction density, computes accuracy metrics against ground truth, and optionally exports disparity/depth maps and side-by-side visualization PNGs.

### Run All Shallow Methods with Tuned Parameters
```bash
uv run python scripts/run_shallow_inference.py \
    --dataset eth3d \
    --root data/eth3d \
    --split val_tune \
    --config configs/shallow_methods_tuned.yaml \
    --output-dir outputs/shallow_tuned
```

### Quick Sanity Test on a Single Sample with Visualizations
```bash
uv run python scripts/run_shallow_inference.py \
    --dataset eth3d \
    --root data/eth3d \
    --split val_tune \
    --max-samples 1 \
    --save-viz
```

### Full Evaluation on `eval_test` Split with Map Export
```bash
uv run python scripts/run_shallow_inference.py \
    --dataset eth3d \
    --root data/eth3d \
    --split eval_test \
    --config configs/shallow_methods_tuned.yaml \
    --output-dir outputs/shallow_eval_test \
    --save-maps \
    --save-viz
```

### Run a Specific Shallow Method
```bash
# Run StereoBM only
uv run python scripts/run_shallow_inference.py --dataset eth3d --root data/eth3d --method StereoBM

# Run StereoSGBM only
uv run python scripts/run_shallow_inference.py --dataset eth3d --root data/eth3d --method StereoSGBM

# Run StereoSGBM with WLS filter
uv run python scripts/run_shallow_inference.py --dataset eth3d --root data/eth3d --method StereoSGBM_WLS
```

### Run Shallow Inference on Unlabeled Custom Pairs
```bash
uv run python scripts/run_shallow_inference.py \
    --dataset custom \
    --root data/custom_eval \
    --method StereoSGBM \
    --save-viz
```

### Shallow Inference Flags & Options
| Flag | Default | Description |
|---|---|---|
| `--dataset` | `eth3d` | Target dataset: `eth3d`, `kitti`, or `custom`. |
| `--root` | `data/eth3d` | Root directory of the dataset. |
| `--split` | `val_tune` | Split partition: `val_tune`, `eval_test`, `train_all`, or `test_visual`. |
| `--method` | `all` | Specific method: `StereoBM`, `StereoSGBM`, `StereoSGBM_WLS`, or `all`. |
| `--config` | `configs/shallow_methods.yaml` | YAML configuration file defining matcher hyperparameters. |
| `--output-dir` | `outputs/shallow_inference` | Directory where CSV metrics, JSON summaries, and maps are written. |
| `--max-samples` | `None` (all) | Maximum number of samples to process. |
| `--warmup` | `3` | Number of unmeasured warmup iterations on the first sample. |
| `--save-maps` | `False` | Save predicted disparity (`_disp.npy`) and depth (`_depth.npy`) arrays. |
| `--save-viz` | `False` | Generate and save side-by-side visual comparison PNGs (Left RGB, Disparity colormap, Metric Depth). |

---

## 7. Stage 3: Deep Stereo Inference & Benchmarking

Deep inference script: [`scripts/run_deep_inference.py`](file:///home/rasas/Documents/Abhiram-Files/CS6350/CS6350-Stereo-Depth-Estimation/cs6350-stereo-depth-estimation/scripts/run_deep_inference.py)

Runs batch or single-pair inference across SOTA deep learning methods:
- **Google HITNet** (`HITNet_ETH3D`, `HITNet_SceneFlow`)
- **Fast-ACVNet+** (`Fast_ACVNet_Generalization`, `Fast_ACVNet_KITTI2015`)
- **Dusk Till Dawn** (`DuskTillDawn_Fast`, `DuskTillDawn_HighRes`)

All methods handle automated checkpoint downloading, ImageNet normalization, resolution scaling, disparity coordinate preservation ($d_{\text{orig}} = d_{\text{model}} \times \frac{W_{\text{orig}}}{W_{\text{model}}}$), and metric depth conversion ($Z = fB/d$).

### Run All Deep Learning Methods
```bash
uv run python scripts/run_deep_inference.py \
    --dataset eth3d \
    --root data/eth3d \
    --split val_tune \
    --config configs/deep_methods.yaml \
    --output-dir outputs/deep_inference \
    --save-viz
```

### Run Google HITNet
```bash
# Run HITNet on ETH3D validation split (eth3d checkpoint)
uv run python scripts/run_deep_inference.py \
    --dataset eth3d \
    --root data/eth3d \
    --split val_tune \
    --method HITNet_ETH3D \
    --save-viz

# Run HITNet with Scene Flow checkpoint
uv run python scripts/run_deep_inference.py \
    --dataset eth3d \
    --root data/eth3d \
    --split val_tune \
    --method HITNet_SceneFlow
```

### Run Fast-ACVNet+ (TPAMI 2023)
```bash
# Run Fast-ACVNet Generalization (zero-shot, trained on Scene Flow)
uv run python scripts/run_deep_inference.py \
    --dataset eth3d \
    --root data/eth3d \
    --split val_tune \
    --method Fast_ACVNet_Generalization \
    --save-viz

# Run Fast-ACVNet KITTI 2015 (fine-tuned on KITTI 2015)
uv run python scripts/run_deep_inference.py \
    --dataset eth3d \
    --root data/eth3d \
    --split val_tune \
    --method Fast_ACVNet_KITTI2015 \
    --save-viz
```

### Run Dusk Till Dawn (ICRA 2024 VFM Stereo)
```bash
# Run Dusk Till Dawn Fast mode (320x192, ~2.7s per frame on CPU)
uv run python scripts/run_deep_inference.py \
    --dataset eth3d \
    --root data/eth3d \
    --split val_tune \
    --method DuskTillDawn_Fast \
    --save-viz

# Run Dusk Till Dawn High-Res mode (640x384)
uv run python scripts/run_deep_inference.py \
    --dataset eth3d \
    --root data/eth3d \
    --split val_tune \
    --method DuskTillDawn_HighRes
```

### Deep Inference Flags & Options
| Flag | Default | Description |
|---|---|---|
| `--dataset` | `eth3d` | Target dataset: `eth3d`, `kitti`, or `custom`. |
| `--root` | `data/eth3d` | Root directory of the dataset. |
| `--split` | `val_tune` | Split partition: `val_tune`, `eval_test`, `train_all`, or `test_visual`. |
| `--method` | `all` | Specific method: `HITNet_ETH3D`, `HITNet_SceneFlow`, `Fast_ACVNet_Generalization`, `Fast_ACVNet_KITTI2015`, `DuskTillDawn_Fast`, `DuskTillDawn_HighRes`, or `all`. |
| `--config` | `configs/deep_methods.yaml` | YAML configuration file defining deep model hyperparameters and checkpoints. |
| `--output-dir` | `outputs/deep_inference` | Directory where CSV metrics, JSON summaries, and visualizations are written. |
| `--max-samples` | `None` (all) | Maximum number of samples to process. |
| `--warmup` | `3` | Number of unmeasured warmup iterations on the first sample. |
| `--save-maps` | `False` | Save predicted disparity (`_disp.npy`) and depth (`_depth.npy`) arrays. |
| `--save-viz` | `False` | Generate and save side-by-side visual comparison PNGs (Left RGB, Disparity colormap, Metric Depth). |


---

---

## 8. Stage 4: Unified Benchmarking Suite

Unified benchmarking runner: [`scripts/benchmark_unified.py`](file:///home/rasas/Documents/Abhiram-Files/CS6350/CS6350-Stereo-Depth-Estimation/cs6350-stereo-depth-estimation/scripts/benchmark_unified.py)

Orchestrates cross-paradigm benchmarking across classical and deep methods on both ETH3D and KITTI 2015, enforcing:
1. **Fair Valid-Pixel Policy:** Evaluates both mutually valid pixels and 100% of ground-truth valid pixels (where invalid/unpredicted pixels count as outliers), exposing the true sparsity vs. coverage trade-off.
2. **Foreground Object Provenance:** Slices metrics on isolated foreground objects (`semantic_gt` for KITTI 2015, `pretrained_seg` via Mask R-CNN for ETH3D).
3. **Decoupled Metric Spaces:** Disparity metrics (EPE, Bad-1.0/2.0/3.0, D1-all) and bounded metric depth metrics ($Z = fB/d$, RMSE, AbsRel for $0.5 \le Z \le 80$ m).
4. **Computational Profiling:** Pure model inference latency, E2E latency, throughput (FPS), peak host RAM footprint (MB), and device specifications.

### Running the Unified Benchmark

```bash
# 1. Run full benchmark across all datasets and methods (ETH3D + KITTI 2015 on val_tune)
uv run python scripts/benchmark_unified.py --dataset all --split val_tune --methods all

# 2. Run specifically on ETH3D with visual comparison figures
uv run python scripts/benchmark_unified.py \
    --dataset eth3d \
    --split val_tune \
    --methods all \
    --save-viz

# 3. Run specifically on KITTI 2015 with visual comparisons
uv run python scripts/benchmark_unified.py \
    --dataset kitti \
    --split val_tune \
    --methods StereoBM,StereoSGBM,Fast_ACVNet_KITTI2015 \
    --save-viz

# 4. Quick smoke run on 2 samples to verify benchmarking pipeline
uv run python scripts/benchmark_unified.py \
    --dataset eth3d \
    --split val_tune \
    --max-samples 2 \
    --methods StereoBM,Fast_ACVNet_Generalization
```

### Unified Benchmark Flags & Options
| Flag | Default | Description |
|---|---|---|
| `--dataset` | `all` | Target dataset: `eth3d`, `kitti`, or `all`. |
| `--eth3d-root` | `data/eth3d` | Root directory for ETH3D. |
| `--kitti-root` | `data/kitti2015` | Root directory for KITTI 2015. |
| `--split` | `val_tune` | Dataset split partition: `val_tune`, `eval_test`, or `train_all`. |
| `--methods` | `all` | Methods to run: `all`, `shallow`, `deep`, or comma-separated list (e.g. `StereoBM,Fast_ACVNet_KITTI2015`). |
| `--shallow-config` | `configs/shallow_methods_tuned.yaml` | YAML configuration for tuned shallow methods. |
| `--deep-config` | `configs/deep_methods.yaml` | YAML configuration for SOTA deep methods. |
| `--eth3d-fg-strategy` | `pretrained_seg` | Strategy for ETH3D foreground masks: `pretrained_seg` (Mask R-CNN), `geometric_proxy`, or `unavailable`. |
| `--output-dir` | `outputs/benchmarks` | Output directory for CSV records, summaries, visualizations, and report. |
| `--max-samples` | `None` (all) | Limit number of evaluated samples per dataset. |
| `--warmup` | `3` | Number of unmeasured warmup iterations before timing. |
| `--min-depth` | `0.5` | Minimum valid depth in meters for bounded metric depth evaluation. |
| `--max-depth` | `80.0` | Maximum valid depth in meters for bounded metric depth evaluation. |
| `--save-viz` | `False` | Generate and save 4-panel visual comparison PNGs (Left RGB, GT Disp, Pred Disp, Pred Depth). |
| `--save-maps` | `False` | Save disparity and depth `.npy` arrays. |

---

## 9. Stage 5: Qualitative Failure Case Analysis & Diagnostics

Stage 5 isolates high-error scenes across both datasets, groups them into physical/algorithmic failure categories (specular reflections, textureless regions, occlusion boundaries, near-field saturation, out-of-distribution transfer), and renders official 3-column assignment grids (`[Estimated Depth Map, Left View, Right View]`), error heatmaps, and cross-method comparisons.

### 9.1 Standard Failure Analysis Run

Execute the automated failure extraction and grid rendering pipeline:

```bash
uv run python scripts/analyze_failures.py \
    --benchmark-csv outputs/benchmarks/unified_benchmark_results.csv \
    --output-dir outputs/failure_cases \
    --top-k 5
```

### 9.2 Custom Options

| Option | Values | Default | Description |
|---|---|---|---|
| `--benchmark-csv` | File path | `outputs/benchmarks/unified_benchmark_results.csv` | Path to benchmark results CSV |
| `--output-dir` | Directory path | `outputs/failure_cases` | Output directory for failure artifacts |
| `--kitti-dir` | Directory path | `data/kitti2015` | Path to KITTI 2015 root directory |
| `--eth3d-dir` | Directory path | `data/eth3d` | Path to ETH3D root directory |
| `--top-k` | Integer | `5` | Number of failure scenes to extract per category |
| `--device` | `cpu`, `cuda`, `auto` | `cpu` | Execution device for rendering models |

### 9.3 Generated Failure Artifacts

Running `analyze_failures.py` populates:
- `outputs/failure_cases/grids_3col/`: Official 3-column assignment grids (`[Estimated Depth Map, Left View, Right View]`).
- `outputs/failure_cases/grids_diagnostic/`: Diagnostic error heatmap grids (`[Left View, Ground Truth, Predicted Disparity, Error Heatmap, Metric Depth]`).
- `outputs/failure_cases/grids_comparative/`: Multi-method comparative panels across challenging scenes (`[Left View, GT, StereoBM, StereoSGBM, Fast-ACVNet, HITNet]`).
- `outputs/failure_cases/failure_catalog.json`: Machine-readable JSON metadata catalog.
- `outputs/failure_cases/failure_analysis_report.md`: Comprehensive Markdown failure analysis report.

---

## 10. Unit & Integration Testing (Pytest)

The repository maintains an automated, zero-regression test suite with **86 automated tests**.

| Command | Scope |
|---|---|
| `uv run pytest` | Run the complete project test suite (86 automated tests). |
| `uv run pytest tests/test_visualization.py` | Test 3-column grid rendering, colormapping, error heatmaps, and failure analyzer catalog logic. |
| `uv run pytest tests/test_scripts.py` | Integration test for all runnable CLI scripts (`run_shallow_inference.py`, `tune_shallow_parameters.py`, `run_deep_inference.py`, `benchmark_unified.py`, `analyze_failures.py`). |
| `uv run pytest tests/test_dtd.py` | Test Dusk Till Dawn VFM matcher (config, resolution validation, checkpoint error handling, synthetic & real inference). |
| `uv run pytest tests/test_acvnet.py` | Test Fast-ACVNet deep matcher (config, checkpoint error handling, synthetic & real inference, resolution scaling). |
| `uv run pytest tests/test_hitnet.py` | Test HITNet deep matcher (config, checkpoint validation, synthetic & real inference, resolution restoration). |
| `uv run pytest tests/test_shallow_stereo.py` | Test shallow stereo matchers (synthetic pairs, invalid handling, calibration depth conversion, config serialization). |
| `uv run pytest tests/test_metrics.py` | Test disparity metrics (mutual vs. all-valid GT policies, EPE, Bad-N, D1-all, density) and metric-depth metrics (RMSE, AbsRel). |
| `uv run pytest tests/test_datasets.py` | Test `StereoSample` schema, split partitioning, and foreground mask handlers (`semantic_gt`, `pretrained_seg`, `geometric_proxy`). |
| `uv run pytest tests/test_io_and_calib.py` | Test PNG decoding, PFM endianness, calibration parsing, and $Z = fB/d$ conversions. |
| `uv run pytest --cov=src` | Run all tests with code coverage analysis. |

---

## 11. Directory & Artifact Reference

Key project directories and output locations:

```
├── checkpoints/
│   ├── hitnet/                     # Google HITNet ONNX weights (eth3d, sceneflow)
│   ├── fast_acvnet/                # Fast-ACVNet+ ONNX weights (generalization, kitti_2015)
│   └── dtd/                        # Dusk Till Dawn PyTorch checkpoint (depth_net_20.pth)
├── configs/
│   ├── datasets.yaml               # Dataset directory paths and default split fractions
│   ├── shallow_methods.yaml        # Untuned default parameters for StereoBM and StereoSGBM
│   ├── shallow_methods_tuned.yaml  # Parameters tuned on ETH3D validation split
│   └── deep_methods.yaml           # SOTA deep methods (HITNet, Fast-ACVNet, Dusk Till Dawn)
├── data/
│   ├── eth3d/                      # ETH3D Low-Res Two-View scenes (im0.png, im1.png, calib.txt, disp0GT.pfm)
│   ├── kitti2015/                  # KITTI 2015 stereo benchmark (image_2, image_3, disp_occ_0, calib_cam_to_cam, obj_map)
│   └── custom_eval/                # User-provided left/right evaluation pairs
├── docs/
│   └── COMMANDS.md                 # This command handbook
├── outputs/
│   ├── benchmarks/                 # Stage 4: Unified benchmarking results
│   │   ├── summary_eth3d_val_tune.csv
│   │   ├── summary_kitti_val_tune.csv
│   │   ├── unified_benchmark_results.csv
│   │   ├── unified_benchmark_report.md
│   │   └── visualizations/
│   ├── failure_cases/              # Stage 5: Qualitative failure cases and diagnostics
│   │   ├── grids_3col/             # 3-column assignment grids [Estimated Depth, Left, Right]
│   │   ├── grids_diagnostic/       # Diagnostic error heatmap panels
│   │   ├── grids_comparative/      # Multi-method landmark comparison panels
│   │   ├── failure_catalog.json    # Machine-readable JSON failure case catalog
│   │   └── failure_analysis_report.md # Markdown failure case analysis report
│   ├── shallow_inference/          # Default shallow inference outputs
│   ├── shallow_tuned/              # Tuned shallow inference outputs
│   └── deep_inference/             # Deep SOTA inference results (HITNet, Fast-ACVNet, DTD)
├── scripts/
│   ├── download_datasets.sh        # Dataset download automation
│   ├── smoke_test_stage1.py        # Dataset & loader smoke testing
│   ├── tune_shallow_parameters.py  # Hyperparameter grid search
│   ├── run_shallow_inference.py    # Batch/single-pair classical stereo runner
│   ├── run_deep_inference.py       # Batch/single-pair SOTA deep stereo runner
│   ├── benchmark_unified.py        # Stage 4: Unified benchmarking suite
│   └── analyze_failures.py         # Stage 5: Failure analysis & grid rendering
├── src/
│   ├── datasets/                   # Loaders, formats, calibration, and foreground masking
│   ├── stereo/                     # Classical and deep SOTA matcher implementations
│   ├── evaluation/                 # Fair evaluation metrics and decoupled depth metrics
│   └── visualization/              # 3-column grids, diagnostic heatmaps, and failure analyzer
└── tests/                          # 86 automated unit and integration tests
```
