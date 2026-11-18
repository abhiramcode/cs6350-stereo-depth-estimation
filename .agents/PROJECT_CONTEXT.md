# Project Context

## Hardware & Environment

| Property | Value |
|---|---|
| OS | Ubuntu 22.04 LTS (kernel 6.8.0-138) |
| CPU | x86_64 |
| GPU | NVIDIA GeForce RTX 3060 (12 GB VRAM) |
| Driver | 580.178.04, CUDA 13.0 (driver level) |
| RAM | 62 GiB |
| Python | 3.13.15 (in `.venv`), 3.10.12 (system), managed via `uv 0.12.18` |
| PyTorch | 2.5.1+cu121 (system-wide, CUDA functional) / optional `torch>=2.1,<3` extra |
| OpenCV | 5.0.0 (in `.venv`) |
| Package Manager | `uv` with `pyproject.toml` |

## Architecture

The project follows a modular pipeline architecture:

```
src/datasets/     → Data loading & format decoding (Stage 1)
src/stereo/       → Matcher implementations (Stages 2–3)
src/evaluation/   → Metrics & benchmarking (Stage 4)
src/visualization/ → Rendering & report generation (Stage 5)
```

### Implemented Components

#### Stage 1: Dataset & Calibration Abstraction
- **`src/datasets/schema.py`**: Universal `Calibration` and `StereoSample` dataclasses with property getters (`left`, `right`, `gt_disp`, `dataset`) and explicit foreground provenance tagging.
- **`src/datasets/io_utils.py`**: Format-specific readers:
  - KITTI 16-bit PNG (`raw / 256.0`, pixels)
  - ETH3D PFM (handling scale/endianness)
  - ETH3D Middlebury `calib.txt` (`baseline` in mm $\to$ metres)
  - COLMAP calibration parsing (`cameras.txt`, `images.txt`)
  - 8-bit image readers (PNG, JPG)
- **`src/datasets/kitti.py`**: KITTI 2015 loader with deterministic 80/20 split (`val_tune` vs `eval_test`).
- **`src/datasets/eth3d.py`**: ETH3D loader auto-detecting flat layout (`data/eth3d/<scene>/`) and nested layout (`two_view_training/`), with `val_tune` (5 scenes) and `eval_test` (22 scenes).
- **`src/datasets/custom_pair.py`**: Evaluation-day loader for arbitrary stereo image pairs with optional calibration.
- **`src/datasets/fg_masking.py`**: Geometric proxy mask (`disp > median`) tagged `geometric_proxy`.

#### Stage 2: Classical / Shallow Stereo Matching
- **`src/stereo/base.py`**: 
  - `StereoMatcher` abstract base class handling color preprocessing (2D, 1-channel, 3-channel, 4-channel uint8), timing isolation, fixed-point conversion, validity masking, and calibration depth conversion.
  - `StereoResult` dataclass tracking predicted disparity, valid mask, metric depth, pure inference time (`elapsed_s`), end-to-end time (`total_elapsed_s`), prediction density (`density_pct`), config, and CPU hardware info.
  - `MatcherConfig` base dataclass with `to_dict()` and `from_dict()`.
- **`src/stereo/shallow_bm_sgbm.py`**:
  - `StereoBMMatcher` and `StereoBMConfig`: OpenCV Block Matching baseline with full parameter dataclass.
  - `StereoSGBMMatcher` and `StereoSGBMConfig`: Semi-Global Block Matching with configurable modes (`MODE_3WAY`, `MODE_HH`), automatic $P_1/P_2$ penalty calculation, and graceful fallback for WLS filtering if `cv2.ximgproc` is absent.
  - `create_matcher()` factory function and `load_shallow_configs()` YAML loader.
- **`src/evaluation/metrics.py`**: Standardized disparity metrics (`EPE`, `Bad-1.0/2.0/3.0`, `D1-all`, `density_pct`) and bounded metric-depth metrics (`RMSE`, `AbsRel`).
- **`scripts/tune_shallow_parameters.py`**: Systematic hyperparameter grid search optimizing accuracy and coverage on validation splits.
- **`scripts/run_shallow_inference.py`**: Batch & single-pair inference runner with warm-up iterations, pure latency measurement, CSV/JSON reporting, `.npy` array export, and side-by-side visual comparison PNGs.
- **`configs/shallow_methods.yaml`** & **`configs/shallow_methods_tuned.yaml`**: Untuned and empirically tuned parameters.
- **`docs/COMMANDS.md`**: Complete command handbook and CLI reference.

#### Stage 3: Deep Learning (SOTA) Methods
- **`src/stereo/deep_hitnet.py`**:
  - `HitnetMatcher` and `HitnetConfig`: Google HITNet (CVPR 2021) ONNX inference adapter.
  - Supports ETH3D (`hitnet_eth3d_480x640.onnx`, 3.5 MB) and SceneFlow (`hitnet_sceneflow_480x640.onnx`, 8.8 MB) models.
  - Automated download from official GitHub release assets into `checkpoints/hitnet/`.
  - Disparity restoration preserving horizontal coordinate scaling: $d_{\text{orig}} = d_{\text{model}} \times \frac{W_{\text{orig}}}{W_{\text{model}}}$.
- **`src/stereo/deep_acvnet.py`**:
  - `ACVNetMatcher` and `ACVNetConfig`: Fast-ACVNet+ (TPAMI 2023 / CVPR 2022) ONNX adapter.
  - Supports `generalization` (SceneFlow pretrained) and `kitti_2015` (fine-tuned) models.
  - Automated streaming HTTP tar extraction in Python into `checkpoints/fast_acvnet/`.
  - ImageNet normalization (`[0.485, 0.456, 0.406]`, `[0.229, 0.224, 0.225]`) and linear upsampling.
- **`src/stereo/deep_dtd.py` & `src/stereo/models/dtd/`**:
  - `DTDMatcher` and `DTDConfig`: Dusk Till Dawn (ICRA 2024) PyTorch adapter.
  - Modular model implementation: DINO ViT-S/8 feature extractor (`dino_v1.py` with `skip_validation=True`), `StereoDepthNet` (`depth_model.py`), and UniMatch cross-attention transformer matching (`unimatch/`).
  - Automated download of official ICRA 2024 weights (`depth_net_20.pth`, 19.4 MB) from Oxford Google Drive.
  - Enforces spatial dimensions as integer multiples of 64 ($320 \times 192$ fast mode, $640 \times 384$ high-res mode).
- **`scripts/run_deep_inference.py`**:
  - Dedicated deep learning inference pipeline supporting all deep models, automated map exports (`.npy`), CSV metric logging, JSON summary generation, and side-by-side visualization PNGs.
- **`configs/deep_methods.yaml`**:
  - Unified configuration file defining `HITNet_ETH3D`, `HITNet_SceneFlow`, `Fast_ACVNet_Generalization`, `Fast_ACVNet_KITTI2015`, `DuskTillDawn_Fast`, and `DuskTillDawn_HighRes`.

#### Stage 4: Unified Benchmarking Suite
- **`src/datasets/fg_masking.py`**:
  - Added `pretrained_seg_mask` using torchvision's Mask R-CNN ResNet-50 FPN with COCO weights.
  - Provenance tagged `pretrained_seg` with score thresholding $\ge 0.5$ and cached weights.
- **`src/datasets/eth3d.py`**:
  - Added `fg_strategy` parameter supporting `"unavailable"` (default), `"pretrained_seg"` (Mask R-CNN), and `"geometric_proxy"`.
- **`src/evaluation/metrics.py`**:
  - Enhanced `compute_disparity_metrics`: Computes mutually valid metrics (`density_pct`, `epe`, `bad_1_0`, `bad_2_0`, `bad_3_0`, `d1_all`) AND all-valid GT metrics (`all_bad_1_0`, `all_bad_2_0`, `all_bad_3_0`, `all_d1`, `all_epe`) penalizing unpredicted pixels as outliers.
  - Enhanced `compute_depth_metrics`: Added density tracking and bounded metric depth validation ($0.5\text{ m} \le Z \le 80.0\text{ m}$).
- **`scripts/benchmark_unified.py`**:
  - Unified non-interactive benchmark runner across classical and deep models on ETH3D and KITTI 2015.
  - Dual-policy evaluation: mutually valid pixels vs all-valid ground truth pixels.
  - Semantic foreground slicing (`obj_map` for KITTI, Mask R-CNN for ETH3D) vs whole-image evaluation.
  - Hardware profiling: CPU name, core count, platform, peak host RAM (RSS via `resource.getrusage`), and peak VRAM.
  - Generates `unified_benchmark_results.csv`, `summary_<dataset>_<split>.csv`, and `unified_benchmark_report.md`.

#### Stage 5: Qualitative Failure Case Analysis & Visualization
- **`src/visualization/grids.py`**:
  - `render_3column_grid`: Exact 3-column assignment visual layout `[Estimated Depth Map, Left View, Right View]`.
  - `render_diagnostic_grid`: 5-column diagnostic grid `[Left View, Ground Truth, Predicted Disparity, Error Heatmap, Metric Depth]`.
  - `render_multimethod_grid`: Side-by-side comparative grid across multiple methods.
  - `compute_error_heatmap`: Pixel-wise absolute disparity error, outlier thresholding, and colormapped error visualization.
  - `colorize_disparity` and `colorize_depth`: High-contrast robust colormapping with masked invalids.
- **`src/visualization/failure_analysis.py`**:
  - `build_failure_case_catalog`: Extraction and classification of failure modes into 5 concrete physical/algorithmic categories.
  - `export_failure_catalog`: JSON metadata serialization.
- **`scripts/analyze_failures.py`**:
  - CLI runner generating 3-column grids, diagnostic grids, multi-method comparisons, and markdown reports.
  - Populates `outputs/failure_cases/`.

## Dataset Locations

| Dataset | Path | Status | Details |
|---|---|---|---|
| ETH3D | `data/eth3d/` | ✅ Downloaded & verified | 27 train scenes (with GT) + 20 test scenes (no GT) |
| KITTI 2015 | `data/kitti2015/` | ✅ Downloaded, calibrated & verified | 200 train scenes (with GT & obj_map) + 200 test scenes |
| Custom | `data/custom_eval/` | ✅ Directory created | Ready for evaluation-day image pairs |

## Test Suite
- **86 automated unit & integration tests** passing (`tests/test_visualization.py`, `tests/test_datasets.py`, `tests/test_io_and_calib.py`, `tests/test_shallow_stereo.py`, `tests/test_hitnet.py`, `tests/test_acvnet.py`, `tests/test_dtd.py`, `tests/test_metrics.py`, `tests/test_scripts.py`).

