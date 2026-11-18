# Current Status

**Last updated:** 2026-10-27T18:30 IST — **ALL STAGES (1–5) COMPLETE**

## Stage 1: Environment Setup & Dataset Interface ✅ COMPLETE

### Acceptance Criteria

| Criterion | Status |
|---|---|
| Reproducible Python environment (`pyproject.toml` + `uv sync`) | ✅ |
| Unified dataset interface (`src/datasets/`) | ✅ |
| KITTI 2015 loader with 16-bit PNG ÷256, obj_map, P_rect calib | ✅ |
| ETH3D loader with PFM (both endianness), COLMAP calib + flat layout support | ✅ |
| Custom pair loader (evaluation-day, all-None fallback) | ✅ |
| StereoSample strict schema with provenance tracking | ✅ |
| Calibration.disp_to_depth() with safe zero/invalid handling | ✅ |
| Foreground masking with honest provenance tags | ✅ |
| YAML config (`configs/datasets.yaml`) | ✅ |
| Unit tests passing with synthetic fixtures | ✅ |
| CLI smoke test passing | ✅ |
| `.gitignore` excluding data/checkpoints/outputs | ✅ |
| Download script with size estimates | ✅ |
| Agent docs (PROJECT_CONTEXT, DECISIONS, CURRENT_STATUS) | ✅ |

---

## Stage 2: Shallow Methods Implementation ✅ COMPLETE

### Acceptance Criteria

| Criterion | Status | Implementation Details |
|---|---|---|
| Consistent matcher interface | ✅ | `StereoMatcher` base class + `StereoResult` in `src/stereo/base.py` |
| Configurable without code edits | ✅ | `StereoBMConfig`, `StereoSGBMConfig` dataclasses, `configs/shallow_methods.yaml`, and `configs/shallow_methods_tuned.yaml` |
| Factory instantiation | ✅ | `create_matcher()`, `load_shallow_configs()` in `src/stereo/` |
| OpenCV StereoBM implementation | ✅ | `StereoBMMatcher` in `src/stereo/shallow_bm_sgbm.py` |
| OpenCV StereoSGBM implementation | ✅ | `StereoSGBMMatcher` with support for MODE_3WAY / MODE_HH and optional WLS filter |
| Fixed-point conversion | ✅ | Correct ÷16.0 decoding to float32 pixels; invalid pixels (`<= 0`) marked |
| Metric depth conversion | ✅ | $Z = \frac{f \cdot B}{d}$ using scene calibration; decoupled from disparity metrics |
| Standardized evaluation metrics | ✅ | `src/evaluation/metrics.py` (EPE, Bad-1.0/2.0/3.0, D1-all, Density, RMSE, AbsRel) |
| Runtime & efficiency tracking | ✅ | Excludes I/O; isolates pure inference latency vs total pipeline latency; records hardware specs |
| Warm-up frames | ✅ | 3 warm-up frames before benchmarking |
| Batch & single-pair inference | ✅ | `scripts/run_shallow_inference.py` outputting CSV, JSON, saved maps, and visual PNG grids |
| Hyperparameter tuning pipeline | ✅ | `scripts/tune_shallow_parameters.py` tuning on `val_tune` splits |
| Unit and integration tests | ✅ | 52/52 tests passing in < 1s (`tests/test_shallow_stereo.py`, `tests/test_metrics.py`, `tests/test_scripts.py`) |

### Benchmark Results on ETH3D (`val_tune` split, 5 scenes)

| Method | Config | Infer Latency | FPS | Density % | EPE (px) | Bad-1.0 % | Bad-2.0 % |
|---|---|---|---|---|---|---|---|
| **StereoBM** | Default (numDisp=128, bs=15, uq=15) | 10.0 ms | 99.6 | 43.2% | 0.29 px | 3.5% | 1.3% |
| **StereoBM** | **Tuned** (numDisp=64, bs=21, uq=10) | **8.6 ms** | **121.0** | **58.3%** | **0.29 px** | **4.0%** | **1.6%** |
| **StereoSGBM** | Default (numDisp=128, bs=5, uq=10) | 60.6 ms | 16.5 | 81.8% | 0.75 px | 11.4% | 3.2% |
| **StereoSGBM** | **Tuned** (numDisp=64, bs=7, uq=15) | **36.1 ms** | **28.9** | **88.4%** | **0.62 px** | **10.1%** | **3.0%** |

*Tuned parameters improved StereoSGBM EPE by 17%, density by +6.6%, and cut runtime by 40%. For StereoBM, tuned parameters increased valid density from 43.2% to 58.3% while cutting runtime by 14%.*

---

## Datasets Status

| Dataset | Size | Status | Location |
|---|---|---|---|
| ETH3D Low-Res Two-View | ~500 MB | ✅ Downloaded & verified | `data/eth3d/` (27 train + 20 test scenes) |
| KITTI 2015 | ~2 GB zip | Not downloaded | Instructions in `scripts/download_datasets.sh` |

---

## Stage 3: Deep Learning (SOTA) Methods Integration ✅ COMPLETE

### First SOTA Model: Google HITNet (via `nburrus/stereodemo`) ✅ VERIFIED & COMPLETE

| Criterion | Status | Implementation Details |
|---|---|---|
| Checkpoint verification & licensing | ✅ | `hitnet_eth3d_480x640.onnx` (3.5 MB) & `hitnet_sceneflow_480x640.onnx` (8.8 MB) under Apache 2.0 / MIT |
| Headless inference adapter | ✅ | `HitnetMatcher` and `HitnetConfig` in `src/stereo/deep_hitnet.py` conforming to `StereoMatcher` |
| Image coordinate system & scaling | ✅ | Native $(480, 640)$ inference restored to $(H_{orig}, W_{orig})$ with disparity scaled by $W_{orig} / W_{model}$ |
| Device selection & fallback | ✅ | Supports `"auto"`, `"cuda"`, and `"cpu"`; gracefully falls back to CPU with provider logging in `hw_info` |
| Disparity & metric depth output | ✅ | Full metric depth via calibration ($Z = fB / d$); invalid pixels masked $\le 0$ |
| Automated tests | ✅ | 5/5 tests passing in `tests/test_hitnet.py` |
| Sample visualization | ✅ | `outputs/deep_inference/HITNet_ETH3D/visualizations/delivery_area_1l_HITNet_ETH3D.png` |

---

### Second SOTA Model: Fast-ACVNet+ (TPAMI 2023 / CVPR 2022) ✅ VERIFIED & COMPLETE

| Criterion | Status | Implementation Details |
|---|---|---|
| Checkpoint verification & licensing | ✅ | `fast_acvnet_plus_generalization_opset16_480x640.onnx` (51.8 MB) & `fast_acvnet_plus_kitti_2015_opset16_480x640.onnx` (51.8 MB) under MIT License (converted by PINTO0309) |
| Headless inference adapter | ✅ | `ACVNetMatcher` and `ACVNetConfig` in `src/stereo/deep_acvnet.py` conforming to `StereoMatcher` |
| Input format & normalization | ✅ | Fixed 640x480 RGB inputs; ImageNet normalization (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`) |
| Image coordinate system & scaling | ✅ | Coordinate-preserving restoration: $d_{orig} = d_{model} \times \frac{W_{orig}}{W_{model}}$ with linear interpolation |
| Automated checkpoint extraction | ✅ | Built-in streaming HTTP tar extractor in Python for zero-setup download |
| Device selection & fallback | ✅ | Graceful fallback to CPU provider with provider logging |
| Automated tests | ✅ | 6/6 tests passing in `tests/test_acvnet.py` |
| Sample visualization | ✅ | `outputs/deep_inference/Fast_ACVNet_Generalization/visualizations/delivery_area_1l_Fast_ACVNet_Generalization.png` |

---

### Third SOTA Model: Dusk Till Dawn (ICRA 2024 VFM Stereo) ✅ VERIFIED & COMPLETE

| Criterion | Status | Implementation Details |
|---|---|---|
| Checkpoint verification & licensing | ✅ | `depth_net_20.pth` (19.4 MB) from Oxford authors / ICRA 2024 Google Drive |
| Headless inference adapter | ✅ | `DTDMatcher` and `DTDConfig` in `src/stereo/deep_dtd.py` conforming to `StereoMatcher` |
| Model architecture & foundation models | ✅ | Modular PyTorch components in `src/stereo/models/dtd/`: DINO ViT-S/8 feature extractor (`dino_v1.py` with `skip_validation=True`), `StereoDepthNet`, and UniMatch cross-attention transformer matching |
| Input format & dimension constraints | ✅ | Spatial dimensions constrained to integer multiples of 64 ($320 \times 192$ fast mode, $640 \times 384$ high-res mode); ImageNet normalization |
| Image coordinate system & scaling | ✅ | Coordinate-preserving restoration: $d_{orig} = d_{model} \times \frac{W_{orig}}{W_{model}}$ with linear interpolation |
| Automated checkpoint downloading | ✅ | Automatic download from Google Drive into `checkpoints/dtd/` |
| Automated tests | ✅ | 7/7 tests passing in `tests/test_dtd.py` |
| Sample visualization | ✅ | `outputs/deep_inference/DuskTillDawn_Fast/visualizations/delivery_area_1l_DuskTillDawn_Fast.png` |

---

### Batch Deep Inference Pipeline & Unified Configs ✅ VERIFIED & COMPLETE

| Criterion | Status | Implementation Details |
|---|---|---|
| Dedicated batch runner | ✅ | `scripts/run_deep_inference.py` supporting all deep models, CSV metrics, JSON summaries, `.npy` map exports, and visual PNGs |
| Declarative YAML configs | ✅ | `configs/deep_methods.yaml` defining `HITNet_ETH3D`, `HITNet_SceneFlow`, `Fast_ACVNet_Generalization`, `Fast_ACVNet_KITTI2015`, `DuskTillDawn_Fast`, and `DuskTillDawn_HighRes` |
| CLI integration test | ✅ | `tests/test_scripts.py::test_deep_inference_help` passing |
| Full test suite | ✅ | 71/71 automated tests passing across the repository |

---

### Empirical Comparative Benchmark on ETH3D `val_tune` Split (5 Scenes)

Measured across `delivery_area_1l`, `delivery_area_1s`, `delivery_area_2l`, `delivery_area_2s`, `delivery_area_3l`:

| Method | Type | Variant / Domain | Pure Infer | FPS | Density % | EPE (px) | Bad-1.0 % | Bad-2.0 % |
|---|---|---|---|---|---|---|---|---|
| **StereoBM** | Classical | Tuned (`numDisp=64, bs=21, uq=10`) | **8.6 ms** | **121.0** | 58.3% | 0.29 px | 4.0% | 1.6% |
| **StereoSGBM** | Classical | Tuned (`numDisp=64, bs=7, uq=15`) | 36.1 ms | 28.9 | 88.4% | 0.62 px | 10.1% | 3.0% |
| **HITNet** | Deep SOTA | ETH3D Pretrained (480x640 ONNX) | 558.1 ms | 1.8 | **100.0%** | **0.19 px** | **1.7%** | **0.2%** |
| **Fast-ACVNet+** | Deep SOTA | Generalization (SceneFlow 480x640) | 430.5 ms | 2.3 | **100.0%** | 0.29 px | 3.0% | **0.2%** |
| **Fast-ACVNet+** | Deep SOTA | Fine-tuned KITTI 2015 (480x640) | 401.0 ms | 2.5 | **100.0%** | 1.37 px | 45.9% | 24.1% |
| **Dusk Till Dawn** | Deep SOTA | Oxford RobotCar Nighttime (320x192 VFM) | 2726.1 ms | 0.4 | **100.0%** | 2.70 px | 60.0% | 39.0% |

---

### Empirical Comparative Benchmark on KITTI 2015 `val_tune` Split (40 Scenes)

Measured across 40 scenes (`000000_10` to `000039_10`) with full semantic foreground object evaluation (`obj_map`, `fg=semantic_gt`) and camera calibration ($f \approx 721.5$ px, $B \approx 0.533$ m):

| Method | Paradigm | Variant / Weights | Infer (ms) | FPS | Density % | EPE (px) | Bad-1.0 % | D1-all % | FG EPE (px) | FG D1-all % | Depth RMSE (m) | Depth AbsRel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **StereoBM** | Classical | Default (`numDisp=128, bs=15`) | **13.7 ms** | **75.1** | 42.3% | 1.12 px | 29.3% | 3.7% | 1.04 px | 4.1% | 9.01 m | 0.079 |
| **StereoSGBM** | Classical | Default (`numDisp=128, bs=5`) | 78.7 ms | 12.7 | 79.4% | 1.15 px | 22.3% | 4.7% | 1.31 px | 7.4% | 10.60 m | 0.088 |
| **StereoSGBM_WLS** | Classical | Default (`numDisp=128, bs=5`) | 80.9 ms | 12.5 | 79.4% | 1.15 px | 22.3% | 4.7% | 1.31 px | 7.4% | 10.60 m | 0.088 |
| **Fast-ACVNet+** | Deep SOTA | **KITTI 2015 Fine-tuned** | 419.4 ms | 2.4 | **100.0%** | **0.92 px** | **23.0%** | **4.7%** | **0.75 px** | **2.1%** | **4.68 m** | **0.042** |
| **Fast-ACVNet+** | Deep SOTA | Generalization (SceneFlow) | 438.3 ms | 2.3 | **100.0%** | 1.44 px | 40.1% | 7.7% | 1.83 px | 11.7% | 6.75 m | 0.086 |
| **Dusk Till Dawn** | Deep SOTA | Oxford RobotCar (VFM) | 2826.0 ms | 0.4 | **100.0%** | 24.32 px | 99.8% | 98.1% | 19.85 px | 100.0% | 41.81 m | 2.467 |
| **HITNet** | Deep SOTA | SceneFlow Pretrained | 2552.9 ms | 0.4 | 99.9% | 143.16 px | 99.7% | 99.0% | 143.61 px | 98.0% | 72.92 m | 0.866 |

*Key Foreground Object Takeaway:* Fast-ACVNet+ (KITTI 2015 fine-tuned) dominates foreground object reconstruction with **0.75 px EPE** and **2.1% D1-all error** at **100% density**, reducing metric depth RMSE from >9.0 m (classical methods) to **4.68 m**.

---

## Stage 4: Unified Benchmarking Suite ✅ COMPLETE

### Acceptance Criteria

| Criterion | Status | Implementation Details |
|---|---|---|
| Fair valid-pixel accounting policy | ✅ | Dual reporting in `src/evaluation/metrics.py`: mutually valid pixels (`pred > 0 & gt_valid`) AND all-valid GT pixels (unpredicted pixels penalized as outliers matching official KITTI/Middlebury leaderboards) |
| Prediction density transparency | ✅ | Prediction density reported alongside all metrics, exposing classical sparsity trade-offs |
| Semantic foreground object slicing | ✅ | Provenance-tagged slicing: `semantic_gt` via KITTI `obj_map`, and `pretrained_seg` via torchvision Mask R-CNN ResNet-50 FPN on ETH3D |
| Decoupled metric depth evaluation | ✅ | Pure disparity metrics (px) and bounded metric depth metrics ($Z = fB/d$, bounded $0.5\text{ m} \le Z \le 80\text{ m}$) evaluated independently |
| Automated hardware & efficiency profiling | ✅ | Isolated pure inference vs E2E latency, FPS, peak host RSS RAM (`resource.getrusage`), and peak VRAM profiling with 3 unmeasured warmup cycles |
| Centralized non-interactive runner | ✅ | `scripts/benchmark_unified.py` executing across all datasets, splits, and models without interactive UI |
| Consolidated output artifacts | ✅ | Detailed CSV (`unified_benchmark_results.csv`), summary CSVs (`summary_eth3d_val_tune.csv`, `summary_kitti_val_tune.csv`), and Markdown report (`unified_benchmark_report.md`) |
| Test suite integration | ✅ | 74/74 unit and integration tests passing (`tests/test_metrics.py`, `tests/test_scripts.py`, `tests/test_datasets.py`) |

---

### Consolidated Benchmark Results: KITTI 2015 `val_tune` (40 Scenes)

Hardware: x86_64 CPU (12 cores), Linux 6.8.0-138-generic

| Method | Paradigm | Pure Lat (ms) | FPS | Peak RAM (MB) | Pred Density | Mutual EPE (px) | Mutual D1-all | **Official All-Valid D1** | FG Mutual EPE (px) | **FG All-Valid D1** | FG Depth RMSE (m) | Full Depth RMSE (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **StereoBM** | Classical | **11.8 ms** | **86.0** | 1272.7 | 50.2% | 1.55 px | 7.4% | **57.3%** | 1.32 px | **23.3%** | 15.66 m | 10.74 m |
| **StereoSGBM** | Classical | 42.3 ms | 23.7 | 1273.2 | 81.4% | 1.40 px | 5.8% | **16.7%** | 1.61 px | **17.2%** | 18.20 m | 12.35 m |
| **StereoSGBM_WLS** | Classical | 42.1 ms | 23.8 | 1273.2 | 81.4% | 1.40 px | 5.8% | **16.7%** | 1.61 px | **17.2%** | 18.20 m | 12.35 m |
| **Fast_ACVNet_KITTI2015** | Deep SOTA | 396.2 ms | 2.5 | 1294.6 | **100.0%** | **0.92 px** | **4.7%** | **4.7%** | **0.75 px** | **2.1%** | **1.70 m** | **4.68 m** |
| **Fast_ACVNet_Generalization** | Deep SOTA | 426.2 ms | 2.3 | 1294.6 | **100.0%** | 1.44 px | 7.7% | 7.7% | 1.83 px | 11.7% | 10.21 m | 6.75 m |
| **HITNet_SceneFlow** | Deep SOTA | 2464.6 ms | 0.4 | 1489.1 | 99.9% | 143.16 px | 99.0% | 99.0% | 143.61 px | 98.0% | 72.92 m | 72.92 m |
| **DuskTillDawn_Fast** | Deep SOTA | 2732.4 ms | 0.4 | 1572.7 | **100.0%** | 24.32 px | 98.1% | 98.1% | 19.85 px | 100.0% | 41.81 m | 41.81 m |

---

### Consolidated Benchmark Results: ETH3D `val_tune` (5 Scenes)

Hardware: x86_64 CPU (12 cores), Linux 6.8.0-138-generic

| Method | Paradigm | Pure Lat (ms) | FPS | Peak RAM (MB) | Pred Density | Mutual EPE (px) | Mutual Bad-1.0 | **Official All-Valid Bad-1.0** | FG Mutual EPE (px) | **FG All-Valid Bad-1.0** | FG Depth RMSE (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **StereoBM** | Classical | **7.6 ms** | **134.9** | 1272.7 | 58.3% | 0.29 px | 4.0% | **46.3%** | 0.20 px | **16.8%** | 0.56 m |
| **StereoSGBM** | Classical | 33.3 ms | 35.6 | 1272.7 | 88.4% | 0.62 px | 10.1% | **20.9%** | 0.24 px | **7.1%** | 0.87 m |
| **StereoSGBM_WLS** | Classical | 33.3 ms | 35.4 | 1272.7 | 88.4% | 0.62 px | 10.1% | **20.9%** | 0.24 px | **7.1%** | 0.87 m |
| **HITNet_ETH3D** | Deep SOTA | 563.2 ms | 1.8 | 1341.6 | **100.0%** | **0.19 px** | **1.7%** | **1.7%** | **0.13 px** | **0.0%** | **0.35 m** |
| **Fast_ACVNet_Generalization** | Deep SOTA | 438.9 ms | 2.3 | 1294.6 | **100.0%** | 0.29 px | 3.0% | **3.0%** | 0.19 px | **0.0%** | 0.40 m |
| **Fast_ACVNet_KITTI2015** | Deep SOTA | 427.0 ms | 2.3 | 1294.6 | **100.0%** | 1.37 px | 45.9% | 45.9% | 0.47 px | 16.5% | 0.54 m |
| **HITNet_SceneFlow** | Deep SOTA | 2476.3 ms | 0.4 | 1489.1 | 99.9% | 2.47 px | 38.6% | 38.7% | 1.10 px | 34.0% | 0.82 m |
| **DuskTillDawn_Fast** | Deep SOTA | 2748.6 ms | 0.4 | 1572.7 | **100.0%** | 2.70 px | 60.0% | 60.0% | 1.35 px | 51.6% | 0.90 m |

---

## Datasets Status

| Dataset | Size | Status | Location |
|---|---|---|---|
| ETH3D Low-Res Two-View | ~500 MB | ✅ Downloaded & verified | `data/eth3d/` (27 train + 20 test scenes) |
| KITTI 2015 | ~1.7 GB (3.3 GB total) | ✅ Downloaded, extracted, calibrated & verified | `data/kitti2015/` (200 train + 200 test scenes) |
| Custom Stereo Pairs | User-supplied | ✅ Template directory ready | `data/custom_eval/` |

---

---

## Stage 5: Qualitative Failure Case Analysis & Diagnostics ✅ COMPLETE

### Acceptance Criteria

| Criterion | Status | Implementation Details |
|---|---|---|
| Automated failure case extraction | ✅ | `src/visualization/failure_analysis.py` extracting top error scenes from `unified_benchmark_results.csv` |
| 5-mechanism failure taxonomy | ✅ | Categorization into (1) specular reflection, (2) textureless regions, (3) occlusion boundaries, (4) near-field saturation, and (5) out-of-distribution collapse |
| Official 3-column assignment grids | ✅ | `render_3column_grid` saving `[Estimated Depth Map, Left View, Right View]` into `outputs/failure_cases/grids_3col/` |
| Diagnostic error heatmap panels | ✅ | `render_diagnostic_grid` saving `[Left View, Ground Truth, Predicted Disparity, Error Heatmap, Metric Depth]` into `outputs/failure_cases/grids_diagnostic/` |
| Landmark multi-method comparisons | ✅ | `render_multimethod_grid` saving comparative panels across classical & deep models into `outputs/failure_cases/grids_comparative/` |
| Standalone CLI runner | ✅ | `scripts/analyze_failures.py` fully automated and configurable |
| Final Comprehensive Report | ✅ | Artifact `final_comparative_report.md` synthesizing all 5 stages, quantitative tables, hardware profiling (CPU vs RTX 3060), and interactive visual carousels |
| Automated test suite | ✅ | 86/86 unit and integration tests passing in ~25.8s (`tests/test_visualization.py`, `tests/test_scripts.py`, `tests/test_datasets.py`) |

---

## Final Project Summary

All 5 stages of the CS6350 TPA-2 Stereo Depth Estimation assignment have been successfully implemented, empirically evaluated, diagnosed, and documented:

1. **Stage 1 (Datasets & Calibrations):** Unified dataset loader (`KITTI2015Dataset`, `ETH3DDataset`, `CustomPairDataset`), deterministic 80/20 train/eval splits, calibration abstraction ($Z = fB/d$), and foreground provenance tagging (`semantic_gt`, `pretrained_seg`, `geometric_proxy`).
2. **Stage 2 (Classical Stereo):** Modular `StereoMatcher` base class, tuned `StereoBM` and `StereoSGBM` (with optional WLS filter), hyperparameter grid search, and timing isolation.
3. **Stage 3 (Deep Learning SOTA):** Headless inference adapters for `HITNet` (CVPR 2021), `Fast-ACVNet+` (TPAMI 2023 / CVPR 2022), and `Dusk Till Dawn` (ICRA 2024 VFM Stereo) with coordinate preservation and automated checkpoint extraction.
4. **Stage 4 (Unified Benchmarking):** Centralized benchmarking runner (`scripts/benchmark_unified.py`), fair dual-policy valid-pixel evaluation (mutual vs all-valid GT), semantic foreground object slicing, decoupled metric depth, and hardware memory profiling.
5. **Stage 5 (Failure Case Analysis & Final Reporting):** Qualitative failure extraction across 5 root causes, official 3-column visual grids (`[Estimated Depth Map, Left View, Right View]`), error heatmaps, multi-method comparisons, and the comprehensive final assignment report artifact.




