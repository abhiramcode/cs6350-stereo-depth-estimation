# Comparative Stereo Depth Estimation: Classical vs. Deep Learning

A reproducible benchmark and evaluation framework comparing classical computer vision stereo matching algorithms with state-of-the-art deep learning architectures. Evaluated on the **KITTI 2015 Stereo Benchmark**, the **ETH3D Low-Resolution Two-View Dataset**, and user-provided custom stereo pairs.

---

## Overview

Stereo depth estimation computes three-dimensional scene structure from a pair of rectified, horizontally displaced 2D images. This project implements a standardized, fair benchmarking pipeline that compares:

1. **Classical / Shallow Approaches**:
   - **Block Matching (StereoBM)**: Fast local sum-of-absolute-differences (SAD) block matching.
   - **Semi-Global Block Matching (StereoSGBM)**: Multi-directional dynamic programming optimizing a global smoothness energy function (Hirschmüller, 2008), with optional Weighted Least Squares (WLS) edge-preserving filtering.
2. **Deep Learning Architectures (SOTA)**:
   - **Google HITNet** (CVPR 2021): Hierarchical iterative tile refinement network operating on compact planar disparity hypotheses.
   - **Fast-ACVNet+** (TPAMI 2023 / CVPR 2022): Accurate and lightweight stereo matching via Attention Concatenation Volume.
   - **Dusk Till Dawn (DTD)** (ICRA 2024): Self-supervised stereo depth estimation leveraging Visual Foundation Models (DINO ViT-S/8) and UniMatch cross-attention transformers.

### Key Engineering Features

- **Decoupled Evaluation**: Disparity error metrics (pixels) and metric depth error metrics (metres, where $Z = \frac{f \cdot B}{d}$) are evaluated independently to prevent distant background points ($d \to 0$) from skewing error calculations.
- **Fair Valid-Pixel & Density Accounting**: Classical methods leave invalid holes (`<= 0`) in textureless or occluded regions. This framework tracks **Prediction Density (%)** alongside error metrics computed on mutually valid pixels and all ground-truth valid pixels.
- **Resolution & Coordinate System Preservation**: Deep models run inference at fixed resolutions (e.g. 640x480 or 320x192); disparity maps are scaled back to native resolution with horizontal coordinate compensation ($d_{\text{orig}} = d_{\text{model}} \times \frac{W_{\text{orig}}}{W_{\text{model}}}$).
- **Unified Data Schema**: A single `StereoSample` dataclass normalizes images, calibrations, disparity maps, and foreground masks across datasets, with explicit provenance tagging for semantic object masks versus geometric heuristics.
- **Inference Timing Isolation**: Hardware execution time excludes data loading and disk I/O, preceded by unmeasured warm-up iterations to ensure consistent runtime reporting.
- **Automated Hyperparameter Optimization**: Automated grid search on validation splits to discover optimal matching windows, search ranges, penalty weights, and uniqueness thresholds.
- **Failure Mode Diagnostics & Visual Grids**: Automated categorization of breakdown mechanisms (specular reflections, textureless regions, occlusions, near-field saturation) and rendering of official 3-column assignment grids (`[Estimated Depth Map, Left View, Right View]`) and error heatmaps.

For the complete list of CLI commands, flags, and quick-reference options, consult the **[Command Handbook](docs/COMMANDS.md)**.

---

## Repository Structure

```
├── checkpoints/                    # [Git-ignored] Model weights & checkpoints
│   ├── hitnet/                     # Google HITNet ONNX weights (eth3d, sceneflow)
│   ├── fast_acvnet/                # Fast-ACVNet+ ONNX weights (generalization, kitti_2015)
│   └── dtd/                        # Dusk Till Dawn PyTorch checkpoint (depth_net_20.pth)
├── configs/                        # Declarative YAML parameter configurations
│   ├── datasets.yaml               # Paths and split partition definitions
│   ├── shallow_methods.yaml        # Default parameters for StereoBM and StereoSGBM
│   ├── shallow_methods_tuned.yaml  # Empirically tuned parameters from validation grid search
│   └── deep_methods.yaml           # Configurations for HITNet, Fast-ACVNet, and Dusk Till Dawn
├── data/                           # [Git-ignored] Dataset directories
│   ├── eth3d/                      # ETH3D Low-Res Two-View benchmark
│   ├── kitti2015/                  # KITTI 2015 stereo benchmark
│   └── custom_eval/                # User-supplied left/right evaluation stereo pairs
├── docs/                           # Documentation and guides
│   └── COMMANDS.md                 # Complete CLI handbook & command reference
├── outputs/                        # [Git-ignored] Predictions, visualizations & metrics
│   ├── benchmarks/                 # Stage 4: Unified benchmarking results & reports
│   │   ├── summary_eth3d_val_tune.csv
│   │   ├── summary_kitti_val_tune.csv
│   │   ├── unified_benchmark_results.csv
│   │   └── unified_benchmark_report.md
│   ├── failure_cases/              # Stage 5: Failure analysis & qualitative grids
│   │   ├── grids_3col/             # 3-column assignment grids [Estimated Depth, Left, Right]
│   │   ├── grids_diagnostic/       # Diagnostic error heatmaps & metric depth
│   │   ├── grids_comparative/      # Multi-method landmark scene comparisons
│   │   ├── failure_catalog.json    # Machine-readable JSON failure catalog
│   │   └── failure_analysis_report.md # Markdown failure case analysis report
│   ├── shallow_inference/          # Default classical inference runs
│   ├── shallow_tuned/              # Results produced by tuned classical configurations
│   └── deep_inference/             # Deep SOTA inference results (HITNet, Fast-ACVNet, DTD)
├── scripts/                        # Standalone execution pipelines
│   ├── download_datasets.sh        # Automated dataset downloader and layout setup
│   ├── smoke_test_stage1.py        # Dataset loading and calibration integrity verification
│   ├── tune_shallow_parameters.py  # Hyperparameter grid search optimizer
│   ├── run_shallow_inference.py    # Batch/single-pair classical stereo runner
│   ├── run_deep_inference.py       # Batch/single-pair SOTA deep stereo runner
│   ├── benchmark_unified.py        # Unified benchmarking suite across datasets & methods
│   └── analyze_failures.py         # Failure case extraction, categorization & grid rendering
├── src/                            # Modular source code
│   ├── datasets/                   # Unified loaders for KITTI, ETH3D, and custom pairs
│   │   └── fg_masking.py           # Foreground masks (semantic_gt, Mask R-CNN pretrained_seg, geometric)
│   ├── stereo/                     # Matcher wrappers (base, StereoBM, StereoSGBM, HITNet, ACVNet, DTD)
│   │   └── models/dtd/             # Dusk Till Dawn & UniMatch modular PyTorch components
│   ├── evaluation/                 # Standardized disparity & metric-depth evaluation metrics
│   └── visualization/              # 3-column grids, diagnostic heatmaps, and failure analyzer
└── tests/                          # 86 automated unit and integration tests
```

---

## Installation & Environment Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, deterministic dependency resolution and virtual environment isolation.

### Prerequisites
- Python $\ge$ 3.10
- Linux / macOS / Windows
- `p7zip` / `7z` (required for extracting ETH3D archives)

### Setup Instructions

```bash
# Clone the repository
git clone <repo-url>
cd cs6350-stereo-depth-estimation

# Synchronize virtual environment with development dependencies
uv sync --extra dev

# Run the automated test suite to verify setup (synthetic fixtures; no data download required)
uv run pytest
```

---

## Supported Datasets

| Dataset | Modality / Environment | Disparity Format | Ground Truth Availability | Calibration Parameters |
|---|---|---|---|---|
| **ETH3D Low-Res Two-View** | Indoor & outdoor scenes under varied illumination | `.pfm` (float32, pixels) | Available for 27 training scenes; 20 test scenes are unlabeled | $f_{\text{px}} \approx 542$, $B \approx 0.06$ m (Middlebury `calib.txt`) |
| **KITTI 2015** | Autonomous driving scenes with dynamic traffic | 16-bit `.png` ($\text{raw} / 256.0$, pixels) | Available for 200 training scenes; 200 test scenes are unlabeled | $f_{\text{px}} \approx 721.5$, $B \approx 0.54$ m (`calib_cam_to_cam.txt`) |
| **Custom Stereo Pairs** | Unlabeled assessment pairs | None | None | Optional user-supplied `calib.yaml` |

### Downloading Datasets

Use the included helper script to acquire and unpack datasets:

```bash
# Download and extract ETH3D Low-Res Two-View benchmark (~500 MB)
bash scripts/download_datasets.sh eth3d

# Set up directory for KITTI 2015 (requires manual download of data_scene_flow.zip)
bash scripts/download_datasets.sh kitti

# Set up drop-in template folders for custom stereo pairs
bash scripts/download_datasets.sh custom
```

Detailed dataset download instructions and directory layouts are documented in **[docs/COMMANDS.md](docs/COMMANDS.md#3-dataset-acquisition--directory-setup)**.

---

## Usage Workflow

The framework operates in three primary workflows: **Verification**, **Hyperparameter Optimization**, and **Inference & Benchmarking**.

```
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│   1. Data Verification  │ ──> │   2. Parameter Tuning   │ ──> │ 3. Benchmark Inference  │
│  (smoke_test_stage1.py) │     │(tune_shallow_parameters)│     │(run_shallow_inference.py)│
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

### 1. Data Loading & Calibration Integrity Check
Ensure images, calibration matrices, ground-truth disparity scales, and masks load consistently:

```bash
# Verify against synthetic in-memory fixtures (no real files required)
uv run python scripts/smoke_test_stage1.py --synthetic

# Verify real ETH3D data directory
uv run python scripts/smoke_test_stage1.py --dataset eth3d --root data/eth3d
```

### 2. Tuning Classical Hyperparameters
Classical methods (BM/SGBM) require parameter tuning tailored to the camera baseline and scene depth dynamics. The tuning script explores combinations of `numDisparities`, `blockSize`, `uniquenessRatio`, and matching algorithms on the validation split (`val_tune`):

```bash
# Run full grid search on ETH3D validation split
uv run python scripts/tune_shallow_parameters.py --dataset eth3d --root data/eth3d --split val_tune

# Run quick parameter search with reduced grid
uv run python scripts/tune_shallow_parameters.py --dataset eth3d --root data/eth3d --split val_tune --quick
```
*Tuned configurations are automatically saved to `configs/shallow_methods_tuned.yaml`.*

### 3. Running Inference & Generating Visualizations
Execute inference over any dataset split or individual pair across classical and deep methods. Generates CSV metric sheets, JSON runtime logs, `.npy` disparity/depth maps, and side-by-side comparison images:

```bash
# Run tuned classical methods on ETH3D validation split
uv run python scripts/run_shallow_inference.py \
    --dataset eth3d \
    --root data/eth3d \
    --split val_tune \
    --config configs/shallow_methods_tuned.yaml \
    --output-dir outputs/shallow_tuned \
    --save-viz

# Run SOTA deep learning methods on ETH3D validation split
uv run python scripts/run_deep_inference.py \
    --dataset eth3d \
    --root data/eth3d \
    --split val_tune \
    --config configs/deep_methods.yaml \
    --output-dir outputs/deep_inference \
    --save-viz

# Run inference on custom left/right stereo image pairs
uv run python scripts/run_deep_inference.py \
    --dataset custom \
    --root data/custom_eval \
    --method Fast_ACVNet_Generalization \
    --save-viz
```

For all available flags (such as `--warmup`, `--max-samples`, `--save-maps`, and `--method`), refer to the **[Command Handbook](docs/COMMANDS.md)**.

---

## Evaluation Metrics

Accuracy and efficiency are quantified using standard stereo evaluation conventions:

### Disparity Metrics (Pixel Space)
- **End-Point Error (EPE)**: Mean absolute disparity difference $\frac{1}{|V|} \sum_{p \in V} |d_p - \hat{d}_p|$.
- **Bad-1.0 / Bad-2.0 / Bad-3.0**: Percentage of valid pixels where absolute disparity error exceeds $1.0$, $2.0$, or $3.0$ pixels.
- **D1-all** (KITTI official metric): Percentage of pixels where error $> 3.0$ px **and** $> 5\%$ of ground-truth disparity:
  $$\text{D1-all} = \frac{1}{|V|} \sum_{p \in V} \mathbb{I}\left(|d_p - \hat{d}_p| > 3 \text{ px} \land \frac{|d_p - \hat{d}_p|}{d_p} > 0.05\right) \times 100\%$$
- **Prediction Density (%)**: Percentage of ground-truth valid pixels for which the algorithm returned a valid match ($\hat{d}_p > 0$).

### Metric Depth Metrics (3D Space in Metres)
Depth is derived from disparity using camera focal length $f$ and baseline $B$:
$$Z = \frac{f \cdot B}{d}$$
- **RMSE**: $\sqrt{\frac{1}{|V|} \sum_{p \in V} (Z_p - \hat{Z}_p)^2}$ evaluated within bounds ($0.5\text{ m} \le Z \le 80\text{ m}$).
- **AbsRel**: $\frac{1}{|V|} \sum_{p \in V} \frac{|Z_p - \hat{Z}_p|}{Z_p}$.

### Computational Efficiency
- **Pure Inference Latency (ms)**: Measured wall-clock time strictly wrapping algorithmic computation, excluding image read, color conversion, and post-processing saving.
- **End-to-End Latency (ms)**: Complete sample processing cycle.
- **Throughput (FPS)**: Frames processed per second based on pure inference time.

---

### Benchmark Results on KITTI 2015 `val_tune` Split (40 Scenes)

Measured across 40 scenes (`000000_10` to `000039_10`) evaluating full scene, isolated foreground traffic objects (`obj_map`, `fg=semantic_gt`), and camera calibration ($f \approx 721.5$ px, $B \approx 0.533$ m):

| Method | Paradigm | Pure Infer | FPS | Density % | Mutual EPE | Mutual D1 % | Official All-D1 % | FG Mutual EPE | FG Official All-D1 % | FG Depth RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| **StereoBM** | Classical | **11.8 ms** | **86.0** | 50.2% | 1.55 px | 7.4% | **57.3%** | 1.32 px | **23.3%** | 15.66 m |
| **StereoSGBM** | Classical | 42.3 ms | 23.7 | 81.4% | 1.40 px | 5.8% | **16.7%** | 1.61 px | **17.2%** | 18.20 m |
| **Fast_ACVNet+ (KITTI)** | Deep SOTA | 396.2 ms | 2.5 | **100.0%** | **0.92 px** | **4.7%** | **4.7%** | **0.75 px** | **2.1%** | **1.70 m** |
| **Fast_ACVNet+ (SceneFlow)** | Deep SOTA | 426.2 ms | 2.3 | **100.0%** | 1.44 px | 7.7% | 7.7% | 1.83 px | 11.7% | 10.21 m |
| **HITNet (SceneFlow)** | Deep SOTA | 2464.6 ms | 0.4 | 99.9% | 143.16 px | 99.0% | 99.0% | 143.61 px | 98.0% | 93.59 m |
| **Dusk Till Dawn (VFM)** | Deep SOTA | 2732.4 ms | 0.4 | **100.0%** | 24.32 px | 98.1% | 98.1% | 19.85 px | 100.0% | 54.50 m |

---

### Benchmark Results on ETH3D `val_tune` Split (5 Scenes)

Measured across 5 scenes (`delivery_area_1l`, `delivery_area_1s`, `delivery_area_2l`, `delivery_area_2s`, `delivery_area_3l`) with Mask R-CNN instance segmentation foreground masks (`fg=pretrained_seg`):

| Method | Paradigm | Pure Infer | FPS | Density % | Mutual EPE | Mutual Bad-1.0 % | Official All-Bad-1.0 % | FG Mutual EPE | FG Official Bad-1.0 % | FG Depth RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| **StereoBM** | Classical | **7.6 ms** | **134.9** | 58.3% | 0.29 px | 4.0% | **46.3%** | 0.20 px | 16.8% | 0.56 m |
| **StereoSGBM** | Classical | 33.3 ms | 35.6 | 88.4% | 0.62 px | 10.1% | **20.9%** | 0.24 px | 7.1% | 0.87 m |
| **HITNet (ETH3D)** | Deep SOTA | 563.2 ms | 1.8 | **100.0%** | **0.19 px** | **1.7%** | **1.7%** | **0.13 px** | **0.0%** | **0.35 m** |
| **Fast-ACVNet+ (Gen)** | Deep SOTA | 438.9 ms | 2.3 | **100.0%** | 0.29 px | 3.0% | **3.0%** | 0.19 px | **0.0%** | **0.40 m** |
| **Fast-ACVNet+ (KITTI)** | Deep SOTA | 427.0 ms | 2.3 | **100.0%** | 1.37 px | 45.9% | 45.9% | 1.24 px | 10.4% | 1.86 m |
| **Dusk Till Dawn (VFM)** | Deep SOTA | 2748.6 ms | 0.4 | **100.0%** | 2.70 px | 60.0% | 60.0% | 4.35 px | 43.8% | 5.74 m |

*Key Insights & Analysis:*
- **The Fair Valid-Pixel Sparsity Fallacy**: StereoBM achieves an apparent mutual D1-all of 7.4% on KITTI only because it discards 49.8% of pixels. Under official benchmark rules (missing prediction = outlier), its true outlier rate is **57.3%**. Deep methods achieve 100% density with true low error.
- **Foreground Object Mastery**: On traffic participants (`obj_map`), `Fast-ACVNet_KITTI2015` achieves **0.75 px EPE** and **2.1% D1-all error** at **100% density**, cutting foreground depth RMSE to **1.70 m** (an ~90% error reduction from StereoBM's 15.66 m).
- **In-Domain vs Cross-Domain Generalization**: Domain-specific fine-tuning is decisive: HITNet dominates ETH3D (0.19 px EPE), while Fast-ACVNet+ fine-tuned on KITTI dominates automotive stereo (0.92 px global, 0.75 px FG).
- **Throughput / Efficiency Sweet Spot**: Fast-ACVNet+ infers at ~396 ms on CPU (~2.5 FPS) and real-time (>30 FPS) on GPUs, providing the best accuracy-to-compute ratio among deep models.

---

## Testing & Continuous Verification

All modules are accompanied by unit and regression test coverage (86 passing tests in ~25s):

```bash
# Run all 86 unit and integration tests
uv run pytest

# Run specific test suites
uv run pytest tests/test_visualization.py   # 3-column grids, colormaps, error heatmaps, failure analyzer
uv run pytest tests/test_scripts.py         # CLI integration tests for all 5 executable pipelines
uv run pytest tests/test_dtd.py            # Dusk Till Dawn VFM matcher tests
uv run pytest tests/test_acvnet.py         # Fast-ACVNet deep matcher tests
uv run pytest tests/test_hitnet.py         # Google HITNet deep matcher tests
uv run pytest tests/test_shallow_stereo.py  # Classical matcher tests
uv run pytest tests/test_metrics.py         # Evaluation metric calculations (fair valid-pixel policy)
uv run pytest tests/test_datasets.py        # Dataset loaders and schema checks (Mask R-CNN seg)
uv run pytest tests/test_io_and_calib.py    # Calibration and image I/O
```

---

## References

- **Dusk Till Dawn Paper**: Madhu Vankadari, Samuel Hodgson, Sangyun Shin, Kaichen Zhou, Andrew Markham, Niki Trigoni, *Dusk Till Dawn: Self-supervised Nighttime Stereo Depth Estimation using Visual Foundation Models*, IEEE International Conference on Robotics and Automation (ICRA), 2024. [arXiv:2404.03579](https://arxiv.org/abs/2404.03579)
- **Fast-ACVNet Paper**: Gangwei Xu, Yun Wang, Junda Cheng, Jinhui Tang, Xin Yang, *Accurate and Efficient Stereo Matching via Attention Concatenation Volume*, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2023. [arXiv:2209.12699](https://arxiv.org/abs/2209.12699)
- **HITNet Paper**: Vladimir Tankovich et al., *HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching*, CVPR 2021.
- **OpenCV Stereo Tutorial**: [Depth Map from Stereo Images](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)
- **Heiko Hirschmüller (2008)**: *Stereo Processing by Semiglobal Matching and Mutual Information*, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).
- **KITTI 2015 Stereo Benchmark**: Moritz Menze and Andreas Geiger, *Object Scene Flow for Autonomous Vehicles*, CVPR 2015.
- **ETH3D Benchmark**: Thomas Schöps et al., *A Multi-View Stereo Benchmark with High-Resolution Images and Multi-Camera Videos*, CVPR 2017.
- **Detailed Command Reference**: See [`docs/COMMANDS.md`](docs/COMMANDS.md).


