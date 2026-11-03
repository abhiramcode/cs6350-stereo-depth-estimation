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
│   ├── shallow_inference/          # Default classical inference runs
│   ├── shallow_tuned/              # Results produced by tuned classical configurations
│   └── deep_inference/             # Deep SOTA inference results (HITNet, Fast-ACVNet, DTD)
├── scripts/                        # Standalone execution pipelines
│   ├── download_datasets.sh        # Automated dataset downloader and layout setup
│   ├── smoke_test_stage1.py        # Dataset loading and calibration integrity verification
│   ├── tune_shallow_parameters.py  # Hyperparameter grid search optimizer
│   ├── run_shallow_inference.py    # Batch/single-pair classical stereo runner
│   └── run_deep_inference.py       # Batch/single-pair SOTA deep stereo runner
├── src/                            # Modular source code
│   ├── datasets/                   # Unified loaders for KITTI, ETH3D, and custom pairs
│   ├── stereo/                     # Matcher wrappers (base, StereoBM, StereoSGBM, HITNet, ACVNet, DTD)
│   │   └── models/dtd/             # Dusk Till Dawn & UniMatch modular PyTorch components
│   └── evaluation/                 # Standardized disparity & metric-depth evaluation metrics
└── tests/                          # 71 automated unit and integration tests
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

### Benchmark Results on ETH3D `val_tune` Split (5 Scenes)

Measured across 5 scenes (`delivery_area_1l`, `delivery_area_1s`, `delivery_area_2l`, `delivery_area_2s`, `delivery_area_3l`):

| Method | Type | Configuration / Domain | Pure Infer (ms) | Throughput (FPS) | Density (%) | EPE (px) | Bad-1.0 (%) | Bad-2.0 (%) |
|---|---|---|---|---|---|---|---|---|
| **StereoBM** | Classical | Tuned (`numDisp=64, bs=21, uq=10`) | **8.6 ms** | **121.0** | 58.3% | 0.29 px | 4.0% | 1.6% |
| **StereoSGBM** | Classical | Tuned (`numDisp=64, bs=7, uq=15`) | 36.1 ms | 28.9 | 88.4% | 0.62 px | 10.1% | 3.0% |
| **HITNet** | Deep SOTA | ETH3D Pretrained (480x640 ONNX) | 558.1 ms | 1.8 | **100.0%** | **0.19 px** | **1.7%** | **0.2%** |
| **Fast-ACVNet+** | Deep SOTA | Generalization (Scene Flow 480x640) | 430.5 ms | 2.3 | **100.0%** | 0.29 px | 3.0% | **0.2%** |
| **Fast-ACVNet+** | Deep SOTA | Fine-tuned KITTI 2015 (480x640) | 401.0 ms | 2.5 | **100.0%** | 1.37 px | 45.9% | 24.1% |
| **Dusk Till Dawn** | Deep SOTA | Oxford RobotCar Nighttime (320x192 VFM) | 2726.1 ms | 0.4 | **100.0%** | 2.70 px | 60.0% | 39.0% |

---

### Benchmark Results on KITTI 2015 `val_tune` Split (40 Scenes)

Measured across 40 scenes (`000000_10` to `000039_10`) with full semantic foreground object evaluation (`obj_map`, `fg=semantic_gt`) and camera calibration ($f \approx 721.5$ px, $B \approx 0.533$ m):

| Method | Paradigm | Variant / Weights | Infer (ms) | FPS | Density % | EPE (px) | D1-all % | FG EPE (px) | FG D1-all % | Depth RMSE (m) | Depth AbsRel |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **StereoBM** | Classical | Default (`numDisp=128, bs=15`) | **13.7 ms** | **75.1** | 42.3% | 1.12 px | 3.7% | 1.04 px | 4.1% | 9.01 m | 0.079 |
| **StereoSGBM** | Classical | Default (`numDisp=128, bs=5`) | 78.7 ms | 12.7 | 79.4% | 1.15 px | 4.7% | 1.31 px | 7.4% | 10.60 m | 0.088 |
| **Fast-ACVNet+** | Deep SOTA | **KITTI 2015 Fine-tuned** | 419.4 ms | 2.4 | **100.0%** | **0.92 px** | **4.7%** | **0.75 px** | **2.1%** | **4.68 m** | **0.042** |
| **Fast-ACVNet+** | Deep SOTA | Generalization (SceneFlow) | 438.3 ms | 2.3 | **100.0%** | 1.44 px | 7.7% | 1.83 px | 11.7% | 6.75 m | 0.086 |
| **Dusk Till Dawn** | Deep SOTA | Oxford RobotCar (VFM) | 2826.0 ms | 0.4 | **100.0%** | 24.32 px | 98.1% | 19.85 px | 100.0% | 41.81 m | 2.467 |

*Key Insights & Analysis:*
- **Foreground Object Mastery**: On traffic participants (`obj_map`), `Fast-ACVNet_KITTI2015` achieves a stunning **0.75 px EPE** and **2.1% D1-all error** at **100% density**, completely resolving windshields and specular body panels where classical StereoBM drops >53% of pixels.
- **Metric 3D Depth Superiority**: Fast-ACVNet reduces depth RMSE to **4.68 metres** (vs >9.0 metres for classical methods) and achieves an **AbsRel of 0.042** (4.2% relative error).
- **In-Domain vs Cross-Domain Generalization**: Fine-tuning specifically on KITTI 2015 improves foreground object accuracy by **82% over the zero-shot generalization model** (2.1% vs 11.7% FG D1-all), demonstrating the crucial value of in-domain supervision for autonomous navigation.
- **Throughput / Efficiency Sweet Spot**: Fast-ACVNet+ infers at ~420 ms on CPU (~2.4 FPS), providing dense, sub-pixel disparity maps without requiring multi-gigabyte GPU clusters.

---

## Testing & Continuous Verification

All modules are accompanied by unit and regression test coverage (71 passing tests):

```bash
# Run all unit and integration tests
uv run pytest

# Run specific test suites
uv run pytest tests/test_dtd.py            # Dusk Till Dawn VFM matcher tests
uv run pytest tests/test_acvnet.py         # Fast-ACVNet deep matcher tests
uv run pytest tests/test_hitnet.py         # Google HITNet deep matcher tests
uv run pytest tests/test_shallow_stereo.py  # Classical matcher tests
uv run pytest tests/test_metrics.py         # Evaluation metric calculations
uv run pytest tests/test_datasets.py        # Dataset loaders and schema checks
uv run pytest tests/test_io_and_calib.py    # Calibration and image I/O
uv run pytest tests/test_scripts.py         # CLI integration tests
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


