# Phase A — Reconnaissance Report

## System Environment

| Property | Value |
|---|---|
| **OS** | Ubuntu 22.04 (Linux 6.8.0-138-generic, x86_64) |
| **Python** | 3.10.12 (system `/usr/bin/python3`) |
| **uv** | 0.12.18 |
| **GPU** | NVIDIA GeForce RTX 3060 (12 GB VRAM, 340 MiB in use) |
| **NVIDIA Driver** | 580.178.04 |
| **CUDA (Driver)** | 13.0 |
| **PyTorch** | 2.5.1+cu121 (system-wide, CUDA available via cu121) |
| **OpenCV** | 4.12.0 (system-wide) |
| **NumPy** | 2.2.6 (system-wide) |
| **RAM** | 62 GiB total, ~58 GiB available |
| **Disk** | **CRITICAL: Only 2.7 GB free on `/dev/sda4`** |
| **CUDA Toolkit (nvcc)** | Not installed system-wide |

## Repository State

- **Git**: Not initialized (no `.git` directory)
- **Existing files**: Only `.agents/` directory with `ACTION_PLAN.md`, `REFERENCES.md`, `Verdicts_on_feedback_proposal.md`, and `rules/master-rule.md`
- **No existing code, tests, configs, or `pyproject.toml`**
- **No virtual environment or `.venv` present**

## Dataset Specifications (Verified from References)

### KITTI 2015
- **Source**: `data_scene_flow.zip` from cvlibs.net (~2 GB download)
- **Layout**: `training/` (200 scenes) + `testing/` (200 scenes, no GT)
- **Images**: `image_2/` (left), `image_3/` (right) — rectified color, ~1242x375
- **GT Disparity**: `disp_occ_0/` — 16-bit PNG, decode: `float32 / 256.0`, valid: `raw_png > 0`
- **Foreground Mask**: `obj_map/` — semantic object labels (vehicles/pedestrians)
- **Calibration**: `calib_cam_to_cam/` — `P_rect_02` and `P_rect_03` projection matrices

### ETH3D Low-Res Two-View
- **Source**: ETH3D website (less than 500 MB total)
- **Layout**: `two_view_training/` (27 scenes) + `two_view_test/` (20 scenes, no GT)
- **Images**: Rectified stereo pairs
- **GT Disparity**: `.pfm` files with endianness/scale header
- **Valid mask**: `np.isfinite(disp) and (disp > 0)`
- **Calibration**: `calib.txt` per scene

### Unseen Custom Dataset (Evaluation Day)
- Arbitrary left/right stereo pairs
- No guaranteed GT, calibration, or masks
- Must gracefully handle all-None metadata

## Split Strategy
- Official `test` splits lack public GT — qualitative only
- Official `training` splits — deterministic 80/20 partition:
  - `eval_test` (80%): final quantitative reporting
  - `val_tune` (20%): shallow hyperparameter tuning

## Blockers

1. **Disk space critically low (2.7 GB free)** — Cannot download datasets until space is freed
2. **No system CUDA toolkit (nvcc)** — PyTorch cu121 bundles runtime, but custom CUDA kernel compilation (Stage 3 ACVNet) will need toolkit later
3. **ETH3D `calib.txt` format** — exact field names to verify with real data
4. **`fg_mask` for ETH3D** — only geometric proxy available; semantic masking deferred
