# Fast-ACVNet+ Integration Report (Stage 3 SOTA Deep Stereo)

> [!NOTE]
> **Model:** Fast Attention Concatenation Volume Network (**Fast-ACVNet+**)
> **Authors:** Gangwei Xu, Yun Wang, Junda Cheng, Jinhui Tang, Xin Yang (IEEE TPAMI 2023 / CVPR 2022)
> **Checkpoints:** `fast_acvnet_plus_generalization_opset16_480x640.onnx` & `fast_acvnet_plus_kitti_2015_opset16_480x640.onnx` (MIT License)

---

## 1. Verification Checklist & Assumptions

| Requirement / Property | Verification Status | Implementation Evidence |
|---|---|---|
| **Checkpoint Existence & License** | ✅ Verified | PINTO0309 Model Zoo 338 (Wasabi S3 mirror), MIT License matching official `gangweiX/Fast-ACVNet`. Models stored in `checkpoints/fast_acvnet/` (51.8 MB each). |
| **Training Domain** | ✅ Verified | `generalization` variant trained on synthetic Scene Flow (35k pairs) for cross-domain transfer; `kitti_2015` fine-tuned on real automotive driving scenes. |
| **Input Specification** | ✅ Verified | 3-channel RGB image pairs resized to $(480 \times 640)$ with standard ImageNet normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$). |
| **Output Specification** | ✅ Verified | Raw pixel disparity tensor $[1, 1, 480, 640]$ squeezed to $(480, 640)$ float32. |
| **Coordinate System Preservation** | ✅ Verified | Resized back to $(W_{orig}, H_{orig})$ with linear scaling: $d_{orig} = d_{model} \times \frac{W_{orig}}{W_{model}}$. Invalid/negative values clamped to 0. |
| **Metric Depth Decoupling** | ✅ Verified | Disparity metrics (px) evaluated independently from metric depth ($Z = \frac{f \cdot B}{d}$ in metres). |
| **Execution Providers** | ✅ Verified | ONNX Runtime with automatic device selection (`"auto"`, `"cuda"`, `"cpu"`) and fallback to CPU. |
| **Zero Git Commands** | ✅ Followed | Zero git commands executed per user instruction. |

---

## 2. Comparative Benchmark on ETH3D (`delivery_area_1l`, $489 \times 942$ px)

All methods benchmarked on identical hardware (Intel CPU / RTX 3060) and identical ground-truth valid masks:

| Method | Paradigm | Configuration / Domain | Pure Infer | FPS | Density % | EPE (px) | Bad-1.0 % | Bad-2.0 % |
|---|---|---|---|---|---|---|---|---|
| **StereoBM** | Classical | Tuned (`numDisp=64, bs=21, uq=10`) | **8.6 ms** | **121.0** | 65.7% | **0.23 px** | 4.0% | 1.6% |
| **StereoSGBM** | Classical | Tuned (`numDisp=64, bs=7, uq=15`) | 43.1 ms | 23.2 | 91.3% | 0.36 px | 8.3% | 1.4% |
| **HITNet** | Deep SOTA | ETH3D Pretrained (480x640) | 551.7 ms | 1.8 | **100.0%** | **0.23 px** | 3.2% | 0.6% |
| **Fast-ACVNet+** | Deep SOTA | **Generalization** (Scene Flow) | 429.4 ms | 2.3 | **100.0%** | 0.26 px | **2.8%** | **0.1%** |
| **Fast-ACVNet+** | Deep SOTA | KITTI 2015 Fine-tuned | 396.6 ms | 2.5 | **100.0%** | 1.74 px | 50.4% | 33.6% |

---

## 3. Qualitative Visualizations

Side-by-side comparison of **Left Image**, **Predicted Disparity** (plasma colormap), and **Metric Depth** ($Z = fB/d$, turbo colormap):

### Fast-ACVNet+ Generalization (Trained on Scene Flow)
![Fast-ACVNet Generalization on ETH3D delivery_area_1l](/home/rasas/.gemini/antigravity-ide/brain/b248e95d-a5bf-4dba-b6e8-a5b96dc2c879/fast_acvnet_generalization_viz.png)

### Fast-ACVNet+ KITTI 2015 (Fine-tuned on Autonomous Driving)
![Fast-ACVNet KITTI 2015 on ETH3D delivery_area_1l](/home/rasas/.gemini/antigravity-ide/brain/b248e95d-a5bf-4dba-b6e8-a5b96dc2c879/fast_acvnet_kitti_viz.png)

> [!TIP]
> **Key Insight — The Cross-Domain Generalization Gap:**
> Notice how the `Generalization` model cleanly reconstructs the indoor hallway walls, ceiling lamps, and distant doorways with smooth continuous disparity. Conversely, the `KITTI 2015` fine-tuned model suffers significant bias when applied zero-shot to indoor scenes because its priors are overfitted to horizontal road planes, vehicle proportions, and outdoor automotive disparities.

---

## 4. Software Architecture & Verification

The adapter is implemented in [`src/stereo/deep_acvnet.py`](file:///home/rasas/Documents/Abhiram-Files/CS6350/CS6350-Stereo-Depth-Estimation/cs6350-stereo-depth-estimation/src/stereo/deep_acvnet.py):
- Conforms to the standard [`StereoMatcher`](file:///home/rasas/Documents/Abhiram-Files/CS6350/CS6350-Stereo-Depth-Estimation/cs6350-stereo-depth-estimation/src/stereo/base.py) interface.
- Includes automated streaming HTTP tar extraction (`urllib` + `tarfile`) so new machines download weights with zero manual steps.
- Unit and integration tests in [`tests/test_acvnet.py`](file:///home/rasas/Documents/Abhiram-Files/CS6350/CS6350-Stereo-Depth-Estimation/cs6350-stereo-depth-estimation/tests/test_acvnet.py) (6/6 passing).
- Complete test suite: **63/63 passing** across the repository.
