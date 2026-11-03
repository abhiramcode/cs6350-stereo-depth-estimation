## Selected Datasets

From the five dataset options provided in the assignment (`KITTI`, `NYU Depth Dataset V2`, `ETH3D Stereo Dataset`, `Apolloscapes`, and `DrivingStereo Dataset`), choose any two to compare the methods and demonstrate the results:

1. **KITTI (specifically KITTI 2015)**

    * **Why chosen:** Contains pre-rectified stereo pairs (`Left View` and `Right View`), compact download size (~2 GB for the 200 training scenes), native ground-truth foreground object masks (vehicles/pedestrians) that directly align with the assignment's foreground object accuracy goal, and out-of-the-box compatibility with the required deep learning baselines.




2. **ETH3D Stereo Dataset (Low-Res Two-View Benchmark)**

    * **Why chosen:** Extremely lightweight (<500 MB total for 27 training and 20 test stereo pairs), pre-rectified images with dense laser-scanned ground truth, and covers diverse indoor/outdoor object geometries without heavy storage or preprocessing overhead.



(Note: `Apolloscapes` and `DrivingStereo Dataset` are excluded due to massive storage requirements, and `NYU Depth Dataset V2` is excluded because it is captured with an RGB-D Kinect sensor rather than a native rectified binocular stereo rig.)

---

## AI Agent Reference Specification & Action Plan

### Project Context & System Goal

* **Assignment:** Computer Vision (CS6350) TPA-2 — Comparison of the performance of SOTA shallow and deep stereo depth estimation techniques for a foreground object to achieve high accuracy.


* **Primary Objective:** Explore and compare shallow learning-based (traditional computer vision) and deep learning-based (neural network) stereo depth estimation methods across **accuracy** and **computational efficiency**.


* **System I/O Contract:**
    * **Input:** Rectified Stereo View (`Left View`, `Right View`).


    * **Output:** `Estimated Depth Map` (or disparity map converted to metric depth via camera baseline and focal length).



---

### Stage 1: Environment Setup & Dataset Acquisition

* **Goal:** Establish a reproducible workspace and standardize both datasets into a unified directory and loader interface.
* **Agent Tasks:**
    1. Initialize a UV Python environment with OpenCV, PyTorch (CUDA-enabled), and dependencies required by `Stereo Demo` ([https://github.com/nburrus/stereodemo](https://github.com/nburrus/stereodemo)).


    2. Download **KITTI 2015** (`data_scene_flow.zip`) and **ETH3D Stereo Dataset** (Low-Res Two-View dataset). Note: Only the official `training` splits (200 scenes in KITTI 2015, 27 scenes in ETH3D) provide public ground truth. Partition the labeled `training` sets into `val_tune` (20% for shallow hyperparameter tuning) and `eval_test` (80% for final quantitative reporting). Reserve official unlabeled `test` splits for qualitative visual inspection only.
    3. Build a unified data loader yielding a structured sample dictionary with explicit metadata (`left_img`, `right_img`, `gt_disparity`, `valid_mask`, `fg_mask`, `fg_mask_provenance`, `calib_params`, `scene_id`).
        * **KITTI 2015 Decoding:** Decode 16-bit PNG ground truth via `disp = png.astype(np.float32) / 256.0` with `valid_mask = (png > 0)`. Extract $f$ (pixels) and $B$ (meters) from `calib_cam_to_cam` (`P_rect_02` and `P_rect_03`). Load `obj_map` as `fg_mask` (`fg_mask_provenance="semantic_gt"`).
        * **ETH3D Decoding:** Decode `.pfm` ground truth handling scale/endianness with `valid_mask = np.isfinite(disp) & (disp > 0)`. Extract $f$ and $B$ from `calib.txt`.
        * **Foreground Masks on ETH3D & Unknown Datasets:** Do not conflate geometric thresholding with semantic objects. Provide a pre-trained instance segmentation wrapper (e.g., Torchvision Mask R-CNN) to generate consistent foreground masks (`fg_mask_provenance="pretrained_seg"`), and expose median-disparity thresholding strictly as a separate geometric proxy (`fg_mask_provenance="geometric_proxy"`).
        * **Evaluation-Day Custom Folder Support:** Include an `UnlabeledCustomStereoDataset` adapter that accepts any directory of `(left, right)` image pairs with optional calibration for unseen assessment-day inference.


* **Stage Deliverable:** Verified dataset directory and a standardized loader script outputting aligned stereo pairs and foreground masks.

---

### Stage 2: Shallow Methods Implementation

* **Goal:** Implement traditional computer vision stereo matching pipelines to serve as the shallow baseline.


* **Target Methods:**
    * **Stereo Vision** based on the OpenCV depth map tutorial ([https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)).


    * **Block Matching (BM)** referencing LearnOpenCV ([https://learnopencv.com/depth-perception-using-stereo-camera-python-c/](https://learnopencv.com/depth-perception-using-stereo-camera-python-c/)).


    * **Semi-Global Block Matching (SGBM)** referencing DigitalNuage ([https://www.digitalnuage.com/disparity-and-depth-estimation-from-stereo-camera](https://www.digitalnuage.com/disparity-and-depth-estimation-from-stereo-camera)) and MathWorks ([https://www.mathworks.com/help/visionhdl/ug/stereoscopic-disparity.html](https://www.mathworks.com/help/visionhdl/ug/stereoscopic-disparity.html)).




* **Agent Tasks:**
    1. Implement `StereoBM` and `StereoSGBM` wrappers accepting standardized stereo pairs.


    2. Tune core hyperparameters (`numDisparities`, `blockSize`, uniqueness ratio, speckle filtering) on a validation subset of KITTI 2015 and ETH3D.


    3. Convert predicted disparity $d$ into metric depth $Z = \frac{f \cdot B}{d}$ using each scene's focal length $f$ and baseline $B$.


* **Stage Deliverable:** Parameter-tuned shallow inference pipeline generating disparity/depth maps and logging per-frame CPU runtime.

---

### Stage 3: Deep Learning (SOTA) Methods Integration

* **Goal:** Set up pre-trained deep stereo networks for zero-shot / fine-tuned inference without training from scratch.


* **Target Methods:**
    * **Stereo Demo Collection** ([https://github.com/nburrus/stereodemo](https://github.com/nburrus/stereodemo)) to run bundled SOTA models (e.g., CREStereo, RAFT-Stereo, HitNet).


    * **ACVNet:** *Accurate and Efficient Stereo Matching via Attention Concatenation Volume* (G. Xu et al., IEEE TPAMI, 2024).


    * **Dusk Till Dawn (Optional/Low-Light Evaluation):** *Self-supervised Nighttime Stereo Depth Estimation using Visual Foundation Models* (Madhu Vankadari et al., 2024).




* **Agent Tasks:**
    1. Clone and configure `stereodemo` and official pre-trained weights for **ACVNet**.


    2. Create a batch inference wrapper that feeds the exact same KITTI 2015 and ETH3D evaluation splits through the selected deep models.


    3. Normalize output resolutions and convert disparities to metric depth maps matching the ground-truth format.




* **Stage Deliverable:** Automated deep learning inference pipeline outputting predicted depth maps and logging GPU/CPU execution times.

---

### Stage 4: Benchmarking (Accuracy & Computational Efficiency)

* **Goal:** Quantitatively evaluate and compare shallow and deep methods on full scenes and isolated foreground objects.


* **Agent Tasks:**
    1. **Fair Valid-Pixel Policy:** Evaluate all methods across the exact same ground-truth valid pixels (`valid_mask`). Because shallow methods (`BM`/`SGBM`) produce sparse invalid holes (`pred_disp <= 0`), report prediction **Density (%)** alongside accuracy on both (a) all valid GT pixels (interpolating or penalizing missing predictions) and (b) mutually valid pixels.
    2. **Separated Disparity vs. Metric Depth Accuracy:** Report pixel-disparity metrics and metric-depth metrics in distinct table groups, evaluated separately on **Full Image (`all`)** and **Foreground Regions (`fg`, annotated with mask provenance)**:
        * **Disparity Metrics (px):** `EPE` (End-Point Error), `D1-all` / `Bad-3.0` (% error $> 3$ px and $> 5\%$ for KITTI), and `Bad-1.0` / `Bad-2.0` (% error $> 1$ px and $> 2$ px for ETH3D).
        * **Metric Depth Metrics (meters, where $Z = f \cdot B / d$):** `RMSE` and `AbsRel` evaluated within valid depth bounds ($0.5\text{ m} \le Z \le 80\text{ m}$ for KITTI; scene max range for ETH3D) to prevent distant background pixels ($d \to 0$) from exploding error variance.
    3. **Computational Efficiency Evaluation:** Exclude model loading and dataset I/O from runtime metrics. Perform 3 warm-up frames, enforce `torch.cuda.synchronize()` before and after inference, and record pure inference latency (ms/frame), end-to-end latency (including pre/post-processing), FPS, hardware device, and peak VRAM/RAM usage.




* **Stage Deliverable:** Consolidated CSV/table comparing all shallow and deep methods across both datasets.


---

### Stage 5: Failure Case Analysis & Final Reporting

* **Goal:** Identify edge cases where deep learning (or both shallow and deep) methods fail to estimate depth accurately, and assemble visual comparisons.


* **Agent Tasks:**
    1. Filter test scenes with the highest error spikes to isolate specific failure cases as required by the problem statement:

        * Specular/reflective surfaces (car windshields, glossy indoor surfaces).
        * Textureless regions (uniform walls, road surfaces, sky).
        * Occlusion boundaries and thin structures (fences, poles, foliage around foreground objects).
    * Harsh lighting or shadows.

    2. Generate side-by-side visual grids displaying `Depth Map`, `Left View`, and `Right View` as illustrated in the expected output format, alongside ground-truth error heatmaps.


    3. Compile the quantitative tables, runtime charts, and failure-case analyses into the final assignment report.


* **Stage Deliverable:** Complete comparative report and visualization bundle satisfying all CS6350 TPA-2 requirements.


---