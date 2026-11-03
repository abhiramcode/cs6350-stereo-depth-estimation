The following off-the-shelf code repositories and reference scripts map directly to the five stages of your action plan to satisfy the CS6350 TPA-2 problem statement. AI agents can clone these repositories and use or lightly modify the specified scripts instead of writing code from scratch.

---

### Master Code & Repository Matrix

| Action Plan Stage | Method / Component | Source URL / Repository | Usage Mode | Target Files for Agents |
| --- | --- | --- | --- | --- |
| **Stage 1: Datasets** | **KITTI 2015** & **ETH3D Stereo Dataset** Loaders | [https://github.com/princeton-vl/RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) <br> | Modify-and-Use | `core/stereo_datasets.py`, `core/utils/frame_utils.py`, `download_datasets.sh` |
| **Stage 2: Shallow** | **Stereo Vision (Class Baseline)** (`StereoBM`) | [https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)<br> | Modify-and-Use | Standalone tutorial script (`cv2.StereoBM_create`) |
| **Stage 2: Shallow** | **Block Matching (BM)**<br> | [https://github.com/spmallick/learnopencv](https://github.com/spmallick/learnopencv) (`Depth-Perception-Using-Stereo-Camera`) | Modify-and-Use | `python/depth_perception.py` (companion to [https://learnopencv.com/depth-perception-using-stereo-camera-python-c/](https://learnopencv.com/depth-perception-using-stereo-camera-python-c/))|
| **Stage 2: Shallow** | **Semi-Global Block Matching (SGBM)**<br> | [https://www.digitalnuage.com/disparity-and-depth-estimation-from-stereo-camera](https://www.digitalnuage.com/disparity-and-depth-estimation-from-stereo-camera) | Modify-and-Use | SGBM + WLS filter script (`cv2.StereoSGBM_create`) |
| **Stage 3: Deep (SOTA)** | **Stereo Demo** (Multi-Model Collection)| [https://github.com/nburrus/stereodemo](https://github.com/nburrus/stereodemo) | **Download-and-Use** | `stereodemo/methods.py`, `scripts/stereodemo` |
| **Stage 3: Deep (SOTA)** | **ACVNet** (Attention Concatenation Volume) | [https://github.com/gangweiX/ACVNet](https://github.com/gangweiX/ACVNet) | **Download-and-Use** | `models/acv.py`, `main_kitti.py`, `save_disp.py` |
| **Stage 3: Deep (SOTA)** | **Fast-ACVNet** (ONNX Standalone Alternative) | [https://github.com/ibaiGorordo/ONNX-FastACVNet-Depth-Estimation](https://github.com/ibaiGorordo/ONNX-FastACVNet-Depth-Estimation) | **Download-and-Use** | `kitti_stereo_depth_estimation.py`, `fast_acvnet.py` |
| **Stage 3: Deep (SOTA)** | **Dusk Till Dawn** (Nighttime / VFM Stereo) | [https://github.com/madhubabuv/dtd](https://github.com/madhubabuv/dtd) | Modify-and-Use | Inference and evaluation scripts |
| **Stages 4 & 5: Eval** | **Unified Benchmarking & Metrics** | [https://github.com/XiandaGuo/OpenStereo](https://github.com/XiandaGuo/OpenStereo) | Modify-and-Use | `stereo/evaluation/metric_per_image.py` |

---

### Stage-by-Stage Agent Implementation Guide

#### Stage 1: Dataset Download & Unified Data Loader

* **Assignment Requirement:** Explore the dataset options and choose any two (`KITTI 2015` and `ETH3D Stereo Dataset`) to compare the methods and demonstrate the results, taking `Left View` and `Right View` as input to produce an `Estimated Depth Map`.


* **Off-the-Shelf Code to Grab:**
    * **Dataset Download & Loaders:** Clone [https://github.com/princeton-vl/RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo).
    * Run `download_datasets.sh` to automatically fetch the **ETH3D** two-view stereo benchmark.
    * Download **KITTI 2015** (`data_scene_flow.zip`) directly from [https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo).
    * Extract `core/stereo_datasets.py` and `core/utils/frame_utils.py` (which contains a ready-made `.pfm` ground-truth disparity reader `readPFM` for ETH3D and 16-bit `.png` disparity reader for KITTI).


* **Agent Adaptation Instructions:**
    * **Ground-Truth Split Handling:** Only `training/` folders in KITTI 2015 (`disp_occ_0`) and ETH3D (`two_view_training_gt`) contain ground-truth labels. Do not run supervised evaluation metrics on `testing/` splits.
    * **Exact Disparity Scaling:** Ensure `frame_utils.py` decodes KITTI 16-bit PNGs by dividing by `256.0` (`disp = np.array(img).astype(np.float32) / 256.0`, invalid where `0`), whereas OpenCV `StereoBM`/`StereoSGBM` fixed-point outputs in Stage 2 are divided by `16.0`.
    * Modify the `KITTI` class in `core/stereo_datasets.py` to also load the `obj_map` PNG mask from the KITTI 2015 training folder so foreground objects are isolated for evaluation.
    * **Foreground Mask Provenance:** For ETH3D and custom unlabeled scenes, do not treat median disparity thresholding (`gt_disp > median`) as a true semantic object mask. Label `gt_disp > median` explicitly as `geometric_proxy`, and include an optional Torchvision segmentation helper (`torchvision.models.detection.maskrcnn_resnet50_fpn`) to extract true semantic foreground masks when required.



---

#### Stage 2: Shallow Learning-Based Methods

* **Assignment Requirement:** Implement traditional computer vision techniques including `Stereo Vision (taught in class)`, `Block Matching`, and `Semi-Global Block Matching`.


* **Off-the-Shelf Code to Grab:**
    * Stereo Vision Baseline & Block Matching (`StereoBM`):


        * Reference tutorial provided in the problem statement: [https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html).


        * Reference article provided in the problem statement: [https://learnopencv.com/depth-perception-using-stereo-camera-python-c/](https://learnopencv.com/depth-perception-using-stereo-camera-python-c/).


        * Direct GitHub Code: Clone [https://github.com/spmallick/learnopencv](https://github.com/spmallick/learnopencv) and navigate to `Depth-Perception-Using-Stereo-Camera/`. It provides ready-made Python code for `cv2.StereoBM_create()` and disparity-to-depth conversion.


    * Semi-Global Block Matching (`StereoSGBM`):


        * Reference tutorials provided in the problem statement: [https://www.digitalnuage.com/disparity-and-depth-estimation-from-stereo-camera](https://www.digitalnuage.com/disparity-and-depth-estimation-from-stereo-camera) and [https://www.mathworks.com/help/visionhdl/ug/stereoscopic-disparity.html](https://www.mathworks.com/help/visionhdl/ug/stereoscopic-disparity.html).


        * Direct GitHub Code: Also included inside [https://github.com/nburrus/stereodemo](https://github.com/nburrus/stereodemo) under `stereodemo/methods.py` (`OpenCVStereoMatcher` class), which wraps both `StereoBM` and `StereoSGBM` with WLS post-filtering (`cv2.ximgproc.createDisparityWLSFilter`).



* **Agent Adaptation Instructions:**
    * Strip out the interactive OpenCV GUI trackbars (`cv2.createTrackbar`) from the LearnOpenCV script and pass stereo pairs from the Stage 1 loader directly.
    * Divide the raw output of `stereo.compute(left, right)` by `16.0` (`disp = raw_disp.astype(np.float32) / 16.0`) to convert OpenCV's 16-bit fixed-point disparity into true pixel disparity before computing depth.



---

#### Stage 3: Deep Learning-Based (SOTA) Methods

* **Assignment Requirement:** Explore and compare deep learning-based SOTA stereo methods, including the `Stereo Demo` collection, `Accurate and Efficient Stereo Matching via Attention Concatenation Volume` (`ACVNet`), and `Dusk Till Dawn: Self-supervised Nighttime Stereo Depth Estimation using Visual Foundation Models`.


* **Off-the-Shelf Code to Grab:**
    1. Stereo Demo Collection (Primary Turnkey Repo):

        * **Repo:** [https://github.com/nburrus/stereodemo](https://github.com/nburrus/stereodemo).
        * **Why it fits:** Directly specified in the problem statement. It automatically downloads pre-trained weights and runs inference for multiple SOTA deep models (`CREStereo`, `RAFT-Stereo`, `HITNet`, `LEAStereo`) plus OpenCV `BM`/`SGBM` in a single Python package.


    2. ACVNet (Official PyTorch Repo):

        * **Repo:** [https://github.com/gangweiX/ACVNet](https://github.com/gangweiX/ACVNet) (implements G. Xu et al., *Accurate and Efficient Stereo Matching via Attention Concatenation Volume*, IEEE TPAMI 2024).
        * **Files to use:** Use `save_disp.py` and `models/acv.py` with the author-provided KITTI 2015 pre-trained checkpoint (`acvnet_kitti2015.ckpt`) linked in the repository README.
        * **Lightweight ONNX Alternative:** If PyTorch/CUDA version conflicts arise, agents can grab [https://github.com/ibaiGorordo/ONNX-FastACVNet-Depth-Estimation](https://github.com/ibaiGorordo/ONNX-FastACVNet-Depth-Estimation) (`kitti_stereo_depth_estimation.py`), which runs pre-trained Fast-ACVNet in pure ONNX Runtime with zero compilation.


    3. Dusk Till Dawn (Nighttime VFM Stereo Repo):

        - **Repo:** [https://github.com/madhubabuv/dtd](https://github.com/madhubabuv/dtd) (implements Madhu Vankadari et al., *Dusk Till Dawn: Self-supervised Nighttime Stereo Depth Estimation using Visual Foundation Models*, 2024).



* **Agent Adaptation Instructions:**
    - Instead of using `stereodemo`'s interactive UI window, import the model classes directly from `stereodemo/methods.py` into a headless batch runner script.
    - Wrap each model's `predict()` call with `time.perf_counter()` and `torch.cuda.max_memory_allocated()` to log runtime and memory for the computational efficiency comparison.





---

#### Stages 4 & 5: Evaluation Metrics, Foreground Masking & Failure Visualization

- **Assignment Requirement:** Compare methods on accuracy and computational efficiency for a foreground object, identify failure cases where deep learning or both methods fail, and format visual outputs with `Depth Map`, `Left View`, and `Right View`.


- **Off-the-Shelf Code to Grab:**  
    - **Evaluation Metrics:** Grab `evaluate_stereo.py` from [https://github.com/princeton-vl/RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) or `utils/metrics.py` from [https://github.com/gangweiX/ACVNet](https://github.com/gangweiX/ACVNet). Both scripts contain verified implementations of **EPE** (End-Point Error) and **D1-all / Bad-3%** (percentage of pixels with error $> 3$ px).
    - **Visualization:** Grab `save_disp.py` from [https://github.com/gangweiX/ACVNet](https://github.com/gangweiX/ACVNet) or `matplotlib` colormap utilities (`cv2.COLORMAP_JET` / `plt.cm.jet`) to render disparity/depth maps matching the sample `Depth Map` color scheme in the problem statement.




- **Agent Adaptation Instructions:**
    - **Foreground-Specific Metric Slicing:** In `evaluate_stereo.py`, compute metrics twice per image: once over all valid ground-truth pixels (`valid_mask = gt_disp > 0`), and once intersected with the foreground object mask (`fg_valid_mask = (gt_disp > 0) & (fg_mask > 0)`) to report `D1-fg` and `EPE-fg`.

    - **Automated Failure Case Extraction:** Have the evaluation script sort all test images by `D1-fg` descending and automatically export the top 5 worst-performing scenes into a `failure_cases/` folder for qualitative analysis.

    - **3-Column Output Grid:** Use `matplotlib.pyplot.subplots(N, 3)` to concatenate `[Estimated Depth Map, Left View, Right View]` horizontally, replicating the exact layout in Section 4 ("Expected Input and Output") of the problem statement.





---