# Technical Decisions Log

## D1: Disparity Decoding Conventions

**Decision:** Enforce exact format-specific decoding in `io_utils.py` and `src/stereo/`:
- KITTI 2015: `disp = uint16_png / 256.0`, `valid = raw > 0`
- ETH3D: PFM with endianness from scale header, `valid = isfinite(d) & (d > 0)`
- OpenCV BM/SGBM (Stage 2): raw 16-bit fixed-point `disp = raw.astype(np.float32) / 16.0`, invalid `disp <= 0`

**Rationale:** Mixing up 256 vs 16 scaling destroys metric calculations. Verified against RAFT-Stereo `frame_utils.py` and OpenCV specifications.

## D2: Train/Eval Split Strategy

**Decision:** Deterministic fixed splits (no random seed required):
- KITTI 2015: indices 0–39 = `val_tune` (20%), 40–199 = `eval_test` (80%)
- ETH3D: first 5 scenes alphabetically (`delivery_area_1l` through `delivery_area_3l`) = `val_tune`, remaining 22 = `eval_test`
- Official `test` splits → `test_visual` (qualitative only, no ground truth public)

**Rationale:** Complete reproducibility across agent turns and user runs without seed configuration. 80/20 matches ACTION_PLAN specification.

## D3: Foreground Mask Provenance

**Decision:** Every foreground mask carries a strict provenance tag:
- `semantic_gt`: KITTI `obj_map` ground-truth annotations
- `geometric_proxy`: median-disparity threshold (explicitly not semantic)
- `pretrained_seg`: Mask R-CNN / DeepLabV3 output (deferred to Stage 4/5)
- `unavailable`: no mask exists

**Rationale:** Prevents conflating geometric heuristics with semantic annotations (Verdicts issue #1).

## D4: ETH3D Calibration Format & Parsing

**Decision:** Prefer Middlebury-style `calib.txt` co-located in scene directories, with COLMAP (`cameras.txt`, `images.txt`) fallback:
- In `calib.txt`: `baseline` is provided in **millimetres** and must be converted to **metres** ($B_{\text{m}} = B / 1000.0$).
- Focal length $f_{\text{px}}$ is parsed from `cam0` intrinsic matrix $[0, 0]$.
- Principal point x-offset `doffs` is preserved.

**Rationale:** Real ETH3D dataset downloads supply `calib.txt` directly. Converting mm to metres is critical for metric depth $Z = fB/d$.

## D5: Missing Data Policy

**Decision:** Never fabricate. All optional fields default to `None`. The `CustomPairDataset` returns `None` for GT, valid_mask, fg_mask, and calib when files are absent.

**Rationale:** Master rule: "Never invent successful test results, benchmark numbers, downloaded files."

## D6: Depth Conversion Safety

**Decision:** `Calibration.disp_to_depth()` replaces zero/negative/NaN/inf disparities with `inf` before dividing, producing `depth ≈ 0` for invalid pixels.

**Rationale:** Prevents division-by-zero crashes; downstream metrics cap depth at valid ranges ($0.5\text{ m} \le Z \le 80\text{ m}$).

## D7: ETH3D Flat Directory Layout Support

**Decision:** `ETH3DDataset` auto-detects both flat layout (actual official download where all 47 scene directories reside directly under `data/eth3d/`) and nested layout (`two_view_training/`, `two_view_training_gt/`, `two_view_test/`).

**Rationale:** Real downloads from `www.eth3d.net` extract all scene folders into the root directory. Detecting both prevents manual re-structuring by users.

## D8: Matcher Interface & Isolated Timing Policy

**Decision:** Base class `StereoMatcher` and result dataclass `StereoResult`:
- Strictly isolate pure inference time (`elapsed_s`) using `time.perf_counter()` directly wrapping `_compute_disparity`.
- Preprocessing (RGB $\to$ grayscale uint8) and post-processing (fixed-point $\div 16.0$, validity masking, metric depth conversion) are tracked separately in `total_elapsed_s`.
- Benchmark runs execute 3 unmeasured warm-up iterations on the first sample before logging latency.

**Rationale:** Required by Stage 4/Action Plan benchmarking protocols: model loading, dataset I/O, and disk writes must never contaminate algorithmic throughput and latency metrics.

## D9: Fair Valid-Pixel Policy for Classical Methods

**Decision:**
- Shallow methods (`StereoBM`, `StereoSGBM`) produce sparse invalid holes (`disp <= 0`) in occlusions, textureless regions, or failed uniqueness checks.
- Report **Prediction Density (%)** alongside all evaluation metrics.
- Disparity accuracy (EPE, Bad-1.0/2.0/3.0, D1-all) is evaluated on mutually valid pixels (`valid_gt & (pred > 0)`), preventing arbitrary interpolation bias while maintaining complete transparency regarding prediction sparsity.

**Rationale:** Follows Section 4 of ACTION_PLAN to ensure fair, objective comparisons with dense deep learning methods.

## D10: Hyperparameter Tuning Composite Scoring

**Decision:** In `scripts/tune_shallow_parameters.py`, parameter candidates on `val_tune` are ranked using a composite objective:
$$\text{Score} = \text{Bad-1.0} + 0.3 \times (100 - \text{Density})$$
(Lower is better).

**Rationale:** Simply minimizing EPE or Bad-1.0 causes classical matchers to set aggressive uniqueness and texture thresholds, rejecting 80%+ of pixels and reporting artificially low error on a tiny subset. The composite penalty encourages low error while penalizing low valid coverage.

## D11: Disparity Coordinate Scaling on Spatial Resizing

**Decision:** When deep learning models require fixed-dimension inputs (e.g. 640x480 for HITNet and Fast-ACVNet, 320x192 / 640x384 for Dusk Till Dawn), input RGB images are resized via `cv2.INTER_AREA`. After forward inference, the raw disparity map is resized back to $(W_{\text{orig}}, H_{\text{orig}})$ using linear interpolation and scaled by the horizontal aspect ratio:
$$d_{\text{orig}}(x, y) = d_{\text{model}}(x', y') \times \frac{W_{\text{orig}}}{W_{\text{model}}}$$

**Rationale:** Disparity $d$ represents horizontal shift ($x_L - x_R$) in pixel units. Spatial scaling of image coordinates $x \to x \cdot s_x$ scales the physical pixel disparity by exactly $s_x = W_{\text{orig}} / W_{\text{model}}$. Omitting this scaling factor introduces catastrophic systematic error proportional to the resizing ratio.

## D12: Fast-ACVNet+ ONNX Export & Streamed Weight Extraction

**Decision:** Utilize PINTO0309's validated ONNX exports of Fast-ACVNet+ (opset 16, fixed 480x640 shape). To avoid multi-gigabyte tar archives or external extraction tools, implement a Python streaming HTTP tar extractor in `ACVNetMatcher._ensure_checkpoint()` that extracts the exact target `.onnx` file directly into `checkpoints/fast_acvnet/`.

**Rationale:** Zero external manual setup required for users while maintaining complete determinism and compatibility with Python 3.13 without native compilation tools.

## D13: Dusk Till Dawn Spatial Dimension Constraints

**Decision:** DTD pairs DINO ViT-S/8 feature extraction (stride 4, patch 8) with UniMatch cross-attention (window splits $N=8$). Consequently, input spatial dimensions $(H, W)$ must be integer multiples of 64 ($8 \times 8$). Supported modes are configured as:
- `DuskTillDawn_Fast`: $320 \times 192$ (fast inference, $\sim 2.7$s on CPU)
- `DuskTillDawn_HighRes`: $640 \times 384$ (high-resolution, $\sim 34$s on CPU)

**Rationale:** Arbitrary dimensions fail UniMatch's internal tensor splitting assertions (`H % num_splits == 0`). Enforcing explicit multiples of 64 guarantees numerical stability without runtime exceptions.

## D14: PyTorch CPU-only Footprint Preservation on Limited Disk Budget

**Decision:** Install `torch==2.14.1+cpu` and `torchvision==0.29.1+cpu` in the `.venv` environment, consuming $\sim 190$ MB disk space instead of the full CUDA bundle ($\sim 2.5$ GB).

**Rationale:** Workspace filesystem has $\sim 7.6$ GB remaining disk space. Avoiding multi-gigabyte CUDA runtime wheels preserves disk budget while keeping the entire test suite and inference pipeline fully operational and lightning-fast.

## D15: DINO Hub Loading Rate-Limit Resilience

**Decision:** Pass `skip_validation=True` to `torch.hub.load("facebookresearch/dino:main", "dino_vits8", skip_validation=True)`.

**Rationale:** PyTorch Hub queries the GitHub API to check repository forks and hits HTTP 403 rate limits on CI/CD and unauthenticated developer machines. Setting `skip_validation=True` forces PyTorch Hub to use local cache directly without external API calls.

