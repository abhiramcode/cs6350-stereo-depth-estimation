"""Headless inference adapter for Fast-ACVNet stereo matching model.

Fast-ACVNet (Accurate and Efficient Stereo Matching via Attention Concatenation Volume,
IEEE TPAMI 2023 / CVPR 2022) utilizes attention weights to generate a compact 3D cost
volume, significantly reducing computation and memory while maintaining high accuracy.

The ONNX models (converted from the official PyTorch implementation by PINTO0309) run
via ONNX Runtime with zero custom CUDA C++ compilation requirements.
Pre-trained variants include:
- "generalization": Trained on Scene Flow for robust cross-domain generalization (ETH3D, Middlebury).
- "kitti_2015": Fine-tuned on the KITTI 2015 benchmark.
- "sceneflow": Pre-trained on Scene Flow benchmark.

Coordinate System & Resolution Restoration:
    Because disparity represents horizontal pixel displacement (d = x_L - x_R), resizing
    an input image by horizontal scale factor s_x = W_orig / W_model scales the true physical
    disparity linearly by s_x.
    This adapter:
    1. Resizes input RGB pairs to the model's native resolution (W_model, H_model) via cv2.INTER_AREA.
    2. Normalizes pixel values via standard ImageNet mean and std.
    3. Runs ONNX inference to obtain disparity at model resolution.
    4. Resizes the predicted disparity map back to (W_orig, H_orig) via bilinear or nearest interpolation.
    5. Scales the disparity values by s_x = W_orig / W_model to preserve physical disparity coordinates.

Reference:
    - TPAMI 2023 Paper: https://arxiv.org/abs/2209.12699
    - Official PyTorch: https://github.com/gangweiX/Fast-ACVNet
    - ONNX Implementation: https://github.com/ibaiGorordo/ONNX-FastACVNet-Depth-Estimation
    - PINTO0309 Model Zoo: Model 338 (Fast-ACVNet)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import tarfile
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

import cv2
import numpy as np

from src.stereo.base import MatcherConfig, StereoMatcher

logger = logging.getLogger(__name__)

# Pinto Model Zoo Wasabi S3 tar archive containing pre-converted Fast-ACVNet ONNX models
FAST_ACVNET_ARCHIVE_URL = (
    "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/338_Fast-ACVNet/fast_acvnet_plus_onnx_gridsample.tar.gz"
)

# Standard ImageNet normalization parameters used by Fast-ACVNet
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


@dataclass
class ACVNetConfig(MatcherConfig):
    """Configuration for Fast-ACVNet ONNX inference.

    Attributes:
        method_name: Identifier for logging/benchmarking (e.g. "Fast-ACVNet").
        model_variant: "generalization" (Scene Flow cross-domain), "kitti_2015", or "sceneflow".
        resolution: "640x480", "512x288", "960x544", or "320x192" (Width x Height).
        interpolation: Disparity upsampling interpolation ("linear" or "nearest").
        checkpoint_path: Explicit path to .onnx file. If None, resolves to checkpoints/fast_acvnet/.
        device: "auto", "cuda", or "cpu".
        auto_download: Whether to stream and extract checkpoint from official archive if missing.
    """

    method_name: str = "Fast-ACVNet"
    model_variant: str = "generalization"
    resolution: str = "640x480"
    interpolation: str = "linear"
    checkpoint_path: Optional[str] = None
    device: str = "auto"
    auto_download: bool = True


class ACVNetMatcher(StereoMatcher):
    """Headless ONNX Runtime adapter for Fast-ACVNet stereo matching."""

    def __init__(self, config: Optional[ACVNetConfig] = None) -> None:
        config = config or ACVNetConfig()
        super().__init__(config)
        self.config: ACVNetConfig = config

        self._model_path = self._resolve_model_path()
        self._ensure_checkpoint()

        self._session, self._active_providers = self._init_session()

        # Parse model I/O signatures
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        self._input_names = [inp.name for inp in inputs]
        self._output_names = [out.name for out in outputs]

        # Expected input shape: [1, 3, H, W]
        in_shape = inputs[0].shape
        self._expected_channels = in_shape[1]
        self._model_height = in_shape[2]
        self._model_width = in_shape[3]

        # Fast-ACVNet operates on 3-channel RGB images
        self.input_color = "rgb"

        logger.info(
            "Loaded Fast-ACVNet model from %s [inputs: %s, shape: (%dx%d, C=%d), providers: %s]",
            self._model_path.name,
            self._input_names,
            self._model_width,
            self._model_height,
            self._expected_channels,
            self._active_providers,
        )

    def _resolve_model_path(self) -> Path:
        if self.config.checkpoint_path:
            return Path(self.config.checkpoint_path)

        cols, rows = self.config.resolution.lower().split("x")
        filename = f"fast_acvnet_plus_{self.config.model_variant.lower()}_opset16_{rows}x{cols}.onnx"
        return Path("checkpoints/fast_acvnet") / filename

    def _ensure_checkpoint(self) -> None:
        if self._model_path.is_file():
            return

        filename = self._model_path.name

        if not self.config.auto_download:
            raise FileNotFoundError(
                f"Fast-ACVNet checkpoint missing at {self._model_path}. "
                f"Enable auto_download=True or obtain '{filename}' from {FAST_ACVNET_ARCHIVE_URL}"
            )

        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Fast-ACVNet checkpoint %s missing. Streaming extraction from archive %s...",
            filename,
            FAST_ACVNET_ARCHIVE_URL,
        )

        req = urllib.request.Request(
            FAST_ACVNET_ARCHIVE_URL,
            headers={"User-Agent": "Mozilla/5.0 (StereoBenchmark-Agent)"},
        )
        extracted = False
        with urllib.request.urlopen(req) as response:
            with tarfile.open(fileobj=response, mode="r|gz") as tar:
                for member in tar:
                    if member.name.endswith(filename):
                        logger.info("Found %s in archive (%.1f MB). Extracting...", member.name, member.size / 1e6)
                        member_file = tar.extractfile(member)
                        if member_file is not None:
                            with open(self._model_path, "wb") as out_f:
                                out_f.write(member_file.read())
                            extracted = True
                            break

        if not extracted or not self._model_path.is_file() or self._model_path.stat().st_size < 1_000_000:
            if self._model_path.is_file():
                self._model_path.unlink()
            raise RuntimeError(
                f"Failed to extract valid Fast-ACVNet checkpoint for {filename} from {FAST_ACVNET_ARCHIVE_URL}"
            )

        logger.info("Successfully extracted %s (%.1f MB)", filename, self._model_path.stat().st_size / 1e6)

    def _init_session(self):
        import onnxruntime as ort

        device = self.config.device.lower()
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:  # "auto"
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(str(self._model_path), sess_options=opts, providers=providers)
        active_providers = session.get_providers()
        return session, active_providers

    def get_hw_info(self) -> Dict[str, Any]:
        info = super().get_hw_info()
        info["device"] = self.config.device
        info["onnx_providers"] = self._active_providers
        return info

    def _compute_disparity(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        """Run Fast-ACVNet inference and return disparity in original image pixel units.

        Args:
            left:  ``(H, W, 3) uint8`` RGB image or ``(H, W) uint8`` grayscale image.
            right: ``(H, W, 3) uint8`` RGB image or ``(H, W) uint8`` grayscale image.

        Returns:
            ``(H, W) float32`` disparity map in original pixel coordinates.
        """
        orig_h, orig_w = left.shape[:2]

        # 1. Ensure 3-channel RGB
        if left.ndim == 2:
            left_rgb = cv2.cvtColor(left, cv2.COLOR_GRAY2RGB)
            right_rgb = cv2.cvtColor(right, cv2.COLOR_GRAY2RGB)
        elif left.shape[2] == 1:
            left_rgb = cv2.cvtColor(left[:, :, 0], cv2.COLOR_GRAY2RGB)
            right_rgb = cv2.cvtColor(right[:, :, 0], cv2.COLOR_GRAY2RGB)
        else:
            left_rgb = left
            right_rgb = right

        # 2. Resize to native model resolution (W_model, H_model)
        target_size = (self._model_width, self._model_height)
        left_resized = cv2.resize(left_rgb, target_size, interpolation=cv2.INTER_AREA)
        right_resized = cv2.resize(right_rgb, target_size, interpolation=cv2.INTER_AREA)

        # 3. Standard ImageNet normalization: (img / 255.0 - mean) / std
        left_norm = (left_resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        right_norm = (right_resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

        # 4. Transpose to [1, 3, H, W]
        left_tensor = np.transpose(left_norm, (2, 0, 1))[np.newaxis, ...]
        right_tensor = np.transpose(right_norm, (2, 0, 1))[np.newaxis, ...]

        # 5. ONNX Inference
        outputs = self._session.run(
            self._output_names,
            {self._input_names[0]: left_tensor, self._input_names[1]: right_tensor},
        )
        raw_disp = outputs[0][0, 0]  # Shape (H_model, W_model)

        # 6. Resolution restoration & coordinate system preservation
        # Disparity is horizontal pixel displacement: d_orig = d_model * (W_orig / W_model)
        x_scale = float(orig_w) / float(self._model_width)
        interp_mode = cv2.INTER_LINEAR if self.config.interpolation.lower() == "linear" else cv2.INTER_NEAREST
        disp_resized = cv2.resize(raw_disp, (orig_w, orig_h), interpolation=interp_mode)
        disp_orig = disp_resized * np.float32(x_scale)

        # 7. Mask invalid / negative / non-finite values
        disp_orig[disp_orig <= 0] = 0.0
        disp_orig[~np.isfinite(disp_orig)] = 0.0

        return disp_orig.astype(np.float32)
