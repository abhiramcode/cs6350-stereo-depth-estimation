"""Headless inference adapter for Dusk Till Dawn (DTD) stereo matching model.

Dusk Till Dawn (Self-supervised Nighttime Stereo Depth Estimation using Visual
Foundation Models, ICRA 2024) leverages features extracted from visual foundation
models (DINO ViT-S/8) coupled with cross-attention transformer matching (UniMatch)
and self-supervised geometry constraints for robust stereo depth estimation.

Coordinate System & Resolution Restoration:
    Because disparity represents horizontal pixel displacement (d = x_L - x_R), resizing
    an input image by horizontal scale factor s_x = W_orig / W_model scales the true physical
    disparity linearly by s_x.
    This adapter:
    1. Resizes input RGB pairs to the model's native resolution (W_model, H_model) via cv2.INTER_AREA.
       Note: Both W_model and H_model must be integer multiples of 64 to satisfy DINO patch stride
       and transformer window split requirements.
    2. Normalizes pixel values via standard ImageNet mean and std.
    3. Extracts DINO ViT-S/8 token representations across layers [5, 11].
    4. Runs DTD cross-attention matching and convex upsampling.
    5. Resizes the predicted disparity back to (W_orig, H_orig) and scales by s_x = W_orig / W_model.

Reference:
    - ICRA 2024 Paper: https://arxiv.org/abs/2404.03579
    - Source Repository: https://github.com/madhubabuv/dtd
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.stereo.base import MatcherConfig, StereoMatcher

logger = logging.getLogger(__name__)

# Official DTD checkpoint from Oxford / ICRA 2024 authors
DTD_GDRIVE_FILE_ID = "1dgUKBf-UKpOZp_3681iS_v3hgBs4Ip9a"
DTD_DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={DTD_GDRIVE_FILE_ID}"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


@dataclass
class DTDConfig(MatcherConfig):
    """Configuration for Dusk Till Dawn (DTD) inference.

    Attributes:
        method_name: Identifier for logging/benchmarking (e.g. "DuskTillDawn").
        resolution: "320x192" (fast, default) or "640x384" (Width x Height, multiples of 64).
        interpolation: Disparity upsampling interpolation ("linear" or "nearest").
        checkpoint_path: Explicit path to .pth file. If None, resolves to checkpoints/dtd/depth_net_20.pth.
        device: "auto", "cuda", or "cpu".
        auto_download: Whether to fetch checkpoint from Google Drive if missing.
    """

    method_name: str = "DuskTillDawn"
    resolution: str = "320x192"
    interpolation: str = "linear"
    checkpoint_path: Optional[str] = None
    device: str = "auto"
    auto_download: bool = True


class DTDMatcher(StereoMatcher):
    """Headless PyTorch adapter for Dusk Till Dawn (DTD) stereo matching."""

    def __init__(self, config: Optional[DTDConfig] = None) -> None:
        config = config or DTDConfig()
        super().__init__(config)
        self.config: DTDConfig = config

        self._model_path = self._resolve_model_path()
        self._ensure_checkpoint()

        # Parse resolution (Width x Height)
        cols_str, rows_str = self.config.resolution.lower().split("x")
        self._model_width = int(cols_str)
        self._model_height = int(rows_str)

        if self._model_width % 64 != 0 or self._model_height % 64 != 0:
            raise ValueError(
                f"DTD resolution {self.config.resolution} ({self._model_width}x{self._model_height}) "
                "must be integer multiples of 64 for DINO patch and transformer window splits."
            )

        # Device resolution
        device_req = self.config.device.lower()
        if device_req == "cuda" and torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif device_req == "auto" and torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        # Initialize model
        self._model = self._init_model()

        # DTD operates on 3-channel RGB images
        self.input_color = "rgb"

        logger.info(
            "Loaded Dusk Till Dawn model from %s [resolution: %dx%d, device: %s]",
            self._model_path.name,
            self._model_width,
            self._model_height,
            self._device,
        )

    def _resolve_model_path(self) -> Path:
        if self.config.checkpoint_path:
            return Path(self.config.checkpoint_path)
        return Path("checkpoints/dtd/depth_net_20.pth")

    def _ensure_checkpoint(self) -> None:
        if self._model_path.is_file():
            return

        if not self.config.auto_download:
            raise FileNotFoundError(
                f"DTD checkpoint missing at {self._model_path}. "
                f"Enable auto_download=True or download from {DTD_DOWNLOAD_URL}"
            )

        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("DTD checkpoint missing at %s. Downloading from Google Drive...", self._model_path)

        req = urllib.request.Request(DTD_DOWNLOAD_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(self._model_path, "wb") as out_f:
            out_f.write(resp.read())

        if not self._model_path.is_file() or self._model_path.stat().st_size < 1_000_000:
            if self._model_path.is_file():
                self._model_path.unlink()
            raise RuntimeError(f"Failed to download valid DTD checkpoint to {self._model_path}")

        logger.info("Downloaded DTD checkpoint (%.1f MB)", self._model_path.stat().st_size / 1e6)

    def _init_model(self):
        from src.stereo.models.dtd.depth_model import StereoDepthNet

        class DTDArgs:
            device = str(self._device)

        net = StereoDepthNet(args=DTDArgs(), device=self._device)

        # Load weights
        ckpt = torch.load(self._model_path, map_location=self._device, weights_only=False)
        state_dict = {}
        for k, v in ckpt.items():
            k_clean = k.replace("model.", "")
            state_dict[k_clean] = v

        net.model.load_state_dict(state_dict, strict=False)
        net.eval()
        return net

    def get_hw_info(self) -> Dict[str, Any]:
        info = super().get_hw_info()
        info["device"] = str(self._device)
        info["backend"] = "pytorch"
        info["cuda_available"] = torch.cuda.is_available()
        return info

    def _compute_disparity(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        """Run DTD inference and return disparity in original image pixel units.

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

        # 2. Resize to model native resolution
        target_size = (self._model_width, self._model_height)
        l_resized = cv2.resize(left_rgb, target_size, interpolation=cv2.INTER_AREA)
        r_resized = cv2.resize(right_rgb, target_size, interpolation=cv2.INTER_AREA)

        # 3. Standard ImageNet normalization: (img / 255.0 - mean) / std
        l_norm = (l_resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        r_norm = (r_resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

        # 4. Convert to PyTorch tensors [1, 3, H, W] on target device
        l_tensor = torch.from_numpy(l_norm).permute(2, 0, 1).unsqueeze(0).float().to(self._device)
        r_tensor = torch.from_numpy(r_norm).permute(2, 0, 1).unsqueeze(0).float().to(self._device)

        # 5. Model forward pass
        with torch.no_grad():
            outputs, _, _ = self._model(l_tensor, r_tensor)
            raw_disp_tensor = outputs[0][0].detach().cpu()
            raw_disp = raw_disp_tensor.numpy()

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
