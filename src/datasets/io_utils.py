"""Low-level I/O helpers for stereo dataset formats.

* KITTI 2015: 16-bit PNG disparity (÷ 256.0, 0 = invalid)
* ETH3D:      PFM disparity (handles both endianness variants)
* Generic:    Image reading with grayscale → RGB promotion

All functions are deterministic and side-effect-free.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np


# ===================================================================== #
#  Image I/O
# ===================================================================== #

def read_image_rgb(path: str | Path) -> np.ndarray:
    """Read an image file and return an ``(H, W, 3)`` uint8 RGB array.

    Grayscale images are replicated to 3 channels.  RGBA images are
    truncated to RGB.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")

    if img.ndim == 2:
        # grayscale → 3-channel
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    # OpenCV loads BGR — convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)


# ===================================================================== #
#  KITTI 2015 disparity PNG
# ===================================================================== #

def read_kitti_disp_png(
    path: str | Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read a KITTI 2015 16-bit disparity PNG.

    Returns:
        disp:  ``float32`` disparity in **pixels** (``uint16 / 256.0``).
        valid: ``bool`` mask — ``True`` where disparity is valid (``raw > 0``).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"KITTI disp PNG not found: {path}")

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != np.uint16:
        raise ValueError(
            f"Expected 16-bit PNG, got {raw.dtype if raw is not None else 'None'}: {path}"
        )

    valid = raw > 0
    disp = raw.astype(np.float32) / 256.0
    disp[~valid] = 0.0
    return disp, valid


def read_kitti_obj_map(path: str | Path) -> Optional[np.ndarray]:
    """Read a KITTI 2015 ``obj_map`` PNG and return a boolean foreground mask.

    Foreground = any pixel with label > 0 (background / road = 0).
    Returns ``None`` if the file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        return None

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None

    return raw > 0  # bool array


# ===================================================================== #
#  ETH3D PFM disparity
# ===================================================================== #

def read_pfm(path: str | Path) -> np.ndarray:
    """Read a Portable FloatMap (``.pfm``) file.

    Handles both little-endian (negative scale) and big-endian (positive
    scale) variants.  The returned array has shape ``(H, W)`` or ``(H, W, 3)``
    and dtype ``float32``.

    Reference: RAFT-Stereo ``frame_utils.readPFM``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PFM file not found: {path}")

    with open(path, "rb") as f:
        header = f.readline().rstrip()
        if header == b"PF":
            color = True
        elif header == b"Pf":
            color = False
        else:
            raise ValueError(f"Not a PFM file (header={header!r}): {path}")

        dim_line = f.readline()
        dim_match = re.match(rb"^(\d+)\s(\d+)\s$", dim_line)
        if not dim_match:
            raise ValueError(f"Malformed PFM dimension line: {dim_line!r}")
        width, height = int(dim_match.group(1)), int(dim_match.group(2))

        scale = float(f.readline().rstrip())
        endian = "<" if scale < 0 else ">"
        scale = abs(scale)

        data = np.fromfile(f, endian + "f")

    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    # PFM stores rows bottom-to-top
    data = np.flipud(data).copy()

    if scale != 1.0:
        data *= scale

    return data.astype(np.float32)


def read_eth3d_disp_pfm(
    path: str | Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read an ETH3D disparity PFM and compute a validity mask.

    Returns:
        disp:  ``float32`` disparity in pixels.
        valid: ``bool`` mask — ``True`` where disparity is finite and > 0.
    """
    disp = read_pfm(path)
    if disp.ndim == 3:
        # take first channel if colour PFM
        disp = disp[:, :, 0]
    valid = np.isfinite(disp) & (disp > 0)
    return disp, valid


# ===================================================================== #
#  KITTI calibration parser
# ===================================================================== #

def parse_kitti_calib(calib_path: str | Path) -> Tuple[float, float]:
    """Parse a KITTI ``calib_cam_to_cam.txt`` file.

    Extracts:
        f_px:       Focal length from ``P_rect_02`` (element ``[0,0]``).
        baseline_m: Baseline in metres from the x-translation difference
                    between ``P_rect_02`` and ``P_rect_03``.

    The 3×4 projection matrices are stored as 12 space-separated floats::

        P_rect_02: f 0 cx tx  0 f cy 0  0 0 1 0
        P_rect_03: f 0 cx tx' 0 f cy 0  0 0 1 0

    where ``tx = -f * baseline_02`` and ``tx' = -f * baseline_03``.
    Baseline = ``|tx' - tx| / f``.
    """
    calib_path = Path(calib_path)
    if not calib_path.is_file():
        raise FileNotFoundError(f"KITTI calib file not found: {calib_path}")

    p_rect: dict[str, list[float]] = {}
    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Lines look like:  P_rect_02: 7.21e+02 0.00 ...
            if ":" in line:
                key, vals = line.split(":", 1)
                key = key.strip()
                if key in ("P_rect_02", "P_rect_03"):
                    p_rect[key] = [float(v) for v in vals.strip().split()]

    if "P_rect_02" not in p_rect or "P_rect_03" not in p_rect:
        raise ValueError(
            f"Could not find P_rect_02 and P_rect_03 in {calib_path}"
        )

    p2 = p_rect["P_rect_02"]  # 12 values → 3×4
    p3 = p_rect["P_rect_03"]

    f_px = p2[0]  # P[0,0] = focal length
    # tx_02 = P_rect_02[0,3] = -f * b_02  →  b_02 = -tx_02 / f
    # tx_03 = P_rect_03[0,3] = -f * b_03  →  b_03 = -tx_03 / f
    # baseline = b_03 - b_02  (b_02 is typically 0)
    tx_02 = p2[3]
    tx_03 = p3[3]
    baseline_m = abs(tx_03 - tx_02) / f_px

    return f_px, baseline_m


# ===================================================================== #
#  ETH3D COLMAP calibration parser
# ===================================================================== #

def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert a COLMAP quaternion ``(qw, qx, qy, qz)`` to a 3×3 rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx**2 + qz**2),  2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),      1 - 2*(qx**2 + qy**2)],
    ])


def parse_eth3d_calib(
    cameras_txt: str | Path,
    images_txt: str | Path,
) -> Tuple[float, float]:
    """Parse ETH3D COLMAP-format calibration files.

    Args:
        cameras_txt: Path to ``cameras.txt`` (PINHOLE intrinsics).
        images_txt:  Path to ``images.txt`` (extrinsics with quaternion + t).

    Returns:
        f_px:       Focal length from the first camera entry (``fx``).
        baseline_m: Euclidean distance between the two camera centres.
    """
    cameras_txt = Path(cameras_txt)
    images_txt = Path(images_txt)

    # --- Parse cameras.txt ---
    f_px = None
    with open(cameras_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
            model = parts[1]
            if model == "PINHOLE":
                # params: fx fy cx cy
                f_px = float(parts[4])
            else:
                # For other models, fx is still typically the first param
                f_px = float(parts[4])
            break  # take the first camera

    if f_px is None:
        raise ValueError(f"No camera found in {cameras_txt}")

    # --- Parse images.txt ---
    # Each image has two lines:
    #   IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    #   POINTS2D[] (may be empty)
    centres = []
    with open(images_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 10:
                # This is an image line (not 2D-points line)
                qvec = np.array([float(parts[1]), float(parts[2]),
                                 float(parts[3]), float(parts[4])])
                tvec = np.array([float(parts[5]), float(parts[6]),
                                 float(parts[7])])
                R = _qvec_to_rotmat(qvec)
                centre = -R.T @ tvec
                centres.append(centre)

    if len(centres) < 2:
        raise ValueError(
            f"Expected ≥ 2 images in {images_txt}, found {len(centres)}"
        )

    baseline_m = float(np.linalg.norm(centres[0] - centres[1]))
    return f_px, baseline_m


# ===================================================================== #
#  ETH3D Middlebury-style calib.txt parser
# ===================================================================== #

def parse_eth3d_calib_txt(calib_txt: str | Path) -> Tuple[float, float, float, int]:
    """Parse an ETH3D Middlebury-style ``calib.txt`` file.

    Format::

        cam0=[fx 0 cx; 0 fy cy; 0 0 1]
        cam1=[fx 0 cx; 0 fy cy; 0 0 1]
        doffs=<float>         # x-offset difference between cam0 and cam1
        baseline=<float>      # baseline in **millimetres**
        width=<int>
        height=<int>
        ndisp=<int>           # recommended max disparity

    Returns:
        (f_px, baseline_m, doffs, ndisp)
    """
    calib_txt = Path(calib_txt)
    if not calib_txt.is_file():
        raise FileNotFoundError(f"ETH3D calib.txt not found: {calib_txt}")

    fields: dict[str, str] = {}
    with open(calib_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                fields[key.strip()] = val.strip()

    # Extract focal length from cam0 matrix string:
    #   "[fx 0 cx; 0 fy cy; 0 0 1]"
    cam0_str = fields.get("cam0", "")
    cam0_str = cam0_str.strip("[]")
    cam0_vals = [float(v) for v in cam0_str.replace(";", " ").split()]
    f_px = cam0_vals[0]  # fx

    baseline_mm = float(fields.get("baseline", "0"))
    baseline_m = baseline_mm / 1000.0  # mm → m

    doffs = float(fields.get("doffs", "0"))
    ndisp = int(fields.get("ndisp", "256"))

    return f_px, baseline_m, doffs, ndisp

