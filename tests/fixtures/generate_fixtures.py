"""Generate tiny synthetic test fixtures for unit tests.

This module creates minimal files that replicate the exact binary formats
of KITTI 2015 and ETH3D datasets, enabling offline testing without any
real dataset downloads.

Run directly to regenerate fixtures::

    python -m tests.fixtures.generate_fixtures
"""

from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np


FIXTURES_DIR = Path(__file__).parent


def _make_tiny_image(h: int, w: int, seed: int = 42) -> np.ndarray:
    """Create a tiny deterministic RGB image."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


# ===================================================================== #
#  KITTI 2015 fixtures
# ===================================================================== #

def create_kitti_fixtures(base: Path, num_scenes: int = 3) -> None:
    """Create minimal KITTI 2015 training directory structure."""
    H, W = 48, 64  # tiny resolution

    for subdir in ("image_2", "image_3", "disp_occ_0", "obj_map", "calib_cam_to_cam"):
        (base / "training" / subdir).mkdir(parents=True, exist_ok=True)

    for idx in range(num_scenes):
        tag = f"{idx:06d}_10"
        seed = idx * 10

        # Left and right images
        left = _make_tiny_image(H, W, seed)
        right = _make_tiny_image(H, W, seed + 1)
        cv2.imwrite(str(base / "training" / "image_2" / f"{tag}.png"),
                     cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(base / "training" / "image_3" / f"{tag}.png"),
                     cv2.cvtColor(right, cv2.COLOR_RGB2BGR))

        # 16-bit disparity PNG: values in [0, 25600] → disp in [0, 100] px
        rng = np.random.RandomState(seed + 2)
        raw_disp = rng.randint(0, 25601, (H, W), dtype=np.uint16)
        # Set 20% of pixels to 0 (invalid)
        invalid_mask = rng.random((H, W)) < 0.2
        raw_disp[invalid_mask] = 0
        cv2.imwrite(str(base / "training" / "disp_occ_0" / f"{tag}.png"), raw_disp)

        # Object map: some foreground (>0), some background (0)
        obj = np.zeros((H, W), dtype=np.uint8)
        obj[10:30, 15:50] = 1  # a rectangular "object"
        cv2.imwrite(str(base / "training" / "obj_map" / f"{tag}.png"), obj)

        # Calibration file
        f_px = 721.5377
        # P_rect_02: f 0 cx tx   0 f cy 0   0 0 1 0
        # baseline_02 = 0 → tx_02 = 0
        # baseline_03 = 0.5372 → tx_03 = -f * 0.5372
        tx_02 = 0.0
        tx_03 = -f_px * 0.5372
        calib_path = base / "training" / "calib_cam_to_cam" / f"{idx:06d}.txt"
        with open(calib_path, "w") as f:
            f.write(f"P_rect_02: {f_px:.4f} 0.0000 6.095593e+02 {tx_02:.4f} "
                    f"0.0000 {f_px:.4f} 1.728540e+02 0.0000 "
                    f"0.0000 0.0000 1.0000 0.0000\n")
            f.write(f"P_rect_03: {f_px:.4f} 0.0000 6.095593e+02 {tx_03:.4f} "
                    f"0.0000 {f_px:.4f} 1.728540e+02 0.0000 "
                    f"0.0000 0.0000 1.0000 0.0000\n")


# ===================================================================== #
#  ETH3D fixtures
# ===================================================================== #

def write_pfm(path: Path, data: np.ndarray, little_endian: bool = True) -> None:
    """Write a PFM file with the specified endianness."""
    H, W = data.shape[:2]
    color = data.ndim == 3

    with open(path, "wb") as f:
        f.write(b"PF\n" if color else b"Pf\n")
        f.write(f"{W} {H}\n".encode())
        scale = -1.0 if little_endian else 1.0
        f.write(f"{scale}\n".encode())

        # PFM stores bottom-to-top
        flipped = np.flipud(data).astype(np.float32)
        if not little_endian:
            flipped = flipped.byteswap()
        flipped.tofile(f)


def create_eth3d_fixtures(
    base: Path,
    num_scenes: int = 3,
    big_endian_scene: int = 1,
) -> None:
    """Create minimal ETH3D two-view directory structure."""
    H, W = 32, 48
    scene_names = [f"test_scene_{i}" for i in range(num_scenes)]

    for scene in scene_names:
        img_dir = base / "two_view_training" / scene
        gt_dir = base / "two_view_training_gt" / scene
        cal_dir = base / "calibration" / scene
        img_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        cal_dir.mkdir(parents=True, exist_ok=True)

    for i, scene in enumerate(scene_names):
        seed = 100 + i * 10
        img_dir = base / "two_view_training" / scene
        gt_dir = base / "two_view_training_gt" / scene
        cal_dir = base / "calibration" / scene

        # Images
        left = _make_tiny_image(H, W, seed)
        right = _make_tiny_image(H, W, seed + 1)
        cv2.imwrite(str(img_dir / "im0.png"), cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(img_dir / "im1.png"), cv2.cvtColor(right, cv2.COLOR_RGB2BGR))

        # Disparity PFM
        rng = np.random.RandomState(seed + 2)
        disp = rng.uniform(1.0, 80.0, (H, W)).astype(np.float32)
        # Set some pixels to inf (invalid)
        disp[0, 0] = np.inf
        disp[H - 1, W - 1] = -1.0  # negative = invalid

        is_big = (i == big_endian_scene)
        write_pfm(gt_dir / "disp0GT.pfm", disp, little_endian=not is_big)

        # COLMAP cameras.txt
        f_px = 500.0 + i * 10
        with open(cal_dir / "cameras.txt", "w") as f:
            f.write("# Camera list\n")
            f.write(f"1 PINHOLE {W} {H} {f_px} {f_px} {W/2} {H/2}\n")
            f.write(f"2 PINHOLE {W} {H} {f_px} {f_px} {W/2} {H/2}\n")

        # COLMAP images.txt — two cameras with known baseline
        baseline = 0.12 + i * 0.01  # metres
        with open(cal_dir / "images.txt", "w") as f:
            f.write("# Image list\n")
            # Camera 1 at origin: q=(1,0,0,0), t=(0,0,0)
            f.write(f"1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 im0.png\n")
            f.write("\n")  # empty 2D points line
            # Camera 2 translated along x-axis by baseline
            # For world-to-camera: t = -R @ C → with R=I, t_x = -baseline
            # But we want C2 = (baseline, 0, 0) → t2 = -R@C2 = (-baseline, 0, 0)
            f.write(f"2 1.0 0.0 0.0 0.0 {-baseline:.6f} 0.0 0.0 2 im1.png\n")
            f.write("\n")

    return scene_names


# ===================================================================== #
#  Custom pair fixtures
# ===================================================================== #

def create_custom_fixtures(base: Path, num_pairs: int = 2) -> None:
    """Create minimal custom stereo pair directory."""
    H, W = 40, 56
    left_dir = base / "left"
    right_dir = base / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_pairs):
        seed = 200 + i * 10
        left = _make_tiny_image(H, W, seed)
        right = _make_tiny_image(H, W, seed + 1)
        cv2.imwrite(str(left_dir / f"{i:03d}.png"),
                     cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(right_dir / f"{i:03d}.png"),
                     cv2.cvtColor(right, cv2.COLOR_RGB2BGR))


# ===================================================================== #
#  Main
# ===================================================================== #

if __name__ == "__main__":
    print(f"Generating fixtures in {FIXTURES_DIR}")
    create_kitti_fixtures(FIXTURES_DIR / "kitti2015_mini")
    create_eth3d_fixtures(FIXTURES_DIR / "eth3d_mini")
    create_custom_fixtures(FIXTURES_DIR / "custom_mini")
    print("Done.")
