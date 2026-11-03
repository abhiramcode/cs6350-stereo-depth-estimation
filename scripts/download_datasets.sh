#!/usr/bin/env bash
# ─── Dataset Download Helper ──────────────────────────────────────────
#
# Usage:
#   bash scripts/download_datasets.sh [kitti|eth3d|all]
#
# Prerequisites:
#   - Sufficient disk space (KITTI ~6 GB extracted, ETH3D ~500 MB)
#   - wget or curl installed
#   - For KITTI: You may need to accept terms at the KITTI website first
#
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"

# ═══════════════════════════════════════════════════════════════════════
#  ETH3D Low-Res Two-View
# ═══════════════════════════════════════════════════════════════════════
download_eth3d() {
    local ETH3D_DIR="${DATA_DIR}/eth3d"
    mkdir -p "$ETH3D_DIR"

    echo "═══ Downloading ETH3D Low-Res Two-View ═══"
    echo "  Target: ${ETH3D_DIR}"
    echo "  Size:   ~500 MB total"
    echo ""

    # Training images
    local TRAIN_URL="https://www.eth3d.net/data/two_view_training.7z"
    # Training ground truth
    local TRAIN_GT_URL="https://www.eth3d.net/data/two_view_training_gt.7z"
    # Test images (no GT)
    local TEST_URL="https://www.eth3d.net/data/two_view_test.7z"

    for url in "$TRAIN_URL" "$TRAIN_GT_URL" "$TEST_URL"; do
        local filename=$(basename "$url")
        if [ ! -f "${ETH3D_DIR}/${filename}" ]; then
            echo "  Downloading ${filename}..."
            wget -q --show-progress -P "$ETH3D_DIR" "$url"
        else
            echo "  Already exists: ${filename}"
        fi
    done

    # Extract
    echo "  Extracting archives (requires p7zip)..."
    cd "$ETH3D_DIR"
    for archive in *.7z; do
        if [ -f "$archive" ]; then
            7z x -y "$archive" > /dev/null
            echo "  Extracted: ${archive}"
        fi
    done

    echo "  ✓ ETH3D download complete"
    echo ""
}


# ═══════════════════════════════════════════════════════════════════════
#  KITTI 2015 Stereo
# ═══════════════════════════════════════════════════════════════════════
download_kitti() {
    local KITTI_DIR="${DATA_DIR}/kitti2015"
    mkdir -p "$KITTI_DIR"

    echo "═══ Downloading KITTI 2015 Stereo ═══"
    echo "  Target: ${KITTI_DIR}"
    echo "  Size:   ~2 GB download, ~6 GB extracted"
    echo ""
    echo "  NOTE: You must first register and accept terms at:"
    echo "    https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo"
    echo ""
    echo "  Download 'data_scene_flow.zip' manually and place it in:"
    echo "    ${KITTI_DIR}/data_scene_flow.zip"
    echo ""

    local ZIP="${KITTI_DIR}/data_scene_flow.zip"
    if [ -f "$ZIP" ]; then
        echo "  Found ${ZIP}, extracting..."
        cd "$KITTI_DIR"
        unzip -n "$ZIP"
        echo "  ✓ KITTI extraction complete"
    else
        echo "  ⚠  ${ZIP} not found."
        echo "  Please download manually from the KITTI website."
        echo ""
        echo "  After downloading, run:"
        echo "    cd ${KITTI_DIR} && unzip data_scene_flow.zip"
    fi
    echo ""
}


# ═══════════════════════════════════════════════════════════════════════
#  Custom eval directory
# ═══════════════════════════════════════════════════════════════════════
setup_custom() {
    local CUSTOM_DIR="${DATA_DIR}/custom_eval"
    mkdir -p "${CUSTOM_DIR}/left" "${CUSTOM_DIR}/right"

    echo "═══ Custom Evaluation Directory ═══"
    echo "  Created: ${CUSTOM_DIR}/left/"
    echo "  Created: ${CUSTOM_DIR}/right/"
    echo ""
    echo "  Drop left/right stereo image pairs into these folders."
    echo "  Optionally create ${CUSTOM_DIR}/calib.yaml with:"
    echo "    f_px: <focal_length_pixels>"
    echo "    baseline_m: <baseline_metres>"
    echo ""
}


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
main() {
    local target="${1:-all}"

    echo "────────────────────────────────────────────────"
    echo " CS6350 Stereo Depth — Dataset Download Helper"
    echo "────────────────────────────────────────────────"
    echo ""
    echo " Disk free: $(df -h "$DATA_DIR" 2>/dev/null | tail -1 | awk '{print $4}')"
    echo ""

    case "$target" in
        eth3d)  download_eth3d ;;
        kitti)  download_kitti ;;
        custom) setup_custom ;;
        all)
            download_eth3d
            download_kitti
            setup_custom
            ;;
        *)
            echo "Usage: $0 [kitti|eth3d|all]"
            exit 1
            ;;
    esac

    echo "Done."
}

main "$@"

# bash scripts/download_datasets.sh eth3d
# uv run python scripts/smoke_test_stage1.py --dataset eth3d --root data/eth3d

# bash scripts/download_datasets.sh kitti
# uv run python scripts/smoke_test_stage1.py --dataset kitti --root data/kitti2015

# bash scripts/download_datasets.sh custom
# uv run python scripts/smoke_test_stage1.py --dataset custom --root data/custom_eval

# bash scripts/download_datasets.sh all
# uv run python scripts/smoke_test_stage1.py --dataset all --root data
