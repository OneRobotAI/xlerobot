#!/usr/bin/env bash
# ============================================================
# SmolVLA Fine-Tuning Script for XLeRobot
# ============================================================
# Usage:
#   bash train.sh
#   bash train.sh --dataset your/xlerobot-clean-table
#
# SmolVLA uses a single training mode — no separate light/full/LoRA.
# Config defaults (freeze_vision_encoder=true, train_expert_only=true)
# are already baked into the model checkpoint.
#
# Requirements:
#   - GPU with 24GB+ VRAM (RTX 4090, A100, etc.)
#   - LeRobot dataset recorded via shared/collect_data.py
#   - pip install 'lerobot[smolvla]'
# ============================================================

set -euo pipefail

# ---- Config ----
DATASET_REPO_ID="your/xlerobot-clean-table"
OUTPUT_DIR="./outputs/smolvla_xlerobot"
MODEL_PATH="lerobot/smolvla_base"     # Start from pretrained base (450M)

# Camera key mapping: XLeRobot dataset keys -> SmolVLA model keys
# XLeRobot:  top, left_wrist, right_wrist
# SmolVLA:   camera1, camera2, camera3
RENAME_MAP='{"observation.images.top": "observation.images.camera1", "observation.images.left_wrist": "observation.images.camera2", "observation.images.right_wrist": "observation.images.camera3"}'

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET_REPO_ID="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --rename-map) RENAME_MAP="$2"; shift 2 ;;
        --help)
            echo "Usage: bash train.sh [options]"
            echo "  --dataset ID       Dataset repo ID (default: your/xlerobot-clean-table)"
            echo "  --output-dir DIR   Output directory (default: ./outputs/smolvla_xlerobot)"
            echo "  --model-path PATH  Base model path (default: lerobot/smolvla_base)"
            echo "  --rename-map JSON  Camera key mapping (default: XLerobot -> SmolVLA)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Build training command ----
# SmolVLA doesn't need freeze_vision_encoder/train_expert_only flags —
# those are the model's config defaults already.
ARGS=""
ARGS+=" --policy.path=${MODEL_PATH}"
ARGS+=" --dataset.repo_id=${DATASET_REPO_ID}"
ARGS+=" --output_dir=${OUTPUT_DIR}"
ARGS+=" --policy.device=cuda"
ARGS+=" --batch_size=64"
ARGS+=" --steps=20000"

# Camera rename map (maps XLeRobot camera keys to SmolVLA's camera1/camera2/camera3)
ARGS+=" --rename_map='${RENAME_MAP}'"

# ----- Print config -----
echo ""
echo "============================================"
echo "  SmolVLA Training"
echo "============================================"
echo "  Dataset:    ${DATASET_REPO_ID}"
echo "  Model:      ${MODEL_PATH}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Steps:      20000"
echo "  Batch:      64"
echo "============================================"
echo ""
echo "Running: lerobot-train ${ARGS}"
echo ""

# ----- Run training -----
if [ -d "venv_smolvla" ]; then
    source venv_smolvla/bin/activate
fi

if ! command -v lerobot-train &> /dev/null; then
    echo "lerobot-train not found. Install with: pip install 'lerobot[smolvla]'"
    exit 1
fi

lerobot-train ${ARGS}

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Model: ${OUTPUT_DIR}/checkpoints/best"
echo ""
echo "  Deploy:"
echo "  python server.py --model-path ${OUTPUT_DIR}/checkpoints/best --port 8000"
echo ""
echo "  Run client:"
echo "  python client.py --server-url http://localhost:8000 --task \"clean table\""
echo "============================================"
