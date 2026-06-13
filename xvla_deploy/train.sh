#!/usr/bin/env bash
# ============================================================
# X-VLA Fine-Tuning Script for XLeRobot Cloth Folding
# ============================================================
# Usage:
#   # Lightweight (only train soft prompts, ~1 hour on 4090)
#   bash train.sh --light
#
#   # Full fine-tuning (~4 hours on 4090)
#   bash train.sh
#
#   # Resume from checkpoint
#   bash train.sh --resume ./outputs/xvla_xlerobot_fold/checkpoints/last
#
# Requirements:
#   - GPU with 24GB+ VRAM (RTX 4090, A100, etc.)
#   - LeRobot dataset recorded via collect_data.py
# ============================================================

set -euo pipefail

# ---- Config ----
DATASET_REPO_ID="your/xlerobot-cloth-fold"
OUTPUT_DIR="./outputs/xvla_xlerobot_fold"
MODEL_PATH="lerobot/xvla-folding"     # Start from cloth-folding checkpoint
HF_USER=""                             # Your HF username (optional)
LIGHT_MODE=false
RESUME_PATH=""

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET_REPO_ID="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --hf-user) HF_USER="$2"; shift 2 ;;
        --light) LIGHT_MODE=true; shift ;;
        --resume) RESUME_PATH="$2"; shift 2 ;;
        --help)
            echo "Usage: bash train.sh [options]"
            echo "  --dataset ID       Dataset repo ID (default: your/xlerobot-cloth-fold)"
            echo "  --output-dir DIR   Output directory (default: ./outputs/xvla_xlerobot_fold)"
            echo "  --model-path PATH  Base model path (default: lerobot/xvla-folding)"
            echo "  --hf-user USER     HF username for pushing model"
            echo "  --light            Lightweight mode (soft prompts only, ~1hr on 4090)"
            echo "  --resume PATH      Resume from checkpoint"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Build training command ----
JOB_NAME="xvla_xlerobot_cloth_fold"
ARGS=""

ARGS+=" --dataset.repo_id=${DATASET_REPO_ID}"
ARGS+=" --output_dir=${OUTPUT_DIR}"
ARGS+=" --job_name=${JOB_NAME}"
ARGS+=" --policy.path=${MODEL_PATH}"
ARGS+=" --policy.dtype=bfloat16"
ARGS+=" --policy.action_mode=auto"
ARGS+=" --policy.max_action_dim=20"
ARGS+=" --policy.max_state_dim=20"
ARGS+=" --policy.device=cuda"

# Resume from checkpoint
if [ -n "$RESUME_PATH" ]; then
    ARGS+=" --resume=true"
    ARGS+=" --resume_path=${RESUME_PATH}"
fi

# Push to Hub (optional)
if [ -n "$HF_USER" ]; then
    ARGS+=" --policy.repo_id=${HF_USER}/xvla-xlerobot-fold"
    ARGS+=" --policy.push_to_hub=true"
fi

# ----- Light mode: only train soft prompts -----
if [ "$LIGHT_MODE" = true ]; then
    echo "============================================"
    echo "  Lightweight Mode: Soft Prompts Only"
    echo "  ~1 hour on RTX 4090, ~9GB VRAM"
    echo "============================================"
    ARGS+=" --policy.freeze_vision_encoder=true"
    ARGS+=" --policy.freeze_language_encoder=true"
    ARGS+=" --policy.train_policy_transformer=false"
    ARGS+=" --policy.train_soft_prompts=true"
    ARGS+=" --steps=5000"
    ARGS+=" --batch_size=8"
    ARGS+=" --optimizer.lr=1e-3"  # Higher LR for soft prompts only
else
    # ----- Full fine-tuning (better results) -----
    echo "============================================"
    echo "  Full Fine-Tuning Mode"
    echo "  ~4 hours on RTX 4090, ~16GB VRAM"
    echo "============================================"
    ARGS+=" --policy.freeze_vision_encoder=false"
    ARGS+=" --policy.freeze_language_encoder=false"
    ARGS+=" --policy.train_policy_transformer=true"
    ARGS+=" --policy.train_soft_prompts=true"
    ARGS+=" --steps=20000"
    ARGS+=" --batch_size=4"
    ARGS+=" --optimizer.lr=1e-4"
fi

# ----- Optional: depth (if you have RealSense) -----
# ARGS+=" --policy.use_depth=true"

# ----- Print config -----
echo ""
echo "============================================"
echo "  Training Configuration"
echo "============================================"
echo "  Dataset:   ${DATASET_REPO_ID}"
echo "  Base model: ${MODEL_PATH}"
echo "  Output:    ${OUTPUT_DIR}"
echo "  Steps:     $(echo $ARGS | grep -oP 'steps=\K[0-9]+')"
echo "  Batch:     $(echo $ARGS | grep -oP 'batch_size=\K[0-9]+')"
echo "  Light:     ${LIGHT_MODE}"
echo "============================================"
echo ""
echo "Running: lerobot-train ${ARGS}"
echo ""

# ----- Run training -----
# Activate virtual environment if it exists
if [ -d "venv_xvla" ]; then
    source venv_xvla/bin/activate
fi

# Check if lerobot-train is available
if ! command -v lerobot-train &> /dev/null; then
    echo "lerobot-train not found. Install with: pip install lerobot[xvla]"
    exit 1
fi

# Execute
lerobot-train ${ARGS}

echo ""
echo "✅ Training complete!"
echo "  Model saved to: ${OUTPUT_DIR}"
echo "  Checkpoints:    ${OUTPUT_DIR}/checkpoints"
if [ -n "$HF_USER" ]; then
    echo "  Pushed to Hub: ${HF_USER}/xvla-xlerobot-fold"
fi
