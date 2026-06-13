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
#   - LeRobot dataset recorded via shared/collect_data.py
# ============================================================

set -euo pipefail

# ---- Config ----
DATASET_REPO_ID="your/xlerobot-cloth-fold"
OUTPUT_DIR="./outputs/xvla_xlerobot_fold"
MODEL_PATH="lerobot/xvla-folding"     # Start from cloth-folding checkpoint
HF_USER=""                             # Your HF username (optional)
MODEL_NAME=""                          # HF model repo name (default: xvla-xlerobot-fold)
PUSH_TO_HUB=false                       # Push model to HuggingFace Hub
LIGHT_MODE=false
RESUME_PATH=""
RENAME_MAP=""                          # Camera key mapping (optional, auto-detects XLerobot)
REPO_ID=""                            # HF repo ID for pushing model
STEPS=""                              # Custom training steps (overrides default)

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET_REPO_ID="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --repo-id) 
            REPO_ID="$2"
            PUSH_TO_HUB=true
            shift 2 ;;
        --hf-user) HF_USER="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --push-to-hub) PUSH_TO_HUB=true; shift ;;
        --light) LIGHT_MODE=true; shift ;;
        --resume) RESUME_PATH="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --rename-map) RENAME_MAP="$2"; shift 2 ;;
        --help)
            echo "Usage: bash train.sh [options]"
            echo "  --dataset ID       Dataset repo ID (default: your/xlerobot-cloth-fold)"
            echo "  --output-dir DIR   Output directory (default: ./outputs/xvla_xlerobot_fold)"
            echo "  --model-path PATH  Base model path (default: lerobot/xvla-folding)"
            echo "  --repo-id ID       HF model repo ID (e.g. username/model-name, implies --push-to-hub)"
            echo "  --hf-user USER     HF username (deprecated, use --repo-id)"
            echo "  --model-name NAME  HF model name (deprecated, use --repo-id)"
            echo "  --push-to-hub      Push model to HuggingFace Hub"
            echo "  --light            Lightweight mode (soft prompts only, ~1hr on 4090)"
            echo "  --resume PATH      Resume from checkpoint"
            echo "  --steps N          Training steps (default: light=5000, full=20000)"
            echo "  --rename-map JSON  Camera key mapping (default: XLerobot top/left_wrist/right_wrist)"
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

# 摄像头键名映射（自定义数据集可能需要）
# XLerobot 默认映射: observation.images.{top,left_wrist,right_wrist} → image,image2,image3

# Push to Hub (optional)
if [ -n "$REPO_ID" ]; then
    # --repo-id 方式
    ARGS+=" --policy.repo_id=${REPO_ID}"
    ARGS+=" --policy.push_to_hub=true"
elif [ -n "$HF_USER" ] && [ "$PUSH_TO_HUB" = true ]; then
    # --hf-user + --model-name + --push-to-hub 方式
    REPO_NAME="${MODEL_NAME:-xvla-xlerobot-fold}"
    ARGS+=" --policy.repo_id=${HF_USER}/${REPO_NAME}"
    ARGS+=" --policy.push_to_hub=true"
else
    ARGS+=" --policy.push_to_hub=false"
fi

# Resume from checkpoint
if [ -n "$RESUME_PATH" ]; then
    ARGS+=" --resume=true"
fi

# Push to Hub (optional)
if [ -n "$HF_USER" ] && [ "$PUSH_TO_HUB" = true ]; then
    ARGS+=" --policy.repo_id=${HF_USER}/xvla-xlerobot-fold"
    ARGS+=" --policy.push_to_hub=true"
fi

# ----- Train with frozen vision encoder (fast, ~13GB) -----
if [ "$LIGHT_MODE" = true ]; then
    echo "============================================"
    echo "  Frozen Vision Encoder Mode"
    echo "  ~13GB VRAM, good for quick iteration"
    echo "============================================"
    ARGS+=" --policy.freeze_vision_encoder=true"
    ARGS+=" --policy.freeze_language_encoder=true"
    ARGS+=" --policy.train_policy_transformer=true"
    ARGS+=" --policy.train_soft_prompts=true"
    ARGS+=" --steps=${STEPS:-15000}"
    ARGS+=" --batch_size=8"
    ARGS+=" --optimizer.lr=1e-3"
else
    # ----- Unfrozen vision encoder (best results, ~15-16GB) -----
    echo "============================================"
    echo "  Unfrozen Vision Encoder Mode"
    echo "  ~15-16GB VRAM, best task performance"
    echo "============================================"
    ARGS+=" --policy.freeze_vision_encoder=false"
    ARGS+=" --policy.freeze_language_encoder=true"
    ARGS+=" --policy.train_policy_transformer=true"
    ARGS+=" --policy.train_soft_prompts=true"
    ARGS+=" --steps=${STEPS:-30000}"
    ARGS+=" --batch_size=4"
    ARGS+=" --optimizer.type=xvla-adamw"
    ARGS+=" --optimizer.lr=1e-4"
    ARGS+=" --optimizer.weight_decay=0.0001"
    ARGS+=" --optimizer.betas=[0.9,0.95]"
    ARGS+=" --scheduler.type=cosine_decay_with_warmup"
    ARGS+=" --scheduler.num_warmup_steps=1000"
    ARGS+=" --scheduler.num_decay_steps=30000"
    ARGS+=" --scheduler.peak_lr=1e-4"
    ARGS+=" --scheduler.decay_lr=2.5e-06"
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
# 摄像头键名映射：用于将数据集的摄像头键名映射到模型期望的名称
# 格式: {"数据集键名1": "模型期望键名1", "数据集键名2": "模型期望键名2", ...}
# XLerobot 默认: top→image, left_wrist→image2, right_wrist→image3
# 模型始终期望: image, image2, image3（按顺序对应三路摄像头）
if [ -n "$RENAME_MAP" ]; then
    lerobot-train ${ARGS} --rename_map "${RENAME_MAP}"
else
    lerobot-train ${ARGS} --rename_map '{"observation.images.top": "observation.images.image", "observation.images.left_wrist": "observation.images.image2", "observation.images.right_wrist": "observation.images.image3"}'
fi

echo ""
echo "✅ Training complete!"
echo "  Model saved to: ${OUTPUT_DIR}"
echo "  Checkpoints:    ${OUTPUT_DIR}/checkpoints"
if [ -n "$HF_USER" ] && [ "$PUSH_TO_HUB" = true ]; then
    echo "  Pushed to Hub: ${HF_USER}/xvla-xlerobot-fold"
fi
