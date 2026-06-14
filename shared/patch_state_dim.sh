#!/usr/bin/env bash
# ============================================================
# 修复 X-VLA 基座模型的状态维度（state_dim）以匹配 XLeRobot
# ============================================================
# X-VLA 基座模型（xvla-folding / xvla-base）的 state_dim=8，
# 但 XLeRobot 2轮版是 16-DOF、3轮版是 17-DOF。
# 不修复的话训练时状态会被截断，推理效果很差。
#
# 用法:
#   bash patch_state_dim.sh                          # 默认修复 xvla-folding, 16-DOF
#   bash patch_state_dim.sh --dof 17                 # 3轮全向版 (17-DOF)
#   bash patch_state_dim.sh --model lerobot/xvla-base
#   bash patch_state_dim.sh --model lerobot/xvla-base --dof 17
#
# 输出（默认放在当前目录下的 patched_models/，避免 /tmp/ 重启丢失）
#   patched_models/xvla-{folding|base}-state{16|17}/   ← 修补后的模型
#   可用 --model-path 指定其他位置
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_SOURCE="lerobot/xvla-folding"
STATE_DIM=16
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
    --model) MODEL_SOURCE="$2"; shift 2 ;;
    --dof) STATE_DIM="$2"; shift 2 ;;
    --model-path) OUTPUT_DIR="$2"; shift 2 ;;
    --help)
        echo "用法: bash patch_state_dim.sh [选项]"
        echo "  --model MODEL      基座模型 (默认: lerobot/xvla-folding)"
        echo "                      可选: lerobot/xvla-base"
        echo "  --dof DOF          状态维度 (默认: 16, 3轮版用 17)"
        echo "  --model-path DIR   输出目录（自动生成）"
        exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

MODEL_NAME=$(basename "$MODEL_SOURCE")
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$SCRIPT_DIR/patched_models/${MODEL_NAME}-state${STATE_DIM}"
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"

# 找本地缓存
CACHE_DIR="$HOME/.cache/huggingface/hub/models--${MODEL_SOURCE/\//--}"
SNAPSHOTS=$(ls -d "$CACHE_DIR/snapshots/"* 2>/dev/null || true)

if [ -z "$SNAPSHOTS" ]; then
    echo "⚠️  本地没有缓存 $MODEL_SOURCE，尝试下载..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$MODEL_SOURCE', local_dir='$OUTPUT_DIR')
" 2>/dev/null && echo "✅ 下载完成" || echo "❌ 下载失败，请检查网络"
    exit 1
fi

SNAPSHOT=$(echo "$SNAPSHOTS" | head -1)
echo "来源: $MODEL_SOURCE ($SNAPSHOT)"
echo "输出: $OUTPUT_DIR"

rm -rf "$OUTPUT_DIR"
cp -rL "$SNAPSHOT" "$OUTPUT_DIR"

python3 -c "
import json
cfg_path = '$OUTPUT_DIR/config.json'
with open(cfg_path) as f:
    cfg = json.load(f)
old = cfg['input_features']['observation.state']['shape'][0]
cfg['input_features']['observation.state']['shape'] = [$STATE_DIM]
with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=2)
print(f'✅ state_dim: {old} → $STATE_DIM')
"

echo ""
echo "训练时使用:"
echo "  bash train.sh --model-path $OUTPUT_DIR ..."
