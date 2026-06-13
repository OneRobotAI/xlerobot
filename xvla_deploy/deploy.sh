#!/usr/bin/env bash
# ============================================================
# X-VLA Cloud Server - One-Click Deployment Script
# ============================================================
# Usage:
#   # Default (uses lerobot/xvla-folding checkpoint, port 8000)
#   bash deploy.sh
#
#   # Custom model and port
#   bash deploy.sh --model-path your/xvla-xlerobot-fold --port 8080
#
#   # Only setup environment (don't start server)
#   bash deploy.sh --setup-only
#
# Tested on: Ubuntu 22.04+, CUDA 12.1+, Python 3.10
# Cloud: Baidu Bǎigě, Alibaba Cloud, AWS, any NVIDIA GPU instance
# ============================================================

set -euo pipefail

# ---- Config ----
MODEL_PATH="lerobot/xvla-folding"
PORT=8000
HOST="0.0.0.0"
SETUP_ONLY=false
VENV_DIR="venv_xvla"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --setup-only) SETUP_ONLY=true; shift ;;
        --venv) VENV_DIR="$2"; shift 2 ;;
        --help)
            echo "Usage: bash deploy.sh [options]"
            echo "  --model-path PATH   Model path or HF ID (default: lerobot/xvla-folding)"
            echo "  --port PORT         Server port (default: 8000)"
            echo "  --host HOST         Bind address (default: 0.0.0.0)"
            echo "  --setup-only        Only install dependencies, don't start server"
            echo "  --venv DIR          Virtual environment directory (default: venv_xvla)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  X-VLA Cloud Server Deployment"
echo "============================================"
echo "  Model:      $MODEL_PATH"
echo "  Port:       $PORT"
echo "  Host:       $HOST"
echo "  Setup only: $SETUP_ONLY"
echo "============================================"

# ============================================================
# Step 1: Check system requirements
# ============================================================
echo ""
echo "[1/5] Checking system requirements..."

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "  ✅ GPU detected: $GPU_NAME (Driver: $CUDA_VERSION)"
else
    echo "  ⚠️  No NVIDIA GPU detected. Will use CPU (slow)."
fi

# Check Python
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version 2>&1)
    echo "  ✅ $PY_VERSION"
else
    echo "  ❌ Python3 not found. Install Python 3.10+: sudo apt install python3 python3-venv"
    exit 1
fi

# Check disk space
AVAIL_GB=$(df -BG "$SCRIPT_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAIL_GB" -lt 20 ]; then
    echo "  ⚠️  Low disk space: ${AVAIL_GB}GB available (recommend 50GB+)"
else
    echo "  ✅ Disk space: ${AVAIL_GB}GB available"
fi

# ============================================================
# Step 2: Install system dependencies
# ============================================================
echo ""
echo "[2/5] Installing system dependencies..."

if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        build-essential \
        python3-dev \
        python3-pip \
        python3-venv \
        git \
        wget \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        2>/dev/null || true
    echo "  ✅ System dependencies installed"
else
    echo "  ⚠️  Not a Debian-based system. Install dependencies manually."
fi

# ============================================================
# Step 3: Create virtual environment
# ============================================================
echo ""
echo "[3/5] Creating Python virtual environment..."

if [ -d "$VENV_DIR" ]; then
    echo "  Virtual environment already exists at $VENV_DIR"
    echo "  Remove it to recreate: rm -rf $VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
    echo "  ✅ Created venv at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ============================================================
# Step 4: Install Python dependencies
# ============================================================
echo ""
echo "[4/5] Installing Python dependencies..."

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Check Python version (Python 3.13+ may have compatibility issues)
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
echo "  Python ${PY_MAJOR}.${PY_MINOR}"
if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 13 ]; then
    echo "  ⚠️  Python 3.13+ may cause issues with some ML libraries."
    echo "     If errors occur, use Python 3.10–3.12."
fi

# Upgrade pip
pip install --upgrade pip setuptools wheel -q

# Install PyTorch with domestic mirror fallback for China users
echo "  Installing PyTorch..."
if pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q 2>/dev/null; then
    :  # success
else
    echo "  ⚠️  PyTorch download failed, trying Aliyun mirror..."
    pip install torch torchvision --index-url https://mirrors.aliyun.com/pytorch/whl/cu121 -q
fi

# Install LeRobot with X-VLA support
echo "  Installing LeRobot with X-VLA..."
pip install lerobot[xvla] -q

# Install server dependencies
echo "  Installing server dependencies..."
pip install fastapi uvicorn requests -q

# Install training dependencies (for fine-tuning)
echo "  Installing training dependencies..."
pip install wandb tensorboard datasets -q

# Verify installation
echo ""
echo "  Verifying installation..."
python3 -c "
import torch
print(f'  ✅ PyTorch {torch.__version__} (CUDA available: {torch.cuda.is_available()})')
try:
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    print(f'  ✅ X-VLA import successful')
except Exception as e:
    print(f'  ⚠️  X-VLA import warning: {e}')
"

echo "  ✅ Dependencies installed"

# ============================================================
# Step 5: Start server (or show instructions)
# ============================================================
echo ""
echo "[5/5] Finalizing..."

if [ "$SETUP_ONLY" = true ]; then
    echo ""
    echo "============================================"
    echo "  ✅ Setup complete!"
    echo "============================================"
    echo ""
    echo "  To start the server manually:"
    echo "    source $VENV_DIR/bin/activate"
    echo "    python server.py --model-path $MODEL_PATH --port $PORT"
    echo ""
    echo "  To test the server health:"
    echo "    curl http://localhost:$PORT/health"
    echo ""
else
    echo "  Starting X-VLA inference server..."
    echo ""
    echo "============================================"
    echo "  🚀 Server starting..."
    echo "  Endpoint: http://$HOST:$PORT/act"
    echo "  Health:   http://$HOST:$PORT/health"
    echo "  Model:    $MODEL_PATH"
    echo "============================================"
    echo ""

    # Start server
    cd "$SCRIPT_DIR"
    exec python server.py --model-path "$MODEL_PATH" --port "$PORT" --host "$HOST"
fi
