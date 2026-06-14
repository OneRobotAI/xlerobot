# X-VLA × XLeRobot Training & Inference Guide

> **Version**: 2.0 | Compatible with: XLeRobot (Dual SO-100 Arms) + X-VLA 0.9B
>
> For hardware setup, environment installation, and data collection, see `shared/GUIDE.md`.
> This document covers X-VLA-specific training and inference deployment only.

---

## 📋 Table of Contents

- [Chapter 1: X-VLA Overview](#chapter-1-x-vla-overview)
- [Chapter 2: Model Training](#chapter-2-model-training)
- [Chapter 3: Inference Deployment](#chapter-3-inference-deployment)
- [Appendix A: Common Errors](#appendix-a-common-errors)
- [Appendix B: Reference](#appendix-b-reference)
- [Appendix C: Command Reference](#appendix-c-command-reference)

---

## Chapter 1: X-VLA Overview

X-VLA (0.9B, ICLR 2026) is a **Soft-Prompted VLA** model based on the Qwen2.5-VL 3B vision-language encoder.

### Key Advantages

| Feature | Description |
|---------|-------------|
| Parameters | **0.9B** (VLM encoder + Action Decoder) |
| Soft Prompt | Only 9M trainable parameters (1%), fast adaptation |
| Cloth Folding | Dedicated checkpoint `lerobot/xvla-folding`, **100% success** |
| Server-Client | Native cloud inference support |
| Action Space | `action_mode=auto` auto-detects dimensions (5/6-DOF bimanual) |

### Comparison

| Metric | X-VLA 0.9B | SmolVLA 450M | ACT/DP |
|--------|-----------|-------------|--------|
| Parameters | **0.9B** | 450M | ~80M |
| Single 4090 Training | ✅ ~9GB | ✅ ~9GB | ✅ 4-6GB |
| Cloth Folding | **100%** (`xvla-folding`) | — | 40-62% |
| Server-Client | **✅ Native** | ❌ DIY | ❌ |
| LeRobot Integration | ✅ Native | ✅ Native | ✅ Native |

> For hardware setup (calibration, data collection), see `shared/GUIDE.md`.

---

## Chapter 2: Model Training

### 2.1 Training Concepts

#### X-VLA Two-Phase Training

```
Phase I:   Large-scale pre-training (done by Ant Group / 2toINF)
  ┌──────────────────────────────────────────┐
  │ Data: 290K episodes, 7 platforms, 5 arms │
  │ Model: Pure Transformer Encoder, 0.9B    │
  │ Output: Base model "lerobot/xvla-base"    │
  └──────────────────────────────────────────┘
                      ↓
Phase II: Domain adaptation (your part)
  ┌──────────────────────────────────────────┐
  │ Method: Soft Prompt (9M params) + option │
  │         to fine-tune policy transformer   │
  │ Data: Your 50-200 demonstrations         │
  │ Output: XLeRobot-specific policy         │
  └──────────────────────────────────────────┘
```

#### What is a Soft Prompt?

```
Traditional approach: Fine-tune all parameters (4B × 16bit ≈ 8GB)
                     Requires lots of data + GPU

X-VLA: Train only the Soft Prompt (9M parameters)
       Soft Prompt = a set of learnable embedding vectors
       They tell the model: "You are now XLeRobot, joints are defined like this..."
```

### 2.2 Running Training

#### Environment Setup

```bash
# Install X-VLA dependencies
cd /home/zach/lerobot
pip install -e ".[xvla]"

# Or via pip
pip install 'lerobot[xvla]'
```

#### Training Command Reference

Run training on a cloud server or local GPU. Uses `train.sh` with the following parameters:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--dataset` | Dataset repo ID | `zonglin11/xlerobot_fold_cloth` |
| `--model-path` | Base model | `lerobot/xvla-folding` or `lerobot/xvla-base` |
| `--steps` | Training steps | `15000` (light) or `30000` (full) |
| `--output-dir` | Output directory | `./outputs/xvla_xlerobot_fold` |
| `--repo-id` | HF model ID (push to Hub) | `zonglin11/xvla-xlerobot-fold` |
| `--rename-map` | Camera key mapping | See below |

**Camera Key Mapping:**

XLeRobot dataset camera keys differ from what X-VLA expects. Use `--rename-map`:

```bash
# XLerobot (2/3-wheel): top, left_wrist, right_wrist
--rename-map '{"observation.images.top": "observation.images.image",
               "observation.images.left_wrist": "observation.images.image2",
               "observation.images.right_wrist": "observation.images.image3"}'

# LeRobot bimanual: left_top, left_wrist, right_wrist
--rename-map '{"observation.images.left_top": "observation.images.image",
               "observation.images.left_wrist": "observation.images.image2",
               "observation.images.right_wrist": "observation.images.image3"}'
```

> `action_mode=auto` automatically detects action dimensions (12 or 16), no manual config needed.

> **⚠️ Important: Fix State Dimension**
> X-VLA base models (`xvla-folding`, `xvla-base`) have `state_dim=8`, but XLeRobot has **16-DOF** (2-wheel) or **17-DOF** (3-wheel). Without fixing this, the state gets truncated during training, causing poor inference.
>
> **Fix before training:**
>
> ```bash
> # 2-wheel diff drive (16-DOF) — outputs to patched_models/
> bash /home/zach/XLeRobot/shared/patch_state_dim.sh
>
> # 3-wheel omni (17-DOF)
> bash /home/zach/XLeRobot/shared/patch_state_dim.sh --dof 17
>
> # From xvla-base
> bash /home/zach/XLeRobot/shared/patch_state_dim.sh --model lerobot/xvla-base
> bash /home/zach/XLeRobot/shared/patch_state_dim.sh --model lerobot/xvla-base --dof 17
>
> # Custom output path
> bash /home/zach/XLeRobot/shared/patch_state_dim.sh --model-path /my/models/xvla-state16
> ```
>
> Then train with the patched model:
> ```bash
> bash train.sh --model-path patched_models/xvla-folding-state16 --dataset ...
> ```
> The script only changes one line in `config.json`: `state_dim: 8 → 16 (or 17)`. Patched models are saved in `shared/patched_models/` (persistent across reboots).

#### Training Examples

**Cloth folding (unfrozen vision encoder, no Hub upload):**

```bash
cd ~/xvla_deploy
conda activate lerobot
export LEROBOT_CACHE=/data/datasets

bash train.sh \
  --dataset zonglin11/xlerobot_fold_cloth \
  --model-path lerobot/xvla-folding \
  --steps 15000 \
  --output-dir ./outputs/xvla_xlerobot_fold_vision \
  --rename-map '{"observation.images.top": "observation.images.image",
                 "observation.images.left_wrist": "observation.images.image2",
                 "observation.images.right_wrist": "observation.images.image3"}'
```

**Cloth folding (upload to HF Hub):**

```bash
bash train.sh \
  --dataset zonglin11/xlerobot_fold_cloth \
  --model-path lerobot/xvla-folding \
  --steps 15000 \
  --output-dir ./outputs/xvla_xlerobot_fold_vision \
  --repo-id zonglin11/xvla-xlerobot-fold-vision \
  --rename-map '{"observation.images.top": "observation.images.image",
                 "observation.images.left_wrist": "observation.images.image2",
                 "observation.images.right_wrist": "observation.images.image3"}'
```

**Clean table (from xvla-base, no upload):**

```bash
bash train.sh \
  --dataset zonglin11/xlerobot_clean_table \
  --model-path lerobot/xvla-base \
  --steps 15000 \
  --output-dir ./outputs/xvla_xlerobot_vision_unfrozen \
  --rename-map '{"observation.images.top": "observation.images.image",
                 "observation.images.left_wrist": "observation.images.image2",
                 "observation.images.right_wrist": "observation.images.image3"}'
```

**Clean table (upload to HF Hub):**

```bash
bash train.sh \
  --dataset zonglin11/xlerobot_clean_table \
  --model-path lerobot/xvla-base \
  --steps 30000 \
  --output-dir ./outputs/xvla_xlerobot_v2 \
  --repo-id zonglin11/xvla-xlerobot-clean-table-v2 \
  --rename-map '{"observation.images.top": "observation.images.image",
                 "observation.images.left_wrist": "observation.images.image2",
                 "observation.images.right_wrist": "observation.images.image3"}'
```

#### Training Log Interpretation

```
Step 1/5000 | loss: 0.8421 | lr: 1e-3 | VRAM: 8.2GB | 3.2 it/s
Step 100/5000 | loss: 0.5213 | lr: 1e-3 | VRAM: 8.3GB | 3.1 it/s
Step 500/5000 | loss: 0.3125 | lr: 1e-3 | VRAM: 8.3GB | 3.1 it/s
Step 1000/5000 | loss: 0.2156 | lr: 1e-3 | VRAM: 8.4GB | 3.1 it/s
```

**Loss interpretation:**
- Initial ~0.8: Model just starting, predictions are inaccurate
- Drops to ~0.2: Model begins to understand the task
- Stabilizes at ~0.1: Training converges
- If loss doesn't decrease: Check dataset quality

#### Training Complete

```
✅ Training complete!
  Model saved to: ./outputs/xvla_xlerobot_fold
  Checkpoints:    ./outputs/xvla_xlerobot_fold/checkpoints

  best/  ← Use for inference deployment
  last/  ← Can continue training
```

### 2.3 Monitoring Training

```bash
# Start TensorBoard
conda activate lerobot
tensorboard --logdir=./outputs/xvla_xlerobot_fold/logs --port=6006

# Access via browser
# http://<cloud-ip>:6006
```

**Key metrics:**

| Metric | Normal Behavior | Meaning |
|--------|----------------|---------|
| `train/loss` | Decreasing | Model is learning |
| `train/grad_norm` | Between 0.1-10 | Gradients stable |
| `train/lr` | Following schedule | Learning rate OK |
| `val/loss` | Decreasing too | No overfitting |

**Early stopping conditions:**
- loss below 0.05 → Model has learned enough
- loss flat for 500 steps → Convergence reached
- loss starts rising → Overfitting, stop early

---

## Chapter 3: Inference Deployment

### 3.1 XLeRobot Local Inference

#### Terminal 1: Start Inference Server

```bash
cd /home/zach/XLeRobot/xvla_deploy
conda activate lerobot

# Use a local checkpoint
python xvla_deploy/server.py \
  --model-path ./outputs/xvla_xlerobot_clean_table/checkpoints/last/pretrained_model \
  --port 8000

# Or load directly from HuggingFace
python xvla_deploy/server.py \
  --model-path zonglin11/xvla_xlerobot_clean_table \
  --port 8000
```

**server.py parameters:**
```
--model-path PATH  Model path or HF ID (default: lerobot/xvla-folding)
--port INT         Server port (default: 8000)
--host TEXT        Bind address (default: 0.0.0.0)
--device TEXT      Inference device (default: auto)
```

#### Terminal 2: Run Client

```bash
cd /home/zach/XLeRobot
conda activate lerobot
export PYTHONPATH=/home/zach/XLeRobot/software/src:$PYTHONPATH

python xvla_deploy/client.py \
  --server-url http://localhost:8000 \
  --task "clean table" \
  --smooth-ratio 0.5
```

**Camera configuration** in `xvla_deploy/client.py` `main()`:

```python
cameras={
    "cam_top": OpenCVCameraConfig(
        index_or_path="/dev/video0", fps=30, width=640, height=480,
        fourcc="MJPG", backend=Cv2Backends.V4L2,
    ),
    "cam_left_wrist": OpenCVCameraConfig(
        index_or_path="/dev/video2", fps=30, width=640, height=480,
        fourcc="MJPG", backend=Cv2Backends.V4L2,
    ),
    "cam_right_wrist": OpenCVCameraConfig(
        index_or_path="/dev/video4", fps=30, width=640, height=480,
        fourcc="MJPG", backend=Cv2Backends.V4L2,
    ),
}
```

**client.py parameters:**
```
--server-url URL      Inference server URL (default: http://localhost:8000)
--task TEXT           Language instruction
--domain-id INT       Domain ID (default: 0)
--denoise-steps INT   Denoising steps (default: 10, range 5-50)
--control-freq FLOAT  Control frequency Hz (default: 30)
--port1 PATH          Left arm bus port (default: /dev/ttyACM0)
--port2 PATH          Right arm bus port (default: /dev/ttyACM1)
```

### 3.2 XLeRobot Remote Inference

#### Cloud Server: Start Inference Service

```bash
conda activate lerobot
cd ~/xvla_deploy

# Upload local model to cloud server (if needed)
scp -P <port> -r /home/zach/XLeRobot/xvla_deploy/outputs/xxx/checkpoints/last/pretrained_model \
  featurize@<ip>:~/xvla_deploy/my_model/

# Start service
python server.py \
  --model-path ~/xvla_deploy/my_model \
  --port 8000
```

#### Local Machine: SSH Tunnel

```bash
ssh -N -f -L 8000:localhost:8000 featurize@<ip> -p <port>
```

#### Run Client

```bash
cd /home/zach/XLeRobot
conda activate lerobot
export PYTHONPATH=/home/zach/XLeRobot/software/src:$PYTHONPATH

python xvla_deploy/client.py \
  --server-url http://localhost:8000 \
  --task "fold the cloth on the table"
```

### 3.3 Bimanual Inference (bi_so_follower)

#### Local

```bash
# Terminal 1: Start server
cd /home/zach/XLeRobot/xvla_deploy
conda activate lerobot
python server.py \
  --model-path ./outputs/bilerobot_fold/checkpoints/last/pretrained_model \
  --port 8000

# Terminal 2: Run bimanual client
cd /home/zach/XLeRobot
conda activate lerobot
export PYTHONPATH=/home/zach/XLeRobot/software/src:$PYTHONPATH
python xvla_deploy/client_bimanual.py \
  --server-url http://localhost:8000 \
  --task "fold the towel on the table"
```

**Bimanual camera configuration** in `client_bimanual.py` `connect_robot()`:

```python
left_arm_config=SOFollowerConfig(
    port=left_port,
    cameras={
        "top": OpenCVCameraConfig(index_or_path="/dev/video0", ...),
        "wrist": OpenCVCameraConfig(index_or_path="/dev/video2", ...),
    },
),
right_arm_config=SOFollowerConfig(
    port=right_port,
    cameras={
        "wrist": OpenCVCameraConfig(index_or_path="/dev/video4", ...),
    },
),
```

#### Remote

```bash
# Cloud: upload model and start
scp -P <port> -r /home/zach/XLeRobot/xvla_deploy/outputs/bilerobot_fold/checkpoints/last/pretrained_model \
  featurize@<ip>:~/xvla_deploy/bimanual_model/
ssh -p <port> featurize@<ip>
conda activate lerobot
cd ~/xvla_deploy
python server.py --model-path ~/xvla_deploy/bimanual_model --port 8000

# Local SSH tunnel
ssh -N -f -L 8000:localhost:8000 featurize@<ip> -p <port>

# Run client
cd /home/zach/XLeRobot
conda activate lerobot
export PYTHONPATH=/home/zach/XLeRobot/software/src:$PYTHONPATH
python xvla_deploy/client_bimanual.py \
  --server-url http://localhost:8000 \
  --task "fold the towel on the table"
```

### 3.4 Verify Service

```bash
# Health check
curl http://localhost:8000/health
# → {"status":"ok","model":"lerobot/xvla-folding","device":"cuda"}
```

### 3.5 Running in Background

```bash
# Keep server running with nohup (survives SSH disconnect)
nohup python server.py \
  --model-path ./outputs/xvla_xlerobot_fold/checkpoints/best \
  --port 8000 > server.log 2>&1 &

tail -f server.log   # View logs
kill %1              # Stop server
```

> Cloud platform port exposure varies. Refer to your platform's documentation (AutoDL custom service, Featurize port mapping, etc.).

---

## Appendix A: Common Errors

### A.1 Training Errors

#### "CUDA out of memory"

```bash
# Reduce batch size
bash train.sh --light --batch-size 2

# Gradient accumulation
# Add to train.sh ARGS:
--optimizer.gradient_accumulation_steps=4

# Gradient checkpointing
--policy.gradient_checkpointing=true
```

#### "Action dimension mismatch"

```bash
# Ensure action_mode=auto is set (default in train.sh)
--policy.action_mode=auto
```

#### "Dataset not found"

```bash
export LEROBOT_CACHE=/data/datasets
# Or check ~/.cache/huggingface/lerobot/
```

### A.2 Inference Errors

#### "Connection refused"

```bash
ps aux | grep server.py      # Check if server is running
netstat -tlnp | grep 8000    # Check if port is listening
# Verify security group settings
```

#### "Model not loaded"

```bash
ls -la ./outputs/xxx/checkpoints/best    # Check model files exist
huggingface-cli whoami                   # Check HF login status
```

#### "Timeout"

```bash
nvidia-smi                    # Check if GPU is busy
# Reduce denoising steps: --denoise-steps 5
ping <cloud-ip>               # Check network latency
```

---

## Appendix B: Reference

### B.1 X-VLA Model Architecture

```
Input                        Output
┌──────┐                    ┌──────┐
│ Image │───┐              ┌→│Action1│
│ 3 cams│   │  ┌──────────┐│ └──────┘
└──────┘   ├─→│ Qwen2.5  ││ ┌──────┐
┌──────┐   │  │ -VL      │├→│Action2│
│ Lang  │───┘  │ 3B       ││ └──────┘
│ Instr │      │ VLM      ││ ┌──────┐
└──────┘      │ Encoder  │├→│ ...   │
              └──────────┘│ └──────┘
┌──────┐      ┌──────────┐│ ┌──────┐
│ Joint │─────→│ Soft     │└→│Action │
│ State │      │ Prompt   │  │(32)  │
└──────┘      │ (9M)     │  └──────┘
              │ Trainable│
              └──────────┘
```

- **Qwen2.5-VL 3B**: Alibaba's multimodal LLM (vision + language)
- **Action Decoder**: Transformer decoder generating action sequences
- **Soft Prompt**: Learnable embedding vectors (9M parameters)

### B.2 Action Space

```
XLeRobot Action Space:

Left Arm (6 dims):
  [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

Right Arm (6 dims):
  [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

Total: 12 dims (arms), plus head and base = 16/17 dims
```

X-VLA internally uses a 20-dim action space (auto-adapted via `action_mode=auto`):
- Training: original dims → pad → 20 dims
- Inference: 20 dims → trim → original dims

---

## Appendix C: Command Reference

### Training

```bash
# Light training (default)
bash train.sh --dataset your/dataset --model-path lerobot/xvla-folding

# Custom steps and output dir
bash train.sh --dataset your/dataset --steps 15000 --output-dir ./outputs/xxx

# Train and push to Hub
bash train.sh --dataset your/dataset --repo-id your/xvla-model
```

### Inference

```bash
# Start server
python server.py --model-path ./outputs/xxx/checkpoints/best --port 8000

# Run in background
nohup python server.py --model-path ./outputs/.../best --port 8000 > server.log 2>&1 &

# Test service
curl http://localhost:8000/health
```

### Parameter Reference

**train.sh:**
```
--dataset ID        Dataset repo ID (default: your/xlerobot-cloth-fold)
--output-dir DIR    Output directory (default: ./outputs/xvla_xlerobot_fold)
--model-path PATH   Base model path (default: lerobot/xvla-folding)
--repo-id ID        HF model ID (push to Hub)
--light             Light mode (default)
--resume PATH       Resume from checkpoint
--rename-map JSON   Camera key mapping
```

**server.py:**
```
--model-path PATH  Model path or HF ID (default: lerobot/xvla-folding)
--port INT         Server port (default: 8000)
--host TEXT        Bind address (default: 0.0.0.0)
--device TEXT      Inference device (default: auto)
```

**client.py:**
```
--server-url URL      Inference server URL (default: http://localhost:8000)
--task TEXT           Language instruction
--domain-id INT       Domain ID (default: 0)
--denoise-steps INT   Denoising steps (default: 10)
--control-freq FLOAT  Control frequency Hz (default: 30)
--port1 PATH          Left arm bus port (default: /dev/ttyACM0)
--port2 PATH          Right arm bus port (default: /dev/ttyACM1)
```
