# SmolVLA × XLeRobot Training & Inference Guide

> **Version**: 1.0 | Compatible with: XLeRobot (Dual SO-100 Arms) + SmolVLA 450M
>
> For hardware setup, environment installation, and data collection, see `shared/GUIDE.md`.
> This document covers SmolVLA-specific training and inference deployment only.

---

## 📋 Table of Contents

- [Chapter 1: SmolVLA Overview](#chapter-1-smolvla-overview)
- [Chapter 2: Model Training](#chapter-2-model-training)
- [Chapter 3: Inference Deployment](#chapter-3-inference-deployment)
- [Appendix A: Common Errors](#appendix-a-common-errors)
- [Appendix B: Reference](#appendix-b-reference)
- [Appendix C: Command Reference](#appendix-c-command-reference)

---

## Chapter 1: SmolVLA Overview

SmolVLA (450M) is HuggingFace's lightweight Vision-Language-Action model, based on the SmolVLM2-500M vision-language encoder with a Flow Matching action expert.

### Key Advantages

| Feature | Description |
|---------|-------------|
| Parameters | **450M** (VLM ~350M + Action Expert ~100M) |
| Training Objective | Flow Matching (continuous, non-autoregressive) |
| Inference VRAM | **~6GB**, 20-30Hz on RTX 3090/4090 |
| Async Inference | Native support (RTC), ~30% faster task completion |
| LoRA Support | ~3GB VRAM for fine-tuning |
| Community Datasets | 487 public datasets available for reference |
| DOF Limit | **32-DOF** (XLeRobot 16/17-DOF auto-adapts) |

### Comparison

| Metric | SmolVLA 450M | X-VLA 0.9B | ACT/DP |
|--------|-------------|-----------|--------|
| Parameters | **450M** | 0.9B | ~80M |
| Single 4090 Training | ✅ ~9GB | ✅ ~9GB | ✅ 4-6GB |
| Async Inference | ✅ Native | ❌ DIY | ❌ |
| Inference VRAM | **~6GB** | ~9GB | ~2GB |
| LeRobot Integration | ✅ Native | ✅ Native | ✅ Native |

> For hardware setup (calibration, data collection), see `shared/GUIDE.md`.

---

## Chapter 2: Model Training

### 2.1 Training Overview

SmolVLA has a single training mode in LeRobot — no "light/full" distinction. `freeze_vision_encoder` and `train_expert_only` are already the model config defaults and don't need to be specified manually.

Training freezes the vision encoder and only trains the Action Expert. ~9GB VRAM, ~4 hours on a single RTX 4090 (20000 steps).

#### Training Architecture

```
Input                            Output
┌──────┐                       ┌──────────┐
│ Image │───┐                 ┌→│ Actions  │
│ 3 cams│   │  ┌───────────┐ │ │ (50-step)│
└──────┘   ├─→│ SmolVLM2   │ │ └──────────┘
┌──────┐   │  │ Vision-Lang│ │
│ Lang  │───┘  │ Encoder   │ │ ┌──────────┐
│ Instr │      │ (Frozen)  │ │→│ Flow      │
└──────┘      └───────────┘ │ │ Matching  │
┌──────┐                    │ │ Action    │
│ Joint │────────────────────┘ │ Expert    │
│ State │                      │(Trainable)│
└──────┘                       └──────────┘
```

- **SmolVLM2-500M**: SigLIP vision encoder + SmolLM2 language decoder
- **Flow Matching**: Continuous diffusion process, denoises from noise to actions
- **Action Expert**: Transformer with interleaved cross/self-attention

### 2.2 Running Training

#### Environment Setup

```bash
# Install SmolVLA dependencies
cd /home/zach/lerobot
pip install -e ".[smolvla]"

# Or via pip
pip install 'lerobot[smolvla]'
```

#### Training Command Reference

Run training on a local GPU or cloud server. Uses `train.sh` with the following parameters:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--dataset` | Dataset repo ID | `your/xlerobot-clean-table` |
| `--model-path` | Base model | `lerobot/smolvla_base` |
| `--output-dir` | Output directory | `./outputs/smolvla_xlerobot` |
| `--rename-map` | Camera key mapping | See below |

**Camera Key Mapping:**

SmolVLA expects camera keys `camera1`, `camera2`, `camera3`. Use `--rename-map`:

```bash
# XLerobot (2/3-wheel): top, left_wrist, right_wrist
--rename-map '{"observation.images.top": "observation.images.camera1",
               "observation.images.left_wrist": "observation.images.camera2",
               "observation.images.right_wrist": "observation.images.camera3"}'

# LeRobot bimanual: left_top, left_wrist, right_wrist
--rename-map '{"observation.images.left_top": "observation.images.camera1",
               "observation.images.left_wrist": "observation.images.camera2",
               "observation.images.right_wrist": "observation.images.camera3"}'
```

#### Training Example

```bash
cd ~/smolvla_deploy
conda activate lerobot
export LEROBOT_CACHE=/data/datasets

bash train.sh \
  --dataset your/xlerobot-clean-table \
  --rename-map '{"observation.images.top": "observation.images.camera1",
                 "observation.images.left_wrist": "observation.images.camera2",
                 "observation.images.right_wrist": "observation.images.camera3"}'
```

This executes:

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=your/xlerobot-clean-table \
  --output_dir=./outputs/smolvla_xlerobot \
  --policy.device=cuda \
  --batch_size=64 \
  --steps=20000 \
  --rename_map='{"observation.images.top": "observation.images.camera1",
                 "observation.images.left_wrist": "observation.images.camera2",
                 "observation.images.right_wrist": "observation.images.camera3"}'
```

#### Training Log Interpretation

```
Step 1/20000 | loss: 0.95 | lr: 1e-4 | VRAM: 8.5GB | 3.5 it/s
Step 1000/20000 | loss: 0.42 | lr: 1e-4 | VRAM: 8.5GB | 3.5 it/s
Step 5000/20000 | loss: 0.21 | lr: 1e-4 | VRAM: 8.6GB | 3.5 it/s
Step 10000/20000 | loss: 0.12 | lr: 1e-4 | VRAM: 8.6GB | 3.4 it/s
```

**Loss interpretation:**
- Initial ~0.95: Model just starting
- Drops to ~0.2: Model begins to understand the task
- Stabilizes at ~0.1: Training converges

#### Training Complete

```
✅ Training complete!
  Model saved to: ./outputs/smolvla_xlerobot
  Checkpoints:    ./outputs/smolvla_xlerobot/checkpoints

  best/  ← Use for inference deployment
  last/  ← Can continue training
```

### 2.3 Monitoring Training

```bash
conda activate lerobot
tensorboard --logdir=./outputs/smolvla_xlerobot/logs --port=6006
```

---

## Chapter 3: Inference Deployment

### 3.1 Local Inference

#### Terminal 1: Start Inference Server

```bash
cd /home/zach/XLeRobot/smolvla_deploy
conda activate lerobot

# Use a fine-tuned model
python server.py \
  --model-path ./outputs/smolvla_xlerobot/checkpoints/best \
  --port 8000

# Or use the pretrained base model (6-DOF single arm, for testing)
python server.py --model-path lerobot/smolvla_base --port 8000
```

**server.py parameters:**
```
--model-path PATH  Model path or HF ID (default: lerobot/smolvla_base)
--port INT         Server port (default: 8000)
--host TEXT        Bind address (default: 0.0.0.0)
--device TEXT      Inference device (default: auto)
```

#### Terminal 2: Run Client

```bash
cd /home/zach/XLeRobot/smolvla_deploy
conda activate lerobot

python client.py \
  --server-url http://localhost:8000 \
  --task "clean table"
```

**client.py parameters:**
```
--server-url URL     SmolVLA server URL (default: http://localhost:8000)
--task TEXT          Language instruction (default: "clean table")
--control-freq FLOAT Control frequency Hz (default: 30)
--smooth-ratio FLOAT Action smoothing ratio (default: 0.3)
--port1 PATH         Left arm bus port (default: /dev/ttyACM0)
--port2 PATH         Right arm bus port (default: /dev/ttyACM1)
```

### 3.2 Remote Inference

#### Cloud Server: Start Inference Service

```bash
conda activate lerobot
cd ~/smolvla_deploy

# Upload local model to cloud server
scp -P <port> -r /home/zach/XLeRobot/smolvla_deploy/outputs/xxx/checkpoints/best \
  featurize@<ip>:~/smolvla_deploy/my_model/

# Start service
python server.py --model-path ~/smolvla_deploy/my_model --port 8000
```

#### Local: SSH Tunnel + Client

```bash
# SSH tunnel
ssh -N -f -L 8000:localhost:8000 featurize@<ip> -p <port>

# Run client
cd /home/zach/XLeRobot/smolvla_deploy
conda activate lerobot
python client.py --server-url http://localhost:8000 --task "clean table"
```

### 3.3 Verify Service

```bash
curl http://localhost:8000/health
# → {"status":"ok","model":"lerobot/smolvla_base","device":"cuda",
#    "state_dim":16,"action_dim":16,"chunk_size":50}
```

### 3.4 Running in Background

```bash
nohup python server.py \
  --model-path ./outputs/smolvla_xlerobot/checkpoints/best \
  --port 8000 > server.log 2>&1 &

tail -f server.log   # View logs
kill %1              # Stop server
```

> Cloud platform port exposure varies. Refer to your platform's documentation (AutoDL custom service, Featurize port mapping, etc.).

---

## Appendix A: Common Errors

### Training Errors

#### "CUDA out of memory"

```bash
# Reduce batch size
lerobot-train --batch_size=16 ...

# Enable gradient checkpointing
--policy.gradient_checkpointing=true
```

#### "Dataset not found"

```bash
export LEROBOT_CACHE=/data/datasets
```

### Inference Errors

#### "Connection refused"

```bash
ps aux | grep server.py    # Check if server is running
netstat -tlnp | grep 8000  # Check if port is listening
```

#### "Model not loaded"

```bash
ls -la ./outputs/xxx/checkpoints/best
huggingface-cli whoami
```

---

## Appendix B: Reference

### B.1 SmolVLA Model Architecture

```
Input                           Output
┌──────┐                    ┌──────────┐
│ Image │───┐              ┌→│ action 1 │
│ 3 cams│   │  ┌─────────┐ │ └──────────┘
└──────┘   ├─→│ SmolVLM2 │ │ ┌──────────┐
┌──────┐   │  │ 500M     │→│→│ action 2 │
│ Lang  │───┘  │ VLM     │ │ └──────────┘
│ Instr │      │ (Frozen) │ │ ┌──────────┐
└──────┘      └────┬────┘ │→│ ...      │
                   │      │ └──────────┘
┌──────┐      ┌────┴────┐ │ ┌──────────┐
│ Joint │─────→│ Action  │ └→│ action 50│
│ State │      │ Expert  │   └──────────┘
└──────┘      │(Train.) │
              └─────────┘
```

- **SmolVLM2-500M**: HuggingFace vision-language model (SigLIP visual encoder + SmolLM2 language decoder)
- **Action Expert**: Flow Matching Transformer with interleaved cross/self-attention layers
- **Output**: 50-step continuous action chunk (chunk_size=50)

### B.2 Action Space

SmolVLA supports up to **32-DOF** action/state spaces, auto-adapting to XLeRobot:

| Version | DOF | Distribution | SmolVLA Handling |
|---------|-----|-------------|-----------------|
| 2-wheel diff drive | **16-DOF** | 12 arms + 2 head + 2 base | Pad→32 for training, trim→16 for inference |
| 3-wheel omni | **17-DOF** | 12 arms + 2 head + 3 base | Pad→32 for training, trim→17 for inference |
| Bimanual standard | **12-DOF** | 6 left + 6 right arms | Pad→32 for training, trim→12 for inference |

### B.3 Camera Configuration

SmolVLA expects camera keys: `camera1`, `camera2`, `camera3`

| SmolVLA Key | XLeRobot Key | Position |
|-------------|--------------|----------|
| `observation.images.camera1` | `top` / `left_top` | Overhead global view |
| `observation.images.camera2` | `left_wrist` | Left wrist |
| `observation.images.camera3` | `right_wrist` | Right wrist |

---

## Appendix C: Command Reference

### Training

```bash
# Basic training
bash train.sh --dataset your/dataset

# Custom model and output dir
bash train.sh --dataset your/dataset --model-path lerobot/smolvla_base \
  --output-dir ./outputs/smolvla_xxx
```

### Inference

```bash
# Start server
python server.py --model-path ./outputs/smolvla_xlerobot/checkpoints/best --port 8000

# Run in background
nohup python server.py --model-path ... --port 8000 > server.log 2>&1 &

# Test service
curl http://localhost:8000/health
```

### Parameter Reference

**train.sh:**
```
--dataset ID        Dataset repo ID (default: your/xlerobot-clean-table)
--output-dir DIR    Output directory (default: ./outputs/smolvla_xlerobot)
--model-path PATH   Base model path (default: lerobot/smolvla_base)
--rename-map JSON   Camera key mapping
```

**server.py:**
```
--model-path PATH  Model path or HF ID (default: lerobot/smolvla_base)
--port INT         Server port (default: 8000)
--host TEXT        Bind address (default: 0.0.0.0)
--device TEXT      Inference device (default: auto)
```

**client.py:**
```
--server-url URL     SmolVLA server URL (default: http://localhost:8000)
--task TEXT          Language instruction (default: "clean table")
--control-freq FLOAT Control frequency Hz (default: 30)
--smooth-ratio FLOAT Action smoothing ratio (default: 0.3)
--port1 PATH         Left arm bus port (default: /dev/ttyACM0)
--port2 PATH         Right arm bus port (default: /dev/ttyACM1)
```
