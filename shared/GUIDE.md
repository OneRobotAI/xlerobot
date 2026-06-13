# XLeRobot Operation Guide (Hardware & Common Workflow)

> VLA model agnostic — shared by X-VLA, SmolVLA, and other VLA solutions
> Compatible with XLeRobot (Dual SO-100 Arms) and LeRobot standard bimanual setup (bi_so_follower)

---

## 📋 Table of Contents

- [Chapter 1: Meet Your XLeRobot](#chapter-1-meet-your-xlerobot)
- [Chapter 2: Environment Setup](#chapter-2-environment-setup)
- [Chapter 3: Data Collection](#chapter-3-data-collection)
- [Chapter 4: Model Training & Inference](#chapter-4-model-training--inference)
- [Appendix A: Common Troubleshooting](#appendix-a-common-troubleshooting)
- [Appendix B: Reference](#appendix-b-reference)

---

## Chapter 1: Meet Your XLeRobot

### 1.1 Robot Components

XLeRobot is a low-cost **dual-arm mobile robot** platform consisting of:

| Component | Description |
|-----------|-------------|
| Arms | 2 × SO-100, each 6-DOF (including gripper) |
| Head | 2 servos (pan/tilt) |
| Base | 2-wheel differential drive (16-DOF) or 3-wheel omni (17-DOF) |
| Cameras | 3 USB cameras (top + left/right wrist) |
| Controller | Laptop (Ubuntu) |

### 1.2 VLA Model Guides

Hardware operations in this guide are VLA-agnostic. For model-specific training and inference:

| Model | Guide |
|-------|-------|
| **X-VLA** | `xvla_deploy/GUIDE.md` |
| **SmolVLA** | `smolvla_deploy/GUIDE.md` |
| **Other models** | Check future `xxx_deploy/GUIDE.md` |

### 1.3 USB Port Layout

```
Laptop USB Ports:
┌───────────────────────────────────────────────────────────┐
│  Left Leader Arm     Right Leader Arm                     │
│  /dev/ttyACM2        /dev/ttyACM3                        │
│  Left Follower Arm   Right Follower Arm                  │
│  /dev/ttyACM0        /dev/ttyACM1                        │
│                                                           │
│  Top Camera          Left Wrist Camera  Right Wrist Camera│
│  /dev/video0         /dev/video2        /dev/video4       │
└───────────────────────────────────────────────────────────┘
```

Actual port numbers may vary. Use `lerobot-find-port` to check.

---

## Chapter 2: Environment Setup

### 2.1 Hardware Requirements

#### Local Computer (Data Collection + Optional Training/Inference)

```
Requirements:
├── OS: Ubuntu 22.04 or newer
├── RAM: 16GB+
├── Storage: 100GB+ free
├── USB Ports: At least 5 free USB-A
│   ├── Left Leader Arm × 1
│   ├── Right Leader Arm × 1
│   ├── Left Follower Arm × 1 (XLeRobot body)
│   ├── Right Follower Arm × 1 (XLeRobot body)
│   └── Cameras × 3
├── GPU (optional, for local training/inference):
│   ├── NVIDIA GPU, 24GB+ VRAM (e.g. RTX 4090, training + inference)
│   └── NVIDIA GPU, 6GB+ VRAM (e.g. RTX 3060, inference only)
└── XLeRobot hardware + 2 SO-100 leader arms (for data collection)
```

If you don't have a local GPU, or VRAM is insufficient, use a cloud GPU server (see 2.3).

#### Cloud Server (Optional, when local GPU is insufficient)

```
Requirements:
├── GPU: NVIDIA GPU, 24GB+ VRAM (RTX 4090 minimum)
├── CUDA: 12.1+
├── OS: Ubuntu 22.04
├── Storage: 50GB+
├── Network: Public IP (client needs access)
└── Recommended: [Featurize](https://featurize.cn?s=a6c5c56819ad418ab5c2ca96b794e40f) / AutoDL / Baidu BQG / Alibaba Cloud
```

### 2.2 Local Environment Setup

#### Step 1: Check Python Version

LeRobot requires Python **>= 3.12**.

```bash
python3 --version
# Expected: Python 3.12.x or later
```

If the version is incorrect, create a new environment:

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
```

#### Step 2: Install LeRobot (Base)

Official installation docs: [https://huggingface.co/docs/lerobot/installation](https://huggingface.co/docs/lerobot/installation)

**Install from source (recommended for easy updates):**

LeRobot source is at `/home/zach/lerobot/`:

```bash
cd /home/zach/lerobot
pip install -e .
```

**Install via pip (stable release):**

```bash
pip install lerobot
```

> VLA model-specific extras (e.g. `lerobot[xvla]`, `lerobot[smolvla]`) are documented in each model's `GUIDE.md`.

#### Step 3: Verify Base Installation

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

#### Step 4: Install ffmpeg (Required for Video Decoding)

LeRobot uses [TorchCodec](https://github.com/meta-pytorch/torchcodec) for dataset video decoding, which requires ffmpeg.

```bash
conda install ffmpeg -c conda-forge
```

#### Step 5: Set Up XLerobot Symlink

`client.py` needs to import the XLerobot class to control XLeRobot hardware. The symlink lets Python find the robot modules.

```bash
bash /home/zach/XLeRobot/shared/setup_local.sh
```

**Expected output:**
```
✅ Symlink created: .../lerobot/robots/xlerobot → .../XLeRobot/software/src/robots/xlerobot
✅ XLerobot import successful
```

> This symlink is local only and won't be committed to Git. Anyone cloning this repo needs to run it once.

#### Step 6: Verify XLeRobot Connection (Optional)

```bash
# Connect all cables (data, power, servo)
ls -la /dev/ttyACM*

# Test with keyboard control example
cd /home/zach/XLeRobot/software/examples
python 2_dual_so100_keyboard_ee_control.py
# Press x to exit
```

### 2.3 Cloud Environment Setup (Optional)

If your local machine doesn't have a GPU (or insufficient VRAM), use a cloud GPU server. Skip this section if you have a local GPU.

The cloud server only needs PyTorch + LeRobot + VLA model dependencies — no XLerobot hardware classes.

```bash
# SSH to cloud server
# Featurize uses featurize@, AutoDL uses root@ — adjust as needed
ssh -p <port> featurize@<cloud-ip>

# Check GPU
nvidia-smi

# Create and activate environment
conda create -n lerobot python=3.12 -y
conda activate lerobot

# Install VLA model dependencies (see each model's GUIDE.md)
# e.g. pip install 'lerobot[xvla]'

# Upload deployment package
cd /home/zach/XLeRobot
scp -P <port> -r xxx_deploy/ featurize@<cloud-ip>:~/
# Optionally upload shared/ (hardware tools)
```

> The cloud server doesn't need `setup_local.sh` (it doesn't control robot hardware).

### 2.4 SSH Tunnel (When Cloud Server Has No Public Port)

If the inference service runs on the cloud server's `localhost:8000`, forward the port via SSH tunnel:

```bash
# Run locally (background)
ssh -N -f -L 8000:localhost:8000 featurize@<cloud-ip> -p <ssh-port>

# Verify
curl http://localhost:8000/health
```

Close the tunnel:
```bash
ps aux | grep "ssh -N -f -L 8000" | grep -v grep
kill <process-id>
```

If the cloud platform allows direct port exposure (AutoDL custom service, Alibaba Cloud security group), you can also access via public IP directly.

---

## Chapter 3: Data Collection

### 3.1 USB Ports

#### Finding Devices

```bash
# Disconnect all arms first, then run:
lerobot-find-port
# Output: []

# Connect only the left leader arm:
lerobot-find-port
# Output: ['/dev/ttyACM0']

# Connect one by one and record:
# Left Leader Arm:   /dev/ttyACM2
# Right Leader Arm:  /dev/ttyACM3
# Left Follower Arm: /dev/ttyACM0
# Right Follower Arm:/dev/ttyACM1
```

> Port numbers may change each time you plug/unplug. Re-confirm before each use.

#### Optional: Fix USB Ports with udev Rules

If you need fixed USB port assignments, use udev rules to bind by device serial number:

```bash
# Check device serial numbers
udevadm info --name=/dev/ttyACM0 | grep ID_SERIAL

# Create udev rules
sudo tee /etc/udev/rules.d/99-xlerobot.rules << 'EOF'
SUBSYSTEM=="tty", ATTRS{serial}=="<left_leader_serial>", SYMLINK+="xlerobot_left_leader"
SUBSYSTEM=="tty", ATTRS{serial}=="<right_leader_serial>", SYMLINK+="xlerobot_right_leader"
SUBSYSTEM=="tty", ATTRS{serial}=="<left_follower_serial>", SYMLINK+="xlerobot_left_follower"
SUBSYSTEM=="tty", ATTRS{serial}=="<right_follower_serial>", SYMLINK+="xlerobot_right_follower"
EOF
sudo udevadm control --reload-rules
```

### 3.2 Calibrating the Arms

Calibration ensures the leader and follower arms output the same motor position values at the same physical position.

#### Calibrating Leader Arms

```bash
lerobot-calibrate \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/ttyACM2 \
  --teleop.right_arm_config.port=/dev/ttyACM3 \
  --teleop.id=bimanual_leader
```

**Steps:**
1. Move left leader arm to zero position (horizontal forward) → press Enter
2. Move each joint of the left leader arm through its full range → press Enter
3. Move right leader arm to zero position → press Enter
4. Move each joint of the right leader arm through its full range → press Enter
5. Calibration data saved to `~/.cache/lerobot/calibration/bimanual_leader.yaml`

#### Calibrating Follower Arms

##### Mode A: Full XLerobot Calibration (Recommended)

Covers all motors (arms + head + base). Requires the symlink first (skip if already done):

```bash
bash /home/zach/XLeRobot/shared/setup_local.sh

python /home/zach/XLeRobot/shared/calibrate_xlerobot.py \
  --port1 /dev/ttyACM0 \
  --port2 /dev/ttyACM1
```

- `port1` (/dev/ttyACM0) → Left arm (ID 1-6) + Head (ID 7-8)
- `port2` (/dev/ttyACM1) → Right arm (ID 1-6) + Base (ID 9-10)

**Calibration process:**
1. Bus1 power off → manually move left arm + head to middle → press Enter
2. Move each left arm + head joint to its limit → press Enter to stop
3. Bus2 power off → manually move right arm to middle → press Enter
4. Move each right arm joint to its limit → press Enter to stop
5. Calibration data auto-saved

##### Mode B: LeRobot Bimanual Calibration (Arms Only)

Calibrates only the 12 arm motors, skipping head and base:

```bash
python /home/zach/XLeRobot/shared/calibrate_xlerobot.py \
  --mode lerobot \
  --port1 /dev/ttyACM0 \
  --port2 /dev/ttyACM1
```

#### Verifying Calibration

```bash
# Move leader arms and check if follower arms follow
python /home/zach/lerobot/src/lerobot/scripts/lerobot_teleoperate.py \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/ttyACM0 \
  --robot.right_arm_config.port=/dev/ttyACM1 \
  --robot.id=xlerobot_follower \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/ttyACM2 \
  --teleop.right_arm_config.port=/dev/ttyACM3 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

If the follower arms move smoothly with the leader arms, calibration is successful.

### 3.3 Recording Demonstration Data

#### Find Camera Indices

```bash
lerobot-find-cameras
```

#### Mode A: XLerobot Recording (record.py)

```bash
python /home/zach/XLeRobot/software/src/record.py \
  --robot.type=xlerobot_2wheels \
  --robot.port1=/dev/ttyACM0 \
  --robot.port2=/dev/ttyACM1 \
  --robot.cameras='{"top": {"type": "opencv", "index_or_path": "/dev/video0", "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG", "backend": "V4L2"}, "left_wrist": {"type": "opencv", "index_or_path": "/dev/video2", "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG", "backend": "V4L2"}, "right_wrist": {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG", "backend": "V4L2"}}' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/ttyACM2 \
  --teleop.right_arm_config.port=/dev/ttyACM3 \
  --teleop.id=bimanual_leader \
  --dataset.repo_id=your/your-dataset-name \
  --dataset.single_task="your task description" \
  --dataset.num_episodes=50 \
  --dataset.fps=30 \
  --dataset.episode_time_s=90 \
  --dataset.reset_time_s=30 \
  --dataset.video=true \
  --display_data=true \
  --dataset.push_to_hub=false
```

**Adjustable parameters:**
| Parameter | Description | Suggested Value |
|-----------|-------------|-----------------|
| `num_episodes` | Number of demonstrations | Start with 50 |
| `episode_time_s` | Max duration per episode (s) | 30-90 |
| `reset_time_s` | Reset time between episodes (s) | 30 |
| `single_task` | Task description | Must match inference |
| `repo_id` | Dataset ID | HF Hub username/dataset-name |

#### Mode B: LeRobot Recording (collect_data.py)

```bash
cd /home/zach/XLeRobot/shared
python collect_data.py \
  --num-episodes 50 \
  --task "your task description" \
  --repo-id your/your-dataset-name \
  --episode-time 30 \
  --fps 30
```

Configure ports and cameras in the script:
```bash
# /home/zach/XLeRobot/shared/collect_data.py lines 32-39
FOLLOWER_LEFT_PORT = "/dev/ttyACM0"    # Left follower arm
FOLLOWER_RIGHT_PORT = "/dev/ttyACM1"   # Right follower arm
LEADER_LEFT_PORT = "/dev/ttyACM2"      # Left leader arm
LEADER_RIGHT_PORT = "/dev/ttyACM3"     # Right leader arm
CAMERA_TOP = "/dev/video0"             # Top camera
CAMERA_LEFT_WRIST = "/dev/video2"      # Left wrist camera
CAMERA_RIGHT_WRIST = "/dev/video4"     # Right wrist camera
```

### 3.4 Recording Tips

**Beginner progression:**
```
Episodes 1-10:   Simple actions (lay flat → single fold)
Episodes 11-20:  Vary starting positions
Episodes 21-30:  Vary操作方法 (single fold → double fold)
Episodes 31-40:  Use different objects
Episodes 41-50:  Mixed scenarios (random positions + methods)
```

**Good vs. Bad demonstrations:**
```
✅ Good:
  - Smooth, continuous motion
  - Complete each step before moving to the next
  - Clear gripper open/close

❌ Bad:
  - Long pauses mid-demonstration
  - Continuing after dropping the object (should reset)
  - Gripper always closed
```

### 3.5 After Data Collection

Recorded data is saved to `~/.cache/huggingface/lerobot/`:

```
~/.cache/huggingface/lerobot/your/your-dataset-name/
├── meta/
│   ├── info.json          ← Feature definitions (action/state dims, cameras)
│   ├── stats.json         ← Normalization statistics (mean/std)
│   └── episodes.jsonl     ← Per-episode metadata
├── data/
│   └── chunk-000/
│       └── episode_*.parquet  ← Action and state time series
└── videos/
    └── chunk-000/
        ├── top/
        │   └── episode_*.mp4   ← Top camera
        ├── left_wrist/
        │   └── episode_*.mp4   ← Left wrist camera
        └── right_wrist/
            └── episode_*.mp4   ← Right wrist camera
```

#### Verify Dataset

Quick check after recording:

```bash
python3 << 'EOF'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

cache_root = os.path.expanduser("~/.cache/huggingface/lerobot")
ds = LeRobotDataset("your/your-dataset-name", root=cache_root)

print(f"Episodes: {ds.num_episodes}")
print(f"Total frames: {len(ds)}")
print(f"FPS: {ds.fps}")

frame = ds[0]
print(f"Action: {frame['action'].shape[-1]} dims")
print(f"State: {frame['observation.state'].shape[-1]} dims")

image_keys = [k for k in frame.keys() if "image" in k]
print(f"Cameras: {len(image_keys)} → {image_keys}")
EOF
```

**Expected output (XLeRobot recording, 2-wheel version):**
```
Episodes: 50
Total frames: 45000
FPS: 30
Action: 16 dims
State: 16 dims
Cameras: 3 → ['observation.images.top', 'observation.images.left_wrist', 'observation.images.right_wrist']
```

#### Upload to Cloud Server (Optional)

If you're using a cloud server for training, upload the dataset:

```bash
# Method A: rsync (recommended)
rsync -avz --progress \
  ~/.cache/huggingface/lerobot/your/your-dataset-name/ \
  root@<cloud-ip>:/data/datasets/your-dataset-name/

# Method B: HuggingFace Hub
huggingface-cli login
python -m lerobot.datasets.push_to_hub \
  --repo-id=your/your-dataset-name \
  --root=~/.cache/huggingface/lerobot/
```

---

## Chapter 4: Model Training & Inference

Training and inference steps depend on your chosen VLA model. See the corresponding `GUIDE.md`:

| Model | Guide |
|-------|-------|
| **X-VLA** | `xvla_deploy/GUIDE.md` |
| **SmolVLA** | `smolvla_deploy/GUIDE.md` |
| **Other** | Future `xxx_deploy/GUIDE.md` |

---

## Appendix A: Common Troubleshooting

### A.1 Dataset Issues

#### "Dataset not found"

```bash
# Set the correct cache path
export LEROBOT_CACHE=/data/datasets
# Or check ~/.cache/huggingface/lerobot/ for data
```

### A.2 Inference Issues

#### "Connection refused"

```bash
ps aux | grep server.py      # Check if server is running
netstat -tlnp | grep 8000    # Check if port is listening
# Verify security group / firewall settings
```

#### "Timeout"

```bash
nvidia-smi                   # Check if GPU is occupied
# Reduce denoising steps (if model supports it)
ping <cloud-ip>              # Check network latency
```

### A.3 Robot Issues

#### Server is up but robot doesn't move

```bash
# Check:
# 1. Client connected to the correct server URL
# 2. Server returns reasonable action values (not all zeros)
# 3. Robot send_action works correctly
python /home/zach/XLeRobot/software/examples/2_dual_so100_keyboard_ee_control.py
```

#### Robot motion is jerky

```bash
# Increase control frequency
python client.py --control-freq 50
# Add action smoothing in client.py before send_action:
#   current_action = current_action * 0.7 + new_action * 0.3
```

#### Slow data transmission

```bash
# Reduce camera resolution in client.py:
cameras={
    "cam_top": OpenCVCameraConfig(..., width=320, height=240),
}
```

---

## Appendix B: Reference

### B.1 Camera Configuration

Typical 3-camera setup:

```
                    ┌─────────────────┐
                    │  top            │ ← Overhead, captures the full table
                    │  (global view)  │
                    └─────────────────┘
                          │
                    ┌─────┴─────┐
                    │   Table    │
                    └───────────┘
               ┌────────┴────────┐
               │                 │
        ┌──────┴──────┐  ┌──────┴──────┐
        │ left_wrist  │  │ right_wrist │
        │ (left wrist)│  │(right wrist)│
        └─────────────┘  └─────────────┘
```

> When using `bi_so_follower`, observation keys get `left_`/`right_` prefixes automatically.
> When using the XLerobot class, keys are `top`, `left_wrist`, `right_wrist`.

### B.2 LeRobot Dataset Format

A single frame contains:

| Field | Shape | Description |
|-------|-------|-------------|
| `action` | (D,) | D-dimensional float array (joint target positions) |
| `observation.state` | (D,) | D-dimensional float array (current joint positions) |
| `observation.images.*` | (H,W,3) | uint8 RGB image |
| `task` | — | Task description string |
| `episode_index` | — | Episode number |
| `frame_index` | — | Frame number within episode |
| `timestamp` | — | Timestamp |

Action dimensions by robot version:

| Version | Action/State Dims | Distribution |
|---------|------------------|--------------|
| 2-wheel diff drive | **16-DOF** | 12 arms + 2 head + 2 base |
| 3-wheel omni | **17-DOF** | 12 arms + 2 head + 3 base |
| Bimanual standard | **12-DOF** | 6 left arm + 6 right arm |
