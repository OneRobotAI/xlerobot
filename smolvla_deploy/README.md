# SmolVLA × XLeRobot 部署指南

> 基于 SmolVLA (450M, HuggingFace) + XLeRobot (双臂 SO-100) 实现机器人推理
> 资源友好：单卡 RTX 4090 可训练，推理仅需 ~6GB VRAM
> 支持 **16-DOF**（2轮差速版）和 **17-DOF**（3轮全向版）

---

## 📋 目录

- [1. 整体架构](#1-整体架构)
- [2. SmolVLA 能控制 XLeRobot 吗？](#2-smolvla-能控制-xlerobot-吗)
- [3. 环境安装](#3-环境安装)
- [4. 数据采集](#4-数据采集)
- [5. 模型训练](#5-模型训练)
- [6. 部署推理服务](#6-部署推理服务)
- [7. 运行客户端](#7-运行客户端)
- [8. 与 X-VLA 对比](#8-与-x-vla-对比)
- [文件索引](#文件索引)

---

## 1. 整体架构

```
┌──────────────────────────────────────────────┐
│ 本地 (你的电脑 + XLeRobot)                    │
│                                               │
│  smolvla_deploy/client.py                     │
│    ├── 获取观测 (关节角度 + 摄像头图像)         │
│    ├── POST /act → 发送到云端推理服务           │
│    └── 执行预测动作 (30Hz 控制循环)             │
│                                               │
│  支持的机器人:                                  │
│    XLerobot (3轮全向, 17-DOF)                  │
│    XLerobot2Wheels (2轮差速, 16-DOF)           │
└──────────────────┬───────────────────────────┘
                   │ HTTP POST /act
                   ▼
┌──────────────────────────────────────────────┐
│ 云端 GPU 服务器 (RTX 4090+)                   │
│                                               │
│  smolvla_deploy/server.py                     │
│    ├── SmolVLAPolicy (450M 参数)              │
│    ├── SmolVLM2 视觉语言编码器                 │
│    ├── Flow Matching 动作专家                  │
│    ├── 预处理: 图像缩放、状态归一化、指令分词    │
│    └── 后处理: 动作反归一化                     │
└──────────────────────────────────────────────┘
```

## 2. SmolVLA 能控制 XLeRobot 吗？

**能。** SmolVLA 原生支持最高 **32-DOF** 的动作/状态空间。

| XLeRobot 版本 | DOF | 关节分布 | SmolVLA 兼容 |
|--------------|-----|---------|-------------|
| 2轮差速版 | **16-DOF** | 12手臂 + 2头部 + 2底盘 | ✅ 自动 pad 到 32 |
| 3轮全向版 | **17-DOF** | 12手臂 + 2头部 + 3底盘 | ✅ 自动 pad 到 32 |

微调时：
- 训练：16/17-DOF 状态/动作自动 pad 到 32 维
- 推理：模型输出 32 维 → 自动裁剪回原始 DOF
- 无需手动配置，`lerobot-train` 自动处理

一个注意事项：预训练 `lerobot/smolvla_base` 是 **6-DOF 单臂模型**（SO-100），所以直接用不会输出正确的 16-DOF 双臂动作。**必须先微调**（在你的 XLeRobot 数据集上微调 20000 steps，约 4 小时）。

---

## 3. 环境安装

### 3.1 本地电脑（数据采集 + 控制机器人）

```bash
# 使用你的 XLeRobot 现有环境，安装 SmolVLA 依赖
pip install 'lerobot[smolvla]'

# 验证安装
python3 << 'EOF'
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
print(" SmolVLA import OK")
from lerobot.policies.factory import make_pre_post_processors
print(" Pre/post-processors OK")
EOF

# 设置 XLerobot 软链接（client.py 需要导入 XLerobot 类）
bash /home/zach/XLeRobot/shared/setup_local.sh
```

### 3.2 云端 GPU 服务器（训练 + 推理）

```bash
# 把 smolvla_deploy/ 传到云服务器
scp -P <端口号> -r /home/zach/XLeRobot/smolvla_deploy/ user@你的云服务器IP:~

# SSH 登录
ssh -p <端口号> user@你的云服务器IP
cd smolvla_deploy

# 创建环境
conda create -n smolvla python=3.12 -y
conda activate smolvla
pip install 'lerobot[smolvla]' fastapi uvicorn

# 验证 GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 4. 数据采集

复用 `shared/` 下的采集工具，采集的 LeRobot 格式数据集可直接用于 SmolVLA 训练。

```bash
cd /home/zach/XLeRobot/shared

# 校准机械臂（只需做一次）
python calibrate_xlerobot.py --port1 /dev/ttyACM0 --port2 /dev/ttyACM1

# 录制 "clean table" 演示数据
python collect_data.py \
  --num-episodes 50 \
  --task "clean table"
```

### 数据格式兼容性

| 维度 | XLeRobot 数据 | SmolVLA 处理 |
|------|-------------|-------------|
| 动作维度 | 16/17 | 自动 pad 到 32 维 |
| 状态维度 | 16/17 | 自动 pad 到 32 维 |
| 相机数 | 3路 (top/left_wrist/right_wrist) | 通过 rename_map 适配 |
| 图像尺寸 | 640×480 | 自动 resize + pad 到 512×512 |

相机键名映射（训练时自动处理）：

```
top           → camera1
left_wrist    → camera2
right_wrist   → camera3
```

---

## 5. 模型训练

SmolVLA 在 LeRobot 里只有一个训练模式。`freeze_vision_encoder` 和 `train_expert_only` 已经是模型 config 的默认值，无需手动指定。

```bash
cd smolvla_deploy

# 如果你的数据集不在默认缓存位置
export LEROBOT_CACHE=/data/datasets

# 开始训练（~9GB VRAM，单卡 4090 约 4 小时）
bash train.sh

# 指定数据集
bash train.sh --dataset your/xlerobot-clean-table
```

### 实际执行的命令

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=your/xlerobot-clean-table \
  --output_dir=./outputs/smolvla_xlerobot \
  --policy.device=cuda \
  --batch_size=64 \
  --steps=20000 \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.left_wrist": "observation.images.camera2", "observation.images.right_wrist": "observation.images.camera3"}'
```

### 训练完成后

```
outputs/smolvla_xlerobot/
├── checkpoints/
│   ├── last/              # 最后一步（可继续训练）
│   └── best/              # 验证集最佳（用于部署）
├── logs/                  # TensorBoard 日志
└── config.yaml            # 配置备份
```

---

## 6. 部署推理服务

### 6.1 启动服务

```bash
cd smolvla_deploy
conda activate smolvla

# 使用你微调好的模型（16/17-DOF）
python server.py \
  --model-path ./outputs/smolvla_xlerobot/checkpoints/best \
  --port 8000

# 或使用预训练 base 模型（仅 6-DOF，用于测试）
python server.py --model-path lerobot/smolvla_base --port 8000
```

### 6.2 启动输出

```
2025-06-13 [INFO] Loading SmolVLA model from: ./outputs/smolvla_xlerobot/checkpoints/best
2025-06-13 [INFO]  Model loaded on cuda (3.2s)
2025-06-13 [INFO]    State dim: 16, Action dim: 16
2025-06-13 [INFO]    Image keys: ['camera1', 'camera2', 'camera3']
2025-06-13 [INFO]    Chunk size: 50
2025-06-13 [INFO]  Pre/post-processors loaded
2025-06-13 [INFO]  Server ready (3.5s)
2025-06-13 [INFO] Starting SmolVLA server...
2025-06-13 [INFO]   Model: ./outputs/smolvla_xlerobot/checkpoints/best
2025-06-13 [INFO]   Listen: 0.0.0.0:8000
2025-06-13 [INFO]   API: POST http://0.0.0.0:8000/act
```

### 6.3 验证服务

```bash
# 健康检查
curl http://localhost:8000/health

# → {"status":"ok","model":".../best","device":"cuda",
#    "state_dim":16,"action_dim":16,"chunk_size":50,
#    "cameras":["camera1","camera2","camera3"]}

# 推理测试
python3 << 'EOF'
import requests, json, numpy as np
resp = requests.post('http://localhost:8000/act', json={
    'proprio': json.dumps(np.zeros(16).tolist()),
    'language_instruction': 'clean table',
    'image0': json.dumps(np.zeros((256,256,3), dtype=np.uint8).tolist()),
    'image1': json.dumps(np.zeros((256,256,3), dtype=np.uint8).tolist()),
    'image2': json.dumps(np.zeros((256,256,3), dtype=np.uint8).tolist()),
})
actions = np.array(resp.json()['action'])
print(f"Action chunk: {actions.shape}")
print(f"Inference: {resp.json()['inference_time_ms']}ms")
print(f"DOF: {actions.shape[1]}")
EOF
```

### 6.4 持久化运行

```bash
nohup python server.py --model-path ./outputs/smolvla_xlerobot/checkpoints/best --port 8000 > server.log 2>&1 &
tail -f server.log
```

### 6.5 SSH 隧道（云服务器没有公网端口时）

```bash
# 本地运行，将云服务器的 8000 端口转发到本地
ssh -N -f -L 8000:localhost:8000 user@你的云服务器IP -p <SSH端口>

# 然后 client.py 用 localhost 连接
python client.py --server-url http://localhost:8000 --task "clean table"
```

---

## 7. 运行客户端

使用 `smolvla_deploy/client.py`，专用客户端，默认任务 "clean table"。

### 7.1 启动客户端

```bash
cd /home/zach/XLeRobot/smolvla_deploy

# 基本用法（连接本地或隧道转发的服务）
python client.py \
  --server-url http://localhost:8000 \
  --task "clean table"

# 连接远程云服务器
python client.py \
  --server-url http://<你的云服务器外网IP>:8000 \
  --task "clean table"
```

### 7.2 启动后的输出

```
  XLeRobot 2-Wheel Diff Drive (16-DOF)
2025-06-13 [INFO] Connecting XLeRobot (port1=/dev/ttyACM0, port2=/dev/ttyACM1)...
2025-06-13 [INFO]  XLeRobot connected
2025-06-13 [INFO] Available cameras: ['cam_top', 'cam_left_wrist', 'cam_right_wrist']
2025-06-13 [INFO] Server reports action_dim=16

[SmolVLA] Starting control loop
  Server:  http://localhost:8000/act
  Task:    'clean table'
  Freq:    30.0Hz
  Smooth:  0.3
  Robot:   2wheels (16-DOF)
  Action:  16-DOF
  Press Ctrl+C to stop

2025-06-13 [INFO] Starting control loop (freq=30.0Hz)
2025-06-13 [INFO] Step 0: Requesting new action chunk...
[SmolVLA] Received chunk: 50 steps, shape (50, 16)
...
```

### 7.3 运行流程

客户端每帧执行以下循环（30Hz）：

```
① 获取观测
   robot.get_observation()
   → 16/17-DOF 观测 (12手臂 + 2头部 + 底盘)
   → 3路摄像头图像 (top, left_wrist, right_wrist)

② 请求云端推理
   POST /act → 发送观测数据
   ← 返回动作块 (50步, 16/17维/步)

③ 执行动作
   逐个执行动作块中的每一步
   同时保持 30Hz 控制频率
   当前块用尽 → 回到步骤① (后台预请求下一块)
```

### 7.4 客户端参数

```
--server-url URL      SmolVLA 推理服务地址 (默认: http://localhost:8000)
--task TEXT           语言指令 (默认: "clean table")
--control-freq FLOAT  控制频率 Hz (默认: 30)
--smooth-ratio FLOAT  动作平滑系数 (默认: 0.3)
--port1 PATH          左臂总线端口 (默认: /dev/ttyACM0)
--port2 PATH          右臂总线端口 (默认: /dev/ttyACM1)
```

---

## 8. 与 X-VLA 对比

### 适用场景

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| 叠衣服 | **X-VLA** | 有专用 `xvla-folding` checkpoint，100% 成功率 |
| 桌面清理 | **SmolVLA** | 社区数据集丰富，适合 clean table 类任务 |
| 边缘设备 (Jetson) | **SmolVLA** | 450M 更轻量，推理更快 |
| 低延迟需求 | **SmolVLA** | 原生异步推理 (RTC) |

### 关键差异

| 对比项 | SmolVLA (本目录) | X-VLA (xvla_deploy/) |
|-------|-----------------|---------------------|
| 参数量 | **450M** | 0.9B |
| 基座模型 | SmolVLM2-500M | Qwen2.5-VL 3B |
| 推理显存 | **~6GB** | ~9GB |
| DOF 上限 | **32** | 20 |
| 训练配置 | 无，config 已内置默认值 | 2种模式（轻量/全量） |
| 叠衣服专用 | ❌ 需微调 | ✅ `xvla-folding` |

### 同时部署两套服务

```bash
# 终端 1: X-VLA 服务 (端口 8000)
cd /home/zach/XLeRobot/xvla_deploy
python server.py --model-path lerobot/xvla-folding --port 8000

# 终端 2: SmolVLA 服务 (端口 8001)
cd /home/zach/XLeRobot/smolvla_deploy
python server.py --model-path ./outputs/smolvla_xlerobot/checkpoints/best --port 8001

# 客户端连接 X-VLA
python /home/zach/XLeRobot/xvla_deploy/client.py --server-url http://localhost:8000

# 或连接 SmolVLA
python /home/zach/XLeRobot/smolvla_deploy/client.py --server-url http://localhost:8001 --task "clean table"
```

---

## 文件索引

所有文件位于 `/home/zach/XLeRobot/smolvla_deploy/`：

| 文件 | 在哪运行 | 用途 |
|------|---------|------|
| `server.py` | 云端（有 GPU） | FastAPI 推理服务，加载 SmolVLAPolicy |
| `client.py` | 本地（连机械臂） | SmolVLA 专用客户端，默认 task "clean table" |
| `train.sh` | 云端（有 GPU） | SmolVLA 微调训练（~9GB VRAM） |
| `requirements.txt` | 两边 | Python 依赖列表 |
| `README.md` | — | 本教程 |

> **共享硬件工具**（与 X-VLA 共用，位于 `shared/`）：
> - `calibrate_xlerobot.py` — 机械臂校准（电机零点设定）
> - `collect_data.py` — 双主臂遥操作录制演示数据（LeRobot 格式）
> - `read_init_pose.py` — 读取当前关节位置用于客户端配置
