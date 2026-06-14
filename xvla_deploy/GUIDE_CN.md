# X-VLA × XLeRobot 训练与推理指南

> **版本**: 2.0 | 适用: XLeRobot (双臂 SO-100) + X-VLA 0.9B
>
> 硬件准备、环境搭建、数据采集等通用步骤请见 `shared/GUIDE.md`。
> 本文档仅包含 X-VLA 特有的训练和推理部署部分。

---

## 📋 目录

- [第一章：X-VLA 简介](#第一章x-vla-简介)
- [第二章：模型训练](#第二章模型训练)
- [第三章：推理服务部署](#第三章推理服务部署)
- [附录A：常见错误](#附录a常见错误)
- [附录B：概念参考](#附录b概念参考)
- [附录C：命令速查表](#附录c命令速查表)

---

## 第一章：X-VLA 简介

X-VLA (0.9B, ICLR 2026) 是一个 **Soft-Prompted VLA** 模型，基于 Qwen2.5-VL 3B 视觉语言编码器。

### 核心优势

| 特性 | 说明 |
|------|------|
| 参数量 | **0.9B**（VLM 编码器 + Action Decoder） |
| Soft Prompt | 仅 9M 可训练参数（1%），快速适配新机器人 |
| 叠衣服 | 专用 checkpoint `lerobot/xvla-folding`，**100% 成功率** |
| Server-Client | 原生支持云端推理 |
| 动作空间 | `action_mode=auto` 自动检测维度（5/6-DOF 双臂通吃） |

### 与同类模型对比

| 对比项 | X-VLA 0.9B | SmolVLA 450M | ACT/DP |
|-------|-----------|-------------|--------|
| 参数量 | **0.9B** | 450M | ~80M |
| 单卡 4090 训练 | ✅ ~9GB | ✅ ~9GB | ✅ 4-6GB |
| 叠衣服 | **100%** (`xvla-folding`) | — | 40-62% |
| Server-Client | **✅ 原生** | ❌ 需自建 | ❌ |
| LeRobot 集成 | ✅ 原生 | ✅ 原生 | ✅ 原生 |

> 硬件相关步骤（校准、数据采集等）请移步 `shared/GUIDE.md`。

---

## 第二章：模型训练

### 2.1 训练原理

#### X-VLA 的两阶段训练

```
Phase I:  大规模预训练（蚂蚁/2toINF 已为你完成）
  ┌──────────────────────────────────────────┐
  │ 数据: 290K episodes, 7平台, 5种机械臂    │
  │ 模型: 纯 Transformer Encoder, 0.9B 参数  │
  │ 输出: 通用基座模型 "lerobot/xvla-base"    │
  └──────────────────────────────────────────┘
                      ↓
Phase II: 领域适配（你需要做的部分）
  ┌──────────────────────────────────────────┐
  │ 方法: Soft Prompt (9M参数) + 可选的      │
  │        微调 policy transformer            │
  │ 数据: 你的 50-200 个演示                 │
  │ 输出: XLeRobot + 叠衣服专用策略           │
  └──────────────────────────────────────────┘
```

#### 什么是 Soft Prompt？

```
传统方法: 微调全部参数 (4B 模型 × 16bit ≈ 8GB)
          需要大量数据 + GPU

X-VLA: 只训练 Soft Prompt (9M 参数)
        Soft Prompt = 一组可学习的嵌入向量
        它们告诉模型: "你现在是 XLeRobot，关节是这样定义的..."
        
        类比:
        你的大脑是基座模型
        Soft Prompt 是你戴上的一副特殊眼镜
        眼镜让你看到 XLeRobot 的世界
        不用重新训练大脑，只需要调整眼镜
```

### 2.2 执行训练

#### 环境准备

```bash
# 安装 X-VLA 依赖
cd /home/zach/lerobot
pip install -e ".[xvla]"

# 或直接 pip 安装
pip install 'lerobot[xvla]'
```

#### 训练命令

训练在云服务器或本地 GPU 上运行。默认使用 `train.sh`，支持以下参数：

| 参数 | 说明 | 示例 |
|------|------|------|
| `--dataset` | 数据集 ID | `zonglin11/xlerobot_fold_cloth` |
| `--model-path` | 基座模型 | `lerobot/xvla-folding` 或 `lerobot/xvla-base` |
| `--steps` | 训练步数 | `15000`（轻量）或 `30000`（全量） |
| `--output-dir` | 输出目录 | `./outputs/xvla_xlerobot_fold` |
| `--repo-id` | HF 模型 ID（上传到 Hub） | `zonglin11/xvla-xlerobot-fold` |
| `--rename-map` | 摄像头键名映射 | 见下方说明 |
| `--light` | 轻量模式（冻结视觉编码器） | 默认 |

**摄像头键名映射：**

XLeRobot 数据集中的摄像头键名与 X-VLA 模型期望的不同，需要通过 `--rename-map` 映射：

```bash
# XLerobot（2轮/3轮版）：top, left_wrist, right_wrist
--rename-map '{"observation.images.top": "observation.images.image",
               "observation.images.left_wrist": "observation.images.image2",
               "observation.images.right_wrist": "observation.images.image3"}'

# LeRobot 双臂：left_top, left_wrist, right_wrist
--rename-map '{"observation.images.left_top": "observation.images.image",
               "observation.images.left_wrist": "observation.images.image2",
               "observation.images.right_wrist": "observation.images.image3"}'
```

> `action_mode=auto` 会自动检测动作维度（12 或 16），无需手动配置。

> **⚠️ 重要：状态维度修复**
> X-VLA 基座模型（`xvla-folding`、`xvla-base`）的 `state_dim=8`，但 XLeRobot 2 轮版是 **16-DOF**、3 轮版是 **17-DOF**。如果不修复，训练时状态会被截断，推理效果会很差。
>
> **训练前必须先修补基座模型：**
>
> ```bash
> # 2 轮差速版（16-DOF）——输出到 shared/patched_models/
> bash /home/zach/XLeRobot/shared/patch_state_dim.sh
> # → shared/patched_models/xvla-folding-state16/
>
> # 3 轮全向版（17-DOF）
> bash /home/zach/XLeRobot/shared/patch_state_dim.sh --dof 17
> # → shared/patched_models/xvla-folding-state17/
>
> # 从 xvla-base 开始
> bash /home/zach/XLeRobot/shared/patch_state_dim.sh --model lerobot/xvla-base
> # → shared/patched_models/xvla-base-state16/
> ```
>
> 训练时用修补后的模型路径，不要直接用 `lerobot/xvla-folding`：
> ```bash
> bash train.sh --model-path shared/patched_models/xvla-folding-state16 --dataset ...
> ```
> `patch_state_dim.sh` 只改 `config.json` 里的一行：`state_dim: 8 → 16（或 17）`，其他配置不变。

#### 训练示例

**叠衣服（解冻视觉编码器，不上传 Hub）：**

先修补基座模型：
```bash
bash /home/zach/XLeRobot/shared/patch_state_dim.sh
```

再训练：
```bash
cd ~/xvla_deploy
conda activate lerobot
export LEROBOT_CACHE=/data/datasets

bash train.sh \
  --dataset zonglin11/xlerobot_fold_cloth \
  --model-path shared/patched_models/xvla-folding-state16 \
  --steps 30000 \
  --output-dir ./outputs/xvla_xlerobot_fold_vision \
  --rename-map '{"observation.images.top": "observation.images.image",
                 "observation.images.left_wrist": "observation.images.image2",
                 "observation.images.right_wrist": "observation.images.image3"}'
```

**叠衣服（上传到 HF Hub）：**

```bash
bash /home/zach/XLeRobot/shared/patch_state_dim.sh
bash train.sh \
  --dataset zonglin11/xlerobot_fold_cloth \
  --model-path shared/patched_models/xvla-folding-state16 \
  --steps 30000 \
  --output-dir ./outputs/xvla_xlerobot_fold_vision \
  --repo-id zonglin11/xvla-xlerobot-fold-vision \
  --rename-map '{"observation.images.top": "observation.images.image",
                 "observation.images.left_wrist": "observation.images.image2",
                 "observation.images.right_wrist": "observation.images.image3"}'
```

**擦桌子（从 xvla-base 开始训练，不上传）：**

```bash
bash /home/zach/XLeRobot/shared/patch_state_dim.sh --model lerobot/xvla-base
bash train.sh \
  --dataset zonglin11/xlerobot_clean_table \
  --model-path shared/patched_models/xvla-base-state16 \
  --steps 30000 \
  --output-dir ./outputs/xvla_xlerobot_vision_unfrozen \
  --rename-map '{"observation.images.top": "observation.images.image",
                 "observation.images.left_wrist": "observation.images.image2",
                 "observation.images.right_wrist": "observation.images.image3"}'
```

**擦桌子（上传到 HF Hub）：**

```bash
bash /home/zach/XLeRobot/shared/patch_state_dim.sh --model lerobot/xvla-base
bash train.sh \
  --dataset zonglin11/xlerobot_clean_table \
  --model-path shared/patched_models/xvla-base-state16 \
  --steps 30000 \
  --output-dir ./outputs/xvla_xlerobot_v2 \
  --repo-id zonglin11/xvla-xlerobot-clean-table-v2 \
  --rename-map '{"observation.images.top": "observation.images.image",
                 "observation.images.left_wrist": "observation.images.image2",
                 "observation.images.right_wrist": "observation.images.image3"}'
```

#### 训练日志解读

```
Step 1/5000 | loss: 0.8421 | lr: 1e-3 | VRAM: 8.2GB | 3.2 it/s
Step 100/5000 | loss: 0.5213 | lr: 1e-3 | VRAM: 8.3GB | 3.1 it/s
Step 500/5000 | loss: 0.3125 | lr: 1e-3 | VRAM: 8.3GB | 3.1 it/s
Step 1000/5000 | loss: 0.2156 | lr: 1e-3 | VRAM: 8.4GB | 3.1 it/s
```

**loss 解读：**
- 初始 ~0.8：模型刚开始，预测不准确
- 下降到 ~0.2：模型开始理解任务模式
- 稳定在 ~0.1：训练收敛
- 如果 loss 不下降：检查数据集是否有问题

#### 训练完成

```
✅ Training complete!
  Model saved to: ./outputs/xvla_xlerobot_fold
  Checkpoints:    ./outputs/xvla_xlerobot_fold/checkpoints
  
  best/  ← 用于推理部署
  last/  ← 可用于继续训练
```

### 2.3 监控训练

```bash
# 启动 TensorBoard
conda activate lerobot
tensorboard --logdir=./outputs/xvla_xlerobot_fold/logs --port=6006

# 本地浏览器访问
# http://<云服务器IP>:6006
```

**关注的关键指标：**

| 指标 | 正常表现 | 含义 |
|------|---------|------|
| `train/loss` | 持续下降 | 模型在学 |
| `train/grad_norm` | 0.1-10 之间 | 梯度稳定 |
| `train/lr` | 按计划衰减 | 学习率正常 |
| `val/loss` | 验证集上也在下降 | 没有过拟合 |

**提前停止条件：**
- loss 降到 0.05 以下 → 基本已学到
- loss 连续 500 步不降 → 已收敛
- loss 开始上升 → 过拟合，应早停

---

## 第三章：推理服务部署

### 3.1 XLeRobot 本地推理

#### 终端 1：启动推理服务

```bash
cd /home/zach/XLeRobot/xvla_deploy
conda activate lerobot

# 用本地 checkpoint
python xvla_deploy/server.py \
  --model-path ./outputs/xvla_xlerobot_clean_table/checkpoints/last/pretrained_model \
  --port 8000

# 或直接从 HuggingFace 加载
python xvla_deploy/server.py \
  --model-path zonglin11/xvla_xlerobot_clean_table \
  --port 8000
```

**server.py 参数：**
```
--model-path PATH  模型路径或 HF ID (默认: lerobot/xvla-folding)
--port INT         服务端口 (默认: 8000)
--host TEXT        绑定地址 (默认: 0.0.0.0)
--device TEXT      推理设备 (默认: auto)
```

#### 终端 2：运行客户端

```bash
cd /home/zach/XLeRobot
conda activate lerobot
export PYTHONPATH=/home/zach/XLeRobot/software/src:$PYTHONPATH

python xvla_deploy/client.py \
  --server-url http://localhost:8000 \
  --task "clean table" \
  --smooth-ratio 0.5
```

**摄像头配置**在 `xvla_deploy/client.py` 的 `main()` 函数里修改：

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

**client.py 参数：**
```
--server-url URL      云端推理服务地址 (默认: http://localhost:8000)
--task TEXT           语言指令 (默认: "fold the towel on the table")
--domain-id INT       领域ID (默认: 0)
--denoise-steps INT   去噪步数 (默认: 10, 范围 5-50)
--control-freq FLOAT  控制频率Hz (默认: 30)
--port1 PATH          左臂总线端口 (默认: /dev/ttyACM0)
--port2 PATH          右臂总线端口 (默认: /dev/ttyACM1)
```

### 3.2 XLeRobot 远程推理

#### 云服务器：启动推理服务

```bash
conda activate lerobot
cd ~/xvla_deploy

# 上传本地模型到云服务器（如有需要）
scp -P <端口> -r /home/zach/XLeRobot/xvla_deploy/outputs/xxx/checkpoints/last/pretrained_model \
  featurize@<IP>:~/xvla_deploy/my_model/

# 启动服务
python server.py \
  --model-path ~/xvla_deploy/my_model \
  --port 8000
```

#### 本地电脑：SSH 隧道

```bash
ssh -N -f -L 8000:localhost:8000 featurize@<IP> -p <端口>
```

#### 运行客户端

```bash
cd /home/zach/XLeRobot
conda activate lerobot
export PYTHONPATH=/home/zach/XLeRobot/software/src:$PYTHONPATH

python xvla_deploy/client.py \
  --server-url http://localhost:8000 \
  --task "fold the cloth on the table"
```

### 3.3 双臂推理（bi_so_follower）

#### 本地

```bash
# 终端 1：启动服务
cd /home/zach/XLeRobot/xvla_deploy
conda activate lerobot
python server.py \
  --model-path ./outputs/bilerobot_fold/checkpoints/last/pretrained_model \
  --port 8000

# 终端 2：运行双臂客户端
cd /home/zach/XLeRobot
conda activate lerobot
export PYTHONPATH=/home/zach/XLeRobot/software/src:$PYTHONPATH
python xvla_deploy/client_bimanual.py \
  --server-url http://localhost:8000 \
  --task "fold the towel on the table"
```

**双臂摄像头配置**在 `client_bimanual.py` 的 `connect_robot()` 方法里修改：

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

#### 远程

```bash
# 云服务器：上传模型并启动
scp -P <端口> -r /home/zach/XLeRobot/xvla_deploy/outputs/bilerobot_fold/checkpoints/last/pretrained_model \
  featurize@<IP>:~/xvla_deploy/bimanual_model/
ssh -p <端口> featurize@<IP>
conda activate lerobot
cd ~/xvla_deploy
python server.py --model-path ~/xvla_deploy/bimanual_model --port 8000

# 本地 SSH 隧道
ssh -N -f -L 8000:localhost:8000 featurize@<IP> -p <端口>

# 运行客户端
cd /home/zach/XLeRobot
conda activate lerobot
export PYTHONPATH=/home/zach/XLeRobot/software/src:$PYTHONPATH
python xvla_deploy/client_bimanual.py \
  --server-url http://localhost:8000 \
  --task "fold the towel on the table"
```

### 3.4 验证服务

```bash
# 健康检查
curl http://localhost:8000/health
# → {"status":"ok","model":"lerobot/xvla-folding","device":"cuda"}
```

### 3.5 后台运行

```bash
# nohup 保持后台运行（SSH 断开后不停止）
nohup python server.py \
  --model-path ./outputs/xvla_xlerobot_fold/checkpoints/best \
  --port 8000 > server.log 2>&1 &

tail -f server.log   # 查看日志
kill %1              # 停止服务
```

> 云服务器的端口开放方式各异，请参照各平台文档操作（AutoDL 自定义服务、Featurize 端口映射等）。

---

## 附录A：常见错误

### A.1 训练时报错

#### "CUDA out of memory"

```bash
# 减小 batch size
bash train.sh --light --batch-size 2

# 梯度累积
# 在 train.sh 的 ARGS 中加:
--optimizer.gradient_accumulation_steps=4

# 梯度检查点
--policy.gradient_checkpointing=true
```

#### "Action dimension mismatch"

```bash
# 确保使用 action_mode=auto（train.sh 中已默认）
--policy.action_mode=auto
```

#### "Dataset not found"

```bash
export LEROBOT_CACHE=/data/datasets
# 或检查 ~/.cache/huggingface/lerobot/ 下是否有数据
```

### A.2 推理时报错

#### "Connection refused"

```bash
ps aux | grep server.py      # 服务在运行？
netstat -tlnp | grep 8000    # 端口在监听？
# 确认安全组开放了端口
```

#### "Model not loaded"

```bash
ls -la ./outputs/xxx/checkpoints/best    # 模型文件存在？
huggingface-cli whoami                   # HF 登录正常？
```

#### "Timeout"

```bash
nvidia-smi                    # GPU 被占用？
# 减少去噪步数: --denoise-steps 5
ping <云服务器IP>             # 网络延迟？
```

---

## 附录B：概念参考

### B.1 X-VLA 模型架构

```
输入                         输出
┌──────┐                    ┌──────┐
│ 图像  │───┐              ┌→│动作(1)│
│ 3路  │   │  ┌──────────┐│ └──────┘
└──────┘   ├─→│ Qwen2.5  ││ ┌──────┐
┌──────┐   │  │ -VL      │├→│动作(2)│
│ 语言  │───┘  │ 3B       ││ └──────┘
│指令   │      │ VLM      ││ ┌──────┐
└──────┘      │ 编码器    │├→│...    │
              └──────────┘│ └──────┘
┌──────┐      ┌──────────┐│ ┌──────┐
│关节状│─────→│ Soft     │└→│动作(32)│
│态    │      │ Prompt   │  └──────┘
└──────┘      │ (9M参数) │
              │ 可训练   │
              └──────────┘
```

- **Qwen2.5-VL 3B**：阿里巴巴多模态大模型（视觉+语言）
- **Action Decoder**：Transformer 解码器，生成动作序列
- **Soft Prompt**：一组可学习的嵌入向量（9M 参数）

### B.2 动作空间

```
XLeRobot 动作空间:

左臂 (6维):
  [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

右臂 (6维):
  [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

总共: 12 维（手臂），加头部和底盘共 16/17 维
```

X-VLA 内部使用 20 维动作空间（`action_mode=auto` 时自动适配）：
- 训练：原始维度 → pad → 20 维
- 推理：20 维 → trim → 原始维度

---

## 附录C：命令速查表

### 训练

```bash
# 先修补基座模型，再用修补后的路径：
#   bash patch_state_dim.sh  →  shared/patched_models/xvla-folding-state16/

bash train.sh --dataset your/dataset --model-path shared/patched_models/xvla-folding-state16

# 指定步数和输出目录
bash train.sh --dataset your/dataset --steps 15000 --output-dir ./outputs/xxx

# 训练并上传到 Hub
bash train.sh --dataset your/dataset --repo-id your/xvla-model
```

### 推理

```bash
# 启动服务
python server.py --model-path ./outputs/xxx/checkpoints/best --port 8000

# 后台运行
nohup python server.py --model-path ./outputs/.../best --port 8000 > server.log 2>&1 &

# 测试服务
curl http://localhost:8000/health
```

### 参数参考

**train.sh：**
```
--dataset ID        数据集ID (默认: your/xlerobot-cloth-fold)
--output-dir DIR    输出目录 (默认: ./outputs/xvla_xlerobot_fold)
--model-path PATH   基座模型路径 (默认: lerobot/xvla-folding)
--repo-id ID        HF 模型 ID（推送模型到 Hub）
--light             轻量模式（默认）
--resume PATH       从 checkpoint 继续训练
--rename-map JSON   摄像头键名映射
```

**server.py：**
```
--model-path PATH  模型路径或 HF ID (默认: lerobot/xvla-folding)
--port INT         服务端口 (默认: 8000)
--host TEXT        绑定地址 (默认: 0.0.0.0)
--device TEXT      推理设备 (默认: auto)
```

**client.py：**
```
--server-url URL     云端推理服务地址 (默认: http://localhost:8000)
--task TEXT          语言指令
--domain-id INT      领域ID (默认: 0)
--denoise-steps INT  去噪步数 (默认: 10)
--control-freq FLOAT 控制频率Hz (默认: 30)
--port1 PATH         左臂总线端口 (默认: /dev/ttyACM0)
--port2 PATH         右臂总线端口 (默认: /dev/ttyACM1)
```
