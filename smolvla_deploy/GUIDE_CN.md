# SmolVLA × XLeRobot 训练与推理指南

> **版本**: 1.0 | 适用: XLeRobot (双臂 SO-100) + SmolVLA 450M
>
> 硬件准备、环境搭建、数据采集等通用步骤请见 `shared/GUIDE.md`。
> 本文档仅包含 SmolVLA 特有的训练和推理部署部分。

---

## 📋 目录

- [第一章：SmolVLA 简介](#第一章smolvla-简介)
- [第二章：模型训练](#第二章模型训练)
- [第三章：推理服务部署](#第三章推理服务部署)
- [附录A：常见错误](#附录a常见错误)
- [附录B：概念参考](#附录b概念参考)
- [附录C：命令速查表](#附录c命令速查表)

---

## 第一章：SmolVLA 简介

SmolVLA (450M) 是 HuggingFace 推出的轻量级 Vision-Language-Action 模型，基于 SmolVLM2-500M 视觉语言编码器和 Flow Matching 动作专家。

### 核心优势

| 特性 | 说明 |
|------|------|
| 参数量 | **450M**（VLM ~350M + Action Expert ~100M） |
| 训练目标 | Flow Matching（连续动作空间，非自回归） |
| 推理显存 | **~6GB**，RTX 3090/4090 可达 20-30Hz |
| 异步推理 | 原生支持（RTC），任务完成快 30% |
| LoRA 支持 | ~3GB VRAM 即可微调 |
| 社区数据集 | 487 个公开数据集可供参考 |
| DOF 上限 | **32-DOF**（XLeRobot 16/17-DOF 自动适配） |

### 与同类模型对比

| 对比项 | SmolVLA 450M | X-VLA 0.9B | ACT/DP |
|-------|-------------|-----------|--------|
| 参数量 | **450M** | 0.9B | ~80M |
| 单卡 4090 训练 | ✅ ~9GB | ✅ ~9GB | ✅ 4-6GB |
| 异步推理 | ✅ 原生 | ❌ 需自建 | ❌ |
| 推理显存 | **~6GB** | ~9GB | ~2GB |
| LeRobot 集成 | ✅ 原生 | ✅ 原生 | ✅ 原生 |

> 硬件相关步骤（校准、数据采集等）请移步 `shared/GUIDE.md`。

---

## 第二章：模型训练

### 2.1 训练说明

SmolVLA 在 LeRobot 里只有一种训练模式，没有"轻量/全量"之分。`freeze_vision_encoder` 和 `train_expert_only` 已经是模型 config 的默认值，无需手动指定。

训练时冻结视觉编码器，只训练动作专家（Action Expert），~9GB VRAM，单卡 4090 约 4 小时（20000 steps）。

#### 训练原理

```
SmolVLA 训练流程:

输入                             输出
┌──────┐                       ┌──────────┐
│ 图像  │───┐                 ┌→│ 动作序列  │
│ 3路   │   │  ┌───────────┐ │ │ (50步)   │
└──────┘   ├─→│ SmolVLM2   │ │ └──────────┘
┌──────┐   │  │ 视觉语言    │ │
│ 语言  │───┘  │ 编码器     │ │ ┌──────────┐
│指令   │      │ (冻结)     │ │→│ Flow      │
└──────┘      └───────────┘ │ │ Matching  │
┌──────┐                    │ │ 动作专家   │
│关节状│────────────────────┘ │ (可训练)   │
│态    │                      └──────────┘
└──────┘
```

- **SmolVLM2-500M**：SigLIP 视觉编码器 + SmolLM2 语言解码器
- **Flow Matching**：连续扩散过程，从噪声逐步去噪到动作
- **Action Expert**：带交叉注意力/自注意力的 Transformer

### 2.2 执行训练

#### 环境准备

```bash
# 安装 SmolVLA 依赖
cd /home/zach/lerobot
pip install -e ".[smolvla]"

# 或直接 pip 安装
pip install 'lerobot[smolvla]'
```

#### 训练命令

训练在本地 GPU 或云服务器上运行。使用 `train.sh`，支持以下参数：

| 参数 | 说明 | 示例 |
|------|------|------|
| `--dataset` | 数据集 ID | `your/xlerobot-clean-table` |
| `--model-path` | 基座模型 | `lerobot/smolvla_base` |
| `--output-dir` | 输出目录 | `./outputs/smolvla_xlerobot` |
| `--rename-map` | 摄像头键名映射 | 见下方说明 |

**摄像头键名映射：**

SmolVLA 期望的摄像头键名是 `camera1`、`camera2`、`camera3`，需要通过 `--rename-map` 映射：

```bash
# XLerobot（2轮/3轮版）：top, left_wrist, right_wrist
--rename-map '{"observation.images.top": "observation.images.camera1",
               "observation.images.left_wrist": "observation.images.camera2",
               "observation.images.right_wrist": "observation.images.camera3"}'

# LeRobot 双臂：left_top, left_wrist, right_wrist
--rename-map '{"observation.images.left_top": "observation.images.camera1",
               "observation.images.left_wrist": "observation.images.camera2",
               "observation.images.right_wrist": "observation.images.camera3"}'
```

#### 训练示例（XLeRobot 16-DOF）

```bash
cd ~/smolvla_deploy
conda activate lerobot
export LEROBOT_CACHE=/data/datasets

bash train.sh \
  --dataset zonglin11/xlerobot_fold_towel \
  --output-dir ./outputs/smolvla_xlerobot_fold_towel \
  --batch-size 16 \
  --repo-id zonglin11/smolvla-xlerobot-fold-towel \
  --rename-map '{"observation.images.top": "observation.images.camera1",
                 "observation.images.left_wrist": "observation.images.camera2",
                 "observation.images.right_wrist": "observation.images.camera3"}'
```

实际执行的是：

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=zonglin11/xlerobot_fold_towel \
  --output_dir=./outputs/smolvla_xlerobot_fold_towel \
  --policy.device=cuda \
  --batch_size=16 \
  --steps=20000 \
  --policy.push_to_hub=true \
  --policy.repo_id=zonglin11/smolvla-xlerobot-fold-towel \
  --rename_map='{"observation.images.top": "observation.images.camera1",
                 "observation.images.left_wrist": "observation.images.camera2",
                 "observation.images.right_wrist": "observation.images.camera3"}'
```

#### 训练示例（双臂 12-DOF）

如果使用 `bi_so_follower` 双臂数据（12-DOF，仅手臂，无头部和底盘），相机键名映射不同：

```bash
cd ~/smolvla_deploy
conda activate lerobot
export LEROBOT_CACHE=/data/datasets

bash train.sh \
  --dataset your/双臂数据集 \
  --rename-map '{"observation.images.left_top": "observation.images.camera1",
                 "observation.images.left_wrist": "observation.images.camera2",
                 "observation.images.right_wrist": "observation.images.camera3"}'
```

#### 训练日志解读

```
Step 1/20000 | loss: 0.95 | lr: 1e-4 | VRAM: 8.5GB | 3.5 it/s
Step 1000/20000 | loss: 0.42 | lr: 1e-4 | VRAM: 8.5GB | 3.5 it/s
Step 5000/20000 | loss: 0.21 | lr: 1e-4 | VRAM: 8.6GB | 3.5 it/s
Step 10000/20000 | loss: 0.12 | lr: 1e-4 | VRAM: 8.6GB | 3.4 it/s
```

**loss 解读：**
- 初始 ~0.95：模型刚开始，预测不准确
- 下降到 ~0.2：模型开始理解任务模式
- 稳定在 ~0.1：训练收敛

#### 训练完成

```
✅ Training complete!
  Model saved to: ./outputs/smolvla_xlerobot
  Checkpoints:    ./outputs/smolvla_xlerobot/checkpoints
  
  best/  ← 用于推理部署
  last/  ← 可用于继续训练
```

### 2.3 监控训练

```bash
conda activate lerobot
tensorboard --logdir=./outputs/smolvla_xlerobot/logs --port=6006
```

---

## 第三章：推理服务部署

### 3.1 本地推理

#### 终端 1：启动推理服务

```bash
cd /home/zach/XLeRobot/smolvla_deploy
conda activate lerobot

# 用微调好的模型
python server.py \
  --model-path ./outputs/smolvla_xlerobot/checkpoints/best \
  --port 8000

# 或使用预训练 base 模型（仅 6-DOF 单臂，用于测试）
python server.py --model-path lerobot/smolvla_base --port 8000
```

**server.py 参数：**
```
--model-path PATH  模型路径或 HF ID (默认: lerobot/smolvla_base)
--port INT         服务端口 (默认: 8000)
--host TEXT        绑定地址 (默认: 0.0.0.0)
--device TEXT      推理设备 (默认: auto)
```

#### 终端 2：运行客户端

```bash
cd /home/zach/XLeRobot/smolvla_deploy
conda activate lerobot

python client.py \
  --server-url http://localhost:8000 \
  --task "clean table"
```

**client.py 参数：**
```
--server-url URL     SmolVLA 推理服务地址 (默认: http://localhost:8000)
--task TEXT          语言指令 (默认: "clean table")
--control-freq FLOAT 控制频率 Hz (默认: 30)
--smooth-ratio FLOAT 动作平滑系数 (默认: 0.3)
--port1 PATH         左臂总线端口 (默认: /dev/ttyACM0)
--port2 PATH         右臂总线端口 (默认: /dev/ttyACM1)
```

#### 双臂推理（12-DOF，仅手臂）

如果模型是用双臂数据（bi_so_follower）训练的，推理时用 `xvla_deploy/client_bimanual.py`（HTTP API 与 SmolVLA 服务端兼容，直接复用）：

```bash
cd /home/zach/XLeRobot
conda activate lerobot
export PYTHONPATH=/home/zach/XLeRobot/software/src:$PYTHONPATH

python xvla_deploy/client_bimanual.py \
  --server-url http://localhost:8000 \
  --task "fold the towel"
```

> `client_bimanual.py` 的 HTTP API 与 SmolVLA 服务端兼容，直接复用。
> 双臂摄像头键名：`left_top`、`left_wrist`、`right_wrist`（bi_so_follower 会自动加 `left_`/`right_` 前缀）。

### 3.2 远程推理

#### 云服务器：启动推理服务

```bash
conda activate lerobot
cd ~/smolvla_deploy

# 上传本地模型到云服务器
scp -P <端口> -r /home/zach/XLeRobot/smolvla_deploy/outputs/xxx/checkpoints/best \
  featurize@<IP>:~/smolvla_deploy/my_model/

# 启动服务
python server.py --model-path ~/smolvla_deploy/my_model --port 8000
```

#### 本地电脑：SSH 隧道 + 客户端

```bash
# SSH 隧道
ssh -N -f -L 8000:localhost:8000 featurize@<IP> -p <端口>

# 运行客户端
cd /home/zach/XLeRobot/smolvla_deploy
conda activate lerobot
python client.py --server-url http://localhost:8000 --task "clean table"
```

### 3.3 验证服务

```bash
curl http://localhost:8000/health
# → {"status":"ok","model":"lerobot/smolvla_base","device":"cuda",
#    "state_dim":16,"action_dim":16,"chunk_size":50}
```

### 3.4 后台运行

```bash
nohup python server.py \
  --model-path ./outputs/smolvla_xlerobot/checkpoints/best \
  --port 8000 > server.log 2>&1 &

tail -f server.log   # 查看日志
kill %1              # 停止服务
```

> 云服务器的端口开放方式各异，请参照各平台文档操作（AutoDL 自定义服务、Featurize 端口映射等）。

---

## 附录：常见问题排查

### 摄像头画面过亮或偏色

客户端启动时会用 v4l2-ctl 设置摄像头参数。如果颜色异常，修改 `client.py` 中的配置：

```python
# auto_exposure=3（自动）或 1（手动）
# contrast、saturation、brightness——按需调节
```

### 服务端报错 "size of tensor a (12) must match size of tensor b (16)"

客户端只发送了 12 维手臂关节，模型需要完整的 16 维状态（12 手臂 + 2 头部 + 2 底盘）。确保 `_build_proprio()` 方法包含全部 16 个维度。

### 服务端报错 "expected input to have 3 channels, but got 480 channels"

图像必须是 (C, H, W) 格式，不能是 (H, W, C)。服务端已自动处理此转换。如果自己实现客户端，记得用 `.permute(2, 0, 1)` 转换通道顺序，并缩放到 float32 [0, 1]。

## 附录A：常见错误

### 训练时报错

#### "CUDA out of memory"

```bash
# 减小 batch size
lerobot-train --batch_size=16 ...

# 启用梯度检查点
--policy.gradient_checkpointing=true
```

#### "Dataset not found"

```bash
export LEROBOT_CACHE=/data/datasets
```

### 推理时报错

#### "Connection refused"

```bash
ps aux | grep server.py    # 服务在运行？
netstat -tlnp | grep 8000  # 端口在监听？
```

#### "Model not loaded"

```bash
ls -la ./outputs/xxx/checkpoints/best
huggingface-cli whoami
```

---

## 附录B：概念参考

### B.1 SmolVLA 模型架构

```
输入                          输出
┌──────┐                    ┌──────────┐
│ 图像  │───┐              ┌→│ action 1 │
│ 3路   │   │  ┌─────────┐ │ └──────────┘
└──────┘   ├─→│ SmolVLM2 │ │ ┌──────────┐
┌──────┐   │  │ 500M     │→│→│ action 2 │
│ 语言  │───┘  │ VLM     │ │ └──────────┘
│指令   │      │ (冻结)   │ │ ┌──────────┐
└──────┘      └────┬────┘ │→│ ...      │
                   │      │ └──────────┘
┌──────┐      ┌────┴────┐ │ ┌──────────┐
│关节状│─────→│ Action  │ └→│ action 50│
│态    │      │ Expert  │   └──────────┘
└──────┘      │ (可训练)│
              └─────────┘
```

- **SmolVLM2-500M**：HuggingFace 视觉语言模型（SigLIP 视觉编码器 + SmolLM2 语言解码器）
- **Action Expert**：Flow Matching Transformer，交替交叉注意力和自注意力层
- **输出**：50 步连续动作序列（chunk_size=50）

### B.2 动作空间

SmolVLA 支持最高 **32-DOF** 的动作/状态空间，自动适配 XLeRobot：

| 版本 | DOF | 分布 | SmolVLA 处理 |
|------|-----|------|-------------|
| 2轮差速版 | **16-DOF** | 12手臂 + 2头部 + 2底盘 | 训练时 pad→32，推理时 trim→16 |
| 3轮全向版 | **17-DOF** | 12手臂 + 2头部 + 3底盘 | 训练时 pad→32，推理时 trim→17 |
| 双臂标准版 | **12-DOF** | 6左臂 + 6右臂 | 训练时 pad→32，推理时 trim→12 |

### B.3 摄像头配置

SmolVLA 期望的摄像头键名：`camera1`、`camera2`、`camera3`

| SmolVLA 键名 | XLeRobot 键名 | 位置 |
|-------------|--------------|------|
| `observation.images.camera1` | `top` / `left_top` | 头顶全局 |
| `observation.images.camera2` | `left_wrist` | 左手腕 |
| `observation.images.camera3` | `right_wrist` | 右手腕 |

---

## 附录C：命令速查表

### 训练

```bash
# 基础训练
bash train.sh --dataset your/dataset

# 指定模型和输出目录
bash train.sh --dataset your/dataset --model-path lerobot/smolvla_base \
  --output-dir ./outputs/smolvla_xxx
```

### 推理

```bash
# 启动服务
python server.py --model-path ./outputs/smolvla_xlerobot/checkpoints/best --port 8000

# 后台运行
nohup python server.py --model-path ... --port 8000 > server.log 2>&1 &

# 测试服务
curl http://localhost:8000/health
```

### 参数参考

**train.sh：**
```
--dataset ID        数据集 ID (默认: your/xlerobot-clean-table)
--output-dir DIR    输出目录 (默认: ./outputs/smolvla_xlerobot)
--model-path PATH   基座模型路径 (默认: lerobot/smolvla_base)
--rename-map JSON   摄像头键名映射
```

**server.py：**
```
--model-path PATH  模型路径或 HF ID (默认: lerobot/smolvla_base)
--port INT         服务端口 (默认: 8000)
--host TEXT        绑定地址 (默认: 0.0.0.0)
--device TEXT      推理设备 (默认: auto)
```

**client.py：**
```
--server-url URL     SmolVLA 推理服务地址 (默认: http://localhost:8000)
--task TEXT          语言指令 (默认: "clean table")
--control-freq FLOAT 控制频率 Hz (默认: 30)
--smooth-ratio FLOAT 动作平滑系数 (默认: 0.3)
--port1 PATH         左臂总线端口 (默认: /dev/ttyACM0)
--port2 PATH         右臂总线端口 (默认: /dev/ttyACM1)
```
