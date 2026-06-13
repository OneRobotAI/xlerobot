# X-VLA × XLeRobot 双臂叠衣服完整教程

> 基于 X-VLA (0.9B, ICLR 2026) + XLeRobot (双臂 SO-100) 实现叠衣服
> 资源友好：单卡 RTX 4090 可训练，推理仅需 ~9GB VRAM

---

## 📋 目录

- [1. 整体架构](#1-整体架构)
- [2. 为什么选 X-VLA](#2-为什么选-x-vla)
- [3. 硬件准备](#3-硬件准备)
- [4. 软件环境](#4-软件环境)
- [Step 1: 数据采集](#step-1-数据采集)
- [Step 2: 数据集准备](#step-2-数据集准备)
- [Step 3: 云端训练](#step-3-云端训练)
- [Step 4: 部署推理服务](#step-4-部署推理服务)
- [Step 5: 运行客户端](#step-5-运行客户端)
- [常见问题](#常见问题)
- [文件索引](#文件索引)

---

## 1. 整体架构

```
┌─────────────────────────────────────────────────────┐
│  本地 (你的电脑 + XLeRobot)                          │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │ ① 数据采集                                     │   │
│  │   双主臂遥操作 → 录制叠衣服演示数据                │   │
│  │   collect_data.py                              │   │
│  └──────────┬───────────────────────────────────┘   │
│             │ 数据集 (LeRobot格式)                    │
│             ▼                                       │
│  ┌──────────────────────────────────────────────┐   │
│  │ ⑤ 执行推理                                     │   │
│  │   client.py → 发送观测 → 接收动作 → 控制机器人   │   │
│  │   ↕ HTTP POST /act                             │   │
│  └──────────┬───────────────────────────────────┘   │
│             │ 网络 (WiFi/5G)                         │
└─────────────┼───────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────┐
│  云端 GPU 服务器 (百舸/阿里云/AWS/AutoDL)             │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │ ② 环境搭建 + ③ 训练                            │   │
│  │   deploy.sh (一键安装)                         │   │
│  │   train.sh (LoRA微调, ~1-4小时)                │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │ ④ 推理服务                                     │   │
│  │   server.py (FastAPI, 加载训练好的模型)          │   │
│  │   暴露 POST /act 接口                          │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## 2. 为什么选 X-VLA

### 与 LeRobot 生态其他算法对比

| 对比项 | X-VLA 0.9B | LingBot-VLA 4B | LingBot-VA | SmolVLA | ACT/DP |
|--------|-----------|---------------|-----------|---------|--------|
| 参数量 | **0.9B** | 4.2B | ~23GB文件 | 2.25B | ~80M |
| LoRA可训练参数 | **9M (1%)** | 4.4M (0.11%) | ❌ 无人做 | 需自定义 | 全量 |
| 单卡4090训练 | **~9GB** | ~9GB (TwinRL) | ❌ 需8×A800 | ~9GB | 4-6GB |
| 叠衣服能力 | **`xvla-folding` 100%** | GM-100含叠衣服 | 叠裤子70% | — | 40-62% |
| 双臂RoboTwin | **70%** | 86-88% | **92.9%** | — | — |
| Server-Client架构 | **原生支持** | 需自建 | ✅原生 | 需自建 | ❌ |
| LeRobot集成 | **原生** | 独立仓库 | 独立仓库 | 原生 | 原生 |
| 论文 | **ICLR 2026** | 预印本 | RSS 2026 | — | — |

**结论**：X-VLA 在资源需求、叠衣服效果、部署便利性三者之间取得了最好的平衡。

### 核心优势

1. **Soft Prompt 机制**：用 9M 参数（1%）就能适配新机器人，其他模型做不到
2. **原生 Server-Client**：直接支持云端推理，不需要自己写服务端
3. **叠衣服专用 checkpoint**：`lerobot/xvla-folding` 在 Soft-FOLD 数据集上 100% 成功率
4. **auto action mode**：自动检测动作维度，5-DOF/6-DOF 双臂通吃
5. **LeRobot 原生集成**：`pip install lerobot[xvla]` 一条命令搞定

## 3. 硬件准备

### 你需要

| 硬件 | 数量 | 说明 |
|------|------|------|
| XLeRobot (含双臂) | 1台 | 你的机器人，含 2 个从臂 |
| 额外主臂 (SO-100) | 2个 | 用于遥操作采集数据，USB 连接电脑 |
| USB 摄像头 | 3个 | 顶部全局 + 左右手腕视角，建议 640×480 |
| GPU 云服务器 | 1台 | 训练和推理用，RTX 4090 起步 |

### USB 端口布局（数据采集时）

```
笔记本电脑 USB 口:
┌────────────────────────────────────────────────────┐
│  左主臂 (Leader)    右主臂 (Leader)               │
│  /dev/ttyACM2        /dev/ttyACM3                 │
│  左从臂 (Follower)  右从臂 (Follower)             │
│  /dev/ttyACM0        /dev/ttyACM1                 │
│                                                    │
│  顶部摄像头          左手腕摄像头    右手腕摄像头    │
│  /dev/video0         /dev/video2     /dev/video4  │
└────────────────────────────────────────────────────┘
```

实际端口号可能不同，用 `lerobot-find-port` 查看。

## 4. 软件环境

### 4.1 本地电脑（数据采集 + 控制机器人）

```bash
# 激活你的 XLeRobot 环境（现有环境应该已经包含 LeRobot）
# 如果没有，安装：
pip install lerobot[xvla]

# 验证
python -c "from lerobot.policies.xvla.modeling_xvla import XVLAPolicy; print('✅ X-VLA OK')"

# 设置 XLerobot 软链接（client.py 需要导入 XLerobot 类）
bash /home/zach/XLeRobot/xvla_deploy/setup_local.sh
```

### 4.2 云端 GPU 服务器（训练 + 推理）

把 `xvla_deploy/` 整个目录传到云服务器：

```bash
scp -P <端口号> -r /home/zach/XLeRobot/xvla_deploy/ user@你的云服务器IP:~

# 登录云服务器
ssh -p <端口号> user@你的云服务器IP
cd xvla_deploy

# 激活已有 conda 环境（如果没有，先创建）
conda activate lerobot
# conda create -n lerobot python=3.12 && conda activate lerobot && pip install 'lerobot[xvla]'
```

> 云端只需要 LeRobot + X-VLA，`deploy.sh` 和 `setup_local.sh` 都不需要。
> 云服务器不控制机器人硬件，不需要 XLerobot 类。

---

## Step 1: 数据采集

在连接了全部硬件的本地电脑上运行。

### 1.1 校准机械臂

```bash
# 校准主臂（只需要做一次）
lerobot-calibrate --teleop.type=so100_leader --teleop.port=/dev/ttyACM2 --teleop.id=left_leader
lerobot-calibrate --teleop.type=so100_leader --teleop.port=/dev/ttyACM3 --teleop.id=right_leader

# 校准从臂
lerobot-calibrate \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/ttyACM0 \
  --robot.right_arm_port=/dev/ttyACM1 \
  --robot.id=xlerobot_follower
```

### 1.2 录制演示数据

```bash
python collect_data.py \
  --num-episodes 50 \
  --task "fold the towel on the table" \
  --repo-id your/xlerobot-cloth-fold
```

### 1.3 数据格式

每条演示包含（30 秒 × 30fps = 900 帧）：

```
action:             12维 [左臂6: shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper,
                          右臂6: 同上]
observation.state:  12维 [同上]
cam_top:            640×480 RGB 视频
cam_left_wrist:     640×480 RGB 视频
cam_right_wrist:    640×480 RGB 视频
```

### 1.4 录制技巧

- **从简单开始**：先对折毛巾（10-20 个演示），再尝试 T 恤
- **固定视角**：摄像头位置固定不动，X-VLA 依赖视觉特征
- **一致性**：每次叠法尽量一致，减少策略困惑
- **速度均匀**：遥操作时动作不要太快或太慢

---

## Step 2: 数据集准备

### 2.1 验证数据集

```bash
python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('your/xlerobot-cloth-fold')
print(f'Episode数: {ds.num_episodes}')
print(f'总帧数: {len(ds)}')
print(f'动作维度: {ds[0][\"action\"].shape}')     # 应该是 (12,)
print(f'状态维度: {ds[0][\"observation.state\"].shape}')  # 应该是 (12,)
"
```

### 2.2 上传到云服务器

```bash
# 方式 A: rsync 直传（推荐，不用过 Hub）
rsync -avz --progress \
  ~/.cache/huggingface/lerobot/your/xlerobot-cloth-fold/ \
  user@云服务器IP:/data/datasets/xlerobot-cloth-fold/

# 方式 B: 推送到 HuggingFace Hub（需要网络通畅）
python -m lerobot.datasets.push_to_hub \
  --repo-id=your/xlerobot-cloth-fold \
  --root=~/.cache/huggingface/lerobot/
```

---

## Step 3: 云端训练

在 GPU 云服务器上运行。

### 3.1 训练模式选择

| 模式 | 命令 | 时间 (单卡4090) | VRAM | 适用场景 |
|------|------|----------------|------|---------|
| 🟢 **轻量模式** | `bash train.sh --light` | **~1 小时** | ~9GB | 快速验证，推荐起步 |
| 🔵 全量模式 | `bash train.sh` | ~4 小时 | ~16GB | 追求更好效果 |
| 🟣 微调专用checkpoint | `bash train.sh --model-path lerobot/xvla-folding` | — | — | 叠衣服专用初始化 |

### 3.2 轻量模式（推荐先跑这个）

```bash
cd xvla_deploy

# 激活虚拟环境（如果 deploy 时创建了）
source venv_xvla/bin/activate

# 指定数据集路径（如果不在默认缓存位置）
export LEROBOT_CACHE=/data/datasets

# 开始训练
bash train.sh --light
```

实际执行的是：

```bash
lerobot-train \
  --dataset.repo_id=your/xlerobot-cloth-fold \
  --output_dir=./outputs/xvla_xlerobot_fold \
  --policy.path="lerobot/xvla-folding" \
  --policy.dtype=bfloat16 \
  --policy.action_mode=auto \
  --policy.max_action_dim=20 \
  --policy.freeze_vision_encoder=true \
  --policy.freeze_language_encoder=true \
  --policy.train_policy_transformer=false \
  --policy.train_soft_prompts=true \
  --steps=5000 \
  --batch_size=8
```

**原理**：只训练 X-VLA 的 soft prompt（9M 参数，1%），冻结 Qwen2.5-VL 全部权重。这是 X-VLA 论文中验证过的 Phase II 适配方式，在 LIBERO 上能达到 93% 成功率（vs 全量 98%）。

### 3.3 全量模式（效果更好）

```bash
bash train.sh
```

多训练 policy transformer 层，但不 freeze VLM 会导致显存需求上升到 ~16GB。

### 3.4 训练完成后

```
outputs/xvla_xlerobot_fold/
├── checkpoints/
│   ├── last/              # 最后一步
│   └── best/              # 验证集最佳（用于部署）
├── logs/                  # TensorBoard 日志
└── config.yaml            # 配置备份
```

---

## Step 4: 部署推理服务

在 GPU 云服务器上运行。

### 4.1 启动服务

```bash
# 使用预训练的叠衣服 checkpoint（不训练也能跑）
bash deploy.sh --port 8000

# 或使用你微调好的模型
bash deploy.sh \
  --model-path ./outputs/xvla_xlerobot_fold/checkpoints/best \
  --port 8000
```

### 4.2 验证服务

```bash
# 健康检查
curl http://localhost:8000/health
# → {"status":"ok","model":"lerobot/xvla-folding","device":"cuda"}

# 推理测试
python -c "
import requests, json, numpy as np
resp = requests.post('http://localhost:8000/act', json={
    'proprio': json.dumps(np.zeros(12).tolist()),
    'language_instruction': 'fold the towel',
    'image0': json.dumps(np.zeros((256,256,3), dtype=np.uint8).tolist()),
    'image1': json.dumps(np.zeros((256,256,3), dtype=np.uint8).tolist()),
    'image2': json.dumps(np.zeros((256,256,3), dtype=np.uint8).tolist()),
    'domain_id': 0,
    'steps': 10,
})
print('动作形状:', np.array(resp.json()['action']).shape)
print('推理耗时:', resp.json()['inference_time_ms'], 'ms')
"
```

### 4.3 开放端口（云服务器安全组设置）

确保云服务器的安全组/防火墙开放了你指定的端口（默认 8000）：

```
阿里云: 安全组 → 入方向 → 添加 8000/TCP
百度百舸: 访问控制 → 安全组 → 添加规则
AutoDL: 自定义服务 → 暴露端口 8000
```

### 4.4 API 说明

```
POST http://<服务器外网IP>:8000/act

请求:
{
  "proprio": "[...]",              # JSON序列化的12维关节角度数组
  "language_instruction": "string", # 语言指令
  "image0": "[...]",               # 顶部摄像头 (H,W,3) uint8
  "image1": "[...]",               # (可选) 左手腕摄像头
  "image2": "[...]",               # (可选) 右手腕摄像头
  "domain_id": 0,                  # 领域ID，默认0
  "steps": 10                      # 去噪步数，越小越快
}

响应:
{
  "action": [[...], ...],          # 动作序列，shape (chunk_size, 12)
  "inference_time_ms": 45.2
}
```

---

## Step 5: 运行客户端

回到连接 XLeRobot 的本地电脑。

### 5.1 启动客户端

```bash
cd /home/zach/XLeRobot/xvla_deploy

python client.py \
  --server-url http://<你的云服务器外网IP>:8000 \
  --task "fold the towel on the table" \
  --port1 /dev/ttyACM0 \
  --port2 /dev/ttyACM1
```

### 5.2 运行流程

客户端每帧执行以下循环（30Hz）：

```
① 获取观测
   robot.get_observation()
   → 12维关节位置 (shoulder_pan, lift, elbow, wrist_flex, wrist_roll, gripper) × 2
   → 3路摄像头图像 (top, left_wrist, right_wrist)

② 请求云端推理
   POST /act → 发送观测数据
   ← 返回动作块 (32步, 12维/步)

③ 执行动作
   逐个执行动作块中的每一步
   同时保持 30Hz 控制频率
   当前块用尽 → 回到步骤①

④ 异步流水线
   执行当前块的同时，后台提前请求下一块
   网络延迟被完全隐藏
```

### 5.3 安全机制

- **服务不可用自动停止**：连不上服务器时安全停止，不执行危险动作
- **超时保护**：请求超过 10 秒自动放弃
- **键盘中断**：Ctrl+C 安全断开机器人

---

## 常见问题

### Q: 没有两个额外的主臂怎么办？

可以用 xlerobot 现有的其他遥操作方式：

```
键盘:   software/examples/2_dual_so100_keyboard_ee_control.py
Xbox:   software/examples/5_xlerobot_teleop_xbox.py
VR:     software/examples/8_vr_teleop_with_dataset_recording.py
```

虽然没有主臂直观，但采集的数据格式完全一样，不影响后续训练。

### Q: 云服务器选哪个平台？

| 平台 | GPU 类型 | 价格参考 | 教程支持 |
|------|---------|---------|---------|
| **百度百舸** | A800/A100 | 按需 | ✅ LingBot 官方教程 |
| **阿里云** | A100/4090 | 按需 | 需参考本项目 |
| **AutoDL** | 4090/3090 | **约 2-4元/小时** | 通用 Ubuntu |
| **恒源云** | 4090/A100 | 约 3-5元/小时 | 通用 Ubuntu |

> 建议：先用 AutoDL 租个 4090（~2元/小时）验证，效果满意后再到百舸上跑大规模训练。

### Q: 到底需要多少数据？

| 任务复杂度 | 最少演示数 | 推荐演示数 |
|-----------|----------|----------|
| 简单对折（毛巾） | 20 | 50 |
| 多次对折（毛巾） | 50 | 100 |
| T 恤折叠 | 100 | 200 |

X-VLA 官方叠衣服论文用了 1,200 条，但那是高精度叠衣。简单对折 50 条就能看到效果。

### Q: 训练显存不够怎么办？

```bash
# 方案1: 减小 batch size
bash train.sh --light --batch-size 2

# 方案2: 梯度累积
# 在 train.sh 的 ARGS 中加:
--optimizer.gradient_accumulation_steps=4

# 方案3: 梯度检查点
# 在 train.sh 的 ARGS 中加:
--policy.gradient_checkpointing=true

# 方案4: 缩小编码图像尺寸
# 在 train.sh 的 ARGS 中加:
--policy.input_size=224
```

### Q: 推理延迟太高怎么办？

| 问题 | 解决方法 |
|------|---------|
| 网络延迟大 | 云服务器选离你最近的区域 |
| 推理计算慢 | 减小 `--denoise-steps` (10→5)，速度翻倍 |
| 图像传输慢 | 减小图像分辨率 (640→320) |
| 动作不流畅 | 增大 `--control-freq` (30→50) |

### Q: 叠衣服效果不好怎么办？

1. **增加数据量**：50 → 100 → 200 个演示
2. **增加多样性**：换不同布料，不同初始摆放位置
3. **全量微调**：不用 `--light`，用 `bash train.sh`
4. **数据增强**：训练时加光照变化、图像噪声
5. **更好的 checkpoint**：如果 `xvla-folding` 效果不够，用 `lerobot/xvla-base` 从零在你的数据上微调

### Q: 5-DOF 和 6-DOF 双臂能同时兼容吗？

可以。X-VLA 的 `action_mode=auto` 自动检测数据集的动作维度：
- 6-DOF 双臂 → 12 维（标准，你的 XLeRobot）
- 5-DOF 双臂 → 10 维（少 wrist_flex）
- 混合使用 → 各自录制各自训练，模型结构不变

### Q: 训练完以后怎么把模型从云服务器拿回来？

```bash
# 从云服务器下载 checkpoint
scp -r user@云服务器IP:~/xvla_deploy/outputs/xvla_xlerobot_fold/checkpoints/best \
  /home/zach/XLeRobot/xvla_deploy/my_model/

# 本地重新启动推理服务（如果有 GPU）
python server.py --model-path ./my_model --port 8000
```

---

## 文件索引

所有文件位于 `/home/zach/XLeRobot/xvla_deploy/`：

| 文件 | 在哪运行 | 用途 |
|------|---------|------|
| `collect_data.py` | 本地（连机械臂） | 双主臂遥操作录制演示 |
| `train.sh` | 云端（有 GPU） | X-VLA 微调训练 |
| `server.py` | 云端（有 GPU） | FastAPI 推理服务 |
| `client.py` | 本地（连机械臂） | 连接云端，控制机器人 |
| `deploy.sh` | 云端（有 GPU） | 一键安装 + 启动服务 |
| `requirements.txt` | 两边 | Python 依赖列表 |
| `README.md` | — | 本教程
