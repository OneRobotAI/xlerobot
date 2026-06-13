# XLeRobot 实操指南（硬件 + 通用流程 篇）

> VLA 模型无关，X-VLA / SmolVLA 等方案共用
> 适用于 XLeRobot (双臂 SO-100) 全系列 及 LeRobot 双臂标准版 (bi_so_follower)
> 各 VLA 模型的训练与推理部署，见各自目录下的 GUIDE.md

---

## 📋 目录

- [第一章：认识你的 XLeRobot](#第一章认识你的-xlerobot)
- [第二章：环境搭建](#第二章环境搭建)
- [第三章：数据采集](#第三章数据采集)
- [第四章：模型训练与推理](#第四章模型训练与推理)
- [附录A：通用排错](#附录a通用排错)
- [附录B：概念参考](#附录b概念参考)

---

## 第一章：认识你的 XLeRobot

### 1.1 机器人构成

XLeRobot 是一个低成本的**双臂移动机器人**平台，包含：

| 部件 | 说明 |
|------|------|
| 双臂 | 2 × SO-100，每臂 6-DOF（含夹爪） |
| 头部 | 2 个舵机（pan/tilt） |
| 底盘 | 2 轮差速版（16-DOF）或 3 轮全向版（17-DOF） |
| 摄像头 | 3 路 USB 摄像头（顶部 + 左右手腕） |
| 主控 | 笔记本电脑（Ubuntu） |

### 1.2 各 VLA 模型的操作手册

每个 VLA 模型的训练和推理部署步骤，见各自目录下的独立手册：

- **X-VLA** → `xvla_deploy/GUIDE.md`
- **SmolVLA** → `smolvla_deploy/GUIDE.md`
- **其他模型** → 后续新增 `xxx_deploy/GUIDE.md`

本指南只覆盖与模型无关的硬件操作和环境准备。

### 1.3 USB 端口布局

```
笔记本电脑 USB 口:
┌───────────────────────────────────────────────────────────┐
│  左主臂 (Leader)    右主臂 (Leader)                      │
│  /dev/ttyACM2        /dev/ttyACM3                        │
│  左从臂 (Follower)  右从臂 (Follower)                    │
│  /dev/ttyACM0        /dev/ttyACM1                        │
│                                                           │
│  顶部摄像头          左手腕摄像头    右手腕摄像头          │
│  /dev/video0         /dev/video2     /dev/video4          │
└───────────────────────────────────────────────────────────┘
```

实际端口号可能不同，用 `lerobot-find-port` 查看。

---

## 第二章：环境搭建

### 2.1 你需要准备的硬件

#### 本地电脑（数据采集 + 可选训练/推理）

```
电脑要求:
├── 操作系统: Ubuntu 22.04 或更新
├── 内存: 16GB+
├── 存储: 100GB+ 空闲（数据存储）
├── USB 端口: 至少 5 个空闲 USB-A
│   ├── 左主臂 (Leader) × 1
│   ├── 右主臂 (Leader) × 1
│   ├── 左从臂 (Follower) × 1  (XLeRobot 本体)
│   ├── 右从臂 (Follower) × 1  (XLeRobot 本体)
│   └── 摄像头 × 3
├── GPU（可选，用于本地训练/推理）:
│   ├── NVIDIA GPU, 24GB+ VRAM（如 RTX 4090，可训练+推理）
│   └── NVIDIA GPU, 6GB+ VRAM（如 RTX 3060，仅推理）
└── 已有: XLeRobot (你的现有硬件)
    需要额外购买: 2个 SO-100 主臂（用于遥操作采集）
```

如果没有本地 GPU，或显存不够，可以使用云服务器进行训练和推理（见 2.3）。

#### 云端服务器（可选，本地 GPU 不够时使用）

```
服务器要求:
├── GPU: NVIDIA GPU, 24GB+ VRAM (RTX 4090 起步)
├── CUDA: 12.1+
├── 系统: Ubuntu 22.04
├── 存储: 50GB+
├── 网络: 公网 IP（客户端需要访问）
└── 推荐: [Featurize](https://featurize.cn?s=a6c5c56819ad418ab5c2ca96b794e40f) / AutoDL / 百度百舸 / 阿里云
```

### 2.2 本地环境安装

#### 步骤 1：确认 Python 版本

LeRobot 要求 Python **>= 3.12**。

```bash
python3 --version
# 期望输出: Python 3.12.x 或更高
```

如果版本不对，创建新环境：

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
```

#### 步骤 2：安装 LeRobot（基础）

官方安装文档：[https://huggingface.co/docs/lerobot/installation](https://huggingface.co/docs/lerobot/installation)

**源码安装（推荐，方便更新）：**

LeRobot 源码在 `/home/zach/lerobot/`：

```bash
cd /home/zach/lerobot
pip install -e .
```

**pip 安装（稳定版）：**

```bash
pip install lerobot
```

> 各个 VLA 模型的额外依赖（如 `lerobot[xvla]`、`lerobot[smolvla]`），见各模型目录下的 `GUIDE.md`。

#### 步骤 3：验证基础安装

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

#### 步骤 4：安装 ffmpeg（视频解码必需）

LeRobot 使用 [TorchCodec](https://github.com/meta-pytorch/torchcodec) 解码数据集视频，依赖 ffmpeg。

```bash
conda install ffmpeg -c conda-forge
```

#### 步骤 5：设置 XLerobot 软链接

`client.py` 需要导入 XLerobot 类来控制 XLeRobot 硬件，软链接让 Python 能找到项目中的机器人代码。

```bash
bash /home/zach/XLeRobot/shared/setup_local.sh
```

**预期输出：**
```
✅ 已创建软链接: .../lerobot/robots/xlerobot → .../XLeRobot/software/src/robots/xlerobot
✅ XLerobot 导入成功
```

> 软链接只在本地有效，不会进 Git。其他人克隆仓库后也需要运行一次。

#### 步骤 6：验证 XLeRobot 连接（可选）

```bash
# 连接好所有数据线、电源、舵机线
ls -la /dev/ttyACM*

# 在 examples 目录下测试
cd /home/zach/XLeRobot/software/examples
python 2_dual_so100_keyboard_ee_control.py
# 按 x 退出
```

### 2.3 云端环境安装（可选）

如果本地电脑没有 GPU（或显存不够），可以使用云 GPU 服务器进行训练和推理。如果有本地 GPU，可以跳过本节，直接在本地操作。

云端不需要 XLerobot 硬件类，只需要 PyTorch + LeRobot + VLA 模型依赖。

```bash
# SSH 登录云服务器
# featurize 平台用 featurize@，AutoDL 用 root@，其他平台请按需替换
ssh -p <端口号> featurize@<云服务器IP>

# 检查 GPU
nvidia-smi

# 创建并激活环境
conda create -n lerobot python=3.12 -y
conda activate lerobot

# 安装对应 VLA 模型依赖（见各模型 GUIDE.md，如 X-VLA / SmolVLA 等）
# pip install 'lerobot[xvla]'

# 上传部署包（按需上传对应模型的 deploy 目录）
cd /home/zach/XLeRobot
scp -P <端口号> -r xxx_deploy/ featurize@<云服务器IP>:~/
# 如有需要，也可上传 shared/（硬件工具）
```

> 云服务器不需要 `setup_local.sh`（不控制机器人硬件）。

### 2.4 SSH 隧道（云服务器没有公网端口时）

推理服务跑在云服务器的 `localhost:8000`，用 SSH 隧道转发到本地：

```bash
# 本地运行（后台）
ssh -N -f -L 8000:localhost:8000 featurize@<云服务器IP> -p <SSH端口>

# 验证
curl http://localhost:8000/health
```

关闭隧道：
```bash
ps aux | grep "ssh -N -f -L 8000" | grep -v grep
kill <进程ID>
```

如果云平台允许开放端口（AutoDL 自定义服务、阿里云安全组），也可直接开放后用公网 IP 访问。

---

## 第三章：数据采集

### 3.1 USB 端口

#### 查找设备

```bash
# 先不连任何机械臂，运行:
lerobot-find-port
# 输出: []

# 只连左主臂:
lerobot-find-port
# 输出: ['/dev/ttyACM0']

# 逐个连接并记录:
# 左主臂: /dev/ttyACM2
# 右主臂: /dev/ttyACM3
# 左从臂: /dev/ttyACM0
# 右从臂: /dev/ttyACM1
```

> 端口号每次插拔可能变化，每次使用前重新确认即可。

#### 固定 USB 端口（可选）

如果需要固定 USB 端口号，可以用 udev 规则绑定设备序列号：

```bash
# 查看设备序列号
udevadm info --name=/dev/ttyACM0 | grep ID_SERIAL

# 创建 udev 规则
sudo tee /etc/udev/rules.d/99-xlerobot.rules << 'EOF'
SUBSYSTEM=="tty", ATTRS{serial}=="<左主臂序列号>", SYMLINK+="xlerobot_left_leader"
SUBSYSTEM=="tty", ATTRS{serial}=="<右主臂序列号>", SYMLINK+="xlerobot_right_leader"
SUBSYSTEM=="tty", ATTRS{serial}=="<左从臂序列号>", SYMLINK+="xlerobot_left_follower"
SUBSYSTEM=="tty", ATTRS{serial}=="<右从臂序列号>", SYMLINK+="xlerobot_right_follower"
EOF
sudo udevadm control --reload-rules
```

### 3.2 校准机械臂

校准的目的是确保主臂和从臂在相同物理位置时输出相同的电机位置值。

#### 校准主臂

```bash
lerobot-calibrate \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/ttyACM2 \
  --teleop.right_arm_config.port=/dev/ttyACM3 \
  --teleop.id=bimanual_leader
```

**操作步骤：**
1. 左主臂移到零位（水平伸直，朝向正前方）→ 回车
2. 依次活动左主臂各关节到极限 → 回车
3. 右主臂移到零位 → 回车
4. 依次活动右主臂各关节到极限 → 回车
5. 校准数据保存到 `~/.cache/lerobot/calibration/bimanual_leader.yaml`

#### 校准从臂

##### 模式 A：XLerobot 全校准（推荐）

覆盖全部电机（含头部、底盘）。需要先创建软链接（如果之前已做可跳过）：

```bash
bash /home/zach/XLeRobot/shared/setup_local.sh

python /home/zach/XLeRobot/shared/calibrate_xlerobot.py \
  --port1 /dev/ttyACM0 \
  --port2 /dev/ttyACM1
```

- `port1` (/dev/ttyACM0) → 左臂（ID 1-6）+ 头部（ID 7-8）
- `port2` (/dev/ttyACM1) → 右臂（ID 1-6）+ 底盘（ID 9-10）

**校准过程：**
1. bus1 断电 → 手动将左臂 + 头部移到中间位置 → 回车
2. 依次活动左臂 + 头部各关节到极限 → 回车停止
3. bus2 断电 → 手动将右臂移到中间位置 → 回车
4. 依次活动右臂关节到极限 → 回车停止
5. 校准数据自动保存

##### 模式 B：LeRobot 双臂校准（仅手臂）

只校准 12 个手臂电机，跳过头部和底盘：

```bash
python /home/zach/XLeRobot/shared/calibrate_xlerobot.py \
  --mode lerobot \
  --port1 /dev/ttyACM0 \
  --port2 /dev/ttyACM1
```

#### 验证校准

```bash
# 移动主臂，看从臂是否跟随
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

如果从臂能平滑跟随主臂运动，校准成功。

### 3.3 录制演示数据

#### 查询摄像头序号

```bash
lerobot-find-cameras
```

#### 模式 A：XLerobot 采集（record.py）

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
  --dataset.repo_id=your/xlerobot-your-task \
  --dataset.single_task="your task description" \
  --dataset.num_episodes=50 \
  --dataset.fps=30 \
  --dataset.episode_time_s=90 \
  --dataset.reset_time_s=30 \
  --dataset.video=true \
  --display_data=true \
  --dataset.push_to_hub=false
```

**可调参数：**
| 参数 | 说明 | 建议值 |
|------|------|--------|
| `num_episodes` | 演示数 | 起步 50，越多越好 |
| `episode_time_s` | 每段最大秒数 | 30-90 |
| `reset_time_s` | 每段间重置时间 | 30 |
| `single_task` | 任务描述 | 要与训练/推理一致 |
| `repo_id` | 数据集 ID | HF Hub 用户名/数据集名 |

#### 模式 B：LeRobot 采集（collect_data.py）

```bash
cd /home/zach/XLeRobot/shared
python collect_data.py \
  --num-episodes 50 \
  --task "your task description" \
  --repo-id your/your-dataset-name \
  --episode-time 30 \
  --fps 30
```

端口和摄像头在脚本里改：
```bash
# /home/zach/XLeRobot/shared/collect_data.py 第 32-39 行
FOLLOWER_LEFT_PORT = "/dev/ttyACM0"    # 左从臂
FOLLOWER_RIGHT_PORT = "/dev/ttyACM1"   # 右从臂
LEADER_LEFT_PORT = "/dev/ttyACM2"      # 左主臂
LEADER_RIGHT_PORT = "/dev/ttyACM3"     # 右主臂
CAMERA_TOP = "/dev/video0"             # 顶部摄像头
CAMERA_LEFT_WRIST = "/dev/video2"      # 左手腕摄像头
CAMERA_RIGHT_WRIST = "/dev/video4"     # 右手腕摄像头
```

### 3.4 录制技巧

**新手建议：**
```
第 1-10 个:   简单动作（平铺 → 一次对折）
第 11-20 个:  改变初始位置
第 21-30 个:  改变操作方式（对折 → 再对折）
第 31-40 个:  换不同物体
第 41-50 个:  混合场景（随机位置 + 随机方式）
```

**好的演示 vs 差的演示：**
```
✅ 好的:
  - 动作流畅不卡顿
  - 物体完全处理好再进入下一步
  - 夹爪开合清晰

❌ 差的:
  - 中途停顿太久
  - 物体掉落后继续（应重置）
  - 夹爪一直闭合不松开
```

### 3.5 数据采集完成后

录制的数据保存在 `~/.cache/huggingface/lerobot/` 下：

```
~/.cache/huggingface/lerobot/your/your-dataset-name/
├── meta/
│   ├── info.json          ← 特征定义（action/state 维度、相机路数）
│   ├── stats.json         ← 归一化统计量（均值/标准差）
│   └── episodes.jsonl     ← 每个 episode 的元数据
├── data/
│   └── chunk-000/
│       └── episode_*.parquet  ← 动作和关节状态的时间序列
└── videos/
    └── chunk-000/
        ├── top/
        │   └── episode_*.mp4   ← 顶部摄像头
        ├── left_wrist/
        │   └── episode_*.mp4   ← 左手腕摄像头
        └── right_wrist/
            └── episode_*.mp4   ← 右手腕摄像头
```

#### 验证数据集

采集完成后可以用以下命令快速检查数据是否正确：

```bash
python3 << 'EOF'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

cache_root = os.path.expanduser("~/.cache/huggingface/lerobot")
ds = LeRobotDataset("your/your-dataset-name", root=cache_root)

print(f"演示数: {ds.num_episodes}")
print(f"总帧数: {len(ds)}")
print(f"FPS: {ds.fps}")

frame = ds[0]
print(f"动作: {frame['action'].shape[-1]} 维")
print(f"状态: {frame['observation.state'].shape[-1]} 维")

image_keys = [k for k in frame.keys() if "image" in k]
print(f"摄像头: {len(image_keys)} 路 → {image_keys}")
EOF
```

**预期输出（XLeRobot 录制，2轮版）：**
```
演示数: 50
总帧数: 45000
FPS: 30
动作: 16 维
状态: 16 维
摄像头: 3 路 → ['observation.images.top', 'observation.images.left_wrist', 'observation.images.right_wrist']
```

#### 上传到云服务器（可选）

如果使用云服务器训练，需要把数据集传上去：

```bash
# 方式 A：rsync（推荐）
rsync -avz --progress \
  ~/.cache/huggingface/lerobot/your/your-dataset-name/ \
  root@<你的云服务器IP>:/data/datasets/your-dataset-name/

# 方式 B：HuggingFace Hub
huggingface-cli login
python -m lerobot.datasets.push_to_hub \
  --repo-id=your/your-dataset-name \
  --root=~/.cache/huggingface/lerobot/
```

---

## 第四章：模型训练与推理

训练和推理步骤取决于你使用的 VLA 模型，请参见对应目录的 `GUIDE.md`：

| 模型 | 手册 |
|------|------|
| X-VLA | `xvla_deploy/GUIDE.md` |
| SmolVLA | `smolvla_deploy/GUIDE.md` |
| 其他 | 后续新增 `xxx_deploy/GUIDE.md` |

---

## 附录A：通用排错

### A.1 数据集相关

#### "Dataset not found"

```bash
# 原因: 数据集路径不对
# 解决:
export LEROBOT_CACHE=/data/datasets
# 或检查 ~/.cache/huggingface/lerobot/ 下是否有数据
```

### A.2 推理相关

#### "Connection refused"

```bash
# 原因: 服务没启动或端口不对
# 解决:
ps aux | grep server.py      # 确认服务在运行
netstat -tlnp | grep 8000    # 确认端口监听
# 确认安全组开放了端口
```

#### "Timeout"

```bash
# 原因: 推理请求超过 10 秒
# 解决:
nvidia-smi                   # 检查 GPU 是否被占用
# 减少去噪步数（如果模型支持）
ping <云服务器IP>            # 检查网络延迟
```

### A.3 机器人相关

#### 服务正常但机器人不动

```bash
# 检查:
# 1. 客户端是否连接到正确的服务器 URL
# 2. 服务器返回的动作值是否合理（不是全零）
# 3. 机器人的 send_action 是否正常
python /home/zach/XLeRobot/software/examples/2_dual_so100_keyboard_ee_control.py
```

#### 机器人动作抖动

```bash
# 原因: 推理动作不平滑
# 解决:
# 1. 增大控制频率
python client.py --control-freq 50
# 2. 加动作平滑
# 修改 client.py, 在 send_action 之前加:
#   current_action = current_action * 0.7 + new_action * 0.3
```

#### 数据传输慢

```bash
# 原因: 摄像头图像太大
# 解决: 减小图像分辨率
# 修改 client.py 中的相机配置:
cameras={
    "cam_top": OpenCVCameraConfig(..., width=320, height=240),
}
```

---

## 附录B：概念参考

### B.1 摄像头配置说明

三路摄像头的典型位置：

```
                    ┌─────────────────┐
                    │  top            │ ← 头顶朝下，拍到桌面全貌
                    │  (全局视角)     │
                    └─────────────────┘
                          │
                    ┌─────┴─────┐
                    │  桌  面    │
                    └───────────┘
               ┌────────┴────────┐
               │                 │
        ┌──────┴──────┐  ┌──────┴──────┐
        │ left_wrist  │  │ right_wrist │
        │ (左腕,看左边)│  │ (右腕,看右边)│
        └─────────────┘  └─────────────┘
```

> 使用 `bi_so_follower` 时，观察键名自动加上 `left_`/`right_` 前缀。
> 使用 XLerobot 类时，键名为 `top`、`left_wrist`、`right_wrist`。

### B.2 LeRobot 数据集格式

一个 frame（一帧数据）包含：

| 字段 | 形状 | 说明 |
|------|------|------|
| `action` | (D,) | D 维浮点数组（关节目标位置） |
| `observation.state` | (D,) | D 维浮点数组（当前关节位置） |
| `observation.images.*` | (H,W,3) | uint8 RGB 图像 |
| `task` | — | 字符串任务描述 |
| `episode_index` | — | 所属演示编号 |
| `frame_index` | — | 演示内第几帧 |
| `timestamp` | — | 时间戳 |

动作维度取决于机器人版本：

| 版本 | 动作/状态维度 | 分布 |
|------|-------------|------|
| 2轮差速版 | **16-DOF** | 12手臂 + 2头部 + 2底盘 |
| 3轮全向版 | **17-DOF** | 12手臂 + 2头部 + 3底盘 |
| 双臂标准版 | **12-DOF** | 6左臂 + 6右臂 |
