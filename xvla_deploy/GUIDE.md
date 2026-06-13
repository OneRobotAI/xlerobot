# X-VLA × XLeRobot 双臂叠衣服 — 完整实操指南

> 版本: 1.2 | 最后更新: 2026-05-29
> 适用: XLeRobot (双臂 SO-100) + X-VLA 0.9B
>
> **v1.2 变更**：云端环境改用 conda 方式，增加 SSH 隧道连接方式。

---

## 目录

- [第一章：背景与选型](#第一章背景与选型)
- [第二章：环境搭建](#第二章环境搭建)
- [第三章：数据采集](#第三章数据采集)
- [第四章：数据集检查与上传](#第四章数据集检查与上传)
- [第五章：模型训练](#第五章模型训练)
- [第六章：推理服务部署](#第六章推理服务部署)
- [第七章：机器人控制客户端](#第七章机器人控制客户端)
- [第八章：效果评估与迭代](#第八章效果评估与迭代)
- [附录A：常见错误解决](#附录a常见错误解决)
- [附录B：概念详解](#附录b概念详解)
- [附录C：命令速查表](#附录c命令速查表)

---

## 第一章：背景与选型

### 1.1 你面对的问题

用双臂机器人叠衣服是一个典型的**柔性物体操作**问题。不同于抓取刚性物体（杯子、方块），布料的特点是：

- **高自由度**：布料有无穷多种形状
- **自遮挡**：折叠过程中部分布料被遮挡
- **摩擦变化**：不同布料表面摩擦系数不同
- **需要双手机调**：一只手展开，另一只手折叠

### 1.2 为什么 X-VLA 是最优解

X-VLA 是一个 **Soft-Prompted VLA** 模型，它的核心创新是：

```
传统 VLA: 
  一个模型只能控制一种机器人，换机器人需要重新训练全部参数

X-VLA:
  同一个基座模型 + 不同的 soft prompt = 不同的机器人
  
  类比:
  基座模型 = 一个人的大脑 (通用智能)
  Soft prompt = 这个人的眼镜/助听器 (适配特定硬件)
  换机器人只需换"眼镜"，不用换"大脑"
```

这对你的意义：

| 你的需求 | X-VLA 如何满足 |
|---------|---------------|
| 资源有限 | 0.9B 参数，LoRA 仅 9M 参数可训练 |
| 叠衣服 | 有专用 checkpoint `lerobot/xvla-folding`，100% 成功率 |
| 云端推理 | 原生 Server-Client 架构，一行代码启动服务 |
| 5/6-DOF兼容 | `action_mode=auto` 自动检测动作维度 |

关键数据对比：

```
                        ACT    DP    SmolVLA   X-VLA    LingBot-V4B   LingBot-VA
参数量                  80M   100M   2.25B     0.9B     4.2B          ~23GB文件
单卡训练                ✅     ✅     ✅       ✅       ✅(LoRA)      ❌需8卡
叠衣服成功率(500演示)   40%    62%    —        100%*    —             70%(叠裤子)
Server-Client           ❌     ❌    ❌        ✅       ❌            ✅
LeRobot原生             ✅     ✅    ✅        ✅       ❌            ❌

* lerobot/xvla-folding 在 Soft-FOLD 数据集上
```

---

## 第二章：环境搭建

### 2.1 你需要准备的硬件

#### 本地（数据采集用）

```
电脑要求:
├── 操作系统: Ubuntu 20.04 / 22.04
├── 内存: 16GB+
├── 存储: 100GB+ 空闲（数据存储）
├── USB 端口: 至少 5 个空闲 USB-A
│   ├── 左主臂 (Leader) × 1
│   ├── 右主臂 (Leader) × 1
│   ├── 左从臂 (Follower) × 1  (XLeRobot 本体)
│   ├── 右从臂 (Follower) × 1  (XLeRobot 本体)
│   └── 摄像头 × 3
└── 已有: XLeRobot (你的现有硬件)
    需要额外购买: 2个 SO-100 主臂
```

#### 云端（训练 + 推理用）

```
服务器要求:
├── GPU: NVIDIA GPU, 24GB+ VRAM (RTX 4090 起步)
├── CUDA: 12.1+
├── 系统: Ubuntu 22.04
├── 存储: 50GB+
├── 网络: 公网 IP（客户端需要访问）
└── 推荐: AutoDL / 百度百舸 / 阿里云 GPU 实例
```

### 2.2 本地环境安装

#### 步骤 1：确认 Python 版本

```bash
python3 --version
# 期望输出: Python 3.10.x 或 3.11.x
# 如果版本不对，安装:
# sudo apt install python3.10 python3.10-venv
```

#### 步骤 2：安装/更新 LeRobot 及 X-VLA

```bash
# 使用你的 XLeRobot 现有环境
# 先检查现有 LeRobot 版本
pip show lerobot
# 如果已安装，确保包含 X-VLA 支持:
pip install --upgrade lerobot[xvla]
```

#### 步骤 3：验证安装

```bash
python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
print("X-VLA import: ✅")

# XLeRobot 特有（需要 XLeRobot fork，pip install 上游版本没有）
try:
    from lerobot.robots.xlerobot import XLerobotConfig, XLerobot
    print("XLeRobot import: ✅ (XLerobot with mobile base + head)")
except ImportError:
    from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
    print("XLeRobot import: ✅ (using upstream bi_so_follower, arms only)")
EOF
```

**预期输出：**
```
PyTorch: 2.5.1+cu121
CUDA available: False  (本地电脑通常没有 GPU，这是正常的)
X-VLA import: ✅
XLeRobot import: ✅
```

#### 步骤 4：确认 XLeRobot 连接

```bash
# 确认 USB 设备
ls -la /dev/ttyACM*

# 在 XLeRobot 的 examples 目录下测试连接
cd /home/zach/XLeRobot/software/examples
# 用你的键盘控制脚本测试双臂是否能正常控制
python 2_dual_so100_keyboard_ee_control.py
# 按 x 退出
```

#### 步骤 5：设置 XLerobot 软链接

```bash
# client.py 需要导入 XLerobot 类来控制 XLeRobot 硬件
# 软链接让 Python 能找到 XLeRobot 项目中的机器人代码
bash /home/zach/XLeRobot/xvla_deploy/setup_local.sh
```

**预期输出：**
```
✅ 已创建软链接: .../lerobot/robots/xlerobot → .../XLeRobot/software/src/robots/xlerobot
✅ XLerobot 导入成功
```

> 这个软链接只在本地有效，不会进 Git。其他人克隆你的仓库后也需要运行一次。

### 2.3 云端环境安装

云端只需要 PyTorch + LeRobot + X-VLA，**不需要 XLerobot 类**（推理服务只用 `XVLAPolicy`）。如果你已经配好了 conda 环境，可以直接用。

#### 步骤 1：确认已有环境

```bash
# SSH 登录云服务器
ssh -p <端口号> featurize@<云服务器IP>

# 检查 GPU
nvidia-smi

# 激活已有 conda 环境
conda activate lerobot

# 验证关键依赖
python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
print("X-VLA import: ✅")

from lerobot.robots.bi_so_follower import BiSOFollower
print("bi_so_follower import: ✅")
EOF
```

**预期输出：**
```
PyTorch: 2.11.0+cu130, CUDA: True
X-VLA import: ✅
bi_so_follower import: ✅
```

> 如果你的云服务器还没有 LeRobot 环境，可以新建一个：
> ```bash
> conda create -n lerobot python=3.12
> conda activate lerobot
> pip install 'lerobot[xvla]'
> ```

#### 步骤 2：上传部署包

```bash
# 在本地电脑上运行
cd /home/zach/XLeRobot
scp -P <端口号> -r xvla_deploy/ featurize@<云服务器IP>:~/
```

#### 步骤 3：启动推理服务

```bash
# 在云服务器上运行
conda activate lerobot
cd ~/xvla_deploy
python server.py --model-path lerobot/xvla-folding --port 8000 --host 0.0.0.0
```

**启动后输出：**
```
[INFO] Loading X-VLA model from: lerobot/xvla-folding
[INFO] ✅ Loaded via LeRobot XVLAPolicy on cuda
[INFO] Model loaded in 2.3s
[INFO] Starting X-VLA server...
[INFO]   Listen: 0.0.0.0:8000
[INFO]   API: POST http://0.0.0.0:8000/act
[INFO] Application startup complete.
```

> **云服务器不需要 `setup_local.sh`**。`setup_local.sh` 只在本地方需要，因为 `client.py` 要控制 XLeRobot 硬件。
> 云端只跑推理服务，只用 `XVLAPolicy`，上游 pip 包自带。`deploy.sh` 也是可选的——如果你已经有 conda 环境，直接用就行。

#### 步骤 4：连接推理服务（SSH 隧道，推荐）

推理服务跑在云服务器的 `localhost:8000`，多数云平台不会直接开放这个端口。用 SSH 隧道把本地方问转发过去，无需配置防火墙。

```bash
# 在本地电脑上运行（后台运行，不占终端）
ssh -N -f -L 8000:localhost:8000 featurize@<云服务器IP> -p <SSH端口>
```

> `-N` 不执行远程命令（纯端口转发），`-f` 后台运行。如果想关闭隧道：
> ```bash
> # 找到 ssh 进程并杀掉
> ps aux | grep "ssh -N -f -L 8000" | grep -v grep
> kill <进程ID>
> ```

隧道建立后，另开一个本地终端测试：

```bash
curl http://localhost:8000/health
```

**预期输出：**
```json
{"status":"ok","model":"lerobot/xvla-folding","device":"cuda"}
```

之后 `client.py` 也用 `localhost` 连接：

```bash
python client.py --server-url http://localhost:8000
```

> 隧道建立后一直保持到进程关闭。用 `ps aux | grep ssh` 找到进程后 `kill` 即可关闭。
>
> 如果云平台允许开放端口（如 AutoDL 的自定义服务、阿里云安全组），也可直接开放 8000 端口，
> 然后用公网 IP 访问：`curl http://<云服务器IP>:8000/health`。

---

## 第三章：数据采集

### 3.1 硬件连接

#### 连接示意图

你的电脑有多个 USB 口，四个机械臂分别连接到不同的 USB 口。

```bash
# 找出每个 USB 口对应哪个设备
# 先不连接任何机械臂，运行:
python -m lerobot.scripts.find_port
# 输出: []

# 然后只连接左主臂，运行:
python -m lerobot.scripts.find_port
# 输出: ['/dev/ttyACM0']

# 逐个连接并记录端口
# 最终你应该得到:
# 左主臂: /dev/ttyACM2
# 右主臂: /dev/ttyACM3
# 左从臂: /dev/ttyACM0
# 右从臂: /dev/ttyACM1
```

> **注意**：端口号每次插拔可能变化。建议给 USB 设备绑定固定名称，或者在每次使用前重新确认。

#### 固定 USB 端口（可选但推荐）

```bash
# 查看每个设备的序列号
udevadm info --name=/dev/ttyACM0 | grep ID_SERIAL

# 创建 udev 规则绑定固定名称
sudo tee /etc/udev/rules.d/99-xlerobot.rules << 'EOF'
SUBSYSTEM=="tty", ATTRS{serial}=="<左主臂序列号>", SYMLINK+="xlerobot_left_leader"
SUBSYSTEM=="tty", ATTRS{serial}=="<右主臂序列号>", SYMLINK+="xlerobot_right_leader"
SUBSYSTEM=="tty", ATTRS{serial}=="<左从臂序列号>", SYMLINK+="xlerobot_left_follower"
SUBSYSTEM=="tty", ATTRS{serial}=="<右从臂序列号>", SYMLINK+="xlerobot_right_follower"
EOF
sudo udevadm control --reload-rules
```

### 3.2 校准机械臂

#### 什么是校准？

校准的目的是确保**主臂和从臂在相同物理位置时，输出相同的电机位置值**。因为每个电机的零点和刻度有微小差异，必须校准。

#### 校准主臂

```bash
# 左主臂
lerobot-calibrate \
  --teleop.type=so100_leader \
  --teleop.port=/dev/ttyACM2 \
  --teleop.id=left_leader

# 按照提示:
# 1. 将手臂移动到零位（水平伸直，朝向正前方）
# 2. 按回车确认
# 3. 手臂会自动检测各关节范围
# 4. 校准数据保存到 ~/.cache/lerobot/calibration/left_leader.yaml
```

```bash
# 右主臂
lerobot-calibrate \
  --teleop.type=so100_leader \
  --teleop.port=/dev/ttyACM3 \
  --teleop.id=right_leader
```

#### 校准从臂

XLerobot 的从臂校准用专用脚本，覆盖全部 17 个电机：

```bash
# 需要先运行 setup_local.sh（确保 XLerobot 可导入）
bash /home/zach/XLeRobot/xvla_deploy/setup_local.sh

# 校准全部电机（左臂6 + 右臂6 + 头部2 + 底盘3）
python /home/zach/XLeRobot/xvla_deploy/calibrate_xlerobot.py \
  --port1 /dev/ttyACM0 \
  --port2 /dev/ttyACM1
```

**校准过程中：**
1. 关掉 bus1 扭矩 → 手动将**左臂 + 头部**所有关节移到中间位置 → 按回车
2. 依次活动**左臂 + 头部**每个关节到极限 → 按回车停止
3. 关掉 bus2 扭矩 → 手动将**右臂**关节移到中间位置 → 按回车
4. 依次活动**右臂**关节到极限 → 按回车停止
5. 校准数据自动保存

> `lerobot-calibrate` 的 `--robot.type=bi_so_follower` 只校准 12 个手臂电机，
> XLerobot 校准脚本会覆盖全部 17 个电机（含头部 2 个、底盘 3 个）。

#### 验证校准

```bash
# 快速测试：移动主臂，看从臂是否跟随
python -m lerobot.scripts.teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/ttyACM0 \
  --robot.right_arm_config.port=/dev/ttyACM1 \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/ttyACM2 \
  --teleop.right_arm_config.port=/dev/ttyACM3 \
  --display_data=true
```

如果从臂能平滑跟随主臂运动，校准成功。

### 3.3 运行数据采集脚本

#### 基本用法

```bash
cd /home/zach/XLeRobot/xvla_deploy

python collect_data.py \
  --num-episodes 50 \
  --task "fold the towel on the table"
```

#### 参数说明

```
--num-episodes 50      录制 50 个演示（每个约 30 秒）
--task "fold..."       任务描述，会存入数据集
--repo-id your/...     数据集在 HuggingFace 上的 ID
--episode-time 30      每个 episode 最长时长（秒）
--fps 30               录制帧率
--dry-run              试运行，不实际录制
--find-ports           自动检测 USB 端口
```

#### 录制过程

```
启动后你会看到类似输出:

============================================================
  X-VLA Data Collection for Cloth Folding
============================================================
  Task:            fold the towel on the table
  Episodes:        50
  Duration/ep:     30s
  FPS:             30
  Output repo:     your/xlerobot-cloth-fold
  Leader arms:     /dev/ttyACM2, /dev/ttyACM3
  Follower arms:   /dev/ttyACM0, /dev/ttyACM1
============================================================

然后会弹出一个窗口显示摄像头画面。
在窗口中按 R 开始录制。

录制过程中:
- 你双手操控两个主臂
- 从臂跟随运动
- 所有数据（关节角度 + 摄像头画面）被记录
- 当前 episode 编号显示在窗口上

一个 episode 完成后:
- 按 R 开始下一个
- 或按 Q 退出

完成 50 个 episode 后自动结束。
```

#### 录制技巧

**新手建议：**
```
第 1-10 个:  简单对折（毛巾平铺 → 对折一次）
第 11-20 个: 改变初始位置（毛巾放在桌面不同位置）
第 21-30 个: 改变折法（对折 → 再对折）
第 31-40 个: 换不同布料（薄毛巾 → 厚毛巾）
第 41-50 个: 混合场景（随机折法 + 随机位置）
```

**好的演示 vs 差的演示：**

```
✅ 好的:
  - 动作流畅不卡顿
  - 布料完全展开再折叠
  - 夹爪开合清晰

❌ 差的:
  - 中途停顿太久
  - 布料掉落后继续（应重置）
  - 夹爪一直闭合不松开
```

### 3.4 数据采集完成后

录制的数据保存在：

```bash
~/.cache/huggingface/lerobot/your/xlerobot-cloth-fold/

# 结构如下:
meta/
  info.json          ← 数据集元信息（特征定义、帧率等）
  stats.json         ← 归一化统计量
  episodes.jsonl     ← 每个 episode 的元数据
data/
  chunk-000/
    episode_*.parquet  ← 动作和关节状态的时间序列数据
videos/
  chunk-000/
    left_top/          ← 顶部摄像头（来自左臂相机配置）
    left_wrist/        ← 左手腕摄像头
    right_wrist/       ← 右手腕摄像头（来自右臂相机配置）
      episode_*.mp4   ← 摄像头视频
```

---

## 第四章：数据集检查与上传

### 4.1 本地检查数据集

```bash
# 用 Python 检查数据集内容
python3 << 'EOF'
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("your/xlerobot-cloth-fold")

print(f"{'='*50}")
print(f"数据集统计")
print(f"{'='*50}")
print(f"Episodes (演示数):   {ds.num_episodes}")
print(f"Total frames (总帧数): {len(ds)}")
print(f"FPS:                {ds.fps}")
print(f"Duration (总时长):    {len(ds)/ds.fps/60:.1f} 分钟")
print()

# 检查第一帧
frame = ds[0]
print(f"{'='*50}")
print(f"特征检查")
print(f"{'='*50}")
for key, value in frame.items():
    if hasattr(value, 'shape'):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  {key}: {value}")

print()

# 检查动作维度
action = frame["action"]
print(f"动作维度: {action.shape[-1]}")
print(f"  前6个 (左臂): {action[:6]}")
print(f"  后6个 (右臂): {action[6:]}")

state = frame["observation.state"]
print(f"状态维度: {state.shape[-1]}")

# 检查摄像头
image_keys = [k for k in frame.keys() if "image" in k]
print(f"摄像头: {image_keys}")
for k in image_keys:
    print(f"  {k}: {frame[k].shape}")
EOF
```

**预期输出（bi_so_follower 录制）**：
```
动作维度: 12
状态维度: 12
摄像头: ['observation.images.left_top', 'observation.images.left_wrist', 'observation.images.right_wrist']
```

**如果输出中 action 维度是 12，说明数据格式正确。**

### 4.2 上传到云服务器

#### 方式 A：通过 rsync（推荐）

```bash
# 本地电脑运行
# 先确认数据集路径
ls ~/.cache/huggingface/lerobot/your/xlerobot-cloth-fold/

# 传输到云服务器
rsync -avz --progress \
  ~/.cache/huggingface/lerobot/your/xlerobot-cloth-fold/ \
  root@<你的云服务器IP>:/data/datasets/xlerobot-cloth-fold/
```

#### 方式 B：通过 HuggingFace Hub

```bash
# 如果你能访问 HuggingFace Hub
# 1. 登录
huggingface-cli login

# 2. 推送数据集
python -m lerobot.datasets.push_to_hub \
  --repo-id=your/xlerobot-cloth-fold \
  --root=~/.cache/huggingface/lerobot/
```

#### 方式 C：通过网盘中转

```bash
# 如果 rsync 太慢（跨国传输），先用网盘
# 本地: 压缩后上传到百度网盘
tar -czf cloth_fold_dataset.tar.gz \
  -C ~/.cache/huggingface/lerobot/ your/xlerobot-cloth-fold/

# 云服务器: 下载并解压
tar -xzf cloth_fold_dataset.tar.gz -C /data/datasets/
```

### 4.3 云端验证

在云服务器上确认数据集可用：

```bash
# 设置数据集路径（如果不是默认位置）
export LEROBOT_CACHE=/data/datasets

# 验证
python3 << 'EOF'
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("your/xlerobot-cloth-fold")
print(f"Episodes: {ds.num_episodes}")
print(f"Frames: {len(ds)}")
print(f"Action dim: {ds[0]['action'].shape}")
print("✅ 数据集可用")
EOF
```

---

## 第五章：模型训练

### 5.1 训练原理详解

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
  │ 方法: 新的 Soft Prompt (9M参数) + 可选的  │
  │        微调 policy transformer            │
  │ 数据: 你的 50-200 个叠衣服演示            │
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

### 5.2 轻量模式训练（推荐起步）

#### 执行训练

```bash
# 在云服务器上
cd ~/xvla_deploy
conda activate lerobot

# 确保数据集路径
export LEROBOT_CACHE=/data/datasets

# 开始轻量训练
bash train.sh --light
```

#### 训练过程日志解读

```
[INFO] Training configuration:
  Dataset:    your/xlerobot-cloth-fold
  Base model: lerobot/xvla-folding
  Steps:      5000
  Light:      true

[INFO] Loading dataset...
Found 50 episodes, 45000 frames
✅ Dataset loaded

[INFO] Loading model...
Loading checkpoint from lerobot/xvla-folding
...
✅ Model loaded, 900M parameters
  Trainable: 9,000,000 (soft prompts only)

Step 1/5000 | loss: 0.8421 | lr: 1e-3 | VRAM: 8.2GB | 3.2 it/s
Step 100/5000 | loss: 0.5213 | lr: 1e-3 | VRAM: 8.3GB | 3.1 it/s
Step 500/5000 | loss: 0.3125 | lr: 1e-3 | VRAM: 8.3GB | 3.1 it/s
Step 1000/5000 | loss: 0.2156 | lr: 1e-3 | VRAM: 8.4GB | 3.1 it/s
...

loss 解读:
  - 初始 ~0.8: 模型刚开始，预测不准确
  - 下降到 ~0.2: 模型开始理解叠衣服模式
  - 稳定在 ~0.1: 训练收敛
  - 如果 loss 不下降: 检查数据集是否有问题
```

#### 训练完成

```
✅ Training complete!
  Model saved to: ./outputs/xvla_xlerobot_fold
  Checkpoints:    ./outputs/xvla_xlerobot_fold/checkpoints
  
  best/  ← 用于推理部署
  last/  ← 可用于继续训练
```

### 5.3 全量模式训练（效果更好）

```bash
bash train.sh
```

与轻量模式的区别：

```
轻量模式:               全量模式:
───────────────────────────────────────────
VRAM: ~9GB              VRAM: ~16GB
时间: ~1小时             时间: ~4小时
训练: soft prompt only   训练: soft prompt + transformer
效果: 良好                效果: 更好
适合: 快速验证            适合: 最终训练
```

### 5.4 监控训练

#### TensorBoard

```bash
# 在云服务器上
conda activate lerobot
tensorboard --logdir=./outputs/xvla_xlerobot_fold/logs --port=6006

# 然后在本地浏览器访问
# http://<云服务器IP>:6006
```

**关注的关键指标：**

```
train/loss       应该持续下降          → 模型在学
train/grad_norm  应该在 0.1-10 之间    → 梯度稳定
train/lr         按计划衰减           → 学习率正常
val/loss         验证集上也在下降       → 没有过拟合
```

#### 提前停止

如果看到以下情况可以提前停止（Ctrl+C）：

```
loss 降到 0.05 以下：  基本已经学到了
loss 连续 500 步不降：  可能已经收敛，不需要继续
loss 开始上升：         过拟合了，应该早停
```

### 5.5 从头训练（不使用叠衣服 checkpoint）

如果你想从通用基座开始训练，而不是从叠衣服专用 checkpoint 开始：

```bash
bash train.sh --model-path lerobot/xvla-base
```

这会从 X-VLA 通用基座开始训练（没有叠衣服先验知识），需要更多数据（~200 个演示）。

---

## 第六章：推理服务部署

### 6.1 启动推理服务

#### 用你训练的模型

```bash
# 在云服务器上
cd ~/xvla_deploy
conda activate lerobot

# 启动服务，加载你微调好的模型
python server.py \
  --model-path ./outputs/xvla_xlerobot_fold/checkpoints/best \
  --port 8000 \
  --host 0.0.0.0
```

#### 或用预训练 checkpoint（不训练也能试）

```bash
# 直接使用 X-VLA 的叠衣服预训练模型
python server.py \
  --model-path lerobot/xvla-folding \
  --port 8000
```

**启动后输出：**
```
[INFO] Loading X-VLA model from: lerobot/xvla-folding
[INFO] ✅ Loaded via LeRobot XVLAPolicy on cuda
[INFO] Model loaded in 2.3s

[INFO] Starting X-VLA server...
[INFO]   Listen: 0.0.0.0:8000
[INFO]   API: POST http://0.0.0.0:8000/act
[INFO] Application startup complete.
```

### 6.2 测试服务

#### 健康检查

```bash
# 在云服务器或本地
curl http://<云服务器IP>:8000/health

# 期望:
{"status":"ok","model":"lerobot/xvla-folding","device":"cuda"}
```

#### 推理测试

```bash
# 在本地电脑或云服务器
python3 << 'EOF'
import requests
import json
import numpy as np

SERVER = "http://<云服务器IP>:8000/act"

# 模拟一个观测（全零数据）
payload = {
    "proprio": json.dumps(np.zeros(12).tolist()),
    "language_instruction": "fold the towel",
    "image0": json.dumps(np.zeros((256, 256, 3), dtype=np.uint8).tolist()),
    "image1": json.dumps(np.zeros((256, 256, 3), dtype=np.uint8).tolist()),
    "image2": json.dumps(np.zeros((256, 256, 3), dtype=np.uint8).tolist()),
    "domain_id": 0,
    "steps": 10,
}

resp = requests.post(SERVER, json=payload, timeout=30)
data = resp.json()

actions = np.array(data["action"])
print(f"✅ 推理成功!")
print(f"  动作序列形状: {actions.shape}")       # 应该是 (chunk_size, 12)
print(f"  推理耗时: {data['inference_time_ms']}ms")
print(f"  动作序列长度: {actions.shape[0]}步")
print(f"  动作维度: {actions.shape[1]}维")
print(f"  前3步动作值:\n{actions[:3]}")
EOF
```

**期望输出：**
```
✅ 推理成功!
  动作序列形状: (32, 12)
  推理耗时: 45.2ms
  动作序列长度: 32步
  动作维度: 12维
  前3步动作值:
  [[...], [...], [...]]
```

### 6.3 设置安全组

对于不同的云平台，开放端口的操作：

#### AutoDL

```bash
# AutoDL 控制台 → 容器实例 → 自定义服务
# 添加: 端口 8000，TCP
```

#### 阿里云

```bash
# 阿里云控制台 → 安全组 → 配置规则
# 入方向 → 添加安全组规则
#   端口范围: 8000/8000
#   授权对象: 0.0.0.0/0 (全通) 或你的本地 IP
```

#### 百度百舸

```bash
# 百度云控制台 → 安全组 → 添加规则
# 入方向: TCP 8000
```

### 6.4 持久化运行

```bash
# 用 nohup 保持后台运行
nohup python server.py \
  --model-path ./outputs/xvla_xlerobot_fold/checkpoints/best \
  --port 8000 \
  > server.log 2>&1 &

# 查看日志
tail -f server.log

# 停止服务
kill %1
```

或者用 systemd 做成服务：

```bash
# 创建服务文件
sudo tee /etc/systemd/system/xvla-server.service << 'EOF'
[Unit]
Description=X-VLA Inference Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/xvla_deploy
ExecStart=/home/featurize/miniforge3/envs/lerobot/bin/python server.py \
  --model-path /root/xvla_deploy/outputs/xvla_xlerobot_fold/checkpoints/best \
  --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 启动服务
sudo systemctl daemon-reload
sudo systemctl enable xvla-server
sudo systemctl start xvla-server
sudo systemctl status xvla-server
```

---

## 第七章：机器人控制客户端

### 7.1 client.py 代码详解

```python
# client.py
#
# 这个文件的作用:
# 1. 连接 XLeRobot 硬件
# 2. 获取观测（关节角度 + 摄像头图像）
# 3. 发送到云端推理服务
# 4. 接收预测的动作
# 5. 控制机器人执行动作

# 核心数据结构:
#
# 观测 (observation):
#   left_arm_shoulder_pan.pos    左臂关节1 (肩部旋转)
#   left_arm_shoulder_lift.pos   左臂关节2 (肩部抬升)
#   left_arm_elbow_flex.pos      左臂关节3 (肘部)
#   left_arm_wrist_flex.pos      左臂关节4 (手腕俯仰)
#   left_arm_wrist_roll.pos      左臂关节5 (手腕旋转)
#   left_arm_gripper.pos         左臂夹爪
#   right_arm_...                右臂 (同上 6 个)
#   left_top                       顶部摄像头图像 (640,480,3) — 来自左臂相机
#   left_wrist                     左手腕摄像头 (640,480,3) — 来自左臂相机
#   right_wrist                    右手腕摄像头 (640,480,3) — 来自右臂相机
#   (bi_so_follower 会给每只手臂的相机加上 left_/right_ 前缀)
#
# 动作 (action):
#   12维数组 [左臂6 + 右臂6], 顺序同观测

# 控制流程:
# while True:
#   obs = robot.get_observation()           # 获取观测
#   proprio = extract_joints(obs)            # 提取关节位置
#   images = extract_images(obs)             # 提取摄像头图像
#   payload = build_payload(proprio, images) # 组装请求
#   response = requests.post(url, payload)   # 发送到云端
#   action_chunk = response["action"]        # 接收动作序列
#   for action in action_chunk:              # 逐个执行
#     robot.send_action(action)
#     time.sleep(1/30)
```

### 7.2 启动客户端

```bash
cd /home/zach/XLeRobot/xvla_deploy

python client.py \
  --server-url http://<你的云服务器外网IP>:8000 \
  --task "fold the towel on the table"
```

### 7.3 启动后的输出

```
[INFO] Connecting XLeRobot (port1=/dev/ttyACM0, port2=/dev/ttyACM1)...
[INFO] ✅ XLeRobot connected
[INFO] Available cameras: ['cam_top', 'cam_left_wrist', 'cam_right_wrist']
[INFO] Starting control loop (freq=30.0Hz)
[INFO] Server: http://xxx.xxx.xxx.xxx:8000/act
[INFO] Task: 'fold the towel on the table'
[INFO] Press Ctrl+C to stop

[INFO] Step 0: Requesting new action chunk from cloud...
[INFO]   Received chunk of 32 actions
[INFO] Step 50: chunk 18/32
[INFO] Step 100: chunk 4/32       ← 自动请求了下一个chunk
[INFO] Step 150: chunk 20/32
...
```

### 7.4 动作分块机制说明

```
云端推理: +-------+      +-------+      +-------+
          |chunk 1|─────→|chunk 2|─────→|chunk 3|
          |32步   |      |32步   |      |32步   |
          +-------+      +-------+      +-------+
                ↓              ↓              ↓
机器人执行:  [1][2]..[32]   [1][2]..[32]   [1][2]..
                ↑              ↑
            执行chunk1时    执行chunk2时
            后台请求chunk2  后台请求chunk3

好处:
  1. 网络延迟被隐藏（执行的同时在请求下一块）
  2. 动作更平滑（32步连续动作，不等待推理）
  3. 容错（一块失败，当前块继续执行完）
```

### 7.5 安全停止

```bash
# 在客户端终端按 Ctrl+C
[INFO] 🛑 Interrupted by user
[INFO] XLeRobot disconnected
[INFO] Client stopped

# 机器人会安全停止（不断电，只是停止发送指令）
```

---

## 第八章：效果评估与迭代

### 8.1 首次测试流程

```
完整测试流程:

1. 准备工作
   □ 桌面放好叠好的毛巾/布料
   □ 开启云端推理服务
   □ 连接 XLeRobot

2. 运行客户端
   □ python client.py --server-url ...
   □ 观察机器人是否开始动作

3. 评估指标
   □ 成功率: 10 次尝试中成功几次
   □ 完成时间: 从开始到完成需要多久
   □ 动作平滑度: 动作是否流畅
   □ 失败模式: 哪里出问题（抓不住？叠不好？）

4. 记录失败模式
   □ 拍照/录像记录下来
   □ 分析是数据问题还是策略问题
```

### 8.2 迭代改进

```
效果不好? → 分析原因:

┌─ 夹爪没有正确抓住布料?
│   → 增加夹爪开合的数据多样性
│   → 录制更多抓取不同位置布料的演示
│
├─ 叠到一半卡住?
│   → 检查是否有障碍物
│   → 增加从不同角度叠的演示
│
├─ 动作抖动?
│   → 增加控制频率 (--control-freq 50)
│   → 检查机械臂是否松动
│
├─ 完全不会动?
│   → 检查网络连接
│   → 检查推理服务是否正常运行
│   → 检查模型是否正确加载
│
└─ 效果还行但不够好?
    → 增加数据量
    → 全量微调
    → 增加数据多样性
```

### 8.3 数据扩增策略

如果 50 个演示不够，可以通过以下方式扩增：

```
回合 1: 50 个演示
  ├── 效果还行? → 部署使用
  └── 不好? → 继续

回合 2: +50 个演示 (共100)
  ├── 新增: 布料不同摆放位置
  ├── 新增: 不同折法
  └── 训练 → 测试

回合 3: +50 个演示 (共150)
  ├── 新增: 不同布料
  ├── 新增: 不同光照条件
  └── 训练 → 测试
```

---

## 附录A：常见错误解决

### A.1 训练时报错

#### "CUDA out of memory"

```bash
# 原因: VRAM 不够
# 解决: 
# 1. 减小 batch size
bash train.sh --light --batch-size 2

# 2. 使用梯度累积（不增加VRAM但等效大batch）
# 在 train.sh 的 ARGS 中加:
--optimizer.gradient_accumulation_steps=4

# 3. 启用梯度检查点（以计算换显存）
# 在 train.sh 的 ARGS 中加:
--policy.gradient_checkpointing=true
```

#### "Action dimension mismatch"

```bash
# 原因: 数据集动作维度与模型期望不匹配
# 解决: 确保使用 action_mode=auto
# 在 train.sh 中已默认设置
--policy.action_mode=auto
```

#### "Dataset not found"

```bash
# 原因: 数据集路径不对
# 解决: 设置正确的缓存路径
export LEROBOT_CACHE=/data/datasets
# 或检查 ~/.cache/huggingface/lerobot/ 下是否有数据
```

### A.2 推理时报错

#### "Connection refused"

```bash
# 原因: 服务没启动或端口不对
# 解决:
# 1. 确认服务正在运行
ps aux | grep server.py

# 2. 确认端口监听
netstat -tlnp | grep 8000

# 3. 确认安全组开放了端口
```

#### "Model not loaded"

```bash
# 原因: 模型加载失败
# 解决:
# 1. 检查模型路径
ls -la ./outputs/xvla_xlerobot_fold/checkpoints/best

# 2. 检查 HuggingFace 连接（如果用远程模型）
huggingface-cli whoami
```

#### "Timeout"

```bash
# 原因: 推理请求超过 10 秒
# 解决:
# 1. 检查 GPU 是否被其他进程占用
nvidia-smi

# 2. 减少去噪步数
# 在 client.py 启动参数加 --denoise-steps 5

# 3. 检查网络延迟
ping <云服务器IP>
```

### A.3 机器人不动作

#### 服务正常但机器人不动

```bash
# 检查:
# 1. 客户端是否连接到正确的服务器 URL
# 2. 服务器返回的动作值是否合理（不是全零）
# 3. 机器人的 send_action 是否正常
#    可以用之前 XLeRobot 的键盘测试脚本确认:
python /home/zach/XLeRobot/software/examples/2_dual_so100_keyboard_ee_control.py
```

#### 机器人动作抖动

```bash
# 原因: 推理动作不平滑
# 解决:
# 1. 增大控制频率
python client.py --control-freq 50

# 2. 在客户端加动作平滑
# 修改 client.py, 在 send_action 之前加:
#   current_action = current_action * 0.7 + new_action * 0.3
```

### A.4 数据传输慢

```bash
# 原因: 摄像头图像太大
# 解决: 减小图像分辨率
# 修改 client.py 中的相机配置:
cameras={
    "cam_top": OpenCVCameraConfig(..., width=320, height=240),
    ...
}
# 或者修改 server.py 中缩放到 256x256
```

---

## 附录B：概念详解

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

**说明：**
- Qwen2.5-VL 3B 是阿里巴巴的多模态大模型（视觉+语言）
- X-VLA 取其编码器部分，加上 Action Decoder
- Soft Prompt 是一组嵌入向量，告诉模型"你现在是 XLeRobot"
- 训练时 Qwen2.5-VL 可以冻结，只训练 Soft Prompt（轻量模式）

### B.2 动作空间说明

```
你的 XLeRobot 动作空间:

左臂 (6维):
  [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

右臂 (6维):
  [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

总共: 12 维

X-VLA 内部使用 20 维动作空间:
  [EE_pos(3), EE_rot_6d(6), gripper(1), padding(10)]

action_mode=auto 时:
  训练: 12维 → pad → 20维 (自动补零)
  推理: 20维 → trim → 12维 (自动裁剪)
```

### B.3 摄像头配置说明

```
三路摄像头的典型位置:

                    ┌─────────────────┐
                    │  left_top       │ ← 头顶朝下，拍到桌面全貌
                    │  (全局视角)     │   挂在左臂相机配置下
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

> **注意**：使用 `bi_so_follower` 时，观察键名为 `left_top`、`left_wrist`、`right_wrist`（自动加上 `left_`/`right_` 前缀）。
> 如果你用 `client.py`，需要把 `camera_keys` 改为 `["left_top", "left_wrist", "right_wrist"]`。
```

### B.4 LeRobot 数据集格式

```
LeRobot v3.0 格式（你录制的数据）:

目录结构:
  xlerobot-cloth-fold/
    meta/
      info.json          ← 特征定义 (action:12维, state:12维, 图像:3路)
      stats.json         ← 每个特征的均值/标准差 (用于归一化)
      episodes.jsonl     ← 每个episode的起始/结束时间
    data/
      chunk-000/
        episode_000000.parquet  ← 第0个episode的动作+状态 (Parquet格式)
        episode_000001.parquet  ← 第1个episode
        ...
    videos/
      chunk-000/
        left_top/
          episode_000000.mp4   ← 第0个episode的顶部摄像头视频
        left_wrist/
          episode_000000.mp4
        right_wrist/
          episode_000000.mp4

一个 frame (一帧数据) 包含:
  action:                   12维浮点数组 (关节目标位置)
  observation.state:        12维浮点数组 (当前关节位置)
  observation.images.left_top:    640×480×3 uint8 数组 (RGB图像)
  observation.images.left_wrist:  640×480×3 uint8 数组
  observation.images.right_wrist: 640×480×3 uint8 数组
  task:               字符串 "fold the towel on the table"
  episode_index:      整数 (属于第几个演示)
  frame_index:        整数 (演示内的第几帧)
  timestamp:          浮点数 (时间戳)
```

---

## 附录C：命令速查表

### 本地（XLeRobot 电脑）

```bash
# 校准
lerobot-calibrate --teleop.type=so100_leader --teleop.port=/dev/ttyACM2 --teleop.id=left_leader
lerobot-calibrate --teleop.type=so100_leader --teleop.port=/dev/ttyACM3 --teleop.id=right_leader
python calibrate_xlerobot.py --port1 /dev/ttyACM0 --port2 /dev/ttyACM1
```

### 云端 GPU 服务器

```bash
# 激活 conda 环境（如果没有，先创建）
conda activate lerobot
# 或: conda create -n lerobot python=3.12 && conda activate lerobot && pip install 'lerobot[xvla]'

# 轻量训练
bash train.sh --light

# 全量训练
bash train.sh

# 启动推理服务
python server.py --model-path ./outputs/xvla_xlerobot_fold/checkpoints/best --port 8000

# 后台运行推理服务
nohup python server.py --model-path ./outputs/.../best --port 8000 > server.log 2>&1 &

# 测试服务
curl http://localhost:8000/health
```

### 所有参数参考

**client.py:**
```
--server-url URL        云端推理服务地址 (默认: http://localhost:8000)
--task TEXT             语言指令 (默认: "fold the towel on the table")
--domain-id INT         领域ID (默认: 0)
--denoise-steps INT     去噪步数 (默认: 10, 范围 5-50)
--control-freq FLOAT    控制频率Hz (默认: 30)
--port1 PATH            左臂总线端口 (默认: /dev/ttyACM0)
--port2 PATH            右臂总线端口 (默认: /dev/ttyACM1)
```

**server.py:**
```
--model-path PATH       模型路径或HF ID (默认: lerobot/xvla-folding)
--port INT              服务端口 (默认: 8000)
--host TEXT             绑定地址 (默认: 0.0.0.0)
--device TEXT           推理设备 (默认: auto)
```

**train.sh:**
```
--dataset ID            数据集ID (默认: your/xlerobot-cloth-fold)
--output-dir DIR        输出目录 (默认: ./outputs/xvla_xlerobot_fold)
--model-path PATH       基座模型路径 (默认: lerobot/xvla-folding)
--hf-user USER          HF用户名 (可选，用于推送模型)
--light                 轻量模式 (soft prompts only)
--resume PATH           从checkpoint继续训练
```

**collect_data.py:**
```
--repo-id ID            数据集ID (默认: your/xlerobot-cloth-fold)
--num-episodes INT      演示数 (默认: 50)
--task TEXT             任务描述 (默认: "fold the towel on the table")
--episode-time INT      每段最长秒数 (默认: 30)
--fps INT               录制帧率 (默认: 30)
--dry-run               试运行不录制
--find-ports            自动检测USB端口
```

**deploy.sh:**
```
--model-path PATH       模型路径 (默认: lerobot/xvla-folding)
--port INT              服务端口 (默认: 8000)
--host TEXT             绑定地址 (默认: 0.0.0.0)
--setup-only            只装环境不启动服务
--venv DIR              虚拟环境目录 (默认: venv_xvla)
```
