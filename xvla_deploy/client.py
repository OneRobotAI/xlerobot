#!/usr/bin/env python3
"""
X-VLA Cloud Inference Client for XLeRobot

Usage:
  python client.py --server-url http://<cloud-ip>:8000

Connects to XLeRobot hardware, streams observations to cloud X-VLA server,
and executes predicted actions for cloth folding.

Requirements:
  - XLeRobot hardware connected (dual SO-100 arms)
  - Cameras configured (see config note below)
  - Cloud X-VLA inference server running
"""

import argparse
import json
import logging
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import requests

# XLeRobot imports — 自动检测硬件版本
try:
    from lerobot.robots.xlerobot_2wheels import XLerobot2WheelsConfig as RobotConfig
    from lerobot.robots.xlerobot_2wheels import XLerobot2Wheels as RobotClass
    ROBOT_VERSION = "2wheels"
    print(f"🔧 检测到 XLeRobot 2 轮差速版")
except ImportError:
    from lerobot.robots.xlerobot import XLerobotConfig as RobotConfig
    from lerobot.robots.xlerobot import XLerobot as RobotClass
    ROBOT_VERSION = "3wheels"
    print(f"🔧 检测到 XLeRobot 3 轮全向版")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger("xvla_client")

# ============================================================
# Camera Configuration
# ============================================================
# By default, XLeRobot config has cameras commented out.
# You need to configure them before running this client.
#
# Option A: Edit config_xlerobot.py to enable cameras:
#   software/src/robots/xlerobot/config_xlerobot.py
#
# Option B: Pass camera config via environment or code (see __main__ below).
#
# Example camera config for 3 cameras:
#   cameras = {
#       "cam_top": OpenCVCameraConfig(index_or_path="/dev/video0", fps=30, width=640, height=480),
#       "cam_left_wrist": OpenCVCameraConfig(index_or_path="/dev/video2", fps=30, width=640, height=480),
#       "cam_right_wrist": OpenCVCameraConfig(index_or_path="/dev/video4", fps=30, width=640, height=480),
#   }

# ============================================================
# Joint definitions for XLeRobot
# ============================================================
LEFT_ARM_JOINTS = [
    "left_arm_shoulder_pan",
    "left_arm_shoulder_lift",
    "left_arm_elbow_flex",
    "left_arm_wrist_flex",
    "left_arm_wrist_roll",
    "left_arm_gripper",
]

RIGHT_ARM_JOINTS = [
    "right_arm_shoulder_pan",
    "right_arm_shoulder_lift",
    "right_arm_elbow_flex",
    "right_arm_wrist_flex",
    "right_arm_wrist_roll",
    "right_arm_gripper",
]

ALL_ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS  # 12 joints total

# 数据集初始手臂位置（从训练数据第一帧的 observation.state 提取）
# 推理前先平滑移动到该位置，避免一启动就乱跳
INITIAL_ARM_POS = {
    "left_arm_shoulder_pan.pos": -4.0,
    "left_arm_shoulder_lift.pos": -99.6,
    "left_arm_elbow_flex.pos": 93.6,
    "left_arm_wrist_flex.pos": 67.2,
    "left_arm_wrist_roll.pos": 2.8,
    "left_arm_gripper.pos": 1.0,
    "right_arm_shoulder_pan.pos": 3.7,
    "right_arm_shoulder_lift.pos": -85.3,
    "right_arm_elbow_flex.pos": 92.8,
    "right_arm_wrist_flex.pos": 52.2,
    "right_arm_wrist_roll.pos": -1.9,
    "right_arm_gripper.pos": 2.2,
}

# 默认动作平滑系数
# 太小（0.02）会导致动作几乎不可见
# 建议值: 0.3（快速响应）, 0.5（激进）, 0.1（保守）
DEFAULT_SMOOTH_RATIO = 0.3

# ============================================================
# XLeRobot Client
# ============================================================


class XVLAXLeRobotClient:
    """Connects XLeRobot to cloud X-VLA inference server."""

    def __init__(
        self,
        server_url: str,
        task_instruction: str = "fold the towel on the table",
        domain_id: int = 0,
        denoise_steps: int = 10,
        control_freq: float = 30.0,
        smooth_ratio: float = DEFAULT_SMOOTH_RATIO,
        camera_keys: list[str] | None = None,
    ):
        self.server_url = server_url.rstrip("/") + "/act"
        self.task_instruction = task_instruction
        self.domain_id = domain_id
        self.denoise_steps = denoise_steps
        self.control_freq = control_freq
        self.smooth_ratio = smooth_ratio
        self.camera_keys = camera_keys or ["cam_top", "cam_left_wrist", "cam_right_wrist"]

        self.robot: RobotClass | None = None

    # ----- robot connection -----

    def connect_robot(self, robot_config: RobotConfig | None = None):
        """Initialize and connect to XLeRobot hardware."""
        if robot_config is None:
            robot_config = RobotConfig()

        logger.info(f"Connecting XLeRobot (port1={robot_config.port1}, port2={robot_config.port2})...")
        self.robot = RobotClass(robot_config)
        self.robot.connect()
        logger.info("✅ XLeRobot connected")

        # Log available cameras
        cam_keys = list(self.robot.cameras.keys())
        logger.info(f"Available cameras: {cam_keys}")

    def disconnect_robot(self):
        """Safely disconnect the robot."""
        if self.robot and self.robot.is_connected:
            self.robot.disconnect()
            logger.info("XLeRobot disconnected")

    # ----- observation extraction -----

    def _build_proprio(self, obs: dict) -> np.ndarray:
        """Extract 12D joint position vector from observation dict."""
        proprio = np.zeros(len(ALL_ARM_JOINTS), dtype=np.float32)
        for i, joint in enumerate(ALL_ARM_JOINTS):
            key = f"{joint}.pos"
            proprio[i] = obs.get(key, 0.0)
        return proprio

    def _build_images(self, obs: dict) -> dict[str, np.ndarray]:
        """Extract camera images from observation dict."""
        images = {}
        for cam_key in self.camera_keys:
            img = obs.get(cam_key)
            if img is not None:
                images[cam_key] = np.asarray(img)
        return images

    def _build_robot_action(self, action_12d: np.ndarray) -> dict:
        """Convert 12D action array to XLeRobot action dict.
        
        The action array is ordered as:
          [left_arm(6), right_arm(6)]
        Maps to XLeRobot's observation.action_features format.
        Model outputs 16D: [arm(12) + head_motor_1 + head_motor_2 + x.vel + theta.vel]
        """
        robot_action = {}
        for i, joint in enumerate(ALL_ARM_JOINTS):
            robot_action[f"{joint}.pos"] = float(action_12d[i])
        # 头部位置：与采集数据时的初始位置一致
        robot_action["head_motor_1.pos"] = 0.0
        robot_action["head_motor_2.pos"] = 1500.0
        # 底盘不动（2轮差速版无 y.vel）
        robot_action["x.vel"] = 0.0
        robot_action["theta.vel"] = 0.0
        return robot_action

    # ----- cloud inference -----

    def query_server(self, proprio: np.ndarray, images: dict[str, np.ndarray]) -> np.ndarray | None:
        """Send observation to cloud server, return predicted actions chunk."""
        # Build payload
        payload = {
            "proprio": self._json_numpy_dumps(proprio),
            "language_instruction": self.task_instruction,
            "domain_id": self.domain_id,
            "steps": self.denoise_steps,
        }

        # Add images (up to 3 cameras as image0, image1, image2)
        for i, cam_key in enumerate(self.camera_keys):
            if cam_key in images:
                payload[f"image{i}"] = self._json_numpy_dumps(images[cam_key])

        try:
            resp = requests.post(self.server_url, json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            actions = np.array(result["action"], dtype=np.float32)
            return actions
        except requests.exceptions.Timeout:
            logger.warning("⏱️ Server timeout, using fallback")
        except requests.exceptions.ConnectionError:
            logger.warning("🔌 Connection failed, is the server running?")
        except Exception as e:
            logger.warning(f"⚠️ Inference error: {e}")
        return None

    @staticmethod
    def _json_numpy_dumps(arr: np.ndarray) -> str:
        """Serialize numpy array to JSON string."""
        return json.dumps(arr.tolist())

    # ----- main control loop -----

    def _move_to_initial_position(self, duration: float = 3.0):
        """平滑移动到训练数据集的初始位置，避免一启动就乱跳。"""
        logger.info(f"平滑移动到初始位置（{duration}秒）...")
        obs = self.robot.get_observation()
        current = {k: obs.get(k, 0.0) for k in INITIAL_ARM_POS}

        freq = 30
        steps = int(duration * freq)
        for s in range(steps + 1):
            t = s / steps  # 0→1
            action = {}
            for key, target in INITIAL_ARM_POS.items():
                cur = current.get(key, 0.0)
                action[key] = cur + (target - cur) * t
            action["head_motor_1.pos"] = 0.0
            action["head_motor_2.pos"] = 1500.0
            action["x.vel"] = 0.0
            action["y.vel"] = 0.0
            action["theta.vel"] = 0.0
            self.robot.send_action(action)
            time.sleep(1.0 / freq)
        logger.info("✅ 已到达初始位置")

    def run(self, num_steps: int = -1):
        """Run the inference control loop.
        
        Args:
            num_steps: Number of control steps (-1 = infinite)
        """
        if self.robot is None:
            raise RuntimeError("Robot not connected. Call connect_robot() first.")

        print(f"[XVLA] 启动控制循环 (freq={self.control_freq}Hz, smooth={self.smooth_ratio})")
        print(f"[XVLA] Server: {self.server_url}")
        print(f"[XVLA] Task: '{self.task_instruction}'")
        print("[XVLA] Press Ctrl+C to stop")
        logger.info(f"Starting control loop (freq={self.control_freq}Hz)")

        # 平滑移动到训练初始位置，使模型观测与训练分布匹配
        self._move_to_initial_position()

        step = 0
        last_action_chunk = None
        chunk_index = 0
        # 动作平滑：用当前关节位置初始化
        obs = self.robot.get_observation()
        prev_action_12d = self._build_proprio(obs)

        try:
            while num_steps < 0 or step < num_steps:
                loop_start = time.perf_counter()

                # 1. Get observation from robot
                obs = self.robot.get_observation()

                # 1b. 显示摄像头画面（转 BGR 以适配 OpenCV 显示）
                for cam_key in self.camera_keys:
                    if cam_key in obs and obs[cam_key] is not None:
                        img = np.asarray(obs[cam_key])
                        if img.dtype != np.uint8:
                            img = (img * 255).clip(0, 255).astype(np.uint8)
                        # 缩放显示宽度不超过 320
                        h, w = img.shape[:2]
                        if w > 320:
                            scale = 320.0 / w
                            img = cv2.resize(img, None, fx=scale, fy=scale)
                        cv2.imshow(cam_key, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[XVLA] 🛑 按 Q 退出")
                    break

                # 2. Extract proprioception and images
                proprio = self._build_proprio(obs)
                images = self._build_images(obs)

                # 3. If we've exhausted the current action chunk, request a new one
                if last_action_chunk is None or chunk_index >= len(last_action_chunk):
                    logger.info(f"Step {step}: Requesting new action chunk from cloud...")
                    last_action_chunk = self.query_server(proprio, images)
                    chunk_index = 0

                    if last_action_chunk is None:
                        # Server unavailable — stop safely
                        logger.error("Server unavailable, stopping")
                        print("[XVLA] ❌ 服务器无响应，停止")
                        break

                    chunk_len = len(last_action_chunk)
                    print(f"[XVLA] ✅ 收到动作块: {chunk_len} 步, 形状 {last_action_chunk.shape}")
                    logger.info(f"  Received chunk of {chunk_len} actions")

                # 4. Execute the next action from the chunk
                if chunk_index < len(last_action_chunk):
                    action_full = last_action_chunk[chunk_index]  # shape (16,)
                    action_arm = action_full[:12]  # 手臂 12 维

                    # 动作平滑：当前关节位置与模型预测的混合
                    delta = np.zeros(12, dtype=np.float32)
                    if prev_action_12d is not None:
                        delta = action_arm - prev_action_12d
                        action_arm = prev_action_12d + delta * self.smooth_ratio
                    prev_action_12d = action_arm.copy()

                    # 打印动作值（每 30 帧约每秒一次）
                    if step % 30 == 0:
                        max_delta = float(np.max(np.abs(delta)))
                        print(f"  [动作] 模型输出关节0={action_full[0]:+.1f} "
                              f"发送={action_arm[0]:+.1f} 最大差值={max_delta:.1f} "
                              f"平滑={self.smooth_ratio}")

                    robot_action = self._build_robot_action(action_arm)
                    self.robot.send_action(robot_action)
                    chunk_index += 1

                # 5. Maintain control frequency
                elapsed = time.perf_counter() - loop_start
                sleep_time = 1.0 / self.control_freq - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                step += 1

                # Progress log every 50 steps
                if step % 50 == 0:
                    chunk_progress = f"chunk {chunk_index}/{chunk_len}" if last_action_chunk is not None else "no chunk"
                    logger.info(f"Step {step}: {chunk_progress}")

        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
        except Exception as e:
            logger.error(f"❌ Error in control loop: {e}")
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()
            self.disconnect_robot()
            logger.info("Client stopped")


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="X-VLA Cloud Inference Client for XLeRobot")
    parser.add_argument("--server-url", default="http://localhost:8000",
                        help="Cloud X-VLA server URL (default: http://localhost:8000)")
    parser.add_argument("--task", default="fold the towel on the table",
                        help="Language instruction for the task")
    parser.add_argument("--domain-id", type=int, default=0,
                        help="Domain ID for X-VLA model")
    parser.add_argument("--denoise-steps", type=int, default=10,
                        help="X-VLA denoising steps (default: 10)")
    parser.add_argument("--control-freq", type=float, default=30.0,
                        help="Control frequency in Hz (default: 30)")
    parser.add_argument("--smooth-ratio", type=float, default=DEFAULT_SMOOTH_RATIO,
                        help=f"Action smoothing ratio (default: {DEFAULT_SMOOTH_RATIO}). "
                             f"0.02=very smooth/slow, 0.3=responsive, 1.0=no smoothing")
    parser.add_argument("--port1", default="/dev/ttyACM0",
                        help="Left arm + head bus port (default: /dev/ttyACM0)")
    parser.add_argument("--port2", default="/dev/ttyACM1",
                        help="Right arm + base bus port (default: /dev/ttyACM1)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build robot config (cameras can be customized here)
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.configs import Cv2Rotation, Cv2Backends

    robot_config = RobotConfig(
        port1=args.port1,
        port2=args.port2,
        cameras={
            "cam_top": OpenCVCameraConfig(
                index_or_path="/dev/video0", fps=30, width=640, height=480, fourcc="MJPG", backend=Cv2Backends.V4L2,
            ),
            "cam_left_wrist": OpenCVCameraConfig(
                index_or_path="/dev/video2", fps=30, width=640, height=480, fourcc="MJPG", backend=Cv2Backends.V4L2,
            ),
            "cam_right_wrist": OpenCVCameraConfig(
                index_or_path="/dev/video4", fps=30, width=640, height=480, fourcc="MJPG", backend=Cv2Backends.V4L2,
            ),
        },
    )

    # 固定摄像头参数，确保每次启动画面一致
    import subprocess
    for cam_dev in ["/dev/video0", "/dev/video2", "/dev/video4"]:
        try:
            subprocess.run(
                ["v4l2-ctl", "-d", cam_dev,
                 "--set-ctrl", "white_balance_automatic=0",
                 "--set-ctrl", "white_balance_temperature=4600",
                 "--set-ctrl", "auto_exposure=3",
                 "--set-ctrl", "saturation=56",
                 "--set-ctrl", "brightness=0",
                 "--set-ctrl", "contrast=3"],
                capture_output=True, timeout=5
            )
        except Exception:
            pass  # 部分摄像头不支持，忽略

    client = XVLAXLeRobotClient(
        server_url=args.server_url,
        task_instruction=args.task,
        domain_id=args.domain_id,
        denoise_steps=args.denoise_steps,
        control_freq=args.control_freq,
        smooth_ratio=args.smooth_ratio,
    )

    try:
        client.connect_robot(robot_config)
        client.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        client.disconnect_robot()


if __name__ == "__main__":
    main()
