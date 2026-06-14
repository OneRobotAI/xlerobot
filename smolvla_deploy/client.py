#!/usr/bin/env python3
"""
SmolVLA Cloud Inference Client for XLeRobot

Usage:
  python client.py --server-url http://<cloud-ip>:8000

Connects to XLeRobot hardware, streams observations to SmolVLA cloud server,
and executes predicted actions for cleaning tasks.

API compatible with smolvla_deploy/server.py (POST /act, GET /health).

Supports:
  - XLerobot (3-wheel omni, 17-DOF)
  - XLerobot2Wheels (2-wheel diff drive, 16-DOF)
  - Auto-detection of robot version
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

# XLeRobot imports — auto-detect hardware version
try:
    from lerobot.robots.xlerobot_2wheels import XLerobot2WheelsConfig as RobotConfig
    from lerobot.robots.xlerobot_2wheels import XLerobot2Wheels as RobotClass
    ROBOT_VERSION = "2wheels"
    ROBOT_DOF = 16  # 12 arm + 2 head + 2 base (x.vel, theta.vel)
    print(f"  XLeRobot 2-Wheel Diff Drive ({ROBOT_DOF}-DOF)")
except ImportError:
    from lerobot.robots.xlerobot import XLerobotConfig as RobotConfig
    from lerobot.robots.xlerobot import XLerobot as RobotClass
    ROBOT_VERSION = "3wheels"
    ROBOT_DOF = 17  # 12 arm + 2 head + 3 base (x.vel, y.vel, theta.vel)
    print(f"  XLeRobot 3-Wheel Omni ({ROBOT_DOF}-DOF)")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger("smolvla_client")

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

# Smooth initial arm position (from clean-table dataset first frame)
# Prevents sudden jumps when starting inference
INITIAL_ARM_POS = {
    "left_arm_shoulder_pan.pos": 0.0,
    "left_arm_shoulder_lift.pos": -90.0,
    "left_arm_elbow_flex.pos": 90.0,
    "left_arm_wrist_flex.pos": 60.0,
    "left_arm_wrist_roll.pos": 0.0,
    "left_arm_gripper.pos": 1.0,
    "right_arm_shoulder_pan.pos": 0.0,
    "right_arm_shoulder_lift.pos": -90.0,
    "right_arm_elbow_flex.pos": 90.0,
    "right_arm_wrist_flex.pos": 60.0,
    "right_arm_wrist_roll.pos": 0.0,
    "right_arm_gripper.pos": 1.0,
}

SMOOTH_RATIO = 0.3  # Action smoothing coefficient


# ============================================================
# SmolVLA Client
# ============================================================

class SmolVLAXLeRobotClient:
    """Connects XLeRobot to cloud SmolVLA inference server."""

    def __init__(
        self,
        server_url: str,
        task_instruction: str = "clean table",
        control_freq: float = 30.0,
        smooth_ratio: float = SMOOTH_RATIO,
        camera_keys: list[str] | None = None,
    ):
        self.server_url = server_url.rstrip("/") + "/act"
        self.task_instruction = task_instruction
        self.control_freq = control_freq
        self.smooth_ratio = smooth_ratio
        self.camera_keys = camera_keys or ["cam_top", "cam_left_wrist", "cam_right_wrist"]
        self.robot: RobotClass | None = None
        self.action_dim = 12  # Will be updated from server /health

    # ----- robot connection -----

    def connect_robot(self, robot_config: RobotConfig | None = None):
        """Initialize and connect to XLeRobot hardware."""
        if robot_config is None:
            robot_config = RobotConfig()

        logger.info(f"Connecting XLeRobot (port1={robot_config.port1}, port2={robot_config.port2})...")
        self.robot = RobotClass(robot_config)
        self.robot.connect()
        logger.info(" XLeRobot connected")

        cam_keys = list(self.robot.cameras.keys())
        logger.info(f"Available cameras: {cam_keys}")

    def disconnect_robot(self):
        """Safely disconnect the robot."""
        if self.robot and self.robot.is_connected:
            self.robot.disconnect()
            logger.info("XLeRobot disconnected")

    # ----- server info -----

    def fetch_server_info(self):
        """Query /health to get model dimensions (action_dim, state_dim)."""
        try:
            base = self.server_url.replace("/act", "")
            resp = requests.get(f"{base}/health", timeout=5)
            info = resp.json()
            if "action_dim" in info:
                self.action_dim = info["action_dim"]
                logger.info(f"Server reports action_dim={self.action_dim}")
            return info
        except Exception:
            logger.warning("Could not fetch server info, using default action_dim=12")
            return {}

    # ----- observation extraction -----

    def _build_proprio(self, obs: dict) -> np.ndarray:
        """Extract full state vector (16-DOF: 12 arms + 2 head + 2 base)."""
        proprio = np.zeros(ROBOT_DOF, dtype=np.float32)
        for i, joint in enumerate(ALL_ARM_JOINTS):
            proprio[i] = obs.get(f"{joint}.pos", 0.0)
        # Head motors
        if len(proprio) > 12:
            proprio[12] = obs.get("head_motor_1.pos", 0.0)
            proprio[13] = obs.get("head_motor_2.pos", 1500.0)
        # Base velocities
        if len(proprio) > 14:
            proprio[14] = obs.get("x.vel", 0.0)
            if len(proprio) > 15:
                proprio[15] = obs.get("theta.vel", 0.0)
        return proprio

    def _build_images(self, obs: dict) -> dict[str, np.ndarray]:
        """Extract camera images from observation dict."""
        images = {}
        for cam_key in self.camera_keys:
            img = obs.get(cam_key)
            if img is not None:
                images[cam_key] = np.asarray(img)
        return images

    def _build_robot_action(self, action_nd: np.ndarray) -> dict:
        """Convert model action vector to XLeRobot action dict.

        SmolVLA outputs action_dim-D vector. We map:
          [0:12]  → arm joints (6 per arm)
          [12:14] → head motors (if available)
          [14:]   → base velocities
        """
        robot_action = {}
        for i, joint in enumerate(ALL_ARM_JOINTS):
            robot_action[f"{joint}.pos"] = float(action_nd[i])

        # Head motors (if action_dim >= 14)
        robot_action["head_motor_1.pos"] = float(action_nd[12]) if len(action_nd) > 12 else 0.0
        robot_action["head_motor_2.pos"] = float(action_nd[13]) if len(action_nd) > 13 else 1500.0

        # Base velocities
        robot_action["x.vel"] = float(action_nd[14]) if len(action_nd) > 14 else 0.0
        robot_action["y.vel"] = float(action_nd[15]) if len(action_nd) > 15 else 0.0
        robot_action["theta.vel"] = float(action_nd[16]) if len(action_nd) > 16 else 0.0

        return robot_action

    # ----- cloud inference -----

    def query_server(self, proprio: np.ndarray, images: dict[str, np.ndarray]) -> np.ndarray | None:
        """Send observation to cloud server, return predicted action chunk."""
        payload = {
            "proprio": self._json_numpy_dumps(proprio),
            "language_instruction": self.task_instruction,
        }

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
            logger.warning(" Server timeout, using fallback")
        except requests.exceptions.ConnectionError:
            logger.warning(" Connection failed, is the server running?")
        except Exception as e:
            logger.warning(f" Inference error: {e}")
        return None

    @staticmethod
    def _json_numpy_dumps(arr: np.ndarray) -> str:
        return json.dumps(arr.tolist())

    # ----- main control loop -----

    def _move_to_initial_position(self, duration: float = 3.0):
        """Smoothly move to initial position to avoid sudden jumps."""
        logger.info(f"Smooth move to initial position ({duration}s)...")
        obs = self.robot.get_observation()
        current = {k: obs.get(k, 0.0) for k in INITIAL_ARM_POS}

        freq = 30
        steps = int(duration * freq)
        for s in range(steps + 1):
            t = s / steps
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
        logger.info(" At initial position")

    def run(self, num_steps: int = -1):
        """Run the inference control loop.

        Args:
            num_steps: Number of control steps (-1 = infinite)
        """
        if self.robot is None:
            raise RuntimeError("Robot not connected. Call connect_robot() first.")

        # Fetch model dimensions from server
        self.fetch_server_info()

        print(f"\n[SmolVLA] Starting control loop")
        print(f"  Server:  {self.server_url}")
        print(f"  Task:    '{self.task_instruction}'")
        print(f"  Freq:    {self.control_freq}Hz")
        print(f"  Smooth:  {self.smooth_ratio}")
        print(f"  Robot:   {ROBOT_VERSION} ({ROBOT_DOF}-DOF)")
        print(f"  Action:  {self.action_dim}-DOF")
        print("  Press Ctrl+C to stop\n")
        logger.info(f"Starting control loop (freq={self.control_freq}Hz)")

        # Smoothly move to initial position
        self._move_to_initial_position()

        step = 0
        last_action_chunk = None
        chunk_index = 0
        obs = self.robot.get_observation()
        prev_action = self._build_proprio(obs)

        try:
            while num_steps < 0 or step < num_steps:
                loop_start = time.perf_counter()

                # 1. Get observation from robot
                obs = self.robot.get_observation()

                # 1b. Display camera feed
                for cam_key in self.camera_keys:
                    if cam_key in obs and obs[cam_key] is not None:
                        img = np.asarray(obs[cam_key])
                        if img.dtype != np.uint8:
                            img = (img * 255).clip(0, 255).astype(np.uint8)
                        h, w = img.shape[:2]
                        if w > 320:
                            scale = 320.0 / w
                            img = cv2.resize(img, None, fx=scale, fy=scale)
                        cv2.imshow(cam_key, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[SmolVLA] Q pressed, exiting")
                    break

                # 2. Extract proprioception and images
                proprio = self._build_proprio(obs)
                images = self._build_images(obs)

                # 3. Request new action chunk if current one is exhausted
                if last_action_chunk is None or chunk_index >= len(last_action_chunk):
                    logger.info(f"Step {step}: Requesting new action chunk...")
                    last_action_chunk = self.query_server(proprio, images)
                    chunk_index = 0

                    if last_action_chunk is None:
                        logger.error("Server unavailable, stopping")
                        print("[SmolVLA]  Server unavailable, stopping")
                        break

                    print(f"[SmolVLA] Received chunk: {len(last_action_chunk)} steps, "
                          f"shape {last_action_chunk.shape}")

                # 4. Execute next action from chunk
                if chunk_index < len(last_action_chunk):
                    action_raw = last_action_chunk[chunk_index]

                    # Action smoothing
                    if prev_action is not None and len(action_raw) >= len(prev_action):
                        delta = action_raw[:len(prev_action)] - prev_action
                        action_smoothed = prev_action + delta * self.smooth_ratio
                    else:
                        action_smoothed = action_raw[:ROBOT_DOF] if len(action_raw) > ROBOT_DOF else action_raw
                    prev_action = action_smoothed.copy()

                    # Log (every 30 steps)
                    if step % 30 == 0:
                        max_delta = float(np.max(np.abs(delta))) if 'delta' in dir() else 0.0
                        print(f"  [Action] max_delta={max_delta:.1f} "
                              f"smooth={self.smooth_ratio} chunk={chunk_index}/{len(last_action_chunk)}")

                    robot_action = self._build_robot_action(action_smoothed)
                    self.robot.send_action(robot_action)
                    chunk_index += 1

                # 5. Maintain control frequency
                elapsed = time.perf_counter() - loop_start
                sleep_time = 1.0 / self.control_freq - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                step += 1

                if step % 50 == 0:
                    logger.info(f"Step {step}")

        except KeyboardInterrupt:
            logger.info(" Interrupted by user")
        except Exception as e:
            logger.error(f" Error in control loop: {e}")
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()
            self.disconnect_robot()
            logger.info("Client stopped")


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SmolVLA Cloud Inference Client for XLeRobot"
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help="SmolVLA server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--task",
        default="clean table",
        help='Language instruction / task description (default: "clean table")',
    )
    parser.add_argument(
        "--control-freq",
        type=float,
        default=30.0,
        help="Control frequency in Hz (default: 30)",
    )
    parser.add_argument(
        "--smooth-ratio",
        type=float,
        default=SMOOTH_RATIO,
        help=f"Action smoothing ratio (default: {SMOOTH_RATIO})",
    )
    parser.add_argument(
        "--port1",
        default="/dev/ttyACM0",
        help="Left arm + head bus port (default: /dev/ttyACM0)",
    )
    parser.add_argument(
        "--port2",
        default="/dev/ttyACM1",
        help="Right arm + base bus port (default: /dev/ttyACM1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Build robot config with cameras
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.configs import Cv2Backends

    robot_config = RobotConfig(
        port1=args.port1,
        port2=args.port2,
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
        },
    )

    # Fix camera parameters for consistent appearance (comment out if colors look wrong)
    import subprocess
    for cam_dev in ["/dev/video0", "/dev/video2", "/dev/video4"]:
        try:
            subprocess.run(
                ["v4l2-ctl", "-d", cam_dev,
                 "--set-ctrl", "auto_exposure=3",
                 "--set-ctrl", "white_balance_automatic=1",
                 "--set-ctrl", "brightness=0",
                 "--set-ctrl", "contrast=5",
                 "--set-ctrl", "saturation=64"],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass

    client = SmolVLAXLeRobotClient(
        server_url=args.server_url,
        task_instruction=args.task,
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
