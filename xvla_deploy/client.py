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

import numpy as np
import requests

# XLeRobot imports
from lerobot.robots.xlerobot import XLerobotConfig, XLerobot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
        camera_keys: list[str] | None = None,
    ):
        self.server_url = server_url.rstrip("/") + "/act"
        self.task_instruction = task_instruction
        self.domain_id = domain_id
        self.denoise_steps = denoise_steps
        self.control_freq = control_freq
        self.camera_keys = camera_keys or ["cam_top", "cam_left_wrist", "cam_right_wrist"]

        self.robot: XLerobot | None = None

    # ----- robot connection -----

    def connect_robot(self, robot_config: XLerobotConfig | None = None):
        """Initialize and connect to XLeRobot hardware."""
        if robot_config is None:
            robot_config = XLerobotConfig()

        logger.info(f"Connecting XLeRobot (port1={robot_config.port1}, port2={robot_config.port2})...")
        self.robot = XLerobot(robot_config)
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
        """
        robot_action = {}
        for i, joint in enumerate(ALL_ARM_JOINTS):
            robot_action[f"{joint}.pos"] = float(action_12d[i])
        # Keep head and base still during cloth folding
        robot_action["head_motor_1.pos"] = 0.0
        robot_action["head_motor_2.pos"] = 0.0
        robot_action["x.vel"] = 0.0
        robot_action["y.vel"] = 0.0
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

    def run(self, num_steps: int = -1):
        """Run the inference control loop.
        
        Args:
            num_steps: Number of control steps (-1 = infinite)
        """
        if self.robot is None:
            raise RuntimeError("Robot not connected. Call connect_robot() first.")

        logger.info(f"Starting control loop (freq={self.control_freq}Hz)")
        logger.info(f"Server: {self.server_url}")
        logger.info(f"Task: '{self.task_instruction}'")
        logger.info("Press Ctrl+C to stop")

        step = 0
        last_action_chunk = None
        chunk_index = 0

        try:
            while num_steps < 0 or step < num_steps:
                loop_start = time.perf_counter()

                # 1. Get observation from robot
                obs = self.robot.get_observation()

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
                        break

                    logger.info(f"  Received chunk of {len(last_action_chunk)} actions")
                    chunk_len = len(last_action_chunk)

                # 4. Execute the next action from the chunk
                if chunk_index < len(last_action_chunk):
                    action_12d = last_action_chunk[chunk_index]
                    robot_action = self._build_robot_action(action_12d[:12])
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
    parser.add_argument("--port1", default="/dev/ttyACM0",
                        help="Left arm + head bus port (default: /dev/ttyACM0)")
    parser.add_argument("--port2", default="/dev/ttyACM1",
                        help="Right arm + base bus port (default: /dev/ttyACM1)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build robot config (cameras can be customized here)
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.configs import Cv2Rotation

    robot_config = XLerobotConfig(
        port1=args.port1,
        port2=args.port2,
        cameras={
            "cam_top": OpenCVCameraConfig(
                index_or_path="/dev/video0", fps=30, width=640, height=480,
            ),
            "cam_left_wrist": OpenCVCameraConfig(
                index_or_path="/dev/video2", fps=30, width=640, height=480,
            ),
            "cam_right_wrist": OpenCVCameraConfig(
                index_or_path="/dev/video4", fps=30, width=640, height=480,
            ),
        },
    )

    client = XVLAXLeRobotClient(
        server_url=args.server_url,
        task_instruction=args.task,
        domain_id=args.domain_id,
        denoise_steps=args.denoise_steps,
        control_freq=args.control_freq,
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
