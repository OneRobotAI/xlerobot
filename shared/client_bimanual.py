#!/usr/bin/env python3
"""
双臂推理客户端（bi_so_follower + bi_so_leader）
适用于 12 维动作空间，不含头部和底盘

用法:
  python client_bimanual.py --server-url http://localhost:8000
"""

import argparse
import json
import logging
import time
import traceback
from pathlib import Path

import numpy as np
import requests

from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
from lerobot.robots.so_follower import SOFollowerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bimanual_client")

# 12 维关节定义（左右臂各 6 个）
LEFT_JOINTS = [
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
    "left_gripper",
]

RIGHT_JOINTS = [
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
    "right_gripper",
]

ALL_JOINTS = LEFT_JOINTS + RIGHT_JOINTS  # 12 joints


class BimanualClient:
    def __init__(
        self,
        server_url: str,
        task_instruction: str = "fold the towel on the table",
        control_freq: float = 30.0,
        camera_keys: list[str] | None = None,
    ):
        self.server_url = server_url.rstrip("/") + "/act"
        self.task_instruction = task_instruction
        self.control_freq = control_freq
        self.camera_keys = camera_keys or ["left_top", "left_wrist", "right_wrist"]
        self.robot: BiSOFollower | None = None

    def connect_robot(self, left_port: str = "/dev/ttyACM0", right_port: str = "/dev/ttyACM1"):
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.cameras.configs import Cv2Backends

        config = BiSOFollowerConfig(
            left_arm_config=SOFollowerConfig(
                port=left_port,
                cameras={
                    "top": OpenCVCameraConfig(index_or_path="/dev/video0", fps=30, width=640, height=480, fourcc="MJPG", backend=Cv2Backends.V4L2),
                    "wrist": OpenCVCameraConfig(index_or_path="/dev/video2", fps=30, width=640, height=480, fourcc="MJPG", backend=Cv2Backends.V4L2),
                },
            ),
            right_arm_config=SOFollowerConfig(
                port=right_port,
                cameras={
                    "wrist": OpenCVCameraConfig(index_or_path="/dev/video4", fps=30, width=640, height=480, fourcc="MJPG", backend=Cv2Backends.V4L2),
                },
            ),
        )
        logger.info(f"Connecting BiSOFollower (left={left_port}, right={right_port})...")
        self.robot = BiSOFollower(config)
        self.robot.connect()
        logger.info("✅ BiSOFollower connected")
        logger.info(f"Available cameras: {list(self.robot.cameras.keys())}")

    def disconnect_robot(self):
        if self.robot and self.robot.is_connected:
            self.robot.disconnect()
            logger.info("BiSOFollower disconnected")

    def _build_proprio(self, obs: dict) -> np.ndarray:
        proprio = np.zeros(len(ALL_JOINTS), dtype=np.float32)
        for i, joint in enumerate(ALL_JOINTS):
            key = f"{joint}.pos"
            proprio[i] = obs.get(key, 0.0)
        return proprio

    def _build_images(self, obs: dict) -> dict[str, np.ndarray]:
        images = {}
        for cam_key in self.camera_keys:
            img = obs.get(cam_key)
            if img is not None:
                images[cam_key] = np.asarray(img)
        return images

    def _build_robot_action(self, action_12d: np.ndarray) -> dict:
        robot_action = {}
        for i, joint in enumerate(ALL_JOINTS):
            robot_action[f"{joint}.pos"] = float(action_12d[i])
        return robot_action

    def query_server(self, proprio: np.ndarray, images: dict[str, np.ndarray]) -> np.ndarray | None:
        payload = {
            "proprio": json.dumps(proprio.tolist()),
            "language_instruction": self.task_instruction,
            "domain_id": 0,
            "steps": 10,
        }
        for i, cam_key in enumerate(self.camera_keys):
            if cam_key in images:
                payload[f"image{i}"] = json.dumps(images[cam_key].tolist())

        try:
            resp = requests.post(self.server_url, json=payload, timeout=10)
            resp.raise_for_status()
            return np.array(resp.json()["action"], dtype=np.float32)
        except requests.exceptions.Timeout:
            logger.warning("⏱️ Server timeout")
        except requests.exceptions.ConnectionError:
            logger.warning("🔌 Connection failed")
        except Exception as e:
            logger.warning(f"⚠️ Inference error: {e}")
        return None

    def run(self, num_steps: int = -1):
        if self.robot is None:
            raise RuntimeError("Robot not connected. Call connect_robot() first.")

        logger.info(f"Starting control loop (freq={self.control_freq}Hz)")
        logger.info(f"Server: {self.server_url}")
        logger.info(f"Task: '{self.task_instruction}'")

        step = 0
        last_action_chunk = None
        chunk_index = 0
        # 动作平滑：用当前关节位置初始化，让模型从当前位置开始
        obs = self.robot.get_observation()
        prev_action_12d = self._build_proprio(obs)

        try:
            while num_steps < 0 or step < num_steps:
                loop_start = time.perf_counter()
                obs = self.robot.get_observation()
                proprio = self._build_proprio(obs)
                images = self._build_images(obs)

                if last_action_chunk is None or chunk_index >= len(last_action_chunk):
                    logger.info(f"Step {step}: Requesting new action chunk...")
                    last_action_chunk = self.query_server(proprio, images)
                    chunk_index = 0
                    if last_action_chunk is None:
                        logger.error("Server unavailable, stopping")
                        break
                    logger.info(f"  Received chunk of {len(last_action_chunk)} actions")

                if chunk_index < len(last_action_chunk):
                    action_12d = last_action_chunk[chunk_index]
                    # 动作平滑：每次只向模型预测方向移动 2%
                    SMOOTH_RATIO = 0.02
                    if prev_action_12d is not None:
                        delta = action_12d[:12] - prev_action_12d[:12]
                        action_12d = prev_action_12d[:12] + delta * SMOOTH_RATIO
                    prev_action_12d = action_12d.copy()
                    robot_action = self._build_robot_action(action_12d[:12])
                    self.robot.send_action(robot_action)
                    chunk_index += 1

                elapsed = time.perf_counter() - loop_start
                sleep_time = 1.0 / self.control_freq - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                step += 1
                if step % 50 == 0:
                    logger.info(f"Step {step}: chunk {chunk_index}/{len(last_action_chunk) if last_action_chunk is not None else 0}")

        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            traceback.print_exc()
        finally:
            self.disconnect_robot()
            logger.info("Client stopped")


def main():
    parser = argparse.ArgumentParser(description="Bimanual VLA Inference Client")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--task", default="fold the towel on the table")
    parser.add_argument("--left-port", default="/dev/ttyACM0")
    parser.add_argument("--right-port", default="/dev/ttyACM1")
    parser.add_argument("--control-freq", type=float, default=30.0)
    args = parser.parse_args()

    client = BimanualClient(
        server_url=args.server_url,
        task_instruction=args.task,
        control_freq=args.control_freq,
    )
    try:
        client.connect_robot(args.left_port, args.right_port)
        client.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        client.disconnect_robot()


if __name__ == "__main__":
    main()
