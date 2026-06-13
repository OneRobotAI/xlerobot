#!/usr/bin/env python3
"""
读取 XLeRobot 当前关节位置，生成 INITIAL_ARM_POS 配置。

用法:
  # 先把机械臂摆到数据采集时的起始姿势（可用手轻轻掰到正确位置）
  # 然后运行本脚本：
  python shared/read_init_pose.py

输出:
  直接打印可复制粘贴到 client.py 的 INITIAL_ARM_POS 字典
"""

import logging
import sys

logging.basicConfig(level=logging.WARNING)

# 自动检测硬件版本
try:
    from lerobot.robots.xlerobot_2wheels import XLerobot2WheelsConfig as RobotConfig
    from lerobot.robots.xlerobot_2wheels import XLerobot2Wheels as RobotClass
    version = "2轮差速"
except ImportError:
    from lerobot.robots.xlerobot import XLerobotConfig as RobotConfig
    from lerobot.robots.xlerobot import XLerobot as RobotClass
    version = "3轮全向"

ALL_ARM_JOINTS = [
    "left_arm_shoulder_pan", "left_arm_shoulder_lift", "left_arm_elbow_flex",
    "left_arm_wrist_flex", "left_arm_wrist_roll", "left_arm_gripper",
    "right_arm_shoulder_pan", "right_arm_shoulder_lift", "right_arm_elbow_flex",
    "right_arm_wrist_flex", "right_arm_wrist_roll", "right_arm_gripper",
]

def main():
    config = RobotConfig()
    robot = RobotClass(config)
    robot.connect()

    obs = robot.get_observation()
    print(f"\n✅ 已连接 XLeRobot ({version})")
    print(f"\n当前关节位置（复制以下内容到 client.py 的 INITIAL_ARM_POS）：")
    print("=" * 60)
    print("INITIAL_ARM_POS = {")
    for joint in ALL_ARM_JOINTS:
        key = f"{joint}.pos"
        val = obs.get(key, 0.0)
        print(f"    \"{key}\": {val:.1f},")
    print("}")
    print("=" * 60)
    print(f"\n⚠️  确认这是你数据采集时的起始姿势！")
    print(f"   如果不是，请用手把机械臂掰到正确位置后重新运行。")

    robot.disconnect()


if __name__ == "__main__":
    main()
