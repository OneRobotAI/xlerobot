#!/usr/bin/env python3
"""
XLerobot 校准脚本

支持两种模式：
  - xlerobot（默认）：校准全部电机（含头部、底盘）
  - lerobot（双臂）：只校准 12 个手臂电机（标准 LeRobot bi_so_follower）

用法:
  # XLerobot 全校准（推荐，含头部和底盘）
  python shared/calibrate_xlerobot.py

  # 标准 LeRobot 双臂校准（只校准手臂）
  python shared/calibrate_xlerobot.py --mode lerobot

  # 指定端口
  python shared/calibrate_xlerobot.py --port1 /dev/ttyACM0 --port2 /dev/ttyACM1
  python shared/calibrate_xlerobot.py --mode lerobot --port1 /dev/ttyACM0 --port2 /dev/ttyACM1
"""

import argparse
import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def calibrate_xlerobot(port1: str, port2: str, version: str | None):
    """使用 XLerobot 类校准全部电机（含头部、底盘）。"""
    if version == "2wheels":
        from lerobot.robots.xlerobot_2wheels import XLerobot2WheelsConfig, XLerobot2Wheels
        ConfigClass = XLerobot2WheelsConfig
        RobotClass = XLerobot2Wheels
        label = "2轮差速底盘"
    else:
        from lerobot.robots.xlerobot import XLerobotConfig, XLerobot
        ConfigClass = XLerobotConfig
        RobotClass = XLerobot
        label = "3轮全向底盘"

    config = ConfigClass(port1=port1, port2=port2)

    print("=" * 60)
    print(f"  XLerobot 校准（{label}）")
    print("  覆盖电机：左臂6 + 右臂6 + 头部2 + 底盘")
    print("=" * 60)

    robot = RobotClass(config)
    robot.connect(calibrate=True)
    robot.disconnect()
    print("\n✅ 校准完成！数据已保存到", robot.calibration_fpath)


def calibrate_lerobot(port1: str, port2: str):
    """使用标准 LeRobot bi_so_follower 校准 12 个手臂电机。"""
    print("=" * 60)
    print("  LeRobot 双臂校准（bi_so_follower）")
    print("  覆盖电机：左臂6 + 右臂6（不含头部、底盘）")
    print("=" * 60)

    # lerobot-calibrate 是命令行入口，不是 python -m 模块
    cmd = [
        "lerobot-calibrate",
        "--robot.type=bi_so_follower",
        f"--robot.left_arm_config.port={port1}",
        f"--robot.right_arm_config.port={port2}",
        "--robot.id=xlerobot_follower",
    ]
    subprocess.run(cmd, check=True)


def detect_version() -> str:
    """自动检测硬件版本。"""
    try:
        from lerobot.robots.xlerobot_2wheels import XLerobot2WheelsConfig  # noqa: F401
        return "2wheels"
    except ImportError:
        return "3wheels"


def main():
    parser = argparse.ArgumentParser(description="XLerobot 校准")
    parser.add_argument("--port1", default="/dev/ttyACM0",
                        help="左臂+头部总线端口 (默认: /dev/ttyACM0)")
    parser.add_argument("--port2", default="/dev/ttyACM1",
                        help="右臂+底盘总线端口 (默认: /dev/ttyACM1)")
    parser.add_argument("--mode", choices=["xlerobot", "lerobot"], default="xlerobot",
                        help="校准模式: xlerobot（全量，含头部底盘）/ lerobot（仅手臂）")
    parser.add_argument("--version", choices=["3wheels", "2wheels"], default=None,
                        help="硬件版本（默认自动检测）")
    args = parser.parse_args()

    if args.mode == "lerobot":
        calibrate_lerobot(args.port1, args.port2)
    else:
        version = args.version or detect_version()
        calibrate_xlerobot(args.port1, args.port2, version)


if __name__ == "__main__":
    main()
