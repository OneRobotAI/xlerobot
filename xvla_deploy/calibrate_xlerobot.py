#!/usr/bin/env python3
"""
XLerobot 完整校准脚本（17个电机：左臂6 + 右臂6 + 头部2 + 底盘3）

用法:
  python calibrate_xlerobot.py
  python calibrate_xlerobot.py --port1 /dev/ttyACM0 --port2 /dev/ttyACM1
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from lerobot.robots.xlerobot import XLerobotConfig, XLerobot


def main():
    parser = argparse.ArgumentParser(description="XLerobot 完整校准")
    parser.add_argument("--port1", default="/dev/ttyACM0",
                        help="左臂+头部总线端口 (默认: /dev/ttyACM0)")
    parser.add_argument("--port2", default="/dev/ttyACM1",
                        help="右臂+底盘总线端口 (默认: /dev/ttyACM1)")
    args = parser.parse_args()

    config = XLerobotConfig(
        port1=args.port1,
        port2=args.port2,
    )

    print("=" * 60)
    print("  XLerobot 校准")
    print("  覆盖电机：左臂6 + 右臂6 + 头部2 + 底盘3 = 17个")
    print("=" * 60)

    robot = XLerobot(config)
    robot.connect(calibrate=True)
    robot.disconnect()
    print("\n✅ 校准完成！数据已保存到", robot.calibration_fpath)


if __name__ == "__main__":
    main()
