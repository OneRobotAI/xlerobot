#!/usr/bin/env python3
"""
Data Collection for X-VLA Cloth Folding Training

Uses dual leader arms (bi_so100) to teleoperate XLeRobot and record
demonstrations for fine-tuning X-VLA on cloth folding.

Hardware setup:
  Leader arms:  /dev/ttyACM2 (left), /dev/ttyACM3 (right)
  Follower arms: /dev/ttyACM0 (left), /dev/ttyACM1 (right)
  Cameras:      /dev/video0 (top), /dev/video2 (left wrist), /dev/video4 (right wrist)

Usage:
  # Record 50 episodes of cloth folding
  python collect_data.py --num-episodes 50 --task "fold the towel on the table"

  # Dry run (don't record, just preview)
  python collect_data.py --dry-run
"""

import argparse
import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("data_collection")

# ============================================================
# Default hardware config
# ============================================================
FOLLOWER_LEFT_PORT = "/dev/ttyACM0"
FOLLOWER_RIGHT_PORT = "/dev/ttyACM1"
LEADER_LEFT_PORT = "/dev/ttyACM2"
LEADER_RIGHT_PORT = "/dev/ttyACM3"

CAMERA_TOP = "/dev/video0"
CAMERA_LEFT_WRIST = "/dev/video2"
CAMERA_RIGHT_WRIST = "/dev/video4"


def build_lerobot_record_cmd(
    repo_id: str,
    num_episodes: int,
    task: str,
    episode_time_s: int = 30,
    fps: int = 30,
    dry_run: bool = False,
) -> list[str]:
    """Build the lerobot-record command for bimanual data collection."""
    # Cameras are configured per-arm for bi_so_follower
    # Observation keys will be: left_top, left_wrist, right_wrist
    left_cameras_json = (
        '{'
        f'  top: {{"type": "opencv", "index_or_path": "{CAMERA_TOP}", "width": 640, "height": 480, "fps": {fps}}},'
        f'  wrist: {{"type": "opencv", "index_or_path": "{CAMERA_LEFT_WRIST}", "width": 640, "height": 480, "fps": {fps}}}'
        '}'
    )

    right_cameras_json = (
        '{'
        f'  wrist: {{"type": "opencv", "index_or_path": "{CAMERA_RIGHT_WRIST}", "width": 640, "height": 480, "fps": {fps}}}'
        '}'
    )

    cmd = [
        sys.executable, "-m", "lerobot.scripts.record",
        f"--robot.type=bi_so_follower",
        f"--robot.left_arm_config.port={FOLLOWER_LEFT_PORT}",
        f"--robot.right_arm_config.port={FOLLOWER_RIGHT_PORT}",
        f"--robot.id=xlerobot_cloth_fold",
        f"--robot.left_arm_config.cameras='{left_cameras_json}'",
        f"--robot.right_arm_config.cameras='{right_cameras_json}'",
        f"--teleop.type=bi_so_leader",
        f"--teleop.left_arm_config.port={LEADER_LEFT_PORT}",
        f"--teleop.right_arm_config.port={LEADER_RIGHT_PORT}",
        f"--teleop.id=bimanual_leader",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.single_task={task}",
        f"--dataset.fps={fps}",
        f"--dataset.episode_time_s={episode_time_s}",
        "--display_data=true",
    ]

    if dry_run:
        cmd.append("--dry-run")

    return cmd


def find_ports():
    """Find connected USB ports for all arms."""
    logger.info("Finding connected robot ports...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "lerobot.scripts.find_port"],
            capture_output=True, text=True, timeout=30
        )
        logger.info(f"Available ports:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Port find stderr:\n{result.stderr}")
    except Exception as e:
        logger.warning(f"Could not auto-detect ports: {e}")
        logger.info(f"Using defaults: Follower={FOLLOWER_LEFT_PORT},{FOLLOWER_RIGHT_PORT}")
        logger.info(f"                 Leader={LEADER_LEFT_PORT},{LEADER_RIGHT_PORT}")


def main():
    parser = argparse.ArgumentParser(description="Collect cloth folding demonstration data")
    parser.add_argument("--repo-id", default="your/xlerobot-cloth-fold",
                        help="HuggingFace dataset repo ID (default: your/xlerobot-cloth-fold)")
    parser.add_argument("--num-episodes", type=int, default=50,
                        help="Number of episodes to record (default: 50)")
    parser.add_argument("--task", default="fold the towel on the table",
                        help="Task description (default: fold the towel on the table)")
    parser.add_argument("--episode-time", type=int, default=30,
                        help="Max episode duration in seconds (default: 30)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Recording FPS (default: 30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry run (just show command, don't record)")
    parser.add_argument("--find-ports", action="store_true",
                        help="Auto-detect USB ports before recording")
    args = parser.parse_args()

    print("=" * 60)
    print("  X-VLA Data Collection for Cloth Folding")
    print("=" * 60)
    print(f"  Task:            {args.task}")
    print(f"  Episodes:        {args.num_episodes}")
    print(f"  Duration/ep:     {args.episode_time}s")
    print(f"  FPS:             {args.fps}")
    print(f"  Output repo:     {args.repo_id}")
    print(f"  Robot type:      bi_so_follower")
    print(f"  Follower arms:   left={FOLLOWER_LEFT_PORT}, right={FOLLOWER_RIGHT_PORT}")
    print(f"  Leader arms:     left={LEADER_LEFT_PORT}, right={LEADER_RIGHT_PORT}")
    print(f"  Cameras:         left(top+left_wrist) on {CAMERA_TOP},{CAMERA_LEFT_WRIST}")
    print(f"                   right(right_wrist) on {CAMERA_RIGHT_WRIST}")
    print(f"  Obs keys:        left_top, left_wrist, right_wrist")
    print("=" * 60)

    # Find ports if requested
    if args.find_ports:
        find_ports()

    # Build command
    cmd = build_lerobot_record_cmd(
        repo_id=args.repo_id,
        num_episodes=args.num_episodes,
        task=args.task,
        episode_time_s=args.episode_time,
        fps=args.fps,
        dry_run=args.dry_run,
    )

    print(f"\nRunning command:\n{' '.join(cmd)}\n")

    if args.dry_run:
        print("✅ Dry run complete. Run without --dry-run to record.")
        return

    # Execute
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Recording complete! {args.num_episodes} episodes saved.")
        print(f"   Dataset: {args.repo_id}")
        print(f"   Local path: ~/.cache/huggingface/lerobot/{args.repo_id.replace('/', '_')}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Recording failed with code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Recording interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
