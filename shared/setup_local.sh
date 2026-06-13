#!/usr/bin/env bash
# ============================================================
# XLeRobot 本地开发环境设置（VLA 通用）
# ============================================================
# 用法:
#   bash setup_local.sh
#
# 说明:
#   创建软链接，使 client.py 能导入 XLeRobot 的机器人类。
#   支持 3 轮全向版 (xlerobot) 和 2 轮差速版 (xlerobot_2wheels)。
#
#   软链接不会进 Git，每个克隆此仓库的人都需要运行一次。
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
XLE_ROBOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROBOTS_SRC="$XLE_ROBOT_DIR/software/src/robots"

SYMLINKS=(
    "xlerobot"
    "xlerobot_2wheels"
    "xlerobot_mecanum"
)

# 找到 LeRobot 包的安装路径
LEROBOT_DIR=$(python3 -c "import lerobot; import os; print(os.path.dirname(lerobot.__file__))" 2>/dev/null)

if [ -z "$LEROBOT_DIR" ]; then
    echo "❌ 找不到 lerobot 包，请先: pip install lerobot[xvla]"
    exit 1
fi

ROBOTS_DIR="$LEROBOT_DIR/robots"

for name in "${SYMLINKS[@]}"; do
    SRC="$ROBOTS_SRC/$name"
    LINK="$ROBOTS_DIR/$name"

    if [ ! -d "$SRC" ]; then
        echo "⚠️  跳过 $name（源目录不存在: $SRC）"
        continue
    fi

    if [ -L "$LINK" ]; then
        echo "✅ 软链接已存在: $LINK"
    elif [ -d "$LINK" ]; then
        echo "⚠️  目录已存在（非链接），跳过: $LINK"
    else
        ln -s "$SRC" "$LINK"
        echo "✅ 已创建软链接: $LINK → $SRC"
    fi
done

# 验证
echo ""
python3 -c "from lerobot.robots.xlerobot import XLerobot; print('✅ XLerobot（3轮全向）导入成功')"
python3 -c "from lerobot.robots.xlerobot_2wheels import XLerobot2Wheels; print('✅ XLerobot2Wheels（2轮差速）导入成功')"
