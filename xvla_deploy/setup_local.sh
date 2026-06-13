#!/usr/bin/env bash
# ============================================================
# X-VLA × XLeRobot 本地开发环境设置
# ============================================================
# 用法:
#   bash setup_local.sh
#
# 说明:
#   创建软链接，使 client.py 能导入 XLeRobot 的 XLerobot 类
#   (from lerobot.robots.xlerobot import XLerobot)
# ============================================================

set -euo pipefail

# 找到 LeRobot 包的安装路径
LEROBOT_DIR=$(python3 -c "import lerobot; import os; print(os.path.dirname(lerobot.__file__))" 2>/dev/null)

if [ -z "$LEROBOT_DIR" ]; then
    echo "❌ 找不到 lerobot 包，请先: pip install lerobot[xvla]"
    exit 1
fi

ROBOTS_DIR="$LEROBOT_DIR/robots"
XLEROBOT_SRC="/home/zach/XLeRobot/software/src/robots/xlerobot"
XLEROBOT_LINK="$ROBOTS_DIR/xlerobot"

# 检查源目录是否存在
if [ ! -d "$XLEROBOT_SRC" ]; then
    echo "❌ 找不到 XLeRobot 的 xlerobot 模块: $XLEROBOT_SRC"
    echo "   请确保在 XLeRobot 项目目录下运行此脚本"
    exit 1
fi

# 创建软链接
if [ -L "$XLEROBOT_LINK" ]; then
    echo "✅ 软链接已存在: $XLEROBOT_LINK"
elif [ -d "$XLEROBOT_LINK" ]; then
    echo "⚠️  目录已存在（非链接），跳过"
else
    ln -s "$XLEROBOT_SRC" "$XLEROBOT_LINK"
    echo "✅ 已创建软链接: $XLEROBOT_LINK → $XLEROBOT_SRC"
fi

# 验证
python3 -c "from lerobot.robots.xlerobot import XLerobot; print('✅ XLerobot 导入成功')"
