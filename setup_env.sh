#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3}"
if ! "$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 10)' 2>/dev/null; then
  echo "需要 Python 3.10+；可用 PYTHON=/path/to/python ./setup_env.sh 指定。" >&2
  exit 1
fi

if [ ! -f .venv/bin/python ]; then
  echo "创建当前目录下的 .venv..."
  "$PYTHON_BIN" -m venv .venv
fi

echo "安装本地服务依赖..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'import torch, fastapi, uvicorn, markdown_it; print("环境就绪 · PyTorch", torch.__version__)'
if ! command -v npm >/dev/null 2>&1; then
  echo "还需要 Node.js / npm 以安装离线公式字体和图标。" >&2
  exit 1
fi
echo "安装前端离线资源..."
npm install --no-audit --no-fund
echo "运行 ./start.sh 打开学习工作台。"
