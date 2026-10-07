#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f .venv/bin/python ]; then
  echo "未找到环境。先运行 ./setup_env.sh" >&2
  exit 1
fi
if [ ! -f node_modules/katex/dist/katex.min.js ]; then
  echo "前端离线资源缺失。先运行 ./setup_env.sh" >&2
  exit 1
fi
exec .venv/bin/python -m app "$@"
