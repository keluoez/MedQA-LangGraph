#!/usr/bin/env bash
# MedQA-LangGraph Demo 一键启动（Linux / macOS）
# 用法：bash scripts/start_demo.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 1. 后端 API（:8000）
if ! curl -s http://127.0.0.1:8000/api/health > /dev/null 2>&1; then
  echo "[1/3] Installing API dependencies..."
  pip install -q -r "$ROOT/api/requirements.txt"
  echo "[2/3] Starting API server on http://127.0.0.1:8000 ..."
  (cd "$ROOT" && nohup python -m uvicorn api.main:app --port 8000 > /tmp/medqa_api.log 2>&1 &)
  sleep 4
else
  echo "[skip] API already running on :8000"
fi

# 2. 前端（:5173）
if ! curl -s http://127.0.0.1:5173 > /dev/null 2>&1; then
  echo "[3/3] Starting frontend on http://localhost:5173 ..."
  cd "$ROOT/frontend"
  [ -d node_modules ] || npm install
  nohup npm run dev > /tmp/medqa_frontend.log 2>&1 &
  sleep 3
else
  echo "[skip] frontend already running on :5173"
fi

echo ""
echo "Demo ready: http://localhost:5173"
echo "  Tab1 数据集浏览  |  Tab2 多智能体流水线（mock 场景零密钥可演示）"
(command -v xdg-open > /dev/null && xdg-open http://localhost:5173) || (command -v open > /dev/null && open http://localhost:5173) || true
