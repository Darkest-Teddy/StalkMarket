#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$ROOT_DIR/main/Code"

if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.venv/bin/activate"
fi

trap 'kill 0' INT TERM EXIT

echo "[start_dev] Booting FastAPI backend..."
(
  cd "$CODE_DIR"
  python -m uvicorn main:api --reload --port 8000
) &

sleep 1

echo "[start_dev] Serving frontend on http://127.0.0.1:5173 (Ctrl+C to stop)"
(
  cd "$CODE_DIR"
  python -m http.server 5173
) &

wait
