#!/usr/bin/env sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON_PATH="$PROJECT_ROOT/.venv/bin/python"
API_PORT=${API_PORT:-8000}
WEB_PORT=${WEB_PORT:-5173}

if [ ! -x "$PYTHON_PATH" ]; then
  echo "Dependencies are not installed. Run ./scripts/setup.sh first." >&2
  exit 1
fi

"$PYTHON_PATH" -m uvicorn torch_to_vulcan.api:app \
  --host 127.0.0.1 --port "$API_PORT" --reload &
BACKEND_PID=$!
trap 'kill "$BACKEND_PID" 2>/dev/null || true' EXIT INT TERM

cd "$PROJECT_ROOT/web"
npm run dev -- --host 127.0.0.1 --port "$WEB_PORT"
