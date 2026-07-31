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

health() {
  curl --silent --fail --max-time 2 "http://127.0.0.1:$1/api/health" 2>/dev/null | \
    grep -q '"api_version":"0.2"'
}

port_in_use() {
  (command -v lsof >/dev/null 2>&1 && lsof -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1) || \
  (command -v ss >/dev/null 2>&1 && ss -ltn "sport = :$1" | grep -q LISTEN)
}

find_free_port() {
  port=$1
  while port_in_use "$port"; do
    port=$((port + 1))
  done
  echo "$port"
}

find_healthy_port() {
  port=$1
  end=$((port + 10))
  while [ "$port" -lt "$end" ]; do
    if health "$port"; then
      echo "$port"
      return 0
    fi
    port=$((port + 1))
  done
  return 1
}

BACKEND_PID=""
HEALTHY_API_PORT=$(find_healthy_port "$API_PORT" || true)
if [ -n "$HEALTHY_API_PORT" ]; then
  API_PORT=$HEALTHY_API_PORT
  echo "Reusing API: http://127.0.0.1:$API_PORT"
else
  if port_in_use "$API_PORT"; then
    API_PORT=$(find_free_port $((API_PORT + 1)))
    echo "API port was occupied. Using $API_PORT instead."
  fi
  "$PYTHON_PATH" -m uvicorn torch_to_vulcan.api:app \
    --host 127.0.0.1 --port "$API_PORT" --reload &
  BACKEND_PID=$!
  trap 'test -z "$BACKEND_PID" || kill "$BACKEND_PID" 2>/dev/null || true' EXIT INT TERM
fi

HEALTHY_WEB_PORT=$(find_healthy_port "$WEB_PORT" || true)
if [ -n "$HEALTHY_WEB_PORT" ]; then
  WEB_PORT=$HEALTHY_WEB_PORT
  echo "Reusing WebUI: http://127.0.0.1:$WEB_PORT"
  echo "Development services are already running."
  exit 0
fi

if port_in_use "$WEB_PORT"; then
  WEB_PORT=$(find_free_port $((WEB_PORT + 1)))
  echo "WebUI port was occupied. Using $WEB_PORT instead."
fi

cd "$PROJECT_ROOT/web"
TTV_API_PORT="$API_PORT" npm run dev -- --host 127.0.0.1 --port "$WEB_PORT"
