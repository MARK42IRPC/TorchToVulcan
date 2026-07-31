#!/usr/bin/env sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
VENV_PATH="$PROJECT_ROOT/.venv"

command -v python3 >/dev/null 2>&1 || {
  echo "Python 3.11 or later is required." >&2
  exit 1
}
command -v npm >/dev/null 2>&1 || {
  echo "Node.js 20 or later is required." >&2
  exit 1
}

if [ ! -x "$VENV_PATH/bin/python" ]; then
  echo "[setup] Creating Python virtual environment..."
  python3 -m venv "$VENV_PATH"
fi

echo "[setup] Installing Python dependencies..."
"$VENV_PATH/bin/python" -m pip install --upgrade pip
"$VENV_PATH/bin/python" -m pip install -e "$PROJECT_ROOT[dev,web]"

echo "[setup] Installing WebUI dependencies..."
cd "$PROJECT_ROOT/web"
if [ -f package-lock.json ]; then
  npm ci
else
  npm install
fi

echo "[setup] Dependencies are ready."
echo "[setup] Start development with: ./scripts/dev.sh"
