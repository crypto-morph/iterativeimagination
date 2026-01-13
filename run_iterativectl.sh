#!/usr/bin/env bash
# Helper launcher that ensures iterativectl runs inside the expected venv.
# Usage: ./run_iterativectl.sh <iterativectl args>

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_VENV_DIR="$HOME/ComfyUI/venv"
VENV_DIR="${ITERATIVE_VENV:-$DEFAULT_VENV_DIR}"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

create_or_update_venv() {
    echo "[iterativectl] Creating virtualenv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    "$PIP_BIN" install --upgrade pip
    "$PIP_BIN" install -r "$REPO_DIR/requirements.txt"
}

if [[ ! -x "$PYTHON_BIN" ]]; then
    create_or_update_venv
fi

if [[ -n "${ITERATIVE_UPDATE_VENV:-}" ]]; then
    echo "[iterativectl] Updating virtualenv packages"
    "$PIP_BIN" install -r "$REPO_DIR/requirements.txt"
fi

exec "$PYTHON_BIN" "$REPO_DIR/iterativectl" "$@"
