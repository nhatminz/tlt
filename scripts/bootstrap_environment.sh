#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
if [[ ! -x "$VENV_DIR/bin/python" ]];then "$PYTHON_BOOTSTRAP" -m venv "$VENV_DIR";fi
"$VENV_DIR/bin/python" -m pip install --upgrade pip
if [[ -n "${WHEELHOUSE:-}" ]];then
 "$VENV_DIR/bin/python" -m pip install --no-index --find-links "$WHEELHOUSE" -r "$ROOT/requirements.txt"
else
 "$VENV_DIR/bin/python" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
 "$VENV_DIR/bin/python" -m pip install -r "$ROOT/requirements.txt"
fi
printf 'Activate: source %q/bin/activate\n' "$VENV_DIR"
