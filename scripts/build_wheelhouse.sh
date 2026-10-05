#!/usr/bin/env bash
# Run on ONLINE Linux x86_64 + Python3.12, transfer wheelhouse with entire folder.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
WHEELHOUSE="${WHEELHOUSE:-$ROOT/wheelhouse}"
"$PYTHON_BIN" -c 'import sys,platform; assert sys.version_info[:2]==(3,12) and sys.prefix!=sys.base_prefix, "Use isolated Python3.12 builder venv"; assert sys.platform=="linux" and platform.machine()=="x86_64", "Build wheelhouse on Linux x86_64 to match B200 server"'
bash "$ROOT/scripts/bootstrap_upstream.sh"
bash "$ROOT/scripts/bootstrap_flashinfer.sh"
mkdir -p "$WHEELHOUSE"
"$PYTHON_BIN" -m pip install setuptools==82.0.0 wheel==0.45.1 packaging==26.0 apache-tvm-ffi==0.1.0
"$PYTHON_BIN" -m pip wheel --no-deps --no-build-isolation --wheel-dir "$WHEELHOUSE" "$ROOT/upstream/flashinfer"
mkdir -p "$ROOT/wheels"
cp "$WHEELHOUSE/flashinfer_python-0.4.0-py3-none-any.whl" "$ROOT/wheels/"
"$PYTHON_BIN" "$ROOT/scripts/audit_upstream.py" --require-wheel --require-flashinfer
"$PYTHON_BIN" -m pip wheel --find-links "$WHEELHOUSE" --wheel-dir "$WHEELHOUSE" -r "$ROOT/requirements.txt"
"$PYTHON_BIN" -m pip wheel --no-deps --no-build-isolation --wheel-dir "$WHEELHOUSE" "$ROOT/upstream/fastrl/third-party/sglang/python" "$ROOT/upstream/fastrl"
if [[ "${INSTALL_RL:-0}" == 1 ]];then
  "$PYTHON_BIN" -m pip download --no-deps --dest "$WHEELHOUSE" -r "$ROOT/requirements-rl.txt"
fi
# Editable offline installs still need build tools, even with build isolation OFF.
"$PYTHON_BIN" -m pip download --only-binary=:all: --dest "$WHEELHOUSE" setuptools==82.0.0 wheel==0.45.1 packaging==26.0
