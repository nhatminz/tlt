#!/usr/bin/env bash
# Only install into an explicitly selected NEW TLT venv, not SpecNaacl's venv.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2]==(3,12), "Use Python3.12"; assert sys.prefix!=sys.base_prefix, "Activate a dedicated TLT venv"; assert "SpecNaacl" not in sys.prefix, "Never replace SpecNaacl environment"'
bash "$ROOT/scripts/bootstrap_upstream.sh"
wheel_args=()
if [[ "${OFFLINE:-0}" == 1 ]];then
  : "${WHEELHOUSE:?Set prepared wheelhouse path}"
  wheel_args=(--no-index --find-links "$WHEELHOUSE")
fi
"$PYTHON_BIN" -m pip install "${wheel_args[@]}" setuptools==82.0.0 wheel==0.45.1 packaging==26.0 apache-tvm-ffi==0.1.0
mkdir -p "$ROOT/wheels"
if [[ ! -f "$ROOT/wheels/flashinfer_python-0.4.0-py3-none-any.whl" ]];then
  if [[ "${OFFLINE:-0}" == 1 && -f "$WHEELHOUSE/flashinfer_python-0.4.0-py3-none-any.whl" ]];then
    cp "$WHEELHOUSE/flashinfer_python-0.4.0-py3-none-any.whl" "$ROOT/wheels/"
  else
    bash "$ROOT/scripts/bootstrap_flashinfer.sh"
    "$PYTHON_BIN" -m pip wheel --no-deps --no-build-isolation --wheel-dir "$ROOT/wheels" "$ROOT/upstream/flashinfer"
  fi
fi
"$PYTHON_BIN" "$ROOT/scripts/audit_upstream.py" --require-wheel
"$PYTHON_BIN" -m pip install --no-deps "$ROOT/wheels/flashinfer_python-0.4.0-py3-none-any.whl"
"$PYTHON_BIN" -m pip install "${wheel_args[@]}" -r "$ROOT/requirements.txt"
"$PYTHON_BIN" -m pip install --no-deps --no-build-isolation -e "$ROOT/upstream/fastrl/third-party/sglang/python"
"$PYTHON_BIN" -m pip install --no-deps --no-build-isolation -e "$ROOT/upstream/fastrl"
if [[ "${INSTALL_RL:-0}" == 1 ]];then
  if [[ "${OFFLINE:-0}" == 1 ]];then
    "$PYTHON_BIN" -m pip install "${wheel_args[@]}" --no-deps flash-attn==2.8.3
  else
    "$PYTHON_BIN" -m pip install --no-deps -r "$ROOT/requirements-rl.txt"
  fi
fi
"$PYTHON_BIN" -m pip check
