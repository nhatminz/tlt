#!/usr/bin/env bash
# Install prepared wheels only: no indexes, no source builds, no network git clone.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
WHEELHOUSE="${WHEELHOUSE:-$ROOT/wheelhouse}"
INSTALL_RL="${INSTALL_RL:-1}"
"$PYTHON_BIN" -c 'import sys,platform; assert sys.version_info[:2]==(3,12), "Use Python3.12"; assert platform.machine()=="x86_64", "Linux x86_64 wheels required"; assert sys.prefix!=sys.base_prefix, "Activate dedicated TLT venv"; assert "SpecNaacl" not in sys.prefix, "Do not replace SpecNaacl venv"'
export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
# Python's initial ensurepip provides packaging only through pip's vendored copy.
# Install tools FIRST entirely from wheelhouse, then verify the full manifest.
"$PYTHON_BIN" -m pip install --no-index --find-links "$WHEELHOUSE" --no-deps pip==26.2.1 packaging==26.0 wheel==0.45.1 setuptools==82.0.0
verify_args=(--wheelhouse "$WHEELHOUSE")
if [[ "$INSTALL_RL" == 1 ]];then verify_args+=(--rl);fi
"$PYTHON_BIN" "$ROOT/scripts/wheelhouse_manifest.py" "${verify_args[@]}"
"$PYTHON_BIN" -m pip install --no-index --find-links "$WHEELHOUSE" -r "$ROOT/requirements.txt"
# Use the locally built TLT fork wheels, never download stock SGLang or extras.
"$PYTHON_BIN" -m pip install --no-index --find-links "$WHEELHOUSE" --no-deps sglang==0.5.3.post2 verl==0.5.0.dev0
if [[ "$INSTALL_RL" == 1 ]];then
  "$PYTHON_BIN" -m pip install --no-index --find-links "$WHEELHOUSE" --no-deps flash-attn==2.8.3
fi
mkdir -p "$ROOT/wheels"
if [[ ! -f "$ROOT/wheels/flashinfer_python-0.4.0-py3-none-any.whl" ]];then
  cp "$WHEELHOUSE/flashinfer_python-0.4.0-py3-none-any.whl" "$ROOT/wheels/"
fi
if [[ ! -f "$ROOT/upstream/fastrl/verl/trainer/main_fastrl.py" ]];then
  bundle="${FASTRL_GIT_SOURCE:-$ROOT/artifacts/fastrl-bce3df7.bundle}"
  [[ -f "$bundle" || -d "$bundle" ]] || { echo 'ERROR: provide local FASTRL_GIT_SOURCE bundle/mirror; network clone is forbidden' >&2;exit 2; }
  FASTRL_GIT_SOURCE="$bundle" bash "$ROOT/scripts/bootstrap_upstream.sh"
fi
# Legacy extracted source can be retained only if it matches all pinned/patch hashes.
"$PYTHON_BIN" "$ROOT/scripts/audit_upstream.py" --require-wheel
"$PYTHON_BIN" -m pip check
if [[ "${VALIDATE_NATIVE:-0}" == 1 ]];then
  native_args=();if [[ "$INSTALL_RL" == 1 ]];then native_args+=(--rl);fi
  "$PYTHON_BIN" "$ROOT/scripts/validate_environment.py" "${native_args[@]}"
fi
echo 'Offline wheel installation/pip check complete. Run validate_environment.py --rl on B200 before training.'
