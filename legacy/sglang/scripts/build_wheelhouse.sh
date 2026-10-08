#!/usr/bin/env bash
# Run on ONLINE Linux x86_64 + Python3.12, transfer wheelhouse with entire folder.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
WHEELHOUSE="${WHEELHOUSE:-$ROOT/wheelhouse}"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-$ROOT/upstream}"
export UPSTREAM_ROOT
"$PYTHON_BIN" -c 'import sys,platform; assert sys.version_info[:2]==(3,12) and sys.prefix!=sys.base_prefix, "Use isolated Python3.12 builder venv"; assert sys.platform=="linux" and platform.machine()=="x86_64", "Build wheelhouse on Linux x86_64 to match B200 server"'
bash "$ROOT/scripts/bootstrap_upstream.sh"
bash "$ROOT/scripts/bootstrap_flashinfer.sh"
mkdir -p "$WHEELHOUSE"
"$PYTHON_BIN" -m pip install setuptools==82.0.0 wheel==0.45.1 packaging==26.0 apache-tvm-ffi==0.1.0
"$PYTHON_BIN" -m pip wheel --no-deps --no-build-isolation --wheel-dir "$WHEELHOUSE" "$UPSTREAM_ROOT/flashinfer"
mkdir -p "$ROOT/wheels"
cp "$WHEELHOUSE/flashinfer_python-0.4.0-py3-none-any.whl" "$ROOT/wheels/"
"$PYTHON_BIN" "$ROOT/scripts/audit_upstream.py" --upstream-root "$UPSTREAM_ROOT" --require-wheel --require-flashinfer
# Download exact public artifacts without resolving the BROKEN public
# FlashInfer beta metadata; custom FlashInfer was explicitly built above.
"$PYTHON_BIN" "$ROOT/scripts/download_offline_wheels.py" --output "$WHEELHOUSE"
"$PYTHON_BIN" - "$WHEELHOUSE" "$ROOT/artifacts/offline-sdists" <<'PY'
import json,pathlib,subprocess,sys
directory,sdists=map(pathlib.Path,sys.argv[1:])
receipt=json.loads((directory/'download_receipt.json').read_text())
for r in receipt['records']:
    if r['kind']=='sdist':
        subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation',
                        '--wheel-dir',str(directory),str(sdists/r['filename'])],check=True)
PY
"$PYTHON_BIN" -m pip wheel --no-deps --no-build-isolation --wheel-dir "$WHEELHOUSE" "$UPSTREAM_ROOT/fastrl/third-party/sglang/python" "$UPSTREAM_ROOT/fastrl"
if [[ "${INSTALL_RL:-0}" == 1 ]];then
  # Official asset filename embeds build traits in the version, but its
  # METADATA version is2.8.3. pip26 rejects that mismatch in --find-links.
  # Rename only; preserve the official wheel bytes / CUDA ABI unchanged.
  curl --fail --location --retry 2 --max-time 900 \
    'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl' \
    --output "$WHEELHOUSE/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
fi
# Editable offline installs still need build tools, even with build isolation OFF.
"$PYTHON_BIN" -m pip download --only-binary=:all: --dest "$WHEELHOUSE" setuptools==82.0.0 wheel==0.45.1 packaging==26.0
manifest_args=(--wheelhouse "$WHEELHOUSE" --create)
if [[ "${INSTALL_RL:-0}" == 1 ]];then manifest_args+=(--rl);fi
"$PYTHON_BIN" "$ROOT/scripts/wheelhouse_manifest.py" "${manifest_args[@]}"
