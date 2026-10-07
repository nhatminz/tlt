#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export METHOD=tlt
export TARGET_LR="${TARGET_LR:-1e-5}"
export MODEL_KEY="${MODEL_KEY:-qwen25_3b}"
exec bash "$ROOT/run_rl.sh" "$@"
