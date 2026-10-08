#!/usr/bin/env bash
set -euo pipefail
export MODEL_KEY="qwen3_1p7b"
export METHOD="tlt_opd_reflex"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/launch/train_model.sh" "$@"
