#!/usr/bin/env bash
set -euo pipefail
export MODEL_KEY="qwen25_3b"
export METHOD="tlt"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/launch/train_model.sh" "$@"
