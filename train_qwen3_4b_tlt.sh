#!/usr/bin/env bash
set -euo pipefail
export MODEL_KEY="qwen3_4b"
export METHOD="tlt"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/launch/train_model.sh" "$@"
