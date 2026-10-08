#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_KEY="${MODEL_KEY:-qwen25_3b}"
export METHOD="${METHOD:-tlt}"
source "$ROOT/scripts/launch/pretrain_model.sh" "$@"
