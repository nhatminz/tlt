#!/usr/bin/env bash
# REAL two-engine smoke; needs configured local target/draft/data and CUDA stack.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BATCH_SIZE="${BATCH_SIZE:-1}" RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-1}"
export BENCHMARK_PROMPTS="${BENCHMARK_PROMPTS:-2}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
bash "$ROOT/benchmark_pair.sh" "$@"
