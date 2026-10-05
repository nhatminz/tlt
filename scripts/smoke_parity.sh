#!/usr/bin/env bash
# Three real engines: fixed-strategy greedy identity diagnostic, NOT an adaptive speed benchmark.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARITY_DIR="${PARITY_DIR:-$ROOT/outputs/benchmarks/parity_$(date -u +%Y%m%dT%H%M%S_%N)}"
[[ ! -e "$PARITY_DIR" ]] || { echo 'ERROR: choose new PARITY_DIR' >&2;exit 2; }
export BATCH_SIZE="${BATCH_SIZE:-1}" RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-1}"
export BENCHMARK_PROMPTS="${BENCHMARK_PROMPTS:-2}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
export TEMPERATURE=0 REFLEX_PROFILE=0
METHOD=tlt OUTPUT_DIR="$PARITY_DIR/tlt" bash "$ROOT/run_benchmark.sh" --mab-configs '' "$@"
METHOD=tlt_reflex REFLEX_LR=0 OUTPUT_DIR="$PARITY_DIR/reflex_zero" bash "$ROOT/run_benchmark.sh" --mab-configs '' "$@"
"${PYTHON_BIN:-python}" "$ROOT/scripts/compare_upstream_outputs.py" "$PARITY_DIR/tlt/report.json" "$PARITY_DIR/reflex_zero/report.json"
METHOD=tlt_reflex REFLEX_LR=0.05 OUTPUT_DIR="$PARITY_DIR/reflex_active" bash "$ROOT/run_benchmark.sh" --mab-configs '' "$@"
