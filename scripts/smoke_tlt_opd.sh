#!/usr/bin/env bash
# Three native SGLang engines. Requires real model/data/checkpoint and TLT venv.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_DIR="${SMOKE_DIR:-$ROOT/outputs/benchmarks/smoke_opd_$(date -u +%Y%m%dT%H%M%S_%N)}"
export BATCH_SIZE="${BATCH_SIZE:-1}" RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-1}"
export BENCHMARK_PROMPTS="${BENCHMARK_PROMPTS:-2}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
export OPD_PROFILE=0 BENCH_SMOKE=1
METHOD=tlt OUTPUT_DIR="$SMOKE_DIR/tlt" bash "$ROOT/run_benchmark.sh" "$@"
METHOD=tlt_opd_reflex OPD_FAST_LR=0 OUTPUT_DIR="$SMOKE_DIR/opd_zero" bash "$ROOT/run_benchmark.sh" "$@"
METHOD=tlt_opd_reflex OPD_FAST_LR=0.01 OUTPUT_DIR="$SMOKE_DIR/opd_active" bash "$ROOT/run_benchmark.sh" "$@"
