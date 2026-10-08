#!/usr/bin/env bash
# Exact shared adaptive settings at each prompt batch size; optional eager timings.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRID_DIR="${GRID_DIR:-$ROOT/outputs/benchmarks/grid_$(date -u +%Y%m%dT%H%M%S_%N)}"
[[ ! -e "$GRID_DIR" ]] || { echo 'ERROR: choose new GRID_DIR' >&2;exit 2; }
for batch in ${BATCH_SIZES:-1 2 4 8 16 32};do
  BATCH_SIZE="$batch" BENCHMARK_PROMPTS="${BENCHMARK_PROMPTS:-64}" \
    PAIR_DIR="$GRID_DIR/batch_$batch" bash "$ROOT/benchmark_pair.sh" "$@"
done
