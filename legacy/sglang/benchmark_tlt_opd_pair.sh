#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAIR_ROOT="${PAIR_DIR:-$ROOT/outputs/benchmarks/tlt_opd_pair_$(date -u +%Y%m%dT%H%M%S_%N)}"
[[ ! -e "$PAIR_ROOT" ]] || { echo 'ERROR: choose NEW PAIR_DIR' >&2;exit 2; }
IFS=, read -ra batches <<< "${BENCH_BATCH_SIZES:-1,2,4,8,16,32}"
IFS=, read -ra seeds <<< "${BENCH_SEEDS:-42,43}"
export RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-1}"
export BENCHMARK_PROMPTS="${BENCHMARK_PROMPTS:-64}"
for batch in "${batches[@]}";do
  for seed in "${seeds[@]}";do
    BATCH_SIZE="$batch" SEED="$seed" PAIR_DIR="$PAIR_ROOT/b${batch}_s${seed}" \
      bash "$ROOT/benchmark_pair.sh" "$@"
  done
done
if [[ "${DRY_RUN:-false}" != true ]];then
  "${PYTHON_BIN:-python3}" "$ROOT/scripts/summarize_tlt_opd.py" "$PAIR_ROOT"
fi
