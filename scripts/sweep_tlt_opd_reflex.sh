#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWEEP_DIR="${SWEEP_DIR:-$ROOT/outputs/benchmarks/tlt_opd_${MODEL_KEY:-qwen25_3b}_$(date -u +%Y%m%dT%H%M%S_%N)}"
[[ ! -e "$SWEEP_DIR" ]] || { echo 'ERROR: choose new SWEEP_DIR' >&2;exit 2; }
IFS=, read -ra batches <<< "${BENCH_BATCH_SIZES:-1,2,4,8,16,32}"
IFS=, read -ra seeds <<< "${BENCH_SEEDS:-42,43}"
IFS=, read -ra lrs <<< "${OPD_FAST_LRS:-0.001,0.01,0.05,0.1}"
IFS=, read -ra streams <<< "${OPD_STREAMS:-0,1}"
export OPD_PROFILE=0
export RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-1}"
export BENCHMARK_PROMPTS="${BENCHMARK_PROMPTS:-64}"
for batch in "${batches[@]}";do
  for seed in "${seeds[@]}";do
    METHOD=tlt BATCH_SIZE="$batch" SEED="$seed" OUTPUT_DIR="$SWEEP_DIR/b${batch}_s${seed}_tlt" bash "$ROOT/run_benchmark.sh" "$@"
    for lr in "${lrs[@]}";do
      for stream in "${streams[@]}";do
        METHOD=tlt_opd_reflex BATCH_SIZE="$batch" SEED="$seed" OPD_FAST_LR="$lr" OPD_UPDATE_STREAM="$stream" \
          OUTPUT_DIR="$SWEEP_DIR/b${batch}_s${seed}_lr${lr}_stream${stream}" bash "$ROOT/run_benchmark.sh" "$@"
      done
    done
  done
done
if [[ "${DRY_RUN:-false}" != true ]];then
  "${PYTHON_BIN:-python3}" "$ROOT/scripts/summarize_tlt_opd.py" "$SWEEP_DIR"
fi
