#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAIR_DIR="${PAIR_DIR:-$ROOT/outputs/benchmarks/paired_$(date -u +%Y%m%dT%H%M%S_%N)}"
[[ ! -e "$PAIR_DIR" ]] || { echo 'ERROR: choose NEW PAIR_DIR' >&2;exit 2; }
for mode in tlt tlt_opd_reflex; do
  METHOD="$mode" OUTPUT_DIR="$PAIR_DIR/$mode" OPD_PROFILE=0 bash "$ROOT/run_benchmark.sh" "$@"
done
if [[ "${COMPONENT_PROFILE:-0}" == 1 ]];then
  for mode in tlt tlt_opd_reflex; do
    METHOD="$mode" OUTPUT_DIR="$PAIR_DIR/${mode}_components_eager" DISABLE_CUDA_GRAPH=1 OPD_PROFILE=1 \
      bash "$ROOT/run_benchmark.sh" "$@"
  done
fi
