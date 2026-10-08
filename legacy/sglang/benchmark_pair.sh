#!/usr/bin/env bash
# One fair case; the grid wrapper repeats this for batches/seeds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAIR_DIR="${PAIR_DIR:-$ROOT/outputs/benchmarks/paired_$(date -u +%Y%m%dT%H%M%S_%N)}"
[[ ! -e "$PAIR_DIR" ]] || { echo 'ERROR: choose NEW PAIR_DIR' >&2;exit 2; }
export OPD_REQUIRE_CALIBRATED_PROFILE=1
mkdir -p "$PAIR_DIR"
METHOD=tlt OPD_PROFILE=0 OUTPUT_DIR="$PAIR_DIR/tlt" \
  CANONICAL_CONFIG_OUTPUT="$PAIR_DIR/tlt/canonical_config.json" bash "$ROOT/run_benchmark.sh" "$@"
METHOD=tlt_opd_reflex OPD_PROFILE=0 OUTPUT_DIR="$PAIR_DIR/tlt_opd" \
  CANONICAL_CONFIG_OUTPUT="$PAIR_DIR/tlt_opd/canonical_config.json" bash "$ROOT/run_benchmark.sh" "$@"
"${PYTHON_BIN:-python3}" "$ROOT/scripts/check_tlt_opd_pair_config.py" \
  "$PAIR_DIR/tlt/canonical_config.json" "$PAIR_DIR/tlt_opd/canonical_config.json" --output "$PAIR_DIR/config_diff.json"
"${PYTHON_BIN:-python3}" "$ROOT/scripts/pair_order.py" "$PAIR_DIR/tlt/canonical_config.json" \
  --output "$PAIR_DIR/run_order.json" --entries-output "$PAIR_DIR/run_order.entries"
if [[ "${DRY_RUN:-false}" == true ]];then exit 0;fi
position=0
while IFS=: read -r mode folder;do
  position=$((position+1))
  METHOD="$mode" OUTPUT_DIR="$PAIR_DIR/$folder" OPD_PROFILE=0 BENCH_RUN_POSITION="$position" BENCH_RUN_ORDER_POLICY=seed_parity bash "$ROOT/run_benchmark.sh" "$@"
done < "$PAIR_DIR/run_order.entries"
"${PYTHON_BIN:-python3}" "$ROOT/scripts/summarize_tlt_opd.py" "$PAIR_DIR"
if [[ "${COMPONENT_PROFILE:-0}" == 1 ]];then
  while IFS=: read -r mode folder;do
    METHOD="$mode" OUTPUT_DIR="$PAIR_DIR/${mode}_components_eager" DISABLE_CUDA_GRAPH=1 OPD_PROFILE=1 \
      bash "$ROOT/run_benchmark.sh" "$@"
  done < "$PAIR_DIR/run_order.entries"
fi
