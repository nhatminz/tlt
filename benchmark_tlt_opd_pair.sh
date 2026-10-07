#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OPD_FAST_LRS="${OPD_FAST_LRS:-0.01}"
export OPD_STREAMS="${OPD_STREAMS:-1}"
export BENCH_SEEDS="${BENCH_SEEDS:-42}"
exec bash "$ROOT/scripts/sweep_tlt_opd_reflex.sh" "$@"
