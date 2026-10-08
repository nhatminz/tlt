#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BENCH_METHOD="${METHOD:-tlt}"
exec bash "$ROOT/scripts/sweep_tlt_opd_reflex.sh" "$@"
