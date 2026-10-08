#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PRETRAIN_ROOT="${PRETRAIN_ROOT:-$ROOT/../SpecNaacl/outputs/pretrain}"
export MAX_DRAFT_K="${MAX_DRAFT_K:-4}"
exec bash "$ROOT/scripts/tune_opd_proposals.sh" "$@"
