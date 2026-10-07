#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmd=("${PYTHON_BIN:-python3}" "$ROOT/scripts/tune_tlt_opd_proposals.py"
  --models "${OPD_TUNE_MODELS:-${MODEL_KEY:-qwen25_3b}}"
  --pretrain-root "${PRETRAIN_ROOT:-$ROOT/../SpecNaacl/outputs/pretrain}"
  --profile-dir "${TLT_OPD_PROFILE_DIR:-${OPD_PROPOSAL_PROFILE_DIR:-$ROOT/outputs/benchmarks/opd_proposals}}"
  --dtype "${OPD_TUNE_DTYPE:-bf16}" --topk "${OPD_TOPK:-16}"
  --iterations "${OPD_TUNE_ITERATIONS:-30}" --seed "${OPD_TUNE_SEED:-42}")
for pair in OPD_TUNE_OUTPUT:output OPD_TUNE_SHAPES:shapes OPD_TUNE_SLOTS:slots OPD_RANK:rank DRAFT_CONFIG:draft-config DRAFT_CHECKPOINT:draft-checkpoint VOCAB_MAPPING:vocab-mapping; do
  name="${pair%%:*}";flag="${pair#*:}"
  if [[ -n "${!name:-}" ]];then cmd+=("--$flag" "${!name}");fi
done
exec "${cmd[@]}" "$@"
