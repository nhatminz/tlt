#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/launch_common.sh"
prepare_runtime_draft
report="${OPD_EAGLE3_PARITY_REPORT:-$ROOT/outputs/validation/eagle3_${MODEL_KEY}.json}"
cmd=("$PYTHON_BIN" "$ROOT/scripts/validate_tlt_eagle3_parity.py"
  --target "$MODEL" --draft "$DRAFT_EXPORT" --source-checkpoint "$DRAFT_CHECKPOINT"
  --source-config "$DRAFT_CONFIG" --mapping "$VOCAB_MAPPING" --source-root "$SOURCE_SPECNAACL_ROOT"
  --source-python "${SPECNAACL_PYTHON_BIN:-$SOURCE_SPECNAACL_ROOT/.venv/bin/python}" --report "$report")
if [[ "$OPD_ALLOW_UNTRAINED_PROJECTOR" == 1 ]];then cmd+=(--allow-untrained-projector);fi
exec "${cmd[@]}" "$@"
