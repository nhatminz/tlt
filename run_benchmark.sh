#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/launch_common.sh"
SMOKE_MODE="${BENCH_SMOKE:-0}"
for arg in "$@";do if [[ "$arg" == --smoke ]];then SMOKE_MODE=1;fi;done
if [[ "$SMOKE_MODE" == 1 ]];then export OPD_REQUIRE_CALIBRATED_PROFILE=0
else export OPD_REQUIRE_CALIBRATED_PROFILE=1;fi
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/benchmarks/${MODEL_KEY}_${METHOD}_$(date -u +%Y%m%dT%H%M%S_%N)}"
cmd=("$PYTHON_BIN" "$ROOT/benchmark.py" --method "$METHOD" --model "$MODEL" --draft "$DRAFT_EXPORT"
  --dataset "$DATASET_PATH" --output "$OUTPUT_DIR/report.json" --batch-size "$BATCH_SIZE"
  --responses "$RESPONSES_PER_PROMPT" --prompts "${BENCHMARK_PROMPTS:-16}"
  --max-new-tokens "$MAX_NEW_TOKENS" --max-prompt-length "$MAX_PROMPT_LENGTH"
  --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" --seed "$SEED" --tp "$TP_SIZE" --dp "${DP_SIZE:-1}"
  --steps "$SPEC_STEPS" --draft-topk "$SPEC_TOPK" --tree-tokens "$SPEC_TREE_TOKENS"
  --sd-threshold "$SD_THRESHOLD" --mab "$MAB_ALGORITHM" --mab-configs "$MAB_CONFIGS" --mab-buckets "$MAB_BUCKETS"
  --attention-backend "$ATTENTION_BACKEND" --memory-fraction "${MEM_FRACTION:-0.6}")
if [[ "${DISABLE_CUDA_GRAPH:-0}" == 1 ]]; then cmd+=(--disable-cuda-graph);fi
if [[ "$OPD_PROFILE" == 1 ]]; then cmd+=(--profile);fi
if [[ "${BENCH_SMOKE:-0}" == 1 ]];then cmd+=(--smoke);fi
cmd+=("$@")
printf 'Command:';printf ' %q' "${cmd[@]}";printf '\n'
if [[ -n "${CANONICAL_CONFIG_OUTPUT:-}" ]];then
  "${cmd[@]}" --dump-canonical-config "$CANONICAL_CONFIG_OUTPUT";exit 0
fi
if [[ "${DRY_RUN:-false}" == true ]];then "${cmd[@]}" --validate-config;exit 0;fi
prepare_runtime_draft
if [[ "$METHOD" == tlt_opd_reflex && "$SMOKE_MODE" != 1 ]];then
  export OPD_EAGLE3_PARITY_REPORT="${OPD_EAGLE3_PARITY_REPORT:-$ROOT/outputs/validation/eagle3_${MODEL_KEY}.json}"
  bash "$ROOT/scripts/validate_tlt_eagle3_parity.sh"
fi
"${cmd[@]}"
