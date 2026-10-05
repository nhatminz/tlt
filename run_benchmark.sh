#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/launch_common.sh"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/benchmarks/${MODEL_KEY}_${METHOD}_$(date -u +%Y%m%dT%H%M%S_%N)}"
cmd=("$PYTHON_BIN" "$ROOT/benchmark.py" --method "$METHOD" --model "$MODEL" --draft "$DRAFT_EXPORT"
  --dataset "$DATASET_PATH" --output "$OUTPUT_DIR/report.json" --batch-size "$BATCH_SIZE"
  --responses "$RESPONSES_PER_PROMPT" --prompts "${BENCHMARK_PROMPTS:-16}"
  --max-new-tokens "$MAX_NEW_TOKENS" --max-prompt-length "$MAX_PROMPT_LENGTH"
  --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" --seed "$SEED" --tp "$TP_SIZE"
  --steps "$SPEC_STEPS" --draft-topk "$SPEC_TOPK" --tree-tokens "$SPEC_TREE_TOKENS"
  --sd-threshold "$SD_THRESHOLD" --mab "$MAB_ALGORITHM" --mab-configs "$MAB_CONFIGS" --mab-buckets "$MAB_BUCKETS"
  --attention-backend "$ATTENTION_BACKEND" --memory-fraction "${MEM_FRACTION:-0.6}")
if [[ "${DISABLE_CUDA_GRAPH:-0}" == 1 ]]; then cmd+=(--disable-cuda-graph);fi
if [[ "$REFLEX_PROFILE" == 1 ]]; then cmd+=(--profile);fi
cmd+=("$@")
printf 'Command:';printf ' %q' "${cmd[@]}";printf '\n'
if [[ "${DRY_RUN:-false}" == true ]];then "${cmd[@]}" --validate-config;exit 0;fi
prepare_runtime_draft
"${cmd[@]}"
