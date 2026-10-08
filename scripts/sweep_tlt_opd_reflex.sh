#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_KEY="${MODEL_KEY:-qwen25_3b}"
export METHOD="${METHOD:-tlt_opd_reflex}"
source "$ROOT/configs/_shared/b200_common.env"
source "$ROOT/configs/$MODEL_KEY/b200.env"
case "${DATASET,,}" in
 simplelr|simplelr_abel_level3to5) DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/simplelr_abel_level3to5/train.parquet}" ;;
 gsm8k) DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/gsm8k/main/train-00000-of-00001.parquet}" ;;
 dapo) DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet}" ;;
 *) : "${DATASET_PATH:?set existing DATASET_PATH}" ;;
esac
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRAFT_CHECKPOINT="${DRAFT_CHECKPOINT:-$ROOT/../SpecNaacl/outputs/pretrain/$MODEL_KEY/latest_checkpoint}"
BENCH_OUTPUT="${BENCH_OUTPUT:-$ROOT/outputs/benchmarks/tlt_${MODEL_KEY}_$(date -u +%Y%m%dT%H%M%S_%N)}"
export OPD_PROPOSAL_PROFILE_DIR="${OPD_PROPOSAL_PROFILE_DIR:-$ROOT/outputs/benchmarks/opd_proposals}"
cmd=("$PYTHON_BIN" "$ROOT/scripts/benchmark_tlt_opd.py"
 --method "${BENCH_METHOD:-pair}" --target-model "$MODEL" --target-adapter "$TARGET_ADAPTER"
 --draft-checkpoint "$DRAFT_CHECKPOINT" --dataset-path "$DATASET_PATH" --output "$BENCH_OUTPUT"
 --batch-sizes "${BENCH_BATCH_SIZES:-$BATCH_SIZE}" --responses "$RESPONSES_PER_PROMPT"
 --seeds "${BENCH_SEEDS:-42,43}" --iterations "${BENCH_ITERATIONS:-2}" --warmup "${BENCH_WARMUP:-1}"
 --max-length "${BENCH_MAX_LENGTH:-512}" --max-prompt-length "${BENCH_MAX_PROMPT_LENGTH:-256}"
 --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "${TOP_K:-0}"
 --attn-implementation "$ATTENTION_IMPLEMENTATION" --dtype "$MODEL_DTYPE"
 --rank "$OPD_RANK" --topk "$OPD_TOPK" --fast-lrs "${OPD_FAST_LRS:-$OPD_FAST_LR}" --streams "${OPD_STREAMS:-$OPD_UPDATE_STREAM}"
 --visited-weight "$OPD_VISITED_WEIGHT" --frontier-weight "$OPD_FRONTIER_WEIGHT"
 --draft-lr "$DRAFT_LR" --draft-accumulation-steps "$DRAFT_ACCUMULATION_STEPS")
if [[ "${BENCH_ONLINE_DRAFT:-0}" == 1 ]];then cmd+=(--online-draft);fi
cmd+=("$@")
printf 'Output: %s\nCommand:' "$BENCH_OUTPUT";printf ' %q' "${cmd[@]}";printf '\n'
if [[ "${DRY_RUN:-false}" == true ]];then cmd+=(--dry-run);fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
exec "${cmd[@]}"
