#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD="${METHOD:-tlt}"
case "$METHOD" in tlt|tlt_reflex) ;; *) echo 'ERROR: METHOD must be tlt or tlt_reflex' >&2; exit 2;; esac
MODEL_KEY="${MODEL_KEY:-qwen25_3b}"
source "$PROJECT_DIR/configs/$MODEL_KEY/b200.env"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/storage-shared/nlp/minhpn19/data}"
DATASET="${DATASET:-dapo}"
case "$DATASET" in
  dapo) DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet}" ;;
  simplelr) DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/simplelr_abel_level3to5/train.parquet}" ;;
  gsm8k) DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/gsm8k/main/train-00000-of-00001.parquet}" ;;
  *) : "${DATASET_PATH:?Set real local DATASET_PATH}" ;;
esac
SOURCE_SPECNAACL_ROOT="${SOURCE_SPECNAACL_ROOT:-$PROJECT_DIR/../SpecNaacl}"
PRETRAIN_MODEL_ROOT="${PRETRAIN_MODEL_ROOT:-$SOURCE_SPECNAACL_ROOT/outputs/pretrain/$MODEL_KEY}"
DRAFT_CHECKPOINT="${DRAFT_CHECKPOINT:-$PRETRAIN_MODEL_ROOT/latest_checkpoint}"
DRAFT_CONFIG="${DRAFT_CONFIG:-$PRETRAIN_MODEL_ROOT/latest_draft_config.json}"
VOCAB_MAPPING="${VOCAB_MAPPING:-$PRETRAIN_MODEL_ROOT/latest_vocab_mapping.pt}"
# Existing pretrained source is unchanged. Export is an explicit format artifact.
DRAFT_EXPORT="${DRAFT_EXPORT:-$PROJECT_DIR/outputs/draft_exports/$MODEL_KEY}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:--1}"
SEED="${SEED:-42}"
TP_SIZE="${TP_SIZE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SPEC_STEPS="${SPEC_STEPS:-8}"
SPEC_TOPK="${SPEC_TOPK:-4}"
SPEC_TREE_TOKENS="${SPEC_TREE_TOKENS:-48}"
SD_THRESHOLD="${SD_THRESHOLD:-32}"
MAB_ALGORITHM="${MAB_ALGORITHM:-BEG}"
MAB_CONFIGS="${MAB_CONFIGS:-8_4_32,8_4_16,8_4_8}"
MAB_BUCKETS="${MAB_BUCKETS:-1,2,5,21}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
REFLEX_FEATURE_DIM="${REFLEX_FEATURE_DIM:-8}"
REFLEX_LR="${REFLEX_LR:-0.05}"
REFLEX_WEIGHT_DECAY="${REFLEX_WEIGHT_DECAY:-0}"
REFLEX_SEED="${REFLEX_SEED:-$SEED}"
REFLEX_PROFILE="${REFLEX_PROFILE:-0}"
export CUDA_VISIBLE_DEVICES METHOD TLT_REFLEX_METHOD="$METHOD"
export REFLEX_FEATURE_DIM REFLEX_LR REFLEX_WEIGHT_DECAY REFLEX_SEED REFLEX_PROFILE
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/upstream/fastrl/third-party/sglang/python:$PROJECT_DIR/upstream/fastrl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

prepare_runtime_draft() {
  if [[ "${DRY_RUN:-false}" == true ]]; then return; fi
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/validate_environment.py"
  for path in "$MODEL/config.json" "$DRAFT_CONFIG" "$VOCAB_MAPPING" "$DATASET_PATH"; do
    [[ -f "$path" ]] || { echo "ERROR: missing required configured path: $path" >&2; exit 2; }
  done
  [[ -e "$DRAFT_CHECKPOINT" ]] || { echo "ERROR: missing pretrained draft: $DRAFT_CHECKPOINT" >&2; exit 2; }
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/export_specforge_draft.py" --checkpoint "$DRAFT_CHECKPOINT" \
    --config "$DRAFT_CONFIG" --mapping "$VOCAB_MAPPING" --target "$MODEL" --output "$DRAFT_EXPORT"
}
