#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD="${METHOD:-tlt}"
case "$METHOD" in tlt|tlt_opd_reflex) ;; *) echo 'ERROR: METHOD must be tlt or tlt_opd_reflex' >&2; exit 2;; esac
MODEL_KEY="${MODEL_KEY:-qwen25_3b}"
source "$PROJECT_DIR/configs/$MODEL_KEY/b200.env"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/storage-shared/nlp/minhpn19/data}"
DATASET="${DATASET:-simplelr}"
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
DRAFT_EXPORT="${DRAFT_EXPORT:-$PROJECT_DIR/outputs/draft_exports_opd/$MODEL_KEY}"
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
OPD_RANK="${OPD_RANK:-8}"
OPD_TOPK="${OPD_TOPK:-16}"
OPD_FAST_LR="${OPD_FAST_LR:-0.01}"
OPD_VISITED_WEIGHT="${OPD_VISITED_WEIGHT:-1.0}"
OPD_FRONTIER_WEIGHT="${OPD_FRONTIER_WEIGHT:-1.0}"
OPD_UPDATE_STREAM="${OPD_UPDATE_STREAM:-1}"
OPD_PROFILE="${OPD_PROFILE:-0}"
OPD_TRAIN_PROJECTOR="${OPD_TRAIN_PROJECTOR:-0}"
OPD_PROPOSAL_MODE="${OPD_PROPOSAL_MODE:-auto}"
OPD_DENSE_IMPLEMENTATION="${OPD_DENSE_IMPLEMENTATION:-auto}"
OPD_PROPOSAL_PROFILE_DIR="${OPD_PROPOSAL_PROFILE_DIR:-$SOURCE_SPECNAACL_ROOT/outputs/benchmarks/opd_proposals}"
export CUDA_VISIBLE_DEVICES METHOD TLT_REFLEX_METHOD="$METHOD"
export OPD_RANK OPD_TOPK OPD_FAST_LR OPD_VISITED_WEIGHT OPD_FRONTIER_WEIGHT
export OPD_ALLOW_UNTRAINED_PROJECTOR="${OPD_ALLOW_UNTRAINED_PROJECTOR:-0}"
export OPD_UPDATE_STREAM OPD_PROFILE OPD_TRAIN_PROJECTOR OPD_PROPOSAL_MODE OPD_DENSE_IMPLEMENTATION OPD_PROPOSAL_PROFILE_DIR
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/upstream/fastrl/third-party/sglang/python:$PROJECT_DIR/upstream/fastrl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

prepare_runtime_draft() {
  if [[ "${DRY_RUN:-false}" == true ]]; then return; fi
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/validate_environment.py"
  for path in "$MODEL/config.json" "$DRAFT_CONFIG" "$VOCAB_MAPPING" "$DATASET_PATH"; do
    [[ -f "$path" ]] || { echo "ERROR: missing required configured path: $path" >&2; exit 2; }
  done
  [[ -e "$DRAFT_CHECKPOINT" ]] || { echo "ERROR: missing pretrained draft: $DRAFT_CHECKPOINT" >&2; exit 2; }
  export_args=(--checkpoint "$DRAFT_CHECKPOINT" --config "$DRAFT_CONFIG" --mapping "$VOCAB_MAPPING" --target "$MODEL" --output "$DRAFT_EXPORT")
  if [[ -n "${OPD_PROJECTOR_PROVENANCE:-}" ]];then export_args+=(--projector-provenance "$OPD_PROJECTOR_PROVENANCE");fi
  if [[ -n "${OPD_PROJECTOR_TRAINING_DATASET:-}" ]];then export_args+=(--projector-training-dataset "$OPD_PROJECTOR_TRAINING_DATASET");fi
  if [[ -n "${OPD_PROJECTOR_TRAINING_STEPS:-}" ]];then export_args+=(--projector-training-steps "$OPD_PROJECTOR_TRAINING_STEPS");fi
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/export_specforge_draft.py" "${export_args[@]}"
}
