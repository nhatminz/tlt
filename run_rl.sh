#!/usr/bin/env bash
# Official FastRL RL pipeline; does NOT use SpecNaacl's GRPO implementation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/launch_common.sh"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/rl/${MODEL_KEY}_${METHOD}_$(date -u +%Y%m%dT%H%M%S_%N)}"
RL_DATA="${RL_DATA:-$RUN_DIR/data/train.parquet}"
EVAL_DATA="${EVAL_DATA:-$RL_DATA}"
NGPUS="${NGPUS:-1}"
if [[ "${ENABLE_DRAFTER_TRAINING:-false}" != false ]]; then
  echo 'ERROR: upstream background trainer factory is EAGLE1, not EAGLE3. No silent replacement; keep upstream default false.' >&2;exit 2
fi
cmd=("$PYTHON_BIN" "$ROOT/rl.py" --method "$METHOD"
  speculative.enable=true speculative.spec_strategy=EAGLE3
  "speculative.eagle.spec_model_path=$DRAFT_EXPORT" "speculative.bs_threshold=$SD_THRESHOLD"
  "speculative.eagle.spec_steps=$SPEC_STEPS" "speculative.eagle.spec_topk=$SPEC_TOPK"
  "speculative.eagle.spec_verify_tokens=$SPEC_TREE_TOKENS" "speculative.eagle.tune_algorithm=$MAB_ALGORITHM"
  "speculative.eagle.mab_configs=[$MAB_CONFIGS]" "speculative.eagle.mab_bs_threshold=[$MAB_BUCKETS]"
  speculative.train.enable_drafter_training=false
  "data.train_files=$RL_DATA" "data.val_files=$EVAL_DATA" data.return_raw_chat=true data.return_full_prompt=true
  "data.train_batch_size=${RL_BATCH_SIZE:-64}" "data.max_prompt_length=$MAX_PROMPT_LENGTH"
  "data.max_response_length=$MAX_NEW_TOKENS" data.filter_overlong_prompts=true data.truncation=error
  "actor_rollout_ref.model.path=$MODEL" actor_rollout_ref.actor.strategy=fsdp2
  "actor_rollout_ref.actor.optim.lr=${TARGET_LR:-1e-6}" actor_rollout_ref.model.use_remove_padding=true
  "actor_rollout_ref.actor.ppo_mini_batch_size=${RL_MINI_BATCH_SIZE:-4}"
  actor_rollout_ref.actor.use_dynamic_bsz=true actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH+MAX_NEW_TOKENS))"
  "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH+MAX_NEW_TOKENS))"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH+MAX_NEW_TOKENS))"
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 actor_rollout_ref.ref.ulysses_sequence_parallel_size=1
  actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.actor.fsdp_config.param_offload=true actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE" actor_rollout_ref.rollout.name=sglang
  actor_rollout_ref.rollout.mode=sync actor_rollout_ref.rollout.multi_turn.format=hermes
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_MEM_FRACTION:-0.4}"
  "actor_rollout_ref.rollout.temperature=$TEMPERATURE" "actor_rollout_ref.rollout.top_p=$TOP_P"
  "actor_rollout_ref.rollout.top_k=$TOP_K" "actor_rollout_ref.rollout.n=$RESPONSES_PER_PROMPT"
  "actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH+MAX_NEW_TOKENS))"
  "actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=$ATTENTION_BACKEND"
  "+actor_rollout_ref.rollout.engine_kwargs.sglang.random_seed=$SEED"
  actor_rollout_ref.ref.fsdp_config.param_offload=true algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=false trainer.critic_warmup=0 "trainer.logger=[console]"
  trainer.project_name=TltReflex "trainer.experiment_name=${MODEL_KEY}_${METHOD}"
  "trainer.default_local_dir=$RUN_DIR/checkpoints" trainer.val_before_train=false
  "trainer.n_gpus_per_node=$NGPUS" trainer.nnodes=1 "trainer.save_freq=${SAVE_FREQ:-30}"
  trainer.test_freq=-1 "trainer.total_epochs=${NUM_EPOCHS:-1}" "+data.seed=$SEED")
cmd+=("$@")
printf 'Command:';printf ' %q' "${cmd[@]}";printf '\n'
if [[ "${DRY_RUN:-false}" == true ]];then "${cmd[@]}" --validate-config;exit 0;fi
[[ ! -e "$RUN_DIR" ]] || { echo 'ERROR: use a NEW RUN_DIR; upstream resume overrides must be explicit' >&2;exit 2; }
prepare_runtime_draft
"$PYTHON_BIN" "$ROOT/scripts/validate_environment.py" --rl
if [[ ! -f "$RL_DATA" ]];then
  "$PYTHON_BIN" "$ROOT/scripts/prepare_rl_data.py" --input "$DATASET_PATH" --output "$RL_DATA"
fi
mkdir -p "$RUN_DIR/logs"
"${cmd[@]}" 2>&1 | tee "$RUN_DIR/logs/console.log"
