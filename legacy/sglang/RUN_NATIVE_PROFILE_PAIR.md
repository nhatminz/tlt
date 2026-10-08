# B200: tune native profile, validate, smoke, then counterbalanced pair

Run from the updated TltReflex folder next to SpecNaacl. Use the TLT venv;
SpecNaacl remains read-only. Replace qwen25_3b with another existing model key.

## 1. Setup

```bash
cd /workspace/storage-shared/nlp/minhpn19/TltReflex
source .venv/bin/activate
export PYTHON_BIN="$(command -v python)"
export CUDA_VISIBLE_DEVICES=0 MODEL_KEY=qwen25_3b
export SOURCE_SPECNAACL_ROOT="$(cd ../SpecNaacl && pwd)"
export SPECNAACL_PYTHON_BIN="$SOURCE_SPECNAACL_ROOT/.venv/bin/python"
export OPD_RANK=8 OPD_TOPK=16 OPD_FAST_LR=0.01 OPD_UPDATE_STREAM=1
export OPD_PROPOSAL_MODE=auto OPD_PROFILE=0 OPD_TRAIN_PROJECTOR=0
export OPD_PROPOSAL_PROFILE_DIR="$PWD/outputs/benchmarks/opd_proposals"
unset OPD_PROPOSAL_PROFILE OPD_TUNE_OUTPUT
export DRAFT_CHECKPOINT="$SOURCE_SPECNAACL_ROOT/outputs/pretrain/$MODEL_KEY/latest_checkpoint"
export DRAFT_CONFIG="$SOURCE_SPECNAACL_ROOT/outputs/pretrain/$MODEL_KEY/latest_draft_config.json"
export VOCAB_MAPPING="$SOURCE_SPECNAACL_ROOT/outputs/pretrain/$MODEL_KEY/latest_vocab_mapping.pt"

bash scripts/bootstrap_upstream.sh
"$PYTHON_BIN" scripts/validate_environment.py
"$PYTHON_BIN" -m compileall -q tlt_reflex scripts benchmark.py rl.py
for file in *.sh scripts/*.sh; do bash -n "$file"; done
"$PYTHON_BIN" -m pytest -q
```

For the default pretrain checkpoint without trained A, explicitly allow the
head-basis experiment (the runtime warns and the report labels it untrained):

```bash
export OPD_ALLOW_UNTRAINED_PROJECTOR=1
```

For trained A, use the actual trained checkpoint for BOTH modes, keep matching
config/mapping, and record real training dataset/steps as explained in RUN_TLT_OPD.md.
Do not mark an initialized projector trained.

## 2. Tune on this GPU/server; verify native profile reload

```bash
export OPD_TUNE_SEED=42 OPD_TUNE_ITERATIONS=30
bash scripts/tune_tlt_opd_proposals.sh

"$PYTHON_BIN" scripts/validate_tlt_opd_profile.py \
  --draft-config "$DRAFT_CONFIG" --draft-checkpoint "$DRAFT_CHECKPOINT" \
  --vocab-mapping "$VOCAB_MAPPING" --rank "$OPD_RANK" --dtype bf16 --topk "$OPD_TOPK" \
  --profile-dir "$OPD_PROPOSAL_PROFILE_DIR"
```

The printed profile must say `tlt_native`, `validated: true`, and unique contexts.
The loader matches GPU/CC, vocab/rank/dtype, kernel/version and TLT execution hashes.
Old/native-contiguous or SpecNaacl profiles are not official calibration. No
profile tuning occurs during generation. Tuning output lives only in TltReflex.

## 3. Three native smoke modes

```bash
BATCH_SIZE=1 RESPONSES_PER_PROMPT=1 BENCHMARK_PROMPTS=2 MAX_NEW_TOKENS=64 \
  METHOD=tlt bash run_benchmark.sh --smoke --warmup 0

BATCH_SIZE=1 RESPONSES_PER_PROMPT=1 BENCHMARK_PROMPTS=2 MAX_NEW_TOKENS=64 \
  METHOD=tlt_opd_reflex OPD_FAST_LR=0 bash run_benchmark.sh --smoke --warmup 0

BATCH_SIZE=1 RESPONSES_PER_PROMPT=1 BENCHMARK_PROMPTS=2 MAX_NEW_TOKENS=64 \
  METHOD=tlt_opd_reflex OPD_FAST_LR=0.01 bash run_benchmark.sh --smoke --warmup 0
```

Each creates a new output folder. Smoke results cannot be official throughput
comparisons. Equivalently run `bash scripts/smoke_tlt_opd.sh`.

## 4. Real checkpoint representation parity

```bash
export OPD_EAGLE3_PARITY_REPORT="$PWD/outputs/validation/eagle3_${MODEL_KEY}.json"
bash scripts/validate_tlt_eagle3_parity.sh --force
```

`--force` regenerates the certificate after code/checkpoint changes. It requires
both environments and real target/draft assets. Failed parity blocks the pair.

## 5. Counterbalanced official pair

```bash
export OPD_REQUIRE_CALIBRATED_PROFILE=1
export OPD_FAST_LR=0.01 OPD_UPDATE_STREAM=1
BENCH_BATCH_SIZES=1,2,4,8,16,32 BENCH_SEEDS=42,43,44,45 \
RESPONSES_PER_PROMPT=1 BENCHMARK_PROMPTS=64 MAX_NEW_TOKENS=2048 \
MAX_PROMPT_LENGTH=256 bash benchmark_tlt_opd_pair.sh
```

Even seed: TLT -> OPD. Odd seed: OPD -> TLT. The same prompts/order/seed/sampling,
checkpoint, TLT/MAB, graphs, warmup and requested/measured responses are retained.
All non-OPD canonical differences fail before generation. Each b<batch>_s<seed>
contains run_order.json, canonical configs, tlt/report.json, tlt_opd/report.json,
comparison.json, summary.csv and config_diff.json. Root aggregation records run
orders and deltas; no performance win is inferred from AAL alone.

Configuration/order-only check without loading a model:

```bash
DRY_RUN=true BENCH_BATCH_SIZES=1 BENCH_SEEDS=42,43,44,45 \
bash benchmark_tlt_opd_pair.sh
```

This machine's RTX3090 V519 calibration is a synthetic kernel fixture, not a
production/B200 profile. Native smoke and B200 end-to-end performance remain to
be checked on the server with the required assets and pinned TLT stack.
