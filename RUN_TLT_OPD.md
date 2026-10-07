# TLT adaptive speculative rollout + fixed EAGLE3, with/without OPD

Only TltReflex is changed. Both modes retain native scheduler/KV/tree/verifier,
RNG and BEG/MAB at `bce3df7a4d46473912e9b81bf47bca419729557f`. Spot Trainer is
OFF; A is frozen and B adapts online. This is not full Spot-Trainer TLT.

## B200 setup

```bash
cd /workspace/storage-shared/nlp/minhpn19/TltReflex
source .venv/bin/activate                    # separate TLT Python3.12 / Torch2.8 cu128
export PYTHON_BIN="$(command -v python)"
export CUDA_VISIBLE_DEVICES=0
export SOURCE_SPECNAACL_ROOT="$(cd ../SpecNaacl && pwd)"
export SPECNAACL_PYTHON_BIN="$SOURCE_SPECNAACL_ROOT/.venv/bin/python"
export MODEL_KEY=qwen25_3b
export OPD_RANK=8 OPD_TOPK=16 OPD_FAST_LR=0.01 OPD_UPDATE_STREAM=1
export OPD_TRAIN_PROJECTOR=0 OPD_PROFILE=0 OPD_DEBUG=0
export OPD_MAX_SPECULATIVE_BATCH_SIZE=32
export OPD_PROPOSAL_MODE=auto
export OPD_PROPOSAL_PROFILE_DIR="$SOURCE_SPECNAACL_ROOT/outputs/benchmarks/opd_proposals"
unset OPD_PROPOSAL_PROFILE OPD_TUNE_OUTPUT

bash scripts/bootstrap_upstream.sh
"$PYTHON_BIN" scripts/validate_environment.py --rl
"$PYTHON_BIN" -m pytest -q
```

Bootstrap recognizes the previous OPD patch or legacy Fast-LK checkout by exact
hashes, saves it as a backup, fetches/checks the pin and applies the current patch.
It prefers `artifacts/fastrl-bce3df7.bundle` if present; otherwise fetches official
Git. `FASTRL_GIT_SOURCE=/path/to/bundle` supports offline machines. Unknown local
upstream changes are preserved and rejected. Runtime also audits patch identity,
so copying new plugin code while retaining an old verifier patch cannot silently
run the pre-truncation feedback bug.

Default paths match SpecNaacl:

- Target `/workspace/storage-shared/models/Qwen2.5-3B-Instruct`.
- Data `/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/train.parquet`.
- Draft `$SOURCE_SPECNAACL_ROOT/outputs/pretrain/qwen25_3b/latest_checkpoint`.
- Config `.../latest_draft_config.json`, mapping `.../latest_vocab_mapping.pt`.
- Common exported runtime checkpoint `outputs/draft_exports_opd/qwen25_3b`.

All 14 model wrappers remain. `MODEL`, `DATASET_PATH`, `DRAFT_CHECKPOINT`,
`DRAFT_CONFIG`, `VOCAB_MAPPING`, `DRAFT_EXPORT` can override paths. Use a NEW export
path when changing source weights/provenance; both modes use this same export.
Do not reinstall or modify SpecNaacl's environment to make SGLang import there.

## Projector provenance

Official comparison should use a SpecNaacl checkpoint whose A was actually trained
at the draft optimizer boundary. Export preserves existing A exactly and never
initializes over it. `trained` and `head_basis_initialized` are the only accepted
provenances. An arbitrary saved tensor without training/init evidence is rejected.
A generic old string saying "learned at optimizer boundary" is not sufficient
proof that the optimizer really updated A.

For a confirmed trained checkpoint, override DRAFT_CHECKPOINT to that file/dir.
If its old format lacks explicit provenance, an author-confirmed declaration can
be supplied with `OPD_PROJECTOR_PROVENANCE=trained` (exporter option
`--projector-provenance trained`). Do this only with actual training evidence.
The declaration does not train A. Config/mapping remain the ones matching those
base EAGLE3 weights; baseline and OPD still use the identical exported checkpoint.

If you deliberately test the default pretrain checkpoint without trained A:

```bash
export OPD_ALLOW_UNTRAINED_PROJECTOR=1
```

This opts into head-basis A explicitly and prints a warning. The report continues
to label A untrained. Without this opt-in, OPD fails rather than presenting an
untrained projector as a learned one. Baseline TLT ignores A/B at runtime.

## Real EAGLE3 representation validation

```bash
export OPD_EAGLE3_PARITY_REPORT="$PWD/outputs/validation/eagle3_${MODEL_KEY}.json"
bash scripts/validate_tlt_eagle3_parity.sh
```

The tool runs native SGLang EAGLE3 on the real checkpoint, records actual prefill
inputs/head operand/raw logits, shuts down that engine, and replays the same
inputs in SpecNaacl's separate Python environment. It compares exact head input,
raw compact logits, real OPD u and corrected Top16 with the same nonzero B fixture.
No extra transformer forward is injected into the native observation. This is an
offline diagnostic and is forbidden inside a throughput benchmark.

Report: max_abs_head_input_error, max_abs_logits_error, max_abs_u_error,
top16_agreement, probability error, explicit tolerances and passed flag. Defaults:
head0.02, logits0.05, u0.02, position-wise Top16 agreement1.0. A failed/missing
validation blocks OPD benchmarking. The certificate checks exact artifact hashes,
implementation/runtime/GPU and the read-only SpecNaacl source hashes. It cannot
be reused after changing weights, A or the representation implementation.
`--force` reruns validation. A cached valid report avoids repeating model loads.

The default source interpreter is `../SpecNaacl/.venv/bin/python`; set
SPECNAACL_PYTHON_BIN if that environment has another name. A tiny fixture or a
missing dependency/checkpoint cannot produce a passing production certificate.
Both target and source draft resources must exist on B200. The tool currently
requires target safetensors, TP1, one fresh prefill and the checked EAGLE3 variants.

## Native smoke and paired benchmark

```bash
# TLT, OPD LR=0, OPD LR=.01; each in a new engine.
bash scripts/smoke_tlt_opd.sh

# One LR/stream, multiple seeds and actual request batches1..32.
BENCH_SEEDS=42,43 BENCH_BATCH_SIZES=1,2,4,8,16,32 \
OPD_FAST_LRS=0.01 OPD_STREAMS=1 RESPONSES_PER_PROMPT=1 \
BENCHMARK_PROMPTS=64 MAX_NEW_TOKENS=2048 MAX_PROMPT_LENGTH=256 \
bash benchmark_tlt_opd_pair.sh

# Full sweep, same TLT config at each batch/seed.
OPD_FAST_LRS=0.001,0.01,0.05,0.1 OPD_STREAMS=0,1 BENCH_SEEDS=42,43 \
BENCH_BATCH_SIZES=1,2,4,8,16,32 RESPONSES_PER_PROMPT=1 \
BENCHMARK_PROMPTS=64 MAX_NEW_TOKENS=2048 MAX_PROMPT_LENGTH=256 \
bash sweep_tlt_opd_reflex.sh
```

run_benchmark.sh automatically checks/generates the representation certificate
for OPD if needed. Pair runs TLT and OPD sequentially and emits both reports plus
deltas; sweep emits report.json, summary.csv, responses.jsonl, fastest_observed.env.
Identity validation compares weights, actual tokenized prompts, measured samples,
seed/sampling, every TLT/MAB setting, batch, warmup and graphs. Profiling runs,
orphan nodes or invalid contexts cannot be recommended. An observed candidate
requires **verified AAL higher AND tokens/s strictly higher** than its matched
baseline. This does not establish statistical significance or guarantee wins on
new data/seeds. No end-to-end result for this implementation has been measured here.

BATCH_SIZE counts prompts; requests = BATCH_SIZE*RESPONSES_PER_PROMPT. Default
sweep responses1 realizes batches1/2/4/8/16/32; training responses8 matches
SpecNaacl. Native threshold32 is retained. Upstream can remain spec-enabled above
that threshold, so the plugin handles oversized batches through bounded chunks:
all chunks read frozen B_t, accumulate weighted gradients/weight, then apply B once.
No denominator per chunk or sequential adaptation is used. OPD_MAX_SPECULATIVE_BATCH_SIZE
controls feedback scratch capacity, not the scheduler or native tree. Raising it
reduces chunk overhead at the cost of memory. Persistent slot caches retain the
full pool. Logs/report distinguish persistent and scratch MB.

## Proposal profiles and separate profiling

Reuse your checked profile from SpecNaacl; no need to tune on benchmark prompts.
Auto follows the source cost interpolation/argmin over sparse/fused/GEMM, using
device active_count without host reads. Native TLT graph profiles take priority.
A source profile remains a calibration prior: wrapper/compiler differences may
change actual costs. A missing compatible profile prints an uncalibrated fallback.
Explicit incompatible profiles fail. Native offline tuner:

```bash
bash tune_tlt_opd_proposals.sh
# Include larger actual workloads if needed:
OPD_TUNE_SHAPES=1x1,2x1,4x1,8x1,16x1,32x1,8x4,32x4 \
bash tune_tlt_opd_proposals.sh
```

All three backends are measured and checked for bitwise proposal parity. Keys
include GPU/CC/V/r/dtype/TopK/kernel+version hash. No generation-time autotuning.
GEMM workspace is reserved before capture only when explicitly selected or when
calibrated auto regions can choose it. Root B_version/root_B_version avoid head
recomputation and projection when the cached proposal is fresh, including LR0 and
zero-gradient/invalid-teacher feedback. Deep contexts are still cleared for each
new tree. Cached proposals are gathered by stable slot IDs even after batch reorder.

```bash
COMPONENT_PROFILE=1 BATCH_SIZE=8 RESPONSES_PER_PROMPT=1 \
BENCHMARK_PROMPTS=16 MAX_NEW_TOKENS=512 bash benchmark_pair.sh
```

Throughput remains profile OFF. Eager component runs live in separate directories
and do not participate in comparison. Graph-inner event times are unavailable,
not zero. Wait overlaps side-stream work and is excluded from summed OPD section
work; wall time remains authoritative. Peak memory/context error counters include
startup/warmup; selected-state/coverage sums are differenced after warmup.

## Training

```bash
bash train_qwen25_3b_tlt.sh trainer.total_training_steps=2
bash train_qwen25_3b.sh trainer.total_training_steps=2
# Then separate full jobs:
bash train_qwen25_3b_tlt.sh
bash train_qwen25_3b.sh
```

Root-level run_tlt_fair.sh/run_tlt_opd_reflex.sh are equivalent wrappers. Both use
native FastRL GRPO with fixed pretrained EAGLE3, target LR1e-5, responses8, shared
sampling/data config. They do not fake EAGLE3 Spot Trainer. OPD_TRAIN_PROJECTOR=1
fails. Output remains outputs/rl/<model>_<method>_<timestamp>/. TP>1, overlap V2,
DP attention and quantized/scaled draft heads remain explicitly unsupported in OPD.
