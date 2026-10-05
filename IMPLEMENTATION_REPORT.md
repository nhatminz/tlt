# Implementation / validation report

Date: 2026-10-05. New sibling folder `TltReflex/` only.
`git -C ../SpecNaacl status --short` remained empty: no SpecNaacl edits.

## Created / changed

- Full official FastRL archive + included official TLT SGLang fork,
  pinned commit `bce3df7a4d46473912e9b81bf47bca419729557f`.
- Pristine SGLang Python copy, `PROVENANCE.json`, `upstream_hashes.json`,
  `scripts/audit_upstream.py` for protected-source/equivalence checks.
- `tlt_reflex/{state,kernels,integration,telemetry}.py`: slot-indexed dense
  LK Reflex, GPU preallocation, Triton proposal/cache/update, lifecycle hooks,
  opt-in profiling outside captured graphs. Root feedback only.
- `tlt_reflex/{runtime,checkpoint,data}.py`: exact source selection, runtime
  preflight, actual pretrained checkpoint/mapping adapter, local data handling.
- `benchmark.py`, `run_benchmark.sh`, `benchmark_pair.sh`,
  `scripts/smoke_benchmark.sh`, `scripts/compare_upstream_outputs.py`.
- `rl.py`, `run_rl.sh`, `scripts/prepare_rl_data.py`: dispatch official
  `verl.trainer.main_fastrl`; no rewritten RL objective/reward/optimizer.
- Model path configs for all seven existing model keys and two train launchers
  per key: normal file = TLT+Reflex, `_tlt.sh` = TLT-only.
- Exact dependency lock/input/constraints, installer/wheelhouse builder,
  vendored FlashInfer0.4 with official stable-FFI ABI backport and checked wheel.
- Tests, `README_REFLEX.md`, `ENVIRONMENT.md`, `huongdanchay.md`.

Only seven FastRL upstream files differ (details/hashes in provenance):

1. EAGLE worker: initialize plugin before graph capture; normal/root proposals;
   optional phase timing + already-existing accepted CPU counters.
2. EAGLE verification input: feedback from existing teacher + root indices.
3. Normal draft graph runner: preallocated valid-batch GPU scalar for plugin.
4. Draft-extend graph runner: root correction/cache inside real captured path.
5. ReqToTokenPool: alloc/free/clear reset request-local state.
6. Scheduler: optional metrics through existing get_internal_state RPC.
7. VERL constants_ppo: propagate plugin/import identity to remote Ray workers.

No edit to tree construction, top-k utility, native verifier/sampler,
BEG/MAB selection/reward, KV cache algorithm, GRPO objective, reward function,
FSDP trainer, or background trainer implementation. Dependency-only FlashInfer
backport is explicit (official TensorView ABI migration, not a version-check bypass).

## Checks actually run

- Python3.12 `python -m compileall -q .`: PASS (full vendor tree included).
  Existing upstream invalid-escape/string-identity SyntaxWarnings only.
- `bash -n`: PASS for all 21 authored shell scripts.
- CLI dry-run benchmark validation: PASS for tlt and tlt_reflex.
- Real upstream Hydra `fastrl_trainer` schema composition: PASS for both
  methods; also Qwen3-1.7B/Qwen2.5-3B wrappers and one-step override.
- `pytest -q`: **36 PASS**, no skipped tests in the complete run. Real GPU tests
  used available RTX3090 / Torch2.5.1cu124 / Triton3.1 (not B200 target stack).
  Test-only safetensors0.7.0 added to an isolated `/tmp` target, not peer env.
  Includes CPU/GPU analytic update parity, compact vocab19/519/16000,
  sampled/greedy teacher, strided verifier root indices, request reorder,
  repeated context proposals, free/reuse, padded slot0 protection, actual
  CUDA graph capture/replay, no plugin sync/host copies/collectives/loss log.
  Actual upstream proposal Python functions tested pristine vs OFF and empty
  Reflex: same tree parents/selected indices/tokens, KV movement, RNG state;
  deterministic NN stub and shared torch.topk stand in for model/native SG topk.
  This is **not** a claim of full engine equivalence with real model weights.
  Adapter names/shapes/required weights/mapping/idempotence/export corruption
  are checked; state remains FP32 independently of global default dtype;
  real target-family capture APIs prove no double layer-index shift.
- Source audit: PASS, 126 protected hash files + 680 pristine Python files;
  changes confined to seven files above; bundled wheel SHA256 verified.
- Dependency resolver: PASS, full 206-package exact lock targeting Python3.12
  Linux x86_64, upstream era cutoff; CUDA Python/bindings12.8 match cu128.
- FlashInfer patched wheel build: PASS in dedicated Python3.12 builder.
- `python -m pip check`: PASS in existing test env and isolated builder env.
  **Neither is the fully installed 206-package TLT runtime. Full-stack B200
  pip check/native-import validation remains required and runs in installer.**

## Real benchmark/RL not yet validated

Attempted `scripts/smoke_benchmark.sh` with isolated Python3.12: stopped at
preflight, correctly reporting missing Torch2.8.0, Transformers4.57.1,
sgl-kernel0.3.15, FlashInfer0.4.0. No engine was launched and no benchmark data
were generated. Machine also lacks nvcc and configured server target/draft/data
paths. Existing GPU test env has different versions; it was not modified to
pretend to be the official stack. No full training launched.

Thus real B200 native SGLang engine smoke, actual checkpoint model-forward
parity, end-to-end RL, ordinary TP/multi-node execution, full native-extension
compatibility and measured throughput/overhead remain unvalidated. Scripts
are provided to run these on the server, **no fake AAL/tokens/s or speedup claim**.

## Explicit limitations / unavoidable deviations

- Upstream ODT factory chooses EAGLE1, not EAGLE3. EAGLE3 modes keep official
  default ODT OFF, reject enable=true. EAGLE3 opportunistic online training
  is **not implemented** by this plugin. Upstream original code remains intact.
- Non-overlap classic EAGLEWorker only (as upstream FastRL RL launcher);
  V2/DP-attention mapping is not implemented. Unsupported mode fails clearly.
- Dense state/buffers are compact-vocab, local to request slots/TP rank;
  memory scales with configured max running requests and feature dimension.
- RL remains official FastRL full-model FSDP GRPO, not SpecNaacl LoRA GRPO.
  Same settings across tlt/tlt_reflex, but not a claim of identical GRPO setup
  to SpecNaacl as an external comparison.
- Checkpoint adapter rejects unsupported normalization/bias/tied-head variants;
  no lossy conversion, random initialization or vocabulary rebuilding.
- Standalone throughput pass uses CUDA graphs, profiling OFF; optional eager
  component profiling is separate. Adaptive scheduler can react to overhead,
  so same seed is not evidence of response-level bitwise equality.
- New Triton FP32 reductions implement the same analytic LK objective; active
  state parity is tolerance-tested, not claimed bitwise equal to a different
  BLAS reduction order. Empty-state logits/probabilities/topk/tree are exact
  in the tested proposal routines; target sampling code is unchanged.
