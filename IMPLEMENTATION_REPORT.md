# Compatibility, shared sampler and timing — 2026-10-08

Only TltReflex changed. SpecNaacl status remained clean. No libraries were installed,
reinstalled or downgraded. requirements pins remain unchanged. Existing environments
were used for all checks; the exact B200 CUDA stack was not available locally.

## Files / implementation

Exact source ports from the current SpecNaacl:
`helper/transformers_compat.py`, `helper/environment_checks.py`,
`helper/fastgrpo_model.py`, `helper/fastgrpo_generate.py`, `helper/opd_sampling.py`,
`scripts/validate_environment.py`. Source snapshots and SOURCE_MANIFEST.json have
actual SHA256 values computed from the source bytes. The manifest checks66 entries.
Source compatibility/sampler tests and their frozen sampler reference were ported;
the tokenizer fixture was refreshed from source to real Qwen BPE for HF5 inference.

`helper/tlt_generate.py` and retained `helper/opd_generate.py` now import the source
DynamicCache adapter. Production/pretraining use FastGRPOModel's target decoder API
adapter. Native modern target calls keep native tensor/plural behavior; direct
FastGRPO calls keep legacy singular-cache/tuple behavior. Weight names and decoder
outputs are checked against the unadapted model. Existing static cache code works
with both installed Transformers families and remains unchanged.

Sampler strict stays default. Finite is the exact source implementation with a
CUDA async nonfinite assertion, full-row sampling and no scalar host reads/dynamic
valid-row compaction. Both production methods share the same sampler. Config diff
now records/checks sampler_mode as shared configuration, and train metadata records
it. Shell defaults remain strict. Finite is not promoted to B200 production default.

Two timing fixes:

- Target-only decoding adds its target interval to target_time_cost using CUDA events
  only when statistical_time=True. Off adds no event waits. The boundary includes
  target forward/head/sampling, matching the speculative target timing boundary.
- Online benchmark separately measures generation_wall_s, draft_training_wall_s and
  combined_wall_s; the latter is the sum of the adjacent intervals. Separate
  generation_tokens_per_s and combined_tokens_per_s are reported in JSON/CSV;
  tokens_per_s remains the generation alias. Training objective, accumulation and
  optimizer step boundaries are unchanged. Warmup still includes training when
  requested, and partial accumulation is not flushed.

Other adaptations: scripts/benchmark_tlt_opd.py, configs/_shared/b200_common.env,
scripts/check_training_sources.py, scripts/launch/train_model.sh, regression tests,
README/ENVIRONMENT/RUN_TLT_OPD docs. helper/opd_profiles.py, TLT gate/MAB/transition,
OPD math/kernels/optimizer, draft architecture/loss and verifier remain unchanged.

## Actual validation

| Installed environment | Execution checks | Tests |
|---|---|---|
| RTX3090, Torch2.5.1+cu124, Transformers4.51.3, PEFT0.17.1 | GPU validator/pretrain backward/optimizer/scheduler, decoder/cache, LoRA/checkpoint probe pass |306 passed,68 warnings,69.98s |
| RTX3090, Torch2.5.1+cu124, Transformers5.12.1, PEFT0.21.1, datasets5.0.1 | Same real GPU probes pass |306 passed,66 warnings,73.29s |
| CPU, Python3.12.12, Torch2.13.0+cpu, Transformers5.12.1, PEFT0.21.1, Triton3.7.1 | Real CPU validator/pretrain/optimizer/decoder/cache/LoRA/checkpoint probes pass |87 passed,207 skipped (CUDA unavailable),4.21s |

Warnings are existing synthetic profile fallback, intentional strict invalid-logit
fallback and deterministic-fixture messages. Final suites have zero failed tests.
Initial failing tests were fixture issues: old generic WordLevel tokenizer loaded as
Qwen tokenizer under HF5, direct test DraftModel config missing rope_scaling=None,
and differing test runner tuple layouts. They were corrected using the source Qwen
BPE fixture and unchanged production draft config; no model/sampler math was changed.

Sampler tests check FP32/FP16/BF16 and full V151936 tokens, probabilities, sort/top-k
teacher metadata and CPU/CUDA RNG bitwise across seeds, temperatures, top-p/top-k,
ties and strided logits; strict nonfinite fallback matches the frozen old source.
Finite CUDA invalid-logit checks run in isolated subprocesses and fail as required.
No-host-read/dynamic-compaction and CUDA graph capture tests pass on RTX3090.
TLT integration checks acceptance/history/forward counts and compact OPD feedback/B/A
for strict versus finite, sync and async. FP32 atomic feedback reductions use an
explicit eight-epsilon-at-tensor-scale bound; sampled tokens/probabilities/RNG remain
bitwise checks. Timing tests verify every target-only forward is counted, off has no
event waits, adjacent benchmark phase intervals add exactly, and real online-draft
GPU runs retain the existing optimizer cadence.

`python scripts/check_source_manifest.py` passes.
`python scripts/validate_environment.py --require-cuda` passes on both GPU environments.
`python -m compileall -q .` passes, including archived/upstream sources.
`bash -n` passes all69 production + archived shell scripts.
Exact copied FastGRPO generator bytes retain the source's trailing-whitespace EOF;
no source snapshot bytes were altered to hide that inherited formatting.

## Real model smoke and native profile

Modern GPU stack above, real Qwen2.5-1.5B-Instruct and the same existing two-step
FastGRPO pretrain checkpoint, local SimpleLR prompts, batch1/2, responses2, seeds42/43,
LR0/0.01, async stream1, one warmup/measured iteration, max total length112/prompt96.
Both strict and finite run16 measured rows (32 total), covering all three ablations.
Within each mode, run order is counterbalanced and config_diff contains only OPD
changes. Across modes, all48 response token sequences per mode and acceptance/forward
metrics match exactly in this smoke. Every measured row satisfies no-extra-target-
forward. Frozen benchmark training time is0 and combined/generation times match.
This is correctness evidence, not an OPD speedup or fully pretrained quality claim.

TLT-native tuner reran after the execution fingerprint changed: V151936/r8/BF16,
contexts1/2/3/4/6/10/16, tail batch4/K4, seeded scattered active rows0/16/1024/V,
5 timing iterations. All sparse/fused/GEMM bitwise proposal checks pass, and the new
profile loads/validates against the current GPU/execution key. Profile code itself
was not changed. B200 must tune on its own GPU/compiler/runtime stack.

Artifacts under ignored `validation/compat_sampler_metrics/`: validator_hf451.log,
validator_hf512.log, validator_torch213_cpu.log, pytest_hf451.log,
pytest_hf512_final.log, pytest_torch213_cpu.log, compileall_final.log, native profiles,
profile_load.json, real_strict/ and real_finite/ reports/CSV/responses/config diff.

Still required on B200: run validator and tests under the actual Torch2.13 CUDA /
Triton3.7 stack, regenerate native profiles, smoke all three ablations, then validate
finite mode there before any production default change. Commands: RUN_TLT_OPD.md.
