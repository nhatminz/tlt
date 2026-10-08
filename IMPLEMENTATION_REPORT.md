# TLT tail correctness/performance revision — 2026-10-08

Only TltReflex was modified. SpecNaacl status remained clean. OPD math/kernels,
projector optimizer, sampler, model and tree verifier remain byte-identical to the
current SpecNaacl source snapshots. No SGLang/Spot Trainer production dependency.

## Fixes and files

- `helper/tlt_scheduler.py`: exact pending boundary. The Nth qualifying round is
  target-only, then draft-only rebuild, then SD starts at N+1. Tail verification
  capacity = min(initial live batch, threshold) × max configured verification_num.
  Separate effective/speculative AAL; trace records CUDA-event timing/reward basis.
- `helper/tlt_generate.py`: direct one-token target-only path, same sampler and
  request ordering, feature/token history and EOS/swap-remove compaction. No
  PackedTree, tree verifier/mask, OPD state or speculative scratch before transition.
  OPD/tree scratch and async stream initialize only after the triggering round,
  using the actual surviving tail batch. Pending A-gradient sums are preserved;
  B retains the original start/finish reset lifecycle.
- `helper/tlt_timing.py`: reusable stream CUDA events for speculative round/MAB
  reward, inclusive of proposal/verification/feedback/committed append and async
  dependency. Only end event is waited for elapsed time; no device synchronization
  in rollout, including statistical timing. Transition prefill has its own event
  interval and is excluded from MAB reward. Target-only adds no timing event wait.
- `scripts/tune_opd_proposals.py/.sh`, `resolve_opd_profile.py`: flattened unique
  tail workload only, max contexts=min(batch×responses, threshold)×max(strategy K).
  Seed42 sorted scattered active IDs, exact 3-backend parity and profile validation
  before atomic write. OPD math/proposal kernels unchanged.
- `scripts/run_tlt_opd_reflex.sh`, `scripts/launch/train_model.sh`,
  `scripts/sweep_tlt_opd_reflex.sh`: production require calibrated TLT-native profile
  by default; explicit OPD_REQUIRE_CALIBRATED_PROFILE=0 supports development smoke.
- `helper/rollout_metrics.py`, `grpo_speculative.py`, `scripts/benchmark_tlt_opd.py`:
  effective AAL, speculative AAL, target-only/SD rounds, transition time, reward,
  throughput and memory. Optional --profile measures inclusive OPD feature/proposal/
  feedback cost in a separate frozen replay excluded from measured wall/peak/TPS.
  No invented zero when profiling is disabled. Inclusive section cost is not the
  counterfactual slowdown; pair/ablation wall time measures the system effect.
- `configs/_shared/b200_common.env`: default verification capacity also tail-bounded.
- `tests/test_tail_revision.py`, `test_tlt_fastgrpo.py`, `tests/reference/`: old
  root-only-tree implementation frozen strictly as a test reference; native MAB
  parity, full target masks/KV, hidden/history/tokens/RNG, boundary, lazy allocation,
  no-global-sync, terminal trigger, large batch shrink and pending A gradients.
- Source manifest/refresh script and README/RUN_TLT_OPD instructions updated.

## Validation

Final `python -m pytest -q`: **121 passed**, zero failed/skipped, 33 warnings, 58.57s.
Warnings are uncalibrated synthetic OPD fixtures; real-model runs require and load a
validated native profile. compileall, source manifest and `git diff --check` pass;
`bash -n` passes all 69 production + archived legacy shell scripts.

GPU regression tests actually executed on RTX3090 include Qwen2/Qwen3 transitions,
old-path/new-path tokens, CUDA RNG, every target attention mask/KV and returned
hidden/token training histories, no extra target forward, depth8/K4 budgets48/32/16/8,
BEG/reward reference, full V151936 BF16/FP16 sparse/fused/GEMM parity, profile dispatch,
training/resume and analytical projector gradients. Forced finishing fixture starts
with 64 live responses, shrinks to2 before SD and allocates OPD feedback for96 rows,
not 64×48. A pending gradients remain untouched by initialization.

Timing/lifecycle tests forbid torch.cuda.synchronize in rollout for both methods,
with statistical timing on/off. Fully target-only tests additionally forbid any
PackedTree/tree workspace/OPD initialization and elapsed-event wait. Pending rebuild
consumes no CPU/CUDA sampling RNG. Same captured features yield bitwise draft
hidden/logits/KV parity; independent BF16 target prefill comparison uses explicit
numerical tolerance on real-token KV (padding remains masked).

## Real measurements (RTX3090, not B200)

Environment: Python3.10, Torch2.5.1+cu124, Triton3.1.0, Transformers4.51.3,
PEFT0.17.1. Deployment pins Torch2.8/Triton3.4/B200 were not tested here.

Target-only before/after uses a tiny real Qwen2 transformer V97, 1 warmup and
3 measured seeds42/43/44, total max_length32, zero draft forwards. Median speedup
ranges **1.12–1.17×** at batches1/2/4/8/16/32. At batch32, old root-tree path takes
107.44ms vs direct path95.98ms; peak allocated12.28MB vs10.90MB. This measures the
fixture and implementation overhead, not a model-scale production speedup.

Full-model acceptance uses real Qwen2.5-1.5B-Instruct, the same existing two-step
FastGRPO pretrain checkpoint and local SimpleLR prompts, one response per prompt,
max total length112, max prompt96, seeds42/43, 1 warmup/1 measured iteration,
async stream1. 48 measured rows cover TLT / OPD LR0 / OPD LR0.01 and all six batches.
Each row has one transition, target forwards=prefill+rounds, same prompt hashes and
counterbalanced order; every config_diff has only OPD differences.

| Batch | TLT tokens/s | OPD LR0 tokens/s | OPD LR0.01 tokens/s |
|---:|---:|---:|---:|
| 1 | 36.66 | 36.12 | 35.87 |
| 2 | 67.99 | 66.38 | 66.04 |
| 4 | 127.73 | 129.67 | 125.35 |
| 8 | 251.42 | 258.16 | 254.62 |
| 16 | 412.98 | 411.54 | 414.49 |
| 32 | 717.43 | 702.65 | 705.94 |

Values aggregate two seeds; baseline column uses baseline runs paired with LR0.
There are also repeated baseline rows paired with LR0.01 in the raw report. Peak
allocated at batch32 is 5.49GB (TLT), 5.71GB (OPD0), 5.83GB (OPD0.01). This is a short
functional smoke with a two-step draft; **no consistent OPD speedup claim**.

Separate full-model profiling replay (batch1, responses2, seed42) records inclusive
OPD feature/proposal/feedback6.15ms for LR0 and6.74ms for LR0.01 per rollout. These
numbers are excluded from throughput measurements and are not net wall slowdown.

Real native tuner on this GPU uses V151936/r8/BF16, tail max_live32/max_contexts128,
contexts1/2/5/11/25/57/128, scattered rows0/16/1024/V and5 timing iterations. All
backend bitwise parity checks pass; profile reload/execution-key validation pass.
An earlier matrix attempt was discarded when a final scheduler edit changed the
execution fingerprint; the final profile was regenerated and all48 rows rerun.
Production B200 needs fresh calibration on that GPU with the final source tree.

Artifacts: ignored `validation/tlt_tail_fix/` contains pytest_final.log,
profile_load_final.json, final profiles, target_only_performance.json,
real_matrix_final/{report.json,summary.csv,responses.jsonl,strategy_trace.jsonl,config_diff.json},
and profile_replay/. B200 commands: [RUN_TLT_OPD.md](RUN_TLT_OPD.md).
