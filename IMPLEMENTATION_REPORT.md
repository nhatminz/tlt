# FastGRPO rebase validation — 2026-10-08

Implemented **TLT adaptive rollout + FastGRPO drafter**, with optional exact source
ReflexOPD. Production contains no SGLang/SpecForge/EAGLE3 integration or compact vocab.
The former implementation is in `legacy/sglang/`; Spot Trainer is outside this phase.
SpecNaacl remained unchanged (`git status --short` empty at start/end).

## Source and scheduling

`SOURCE_MANIFEST.json` verifies 61 source entries: 45 exact local ports, 14 adapted
ports and two reference-only FastRL snapshots. FastGRPO architecture/training,
OPD correction/optimizer/sampler and FastGRPO tree verifier/kernels are exact ports.
Local `tlt_generate.py` derives from source OPD rollout. New scheduling/workspace/
transition files supply tail gate, BEG, strict capacity and replay. FastRL MAB code
is exact; NumPy RNG state is isolated from global target/model RNG.

Transition retains full target-hidden + shifted token history while making no draft
transformer calls. At the threshold it pre-fills only the draft over live prefixes,
with correct causal mask/positions and already-sampled bonus alignment. Target cache
is reused. Numerical tests check next hidden/logits and KV against source DraftModel,
and independently target-prefilled real-token prefix features/KV. Fixed-strategy TLT
also matches native FastGRPO tokens, RNG and training histories bitwise on Qwen2/Qwen3.

Default depth8/K4 budgets48/32/16/8 are validated against candidate count and allocated
verification capacity. Source adaptive-hyperparameter policy is not called. Same
end-to-end processing boundary measures both modes and includes async OPD cost.
Training retains upstream online objective/optimizer/cadence, with A feedback applied
at the draft optimizer boundary; scheduler/BEG RNG is checkpointed per rank.

## Results

- `python -m pytest -q`: **106 passed**, zero failed/skipped, 30 warnings, 53.95s.
  Warnings are deliberate uncalibrated OPD fixture fallback; actual full-model smoke
  uses a validated TLT-native profile with `OPD_REQUIRE_CALIBRATED_PROFILE=1`.
- compileall and AST checks: 61 production/test Python files pass.
- `bash -n`: 69 production + archived legacy shell scripts pass.
- source manifest, required source checks, `git diff --check`: pass.
- CUDA tests: zero-B/raw logits; full V151936 BF16/FP16 sparse/fused/GEMM parity;
  actual auto dispatch; depth8 buffer bounds; positive/zero-LR strategy replay;
  target-only sampler RNG; EOS/swap-remove; transition state; training losses;
  real GRPO entrypoint update order; bitwise optimizer/model/RNG restore on resume.
- Fixture pair: 16 measured rows, seeds42/43, batches1/2, LR0/0.01; counterbalanced
  order, only OPD config differences, no extra target forward.
- Native tuner: real CUDA, V151936/r8/BF16; deduplicated `1x1,2x1,1x2,2x4` into
  contexts1/2/8. Seed42 scattered rows0/16/1024/V; all backend parity checks pass.
  Written profile loads through ProposalProfile and exact TLT execution-key validation.
- Local real-model smoke: Qwen2.5-1.5B-Instruct + same two-step FastGRPO pretrain
  checkpoint + SimpleLR, 2 responses, total max length128, seeds42/43, batch1,
  LR0/0.01, async stream1, 1 warmup/1 measured rollout each method.
  8 measured rows pass, each with 9 target-only rounds, 19 speculative rounds,
  one real draft-only transition, and target forwards = prefill + verification rounds.
  Config diff and prompt hashes match. No extra target forward from OPD/transition.

Artifacts under ignored `validation/fastgrpo_rebase/`: pytest/tune logs,
`profile_load.json`, profiles, real_pair/, tiny_pair/, training/.
Pair folders contain report.json, summary.csv, responses.jsonl, strategy_trace.jsonl,
config_diff.json and per-case replay traces. Reports explicitly label tiny fixtures.

Test environment: RTX3090 24GB, Python3.10, Torch2.5.1+cu124, Triton3.1.0,
Transformers4.51.3, PEFT0.17.1. B200/Torch2.8/Triton3.4 and full training/matrix
remain to be run by the user. The short two-step checkpoint is a functional smoke,
not evidence of OPD speedup or model quality. Re-tune on B200; source/RTX3090 profiles
cannot pass the TLT-native B200 fingerprint. No official Spot Trainer/performance claim.

B200 commands: [RUN_TLT_OPD.md](RUN_TLT_OPD.md).
