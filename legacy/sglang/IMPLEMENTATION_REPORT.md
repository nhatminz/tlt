# Native tuner and pair order — 2026-10-07

Only four integration/calibration issues were changed. OPD math/state/kernels,
objective, verifier/sampling/RNG, BEG/MAB and EAGLE3 semantics were not rewritten.
SpecNaacl was not modified.

- **Duplicate bucket:** tuner flattens/deduplicates shapes before measurement;
  default effective contexts are [1,2,4,8,16,32,128]. ProposalProfile and the native
  consumer validate the payload before an atomic file write. Invalid payloads
  neither create a bad profile nor overwrite a valid one.
- **Scattered IDs:** sorted randperm with an explicit CUDA generator/seed replaces
  contiguous prefixes. IDs, B rows and bitmap are consistent; seed, previews and
  ID digests are recorded. Calibration is deterministic for each workload.
- **Native only:** official loader requires profile_kind=tlt_native, scattered
  graph calibration, exact GPU/CC/V/r/dtype/TopK/compiler/kernel/version and TLT
  execution fingerprint. SpecNaacl and old contiguous calibrations are rejected.
  Missing native profile + OPD_REQUIRE_CALIBRATED_PROFILE=1 fails with tuner command.
  Default profile directory is TltReflex/outputs/benchmarks/opd_proposals.
- **Order:** actual canonical sampling seed determines physical launch order:
  even seeds TLT->OPD, odd seeds OPD->TLT. run_order.json and report positions make
  this reviewable. Generation-critical canonical fields remain identical.

**Files:** scripts/tune_tlt_opd_proposals.py/.sh, tlt_reflex/profiles.py,
scripts/launch_common.sh, benchmark_pair.sh, scripts/pair_order.py, benchmark.py,
scripts/summarize_tlt_opd.py, scripts/validate_tlt_opd_profile.py, tests and docs.

**Pass:** compileall; bash -n38 scripts; pytest -q **117 passed, 2 skipped** (CPU
BF16; CUDA covered). Regression executes real default tuner shapes/profile reload,
invalid-write protection, deterministic scattered IDs and actual shell order for
seeds42/43/44/45. Existing OPD math/graph/context/version/chunk tests still pass.

**GPU actually measured:** shell tuner on RTX3090 using synthetic V519/r8 FP32
fixture, defaults'7 effective contexts, active rows0/16/128/519, 5 timing samples.
All sparse/fused/GEMM bitwise proposal parity checks pass. New native profile loads
through validate_tlt_opd_profile.py. CUDA graph auto dispatch with this measured
profile matches argmin: at contexts32, S0 sparse, S16 GEMM, S128/S519 fused.
This fixture profile is not a production/B200 calibration.

**Blocked native attempts:** TLT, OPD LR0, OPD LR.01 and counterbalanced pair all
failed the required Python3.12/Torch2.8 cu128 gate on the local Python3.10/Torch2.5
cu124 environment; B200 production model/data/draft resources are absent. No
native end-to-end AAL/tokens/s or B200 advantage is claimed.

Local ignored evidence: validation/pytest_native_tuner_order.txt,
validation/tuner_native_fixture/{load_report.json,auto_selection.json,tune.log},
validation/native_tuner_order_attempts.json.
Step-by-step B200 commands: [RUN_NATIVE_PROFILE_PAIR.md](RUN_NATIVE_PROFILE_PAIR.md).
