# Final interface/fairness report — 2026-10-07

Scope remains **TLT adaptive speculative rollout + fixed EAGLE3**, with/without
OPD; Spot Trainer OFF. SpecNaacl is read-only. OPD math/state/CUDA kernels and the
TLT/SGLang patch were NOT rewritten in this revision.

**Fixed:** run_rl.sh now serializes MAB_CONFIGS as quoted JSON strings: Hydra
receives `["8_4_32","8_4_16","8_4_8"]`, never numeric underscores. Paths/string
values and logger lists are also quoted safely. The effective config rejects
non-string MAB strategies before starting Ray/runtime.

**Benchmark:** pre-generation canonical config dumps use the SAME engine/sampling
builder as generation. Any non-OPD difference fails. Pair defaults to batches
1/2/4/8/16/32 and seeds42/43; each case emits tlt/report.json, tlt_opd/report.json,
comparison.json, summary.csv, config_diff.json and both canonical configs. Grid
summaries do not double-count responses. Delta AAL/tokens/s/wall/memory and OPD
section overhead are explicit. Head-basis and trained-A labels/recommendations
remain separate. Training source checkpoint/dataset/steps are recorded; missing
trained-A source metadata blocks official comparison instead of being invented.
No A training occurs during validation or benchmarking.

**Parity/profile:** real-checkpoint validation now checks head input, raw logits,
u, corrected logits, Top16 IDs AND probability errors. Old incomplete certificates
are rejected. Official generation requires a calibrated profile; no hot-path
autotuning. Offline tuner checks full profile schema/key before reuse. Smoke is
explicitly scoped separately, so acceptance can run smoke -> parity -> offline
tune -> official pair. Provenance guards still apply to smoke.

**Files:** run_rl.sh/rl.py; benchmark.py and pair/sweep runners; scripts/{hydra_value,
check_tlt_opd_pair_config,summarize_tlt_opd,export_specforge_draft,launch_common,
validate_tlt_eagle3_parity,tune_tlt_opd_proposals,smoke_tlt_opd};
tlt_reflex/{benchmark_config,experiments,checkpoint,integration,telemetry,parity};
tests and README/run docs. Core state.py/kernels.py/ported update math are unchanged.

**Actually checked:** compileall passed; bash -n passed for38 scripts;
pytest -q **111 passed, 2 skipped** (CPU BF16; CUDA BF16 tested). All14 launchers
passed real pinned Hydra composition with exact list[str] checks. CPU preflight
ran fair config grids and rejects seed/sampling/checkpoint/graph mismatches;
CUDA tests retain backend/root/version/chunk/slot/finish correctness coverage.
Local environment: RTX3090, Python3.10, Torch2.5.1 cu124, Triton3.1.

**GPU acceptance attempts:** all3 native smoke attempts and official pair failed
the pinned Python3.12/Torch2.8 cu128 environment gate. Real parity and target-GPU
tuner failed because B200 model/draft/config resources are absent. No production
parity errors/AAL/tokens/s/peak VRAM were measured, and no speedup is claimed.
The previous synthetic CUDA/layout measurements remain component evidence only.
TP>1, DP>1 benchmark telemetry, overlap V2 and quantized/scaled OPD heads remain
unsupported. External trained-projector/evaluation dataset disjointness must be
established by the experiment owner; it is not inferred from a filename.

Evidence: local ignored validation/{pytest_final_interface.txt,
hydra_final_interface_launchers.json,native_final_interface_attempts.json}.
B200 sequence/provenance options/output layout: [RUN_TLT_OPD.md](RUN_TLT_OPD.md).
