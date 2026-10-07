# OPD completion report — 2026-10-07

Experiment: **TLT adaptive speculative rollout + fixed EAGLE3** vs the same + OPD.
Spot Trainer is OFF. SpecNaacl was not edited; native TLT semantics remain pinned.

**Fixed:** feedback now runs after native EOS/stop/length truncation and before
flattening; terminal continuations are not valid frontier states. Added bounded
feedback scratch and a single globally normalized chunk update for oversized
native batches. Auto dispatch retains calibrated sparse/fused/GEMM regions.
Device B/root versions skip unchanged-root head GEMM. Context/orphan diagnostics
mask invalid feedback safely and reject invalid comparisons. Cache reclamation
is ordered on the update stream. Runtime audits prevent reuse of the old verifier
patch; bootstrap recognizes/backups the previous OPD checkout.

**Files:** tlt_reflex/{state,kernels,dispatch,checkpoint,integration,profiles,parity,
runtime}.py; the small optional version/deferred-apply hooks in
ported/opd_reflex_kernels.py; patches/fastrl_reflex.patch and prior-OPD hashes;
benchmark.py, benchmark_pair.sh, run_benchmark.sh; scripts/{summarize_tlt_opd,
tune_tlt_opd_proposals,export_specforge_draft,validate_tlt_eagle3_parity,
upgrade_upstream,bootstrap_upstream,launch_common}; four root launch aliases;
regression tests and README/run/implementation docs. Base EAGLE3 weights,
verifier/sampling/RNG, BEG/MAB, GRPO and KV/scheduler algorithms were not rewritten.

**Projector:** saved A is preserved/frozen. Only trained or head_basis_initialized
provenance is accepted. Unknown evidence fails; no random initialization or fake
learned label. Head-basis A requires OPD_ALLOW_UNTRAINED_PROJECTOR=1 and warns.
The exported base checkpoint is identical in both modes. Real-checkpoint parity
is required for official OPD benchmarks; failed/stale/fixture certificates fail.

**Actually run:** pytest -q: **104 passed, 2 skipped** (CPU BF16 variants; CUDA BF16
is tested). Covers native verifier-prefix finish flow, original-source OPD math,
real CUDA graph/backend/mixed-root-version replay, chunk-vs-single-batch update,
invalid-context safety, zero-gradient versioning and recorder protocol. Source
bootstrap/previous-OPD upgrade audits, 14 real Hydra launcher compositions and
shell syntax checks passed. Tests use Python3.10/Torch2.5.1 cu124/Triton3.1 on RTX3090.

**Measured components:** synthetic V519 proposal CUDA graphs at batches1/2/4/8/16/32,
active rows0/16/128/519: sparse/fused/GEMM bitwise parity passed. Synthetic actual
state allocations at pool512/V32000/H2048/BF16: persistent65.75 MB unchanged;
scratch754.34 ->160.42 MB when feedback capacity512 ->32; teacher tiles614.4 ->38.4 MB.
These are component/layout fixtures, not production model peak memory or tokens/s.

**Not validated here:** three native smoke attempts and paired benchmark attempt
all failed the pinned Python3.12/Torch2.8 cu128 environment gate. Real-checkpoint
parity tool attempt failed because B200 assets are absent; its report is explicitly
passed=false. No production AAL/throughput, real checkpoint head-input errors or
B200 speedup is claimed. TP>1, overlap V2, DP attention and quantized/scaled draft
heads remain unsupported. Oversized chunks and masked head GEMM have FP32/native-
dtype numerical tolerances, not a universal bitwise trajectory guarantee.

Evidence is in local validation/{pytest_revision.txt,native_revision_attempts.json,
real_eagle3_revision_unavailable.json,proposal_revision_rtx3090.json,
memory_revision_fixture.json,previous_opd_upgrade_revision.txt}. These artifacts
remain ignored by the user's existing .gitignore. [RUN_TLT_OPD.md](RUN_TLT_OPD.md)
has the exact B200 sequence and projector opt-in/validation commands.
