# TLT adaptive speculative rollout + fixed EAGLE3, with/without OPD

Two production modes: `METHOD=tlt` and `METHOD=tlt_opd_reflex`. Both use the same
native TLT settings and exact EAGLE3 base checkpoint. Spot Trainer is disabled;
this is not full Spot-Trainer TLT. A[H,8] is frozen, B[V,8] is shared per rollout
wave and adapts online. Baseline allocates no OPD runtime tensors/kernels.

The plugin uses exact normalized lm-head inputs, Top16 distributions at actual
expanded contexts, committed-path/valid one-hop frontier feedback after EOS/stop/
length truncation, and the SpecNaacl union/tail objective. Scratch is bounded by
speculative batch capacity, including an exact globally normalized chunk path for
larger native batches. Device auto dispatch uses calibrated sparse/fused/GEMM
costs; root versions avoid redundant head recomputation. Native TLT scheduler/KV,
BEG/MAB, tree construction, target sampler/RNG and verifier are retained.

A trained SpecNaacl projector is preferred. Head-basis A is labelled untrained
and requires `OPD_ALLOW_UNTRAINED_PROJECTOR=1`. Saved A is never overwritten;
unverified training claims are rejected. Official OPD benchmarks require the
real-checkpoint SGLang-vs-SpecNaacl representation certificate and zero orphan/
invalid-context counters. Profiled component runs never rank throughput winners. Canonical preflight rejects
any non-OPD config difference before generation; head-basis and trained-A reports
remain distinct, with checkpoint/training source recorded.

```bash
bash train_qwen25_3b_tlt.sh  # TLT adaptive rollout + fixed EAGLE3
bash train_qwen25_3b.sh      # same + OPD (projector provenance guard applies)
```

[RUN_TLT_OPD.md](RUN_TLT_OPD.md): paths, environment, projector opt-in, real parity,
smoke, paired benchmark, sweeps and all model wrappers.
[OPD_IMPLEMENTATION.md](OPD_IMPLEMENTATION.md): integration and numerical limits.
[IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md): tests/measurements actually run.

Native B200 AAL/tokens/s and real checkpoint parity have not been measured here;
local CUDA fixtures are not substitutes. No end-to-end speedup is claimed.
