# Official TLT + Reflex OPD

Production modes: `METHOD=tlt` and `METHOD=tlt_opd_reflex`. Fast-LK has been
replaced by the current SpecNaacl OPD math, with frozen persistent projector
A[H,8], one shared rollout-wave B[V,8], Top16 context caches keyed by stable
request slots, visited/one-hop expanded frontier feedback, compact teacher
extraction and optional asynchronous updates.

TLT keeps its native scheduler, KV, EAGLE3 recurrence, BEG/MAB, tree builder,
target sampling/RNG and verifier at FastRL commit
`bce3df7a4d46473912e9b81bf47bca419729557f`. The baseline goes through the original
proposal branches and never constructs OPD state or launches OPD kernels.
EAGLE3 Spot Trainer stays disabled in both modes; A is never updated per round.

Read [RUN_TLT_OPD.md](RUN_TLT_OPD.md) for B200 paths, reuse of your SpecNaacl
profile, smoke, paired benchmark, sweep and RL commands. Defaults match the
SpecNaacl model/data/draft paths and OPD rank8/Top16/LR0.01/stream1. Training uses
the official FastRL GRPO pipeline and a separate pinned TLT environment.

```bash
bash train_qwen25_3b_tlt.sh     # official TLT baseline
bash train_qwen25_3b.sh         # TLT + OPD Reflex
```

Tests on RTX3090 include original-source math/Triton parity, real CUDA graph
replay, slot reuse, compact teacher/union, head-only root refresh and executed
upstream proposal/verifier hooks with bounded NN/kernel fixtures. Native SGLang
production-model smoke and end-to-end B200 AAL/throughput remain to be run in the
pinned environment with the actual assets. No TLT speedup is claimed.

Files: [OPD_IMPLEMENTATION.md](OPD_IMPLEMENTATION.md),
[patch](patches/fastrl_reflex.patch), [source provenance](PROVENANCE.json),
[ported OPD source hashes](tlt_reflex/ported/SOURCE.json).
