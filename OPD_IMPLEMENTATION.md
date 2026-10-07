# OPD plugin implementation

The experiment is **TLT adaptive speculative rollout + fixed EAGLE3** vs the same
with OPD. EAGLE3 Spot Trainer remains disabled. SpecNaacl is read-only mathematical
reference; native scheduler, KV, verifier, target sampling/RNG and BEG/MAB at
`bce3df7a4d46473912e9b81bf47bca419729557f` are retained.

- `eagle_info.py`: target verification -> native token/finish loop -> truncate
  accept_index -> OPD -> native flatten/KV commit. Terminal decision states have
  no valid continuation frontier. Existing host finish bookkeeping supplies the
  terminal mask; no added D2H, target forward or RNG draw.
- `checkpoint.py` / `integration.py`: preserve saved A and all base EAGLE3 weights;
  deterministic head-row QR only when A is absent. Accepted provenance is trained
  or head_basis_initialized. Unknown evidence is rejected, never relabelled as
  trained. Untrained A requires OPD_ALLOW_UNTRAINED_PROJECTOR=1 and warns. A stays
  frozen; B[compact_vocab,r] is shared within a request-pool wave.
- `state.py`: full-slot head/u/Top16/norm/valid/expanded caches remain keyed by
  req_pool_indices. Feedback tiles, queues, union and parent/context/path buffers
  use max_speculative_batch_size*max_feedback_nodes. Default capacity derives from
  native threshold, overridden by OPD_MAX_SPECULATIVE_BATCH_SIZE. Because upstream
  stays enabled after transition, larger actual batches use bounded metadata/teacher
  chunks and ONE accumulated rank-r gradient/total-weight update against frozen B_t.
  Target probabilities do not escape to async work. Small batches keep the native
  compact-teacher/union/update side-stream path. Chunking can change FP32 reduction
  association; CPU/CUDA tests compare within the existing numerical tolerances.
- `dispatch.py` / `kernels.py`: same measured-cost interpolation/argmin policy as
  SpecNaacl, implemented as setup cost tables + device active-count interpolation.
  Nonmonotone sparse/fused/GEMM regions are retained. Preallocated sparse workspace
  covers all sparse-winning counts; GEMM storage exists only when explicit or auto
  can select it. Selected preparation runs; other preparation kernels are no-ops.
  Fixed FP32 rank arithmetic and normalization/Top16 merge are preserved.
- `B_version` / `root_B_version[slot]`: after actual representable B writes, advance
  version once. LR0, invalid/zero-weight and zero-gradient rounds do not advance.
  Fresh roots gather cached Top16/u by slot, avoiding head GEMM and feature projection.
  Stale rows alone recompute head logits with a masked native-dtype Triton GEMM and
  the unchanged OPD correction/Top16 arithmetic. Root and deep outputs have separate
  storage. Cache reuse is reported separately from backend rounds; reused roots
  execute no correction backend. BF16 H256/H2048 head comparisons and graph mixed
  version/reorder tests cover this route. Bitwise equality with every cuBLAS head
  shape is not claimed; native model representation needs the real parity certificate.
- Lifecycle cache reclamation is ordered after its reader on the side stream.
  Main-stream waits occur before proposals read B or at explicit reporting boundaries;
  no global CUDA synchronization or new proposal `.item/.cpu/.tolist` is introduced.
- `opd_orphan_nodes` / `opd_invalid_contexts`: validate native selected tree metadata,
  expanded identity, cached validity and parent topology. Bad OPD feedback contexts
  are masked safely; OPD_DEBUG=1 asserts. Native tree/verifier are not changed.
  Official reports require both cumulative counters zero, including warmup.
- `parity.py` / `scripts/validate_tlt_eagle3_parity.py`: isolated real-checkpoint
  native SGLang observation and SpecNaacl/SpecForge replay, with separate environments.
  Head-input/logit/u errors and corrected Top16 agreement are reported. Failed,
  missing, stale or fixture-only certificates cannot unlock official OPD comparison.
- `benchmark.py` / `summarize_tlt_opd.py`: exact artifact/tokenized-prompt/workload/
  engine-config identity guards, scoped experiment name, persistent/scratch MB,
  all requested acceptance/throughput/memory/OPD metrics. Throughput profile OFF;
  components separate. A recommendation needs higher verified AAL and strictly
  higher tokens/s; an observed win is not a significance claim.

OPD union math is unchanged: DraftTop16 + positive unique TargetTop16 + one tail,
no shortlist renormalization; weighted q-p with one batch denominator, compact-mass
conditioning and pre-update target-only q reconstruction. The ported core changes
are limited to an optional actual-write flag and deferring apply for oversized
chunks. No optimizer, projector training, custom attention/scheduler/history/KV
runtime, target re-softmax/sort or transformer forward was ported.

Read [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) for actual validation
results and unresolved limits, and [RUN_TLT_OPD.md](RUN_TLT_OPD.md) for B200 commands.
