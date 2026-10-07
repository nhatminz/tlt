# Reflex OPD port: implementation and validation

## Source mapping

| SpecNaacl source | TLT/SGLang hook |
| --- | --- |
| opd_reflex.py deterministic head basis, persistent A | checkpoint exporter/load_projector and OPDState.projector (FP32/frozen) |
| shared B_fast + active bitmap/list | OPDState on EAGLEWorker, request-pool wave epoch lifecycle |
| exact EAGLE3 compute_compact_logits_with_inputs | draft LogitsProcessorOutput.opd_head_input, separate from auxiliary recurrent states |
| proposal feature + corrected normalized Top16 | root capture, draft_forward, draft CUDA graph, extend and extend CUDA graph |
| expanded feedback context IDs | upstream select_top_k_tokens tree_info -> slot expanded_ids -> EagleVerifyInput.opd_parents/opd_feedback_contexts |
| visited + one-hop expanded frontier | existing accept_index, converted from flattened teacher indices to local nodes on GPU |
| compact teacher extraction | consume existing filtered target_probs inside EagleVerifyInput.verify, before release/finished-row filtering |
| union + one tail + weighted q-p + shared update | ported/opd_reflex_kernels.py (same FP32 math/reconstruction ordering) |
| OPD_UPDATE_STREAM | compact teacher/source_ready -> side-stream union/update -> update_done -> proposal wait |
| proposal tuner/profile | TLT graph tuner with GPU/CC/V/r/dtype/TopK/kernel key; checked source calibration reuse |

No OPDStaticCache, FastGRPO KV compaction, custom scheduler, history, tree
attention or FastGRPO verifier was ported. Only the proposal merge was extracted
from SpecNaacl tree_kernels. The pinned native SGLang tree construction, target
verifier and RNG draws are unchanged. TP>1 head reconstruction explicitly fails;
TP1 is implemented. Quantized/scaled/softcapped draft heads and overlap V2/
DP-attention are also rejected rather than approximated.

## State and lifecycle

A is frozen FP32 [H,r]. B is shared FP32 [compact_vocab,r], never [slots,V,r].
Caches have [slot,1+max_MAB(topk*(steps-1)),H/r/16/2] layout. Only forwarded
beam candidates get context IDs; final-depth/unexpanded candidates remain -1.
Cache reset clears every slot-owned feature/normalization/Top16/valid field.
B persists while requests in a wave remain active, including when one request
finishes and its slot is reused. B resets before a new wave when the pool was
empty. Counters remain cumulative for explicit server-info telemetry.

Correction is raw_logit + (head_input @ A) dot B[token], without random
projection or feature normalization. All hot CUDA buffers are preallocated.
Sparse correction uses append-only active IDs, bitmap and [proposal_rows,S]
scores. Auto chooses sparse/fused with device active_count and an offline-derived
threshold per row workload. Sparse preparation is a no-op when fused is selected.
FP32 ordering and the fixed normalization reduction preserve bitwise parity
across proposal implementations in the tested fixtures. The full-vocab GEMM
score workspace exists only when explicitly selected before capture. There is
no full-vocab probability cache or [contexts,V,r] correction tensor.

Root Top16 outputs have separate storage from deep proposal scratch: upstream
keeps root confidence views until its first expansion. Sharing that storage
would corrupt beam scores. A cached root may also wait across a different
scheduler batch's shared-B update. Before drafting, OPD refreshes its head-only
logits/distribution from cached exact head_input with current B. The root head
matmul output is preallocated. This adds a draft head operation, never a target
or transformer forward; its measured event cost is reported separately. It may
be a throughput bottleneck and needs end-to-end profiling on B200.

Compact teacher extraction reads selected rows of existing post-temperature,
top-k/top-p filtered target_probs. It scans compact probabilities with tile Top16
reductions; it does not re-softmax or re-sort the target vocabulary. Positive
TargetTop16 entries only, sentinel -1/0 otherwise. It also captures compact mass
and raw p at DraftTop16. Full target_probs does not escape to the update stream.
DraftTop16 stays in the union even where p=0. Target-only q reconstructs native
head-dtype logits from cached exact head operand/head row, pre-update B, u and
normalization. The union has one tail and is not shortlist-renormalized.
Zero/nonfinite compact mass contributes neither weight nor update.

Projectors in checkpoint tensor state/top-level/sidecar are preserved; conflicting
copies or invalid ranks/shapes/nonfinite values fail. Export leaves base EAGLE3
weights and mapping unchanged. A head-basis fallback is deterministic and labelled
head_basis_initialized, trained=false. Explicit trained metadata is preserved.
An existing non-basis tensor without training evidence is preserved and labelled
checkpoint_training_unverified; the exporter never manufactures a trained flag.

## Profiles and telemetry

SpecNaacl profiles are read from the sibling outputs/benchmarks/opd_proposals.
GPU/CC/V/r/dtype/TopK/source kernel hash must match. Native TLT graph profiles
have their own wrapper hash and full compiler key, and take priority. A source
profile is a calibration prior for the same arithmetic core; it is not a measured
TLT graph cost and compiler differences may change the optimal threshold. Auto
uses sparse/fused. GEMM remains explicitly available and measured by the offline
tuner, with no GEMM workspace penalty when unused. No auto-tuning happens in
rollout/generation. A malformed/incompatible explicit profile fails clearly.

Benchmark emits AAL (upstream completion_tokens/rounds), verified AAL
(accepted_draft+rounds)/rounds, rounds, accepted/proposed draft tokens, acceptance
rate, generation wall time/tokens/s, engine peak allocated/reserved memory,
selected/visited/frontier/invalid counts, weighted KL, compact mass, target mass
in DraftTop16, active rows mean/max and sparse/fused/dense/GEMM root-proposal
counts. These proposal counts include extend roots and root refreshes and are
not sequence verification-round counts. Peak memory and active max include
engine startup/warmup; cumulative sums/counts are differenced after warmup.

Opt-in eager diagnostic runs expose feature, proposal, root head, teacher
extraction, union, update, wait and total OPD section timings. Graph inner timings
are unavailable (null), not fabricated as zero. Total excludes wait and inclusive
upstream timers to avoid double-counting asynchronous work. Main throughput runs
always have profile OFF. BEG's existing native timing remains unchanged.

Frozen pairs use identical model weights, prompts, seeds, sampling and TLT/MAB
parameters. MAB responds to timings, so token trajectories need not be identical.
Cold OPD leaves raw logits/distribution unchanged mathematically; its fixed FP32
normalization can differ from Torch by floating reduction roundoff. OPD breaks
logit ties by low compact ID. Native top-k(k) may choose different exact ties,
including probability ties caused by underflow. No tree/verifier changes force
candidate-ID parity. Unique-score Top16 prefix parity is tested.

## Validation scope

Read validation/native_smoke_unavailable.json for three actual native smoke
attempts and their errors. The local working CUDA environment is Python3.10 /
Torch2.5.1 cu124 / Triton3.1, required for the installed 3090 driver. Official TLT
requires a separate Python3.12 / Torch2.8 cu128 stack. Production model/data/draft
paths exist only on B200, so native engine smoke, real RL and production AAL/
tokens/s comparisons are not validated here. Do not infer a throughput win.

validation/opd_proposal_rtx3090_fixture.json records actual CUDA graph timings
for synthetic V519/r8 proposal workloads at batches1/2/4/8/16/32, active rows
0/16/128/519 and sparse/fused/GEMM bitwise parity. This is a component fixture,
not a production model benchmark or deployable B200 profile.

The pytest suite covers CPU and real CUDA math/state/stream/graphs, independently
executes the original SpecNaacl CPU feedback and Triton proposal source for parity,
runs actual pinned upstream root/draft/extend-graph/verifier-prefix control flow
with deterministic NN/native-kernel fixtures, and audits unchanged protected
upstream files, sampler RNG/target-forward counts and patch reproducibility.
Two CPU BF16 variants are skipped because BF16 reconstruction is tested on CUDA.
The final test command/result is recorded in validation/pytest.txt.
