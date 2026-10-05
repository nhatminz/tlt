# Review-fix implementation / validation report

Date: 2026-10-05. This report supersedes the earlier local-artifact-only result.
Mode: **TLT adaptive speculative rollout + fixed pretrained EAGLE3**, not full
Spot-Trainer TLT. No edits to SpecNaacl; its git status stayed empty.

## Changes

- Reproducible `scripts/bootstrap_upstream.sh` and tracked
  `patches/fastrl_reflex.patch`: clone official FastRL, detach exact commit
  bce3df7a4d46473912e9b81bf47bca419729557f, save pristine SGLang before patch,
  apply/check/reverse-check, verify commit/file-set/patch and patched-file hashes.
  Existing wrong/dirty/unidentified source is refused, never reset/overwritten.
- Offline git bundle preparation, verified source bootstrap and pytest session
  bootstrap when source is absent. Source tests do not require local wheels.
  `upstream/`, `wheels/`, `artifacts/` remain generated/ignored, but all code,
  patches, hashes and generation instructions are in the deliverable.
- RL conversion uses actual supported reward IDs: DAPO/math_dapo,
  GSM8K/openai/gsm8k, MATH/simplelr/lighteval/MATH; preserves supported original
  math identifiers. Unknown families fail and require an explicit identifier.
  Correct ground-truth normalization, no blanket boxing; actual unchanged
  upstream scorers are exercised for positive rewards.
  Dataset-specific final-answer instructions match upstream GSM8K strict
  `####` and DAPO/AIME Minerva `Answer:` formats. Both methods/benchmark share
  the same formatting. Existing invalid converted data is rejected by validator,
  not silently reused or overwritten.
- Proposal math remains Fast-LK dense state. Normal/eager draft, initial/root
  capture, post-verification eager extend and captured extend all hook before
  original softmax/topk. Normal graph calls the same draft routine. V2 is not
  supported: verified inheritance invokes the existing factory overlap guard.
- Feedback consumes the existing filtered teacher or greedy predictions and
  strided `accept_index[:,0]`, gathers fixed compact mapping and renormalizes.
  No added target forward, target softmax or sampling RNG draws.
- Request-slot ownership unchanged; free/reuse now zeroes Q/psi as well as A
  and validity flags in the same lifecycle kernel. Scratch buffers/addresses
  stay preallocated; no batch-row owner assumptions or state all-reduces.
- Added memory MB (A-only and total owned buffers), correction latency,
  root/extend timing and total proposal latency. Profile events remain OFF in
  production. Eager profiling is a separate pass; nested times cannot be summed
  into net wall overhead. Source acceptance/round counters remain weighted.
- Small-batch BEG benchmark reserves sufficient common request/graph capacity
  for ALL upstream buckets (default minimum32 for bucket21+), preventing an
  empty capture set. It does not submit additional samples or alter BEG rules.
- `smoke_parity.sh` (OFF/LR0/active fixed-greedy diagnostic), adaptive smoke,
  `benchmark_grid.sh` (prompt batches1,2,4,8,16,32), bootstrap/data/path/feedback/
  lifecycle/zero-LR/disabled-profile tests.
- Effective Hydra-config validation rejects EAGLE3 Spot Trainer regardless of
  True/true spelling, before model/Ray launch.

Upstream patch contains ALL seven modified files: eagle_worker.py, eagle_info.py,
both draft graph runners, memory_pool.py, scheduler.py, constants_ppo.py.
No changes to tree builder, verifier/sampler algorithm, BEG/MAB selection,
target distribution, GRPO loss/reward/optimizer or Spot Trainer implementation.

## Reproducibility evidence

An isolated artifact Git repo was built from tracked + new deliverable files
(excluding ignored upstream/wheels). It was committed ONLY in a temporary test
directory and cloned there; the user's repository/index was not committed.

- Fresh clone, no upstream/wheels: direct executable bootstrap against official
  GitHub succeeded; exact commit and seven-file patch verified; unit suite passed.
- Moving generated upstream aside (recoverable equivalent of removing it), then
  pytest alone: session bootstrap using local git bundle succeeded; suite passed.
- Separate offline bootstrap into an empty directory: passed. Repeated bootstrap:
  passed. Wrong HEAD guard: tested, refuses and leaves existing HEAD unchanged.
- Source audit: 126 protected hash files plus 680 pristine Python files; exact
  patched-file fingerprints and patch checksums pass.

The deliverable changes must be included in your commit/upload; the previous
published HEAD without the new scripts/patches is not claimed reproducible.

## Tests actually run

- Final TltReflex-only suite: **64 passed**, real RTX3090 CUDA tests included;
  Torch2.5.1cu124/Triton3.1 test environment, NOT the pinned B200 engine stack.
- Clean-clone suite: **64 passed**, no upstream/wheels present before session;
  offline git-bundle bootstrap automatically reconstructed all required sources.
- Actual upstream Python draft/tree-input routines: pristine vs OFF, empty A,
  and LR0 after feedback give exact parents/indices/tokens/KV moves and RNG state.
  NN and native topk are controlled stubs; not a real checkpoint/engine test.
- Actual captured-extend run_once body: OFF/LR0 exact; nonzero state changes topk
  as expected on CPU/CUDA. Actual CUDA graph capture/replay of plugin passes.
- Actual verifier prefix: greedy/stochastic branches reuse unchanged teacher/
  root metadata; no extra forward/sampler invocation. Native verifier/topk
  filtering are stubs in this harness, not a native-kernel validation claim.
- Root analytic update against dense oracle with compact vocab19/519/16000,
  reorder, simultaneous requests, finished/free/reuse, valid_bs padding,
  stable buffer pointers and LR0 unchanged outputs/teacher: passed.
- Actual pinned default_compute_score dispatch function + real scoring modules:
  correct DAPO/GSM8K/MATH labels yield positive scores. No reward stubs.
- compileall of whole TltReflex tree passed; upstream SyntaxWarnings only.
- Shell syntax passed (all26 authored launchers at time of check).
- CLI config validation and actual upstream Hydra composition passed for both
  methods; configs match excluding run names/paths. Spot enable=True fails
  with explicit fixed-EAGLE3/non-Spot error.
- pip check passed in test/builder environments; NOT a certification of the
  uninstalled full206-package B200 stack. Installer enforces that check there.
- One accidental workspace-root pytest collection was interrupted after unrelated
  sibling/vendor import errors; it is not included in the scoped suite claims.

## FlashInfer packaging correction

The previous “full official PR1960 applied” claim was wrong: Git apply in a
nested ignored directory could skip paths; that PR's TensorView revision also
does not match the exact0.4.0 sdist. Host compilation confirmed old owning
Tensor::operator-> no longer exposes TensorObj fields in stable FFI0.1.

Tracked `flashinfer_stable_ffi.patch` now supplies a small owning-Tensor accessor
compatibility class using typed get(), retaining TensorObj/FFI type/ownership.
Only type aliases/accessor compatibility and beta dependency metadata change;
no CUDA kernel math changes. Bootstrap uses explicit patch directory, verified
official sdist SHA256, dry-run/apply/reverse-check and hashes of all changed files.

- Host C++ accessor + typed function registration: passed.
- Real TVM-FFI compiled host module on NumPy tensors: shape7 and same underlying
  data pointer alias passed (not syntax-only and not a CUDA JIT claim).
- Wheel rebuilt locally and independently from freshly bootstrapped source:
  passed, exact version0.4.0. Auditor verifies packaged ABI code + metadata,
  not a machine-specific wheel-byte hash. All changed source files are hashed.
- Earlier live wheel replaced; old artifact preserved at
  /tmp/tlt-review-old-flashinfer-wheel.whl, no source/model/dataset deletion.
- Native CUDA JIT/FlashInfer sampling/attention compatibility is still untested
  here. Do NOT infer this from host ABI or pure wheel-build success.

## Native smoke / benchmark not completed

Attempted TLT-only, tlt_reflex LR0 and LR.05 engine smoke: each stopped BEFORE
engine launch at preflight, reporting missing Torch2.8.0, Transformers4.57.1,
sgl-kernel0.3.15 and FlashInfer0.4.0 in the isolated Python3.12 builder.
Also no nvcc or configured server model/pretrained draft/data assets here.

Thus no real-model B200 engine equivalence, end-to-end RL, TP/multi-node, native
CUDA JIT, batch-grid timings, measured AAL/tokens/s or speedup result is claimed.
No benchmark JSON/results were fabricated; no full training launched.

## Spot Trainer audit / remaining limits

Both FSDP drafter factory and background factory select EAGLE1 llama/qwen2;
draft_vocab_size wiring is commented out in FSDP setup. Background batch shifts
a single hidden stream; loss uses frozen head plus SmoothL1/CE on that stream,
not EAGLE3 three-feature/compact-head/unrolling training. Porting just class names
would be wrong. Full capture/loss/compact mapping/sync and version invalidation
need a separately verified EAGLE3 Spot Trainer port; not implemented here.

Keep training=false for both methods; enabling it raises. Ordinary TP is
designed in, not hardware-validated; overlap/V2 and DP-attention are explicitly
unsupported, not silently switched to a different proposal/verifier.
