# TltReflex: official FastRL/TLT + request-local FastLKReflex

Hai mode dùng cùng pipeline: `METHOD=tlt` và `METHOD=tlt_reflex`.
Mode hiện tại là **TLT adaptive speculative rollout + fixed pretrained EAGLE3**,
chưa phải full Spot-Trainer TLT. Cả hai mode giữ Spot Trainer OFF.
“Fixed” là các drafter tensors train riêng không được Spot update; target
embedding vẫn được share theo upstream và thay đổi theo target GRPO updates.
Không sửa `../SpecNaacl`; draft pretrained vẫn đọc từ folder đó.
Code FastRL chính thức đã được vendor ở `upstream/fastrl`, commit
`bce3df7a4d46473912e9b81bf47bca419729557f`. Không dùng PyPI SGLang thay fork TLT.
Nguồn: https://github.com/mit-han-lab/fastrl .

## Reproduce từ checkout sạch

`upstream/` và wheels là artifacts được tái tạo, không phải source chỉ có trên
máy developer. Các patch và provenance được track trong repo:

```bash
./scripts/bootstrap_upstream.sh
python -m pytest -q
```

Bootstrap clone official repo, checkout đúng commit, giữ pristine trước patch,
apply/check/reverse-check `patches/fastrl_reflex.patch`, xác nhận SHA256 của
từng file đã sửa và diff file set. Không reset/overwrite tree đang có.
Pytest tự bootstrap source nếu thiếu (không pip-install/model download).
Server offline: tạo `scripts/create_offline_upstream_bundle.sh` trên máy online,
copy git bundle rồi dùng `FASTRL_GIT_SOURCE=/actual/path/to/bundle` với bootstrap.
Installer/wheelhouse builder bootstrap source và FlashInfer ABI artifact riêng;
unit tests không cần wheel/FlashInfer/CUDA environment đầy đủ.

## Implementation

- Proposal normal draft: `EAGLEWorker.draft_forward` trong vendored
  `sglang/srt/speculative/eagle_worker.py`, ngay trước softmax/top-k upstream.
- Proposal root: `capture_for_decode`, cả initial extend và post-verification
  eager extend. CUDA-graph draft-extend có hook riêng trong
  `eagle_draft_extend_cuda_graph_runner.py:run_once`; normal draft graph gọi
  cùng `draft_forward`. Không bỏ sót graph path để Reflex chỉ hoạt động eager.
- Feedback: `EagleVerifyInput.verify` trong `eagle_info.py`, dùng chính
  `target_probs` đã temperature/top-k/top-p của verifier, hoặc `target_predict`
  khi greedy, với `accept_index[:,0]`. Không thêm target forward, teacher
  softmax, RNG draws hay scan tree. Stride của root indices được giữ đúng.
- State: `tlt_reflex/state.py`, FP32 `A[request_slot, compact_vocab, d]`,
  projection/normalization và analytic LK gradient theo SpecNaacl. Root-only
  feedback như default SpecNaacl. Đây là **dense LK Reflex**, không Sparse Reflex.
- Lifecycle: attach vào shared `ReqToTokenPool`; `alloc/free/clear` reset state.
  Proposal lấy owner từ `req_pool_indices`, không dùng batch row làm identity.
  Padded graph rows bị mask bằng GPU scalar `valid_bs` preallocated; không sửa
  attention indices, tránh padded row ghi đè request slot 0.
- Buffers cấp trước capture, fused Triton correction/rank-one update, không
  materialize outer product lớn, không `-log(alpha)` hay `mul_(1)` để log.
  Không `.item()/.cpu()` hoặc global CUDA sync trong production plugin.
  Update và graph replay trên stream upstream: update hoàn tất trước proposal
  tiếp theo theo stream ordering, không tạo side-stream race.
- Ordinary TP: state/R local mỗi rank, logits/hidden đã được upstream TP
  xử lý; plugin không all-reduce giữa requests/workers. Independent rollout
  workers có state riêng. DP-attention token redistribution và overlap/V2
  worker bị từ chối rõ ràng, chưa có mapping adapter/test cho những mode đó.

`METHOD=tlt` factory trả `None`: không allocation Reflex hoặc correction/
feedback kernel; softmax/top-k/tree, BEG/MAB, sampler, verifier và KV cache
upstream giữ nguyên. Hash audit bảo vệ 126 file, so sánh thêm 680 file Python
với bản SGLang pristine. Một copy pristine có thể dùng để kiểm tra engine OFF.

## Checkpoint / tokenizer / features

`scripts/export_specforge_draft.py` chuyển **weights đã train** sang layout
SGLang, không pretrain lại/random fallback/rebuild vocabulary. Kiểm tra tensor
shapes, normalization variants, compact d2t/t2d, required/unknown parameters.
HF tokenizer lấy từ chính target local. Target embeddings và KV cache theo
SGLang upstream. Ba decoder-layer IDs của SpecForge được giữ nguyên trong
config: API `set_eagle3_layers_to_capture` của fork **tự cộng +1** để capture
before-next-layer; không double-shift. Chỉ hỗ trợ ba IDs tăng dần, distinct,
không capture final decoder layer qua API before-layer này.

Hiện adapter hỗ trợ one-layer `LlamaForCausalLMEagle3`, target llama/qwen2/qwen3,
draft hidden = target hidden, riêng compact lm_head, không fc_norm/norm_output/
bias variants. Unsupported/missing/corrupt checkpoint là lỗi, không random.
Export lưu riêng `outputs/draft_exports/<model_key>` với SHA256 provenance,
không overwrite export khác. Nếu thay source checkpoint, dùng `DRAFT_EXPORT`
mới. Cả hai mode dùng cùng export.

## Chạy

Xem [ENVIRONMENT.md](ENVIRONMENT.md) trước: **venv riêng**, không thay stack
SpecNaacl. Các lệnh đầy đủ ở [huongdanchay.md](huongdanchay.md).

```bash
export PYTHON_BIN="$(command -v python)"
CUDA_VISIBLE_DEVICES=0 MODEL_KEY=qwen25_3b bash scripts/smoke_benchmark.sh
CUDA_VISIBLE_DEVICES=0 MODEL_KEY=qwen25_3b bash benchmark_pair.sh
# Component timings là pass eager riêng cho CẢ HAI mode:
CUDA_VISIBLE_DEVICES=0 MODEL_KEY=qwen25_3b COMPONENT_PROFILE=1 bash benchmark_pair.sh
METHOD=tlt MODEL_KEY=qwen25_3b CUDA_VISIBLE_DEVICES=0 bash run_rl.sh
METHOD=tlt_reflex MODEL_KEY=qwen25_3b CUDA_VISIBLE_DEVICES=0 bash run_rl.sh
```

Default model/dataset: path hiện có của SpecNaacl; DAPO math local, 8 responses,
temperature 1/top-p .95/top-k -1, max prompt/new tokens 2048, seed42, TP1,
TLT steps8/topk4/tree48, threshold32, BEG configs8_4_32,8_4_16,8_4_8.
Reflex d8/LR.05/decay0/profileOFF. CUDA graph ON cho benchmark throughput.
Các env override ở `scripts/launch_common.sh`; model paths ở `configs/*/b200.env`.
RL hyperparameters theo official FastRL launcher (FSDP2 full-model GRPO,
LR1e-6, train batch64, mini batch4, dynamic batch, upstream KL/objective/reward),
**không phải** SpecNaacl LoRA GRPO. Bản 1 GPU đặt Ulysses1/TP1; không sửa code
BEG/MAB, optimizer, reward hoặc GRPO objective của upstream.

## Metrics và fairness

`report.json` + `report.responses.jsonl` trong output mỗi run, gồm accepted/
proposed draft tokens, sequence verification rounds, AAL, tokens/s, elapsed,
memory RPC, config và server info. Warmup không nằm trong measured deltas.
Thêm `reflex_state_memory_mb` (A only, decimal MB),
`reflex_buffer_memory_mb` (tất cả owned preallocated buffers),
`proposal_correction_time_ms`, `draft_extend_time_ms`,
`proposal_latency_total_ms = proposal_time_ms + draft_extend_time_ms`.
Proposal total gồm initial/root extend và post-verification extend, không
chỉ tính deep draft loop. Correction/update/profile times chỉ có ở pass eager
opt-in; production pass để null, không tạo số đo giả.

- `upstream_aal = total completion_tokens / sum(response.spec_verify_ct)`:
  định nghĩa chính benchmark FastRL; numerator có cả normal decode/prefill
  khi adaptive SD OFF. Vì vậy không gọi nó pure speculative accepted length.
- `verified_aal = (accepted_draft_tokens + sequence_verification_rounds) /
  sequence_verification_rounds`: rounds SD thực, bao gồm một root/bonus token,
  weighted ratio, không mean batch averages; accepted counter dùng CPU list
  đã có từ verifier, sau truncation khi response finish.
- `draft_acceptance_rate = accepted_draft_tokens / proposed_draft_tokens`,
  denominator dùng số tree nodes thực `draft_token_num-1` mỗi sequence-round,
  không dùng cố định `steps` khi BEG thay tree size.
- CUDA events OFF mặc định. Eager component pass ghi proposal, verification,
  feature/correction/cache/update; info RPC cuối run mới đọc events. Các khoảng
  thời gian lồng nhau không được cộng thành net wall overhead. Throughput pass
  và component pass là hai thí nghiệm khác, không trộn kết quả.
- Adaptive MAB dùng timing để chọn strategy: overhead có thể thay strategy và
  RNG consumption theo tree khác. Cùng seed không đảm bảo same generated text.
  Plugin không thay target distribution; bitwise end-to-end equivalence cần
  kiểm tra riêng với engine/config quyết định giống nhau, không claim từ seed.

## Giới hạn đã xác định, không che bằng fallback

Factory background drafter trainer của commit này chỉ chọn EAGLE1 classes,
không chọn EAGLE3. Code ODT/RL upstream vẫn được giữ nguyên; EAGLE3 launcher
giữ `speculative.train.enable_drafter_training=false` (default upstream).
Enable=true báo lỗi trước khi chạy. Repo **chưa tích hợp EAGLE3 opportunistic
online draft training**; không tự gọi một trainer khác là TLT/SpecForge.
Chưa kiểm chứng engine/model benchmark B200 hoặc end-to-end RL tại máy này.
Xem [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) cho evidence/test limits.

## RL reward data fix

Data conversion preserve valid original math identifiers; nếu thiếu/legacy
invalid, map đúng family: DAPO→`math_dapo`, GSM8K→`openai/gsm8k`,
MATH/simplelr→`lighteval/MATH`. Unknown family phải có `REWARD_DATA_SOURCE`
explicit, không fallback tất cả thành `math`. Ground truth không bọc boxed
cho GSM8K/MATH. Reward upstream không sửa: format final line GSM8K=`#### ...`,
DAPO/AIME=`Answer: ...` (upstream Minerva default), MATH giữ boxed answer.
Hai mode và benchmark dùng cùng formatting này. Tests gọi actual pinned
dispatcher và scoring modules, kiểm tra cả positive reward, không chỉ tên.

## Server validation

```bash
bash scripts/smoke_benchmark.sh          # adaptive TLT vs active Reflex
bash scripts/smoke_parity.sh             # fixed-strategy greedy OFF / LR0 / active
COMPONENT_PROFILE=1 bash scripts/benchmark_grid.sh # batches1,2,4,8,16,32
```

Responses mặc định vẫn8; grid batch size là số prompts, request concurrency
=batch×responses. Fixed/greedy identity diagnostic không phải adaptive speed
benchmark; cùng seed trong adaptive MAB không bảo đảm same strategy/output.
Engine reserve tối thiểu32 request slots khi dùng default BEG buckets1/2/5/21,
để upstream capture cả bucket21+ không bị max(empty) ở smoke batch1. Đây là
capacity chung của hai mode, không tạo thêm responses/concurrency thực.
