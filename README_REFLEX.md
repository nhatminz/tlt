# TLT adaptive rollout + FastGRPO drafter

Repo chạy độc lập bằng FastGRPO full target vocabulary hiện tại của SpecNaacl.
Hai method: `tlt` và `tlt_opd_reflex`. Không dùng SGLang, SpecForge/EAGLE3,
compact vocab, mapping d2t/t2d hoặc Spot Trainer trong production.
`legacy/sglang/` chứa implementation trước rewrite và không nằm trên PYTHONPATH/test discovery.

## Architecture và source

- `helper/modeling_draft.py`: nguyên byte FastGRPO DraftModel: EagleFS,
  DraftDecoderLayer, states/logits MLP, shared target embedding và lm_head.
- `helper/fastgrpo_model.py`, `fastgrpo_training.py`, `opd_reflex.py`,
  `opd_reflex_kernels.py`, `opd_sampling.py`, `opd_optimizer.py`,
  `tree_verification.py`, `tree_kernels.py` giữ nguyên implementation SpecNaacl.
- `helper/tlt_generate.py`: derive từ OPD rollout của source, dùng chung cache,
  tree expansion, confidence pruning, sampling, verification, history và compaction
  cho cả hai method. TLT thay duy nhất scheduling và delayed draft prefill.
- `helper/tlt_mab.py`: nguyên byte FastRL MAB; `tlt_scheduler.py` thêm adaptive
  tail gate, RNG riêng, capacity validation, timing/reward, record/replay và checkpoint state.
- `helper/tlt_transition.py`: rebuild draft-only, không forward target.
- `SOURCE_MANIFEST.json` ghi source SHA256, SHA256 của port, trạng thái exact/adapted.
  Snapshot nằm trong `sources/SpecNaacl/`, `sources/FastGRPO/`, `sources/FastRL/`.
  Chạy `python scripts/check_source_manifest.py` để kiểm tra. Sau khi chủ động sửa
  local adaptations, dùng `python scripts/refresh_port_manifest.py` rồi kiểm tra lại;
  source snapshot hashes được giữ nguyên.
  Launcher dùng `python scripts/check_source_manifest.py --runtime`: vẫn kiểm tra
  hash của helper, GRPO/pretraining và TLT core, nhưng không yêu cầu snapshot lưu trữ
  hoặc hash script/config vận hành. Lệnh không có `--runtime` vẫn audit đầy đủ.

SpecNaacl chỉ được dùng lúc port hoặc làm đường dẫn weights/data; không runtime-import
source sibling. Có thể đặt weights/data ở bất kỳ nơi nào và chạy chỉ folder TltReflex.
Checkpoint EAGLE3 cũ không tương thích; dùng checkpoint FastGRPO sau rewrite của SpecNaacl.

## Transition đúng alignment

Target prefill và các round target-only giữ `features[t]=target_hidden(x[t])`
và `draft_input_ids[t]=x[t+1]`, gồm cả token bonus đã sample. Chưa gọi draft
transformer. Round đạt đủ warmup checks vẫn target-only và đánh dấu pending. Sau round đó,
chạy một draft-only causal prefill trên toàn bộ history còn sống, với padding/position_ids đúng source.
Draft KV, next-feature và lm-head hidden được dựng từ prefix này; target KV tiếp tục
được tái sử dụng. Round tiếp theo mới chạy SD; SD giữ enabled đến hết rollout. Compaction chọn cả history owners,
padding và caches cùng thứ tự; sampler vẫn nhận thứ tự request gốc.

Regression test đối chiếu rebuild với DraftModel source trực tiếp: hidden/logits/KV
bitwise khi cùng captured features. So target-only histories với target prefill
độc lập cùng prefix: real-token hidden/KV và next logits trong tolerance BF16
`atol=rtol=0.025`; padding KV bị mask không có semantic relevance.

## TLT config

| Biến | Default |
|---|---|
| `TLT_BS_THRESHOLD` | `auto`: initial live responses của từng rollout |
| `TLT_SD_WARMUP_CHECKS` | 1 target-only round |
| `TLT_MAB_CONFIGS` | 16 explicit arms, hai arm cho mỗi verify budget 2/4/8/16/32/64/128/160 |
| `TLT_SCHEDULING_MODE` | `budget_aware_beg` |
| `VERIFICATION_CAPACITY` | 512 total verification tokens per round |
| `TLT_MAB_ALGORITHM` | `BEG` |
| `TLT_MAB_BS_THRESHOLDS` | `1,3,5,9,17,33,65,129` |
| `TLT_MAB_WINDOW_SIZE` | 1000 |
| `TLT_MAB_SEED` | training subset seed, default 42 |
| `TLT_STRATEGY_TRACE` | training: `<run>/strategy_trace.jsonl` |
| `TLT_STRATEGY_REPLAY` | empty; optional exact trace path |

`8_4_32` -> depth 8, K4, verification_num32, total_draft31.
`budget_aware_beg` giữ MAB/reward gốc, nhưng chỉ chọn nguyên arm có
`live_batch * verification_num <= VERIFICATION_CAPACITY` và depth/K/verify nằm trong `MAX_*` truyền vào runtime. Nếu bucket ưu tiên
không vừa, chọn bucket hợp lệ lớn nhất; không clamp depth/K/token của arm.
`fastgrpo_matched` gọi trực tiếp `get_adaptive_hyperparameters()` của FastGRPO
với capacity và MAX/MIN limits được truyền vào rollout, giống SpecNaacl.
Defaults: depth5/K8/max_verify160/min_depth3/C0.75, capacity512.
Threshold mặc định tự bằng initial live batch của mỗi rollout; warmup1 transition sau target-only
round đầu tiên và SD bắt đầu ở round tiếp theo. Không nhân threshold để cấp capacity.
Scratch chỉ khởi tạo sau round pending transition, theo batch thực tế còn sống;
target-only không tạo PackedTree, tree mask hoặc OPD scratch.
Invalid candidate counts hoặc capacity override quá nhỏ fail rõ ràng; không clamp strategy.
`BATCH_SIZE` là số prompts; live batch khởi đầu = `BATCH_SIZE * RESPONSES_PER_PROMPT`.

BEG giữ sliding median, batch groups, exploration và stable acceptance length
như FastRL. Reward = stable acceptance length × live batch / processing time.
Timing boundary speculative chung gồm proposal, verification, OPD feedback nếu bật,
compaction và draft append. Dùng CUDA stream events và chỉ chờ event để đọc reward;
không synchronize toàn device. Draft-only transition prefill đo riêng và nằm ngoài reward.
Target-only dùng direct one-token target forward, sampler giữ nguyên và packet EOS/compaction.
Metrics tách `effective_aal` (mọi round) và `speculative_aal` (chỉ SD).
CSV/summary ghi `target_only_rounds`, `speculative_rounds`, `speculative_round_ratio`.
Trace JSONL ghi `actual_draft_depth`, `actual_draft_k`, `actual_verify_tokens` (tổng live batch),
`actual_verify_tokens_per_response`, `selected_mab_strategy` mỗi round.
TLT cảnh báo khi không có SD hoặc ratio < `TLT_MIN_SPECULATIVE_ROUND_RATIO` (default 0.5). Đây là implementation ưu tiên correctness; chưa claim tối ưu throughput.

## Training và OPD

Cả hai dùng objective FastGRPO `2*SmoothL1 + 0.1*soft CE`, cùng draft LR,
AdamW, accumulation và update cadence, cùng GRPO target objective. A chỉ được update
ở draft optimizer boundary. B full-V rollout-local reset mỗi rollout. Feedback dùng
visited+frontier và teacher metadata của target sampler; không thêm target forward.
`tlt` không khởi tạo A/B, OPD proposal kernels hoặc proposal profile.
Đây là **TLT adaptive rollout + FastGRPO drafter**, chưa phải full official TLT Spot Trainer.

Đọc [RUN_TLT_OPD.md](RUN_TLT_OPD.md) để chạy B200 từng bước.

## Benchmark và replay

`scripts/benchmark_tlt_opd.py` tạo `report.json`, `summary.csv`, `responses.jsonl`,
`strategy_trace.jsonl`, `config_diff.json`, và trace riêng từng case/method trong `runs/`.
Config diff kiểm tra mọi field ngoài OPD bằng nhau. Hai method dùng cùng checkpoint,
prompts và thứ tự, seeds, sampling, batch, TLT/MAB và warmup/measured iterations.
Physical order counterbalanced: even seed TLT trước, odd seed OPD trước.
Production OPD mặc định `OPD_REQUIRE_CALIBRATED_PROFILE=1`; smoke có thể override `=0`.
Tuner chỉ phủ tail workload `min(batch*responses, threshold)*max(strategy K)`.
Default là frozen rollout; `BENCH_ONLINE_DRAFT=1`/`--online-draft` thêm đúng draft
training sau mỗi rollout, giữ target frozen để đo draft pipeline. Training GRPO đầy
đủ chạy qua hai train launchers. `OPD_PROFILE=1`/`--profile` đo inclusive feature/proposal/feedback
GPU cost bằng một frozen replay riêng; không tính replay vào wall time/peak memory đo chính.
Đây là thời gian các OPD sections, không phải slowdown counterfactual. Chênh lệch wall time
của ablation/pair cho biết effect end-to-end; giá trị chưa profile được ghi null, không bịa zero.

Để controlled comparison, record baseline bằng `--method tlt`; lấy trace riêng ở
`runs/batch<B>_seed<S>_lr<L>_stream<U>/tlt/strategy_trace.jsonl`. Set
`TLT_STRATEGY_REPLAY=<file>` rồi chạy OPD hoặc pair trên đúng một seed/batch và
cùng iteration/config. Pair controlled replay áp cùng trace cho cả hai method.
Replay kiểm tra rollout/round/live batch/phase và budget. Nếu OPD khiến finish/trajectory
khác đến mức trace không tương thích, fail rõ ràng; không ép token/request để khớp trace.

## Tests

```bash
python -m compileall -q helper scripts tests grpo_speculative.py train_draft.py
python -m pytest -q
python scripts/check_source_manifest.py
```

Tests bao gồm source integrity, native FastGRPO tokens/RNG/history parity trên
Qwen2/Qwen3, real transition hidden/logits/KV, no extra forwards, finishing/compaction,
depth8/K4/verify48/32/16/8, BEG reference traces, OPD full-V kernels và optimizer,
profiles/dispatch/tuner, paired launch configs, GRPO training/resume và pair outputs.

## Compatibility và sampler (2026-10-08)

`helper/transformers_compat.py`, `environment_checks.py`, `fastgrpo_model.py` và
validator được port chính xác từ SpecNaacl mới. Transformers4.51.3 giữ API native;
5.12.1 dùng adapter singular/plural decoder output và DynamicCache legacy list views.
Không đổi attention, weights, objective, tree hoặc KV alignment. Pretrain, TLT và OPD
đều dùng model adapter này. `helper/opd_profiles.py` vẫn giữ adaptation riêng cho TLT.

Requirements vẫn là installation pins tham chiếu. Dùng environment hiện có:

```bash
python scripts/validate_environment.py --require-cuda
```

Validator kiểm tra bounded API families rồi chạy thật draft pretrain/backward,
AdamW/scheduler, target decoder/cache, PEFT LoRA/state và checkpoint roundtrip.
`--strict-versions` yêu cầu exact pins. Không tự cài hoặc downgrade thư viện.

`OPD_SAMPLER_MODE=finite` là default cấu hình cho cả `tlt` và `tlt_opd_reflex`.
Có thể override `strict` để dùng validation/fallback cũ. `finite` dùng nguyên implementation SpecNaacl: device assertion cho
nonfinite sampled logits, không host scalar reads/dynamic valid-row compaction.
Đặt biến trước khi khởi động process; pair config ghi sampler mode trong phần
shared và reject nếu hai method khác mode. Kiểm tra runtime trên stack B200 trước benchmark.

Online-draft benchmark tách `generation_wall_s`, `draft_training_wall_s`,
`combined_wall_s`, `generation_tokens_per_s`, `combined_tokens_per_s`.
`tokens_per_s` là alias generation throughput; combined gồm cả backward/optimizer
theo cadence gốc. Hai wall intervals nối tiếp, không double-count.
`target_time_cost` khi `statistical_time=True` gồm cả target-only forwards;
profiling tắt không thêm event wait.

Benchmark ghi `actual_depth`, `actual_k`, `verification_num` (mỗi response), và
`verification_tokens` (tổng round) trong `strategy_trace.jsonl` (run và aggregate).
OPD experiment fail nếu không có SD rounds hoặc không có feedback states, kể cả
ablation LR0; LR0 được phép có zero updates. Fingerprint scheduler/rollout thay đổi:
tune lại TLT-native proposal profile trên GPU chạy benchmark.

Nhánh OFF dùng cùng raw FP32 scan/merge và low-ID tie ordering như OPD khi B=0,
để LR0 không bị lệch vì softmax BF16 hoặc `torch.topk` tie ordering. Không cấp A/B
hoặc feedback state cho OFF; kernel OPD và target verifier giữ nguyên.
