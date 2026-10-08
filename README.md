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

SpecNaacl chỉ được dùng lúc port hoặc làm đường dẫn weights/data; không runtime-import
source sibling. Có thể đặt weights/data ở bất kỳ nơi nào và chạy chỉ folder TltReflex.
Checkpoint EAGLE3 cũ không tương thích; dùng checkpoint FastGRPO sau rewrite của SpecNaacl.

## Transition đúng alignment

Target prefill và các round target-only giữ `features[t]=target_hidden(x[t])`
và `draft_input_ids[t]=x[t+1]`, gồm cả token bonus đã sample. Chưa gọi draft
transformer. Sau khi live batch <= threshold đủ số checks, chạy một draft-only
causal prefill trên toàn bộ history còn sống, với padding/position_ids đúng source.
Draft KV, next-feature và lm-head hidden được dựng từ prefix này; target KV tiếp tục
được tái sử dụng. SD giữ enabled đến hết rollout. Compaction chọn cả history owners,
padding và caches cùng thứ tự; sampler vẫn nhận thứ tự request gốc.

Regression test đối chiếu rebuild với DraftModel source trực tiếp: hidden/logits/KV
bitwise khi cùng captured features. So target-only histories với target prefill
độc lập cùng prefix: real-token hidden/KV và next logits trong tolerance BF16
`atol=rtol=0.025`; padding KV bị mask không có semantic relevance.

## TLT config

| Biến | Default |
|---|---|
| `TLT_BS_THRESHOLD` | 32 live responses |
| `TLT_SD_WARMUP_CHECKS` | 10 consecutive decode checks |
| `TLT_MAB_CONFIGS` | `8_4_48,8_4_32,8_4_16,8_4_8` |
| `TLT_MAB_ALGORITHM` | `BEG` |
| `TLT_MAB_BS_THRESHOLDS` | `1,2,5,21` |
| `TLT_MAB_WINDOW_SIZE` | 1000 |
| `TLT_MAB_SEED` | training subset seed, default 42 |
| `TLT_STRATEGY_TRACE` | training: `<run>/strategy_trace.jsonl` |
| `TLT_STRATEGY_REPLAY` | empty; optional exact trace path |

`8_4_32` -> depth 8, K4, verification_num32, total_draft31.
Không gọi native `get_adaptive_hyperparameters` trong TLT mode.
Capacity được tính từ initial live batch × max configured verification_num;
verification token/position/mask buffers và feedback capacity đủ worst case.
Invalid candidate counts hoặc capacity override quá nhỏ fail rõ ràng; không clamp strategy.
`BATCH_SIZE` là số prompts; live batch khởi đầu = `BATCH_SIZE * RESPONSES_PER_PROMPT`.

BEG giữ sliding median, batch groups, exploration và stable acceptance length
như FastRL. Reward = stable acceptance length × live batch / processing time.
Timing boundary chung gồm proposal, verification, OPD feedback nếu bật, compaction
và draft append/rebuild. Cả hai method synchronize ở cùng boundary; OPD async cost
được tính. Đây là implementation ưu tiên correctness; chưa claim tối ưu throughput.

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
Default là frozen rollout; `BENCH_ONLINE_DRAFT=1`/`--online-draft` thêm đúng draft
training sau mỗi rollout, giữ target frozen để đo draft pipeline. Training GRPO đầy
đủ chạy qua hai train launchers.

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
