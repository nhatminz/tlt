# Chạy trên B200

Các default model/data giống SpecNaacl. Draft mặc định trỏ sang checkpoint FastGRPO
mới tại `../SpecNaacl/outputs/pretrain/<MODEL_KEY>/latest_checkpoint`. Không dùng
checkpoint compact/EAGLE3 cũ. Nếu copy TltReflex độc lập, override `MODEL`,
`DATASET_PATH`, `DRAFT_CHECKPOINT`, `TARGET_CONFIG` đến các files đã copy.

## 1. Environment

```bash
cd /workspace/storage-shared/nlp/minhpn19/TltReflex
# Activate environment đã cài sẵn trên server; không cần reinstall/downgrade.
source .venv/bin/activate
export PYTHON_BIN="$(command -v python)"
export CUDA_VISIBLE_DEVICES=0
python scripts/validate_environment.py --require-cuda
python scripts/check_source_manifest.py
python -m pytest -q
```

Pins tham chiếu là Torch2.8.0/cu128, Triton3.4.0, Transformers4.51.3, PEFT0.17.1.
Validator cũng hỗ trợ compatible API families, gồm Transformers5.12.1/PEFT0.21.1;
phải pass decoder/cache, draft backward, optimizer, LoRA và checkpoint probes thật.
Chỉ dùng bootstrap khi chủ động muốn tạo environment từ pins; không cần đổi stack B200 hiện tại.
Old wheelhouse SGLang có thể khác dependencies; bootstrap mới dùng requirements mới.
Offline có thể set `WHEELHOUSE=/path/to/new-compatible-wheels`.

## 2. Paths/config dùng chung

```bash
export MODEL_KEY=qwen25_3b
export MODEL=/workspace/storage-shared/models/Qwen2.5-3B-Instruct
export DATASET=simplelr
export DATASET_PATH=/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/train.parquet
export DRAFT_CHECKPOINT="$(realpath ../SpecNaacl/outputs/pretrain/$MODEL_KEY/latest_checkpoint)"
export TARGET_CONFIG="$MODEL/config.json"
export TARGET_ADAPTER=""
export TARGET_LR=1e-6 DRAFT_LR=1e-4
export BATCH_SIZE=8 ACCUMULATION_STEPS=4 DRAFT_ACCUMULATION_STEPS=1
export RESPONSES_PER_PROMPT=8 TRAIN_SUBSET_SEED=42
export TLT_BS_THRESHOLD=auto TLT_SD_WARMUP_CHECKS=1
export TLT_SCHEDULING_MODE=budget_aware_beg
unset TLT_MAB_CONFIGS TLT_MAB_BS_THRESHOLDS  # Dùng đầy đủ budget buckets mới của TLT
export VERIFICATION_CAPACITY=512 MAX_DRAFT_TOKEN_LENGTH=5 MAX_DRAFT_K=8
export MAX_VERIFICATION_NUM=160 MIN_DRAFT_TOKEN_LENGTH=3 DRAFT_TOKEN_LENGTH_C=0.75
export TLT_MAB_ALGORITHM=BEG
export TLT_MAB_SEED=42
export OPD_SAMPLER_MODE=finite
export OPD_RANK=8 OPD_TOPK=16 OPD_FAST_LR=0.01
export OPD_UPDATE_STREAM=1 OPD_TRAIN_PROJECTOR=1
export OPD_PROPOSAL_MODE=auto OPD_DENSE_IMPLEMENTATION=auto
export OPD_PROPOSAL_PROFILE_DIR="$PWD/outputs/benchmarks/opd_proposals"
unset OPD_PROPOSAL_PROFILE OPD_TUNE_OUTPUT
```

## 3. Tune TLT-native profile trên chính B200

```bash
export OPD_TUNE_MODELS="$MODEL_KEY"
export OPD_TUNE_ITERATIONS=30
bash scripts/tune_tlt_opd_proposals.sh
python scripts/validate_tlt_opd_profile.py \
  --target-config "$TARGET_CONFIG" --draft-checkpoint "$DRAFT_CHECKPOINT" \
  --rank 8 --dtype bf16 --topk 16 --profile-dir "$OPD_PROPOSAL_PROFILE_DIR"
export OPD_REQUIRE_CALIBRATED_PROFILE=1
```

Tuner default workload chỉ phủ tail: `max_live=min(BATCH_SIZE*RESPONSES_PER_PROMPT, TLT_BS_THRESHOLD)`
và `max_contexts=max_live*max(TLT strategy K)`. Profile dùng full V; key gồm GPU/CC, Torch/Triton/CUDA, V, rank, dtype, kernel SHA
và TLT execution fingerprint. Source SpecNaacl profile và profile RTX3090 không được
coi là compatible cho TLT trên B200. Tuner dedup `batch*contexts`, scattered active IDs
với seed42, kiểm tra 3 backend parity và load bằng ProposalProfile trước atomic write.
Auto chọn sparse/fused/GEMM bằng measured/interpolated cost. `OPD_FAST_LR` không thay
cách tính OPD; profile dispatch chỉ quyết định implementation kernel.

## 4. GPU smoke ba cấu hình trên cùng weights/prompts

Đây là frozen rollout smoke, chưa update target/draft. Nếu muốn kiểm tra online draft,
thêm `--online-draft`. Mỗi lệnh dùng fresh model/checkpoint; B reset mỗi rollout. Với warmup checks1,
round1 vẫn target-only, prefill drafter sau round1 và round2 mới speculative.

```bash
export BENCH_BATCH_SIZES=1 BENCH_SEEDS=42 BENCH_ITERATIONS=1 BENCH_WARMUP=1
export BENCH_MAX_LENGTH=256 BENCH_MAX_PROMPT_LENGTH=96
export RESPONSES_PER_PROMPT=2
METHOD=tlt BENCH_OUTPUT="$PWD/outputs/smoke/tlt" bash run_benchmark.sh
METHOD=tlt_opd_reflex OPD_FAST_LRS=0 BENCH_OUTPUT="$PWD/outputs/smoke/opd_zero" bash run_benchmark.sh
METHOD=tlt_opd_reflex OPD_FAST_LRS=0.01 BENCH_OUTPUT="$PWD/outputs/smoke/opd_live" bash run_benchmark.sh
```

## 5. Train hai method riêng từ cùng checkpoint ban đầu

```bash
export BATCH_SIZE=8 RESPONSES_PER_PROMPT=8
export GEN_MAX_LENGTH=2048 MAX_PROMPT_LENGTH=2048
unset RESUME RUN_DIR RUN_NAME TLT_STRATEGY_REPLAY
bash train_qwen25_3b_tlt.sh
bash train_qwen25_3b.sh
```

`tlt_opd_reflex` launchers mặc định require calibrated profile; explicit
`OPD_REQUIRE_CALIBRATED_PROFILE=0` chỉ dành cho smoke/development.
`train_qwen25_3b_tlt.sh` = pure TLT; `train_qwen25_3b.sh` = TLT+OPD.
Generic: `bash scripts/run_tlt_fair.sh` và `bash scripts/run_tlt_opd_reflex.sh`.
Các model khác có cùng cặp wrapper. Mỗi run viết output độc lập, giữ cùng init draft,
target adapter, dataset order/seed, optimizer/LR/cadence. Chiến lược có thể adapt khác
vì performance/acceptance khác; đó là system-level comparison đã yêu cầu.

Smoke training ít bước: thêm `--max_grpo_steps 2`. Resume dùng cùng method/config,
`RUN_DIR=<existing-run> RESUME=auto bash train_qwen25_3b.sh`; scheduler/MAB RNG và
metrics được lưu cùng checkpoint rank-local.

## 6. Pair benchmark nhiều batch/seed, counterbalanced

```bash
export RESPONSES_PER_PROMPT=8
export BENCH_BATCH_SIZES=1,2,4,8,16,32
export BENCH_SEEDS=42,43,44,45
export BENCH_ITERATIONS=2 BENCH_WARMUP=1
export BENCH_MAX_LENGTH=512 BENCH_MAX_PROMPT_LENGTH=256
export OPD_FAST_LRS=0,0.01 OPD_STREAMS=1
export BENCH_OUTPUT="$PWD/outputs/benchmarks/tlt_fastgrpo_pair"
bash scripts/sweep_tlt_opd_reflex.sh
```

Even seed chạy TLT→OPD, odd seed OPD→TLT. Đây là frozen rollout benchmark từ cùng
checkpoint. Nếu đo cả draft online training: `BENCH_ONLINE_DRAFT=1 bash scripts/sweep_tlt_opd_reflex.sh`.
Dùng output mới cho mỗi experiment. Xem `report.json`, `summary.csv`, `responses.jsonl`,
`strategy_trace.jsonl`, `config_diff.json`. Config diff phải chỉ có key `opd` khác. `effective_aal` gồm cả target-only;
`speculative_aal` chỉ gồm SD. `tlt_transition_draft_prefill_s` được đo riêng, ngoài MAB reward.
Có thể chạy `OPD_PROFILE=1` cùng frozen benchmark để lấy OPD section timings qua
replay riêng, không thêm replay vào measured wall/throughput/peak. Không dùng tùy chọn này
cùng `BENCH_ONLINE_DRAFT=1`.
Training full GRPO comparison là hai train runs ở bước5, không lẫn với frozen metrics.

## 7. Optional controlled strategy replay

```bash
export BENCH_BATCH_SIZES=1 BENCH_SEEDS=42 BENCH_ITERATIONS=2
BENCH_METHOD=tlt BENCH_OUTPUT="$PWD/outputs/benchmarks/record_tlt" bash scripts/sweep_tlt_opd_reflex.sh
export TLT_STRATEGY_REPLAY="$PWD/outputs/benchmarks/record_tlt/runs/batch1_seed42_lr0.01_stream1/tlt/strategy_trace.jsonl"
BENCH_OUTPUT="$PWD/outputs/benchmarks/controlled_pair" bash scripts/sweep_tlt_opd_reflex.sh
unset TLT_STRATEGY_REPLAY
```

Giữ nguyên responses/length/sampling/checkpoint/iterations. Khi live batch/phase khác
trace, replay fail rõ ràng để tránh gán nhãn controlled cho một workload không khớp.

## Compatibility / finite sampler / online timing

```bash
python scripts/check_source_manifest.py
python scripts/validate_environment.py --require-cuda
python -m compileall -q .
python -m pytest -q
```

Sau khi đổi code/runtime adapter, phải tune lại nếu execution fingerprint không
khớp. Tuner và model/data paths ở trên giữ nguyên. Finite là default cấu hình mới.
Cả hai process phải dùng **cùng một mode**:

```bash
export OPD_SAMPLER_MODE=finite
METHOD=tlt BENCH_OUTPUT="$PWD/outputs/smoke/finite_tlt" bash run_benchmark.sh
METHOD=tlt_opd_reflex OPD_FAST_LRS=0 BENCH_OUTPUT="$PWD/outputs/smoke/finite_zero" bash run_benchmark.sh
METHOD=tlt_opd_reflex OPD_FAST_LRS=0.01 BENCH_OUTPUT="$PWD/outputs/smoke/finite_live" bash run_benchmark.sh
export OPD_SAMPLER_MODE=finite
```

Finite không thực hiện fallback cho sampled logits NaN/Inf; device assertion fail
thay vì thay target distribution. Cấu hình mặc định hiện dùng finite cho cả hai methods; có thể override strict.
Kiểm tra runtime trên stack/GPU B200 thật trước benchmark.

`BENCH_ONLINE_DRAFT=1` giữ objective/accumulation/optimizer cadence gốc và report
riêng generation/training/combined wall time. `tokens_per_s` và
`generation_tokens_per_s` chỉ tính generation; `combined_tokens_per_s` tính cả draft
training. Peak memory trong online benchmark gồm cả hai phases.
