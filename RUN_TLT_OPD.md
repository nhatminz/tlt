# Chạy official TLT và TLT + Reflex OPD trên B200

Code mới chỉ sửa TltReflex. SpecNaacl là nguồn toán học/config/path, không bị sửa.
Hai mode production là `METHOD=tlt` và `METHOD=tlt_opd_reflex`. Fast-LK cũ và biến
`REFLEX_LR` không còn được dùng. Cùng target, pretrained EAGLE3 export, d2t/t2d,
data, sampling và cấu hình TLT/MAB được dùng cho hai mode.

## 1. Đồng bộ code và chuẩn bị runtime

Đưa toàn bộ TltReflex đã sửa lên cạnh SpecNaacl trên B200. Không chép venv CUDA
12.4/Python3.10 từ máy 3090 lên B200. Giữ venv riêng của TLT (Python3.12,
Torch2.8/cu128, SGLang fork pinned), không dùng venv SpecNaacl để chạy TLT.

```bash
cd /workspace/storage-shared/nlp/minhpn19/TltReflex
source .venv/bin/activate                    # venv TLT đã cài đúng stack
export PYTHON_BIN="$(command -v python)"
export CUDA_VISIBLE_DEVICES=0
export SOURCE_SPECNAACL_ROOT="$(cd ../SpecNaacl && pwd)"
export MODEL_KEY=qwen25_3b
export OPD_FAST_LR=0.01
export OPD_UPDATE_STREAM=1
export OPD_RANK=8 OPD_TOPK=16
export OPD_VISITED_WEIGHT=1.0 OPD_FRONTIER_WEIGHT=1.0
export OPD_TRAIN_PROJECTOR=0 OPD_PROFILE=0

bash scripts/bootstrap_upstream.sh
"$PYTHON_BIN" scripts/validate_environment.py --rl
"$PYTHON_BIN" -m pytest -q
```

Bootstrap ưu tiên Git bundle local `artifacts/fastrl-bce3df7.bundle` nếu có;
không có bundle thì fetch repo official, hoặc dùng FASTRL_GIT_SOURCE chỉ định bundle.
Bootstrap fetch đúng commit `bce3df7a4d46473912e9b81bf47bca419729557f`, apply OPD
patch và audit. Checkout Fast-LK cũ chỉ được migrate nếu checksum được nhận diện;
source cũ được giữ trong `upstream.legacy.fast_lk.<timestamp>`. Nếu source upstream
đã được sửa riêng, bootstrap sẽ báo lỗi và giữ nguyên; có thể chuẩn bị checkout mới
bằng `UPSTREAM_ROOT=/path/to/new/upstream bash scripts/bootstrap_upstream.sh`,
sau đó dùng thư mục mới này làm `upstream` của TltReflex. Không cần pretrain lại.

Nếu chưa có venv TLT, dùng hướng dẫn dependency hiện có trong ENVIRONMENT.md /
OFFLINE_WHEELS.md và `scripts/install_environment.sh` hoặc
`scripts/install_offline_wheels.sh`; wheelhouse và các pin không đổi bởi OPD port.

Paths mặc định Qwen2.5-3B, trùng SpecNaacl:

- Target: `/workspace/storage-shared/models/Qwen2.5-3B-Instruct`.
- Data: `/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/train.parquet`.
- Draft: `../SpecNaacl/outputs/pretrain/qwen25_3b/latest_checkpoint`.
- Config: `../SpecNaacl/outputs/pretrain/qwen25_3b/latest_draft_config.json`.
- Mapping: `../SpecNaacl/outputs/pretrain/qwen25_3b/latest_vocab_mapping.pt`.

Exporter tự tạo `outputs/draft_exports_opd/qwen25_3b`. Cả baseline và OPD dùng cùng
artifact này. Đổi namespace export để không overwrite artifact Fast-LK cũ.
Override bằng MODEL, DATASET_PATH, DRAFT_CHECKPOINT, DRAFT_CONFIG, VOCAB_MAPPING,
DRAFT_EXPORT nếu cần. DATASET=dapo/gsm8k chọn cùng data root như SpecNaacl.

## 2. Dùng profile bạn đã tune trong SpecNaacl

```bash
export OPD_PROPOSAL_PROFILE_DIR="$SOURCE_SPECNAACL_ROOT/outputs/benchmarks/opd_proposals"
unset OPD_PROPOSAL_PROFILE OPD_TUNE_OUTPUT
```

Không cần chạy lại lệnh tune SpecNaacl bạn đã chạy. TLT tự tìm JSON khớp GPU,
compute capability, compact vocab, rank, dtype, TopK và fingerprint kernel OPD
đã port. Nếu muốn chỉ định JSON đã có:

```bash
export OPD_PROPOSAL_PROFILE="$SOURCE_SPECNAACL_ROOT/outputs/benchmarks/opd_proposals/<ten-profile-that>.json"
```

Explicit profile sai sẽ fail rõ; tự discovery không tìm được profile phù hợp thì
báo `uncalibrated fallback` và dùng threshold an toàn, không tune trong generation.
Source profile là calibration của kernel gốc. Wrapper và graph của TLT khác, nên
chi phí source không được gọi là benchmark TLT. Phiên bản Torch/Triton khác có
thể đổi crossover; native TLT graph profile được ưu tiên nếu có.

Có thể tune riêng graph TLT sau smoke (không sửa hay ghi vào SpecNaacl):

```bash
MODEL_KEY=qwen25_3b bash scripts/tune_tlt_opd_proposals.sh
```

Output ở `TltReflex/outputs/benchmarks/opd_proposals`. Auto dispatch dùng
sparse/fused dựa trên device active_count, không host `.item()`. GEMM có workspace
riêng và chỉ allocate khi chọn explicit trước capture:
`OPD_PROPOSAL_MODE=gemm`; source chọn GEMM tối ưu không được giả định là tối ưu
cho graph TLT. Tuner đo cả sparse, fused, GEMM và kiểm tra bitwise parity.

## 3. Smoke native trước khi benchmark hoặc train

```bash
bash scripts/smoke_tlt_opd.sh
```

Lệnh này chạy ba engine: TLT, OPD LR=0, OPD LR=0.01; mặc định batch1,
responses1, prompts2, max-new-tokens64. Mỗi mode có report.json / summary.csv /
responses.jsonl trong thư mục smoke riêng. Đây là smoke native cần B200 và model
thật; các CUDA fixture đã kiểm tra trên 3090 không thay thế bước này.

## 4. Frozen benchmark công bằng

Pair một cấu hình LR/stream, các batch1,2,4,8,16,32:

```bash
MODEL_KEY=qwen25_3b BENCH_SEEDS=42,43 \
OPD_FAST_LRS=0.01 OPD_STREAMS=1 \
BENCH_BATCH_SIZES=1,2,4,8,16,32 RESPONSES_PER_PROMPT=1 \
BENCHMARK_PROMPTS=64 MAX_NEW_TOKENS=2048 MAX_PROMPT_LENGTH=256 \
bash benchmark_tlt_opd_pair.sh
```

Sweep toàn bộ LR/stream:

```bash
MODEL_KEY=qwen25_3b OPD_FAST_LRS=0.001,0.01,0.05,0.1 OPD_STREAMS=0,1 \
BENCH_SEEDS=42,43 BENCH_BATCH_SIZES=1,2,4,8,16,32 \
RESPONSES_PER_PROMPT=1 BENCHMARK_PROMPTS=64 MAX_NEW_TOKENS=2048 \
MAX_PROMPT_LENGTH=256 bash scripts/sweep_tlt_opd_reflex.sh
```

BATCH_SIZE là số prompts; tổng requests bằng BATCH_SIZE*RESPONSES_PER_PROMPT.
Responses1 trong sweep cho actual request batches1..32. Training giữ responses8
như SpecNaacl. Benchmark mặc định giữ official BEG/MAB/threshold32; trên batch
requests lớn hơn threshold, TLT có thể tạm disable speculation theo upstream.
Không chỉnh threshold để ép OPD có nhiều feedback hơn.

Sweep xuất `report.json`, `summary.csv`, `responses.jsonl`,
`fastest_observed.env`. Recommendation chỉ có khi delta verified AAL > 0 và
throughput >= baseline của cùng batch/seed. Nếu không có cấu hình đạt cả hai,
file env chỉ có comment. Chưa có kết quả B200 của implementation TLT này.

Thông số giữa cặp: target/draft/prompts/seed/sampling/TLT/MAB đều giống nhau.
BEG/MAB chọn strategy theo measured timing nên trajectory có thể khác; không
claim stochastic token identity trên adaptive runs. Cold logits identity không
ép native floating top-k tie convention: OPD tie chọn compact ID nhỏ trước.

Component profiling chạy riêng, không dùng throughput profiling để claim win:

```bash
COMPONENT_PROFILE=1 MODEL_KEY=qwen25_3b BATCH_SIZE=8 \
RESPONSES_PER_PROMPT=1 BENCHMARK_PROMPTS=16 MAX_NEW_TOKENS=512 \
bash benchmark_pair.sh
```

Pair ghi hai throughput runs profile OFF và hai diagnostic eager runs profile ON
trong các thư mục `*_components_eager`. Inner event timings trong graph không
được đo: các field không available là null. OPD overhead loại wait để tránh
cộng double-count với công việc side stream; wall time/tokens/s là kết luận chính.
Root refresh draft-head time được ghi riêng `opd_root_head_time_ms`.

## 5. Chạy RL hai bước, rồi chạy dài

```bash
# Official FastRL GRPO, cùng fixed pretrained EAGLE3; projector A frozen.
# Hai lệnh là hai training jobs riêng, không chạy đồng thời trên cùng GPU.
bash train_qwen25_3b_tlt.sh trainer.total_training_steps=2
bash train_qwen25_3b.sh trainer.total_training_steps=2

# Sau smoke / benchmark:
bash train_qwen25_3b_tlt.sh
bash train_qwen25_3b.sh
```

RL_BATCH_SIZE mặc định64 (prompt wave của official FastRL), responses8, target
LR1e-5 từ paired launchers; train source/draft/data giống SpecNaacl. Đây không phải
trainer của SpecNaacl: không port draft optimizer hoặc Spot Trainer EAGLE1 sang
EAGLE3. `OPD_TRAIN_PROJECTOR=1` sẽ fail rõ. A được load/frozen, B adapt online.

Output training: `outputs/rl/<model>_<method>_<timestamp>/logs/console.log` và
checkpoints target theo FastRL. OPD state/counters được expose ở server-info RPC;
không thêm polling per-token hoặc hứa có các CSV per-iteration của SpecNaacl.

14 wrappers vẫn được giữ cho qwen25_1p5b/3b/7b/14b, qwen3_1p7b/4b,
llama31_8b: file thường chạy OPD, file `_tlt.sh` chạy baseline.
TP_SIZE=1 là production OPD hiện tại. TP>1, DP-attention, overlap worker V2,
quantized/scaled head fail rõ để tránh reconstruction sai. Baseline TLT giữ
upstream semantics và không allocate OPD state / launch OPD kernels.

Projector mới từ head được ghi `head_basis_initialized`, trained flag=false.
Checkpoint có flag trained được preserve là `trained`; tensor có sẵn nhưng không
có bằng chứng optimizer update được giữ nguyên và ghi
`checkpoint_training_unverified`, không gọi là learned.
