# Lệnh chạy TltReflex trên server

## 1. Cài đúng môi trường một lần

Dùng venv riêng, không sửa `.venv` của SpecNaacl. Server offline cần copy
wheelhouse đã tạo trên máy online; xem [ENVIRONMENT.md](ENVIRONMENT.md).

```bash
cd /workspace/storage-shared/nlp/minhpn19/TltReflex
python3.12 -m venv .venv-tlt
source .venv-tlt/bin/activate
export PYTHON_BIN="$(command -v python)"
OFFLINE=1 INSTALL_RL=1 WHEELHOUSE="$PWD/wheelhouse" bash scripts/install_environment.sh
python -m pip check
python scripts/validate_environment.py --rl
```

Sau này chỉ activate venv và export PYTHON_BIN; không cài lại mỗi lần train.

## 2. Smoke và benchmark standalone trước

Qwen2.5-3B default dùng target:
`/workspace/storage-shared/models/Qwen2.5-3B-Instruct`, data:
`/workspace/storage-shared/nlp/minhpn19/data/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet`.
Draft/config/mapping vẫn ở
`../SpecNaacl/outputs/pretrain/qwen25_3b/latest_checkpoint`,
`latest_draft_config.json`, `latest_vocab_mapping.pt`. Không pretrain lại.

```bash
CUDA_VISIBLE_DEVICES=0 MODEL_KEY=qwen25_3b bash scripts/smoke_benchmark.sh

# Benchmark 2 fresh engine runs, cùng dataset/sampler/scheduler:
CUDA_VISIBLE_DEVICES=0 MODEL_KEY=qwen25_3b BENCHMARK_PROMPTS=128 \
  PAIR_DIR="$PWD/outputs/benchmarks/qwen25_3b_pair_seed42" bash benchmark_pair.sh

# Eager component profiling là pass riêng, không dùng để claim production tokens/s:
CUDA_VISIBLE_DEVICES=0 MODEL_KEY=qwen25_3b COMPONENT_PROFILE=1 \
  PAIR_DIR="$PWD/outputs/benchmarks/qwen25_3b_profile_seed42" bash benchmark_pair.sh
```

PAIR_DIR phải chưa tồn tại; mặc định tự tạo timestamp unique. Outputs:
`PAIR_DIR/tlt/report.json`, `PAIR_DIR/tlt_reflex/report.json`, kèm
`report.responses.jsonl`; component pass ở `*_components_eager/`.
Đổi model bằng MODEL_KEY: `qwen25_1p5b`, `qwen25_3b`, `qwen25_7b`,
`qwen25_14b`, `qwen3_1p7b`, `qwen3_4b`, `llama31_8b`.

Override resource và hyperparameter ví dụ:

```bash
METHOD=tlt_reflex MODEL_KEY=qwen3_1p7b CUDA_VISIBLE_DEVICES=4 \
  MODEL=/workspace/storage-shared/models/Qwen3-1.7B \
  DATASET=simplelr BATCH_SIZE=8 RESPONSES_PER_PROMPT=8 \
  MAX_NEW_TOKENS=2048 TEMPERATURE=1 TOP_P=0.95 SEED=42 \
  REFLEX_FEATURE_DIM=8 REFLEX_LR=0.05 REFLEX_WEIGHT_DECAY=0 \
  bash run_benchmark.sh
```

Không cần override draft path nếu naming SpecNaacl đúng như default.
Checkpoint khác: đặt DRAFT_CHECKPOINT, DRAFT_CONFIG, VOCAB_MAPPING và một
DRAFT_EXPORT mới. Conversion kiểm tra cấu trúc/mapping/features, không random
khi checkpoint thiếu/hỏng. Hai method phải dùng cùng export và settings.

## 3. Chạy GRPO bằng chính pipeline FastRL

Cần INSTALL_RL=1 và `validate_environment.py --rl` pass.
Train batch/LR dưới đây theo upstream, không LoRA trainer SpecNaacl.

```bash
CUDA_VISIBLE_DEVICES=0 MODEL_KEY=qwen25_3b METHOD=tlt \
  RL_BATCH_SIZE=64 RL_MINI_BATCH_SIZE=4 TARGET_LR=1e-6 NUM_EPOCHS=1 \
  bash run_rl.sh

CUDA_VISIBLE_DEVICES=0 MODEL_KEY=qwen25_3b METHOD=tlt_reflex \
  RL_BATCH_SIZE=64 RL_MINI_BATCH_SIZE=4 TARGET_LR=1e-6 NUM_EPOCHS=1 \
  bash run_rl.sh
```

Short first RL smoke (không full training):

```bash
CUDA_VISIBLE_DEVICES=0 MODEL_KEY=qwen25_3b METHOD=tlt \
  RL_BATCH_SIZE=4 RL_MINI_BATCH_SIZE=4 MAX_NEW_TOKENS=64 \
  bash run_rl.sh trainer.total_training_steps=1
CUDA_VISIBLE_DEVICES=0 MODEL_KEY=qwen25_3b METHOD=tlt_reflex \
  RL_BATCH_SIZE=4 RL_MINI_BATCH_SIZE=4 MAX_NEW_TOKENS=64 \
  bash run_rl.sh trainer.total_training_steps=1
```

Hoặc launcher từng model, file thường = TLT+Reflex, `_tlt.sh` = original TLT:

```bash
CUDA_VISIBLE_DEVICES=0 bash train_qwen25_3b.sh
CUDA_VISIBLE_DEVICES=0 bash train_qwen25_3b_tlt.sh
CUDA_VISIBLE_DEVICES=0 bash train_qwen3_1p7b.sh
CUDA_VISIBLE_DEVICES=0 bash train_qwen3_1p7b_tlt.sh
CUDA_VISIBLE_DEVICES=0 bash train_qwen3_4b.sh
CUDA_VISIBLE_DEVICES=0 bash train_qwen3_4b_tlt.sh
```

Hai file tương ứng cũng có cho 1.5B/7B/14B/Llama8B. RL output mặc định:
`outputs/rl/<model>_<method>_<timestamp>/logs/console.log`, `checkpoints/`,
`data/train.parquet` (converted VERL format từ data local). Đặt RUN_DIR để đổi;
không ghi đè một run tồn tại. Muốn resume dùng path run mới và explicit
Hydra override `trainer.resume_mode=resume_path trainer.resume_from_path=...`.
Validation mặc định OFF theo upstream (`test_freq=-1`, val_before_train=false);
EVAL_DATA mặc định trỏ train chỉ để thỏa schema, **không phải held-out eval**.
Muốn đánh giá, cung cấp parquet VERL held-out thật và override tần suất:

```bash
: "${EVAL_DATA:?Hãy đặt EVAL_DATA tới parquet VERL held-out thật của bạn}"
METHOD=tlt_reflex MODEL_KEY=qwen25_3b EVAL_DATA="$EVAL_DATA" \
  bash run_rl.sh trainer.test_freq=30 trainer.val_before_train=true
```

Path heldout ở ví dụ là tham số bắt buộc do người dùng cung cấp, không default
resource giả trong launcher. RL giữ ODT EAGLE3 OFF: upstream trainer factory
chưa support; enable=true là lỗi rõ ràng, không trainer thay thế.

## 4. Config/source/identity checks không train dài

```bash
python scripts/audit_upstream.py
DRY_RUN=true METHOD=tlt bash run_rl.sh
DRY_RUN=true METHOD=tlt_reflex bash run_rl.sh
DRY_RUN=true METHOD=tlt bash run_benchmark.sh
python -m compileall -q .
python -m pytest -q
python -m pip check
```

So sánh engine với fork upstream **chưa sửa** (hai processes riêng):

```bash
METHOD=tlt OUTPUT_DIR="$PWD/outputs/upstream_check/pristine" \
  bash run_benchmark.sh --pristine-upstream
METHOD=tlt OUTPUT_DIR="$PWD/outputs/upstream_check/off" bash run_benchmark.sh
python scripts/compare_upstream_outputs.py \
  outputs/upstream_check/pristine/report.json outputs/upstream_check/off/report.json
```

Adaptive MAB quyết định theo timing nên cùng seed không bảo đảm tree/output
bitwise giống nhau. Script báo fail khi output khác; không che bằng fallback.
Để kiểm tra identity deterministic riêng, dùng **cùng** `MAB_CONFIGS=''`
và `TEMPERATURE=0` cho cả hai, không trộn diagnostic đó với adaptive benchmark.
Launcher dùng `${MAB_CONFIGS:-default}` nên override list rỗng bằng
`--mab-configs ''` trên CLI cho benchmark (không tự tắt MAB mặc định).
