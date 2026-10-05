# Dedicated TLT environment (Linux x86_64 / Python3.12 / B200)

Không dùng venv SpecNaacl. Fork TLT API khác stock SGLang; reinstall stock
SGLang0.5.18/Torch2.13 vào TLT sẽ không cung cấp adaptive/BEG fork này.

Core stack từ metadata chính thức của vendored FastRL/SGLang: Torch2.8.0
(CUDA12.8 wheel), Transformers4.57.1, SGLang fork0.5.3.post2,
sgl-kernel0.3.15, FlashInfer0.4.0, Triton3.4.0. Full 206 exact pins ở
`requirements.txt`, generated từ `requirements.in`, `constraints.txt`,
`dependency-fixes.txt`. Không dùng stdlib như dependency. Core native imports
được kiểm tra bằng `scripts/validate_environment.py`; installer chạy pip check.
Resolver metadata đã pass; **chưa install/import toàn bộ stack trên B200**.

## Online builder / installation

Cần Python3.12, driver hỗ trợ GPU và CUDA wheel; FlashInfer JIT cần CUDA
toolkit/nvcc, GCC/g++, Ninja tương thích trên server. CUDA12.8 toolkit có hỗ
trợ Blackwell sm100. PyTorch wheel chứa CUDA runtime, **không chứa đầy đủ nvcc**.
CUDA Python/bindings được constrain12.8.0 dựa trên NVIDIA package metadata,
thỏa CUTLASS DSL4.2.1 (>=12.8), tránh vô tình kéo CUDA13 vào stack cu128.
Không tự nâng driver hoặc trộn LD_LIBRARY_PATH CUDA không tương thích.

```bash
cd /workspace/storage-shared/nlp/minhpn19/TltReflex
python3.12 -m venv .venv-tlt
source .venv-tlt/bin/activate
export PYTHON_BIN="$(command -v python)"
# Pin torch CUDA build explicitly, then installer locks the remaining stack:
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
INSTALL_RL=1 bash scripts/install_environment.sh
python -m pip check
python scripts/validate_environment.py
python scripts/validate_environment.py --rl
```

`INSTALL_RL=1` cài exact FlashAttention wheel2.8.3 từ README upstream (cp312,
cu12, torch2.8, CXX11 ABI TRUE). Bắt buộc cho official FSDP trainer; benchmark
standalone Triton-attention không yêu cầu FlashAttention. Không cài extras
`verl[sglang]`: extras metadata cũ của upstream sẽ kéo một stock SGLang khác.
SGLang fork và VERL chỉ cài `--no-deps --no-build-isolation -e` từ folder vendor.

## Offline server

Trên **máy có internet**, Linux x86_64 với Python3.12, trong builder venv riêng:

```bash
python3.12 -m venv /tmp/tlt-wheel-builder
source /tmp/tlt-wheel-builder/bin/activate
PYTHON_BIN="$(command -v python)" INSTALL_RL=1 bash scripts/build_wheelhouse.sh
```

Copy nguyên `TltReflex/` gồm `wheelhouse/`, `wheels/`, `upstream/` sang server.
Không chỉ copy root scripts. Bundle mặc định hiện có source vendor và custom
FlashInfer wheel, **không phải** toàn bộ 206 dependency wheels. Script builder
tải/build toàn bộ phần còn lại, không cần git clone ở server.

Trên server có Python3.12 sẵn:

```bash
cd /workspace/storage-shared/nlp/minhpn19/TltReflex
python3.12 -m venv .venv-tlt
source .venv-tlt/bin/activate
export PYTHON_BIN="$(command -v python)"
OFFLINE=1 INSTALL_RL=1 WHEELHOUSE="$PWD/wheelhouse" bash scripts/install_environment.sh
python -m pip check
python scripts/validate_environment.py --rl
```

`OFFLINE=1` không truy cập package index. CUDA toolkit/driver/Python là system
prerequisites, wheelhouse không tự cài chúng. Nên đặt venv/JIT cache trên SSD
local để tránh shared storage treo khi dlopen thư viện CUDA, như lỗi trước đó.
Không dùng cache tạm từ machine khác/architecture khác cho compiled GPU kernels.

## FlashInfer mandatory ABI backport

Public FlashInfer0.4.0 yêu cầu `apache-tvm-ffi==0.1.0b15` đã bị gỡ khỏi PyPI.
Không thể resolve reproducibly bằng wheel public nguyên bản. `upstream/flashinfer`
là official0.4.0 sdist + official PR1960 C++ TensorView ABI migration sang stable
FFI0.1.0, cùng build/runtime metadata tương ứng. Không đổi CUDA sampling math.
Không phải chỉ sửa version check hay dùng `--no-deps` để che incompatibility.
Bundled wheel đã build thành công trên Python3.12; checksum/provenance ở
`PROVENANCE.json`; native JIT/runtime smoke còn phải kiểm tra trên B200.
Nguồn: https://github.com/flashinfer-ai/flashinfer/pull/1960 .

Tái tạo lock (online, dùng uv, không cần cho offline install):

```bash
uv pip compile requirements.in -c constraints.txt --overrides dependency-fixes.txt \
  --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
  --exclude-newer 2026-02-28 --prerelease if-necessary-or-explicit \
  --upgrade --no-annotate --no-header -o requirements.txt
```
