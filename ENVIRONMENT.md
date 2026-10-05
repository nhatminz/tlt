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

Checkout sạch không chứa upstream/wheels. Bootstrap/scripts tái tạo các artifacts
từ pinned git source, SHA256-verified sdist và tracked patches. Online builder
bootstrap chúng trước khi build/download full wheelhouse. Có thể copy nguyên
folder đã chuẩn bị (`upstream/` phải gồm `.git` của fastrl) hoặc chỉ copy checkout,
wheelhouse và offline git bundle:

```bash
PYTHON_BIN="$(command -v python)" bash scripts/create_offline_upstream_bundle.sh
```

Tạo bundle trên máy online; copy `artifacts/fastrl-bce3df7.bundle` sang server.
Đặt `FASTRL_GIT_SOURCE` khi installer/bootstrap để clone local bundle, không cần
internet. Unit tests không cần download/build FlashInfer wheel.

Trên server có Python3.12 sẵn:

```bash
cd /workspace/storage-shared/nlp/minhpn19/TltReflex
python3.12 -m venv .venv-tlt
source .venv-tlt/bin/activate
export PYTHON_BIN="$(command -v python)"
OFFLINE=1 INSTALL_RL=1 FASTRL_GIT_SOURCE="$PWD/artifacts/fastrl-bce3df7.bundle" \
  WHEELHOUSE="$PWD/wheelhouse" bash scripts/install_environment.sh
python -m pip check
python scripts/validate_environment.py --rl
```

`OFFLINE=1` không truy cập package index. CUDA toolkit/driver/Python là system
prerequisites, wheelhouse không tự cài chúng. Nên đặt venv/JIT cache trên SSD
local để tránh shared storage treo khi dlopen thư viện CUDA, như lỗi trước đó.
Không dùng cache tạm từ machine khác/architecture khác cho compiled GPU kernels.

## FlashInfer exact0.4.0 ABI adapter (corrected after review)

Public FlashInfer0.4.0 yêu cầu `apache-tvm-ffi==0.1.0b15` đã bị gỡ khỏi PyPI.
Không thể resolve reproducibly bằng wheel public nguyên bản. `upstream/flashinfer`
là official0.4.0 sdist + `patches/flashinfer_stable_ffi.patch`, metadata và
typed owning-Tensor accessor shim. Stable FFI0.1.0 `Tensor::operator->`
trả Object* thay vì TensorObj*; shim dùng typed `get()`, giữ nguyên object
ownership/FFI type/underlying data pointer và CUDA sampling math.
Claim trước đây “full official PR1960 applied” là **không chính xác**: PR đó
target một TensorView revision khác, không apply trực tiếp lên exact sdist này.
Reference liên quan: https://github.com/flashinfer-ai/flashinfer/pull/1960 .

```bash
bash scripts/bootstrap_flashinfer.sh
python scripts/check_flashinfer_abi.py
```

Đã kiểm tra host C++ accessor/type registration và gọi real TVM-FFI function
trên NumPy tensor (shape/data pointer alias đúng); chưa compile toàn bộ native
CUDA JIT. Wheel rebuild có thể khác SHA do metadata/timestamps; auditor kiểm
tra code ABI thực đóng gói so với tracked hashes, không bắt một hash chỉ có
ở máy developer. Build scripts tạo wheel trước khi install pinned requirements.
Offline muốn bootstrap source riêng: `FLASHINFER_ARCHIVE` trỏ official verified
sdist; nếu đã có verified wheelhouse thì installer không cần sdist.

Nếu folder cũ có fastrl extracted nhưng không `.git`, bootstrap từ chối overwrite.
Chủ động backup nó trước (`mv upstream upstream.legacy.<your-run-id>`) rồi bootstrap;
không dùng reset/rm để che sai source identity.

Tái tạo lock (online, dùng uv, không cần cho offline install):

```bash
uv pip compile requirements.in -c constraints.txt --overrides dependency-fixes.txt \
  --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
  --exclude-newer 2026-02-28 --prerelease if-necessary-or-explicit \
  --upgrade --no-annotate --no-header -o requirements.txt
```
