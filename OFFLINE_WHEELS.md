# Bộ wheel offline cho TltReflex — 2026-10-06

Target: **Python3.12 / Linux x86_64 / glibc>=2.28 / Torch2.8.0 CUDA12.8**.
Không dùng bộ này để ghi đè môi trường SpecNaacl/Torch2.11 hoặc Torch2.13.

Đã chuẩn bị `wheelhouse/`: 211 wheels (~4.80 GiB), gồm toàn bộ206 package
trong requirements, pip/wheel, FlashAttention2.8.3, SGLang fork0.5.3.post2
và VERL0.5.0.dev0 built từ pinned FastRL. FlashInfer là bản0.4.0 có adapter
ABI đã ghi provenance; không lấy wheel public với beta dependency bị gỡ.
Antlr4/Pylatexenc source-only đã build thành wheel, server không cần build chúng.

`MANIFEST.json` ghi tên/version/size/SHA256. `download_receipt.json` ghi URL
và SHA256 chính thức từ PyPI. FlashAttention lấy nguyên bytes release chính thức
cho cu12/torch2.8/CXX11ABI TRUE/cp312. Chỉ rename filename về version2.8.3 để
khớp METADATA, tránh pip26 từ chối local-build suffix; không sửa binary.

## Copy và cài trên server

Copy cả bộ archive được chuẩn bị ở `artifacts/` sang server. Archive chứa folder
`TltReflex/`: code/config/scripts, wheelhouse và offline FastRL git bundle.
Không chứa model/data, venv test, outputs hay driver/CUDA toolkit.
Nếu server đã có project với code riêng, extract archive vào thư mục staging rồi
copy `wheelhouse/`, `artifacts/fastrl-bce3df7.bundle` và các scripts mới; không
ghi đè edits của bạn một cách mù quáng.

```bash
cd /workspace/storage-shared/nlp/minhpn19/TltReflex
python3.12 -m venv .venv-tlt-offline
source .venv-tlt-offline/bin/activate
export PYTHON_BIN="$(command -v python)"

INSTALL_RL=1 WHEELHOUSE="$PWD/wheelhouse" bash scripts/install_offline_wheels.sh
python -m pip check
python scripts/validate_environment.py --rl
```

Installer chỉ dùng `--no-index --find-links`; không source-build, không PyPI,
không GitHub. Thiếu vendored source thì clone local git bundle đã kèm sẵn.
Source legacy extracted không `.git` chỉ được giữ nếu pass audit hashes.
Không cần chạy installer online cũ trên server offline.

Driver B200 và CUDA toolkit/nvcc12.8 vẫn phải có sẵn. Wheels cung cấp CUDA runtime
libraries nhưng không thay NVIDIA kernel driver hay toàn bộ CUDA compiler.
Đã kiểm tra resolver offline; full install/pip-check và native-import evidence
được ghi trong `artifacts/offline-build/` cùng handoff report. Native B200
generation/training smoke vẫn cần chạy trên server, không suy ra từ pip check.

Kết quả thực tế 2026-10-06: đủ211 wheels/SHA256; offline resolver PASS; cài cả
211 package trong venv test mới bằng --no-index PASS; pip check PASS. Torch
2.8.0+cu128/CXX11ABI TRUE, Transformers4.57.1, PEFT0.17.1, Pandas3.0.1 và
LlamaForCausalLM import PASS. FlashAttention và sgl-kernel binary import PASS
khi diagnostic chỉ đúng directory libcudart trong wheel. Không dùng directory
runtime-wheel đó như CUDA toolkit để train: nó không cung cấp nvcc. Import
SG default/FlashInfer optional allocator ở máy test gặp thiếu system CUDA lib/
headers; vì vậy cần CUDA_HOME trỏ toolkit thật trên B200. Chưa benchmark GPU.

Không copy `artifacts/offline-verify-*` hoặc venv test sang server; chỉ dùng
archive/wheelhouse đã đóng gói. Test venv không phải một môi trường portable.

## Cài chọn lọc nếu đã có môi trường TLT đúng stack

Pip sẽ giữ package đã đáp ứng version; không bắt buộc cài lại mọi wheel:

```bash
python -m pip install --no-index --find-links "$PWD/wheelhouse" -r requirements.txt
python -m pip install --no-index --find-links "$PWD/wheelhouse" --no-deps \
  sglang==0.5.3.post2 verl==0.5.0.dev0 flash-attn==2.8.3
python -m pip check
```

`wheels/`, `wheelhouse/`, `artifacts/` vẫn gitignored vì là binary artifacts lớn.
Để chuyển máy dùng archive/scp/rsync, không chỉ git clone. Không upload các
binary này lên GitHub thường bằng một commit nhiều GB.
