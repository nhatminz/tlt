# Offline artifacts report — 2026-10-06

- Target CPython3.12 / Linux x86_64 / glibc>=2.28 / Torch2.8.0+cu128.
- 211 wheels, total4.797 GiB; all206 locked runtime packages plus wheel/pip,
  exact official FlashAttention2.8.3 ABI-TRUE wheel, pinned TLT SGLang fork and VERL.
- Every public PyPI download verified against release metadata SHA256.
  Two pure-Python source-only packages (antlr4-runtime4.9.3, pylatexenc2.10)
  built locally to wheels. No source build needed on offline server.
- Patched FlashInfer wheel audited against tracked source hashes; SGLang/VERL
  wheels built from the verified pinned/patched upstream, not stock PyPI SGLang.
- Official FlashAttention filename had local-version suffix inconsistent with
  METADATA2.8.3. pip26 rejected it in find-links. Only filename normalized;
  binary bytes unchanged. Actual offline resolver then passed.
- MANIFEST.json covers all211 wheel names/versions/sizes/SHA256. Ray's nested
  vendored METADATA excluded; only top-level wheel dist-info inspected.
- Real new venv installation with --no-index/--find-links: PASS. Full pip check:
  No broken requirements found. Existing SpecNaacl/runtime environments unchanged.
- Core imports/model class: PASS (Torch2.8+cu128, CXX11ABI TRUE; Transformers4.57.1,
  PEFT0.17.1, Pandas3.0.1, datasets/safetensors, LlamaForCausalLM).
- FlashAttention and sgl-kernel native binary imports: PASS with diagnostic
  CUDA_HOME selecting wheel's libcudart directory. Default system CUDA paths
  are absent here; SG default import fails there, FlashInfer optional Torch-DLPack
  compilation warns missing CUDA headers. nvcc absent. This is NOT a validated
  CUDA toolkit or native B200 model/engine/training run.
- Existing scoped regression suite:64 passed. Shell/compile checks passed.
- This host's python -m venv on the relocatable uv interpreter initially had
  /install stdlib lookup error; an isolated uv-created venv was used for the
  actual offline-install verification. This host-specific venv failure does not
  change package/wheel platform; server should use its real Python3.12.
- No training, model/dataset replacements, cloud uploads, driver installation,
  or modification to SpecNaacl. Binary artifacts remain gitignored by design.

Installation instructions: OFFLINE_WHEELS.md. Download/source/install logs are
retained locally under artifacts/offline-build (not part of a portable venv).
