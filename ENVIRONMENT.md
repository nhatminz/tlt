# FastGRPO environment

Production dependencies nằm trong `requirements.txt`, copy pins từ SpecNaacl.
Python >=3.10; deployment `.python-version` là 3.12. Torch2.8.0/cu128,
Triton3.4.0, Transformers4.51.3, PEFT0.17.1. Không cần SGLang/verl/SpecForge.

```bash
bash scripts/bootstrap_environment.sh
source .venv/bin/activate
python scripts/validate_environment.py --require-cuda
```

Validation RTX3090 dùng Python3.10, Torch2.5.1/cu124, Triton3.1.0,
Transformers4.51.3 và PEFT0.17.1. Kết quả này không chứng nhận deployment pins/B200.
README/RUN_TLT_OPD.md mới thay tài liệu cũ đã chuyển vào legacy/sglang.
