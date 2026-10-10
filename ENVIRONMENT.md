# FastGRPO installed-stack compatibility

`requirements.txt` giữ installation pins tham chiếu, không phải yêu cầu downgrade
environment hiện có. Python>=3.10. Validator port nguyên từ SpecNaacl hỗ trợ bounded
Torch/Transformers/PEFT API families và thực sự chạy target decoder/cache, draft
pretrain backward, AdamW/scheduler, LoRA/state và checkpoint roundtrip.

```bash
python scripts/validate_environment.py --require-cuda
# Chỉ dùng khi muốn kiểm tra exact reference pins:
python scripts/validate_environment.py --require-cuda --strict-versions
```

Transformers4.51.3 dùng API native; 5.12.1 dùng adapter trong
`helper/transformers_compat.py`. Không đổi draft architecture/attention/loss/RNG.
Không auto-install, auto-downgrade hoặc bỏ API probes.

`OPD_SAMPLER_MODE=finite` default trong training/benchmark config của cả hai methods.
Override `strict` để dùng validation/fallback cũ; chạy validator trên GPU B200 thật.
Chạy cả hai method bằng cùng sampler mode. B200 commands ở RUN_TLT_OPD.md.
