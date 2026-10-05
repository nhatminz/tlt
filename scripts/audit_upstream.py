#!/usr/bin/env python3
"""Read-only source/provenance audit, no network and no heavy imports."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def audit():
    provenance=json.loads((ROOT/'PROVENANCE.json').read_text())
    allowed=set(provenance['fastrl']['modified_files'])
    hashes=json.loads((ROOT/'upstream_hashes.json').read_text())
    changed=[]
    for name,expected in hashes.items():
        actual=hashlib.sha256((ROOT/'upstream/fastrl'/name).read_bytes()).hexdigest()
        if actual!=expected:
            changed.append(name)
            if name not in allowed:raise AssertionError('unexpected upstream algorithm change: '+name)
    # Pristine fork independently covers models, sampler, MAB, backend and tree
    # code, including files outside the targeted hash manifest.
    pristine=ROOT/'upstream/pristine_sglang_python'
    patched=ROOT/'upstream/fastrl/third-party/sglang/python'
    count=0
    for source in pristine.rglob('*.py'):
        relative=source.relative_to(pristine)
        destination=patched/relative
        if source.read_bytes()!=destination.read_bytes():
            name='third-party/sglang/python/'+str(relative)
            if name not in allowed:raise AssertionError('unexpected SGLang change: '+name)
        count+=1
    wheel=ROOT/'wheels/flashinfer_python-0.4.0-py3-none-any.whl'
    if hashlib.sha256(wheel.read_bytes()).hexdigest()!=provenance['flashinfer']['wheel_sha256']:
        raise AssertionError('bundled FlashInfer wheel checksum mismatch')
    return dict(protected_hash_files=len(hashes),pristine_python_files=count,
                changed_files=sorted(changed),flashinfer_wheel_checksum='OK')


if __name__=='__main__':print(json.dumps(audit(),indent=2))
