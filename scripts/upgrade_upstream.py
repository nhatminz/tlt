#!/usr/bin/env python3
"""Recognize the previous Fast-LK checkout; bootstrap replaces it with a backup."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def recognize(path):
    path=Path(path)
    hashes=json.loads((ROOT/'upstream_hashes.json').read_text())
    legacy=json.loads((ROOT/'patches/legacy_fast_lk_source_hashes.json').read_text())
    expected=dict(hashes,**legacy)
    for name,digest in expected.items():
        file=path/name
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
            raise ValueError('unrecognized or modified legacy upstream file: '+name)
    pristine=ROOT/'upstream/pristine_sglang_python'
    if pristine.is_dir():
        for source in pristine.rglob('*.py'):
            name='third-party/sglang/python/'+str(source.relative_to(pristine))
            if name not in legacy and (path/name).read_bytes()!=source.read_bytes():
                raise ValueError('unrecognized legacy SGLang edit: '+name)
    return True

if __name__=='__main__':recognize(sys.argv[1])
