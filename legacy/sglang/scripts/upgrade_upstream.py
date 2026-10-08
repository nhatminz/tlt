#!/usr/bin/env python3
"""Recognize prior OPD/Fast-LK checkouts; bootstrap replaces it with a backup."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def recognize(path):
    path=Path(path)
    hashes=json.loads((ROOT/'upstream_hashes.json').read_text())
    versions=[('previous OPD',ROOT/'patches/previous_opd_source_hashes.json'),
              ('legacy Fast-LK',ROOT/'patches/legacy_fast_lk_source_hashes.json')]
    pristine=ROOT/'upstream/pristine_sglang_python'
    for label,manifest in versions:
        known=json.loads(manifest.read_text());expected=dict(hashes,**known)
        if not all((path/name).is_file() and hashlib.sha256((path/name).read_bytes()).hexdigest()==digest for name,digest in expected.items()):continue
        if pristine.is_dir():
            unchanged=True
            for source in pristine.rglob('*.py'):
                name='third-party/sglang/python/'+str(source.relative_to(pristine))
                if name not in known and (not (path/name).is_file() or (path/name).read_bytes()!=source.read_bytes()):
                    unchanged=False;break
            if not unchanged:continue
        print('Recognized '+label+' source')
        return True
    raise ValueError('unrecognized or locally modified upstream; existing source preserved')

if __name__=='__main__':recognize(sys.argv[1])
