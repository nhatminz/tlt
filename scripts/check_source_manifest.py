#!/usr/bin/env python3
"""Verify self-contained source snapshots and declared exact/modified ports."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def check():
    manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    for entry in manifest['files']:
        snapshot=ROOT/entry['snapshot']
        if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('source snapshot modified: '+entry['snapshot'])
        if entry.get('destination'):
            actual=hashlib.sha256((ROOT/entry['destination']).read_bytes()).hexdigest()
            if actual!=entry['port_sha256']:raise ValueError('port differs from manifest: '+entry['destination'])
            if entry['port_status']=='exact' and actual!=entry['sha256']:raise ValueError('exact port differs from source')
    for name,expected in manifest.get('tlt_algorithm_files',{}).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('TLT implementation differs from manifest: '+name)
    return len(manifest['files'])

if __name__=='__main__':print(f'Source manifest verified: {check()} entries')
