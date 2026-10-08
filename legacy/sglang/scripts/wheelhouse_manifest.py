#!/usr/bin/env python3
"""Create/verify an offline wheel inventory and pinned requirement coverage."""
import argparse
from email.parser import Parser
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wheelhouse',default=str(ROOT/'wheelhouse'))
    p.add_argument('--create',action='store_true');p.add_argument('--rl',action='store_true')
    a=p.parse_args();directory=Path(a.wheelhouse);manifest=directory/'MANIFEST.json'
    if not directory.is_dir():p.error('wheelhouse missing')
    if list(directory.glob('*.part')):p.error('incomplete .part downloads present')
    expected=[Requirement(s) for s in (ROOT/'requirements.txt').read_text().splitlines() if s and not s.startswith('#')]
    expected.extend(Requirement(s) for s in ('wheel==0.45.1','pip==26.2.1','sglang==0.5.3.post2','verl==0.5.0.dev0'))
    if a.rl:expected.append(Requirement('flash-attn==2.8.3'))
    if a.create:
        records=[]
        for file in sorted(directory.glob('*.whl')):
            with zipfile.ZipFile(file) as z:
                # Ray vendors other distributions inside its wheel. Only the
                # TOP-LEVEL dist-info belongs to the installable distribution.
                paths=[n for n in z.namelist() if n.endswith('.dist-info/METADATA') and n.count('/')==1]
                if len(paths)!=1:raise ValueError('invalid wheel metadata: '+file.name)
                meta=Parser().parsestr(z.read(paths[0]).decode())
            records.append(dict(filename=file.name,name=canonicalize_name(meta['Name']),version=meta['Version'],
                                bytes=file.stat().st_size,sha256=sha(file)))
        doc=dict(target=dict(python='3.12',platform='Linux x86_64',torch='2.8.0',cuda='12.8',glibc_min='2.28'),
                 rl_included=a.rl,wheels=records,bytes=sum(r['bytes'] for r in records))
    else:
        if not manifest.is_file():p.error('MANIFEST.json missing; builder must create it')
        doc=json.loads(manifest.read_text())
        for r in doc['wheels']:
            file=directory/r['filename']
            if not file.is_file() or sha(file)!=r['sha256']:raise ValueError('missing/corrupt wheel: '+r['filename'])
    available={r['name']:Version(r['version']) for r in doc['wheels']}
    missing=[str(r) for r in expected if canonicalize_name(r.name) not in available or available[canonicalize_name(r.name)] not in r.specifier]
    if 'verl' not in available:missing.append('official pinned VERL wheel')
    if missing:raise SystemExit('incomplete wheelhouse: '+', '.join(missing))
    if a.create:manifest.write_text(json.dumps(doc,indent=2)+'\n')
    print(json.dumps(dict(wheels=len(doc['wheels']),size_gib=round(doc['bytes']/2**30,3),coverage='PASS',hashes='PASS',rl=a.rl),indent=2))


if __name__=='__main__':main()
