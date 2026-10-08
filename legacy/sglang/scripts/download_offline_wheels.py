#!/usr/bin/env python3
"""Download exact locked wheels for CPython3.12/Linux x86_64, verify PyPI SHA256.

No installation into the current environment. Source-only packages are fetched
separately and must be built to wheels before offline installation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.request
from packaging.tags import cpython_tags,compatible_tags
from packaging.utils import parse_wheel_filename,canonicalize_name

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'TltReflex-offline-builder/1'}),timeout=120)


def fetch(file,directory):
    path=directory/file['filename']
    if path.is_file() and digest(path)==file['digests']['sha256']:return path
    partial=path.with_suffix(path.suffix+'.part')
    for attempt in range(3):
        try:
            with request(file['url']) as response,partial.open('wb') as out:
                shutil.copyfileobj(response,out,length=2**20)
            if digest(partial)!=file['digests']['sha256']:raise ValueError('SHA256 mismatch: '+file['filename'])
            partial.replace(path);return path
        except Exception:
            if attempt==2:raise
            time.sleep(attempt+1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default=str(ROOT/'wheelhouse'))
    p.add_argument('--sdists',default=str(ROOT/'artifacts/offline-sdists'))
    p.add_argument('--jobs',type=int,default=4)
    a=p.parse_args()
    if sys.version_info[:2]!=(3,12):p.error('run with CPython3.12 builder')
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    sdists=Path(a.sdists);sdists.mkdir(parents=True,exist_ok=True)
    platforms=[f'manylinux_2_{minor}_x86_64' for minor in range(28,4,-1)]
    platforms+=['manylinux2014_x86_64','manylinux2010_x86_64','manylinux1_x86_64','linux_x86_64']
    supported=list(cpython_tags((3,12),abis=['cp312'],platforms=platforms))
    supported+=list(compatible_tags((3,12),interpreter='cp312',platforms=platforms))
    rank={t:i for i,t in enumerate(supported)}
    pins=[s.strip() for s in (ROOT/'requirements.txt').read_text().splitlines() if s.strip() and not s.startswith('#')]
    pins+=['wheel==0.45.1','pip==26.2.1']  # actual builder pip version, exact public wheel
    pins=[s for s in pins if not s.startswith('flashinfer-python==')]
    def download(pin):
        name,version=pin.split('==')
        with request(f'https://pypi.org/pypi/{name}/{version}/json') as response:meta=json.load(response)
        candidates=[]
        for f in meta['urls']:
            if f['packagetype']!='bdist_wheel' or f.get('yanked'):continue
            _,_,_,tags=parse_wheel_filename(f['filename'])
            scores=[rank[t] for t in tags if t in rank]
            if scores:candidates.append((min(scores),f))
        if candidates:
            f=min(candidates,key=lambda x:x[0])[1];kind='wheel';directory=output
        else:
            sources=[f for f in meta['urls'] if f['packagetype']=='sdist' and not f.get('yanked')]
            if not sources:raise ValueError('no compatible wheel/source for '+pin)
            f=sources[0];kind='sdist';directory=sdists
        path=fetch(f,directory)
        return dict(requirement=pin,kind=kind,filename=path.name,sha256=f['digests']['sha256'],url=f['url'],size=path.stat().st_size)
    records=[];failures=[]
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        pending={pool.submit(download,pin):pin for pin in pins}
        for future in as_completed(pending):
            try:
                record=future.result();records.append(record)
                print(f'{len(records)}/{len(pins)} {record["kind"]}: {record["filename"]}',flush=True)
            except Exception as e:
                failures.append(dict(requirement=pending[future],error=str(e)))
                print(f'FAILED {pending[future]}: {e}',flush=True)
    receipt=dict(target='CPython3.12 Linux x86_64 manylinux<=2.28',records=sorted(records,key=lambda x:x['requirement']),failures=failures)
    (output/'download_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if failures:raise SystemExit('some artifacts failed; see download_receipt.json, do not call bundle complete')
    print('Source-only wheels to build:',[r['requirement'] for r in records if r['kind']=='sdist'],flush=True)


if __name__=='__main__':main()
