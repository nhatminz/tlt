#!/usr/bin/env python3
"""Read-only source/provenance audit, no network and no heavy imports."""
import hashlib
import json
import argparse
import subprocess
import zipfile
from email.parser import Parser
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def audit(upstream_root=None,require_git=False,require_wheel=False,require_flashinfer=False):
    upstream=Path(upstream_root) if upstream_root else ROOT/'upstream'
    provenance=json.loads((ROOT/'PROVENANCE.json').read_text())
    allowed=set(provenance['fastrl']['modified_files'])
    for key,file in [('fastrl','fastrl_reflex.patch'),('flashinfer','flashinfer_stable_ffi.patch')]:
        field='reflex_patch_sha256' if key=='fastrl' else 'local_patch_sha256'
        if hashlib.sha256((ROOT/'patches'/file).read_bytes()).hexdigest()!=provenance[key][field]:
            raise AssertionError('tracked patch checksum mismatch: '+file)
    hashes=json.loads((ROOT/'upstream_hashes.json').read_text())
    changed=[]
    for name,expected in hashes.items():
        actual=hashlib.sha256((upstream/'fastrl'/name).read_bytes()).hexdigest()
        if actual!=expected:
            changed.append(name)
            if name not in allowed:raise AssertionError('unexpected upstream algorithm change: '+name)
    for name,expected in provenance['fastrl'].get('patched_sha256',{}).items():
        if hashlib.sha256((upstream/'fastrl'/name).read_bytes()).hexdigest()!=expected:
            raise AssertionError('modified file does not match tracked Reflex patch: '+name)
    if require_git:
        actual=subprocess.check_output(['git','-C',str(upstream/'fastrl'),'rev-parse','HEAD'],text=True).strip()
        if actual!=provenance['fastrl']['commit']:raise AssertionError('upstream commit mismatch')
        changes=subprocess.check_output(['git','-C',str(upstream/'fastrl'),'diff','--name-only','HEAD'],text=True).splitlines()
        if set(changes)!=allowed:raise AssertionError('unexpected git diff file set: '+repr(changes))
    # Pristine fork independently covers models, sampler, MAB, backend and tree
    # code, including files outside the targeted hash manifest.
    pristine=upstream/'pristine_sglang_python'
    patched=upstream/'fastrl/third-party/sglang/python'
    if not pristine.is_dir():raise FileNotFoundError('pristine source missing; run scripts/bootstrap_upstream.sh')
    count=0
    for source in pristine.rglob('*.py'):
        relative=source.relative_to(pristine)
        destination=patched/relative
        if source.read_bytes()!=destination.read_bytes():
            name='third-party/sglang/python/'+str(relative)
            if name not in allowed:raise AssertionError('unexpected SGLang change: '+name)
        count+=1
    if require_flashinfer:
        for name,digest in provenance['flashinfer']['patched_sha256'].items():
            if hashlib.sha256((upstream/'flashinfer'/name).read_bytes()).hexdigest()!=digest:
                raise AssertionError('FlashInfer ABI source mismatch: '+name)
    wheel=ROOT/'wheels/flashinfer_python-0.4.0-py3-none-any.whl'
    if not wheel.is_file() and require_wheel:raise FileNotFoundError('FlashInfer wheel missing; run scripts/build_wheelhouse.sh or offline bootstrap_flashinfer.sh')
    if wheel.is_file():
        # Build timestamps/git metadata can change wheel bytes; verify actual
        # packaged ABI code against tracked source, not a local-only old hash.
        with zipfile.ZipFile(wheel) as z:
            metadata=Parser().parsestr(z.read('flashinfer_python-0.4.0.dist-info/METADATA').decode())
            if metadata['Version']!='0.4.0' or 'apache-tvm-ffi==0.1.0' not in metadata.get_all('Requires-Dist',[]):
                raise AssertionError('FlashInfer wheel metadata is not the checked0.4/stable0.1 ABI artifact')
            for name,digest in provenance['flashinfer']['patched_sha256'].items():
                if name.startswith('csrc/'):
                    if hashlib.sha256(z.read('flashinfer/data/'+name)).hexdigest()!=digest:
                        raise AssertionError('FlashInfer wheel ABI code mismatch: '+name)
    return dict(protected_hash_files=len(hashes),pristine_python_files=count,
                changed_files=sorted(changed),flashinfer_wheel_checksum='OK' if wheel.is_file() else 'not built (not needed for source/unit tests)')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--upstream-root');p.add_argument('--require-git',action='store_true');p.add_argument('--require-wheel',action='store_true');p.add_argument('--require-flashinfer',action='store_true')
    a=p.parse_args();print(json.dumps(audit(a.upstream_root,a.require_git,a.require_wheel,a.require_flashinfer),indent=2))
