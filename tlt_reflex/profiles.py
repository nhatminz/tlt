"""Reuse checked SpecNaacl calibration; distinguish TLT graph execution keys."""
import hashlib
import json
import os
from pathlib import Path
from tlt_reflex.ported.profiles import ProposalProfile,execution_key,profile_filename
ROOT=Path(__file__).resolve().parents[1]


def fingerprint(device=None,*,source=False):
    import torch,triton
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for real OPD tuning')
    manifest=json.loads((ROOT/'tlt_reflex/ported/SOURCE.json').read_text())
    digest=manifest['proposal_source_fingerprint']
    if not source:
        h=hashlib.sha256(digest.encode())
        for name in ('kernels.py','dispatch.py','state.py','ported/opd_reflex_kernels.py','ported/merge.py'):
            h.update((ROOT/'tlt_reflex'/name).read_bytes())
        digest=h.hexdigest()
    return dict(gpu=torch.cuda.get_device_name(device),compute_capability=list(torch.cuda.get_device_capability(device)),
        torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,kernel_sha256=digest)


def discover(v,rank,dtype,topk,device=None):
    current=execution_key(fingerprint(device),v,rank,dtype,topk)
    source=execution_key(fingerprint(device,source=True),v,rank,dtype,topk)
    explicit=os.environ.get('OPD_PROPOSAL_PROFILE','')
    directories=[Path(os.environ.get('OPD_PROPOSAL_PROFILE_DIR',str(ROOT.parent/'SpecNaacl/outputs/benchmarks/opd_proposals'))),
                 ROOT/'outputs/benchmarks/opd_proposals']
    paths=[Path(explicit)] if explicit else [p for d in directories for p in sorted(d.glob('*.json'))]
    source_candidate=None
    for path in paths:
        try:
            payload=json.loads(path.read_text());key=payload['execution_key']
            # Source calibration can be reused as an offline dispatch prior.
            # It does not claim graph timing is identical across runtimes.
            match=key==current
            source_match=all(key.get(k)==source[k] for k in ('gpu','compute_capability','vocab','rank','dtype','topk','kernel_sha256'))
            if not match and not source_match:raise ValueError('GPU/vocab/rank/dtype/kernel profile mismatch')
            selector=ProposalProfile(payload)
            if match:return selector,str(path)
            if source_candidate is None:source_candidate=(selector,str(path))
        except (ValueError,KeyError,TypeError,OSError):
            if explicit:raise ValueError(f'incompatible explicit OPD profile: {path}')
    if source_candidate is not None:
        print(f'Reusing SpecNaacl proposal calibration: {source_candidate[1]}; TLT graph overhead was not measured by that profile',flush=True)
        return source_candidate
    return None,''
