"""Strict TLT-native proposal calibration, validated only during setup."""
import hashlib,json,os
from pathlib import Path
from tlt_reflex.ported.profiles import ProposalProfile,execution_key,profile_filename
ROOT=Path(__file__).resolve().parents[1]
PROFILE_KIND='tlt_native'
CALIBRATION_VERSION='tlt-scattered-effective-contexts-v3'


def fingerprint(device=None):
    import torch,triton
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for real TLT OPD tuning')
    kernel=hashlib.sha256()
    for name in ('kernels.py','dispatch.py','state.py','ported/opd_reflex_kernels.py','ported/merge.py'):
        kernel.update(name.encode());kernel.update((ROOT/'tlt_reflex'/name).read_bytes())
    execution=hashlib.sha256(CALIBRATION_VERSION.encode())
    # Runtime hooks, pin and calibration procedure affect TLT graph costs.
    for name in ('patches/fastrl_reflex.patch','PROVENANCE.json','scripts/tune_tlt_opd_proposals.py'):
        execution.update(name.encode());execution.update((ROOT/name).read_bytes())
    execution.update(kernel.digest())
    return dict(gpu=torch.cuda.get_device_name(device),compute_capability=list(torch.cuda.get_device_capability(device)),
        torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,kernel_sha256=kernel.hexdigest(),
        tlt_execution_sha256=execution.hexdigest(),calibration_version=CALIBRATION_VERSION)


def validate_native_profile(payload,key):
    if payload.get('profile_kind')!=PROFILE_KIND:
        raise ValueError('requires TLT-native profile, not SpecNaacl/source calibration')
    actual=payload.get('execution_key',{})
    if 'tlt_execution_sha256' not in actual:raise ValueError('missing TLT execution fingerprint; retune')
    wrong=[name for name,value in key.items() if actual.get(name)!=value]
    if wrong:raise ValueError('incompatible TLT-native profile: '+', '.join(wrong))
    if payload.get('benchmark_metadata',{}).get('active_id_pattern')!='seeded_sorted_randperm':
        raise ValueError('legacy contiguous-active-ID calibration; retune with scattered active IDs')
    if payload.get('benchmark_metadata',{}).get('cuda_graph') is not True:
        raise ValueError('TLT-native profile must measure CUDA graph proposals')
    return ProposalProfile(payload)


def discover(v,rank,dtype,topk,device=None):
    key=execution_key(fingerprint(device),v,rank,dtype,topk)
    explicit=os.environ.get('OPD_PROPOSAL_PROFILE','')
    directory=Path(os.environ.get('OPD_PROPOSAL_PROFILE_DIR',str(ROOT/'outputs/benchmarks/opd_proposals')))
    paths=[Path(explicit)] if explicit else sorted(directory.glob('*.json'))
    for path in paths:
        try:
            payload=json.loads(path.read_text());selector=validate_native_profile(payload,key)
            return selector,str(path)
        except (ValueError,KeyError,TypeError,OSError) as exc:
            if explicit:raise ValueError(f'incompatible explicit TLT-native OPD profile: {path}: {exc}') from exc
    if os.environ.get('OPD_REQUIRE_CALIBRATED_PROFILE','0')=='1':
        raise ValueError('no compatible TLT-native proposal profile; run bash scripts/tune_tlt_opd_proposals.sh on this GPU before benchmarking')
    return None,''
