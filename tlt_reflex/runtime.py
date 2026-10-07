"""Pin vendored imports before any SGLang/VERL module is loaded."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
FAStrl=ROOT/'upstream/fastrl'
SGLANG=FAStrl/'third-party/sglang/python'
PRISTINE=ROOT/'upstream/pristine_sglang_python'


def configure(mode, *, pristine=False):
    global SGLANG
    if mode not in ('tlt','tlt_opd_reflex'):
        raise ValueError('METHOD must be tlt or tlt_opd_reflex')
    if pristine and mode!='tlt':
        raise ValueError('pristine upstream supports METHOD=tlt only')
    SGLANG=PRISTINE if pristine else FAStrl/'third-party/sglang/python'
    os.environ['TLT_REFLEX_METHOD']=mode
    paths=[str(ROOT),str(SGLANG),str(FAStrl)]
    # Ray/torchrun/SGLang child processes inherit exact same module selection.
    os.environ['PYTHONPATH']=os.pathsep.join(paths+[os.environ.get('PYTHONPATH','')])
    for value in reversed(paths):
        if value in sys.path: sys.path.remove(value)
        sys.path.insert(0,value)
    for name in ('sglang','verl'):
        if name in sys.modules:
            module=sys.modules[name]
            expected=SGLANG if name=='sglang' else FAStrl
            if not Path(module.__file__).resolve().is_relative_to(expected):
                raise RuntimeError(f'{name} already imported from wrong environment: {module.__file__}')


def require_runtime(*, rl=False):
    import importlib
    import importlib.util
    import importlib.metadata as metadata
    if sys.version_info[:2]!=(3,12):
        raise RuntimeError('pinned upstream TLT environment requires Python3.12, use a SEPARATE venv')
    pins={'torch':'2.8.0','transformers':'4.57.1','sgl-kernel':'0.3.15','flashinfer-python':'0.4.0','apache-tvm-ffi':'0.1.0'}
    if rl:pins['flash-attn']='2.8.3'
    failures=[]
    for package,expected in pins.items():
        try: actual=metadata.version(package).split('+',1)[0]
        except metadata.PackageNotFoundError: actual='not installed'
        if actual!=expected: failures.append(f'{package}: expected {expected}, found {actual}')
    if failures:
        raise RuntimeError('TLT stack mismatch; do not replace SpecNaacl environment:\n- '+'\n- '.join(failures))
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('real TLT runtime requires CUDA')
    if torch.version.cuda!='12.8': raise RuntimeError('pinned Torch2.8 CUDA12.8 build required, including B200 support')
    if torch.cuda.get_device_capability()<(8,0): raise RuntimeError('unsupported CUDA GPU')
    import shutil
    if shutil.which('nvcc') is None:
        raise RuntimeError('FlashInfer0.4 JIT needs CUDA toolkit/nvcc (12.8 for B200); a Torch CUDA runtime wheel alone is insufficient')
    audit_spec=importlib.util.spec_from_file_location('tlt_source_audit',ROOT/'scripts/audit_upstream.py')
    audit_module=importlib.util.module_from_spec(audit_spec);audit_spec.loader.exec_module(audit_module)
    audit_module.audit(require_git=True)
    sg=importlib.import_module('sglang')
    if not Path(sg.__file__).resolve().is_relative_to(SGLANG):
        raise RuntimeError('wrong SGLang import; must use the official vendored TLT fork')
    # Actual mandatory native extension/API imports; not merely version strings.
    from sgl_kernel import tree_speculative_sampling_target_only
    from sglang.srt.speculative.eagle_worker import EAGLEWorker
    from flashinfer.sampling import top_k_renorm_prob,top_p_renorm_prob
    if rl:
        # Official FSDP trainer uses FA2 unconditionally, unlike standalone
        # SGLang (which can use Triton attention). Fail before launching Ray.
        importlib.import_module('flash_attn.bert_padding')
        importlib.import_module('verl.trainer.main_fastrl')
    return sg
