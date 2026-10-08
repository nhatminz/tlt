#!/usr/bin/env python3
"""Offline TLT graph proposal tuner; real CUDA timings and execution-keyed JSON."""
import argparse
import json
from pathlib import Path
import statistics
import hashlib
import os
import tempfile
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tlt_reflex.ported.profiles import inspect_draft,active_trials,execution_key,profile_filename,ProposalProfile
from tlt_reflex.profiles import fingerprint,validate_native_profile,PROFILE_KIND


DEFAULT_SHAPES='1x1,2x1,4x1,8x1,16x1,32x1,8x4,32x4'


def effective_shapes(shapes):
    """Flatten and deduplicate effective workloads, independent of factorization."""
    if isinstance(shapes,str):shapes=[tuple(map(int,shape.split('x'))) for shape in shapes.split(',')]
    if not shapes or any(len(shape)!=2 or min(shape)<1 for shape in shapes):raise ValueError('tuner shapes must be positive batch x contexts')
    return [(n,1) for n in sorted({b*c for b,c in shapes})]


def write_validated_profile(path,payload,key):
    # Validate the exact same consumer class BEFORE touching the output file.
    ProposalProfile(payload);validate_native_profile(payload,key)
    serialized=json.dumps(payload,indent=2,allow_nan=False)+'\n'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix='.'+path.name+'.',delete=False) as f:
            temporary=Path(f.name);f.write(serialized)
        os.replace(temporary,path)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()


def benchmark_configuration(key,shapes,slots,iterations,seed=42):
    import torch
    from tlt_reflex.state import OPDState
    from tlt_reflex import kernels
    dtype=getattr(torch,key['dtype'].split('.')[-1]);v=key['vocab'];r=key['rank'];k=key['topk'];records=[]
    if iterations<1 or any(count<0 or count>v for count in slots):raise ValueError('invalid tuner iterations/active rows')
    for b,c in effective_shapes(shapes):
        n=b*c
        generator=torch.Generator(device="cuda").manual_seed(seed+n)
        head=torch.nn.Linear(r,v,bias=False,device='cuda',dtype=dtype)
        s=OPDState(b,head,torch.arange(v,device='cuda'),rank=r,topk=k,projector=torch.eye(r,device='cuda'),
            max_contexts=c,max_topk=c,max_nodes=1,max_path=1,proposal_mode='sparse',update_stream=False)
        s.score_workspace=torch.empty(n*v,device='cuda');s.has_gemm=True;raw=torch.randn(b,c,v,device='cuda',dtype=dtype,generator=generator)
        u=torch.randn(b,c,r,device='cuda',generator=generator);trials=[]
        for count in slots:
            s.B_fast.zero_();s.bitmap.zero_();s.active_count.fill_(count)
            ids=torch.randperm(v,device='cuda',generator=generator)[:count].sort().values
            s.active_ids[:count].copy_(ids)
            if count:s.B_fast[ids]=torch.randn(count,r,device='cuda',generator=generator)*.05
            # Lifecycle-only fixture bitmap setup, never a generation operation.
            cpu_ids=ids.cpu()
            bits=torch.zeros((v+31)//32,dtype=torch.int64)
            bits.scatter_add_(0,cpu_ids//32,torch.ones_like(cpu_ids,dtype=torch.int64)<<(cpu_ids%32))
            bits=bits.to(torch.int32)
            s.bitmap.copy_(bits)
            values={};times={}
            for mode in ('sparse','fused','gemm'):
                s.proposal_mode=mode
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):kernels.propose(s,raw,u)
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):out=kernels.propose(s,raw,u)
                graph.replay();values[mode]=tuple(x.clone() for x in out)
                a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);samples=[]
                for _ in range(iterations):
                    a.record();graph.replay();z.record();z.synchronize();samples.append(a.elapsed_time(z))
                times[mode]=statistics.median(samples)
            for mode in ('fused','gemm'):
                if not all(torch.equal(a,z) for a,z in zip(values['sparse'],values[mode])):
                    raise AssertionError(f'TLT proposal parity failed n={n} S={count} {mode}; no profile written')
            trials.append(dict(slots=count,**times,bitwise_parity=True,
                active_ids_sha256=hashlib.sha256(cpu_ids.numpy().tobytes()).hexdigest(),active_ids_preview=cpu_ids[:16].tolist()))
        records.append(dict(contexts=n,shape=f'{b}x{c}',trials=trials))
    return dict(schema_version=3,profile_kind=PROFILE_KIND,execution_key=key,records=records,
        benchmark_metadata=dict(iterations=iterations,statistic='median',unit='ms',cuda_graph=True,seed=seed,active_id_pattern='seeded_sorted_randperm'),
        note='TLT graph proposal-only costs. Auto interpolates calibrated sparse/fused/GEMM costs on device.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',default='qwen25_3b')
    p.add_argument('--pretrain-root',default=str(ROOT.parent/'SpecNaacl/outputs/pretrain'))
    p.add_argument('--profile-dir',default=str(ROOT/'outputs/benchmarks/opd_proposals'))
    p.add_argument('--dtype',default='bf16');p.add_argument('--rank',type=int);p.add_argument('--topk',type=int,default=16)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--iterations',type=int,default=30);p.add_argument('--shapes',default=DEFAULT_SHAPES)
    p.add_argument('--slots',default='');p.add_argument('--output');p.add_argument('--force',action='store_true')
    p.add_argument('--draft-config');p.add_argument('--draft-checkpoint');p.add_argument('--vocab-mapping')
    a=p.parse_args(argv);shapes=effective_shapes(a.shapes);groups={}
    for name in a.models.split(','):
        base=Path(a.pretrain_root)/name
        info=inspect_draft(a.draft_config or base/'latest_draft_config.json',a.draft_checkpoint or base/'latest_checkpoint',
            a.vocab_mapping or base/'latest_vocab_mapping.pt',rank=a.rank,dtype=a.dtype,topk=a.topk)
        key=execution_key(fingerprint(),info['vocab'],info['rank'],info['dtype'],info['topk'])
        groups.setdefault(profile_filename(key),key)
    if a.output and len(groups)!=1:p.error('--output needs exactly one execution key')
    for name,key in groups.items():
        path=Path(a.output) if a.output else Path(a.profile_dir)/name
        if path.exists() and not a.force:
            validate_native_profile(json.loads(path.read_text()),key)
            print('Reuse',path);continue
        slots=list(map(int,a.slots.split(','))) if a.slots else active_trials(key['vocab'],key['topk'])
        report=benchmark_configuration(key,shapes,slots,a.iterations,seed=a.seed)
        write_validated_profile(path,report,key);print(path)

if __name__=='__main__':main()
