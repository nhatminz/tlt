#!/usr/bin/env python3
"""Offline TLT graph proposal tuner; real CUDA timings and execution-keyed JSON."""
import argparse
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tlt_reflex.ported.profiles import inspect_draft,active_trials,context_shapes,execution_key,profile_filename
from tlt_reflex.profiles import fingerprint


def benchmark_configuration(key,shapes,slots,iterations):
    import torch
    from tlt_reflex.state import OPDState
    from tlt_reflex import kernels
    dtype=getattr(torch,key['dtype'].split('.')[-1]);v=key['vocab'];r=key['rank'];k=key['topk'];records=[]
    for b,c in shapes:
        n=b*c
        head=torch.nn.Linear(r,v,bias=False,device='cuda',dtype=dtype)
        s=OPDState(b,head,torch.arange(v,device='cuda'),rank=r,topk=k,projector=torch.eye(r,device='cuda'),
            max_contexts=c,max_topk=c,max_nodes=1,max_path=1,proposal_mode='sparse',update_stream=False)
        s.score_workspace=torch.empty(n*v,device='cuda');s.has_gemm=True;raw=torch.randn(b,c,v,device='cuda',dtype=dtype)
        u=torch.randn(b,c,r,device='cuda');trials=[]
        for count in slots:
            s.B_fast.zero_();s.bitmap.zero_();s.active_count.fill_(count)
            ids=torch.arange(count,device='cuda');s.active_ids[:count]=ids
            if count:s.B_fast[:count]=torch.randn(count,r,device='cuda')*.05
            # Lifecycle-only fixture bitmap setup, never a generation operation.
            bits=torch.full(((v+31)//32,),-1,dtype=torch.int32)
            words,remainder=divmod(count,32);bits[words:]=0
            if remainder:bits[words]=(1<<remainder)-1
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
            trials.append(dict(slots=count,**times,bitwise_parity=True))
        records.append(dict(contexts=n,shape=f'{b}x{c}',trials=trials))
    return dict(schema_version=2,execution_key=key,records=records,
        benchmark_metadata=dict(iterations=iterations,statistic='median',unit='ms',cuda_graph=True),
        note='TLT graph proposal-only costs. Auto interpolates calibrated sparse/fused/GEMM costs on device.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',default='qwen25_3b')
    p.add_argument('--pretrain-root',default=str(ROOT.parent/'SpecNaacl/outputs/pretrain'))
    p.add_argument('--profile-dir',default=str(ROOT/'outputs/benchmarks/opd_proposals'))
    p.add_argument('--dtype',default='bf16');p.add_argument('--rank',type=int);p.add_argument('--topk',type=int,default=16)
    p.add_argument('--iterations',type=int,default=30);p.add_argument('--shapes',default='1x1,2x1,4x1,8x1,16x1,32x1,8x4,32x4')
    p.add_argument('--slots',default='');p.add_argument('--output');p.add_argument('--force',action='store_true')
    p.add_argument('--draft-config');p.add_argument('--draft-checkpoint');p.add_argument('--vocab-mapping')
    a=p.parse_args(argv);groups={}
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
            if json.loads(path.read_text())['execution_key']!=key:raise ValueError('existing profile key mismatch')
            print('Reuse',path);continue
        shapes=[tuple(map(int,s.split('x'))) for s in a.shapes.split(',')]
        slots=list(map(int,a.slots.split(','))) if a.slots else active_trials(key['vocab'],key['topk'])
        report=benchmark_configuration(key,shapes,slots,a.iterations)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n');print(path)

if __name__=='__main__':main()
