#!/usr/bin/env python3
"""Check actual GPU/draft execution key and reload native cost buckets offline."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tlt_reflex.ported.profiles import inspect_draft,ProposalProfile
from tlt_reflex.profiles import discover

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('draft-config','draft-checkpoint','vocab-mapping'):p.add_argument('--'+name,required=True)
    p.add_argument('--profile-dir',default=str(ROOT/'outputs/benchmarks/opd_proposals'));p.add_argument('--profile')
    p.add_argument('--rank',type=int);p.add_argument('--dtype',default='bf16');p.add_argument('--topk',type=int,default=16)
    a=p.parse_args();info=inspect_draft(a.draft_config,a.draft_checkpoint,a.vocab_mapping,rank=a.rank,dtype=a.dtype,topk=a.topk)
    import torch
    os.environ['OPD_REQUIRE_CALIBRATED_PROFILE']='1';os.environ['OPD_PROPOSAL_PROFILE_DIR']=a.profile_dir
    if a.profile:os.environ['OPD_PROPOSAL_PROFILE']=a.profile
    else:os.environ.pop('OPD_PROPOSAL_PROFILE',None)
    selector,path=discover(info['vocab'],info['rank'],getattr(torch,info['dtype'].split('.')[-1]),info['topk'])
    payload=json.loads(Path(path).read_text());ProposalProfile(payload)
    print(json.dumps(dict(profile=path,execution_key=payload['execution_key'],contexts=selector.contexts,
        validated=True,profile_kind=payload['profile_kind']),indent=2))
