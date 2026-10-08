#!/usr/bin/env python3
"""Validate the current full-V TLT proposal profile, before any engine launch."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from helper.opd_profiles import inspect_draft,execution_key,fingerprint,discover_profile,validate_profile

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--target-config',required=True);p.add_argument('--draft-checkpoint',required=True)
    p.add_argument('--profile-dir',default=str(ROOT/'outputs/benchmarks/opd_proposals'));p.add_argument('--profile',default='')
    p.add_argument('--rank',type=int,default=8);p.add_argument('--topk',type=int,default=16);p.add_argument('--dtype',default='bf16')
    a=p.parse_args();d=inspect_draft(a.target_config,a.draft_checkpoint,rank=a.rank,dtype=a.dtype,topk=a.topk)
    key=execution_key(fingerprint(),d['vocab'],d['rank'],d['dtype'],d['topk'])
    path,payload=discover_profile(a.profile_dir,key,a.profile)
    if path is None:p.error('Missing TLT-native profile; run bash scripts/tune_tlt_opd_proposals.sh on this server/GPU')
    selector=validate_profile(payload,key)
    print(json.dumps(dict(validated=True,profile=str(path.resolve()),execution_key=key,contexts=selector.contexts),indent=2))

if __name__=='__main__':main()
