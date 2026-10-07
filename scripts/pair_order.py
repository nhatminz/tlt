#!/usr/bin/env python3
"""Counterbalance physical run order using the effective canonical sampling seed."""
import argparse,json
from pathlib import Path


def ordered_entries(seed):
    entries=[('tlt','tlt'),('tlt_opd_reflex','tlt_opd')]
    return entries if int(seed)%2==0 else entries[::-1]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('canonical');p.add_argument('--output',required=True);p.add_argument('--entries-output',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.canonical).read_text());seed=cfg['workload']['sampling_seed'];entries=ordered_entries(seed)
    Path(a.output).write_text(json.dumps(dict(seed=seed,policy='even seed TLT-first; odd seed OPD-first',
        methods=[e[0] for e in entries],status='planned'),indent=2)+'\n')
    Path(a.entries_output).write_text(''.join(method+':'+folder+'\n' for method,folder in entries))
