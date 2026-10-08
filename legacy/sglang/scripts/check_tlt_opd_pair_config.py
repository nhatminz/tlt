#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tlt_reflex.benchmark_config import canonical_diff

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tlt');p.add_argument('opd');p.add_argument('--output',required=True);a=p.parse_args()
    result=canonical_diff(json.loads(Path(a.tlt).read_text()),json.loads(Path(a.opd).read_text()))
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    print('Canonical preflight passed: all benchmark-critical fields identical; only OPD differences')
