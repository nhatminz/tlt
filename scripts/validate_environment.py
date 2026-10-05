#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tlt_reflex.runtime import configure,require_runtime
import os

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rl',action='store_true');a=p.parse_args()
    configure(os.environ.get('METHOD','tlt'))
    try:
        sg=require_runtime(rl=a.rl)
        print('Pinned TLT runtime imports passed:',sg.__file__)
    except Exception as error:
        raise SystemExit(f'{type(error).__name__}: {error}') from None
