#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tlt_reflex.checkpoint import export


def main():
    p=argparse.ArgumentParser(description='Export existing SpecNaacl pretrained EAGLE3; no training or random fallback')
    for arg in ('checkpoint','config','mapping','target','output'): p.add_argument('--'+arg,required=True)
    a=p.parse_args()
    print(export(a.checkpoint,a.config,a.mapping,a.target,a.output))


if __name__=='__main__': main()
