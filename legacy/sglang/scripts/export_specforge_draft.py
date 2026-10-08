#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tlt_reflex.checkpoint import export


def main():
    p=argparse.ArgumentParser(description='Export existing SpecNaacl pretrained EAGLE3; no training or random fallback')
    for arg in ('checkpoint','config','mapping','target','output'): p.add_argument('--'+arg,required=True)
    p.add_argument('--projector-provenance',choices=['trained','head_basis_initialized'])
    p.add_argument('--projector-training-dataset')
    p.add_argument('--projector-training-steps',type=int)
    a=p.parse_args()
    print(export(a.checkpoint,a.config,a.mapping,a.target,a.output,projector_provenance=a.projector_provenance,
        projector_training_dataset=a.projector_training_dataset,projector_training_steps=a.projector_training_steps))


if __name__=='__main__': main()
