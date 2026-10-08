#!/usr/bin/env python3
"""Reject old/unknown reward identifiers; never overwrite an existing dataset."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tlt_reflex.data import valid_math_source


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args()
    import pandas as pd
    records=pd.read_parquet(a.input,columns=['data_source','reward_model']).to_dict('records')
    if not records:raise ValueError('empty RL dataset')
    for i,r in enumerate(records):
        if not valid_math_source(r['data_source']):
            raise ValueError(f'row {i}: unsupported data_source {r["data_source"]!r}; regenerate with prepare_rl_data.py into a NEW path (old data_source=math is invalid)')
        truth=r['reward_model'].get('ground_truth')
        if not isinstance(truth,str) or not truth.strip():raise ValueError(f'row {i}: missing string ground_truth')
        if r['data_source']=='openai/gsm8k' and '\\boxed{' in truth:
            raise ValueError(f'row {i}: GSM8K requires unboxed numeric ground truth; regenerate old converted data')
    print(f'RL reward identifiers/ground truths verified: {len(records)} rows')


if __name__=='__main__':main()
