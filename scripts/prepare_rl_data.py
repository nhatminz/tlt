#!/usr/bin/env python3
"""Format existing local data for upstream VERL; no download/reward replacement."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tlt_reflex.data import load_rows,prompt_messages


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    output=Path(a.output)
    if output.exists():raise FileExistsError('choose a NEW converted-data path')
    import pandas as pd
    rows=load_rows(a.input)
    prepared=[]
    for i,row in enumerate(rows):
        if row['answer'] is None or not isinstance(row['question'],str) or not row['question'].strip():
            raise ValueError(f'dataset row {i} lacks a real question/ground-truth answer; refusing fabricated reward labels')
        truth=str(row['answer'])
        if '####' in truth:truth=truth.split('####')[-1].strip()
        if '\\boxed{' not in truth:truth='\\boxed{'+truth+'}'
        prepared.append(dict(data_source='math',prompt=prompt_messages(row['question']),ability='math',
            reward_model=dict(style='rule',ground_truth=truth),extra_info=dict(index=i,source_path=a.input)))
    output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(prepared).to_parquet(output,index=False)
    print(output)


if __name__=='__main__':main()
