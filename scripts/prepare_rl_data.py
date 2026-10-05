#!/usr/bin/env python3
"""Format existing local data for upstream VERL; no download/reward replacement."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tlt_reflex.data import load_rows,prepare_records


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--dataset',help='actual dataset family (dapo/gsm8k/math/simplelr); does not override valid original identifier')
    p.add_argument('--data-source',help='explicit supported upstream math reward identifier')
    a=p.parse_args()
    output=Path(a.output)
    if output.exists():raise FileExistsError('choose a NEW converted-data path')
    import pandas as pd
    rows=load_rows(a.input)
    prepared=prepare_records(rows,path=a.input,dataset=a.dataset,data_source=a.data_source)
    output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(prepared).to_parquet(output,index=False)
    print(output)


if __name__=='__main__':main()
