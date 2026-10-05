#!/usr/bin/env python3
"""Compare real engine outputs: pristine upstream vs plugin OFF (not two seeds).

Adaptive MAB is timing driven. If it selects different trees, stochastic bitwise
identity is NOT promised despite exact target distributions. Do not disguise a
failed equality test; report tokens/config and re-run a fixed-strategy, greedy
diagnostic separately from the real adaptive throughput benchmark.
"""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('pristine');p.add_argument('off')
    a=p.parse_args()
    reports=[json.loads(Path(x).read_text()) for x in (a.pristine,a.off)]
    if reports[0]['engine_config']!=reports[1]['engine_config']:
        raise ValueError('engine configs differ; not a valid equivalence check')
    sequences=[]
    for x in (a.pristine,a.off):
        rows=[json.loads(s) for s in Path(x).with_suffix('.responses.jsonl').read_text().splitlines()]
        sequences.append([r.get('output_ids',r.get('text')) for r in rows])
    identical=sequences[0]==sequences[1]
    print(json.dumps(dict(response_outputs_identical=identical,responses=[len(x) for x in sequences]),indent=2))
    if not identical:raise SystemExit('outputs differ; inspect adaptive/timing-driven MAB decisions, not evidence of equivalence')


if __name__=='__main__':main()
