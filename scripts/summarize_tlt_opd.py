"""Aggregate frozen pairs. Recommend only measured AAL AND throughput wins."""
import csv,json
from pathlib import Path
import sys


def summarize(directory):
    root=Path(directory);reports=[json.loads(p.read_text()) for p in sorted(root.glob('*/report.json'))]
    baselines={(r['config']['batch_size'],r['config']['seed']):r for r in reports if r['method']=='tlt'}
    rows=[];eligible=[]
    for r in reports:
        cfg=r['config'];base=baselines[(cfg['batch_size'],cfg['seed'])]
        aal=r['verified_aal'];b_aal=base['verified_aal']
        delta=aal-b_aal if aal is not None and b_aal is not None else None
        ratio=r['tokens_per_s']/base['tokens_per_s']
        good=r['method']=='tlt_opd_reflex' and delta is not None and delta>0 and ratio>=1
        row={key:value for key,value in r.items() if isinstance(value,(str,int,float,bool)) or value is None}
        row.update(batch_size=cfg['batch_size'],seed=cfg['seed'],delta_aal=delta,throughput_ratio=ratio,recommendable=good)
        rows.append(row)
        if good:eligible.append(r)
    root.mkdir(parents=True,exist_ok=True)
    (root/'report.json').write_text(json.dumps(dict(runs=reports,recommendation_rule='delta verified AAL > 0 AND tokens/s >= paired baseline'),indent=2)+'\n')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (root/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    with (root/'responses.jsonl').open('w') as out:
        for p in sorted(root.glob('*/responses.jsonl')):
            for line in p.read_text().splitlines():out.write(json.dumps(dict(run=p.parent.name,response=json.loads(line)))+'\n')
    chosen=max(eligible,key=lambda r:r['tokens_per_s']) if eligible else None
    env='# Only paired measured AAL/throughput wins are recommended.\n'
    if chosen:env+=f"export OPD_FAST_LR={chosen['opd_fast_lr']}\nexport OPD_UPDATE_STREAM={chosen['opd_update_stream']}\n"
    else:env+='# No configuration satisfies the recommendation rule.\n'
    (root/'fastest_observed.env').write_text(env)
    return rows

if __name__=='__main__':summarize(sys.argv[1])
