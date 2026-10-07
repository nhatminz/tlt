"""Validate exact paired experiment identity before reporting deltas/winners."""
import csv,json
from pathlib import Path
import sys

REQUIRED_CONFIG=('model','draft','dataset','batch_size','responses','prompts','max_new_tokens','max_prompt_length',
    'temperature','top_p','top_k','seed','tp','steps','tree_tokens','draft_topk','sd_threshold','mab','mab_configs',
    'mab_buckets','attention_backend','memory_fraction','disable_cuda_graph','warmup','profile')


def validate_pair(base,opd):
    for name,report in [('baseline',base),('opd',opd)]:
        cfg=report['config']
        if any(key not in cfg for key in REQUIRED_CONFIG):raise ValueError(f'{name} missing fairness configuration fields')
        if cfg['profile']:raise ValueError('profiled runs cannot participate in throughput comparison')
        if report.get('spot_trainer_enabled') is not False:raise ValueError('fixed EAGLE3 comparison must explicitly disable Spot Trainer')
        if not report.get('valid_for_official_comparison',False):raise ValueError('invalid run cannot participate in official comparison')
    ignored={'method','output','validate_config'}
    if {k:v for k,v in base['config'].items() if k not in ignored}!={k:v for k,v in opd['config'].items() if k not in ignored}:
        raise ValueError('paired generation configurations differ')
    if base['engine_config']!=opd['engine_config']:raise ValueError('paired engine configurations differ')
    for key in ('artifact_identity','prompt_token_sha256','measured_prompts','generated_responses'):
        if key not in base or key not in opd or base[key]!=opd[key]:raise ValueError('paired checkpoint/prompts/measured samples differ: '+key)
    if opd.get('opd_orphan_nodes')!=0 or opd.get('opd_invalid_contexts')!=0:raise ValueError('OPD tree/context validation counters must be zero')
    if not (opd.get('real_eagle3_parity') or {}).get('passed'):raise ValueError('real EAGLE3 representation parity was not certified')


def summarize(directory):
    root=Path(directory)
    reports=[json.loads(p.read_text()) for p in sorted(root.glob('*/report.json')) if 'components' not in p.parent.name]
    baselines={}
    for r in reports:
        if r['method']=='tlt':
            key=(r['config']['batch_size'],r['config']['seed'])
            if key in baselines:raise ValueError('duplicate paired baseline')
            baselines[key]=r
    rows=[];eligible=[];pairs=[]
    for r in reports:
        cfg=r['config'];key=(cfg['batch_size'],cfg['seed'])
        if key not in baselines:raise ValueError('missing same-batch/seed TLT baseline')
        base=baselines[key]
        if r['method']=='tlt_opd_reflex':validate_pair(base,r)
        aal=r['verified_aal'];b_aal=base['verified_aal']
        delta=aal-b_aal if aal is not None and b_aal is not None else None
        ratio=r['tokens_per_s']/base['tokens_per_s']
        good=r['method']=='tlt_opd_reflex' and delta is not None and delta>0 and ratio>1
        row={key:value for key,value in r.items() if isinstance(value,(str,int,float,bool)) or value is None}
        row.update(batch_size=cfg['batch_size'],seed=cfg['seed'],delta_aal=delta,throughput_ratio=ratio,recommendable=good)
        rows.append(row)
        if r['method']=='tlt_opd_reflex':
            pairs.append(dict(batch_size=cfg['batch_size'],seed=cfg['seed'],tlt=base,opd=r,
                delta=dict(verified_aal=delta,tokens_per_s=r['tokens_per_s']-base['tokens_per_s'],
                    generation_wall_s=r['generation_wall_s']-base['generation_wall_s']),throughput_ratio=ratio))
        if good:eligible.append(r)
    root.mkdir(parents=True,exist_ok=True)
    (root/'report.json').write_text(json.dumps(dict(runs=reports,pairs=pairs,
        experiment='TLT adaptive speculative rollout + fixed EAGLE3 vs same + OPD',
        recommendation_rule='delta verified AAL > 0 AND tokens/s > paired baseline; no statistical significance claimed'),indent=2)+'\n')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (root/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    with (root/'responses.jsonl').open('w') as out:
        for p in sorted(root.glob('*/responses.jsonl')):
            if 'components' in p.parent.name:continue
            for line in p.read_text().splitlines():out.write(json.dumps(dict(run=p.parent.name,response=json.loads(line)))+'\n')
    chosen=max(eligible,key=lambda r:r['tokens_per_s']) if eligible else None
    env='# Observed paired AAL/throughput wins only; validate across seeds before claiming significance.\n'
    if chosen:env+=f"export OPD_FAST_LR={chosen['opd_fast_lr']}\nexport OPD_UPDATE_STREAM={chosen['opd_update_stream']}\n"
    else:env+='# No configuration satisfies the strict recommendation rule.\n'
    (root/'fastest_observed.env').write_text(env)
    return rows

if __name__=='__main__':summarize(sys.argv[1])
