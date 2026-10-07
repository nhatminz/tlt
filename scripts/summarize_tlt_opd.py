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
    ignored={'method','output','validate_config','dump_canonical_config'}
    if {k:v for k,v in base['config'].items() if k not in ignored}!={k:v for k,v in opd['config'].items() if k not in ignored}:
        raise ValueError('paired generation configurations differ')
    if 'canonical_config' in base or 'canonical_config' in opd:
        from tlt_reflex.benchmark_config import canonical_diff
        canonical_diff(base['canonical_config'],opd['canonical_config'])
    if base['engine_config']!=opd['engine_config']:raise ValueError('paired engine configurations differ')
    for key in ('artifact_identity','prompt_token_sha256','measured_prompts','generated_responses'):
        if key not in base or key not in opd or base[key]!=opd[key]:raise ValueError('paired checkpoint/prompts/measured samples differ: '+key)
    if opd.get('opd_orphan_nodes')!=0 or opd.get('opd_invalid_contexts')!=0:raise ValueError('OPD tree/context validation counters must be zero')
    if not (opd.get('real_eagle3_parity') or {}).get('passed'):raise ValueError('real EAGLE3 representation parity was not certified')


def summarize(directory):
    root=Path(directory)
    entries=[]
    for path in sorted(root.rglob('report.json')):
        if path==root/'report.json' or 'components' in str(path.relative_to(root)):continue
        report=json.loads(path.read_text())
        if 'method' in report:entries.append((path,report))
    reports=[report for _,report in entries]
    if not reports:raise ValueError('no measured benchmark reports')
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
        row.update(projector_experiment=r.get('opd_projector_experiment','unspecified'),batch_size=cfg['batch_size'],seed=cfg['seed'],delta_aal=delta,throughput_ratio=ratio,recommendable=good)
        row.update(delta_tokens_per_s=r['tokens_per_s']-base['tokens_per_s'],
            delta_wall_time_s=r['generation_wall_s']-base['generation_wall_s'],
            delta_peak_allocated_gb=(r['peak_allocated_gb']-base['peak_allocated_gb']) if r.get('peak_allocated_gb') is not None and base.get('peak_allocated_gb') is not None else None,
            delta_peak_reserved_gb=(r['peak_reserved_gb']-base['peak_reserved_gb']) if r.get('peak_reserved_gb') is not None and base.get('peak_reserved_gb') is not None else None)
        rows.append(row)
        if r['method']=='tlt_opd_reflex':
            pairs.append(dict(batch_size=cfg['batch_size'],seed=cfg['seed'],tlt=base,opd=r,
                delta=dict(verified_aal=delta,tokens_per_s=r['tokens_per_s']-base['tokens_per_s'],
                    generation_wall_s=r['generation_wall_s']-base['generation_wall_s'],
                    peak_allocated_gb=(r['peak_allocated_gb']-base['peak_allocated_gb']) if r.get('peak_allocated_gb') is not None and base.get('peak_allocated_gb') is not None else None,
                    peak_reserved_gb=(r['peak_reserved_gb']-base['peak_reserved_gb']) if r.get('peak_reserved_gb') is not None and base.get('peak_reserved_gb') is not None else None),
                projector_experiment=r.get('opd_projector_experiment','unspecified'),opd_overhead_ms=r.get('total_opd_overhead_ms'),throughput_ratio=ratio))
        if good:eligible.append(r)
    root.mkdir(parents=True,exist_ok=True)
    (root/'report.json').write_text(json.dumps(dict(runs=reports,pairs=pairs,
        experiment='TLT adaptive speculative rollout + fixed EAGLE3 vs same + OPD',
        recommendation_rule='delta verified AAL > 0 AND tokens/s > paired baseline; no statistical significance claimed'),indent=2)+'\n')
    comparison=json.loads((root/'report.json').read_text())
    comparison['projector_experiments']=sorted({r.get('opd_projector_experiment','unspecified') for r in reports if r['method']=='tlt_opd_reflex'})
    (root/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    if not (root/'config_diff.json').is_file():
        cases=[]
        for path in sorted(root.rglob('config_diff.json')):
            if path==root/'config_diff.json' or 'components' in str(path.relative_to(root)):continue
            item=json.loads(path.read_text())
            if any(not d['path'].startswith('opd.') for d in item.get('differences',[])):raise ValueError('invalid preflight config_diff')
            cases.append(dict(case=str(path.parent.relative_to(root)),**item))
        (root/'config_diff.json').write_text(json.dumps(dict(cases=cases),indent=2)+'\n')
    # Grid root provides requested mode summaries; actual leaf reports stay per case.
    if any(path.parent.parent!=root for path,_ in entries):
        for method,folder in [('tlt','tlt'),('tlt_opd_reflex','tlt_opd')]:
            path=root/folder/'report.json';path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(dict(scope='aggregate mode summary; individual canonical configs are in each case',
                runs=[r for r in reports if r['method']==method]),indent=2)+'\n')

    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (root/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    with (root/'responses.jsonl').open('w') as out:
        for result_path,_ in entries:
            p=result_path.parent/'responses.jsonl'
            if not p.is_file():raise FileNotFoundError(p)
            for line in p.read_text().splitlines():out.write(json.dumps(dict(run=str(p.parent.relative_to(root)),response=json.loads(line)))+'\n')
    variants=sorted({r.get('opd_projector_experiment','unspecified') for r in eligible})
    env='# Observed paired wins only; projector experiments are kept separate.\n'
    recommendations={}
    for variant in variants:
        chosen=max((r for r in eligible if r.get('opd_projector_experiment','unspecified')==variant),key=lambda r:r['tokens_per_s'])
        recommendations[variant]=dict(opd_fast_lr=chosen['opd_fast_lr'],opd_update_stream=chosen['opd_update_stream'],batch_size=chosen['config']['batch_size'],seed=chosen['config']['seed'])
    if len(variants)==1:
        chosen=recommendations[variants[0]]
        env+=f"# Projector experiment: {variants[0]}\nexport OPD_FAST_LR={chosen['opd_fast_lr']}\nexport OPD_UPDATE_STREAM={chosen['opd_update_stream']}\n"
    elif variants:env+='# Multiple projector experiments: see recommendations_by_projector.json; no mixed winner.\n'
    else:env+='# No configuration satisfies the strict recommendation rule.\n'
    (root/'recommendations_by_projector.json').write_text(json.dumps(recommendations,indent=2)+'\n')
    (root/'fastest_observed.env').write_text(env)
    return rows

if __name__=='__main__':summarize(sys.argv[1])
