"""Canonical benchmark definition, shared by preflight and real generation."""
import json,os
from pathlib import Path
from tlt_reflex.experiments import projector_record,experiment_label


def request_capacity(a):
    capacity=a.batch_size*a.responses
    if a.mab_configs and a.mab in ('BEG','PREDEFINED'):
        minimum=max(map(int,a.mab_buckets.split(',')))
        capacity=max(capacity,1<<(minimum-1).bit_length())
    return capacity


def engine_config(a):
    return dict(model_path=a.model,speculative_algorithm='EAGLE3',speculative_draft_model_path=a.draft,
        dtype='bfloat16',tp_size=a.tp,dp_size=a.dp,random_seed=a.seed,disable_overlap_schedule=True,
        speculative_num_steps=a.steps,speculative_eagle_topk=a.draft_topk,speculative_num_draft_tokens=a.tree_tokens,
        apdative_speculative_batch_size_threshold=a.sd_threshold,speculative_eagle_mab_algorithm=a.mab,
        speculative_eagle_mab_configs=a.mab_configs.split(',') if a.mab_configs else [],
        speculative_mab_bs_threshold=list(map(int,a.mab_buckets.split(','))),
        max_running_requests=request_capacity(a),cuda_graph_max_bs=request_capacity(a),
        context_length=a.max_prompt_length+a.max_new_tokens+a.steps+1,
        attention_backend=a.attention_backend,mem_fraction_static=a.memory_fraction,disable_cuda_graph=a.disable_cuda_graph)


def sampling_config(a):
    return dict(n=a.responses,temperature=a.temperature,top_p=a.top_p,top_k=a.top_k,max_new_tokens=a.max_new_tokens)


def canonical_config(a):
    record=projector_record(a.draft)
    return dict(schema_version=1,
        experiment_scope='TLT adaptive speculative rollout + fixed EAGLE3',spot_trainer_enabled=False,
        engine=engine_config(a),sampling=sampling_config(a),
        workload=dict(dataset=str(Path(a.dataset).resolve()),requested_prompts=a.prompts,prompt_order='first N dataset rows in file order',
            prompt_template='tlt_reflex.data.prompt_messages + target tokenizer chat template',
            max_prompt_length=a.max_prompt_length,prompt_batch_size=a.batch_size,responses_per_prompt=a.responses,
            requested_responses=a.prompts*a.responses,warmup_count=a.warmup,sampling_seed=a.seed),
        execution=dict(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES','0'),profile=a.profile,scope='native_smoke' if a.smoke else 'official_benchmark',
            pristine_upstream=a.pristine_upstream,tp=a.tp,dp=a.dp),
        checkpoints=dict(target=str(Path(a.model).resolve()),eagle3_base=str(Path(a.draft).resolve()),
            draft_source=record['source_checkpoint'] or str(Path(os.environ.get('DRAFT_CHECKPOINT',a.draft)).resolve())),
        opd=dict(enabled=a.method=='tlt_opd_reflex',method=a.method,
            experiment=experiment_label(a.method,record['provenance']),projector=record,
            rank=int(os.environ.get('OPD_RANK','8')),topk=int(os.environ.get('OPD_TOPK','16')),
            fast_lr=float(os.environ.get('OPD_FAST_LR','.01')),update_stream=int(os.environ.get('OPD_UPDATE_STREAM','1')),
            visited_weight=float(os.environ.get('OPD_VISITED_WEIGHT','1')),frontier_weight=float(os.environ.get('OPD_FRONTIER_WEIGHT','1')),
            proposal_mode=os.environ.get('OPD_PROPOSAL_MODE','auto'),profile=os.environ.get('OPD_PROPOSAL_PROFILE',''),
            profile_dir=os.environ.get('OPD_PROPOSAL_PROFILE_DIR',''),
            require_calibrated_profile=os.environ.get('OPD_REQUIRE_CALIBRATED_PROFILE','1')=='1',
            allow_untrained_projector=os.environ.get('OPD_ALLOW_UNTRAINED_PROJECTOR','0')=='1',train_projector=False))


def canonical_diff(tlt,opd):
    def flatten(value,prefix=''):
        if isinstance(value,dict):
            result={}
            for key,item in value.items():result.update(flatten(item,prefix+'.'+key if prefix else key))
            return result
        return {prefix:value}
    left,right=flatten(tlt),flatten(opd);diffs=[]
    for key in sorted(left.keys()|right.keys()):
        if key not in left or key not in right or left[key]!=right[key]:
            diffs.append(dict(path=key,tlt=left.get(key),tlt_opd=right.get(key)))
    critical=[d for d in diffs if not d['path'].startswith('opd.')]
    if critical:raise ValueError('benchmark-critical canonical config differs: '+', '.join(d['path'] for d in critical))
    if tlt.get('opd',{}).get('enabled') is not False or opd.get('opd',{}).get('enabled') is not True:
        raise ValueError('pair must contain OPD disabled then OPD enabled')
    if tlt.get('execution',{}).get('scope','official_benchmark')!='official_benchmark':
        raise ValueError('native smoke cannot be an official throughput comparison')
    if tlt.get('execution',{}).get('profile') or opd.get('execution',{}).get('profile'):
        raise ValueError('official throughput pair must have profiling OFF')
    if tlt.get('spot_trainer_enabled') is not False or opd.get('spot_trainer_enabled') is not False:
        raise ValueError('pair must explicitly keep EAGLE3 Spot Trainer disabled')
    return dict(benchmark_critical_fields_identical=True,differences=diffs)
