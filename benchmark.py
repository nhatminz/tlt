"""Official TLT SGLang Engine standalone benchmark; same config, plugin toggle."""
import argparse
import json
import os
from pathlib import Path
import time
from tlt_reflex.runtime import configure,require_runtime


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['tlt','tlt_reflex'],default=os.environ.get('METHOD','tlt'))
    for key in ('model','draft','dataset','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--responses',type=int,default=8)
    p.add_argument('--prompts',type=int,default=16)
    p.add_argument('--max-new-tokens',type=int,default=2048)
    p.add_argument('--max-prompt-length',type=int,default=2048)
    p.add_argument('--temperature',type=float,default=1.)
    p.add_argument('--top-p',type=float,default=.95)
    p.add_argument('--top-k',type=int,default=-1)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--tp',type=int,default=1)
    p.add_argument('--steps',type=int,default=8)
    p.add_argument('--tree-tokens',type=int,default=48)
    p.add_argument('--draft-topk',type=int,default=4)
    p.add_argument('--sd-threshold',type=int,default=32)
    p.add_argument('--mab',default='BEG')
    p.add_argument('--mab-configs',default='8_4_32,8_4_16,8_4_8')
    p.add_argument('--mab-buckets',default='1,2,5,21')
    p.add_argument('--attention-backend',default='triton')
    p.add_argument('--memory-fraction',type=float,default=.6)
    p.add_argument('--disable-cuda-graph',action='store_true')
    p.add_argument('--profile',action='store_true',help='separate diagnostic component run; use eager for inner timings')
    p.add_argument('--pristine-upstream',action='store_true',help='load unmodified official SGLang for a separate TLT-only equivalence check')
    p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--validate-config',action='store_true')
    a=p.parse_args(argv)
    if min(a.batch_size,a.responses,a.prompts,a.max_new_tokens,a.tp,a.steps,a.tree_tokens,a.draft_topk)<1:
        p.error('counts must be positive')
    if a.profile and not a.disable_cuda_graph:
        p.error('inner component profiling requires explicit --disable-cuda-graph for BOTH methods; throughput run is separate')
    if a.pristine_upstream and (a.method!='tlt' or a.profile):
        p.error('pristine upstream comparison requires unprofiled METHOD=tlt')
    return a


def metric_delta(after,before):
    # Single-GPU first; get_server_info exposes one TP-leader state per engine.
    def extract(info):
        states=info.get('internal_states',[])
        return states[0].get('tlt_reflex_metrics',{}) if states else {}
    a,b=extract(after),extract(before)
    return {kind:{key:value-b.get(kind,{}).get(key,0.) for key,value in a.get(kind,{}).items()}
            for kind in ('times_ms','counters')}


def engine_request_capacity(a):
    capacity=a.batch_size*a.responses
    if a.mab_configs and a.mab in ('BEG','PREDEFINED'):
        # Upstream eagerly captures ALL bucket strategies. A batch1 engine
        # limited to one slot would have an empty capture set for bucket21+.
        # Reserve capacity, not extra requests; both methods submit identical
        # B*n responses. No change to BEG thresholds/strategy selection.
        minimum=max(map(int,a.mab_buckets.split(',')))
        capacity=max(capacity,1<<(minimum-1).bit_length())
    return capacity


def main(argv=None):
    a=parse_args(argv)
    if a.validate_config:
        print(json.dumps(vars(a),indent=2));return
    if Path(a.output).exists():raise FileExistsError('use a NEW benchmark output')
    configure(a.method,pristine=a.pristine_upstream)
    os.environ['TLT_TRACE']='1'
    os.environ['REFLEX_PROFILE']='1' if a.profile else '0'
    sg=require_runtime()
    import torch
    from transformers import AutoTokenizer
    from tlt_reflex.data import load_rows,prompt_messages,resolve_data_source
    rows=load_rows(a.dataset)[:a.prompts]
    if not rows:raise ValueError('empty prompt pool')
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    messages=[prompt_messages(r['question'],resolve_data_source(r.get('data_source'),path=a.dataset)) for r in rows]
    prompts=tokenizer.apply_chat_template(messages,
                                         tokenize=False,add_generation_prompt=True)
    prompts=[tokenizer.encode(x)[:a.max_prompt_length] for x in prompts]
    engine_args=dict(model_path=a.model,speculative_algorithm='EAGLE3',speculative_draft_model_path=a.draft,
        dtype='bfloat16',tp_size=a.tp,random_seed=a.seed,disable_overlap_schedule=True,
        speculative_num_steps=a.steps,speculative_eagle_topk=a.draft_topk,
        speculative_num_draft_tokens=a.tree_tokens,
        apdative_speculative_batch_size_threshold=a.sd_threshold,
        speculative_eagle_mab_algorithm=a.mab,speculative_eagle_mab_configs=a.mab_configs.split(',') if a.mab_configs else [],
        speculative_mab_bs_threshold=list(map(int,a.mab_buckets.split(','))),
        max_running_requests=engine_request_capacity(a),cuda_graph_max_bs=engine_request_capacity(a),
        context_length=a.max_prompt_length+a.max_new_tokens+a.steps+1,
        attention_backend=a.attention_backend,mem_fraction_static=a.memory_fraction,
        disable_cuda_graph=a.disable_cuda_graph)
    engine=sg.Engine(**engine_args)
    sampling=dict(n=a.responses,temperature=a.temperature,top_p=a.top_p,top_k=a.top_k,
                  max_new_tokens=a.max_new_tokens)
    try:
        for _ in range(a.warmup):engine.generate(input_ids=prompts[:a.batch_size],sampling_params=sampling)
        baseline=engine.get_server_info()
        collected=[]; elapsed=0.
        for start in range(0,len(prompts),a.batch_size):
            begin=time.perf_counter()
            outputs=engine.generate(input_ids=prompts[start:start+a.batch_size],sampling_params=sampling)
            elapsed+=time.perf_counter()-begin  # synchronous generate includes completion
            collected.extend(outputs)
        final=engine.get_server_info()
        final_metrics=(final.get('internal_states') or [{}])[0].get('tlt_reflex_metrics',{})
        delta=metric_delta(final,baseline)
        counters=delta['counters']; times=delta['times_ms']
        generated=sum(o['meta_info']['completion_tokens'] for o in collected)
        rounds=sum(o['meta_info'].get('spec_verify_ct',0) for o in collected)
        seq_rounds=counters.get('sequence_verification_rounds',0)
        accepted=counters.get('accepted_draft_tokens')
        proposed=counters.get('proposed_draft_tokens')
        report=dict(method=a.method,config=vars(a),engine_config=engine_args,
            generated_tokens=generated,generated_responses=len(collected),generation_wall_s=elapsed,
            tokens_per_s=generated/max(elapsed,1e-9),verification_rounds=rounds,
            # Official FastRL bench definition includes prefill/normal-decode tokens.
            upstream_aal=generated/rounds if rounds else None,
            accepted_draft_tokens=accepted,proposed_draft_tokens=proposed,
            verified_aal=(accepted+seq_rounds)/seq_rounds if seq_rounds else None,
            draft_acceptance_rate=accepted/proposed if proposed else None,
            proposal_time_ms=times.get('proposal_ms') if a.profile else None,
            draft_extend_time_ms=times.get('draft_extend_ms',0.) if a.profile else None,
            proposal_latency_total_ms=sum(times.get(k,0.) for k in ('proposal_ms','draft_extend_ms')) if a.profile else None,
            proposal_correction_time_ms=times.get('reflex_correction_ms',0.) if a.profile else None,
            verification_time_ms=times.get('verification_ms') if a.profile else None,
            reflex_update_time_ms=times.get('reflex_update_ms',0.) if a.profile else None,
            reflex_overhead_ms=sum(times.get(k,0.) for k in ('reflex_feature_ms','reflex_correction_ms','reflex_cache_ms','reflex_update_ms')) if a.profile else None,
            phase_times_ms=times,server_info=final,
            reflex_state_memory_mb=final_metrics.get('reflex_state_memory_mb'),
            reflex_buffer_memory_mb=final_metrics.get('reflex_buffer_memory_mb'),
            note='Throughput pass is unprofiled. Eager component pass is separate; MAB timing-driven strategy/trajectories may differ. No added target forward/sampling. Overlapping intervals cannot infer net wall overhead.')
        path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+'\n')
        path.with_suffix('.responses.jsonl').write_text(''.join(json.dumps(o)+'\n' for o in collected))
        print(json.dumps({k:v for k,v in report.items() if k not in ('server_info','engine_config')},indent=2))
    finally:
        engine.shutdown()


if __name__=='__main__':main()
