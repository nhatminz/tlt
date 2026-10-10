#!/usr/bin/env python3
"""Counterbalanced frozen or online-draft pair on TLT adaptive FastGRPO rollout."""
import argparse
import ast
from copy import deepcopy
from dataclasses import asdict
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from helper.tlt_scheduler import TLTConfig, TLTScheduler


def time_generation_and_training(generate, train=None, *, synchronize, clock=time.perf_counter):
    """Adjacent wall intervals; generation never includes draft backward/update.

    Synchronization belongs only to benchmark phase boundaries. Timing does not
    insert an optimizer step or flush partial accumulation.
    """
    synchronize()
    start=clock()
    output=generate()
    synchronize()
    generated=clock()
    if train is not None:
        train(output)
        synchronize()
        finished=clock()
    else:
        finished=generated
    return output,dict(generation_wall_s=generated-start,
                       draft_training_wall_s=finished-generated,
                       combined_wall_s=finished-start)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('target-model','draft-checkpoint','dataset-path'):
        p.add_argument('--'+name,default='')
    p.add_argument('--target-adapter',default='')
    p.add_argument('--output',required=True)
    p.add_argument('--method',choices=['pair','tlt','tlt_opd_reflex'],default='pair')
    p.add_argument('--batch-sizes',default='1,2,4,8,16,32');p.add_argument('--seeds',default='42,43')
    p.add_argument('--fast-lrs',default='0.01');p.add_argument('--streams',default='1')
    for name,default in [('responses',8),('iterations',2),('warmup',1),('max-length',512),('max-prompt-length',256),('rank',8),('topk',16),('draft-accumulation-steps',1)]:
        p.add_argument('--'+name,type=int,default=default)
    for name,default in [('verification-capacity',512),('max-draft-token-length',5),('max-draft-k',8),
                         ('max-verification-num',160),('min-draft-token-length',3)]:
        p.add_argument('--'+name,type=int,default=int(os.getenv(name.upper().replace('-', '_'), default)))
    p.add_argument('--draft-token-length-c',type=float,default=float(os.getenv('DRAFT_TOKEN_LENGTH_C', '.75')))
    p.add_argument('--temperature',type=float,default=1.);p.add_argument('--top-p',type=float,default=.95)
    p.add_argument('--top-k',type=int,default=0);p.add_argument('--draft-lr',type=float,default=1e-4)
    p.add_argument('--visited-weight',type=float,default=1.);p.add_argument('--frontier-weight',type=float,default=1.)
    p.add_argument('--dtype',choices=['bf16','fp16'],default='bf16');p.add_argument('--attn-implementation',default='sdpa')
    p.add_argument('--online-draft',action='store_true',help='Same upstream loss/optimizer/cadence after each rollout; target remains frozen')
    p.add_argument('--profile',action='store_true',help='Separate frozen replay for inclusive OPD timings; excluded from measured wall time')
    p.add_argument('--strategy-replay',default=os.getenv('TLT_STRATEGY_REPLAY',''))
    p.add_argument('--tiny',action='store_true',help='Explicit real-transformer CUDA fixture; never a full-model benchmark')
    p.add_argument('--dry-run',action='store_true')
    return p


def parse_args(argv=None):
    p=parser();a=p.parse_args(argv)
    for key,cast in [('batch_sizes',int),('seeds',int),('fast_lrs',float),('streams',int)]:
        try:setattr(a,key,[cast(x) for x in getattr(a,key).split(',')])
        except ValueError:p.error('invalid numeric list: '+key)
    if not a.seeds or not a.batch_sizes or min(a.batch_sizes+a.streams)<0 or any(x<1 for x in a.batch_sizes):p.error('positive batch sizes, nonempty seeds required')
    if not a.fast_lrs or min(a.fast_lrs)<0 or not set(a.streams)<={0,1}:p.error('LR>=0 and streams 0/1 required')
    if min(a.responses,a.iterations,a.max_length,a.max_prompt_length,a.rank,a.topk,a.draft_accumulation_steps)<1 or a.warmup<0:p.error('positive sizes, warmup>=0 required')
    if min(a.verification_capacity,a.max_draft_token_length,a.max_draft_k,a.min_draft_token_length)<1 or a.max_verification_num<2 or a.draft_token_length_c<=0:
        p.error('invalid verification/adaptive limits')
    if a.min_draft_token_length>a.max_draft_token_length:p.error('minimum draft depth exceeds maximum')
    if not a.tiny and not all((a.target_model,a.draft_checkpoint,a.dataset_path)):p.error('real target/draft/dataset paths required (or explicit --tiny fixture)')
    if a.profile and a.online_draft:p.error('--profile is a separate frozen replay; run online-draft measurement separately')
    return a


def ordered_methods(seed):
    return ['tlt','tlt_opd_reflex'] if seed%2==0 else ['tlt_opd_reflex','tlt']


def canonical_config(args,batch,seed,method,lr,stream):
    return dict(architecture='TLT adaptive rollout + FastGRPO drafter',
        target_model=args.target_model,target_adapter=args.target_adapter,draft_checkpoint=args.draft_checkpoint,
        dataset=args.dataset_path,fixture=args.tiny,prompt_order='dataset order, no shuffle',
        batch_size=batch,responses=args.responses,seed=seed,temperature=args.temperature,top_p=args.top_p,top_k=args.top_k,
        dtype=args.dtype,attention=args.attn_implementation,max_length=args.max_length,max_prompt_length=args.max_prompt_length,
        warmup=args.warmup,measured_iterations=args.iterations,tlt=asdict(TLTConfig.from_env()),profiling_replay=args.profile,
        sampler_mode=os.getenv('OPD_SAMPLER_MODE','finite'),
        draft_limits=dict(verification_capacity=args.verification_capacity,max_draft_token_length=args.max_draft_token_length,
            max_draft_k=args.max_draft_k,max_verification_num=args.max_verification_num,
            min_draft_token_length=args.min_draft_token_length,draft_token_length_c=args.draft_token_length_c),
        training=dict(online_draft=args.online_draft,objective='fastgrpo_smoothl1_2_ce_0.1',lr=args.draft_lr,
            optimizer='AdamW',accumulation=args.draft_accumulation_steps,update_cadence='existing draft optimizer boundary',target_frozen=True),
        verifier='SpecNaacl FastGRPO PackedTree/sample-once token matching',cuda_graph=False,
        strategy_replay=args.strategy_replay,
        opd=dict(enabled=method=='tlt_opd_reflex',rank=args.rank,topk=args.topk,fast_lr=lr,update_stream=stream,
            visited_weight=args.visited_weight,frontier_weight=args.frontier_weight,
            train_projector=args.online_draft,profile=os.getenv('OPD_PROPOSAL_PROFILE',''),
            profile_dir=os.getenv('OPD_PROPOSAL_PROFILE_DIR',''),proposal_mode=os.getenv('OPD_PROPOSAL_MODE','auto'),
            dense_implementation=os.getenv('OPD_DENSE_IMPLEMENTATION','auto')))


def config_diff(a,b):
    keys=set(a)|set(b);difference={key:dict(tlt=a.get(key),tlt_opd_reflex=b.get(key)) for key in keys if a.get(key)!=b.get(key)}
    if set(difference)-{'opd'}:raise ValueError('unfair pair config: '+str(set(difference)-{'opd'}))
    return dict(only_opd_differences=True,differences=difference,baseline=a,reflex=b)


def validate_opd_experiment(output, method):
    if method == 'tlt_opd_reflex':
        if output.get('tlt_speculative_rounds', 0) <= 0:
            raise ValueError('invalid OPD experiment: no speculative rounds; set TLT_BS_THRESHOLD to initial live batch '
                             'and TLT_SD_WARMUP_CHECKS=1, and allow enough decoding rounds')
        if output.get('opd_feedback_calls', 0) <= 0 or output.get('opd_selected_states', 0) <= 0:
            raise ValueError('invalid OPD experiment: no OPD feedback states')


def load_model(args,method):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,Qwen2Config,Qwen2ForCausalLM
    from helper.fastgrpo_model import FastGRPOModel
    dtype=torch.bfloat16 if args.dtype=='bf16' else torch.float16
    torch.manual_seed(321)
    saved_projector=None
    if args.tiny:
        cfg=Qwen2Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,
            num_key_value_heads=2,max_position_embeddings=512,attention_dropout=0.,torch_dtype=dtype)
        cfg._attn_implementation='sdpa'
        target=Qwen2ForCausalLM(cfg).cuda().to(dtype).eval()
        tokenizer=type('Tokenizer',(),dict(eos_token_id=96))()
    else:
        target=AutoModelForCausalLM.from_pretrained(args.target_model,torch_dtype=dtype,
            attn_implementation=args.attn_implementation,local_files_only=True).cuda().eval()
        tokenizer=AutoTokenizer.from_pretrained(args.target_model,padding_side='left',local_files_only=True)
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        if target.config.model_type=='llama':tokenizer.pad_token,tokenizer.pad_token_id='<|end_of_text|>',128001
    config=deepcopy(target.config);config.rope_scaling=None;config.num_hidden_layers=1;config.torch_dtype=dtype
    model=FastGRPOModel(config,target).cuda().eval()
    if not args.tiny:
        path=Path(args.draft_checkpoint);path=path/'draft.pth' if path.is_dir() else path
        state=torch.load(path,map_location='cpu',weights_only=True)['draft_model'];saved_projector=state.pop('opd_projector',None)
        model.draft_model.load_state_dict(state)
        if args.target_adapter:
            from peft import PeftModel
            model.target_model=PeftModel.from_pretrained(target,args.target_adapter).eval()
    if method=='tlt_opd_reflex':
        model.enable_opd(args.rank)
        if saved_projector is not None:model.load_opd_projector(saved_projector)
    for param in model.draft_model.parameters():param.requires_grad_(args.online_draft)
    return model,tokenizer


def make_batches(args,tokenizer,batch):
    import torch
    if args.tiny:
        return [dict(input_ids=torch.tensor([[3,5,7+i%20]]*batch,device='cuda'),attention_mask=torch.ones((batch,3),device='cuda',dtype=torch.long)) for i in range(args.iterations)]
    from helper.get_QAs import get_QAs_from_path
    data=get_QAs_from_path(args.dataset_path,'train');needed=batch*args.iterations
    if len(data)<needed:raise ValueError(f'need {needed} prompts, found {len(data)}')
    node=next(n for n in ast.parse((ROOT/'grpo_speculative.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='TrainDataCollator')
    scope={};exec(compile(ast.Module(body=[node],type_ignores=[]),'collator','exec'),scope)
    collate=scope['TrainDataCollator'](tokenizer,args.max_prompt_length)
    batches=[]
    for offset in range(0,needed,batch):
        value=collate(data[offset:offset+batch])
        if value['input_ids'].shape[-1]>=args.max_length:raise ValueError('prompt exhausts max_length')
        batches.append({k:value[k].cuda() for k in ('input_ids','attention_mask')})
    return batches


def benchmark(args):
    if args.dry_run:
        return _benchmark(args)
    from helper import opd_sampling
    previous = opd_sampling.SAMPLER_MODE
    opd_sampling.SAMPLER_MODE = os.getenv('OPD_SAMPLER_MODE', 'finite')
    try:
        return _benchmark(args)
    finally:
        opd_sampling.SAMPLER_MODE = previous


def _benchmark(args):
    destination=Path(args.output);destination.mkdir(parents=True,exist_ok=True)
    plan=[]
    for batch in args.batch_sizes:
        for seed in args.seeds:
            for lr in args.fast_lrs:
                for stream in args.streams:
                    methods=ordered_methods(seed) if args.method=='pair' else [args.method]
                    a=canonical_config(args,batch,seed,'tlt',lr,stream);b=canonical_config(args,batch,seed,'tlt_opd_reflex',lr,stream)
                    plan.append(dict(batch=batch,seed=seed,lr=lr,stream=stream,order=methods,config_diff=config_diff(a,b)))
    (destination/'config_diff.json').write_text(json.dumps(plan,indent=2)+'\n')
    if args.dry_run:return dict(dry_run=True,plan=plan)
    import torch
    from helper.specualtive_generate import speculative_generate
    from helper.fastgrpo_training import training_draft_model
    from helper.opd_optimizer import draft_optimizer
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required; no fabricated benchmark')
    rows=[];responses=[];traces=[]
    for case in plan:
        batch,seed,lr,stream=case['batch'],case['seed'],case['lr'],case['stream']
        prompt_hashes=[]
        for position,method in enumerate(case['order'],1):
            case_dir=destination/'runs'/f'batch{batch}_seed{seed}_lr{lr}_stream{stream}'/method
            case_dir.mkdir(parents=True,exist_ok=True)
            (case_dir/'strategy_trace.jsonl').write_text('')
            model,tokenizer=load_model(args,method);batches=make_batches(args,tokenizer,batch)
            digest=hashlib.sha256()
            for entry in batches:
                for key in ('input_ids','attention_mask'):digest.update(entry[key].cpu().numpy().tobytes())
            prompt_hashes.append(digest.hexdigest())
            initial={k:v.detach().clone() for k,v in model.draft_model.state_dict().items()}
            counts={'target':0,'draft':0}
            h=model.target_model.model.layers[0].register_forward_pre_hook(lambda *u:counts.__setitem__('target',counts['target']+1))
            d=model.register_forward_pre_hook(lambda *u:counts.__setitem__('draft',counts['draft']+1))
            cfg=TLTConfig.from_env()
            def reset(measured=False):
                model.draft_model.load_state_dict(initial)
                model.zero_grad(set_to_none=True)
                if method=='tlt_opd_reflex':model.opd_projector_grad_sum.zero_();model.opd_projector_grad_weight.zero_()
                replay=args.strategy_replay if measured else ''
                scheduler=TLTScheduler(cfg,trace_path='',replay_path=replay)
                opt=draft_optimizer(model.draft_model,args.draft_lr) if args.online_draft else None
                return scheduler,opt
            def rollout(entry,actual_seed,scheduler,opt,index,profile=False):
                torch.manual_seed(actual_seed)
                with torch.inference_mode():
                    output=speculative_generate(model,entry['input_ids'],entry['attention_mask'],tokenizer,
                        method=method,tlt_scheduler=scheduler,do_sample=True,repeated_generate_nums=args.responses,
                        temperature=args.temperature,top_p=args.top_p,top_k=args.top_k or None,max_length=args.max_length,
                        verification_capacity=args.verification_capacity,max_draft_token_length=args.max_draft_token_length,
                        max_draft_k=args.max_draft_k,max_verification_num=args.max_verification_num,
                        min_draft_token_length=args.min_draft_token_length,draft_token_length_c=args.draft_token_length_c,
                        return_all_draft_input=args.online_draft,statistical_time=False,opd_rank=args.rank,opd_topk=args.topk,
                        opd_fast_lr=lr,opd_update_stream=bool(stream),opd_visited_weight=args.visited_weight,
                        opd_frontier_weight=args.frontier_weight,opd_train_projector=args.online_draft,opd_profile=profile)
                return output
            def train_rollout(output,entry,opt,index):
                if args.online_draft:
                    for key in ('all_draft_input_states','all_draft_input_ids'):output[key]=[x.clone() for x in output[key]]
                    model.train();model.target_model.eval()
                    training_draft_model(model,output,entry['attention_mask'],repeated_generate_nums=args.responses,
                        max_training_token=1024,max_training_padding_gap=4096,draft_accumulation_steps=args.draft_accumulation_steps)
                    # Match native accumulation: committed at the common draft boundary.
                    if (index+1)%args.draft_accumulation_steps==0:
                        if method=='tlt_opd_reflex':model.apply_opd_projector_gradient()
                        opt.step();opt.zero_grad()
                    model.eval()
            for _ in range(args.warmup):
                scheduler,opt=reset()
                for i,entry in enumerate(batches):
                    warm_output=rollout(entry,seed+i,scheduler,opt,i)
                    if args.online_draft: train_rollout(warm_output,entry,opt,i)
                del warm_output
            scheduler,opt=reset(measured=True)
            try:
                for i,entry in enumerate(batches):
                    replay_state=scheduler.state_dict() if args.profile else None
                    before=counts.copy();torch.cuda.reset_peak_memory_stats()
                    train=(lambda output:train_rollout(output,entry,opt,i)) if args.online_draft else None
                    output,times=time_generation_and_training(lambda:rollout(entry,seed+i,scheduler,opt,i),train,
                        synchronize=torch.cuda.synchronize)
                    wall=times['generation_wall_s']
                    validate_opd_experiment(output, method)
                    forwards={k:counts[k]-before[k] for k in counts}
                    if forwards['target']!=1+output['batch_verification_rounds']:raise AssertionError('extra target transformer forward')
                    row=dict(method=method,batch_size=batch,seed=seed+i,case_seed=seed,iteration=i,fast_lr=lr,stream=stream,run_position=position,
                        prompt_sha256=digest.hexdigest(),**times,generated_tokens=sum(output['response_generated_tokens']),
                        aal=output['effective_aal'],effective_aal=output['effective_aal'],speculative_aal=output['speculative_aal'],
                        target_only_rounds=output['target_only_rounds'], speculative_rounds=output['speculative_rounds'],
                        speculative_round_ratio=output['speculative_round_ratio'],
                        verification_rounds=output['total_decoded_token_num'],batch_verification_rounds=output['batch_verification_rounds'],
                        accepted_draft_tokens=output['total_accepted_draft_tokens'],proposed_draft_tokens=output['total_proposed_draft_tokens'],
                        acceptance_rate=output['draft_acceptance_rate'],target_forwards=forwards['target'],draft_forwards=forwards['draft'],
                        tokens_per_s=sum(output['response_generated_tokens'])/wall,
                        generation_tokens_per_s=sum(output['response_generated_tokens'])/wall,
                        combined_tokens_per_s=sum(output['response_generated_tokens'])/times['combined_wall_s'],
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        no_extra_target_forward=True,
                        **{k:v for k,v in output.items() if k.startswith(('tlt_','opd_')) and isinstance(v,(int,float,str,dict))})
                    rows.append(row)
                    for original in output['tlt_strategy_trace']:
                        traces.append(dict(method=method,case_seed=seed,batch_size_case=batch,fast_lr=lr,stream=stream,**original))
                    with (case_dir/'strategy_trace.jsonl').open('a') as f:
                        for original in output['tlt_strategy_trace']:f.write(json.dumps(original)+'\n')
                    for n,tokens in enumerate(output['generated_token_ids']):
                        responses.append(dict(method=method,seed=seed+i,batch_size=batch,fast_lr=lr,stream=stream,prompt_id=i*batch+n//args.responses,
                            response_index=n%args.responses,tokens=tokens))
                    if args.profile:
                        replay_scheduler=TLTScheduler(cfg,trace_path='',replay_path=args.strategy_replay)
                        replay_scheduler.load_state_dict(replay_state)
                        profiling=rollout(entry,seed+i,replay_scheduler,None,i,profile=True)
                        row['opd_overhead_ms']=profiling['opd_overhead_ms']
                        row['opd_feedback_ms']=profiling['opd_feedback_ms']
                        row['opd_profile_sections_ms']=profiling.get('opd_profile_sections_ms')
                        row['opd_overhead_measured']=profiling['opd_overhead_measured']
                        row['opd_overhead_basis']='separate frozen replay: inclusive feature/proposal/feedback GPU time'
            finally:h.remove();d.remove()
            del model,tokenizer,initial,batches,scheduler,opt,output
            gc.collect();torch.cuda.empty_cache()
        if len(set(prompt_hashes))!=1:raise AssertionError('pair prompt order/content mismatch')
    report=dict(architecture='TLT adaptive rollout + FastGRPO drafter',fixture=args.tiny,
        online_draft=args.online_draft,run_order_policy='seed parity',cases=plan,rows=rows,
        no_extra_target_forward=all(row['no_extra_target_forward'] for row in rows))
    (destination/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,data in [('responses.jsonl',responses),('strategy_trace.jsonl',traces)]:
        (destination/name).write_text(''.join(json.dumps(row)+'\n' for row in data))
    fields=['method','batch_size','seed','case_seed','iteration','fast_lr','stream','run_position','aal','effective_aal','speculative_aal','opd_overhead_ms','opd_feedback_ms','accepted_draft_tokens','proposed_draft_tokens',
        'acceptance_rate','verification_rounds','target_forwards','draft_forwards','generation_wall_s','draft_training_wall_s','combined_wall_s',
        'generation_tokens_per_s','combined_tokens_per_s','tokens_per_s','peak_allocated_bytes',
        'tlt_target_only_rounds','tlt_speculative_rounds','tlt_sd_transition_count','tlt_transition_draft_prefill_s']
    with (destination/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    return report


def main(argv=None):
    args=parse_args(argv);report=benchmark(args)
    print(json.dumps(dict(output=args.output,measured_rows=len(report.get('rows',[])),dry_run=args.dry_run,no_extra_target_forward=report.get('no_extra_target_forward'))))

if __name__=='__main__':main()
