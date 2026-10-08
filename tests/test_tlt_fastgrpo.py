import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM
from helper.fastgrpo_model import FastGRPOModel
from helper.tlt_scheduler import TLTScheduler, TLTConfig, Strategy, AdaptiveTail
from helper.specualtive_generate import speculative_generate
from helper.tlt_transition import prefill_draft_prefix
ROOT=Path(__file__).resolve().parents[1]
CUDA=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA transformer regression')


def tiny(family='qwen2'):
    torch.manual_seed(321)
    config_cls,model_cls=(Qwen2Config,Qwen2ForCausalLM) if family=='qwen2' else (Qwen3Config,Qwen3ForCausalLM)
    config=config_cls(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=256,
        attention_dropout=0.,torch_dtype=torch.bfloat16)
    config._attn_implementation='sdpa'
    target=model_cls(config).cuda().bfloat16().eval()
    dc=deepcopy(config);dc.num_hidden_layers=1;dc.rope_scaling=None
    return FastGRPOModel(dc,target).cuda().eval()


def run(model,method='tlt',seed=42,config=None,**kwargs):
    if method=='tlt_opd_reflex' and model.opd_projector is None:model.enable_opd(8)
    s=kwargs.pop('tlt_scheduler',None) or TLTScheduler(config or TLTConfig(warmup_checks=3))
    counts={'target':0,'draft':0};draft_rounds=[]
    def dh(*unused):counts['draft']+=1;draft_rounds.append(s.round)
    th=model.target_model.model.layers[0].register_forward_pre_hook(lambda *unused:counts.__setitem__('target',counts['target']+1))
    hook=model.register_forward_pre_hook(dh)
    torch.manual_seed(seed)
    try:
        out=speculative_generate(model,torch.tensor([[0,7,9],[3,5,8]]),torch.tensor([[0,1,1],[1,1,1]]),
            SimpleNamespace(eos_token_id=kwargs.pop('eos_token_id',96)),do_sample=True,repeated_generate_nums=2,
            max_length=18,temperature=.8,top_p=.95,statistical_time=False,return_all_draft_input=True,
            method=method,tlt_scheduler=s,**kwargs)
        return out,counts,torch.cuda.get_rng_state(),draft_rounds,s
    finally:th.remove();hook.remove()


def test_source_architecture_and_opd_are_exact_snapshots():
    manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    exact=('modeling_draft.py','fastgrpo_model.py','fastgrpo_training.py','opd_reflex.py',
           'opd_reflex_kernels.py','opd_sampling.py','opd_optimizer.py','tree_verification.py','tree_kernels.py')
    for name in exact:
        entry=next(x for x in manifest['files'] if x['destination']=='helper/'+name)
        assert hashlib.sha256((ROOT/entry['snapshot']).read_bytes()).hexdigest()==entry['sha256']
        assert (ROOT/'helper'/name).read_bytes()==(ROOT/entry['snapshot']).read_bytes()
    assert (ROOT/'helper/tlt_mab.py').read_bytes()==(ROOT/'sources/FastRL/eagle_mab.py').read_bytes()


def test_mapping_gating_capacity_and_no_native_adaptive_call():
    s=Strategy.parse('8_4_32');assert (s.depth,s.k,s.total_draft)==(8,4,31)
    gate=AdaptiveTail(32,3)
    assert [gate.check(b) for b in [64,32,31,64,30,10,1]]==[False]*7
    assert gate.pending and not gate.enabled
    gate.complete_transition();assert gate.check(64)
    for n in (48,32,16,8):assert Strategy.parse(f'8_4_{n}').total_draft==n-1
    with pytest.raises(ValueError,match='available'):Strategy.parse('1_4_48')
    sch=TLTScheduler();assert sch.start_rollout(64)==1536
    with pytest.raises(ValueError,match='no strategy clamping'):sch.start_rollout(64,160)
    tree=ast.parse((ROOT/'helper/tlt_generate.py').read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='get_adaptive_hyperparameters' for n in ast.walk(tree))


def test_mab_selection_reward_metrics_reference_and_rng_state():
    path=ROOT/'sources/FastRL/eagle_mab.py'
    spec=importlib.util.spec_from_file_location('fastrl_ref',path);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    configs='8_4_48,7_4_48,8_4_32,7_4_32,8_4_16,7_4_16,8_4_8,7_4_8'
    cfg=TLTConfig(warmup_checks=1,strategies=configs)
    s=TLTScheduler(cfg);s.start_rollout(32)
    s.gate.check(32);s.gate.complete_transition()
    reference=ref.MABGroupManager(configs.split(','),'BEG',1000,[1,2,5,21])
    private=np.random.RandomState(42)
    for batch,accepted,seconds in [(32,1.,.003),(8,1.5,.005),(2,2.,.007),(1,3.,.010)]*8:
        previous=np.random.get_state()
        selected=s.select(batch)
        np.random.set_state(private.get_state());expected=reference.select_strategy(batch)
        stable=reference.get_stable_accept_length(expected)
        reward=stable*batch/seconds
        reference.record_strategy_metrics(batch,expected,reward,accepted)
        private.set_state(np.random.get_state());np.random.set_state(previous)
        s.record([accepted]*batch,seconds)
        assert selected.name==expected
        assert s.trace[-1]['reward']==reward
        for group in s.manager.groups:
            for key in configs.split(','):
                assert list(s.manager.mabs[group].strategy_metrics[key].rewards)==list(reference.mabs[group].strategy_metrics[key].rewards)
        after=np.random.get_state();assert after[0]==previous[0];assert np.array_equal(after[1],previous[1]);assert after[2:]==previous[2:]
    state=s.state_dict();other=TLTScheduler(cfg);other.load_state_dict(state)
    assert other.select(2)==s.select(2)


@CUDA
@pytest.mark.parametrize('family',['qwen2','qwen3'])
def test_off_never_initializes_opd_and_no_extra_target_forward(monkeypatch,family):
    import helper.opd_reflex as opd
    monkeypatch.setattr(opd.OPDReflex,'__init__',lambda *a,**k:pytest.fail('OPD initialized in tlt'))
    model=tiny(family)
    out,counts,rng,drafts,s=run(model)
    assert model.opd_projector is None
    assert not hasattr(model,'_opd_runtime_cache')
    assert out['opd_backend']=='off'
    assert min(drafts)==3
    assert counts['target']==1+out['batch_verification_rounds']
    assert out['tlt_target_only_rounds']==3 and out['tlt_sd_transition_count']==1
    assert all(row['verification_num']==32 for row in out['tlt_strategy_trace'][3:])


@CUDA
@pytest.mark.parametrize('family',['qwen2','qwen3'])
def test_transition_direct_prefix_hidden_logits_kv_parity(monkeypatch,family):
    import helper.tlt_generate as runtime
    model=tiny(family);captured=[]
    def capture(m,features,ids,padding,**kw):
        result=prefill_draft_prefix(m,features,ids,padding,**kw)
        captured.append((features.clone(),ids.clone(),padding.clone(),
            {k:result[k].clone() for k in ('hidden_states','next_feature_states','position_ids')},
            [[x.clone() for x in layer] for layer in result['past_key_values']]))
        return result
    monkeypatch.setattr(runtime,'prefill_draft_prefix',capture)
    out,counts,*_=run(model)
    assert len(captured)==1 and counts['target']==out['batch_verification_rounds']+1
    features,ids,padding,state,kv=captured[0]
    # Three target-only steps appended to prompt features. Shifted ids end with
    # already sampled target bonus; all earlier ids form the current prefix.
    assert ids.shape==(4,6)
    from helper.modeling_draft import DraftModel
    config=deepcopy(model.target_model.config);config.num_hidden_layers=1;config.rope_scaling=None
    direct=DraftModel(config).cuda();direct.load_state_dict(model.draft_model.state_dict());direct.eval()
    minimum=torch.finfo(model.dtype).min
    mask=torch.triu(torch.full((6,6),minimum,device='cuda',dtype=model.dtype),diagonal=1)[None,None].repeat(4,1,1,1)
    mask.masked_fill_(padding[:,None,None,:],minimum)
    with torch.inference_mode(),torch.amp.autocast('cuda',dtype=model.dtype):
        reference=direct(features,model.embed_tokens(ids),attention_mask=mask,position_ids=state['position_ids'],use_cache=True)
        reference_logits=model.lm_head(reference['hidden_states'][:,-1:])
        transition_logits=model.lm_head(state['hidden_states'])
    torch.testing.assert_close(reference['hidden_states'][:,-1:],state['hidden_states'],rtol=0,atol=0)
    torch.testing.assert_close(reference['next_feature_states'][:,-1:],state['next_feature_states'],rtol=0,atol=0)
    torch.testing.assert_close(reference_logits,transition_logits,rtol=0,atol=0)
    for a,b in zip(reference['past_key_values'],kv):
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
    # Independently recompute target features from that SAME prefix, outside
    # rollout, to test the history alignment rather than only rebuild arithmetic.
    prefix=torch.cat((torch.tensor([[0],[0],[3],[3]],device='cuda'),ids[:,:-1]),1)
    with torch.inference_mode():
        actual=model.target_model(prefix,attention_mask=(~padding).long(),position_ids=state['position_ids'],output_hidden_states=True).hidden_states[-1]
    live=(~padding)[:,:,None].expand_as(actual)
    torch.testing.assert_close(actual[live].float(),features[live].float(),rtol=.025,atol=.025)
    with torch.inference_mode(),torch.amp.autocast('cuda',dtype=model.dtype):
        direct_prefix=direct(actual,model.embed_tokens(ids),attention_mask=mask,position_ids=state['position_ids'],use_cache=True)
        direct_logits=model.lm_head(direct_prefix['hidden_states'][:,-1:])
    torch.testing.assert_close(direct_logits.float(),transition_logits.float(),rtol=.025,atol=.025)
    torch.testing.assert_close(direct_prefix['hidden_states'][:,-1:].float(),state['hidden_states'].float(),rtol=.025,atol=.025)
    for a,b in zip(direct_prefix['past_key_values'],kv):
        for x,y in zip(a,b):
            # HF's 2D-mask prefill unblocks fully masked PAD queries. Their
            # hidden/KV contents are arbitrary and never attended; real-token
            # KV must match. Same captured features above match ALL KV exactly.
            live_kv=(~padding)[:,None,:,None].expand_as(x)
            torch.testing.assert_close(x[live_kv].float(),y[live_kv].float(),rtol=.025,atol=.025)


@CUDA
@pytest.mark.parametrize('stream',[False,True])
def test_on_zero_lr_positive_lr_training_and_b_reset(stream):
    from helper.fastgrpo_training import training_draft_model
    model=tiny();out,counts,rng,_,s=run(model,'tlt_opd_reflex',opd_fast_lr=.01,opd_update_stream=stream,opd_train_projector=True)
    assert counts['target']==1+out['batch_verification_rounds']
    assert out['opd_selected_states']>0 and out['opd_updates']>0
    assert torch.isfinite(model.opd_projector_grad_sum).all()
    for key in ('all_draft_input_states','all_draft_input_ids'):out[key]=[x.clone() for x in out[key]]
    loss=training_draft_model(model,out,torch.tensor([[0,1,1],[1,1,1]]),repeated_generate_nums=2,
        max_training_token=64,max_training_padding_gap=64,draft_accumulation_steps=1)
    assert torch.isfinite(torch.tensor(loss)).all()
    model.apply_opd_projector_gradient();before=model.opd_projector.detach().clone()
    torch.optim.AdamW(model.draft_model.parameters(),lr=1e-4).step()
    assert not torch.equal(before,model.opd_projector)
    zero=run(model,'tlt_opd_reflex',opd_fast_lr=0,opd_update_stream=stream)[0]
    assert zero['opd_updates']==0
    assert model._opd_target_kv_pool.get_seq_length()==0


@CUDA
@pytest.mark.parametrize('verify',[48,32,16,8])
@pytest.mark.parametrize('fast_lr',[0.,.01])
def test_depth8_k4_buffer_bounds_same_budget_and_replay(tmp_path,verify,fast_lr):
    cfg=TLTConfig(warmup_checks=2,strategies=f'8_4_{verify}',buckets=(1,))
    model=tiny();trace=tmp_path/'strategy_trace.jsonl'
    baseline=run(model,config=cfg,tlt_scheduler=TLTScheduler(cfg,trace_path=str(trace)))
    other=tiny()
    reflex=run(other,'tlt_opd_reflex',config=cfg,opd_fast_lr=fast_lr,tlt_scheduler=TLTScheduler(cfg,replay_path=str(trace)))
    a,b=baseline[0],reflex[0]
    assert a['tlt_strategy_trace'] and b['tlt_strategy_trace']
    for x,y in zip(a['tlt_strategy_trace'],b['tlt_strategy_trace']):
        assert (x['strategy'],x['verification_num'],x['batch_size'],x['phase'])==(y['strategy'],y['verification_num'],y['batch_size'],y['phase'])
    assert a['verified_tree_nodes']==b['verified_tree_nodes']
    assert baseline[1]['target']==reflex[1]['target']==1+a['batch_verification_rounds']


@CUDA
def test_target_only_sampling_rng_distribution_matches_source():
    from helper.fastgrpo_generate import sampling
    import helper.tlt_generate as runtime
    model=tiny();inputs=[]
    original=runtime.sample_target_with_metadata
    def observe(logits,**kwargs):
        before=torch.cuda.get_rng_state();result=original(logits,**kwargs);after=torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(before)
        expected=sampling(logits,kwargs.get('top_k'),kwargs.get('top_p'),kwargs['temperature'],kwargs['eos_token_id'])
        assert torch.equal(result[0],expected);assert torch.equal(after,torch.cuda.get_rng_state())
        torch.cuda.set_rng_state(after);inputs.append(logits.shape)
        return result
    from unittest.mock import patch
    with patch.object(runtime,'sample_target_with_metadata',observe):
        out,counts,*_=run(model,config=TLTConfig(bs_threshold=0))
    assert counts['draft']==0 and out['tlt_sd_transition_count']==0
    assert all(shape[1]==1 for shape in inputs)


@CUDA
def test_finishing_compaction_preserves_transition_history(monkeypatch):
    import helper.tlt_generate as runtime
    model=tiny();captures=[]
    original=runtime.prefill_draft_prefix
    def observe(m,features,ids,padding,**kw):
        captures.append((ids.clone(),features.clone()));return original(m,features,ids,padding,**kw)
    monkeypatch.setattr(runtime,'prefill_draft_prefix',observe)
    # Seed 42 produces EOS=72 for response zero at the first target-only round;
    # other rows stay live. Swap-remove happens BEFORE rebuild at round 2.
    out,counts,*_=run(model,eos_token_id=72)
    assert len(out['generated_token_ids'])==4
    assert captures and captures[0][0].shape[0]<4
    assert out['opd_target_row_compactions']>0
    assert counts['target']==1+out['batch_verification_rounds']
    assert all(len(x)==len(y) for x,y in zip(out['all_draft_input_ids'],out['all_draft_input_states']))


def slow_reference():
    path=ROOT/'tests/reference/tlt_generate_before_tail_fix.py'
    spec=importlib.util.spec_from_file_location('tlt_slow_reference',path)
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    return reference


class ReferenceScheduler(TLTScheduler):
    def start_rollout(self,batch,*args,**kwargs):
        super().start_rollout(batch,None,*args[1:],**kwargs)
        self.capacity=batch*max(s.verification_num for s in self.strategies)
        return self.capacity

    def record(self,*args):
        super().record(*args)
        # Reference's draft prefill remains at the next round start, but the
        # completed triggering round and next prefix now match revised TLT.
        if self.gate.pending:self.gate.complete_transition()


@CUDA
@pytest.mark.parametrize('family',['qwen2','qwen3'])
@pytest.mark.parametrize('threshold',[0,32])
def test_fast_path_and_transition_match_previous_verifier_tokens_rng_hidden_and_kv(monkeypatch,family,threshold):
    from unittest.mock import patch
    import helper.tlt_generate as runtime
    cfg=TLTConfig(bs_threshold=threshold,warmup_checks=3)
    reference=slow_reference()
    captured={}
    def execute(generate,model,scheduler,label):
        target_masks=[];kv=[]
        hook=model.target_model.model.layers[0].register_forward_pre_hook(lambda module,args,kw:target_masks.append(kw['attention_mask'].clone()),with_kwargs=True)
        norm=model.target_model.model.norm.register_forward_hook(lambda *unused:kv.append([[x.clone() for x in layer] for layer in model._opd_target_kv_pool]))
        original=generate._cache_set_layer
        def observe(cache,idx,key,value):
            original(cache,idx,key,value)
        torch.manual_seed(42)
        with patch.object(generate,'_cache_set_layer',observe):
            out=generate.speculative_generate(model,torch.tensor([[0,7,9],[3,5,8]]),torch.tensor([[0,1,1],[1,1,1]]),
                SimpleNamespace(eos_token_id=72),do_sample=True,repeated_generate_nums=2,max_length=18,
                temperature=.8,top_p=.95,statistical_time=False,return_all_draft_input=True,
                method='tlt',tlt_scheduler=scheduler)
        hook.remove();norm.remove()
        captured[label]=(out,torch.cuda.get_rng_state(),target_masks,kv)
    execute(reference,tiny(family),ReferenceScheduler(cfg),'old')
    execute(runtime,tiny(family),TLTScheduler(cfg),'new')
    native,rng,masks,kv=captured['old'];result,new_rng,new_masks,new_kv=captured['new']
    assert native['generated_token_ids']==result['generated_token_ids']
    assert torch.equal(rng,new_rng)
    for key in ('all_draft_input_ids','all_draft_input_states'):
        for x,y in zip(native[key],result[key]):torch.testing.assert_close(x,y,rtol=0,atol=0)
    for x,y in zip(masks,new_masks):torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert len(kv)==len(new_kv)
    for old_round,new_round in zip(kv,new_kv):
        for a,b in zip(old_round,new_round):
            for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert native['verification_batches']==result['verification_batches']
