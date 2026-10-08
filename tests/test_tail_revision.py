"""Boundary, lazy memory, synchronization and target-only regression tests."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch
from test_tlt_fastgrpo import tiny,run,TLTConfig,TLTScheduler
from scripts.tune_opd_proposals import tail_workload,parse_args
ROOT=Path(__file__).resolve().parents[1]
CUDA=pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA lifecycle/timing')


def test_native_fastrl_boundary_pending_and_persistent_enable():
    from helper.tlt_scheduler import AdaptiveTail
    gate=AdaptiveTail(32,3)
    assert not gate.check(64)
    assert not gate.check(32) and not gate.pending
    assert not gate.check(32) and not gate.pending
    assert not gate.check(32) and gate.pending and not gate.enabled
    assert not gate.check(32)
    gate.complete_transition();assert gate.check(64)
    scheduler=TLTScheduler(TLTConfig(warmup_checks=2));scheduler.start_rollout(4)
    for _ in range(2):
        assert scheduler.select(4) is None;scheduler.record([1]*4,.01)
    assert scheduler.gate.pending
    scheduler.gate.complete_transition();assert scheduler.select(4)
    scheduler.record([2]*4,.01)
    metrics=scheduler.finish()
    assert metrics['effective_aal']==pytest.approx(4/3)
    assert metrics['speculative_aal']==2
    assert all(row['reward'] is None for row in scheduler.trace[:2])


def test_tuner_actual_tail_region_and_custom_strategies():
    live,k,shapes=tail_workload(32,8,32,'8_4_48,8_4_32,8_4_16,8_4_8')
    assert (live,k)==(32,4) and shapes[-1]==(128,1)
    live,k,shapes=tail_workload(1,2,32,'8_4_48,8_4_32,8_4_16,8_4_8')
    assert (live,k)==(2,4) and shapes[-1]==(8,1)
    with pytest.raises(ValueError,match='never enables'):tail_workload(32,8,0,'8_4_48,8_4_32,8_4_16,8_4_8')
    args=parse_args(['--batch-size','32','--responses','8','--tlt-bs-threshold','16']);assert args.tlt_bs_threshold==16


@CUDA
@pytest.mark.parametrize('method',['tlt','tlt_opd_reflex'])
def test_target_only_allocates_no_speculative_scratch_or_event_waits(monkeypatch,method):
    import helper.tlt_generate as runtime
    from helper.opd_reflex import OPDReflex
    from helper.tlt_timing import StreamTimer
    model=tiny()
    if method=='tlt_opd_reflex':model.enable_opd(8)
    def forbidden(*a,**k):pytest.fail('speculative allocation/synchronization in target-only phase')
    monkeypatch.setattr(runtime,'PackedTree',forbidden)
    monkeypatch.setattr(runtime,'TreeWorkspace',forbidden)
    monkeypatch.setattr(OPDReflex,'__init__',forbidden)
    monkeypatch.setattr(torch.cuda,'synchronize',forbidden)
    monkeypatch.setattr(StreamTimer,'seconds',forbidden)
    # No pending transition: deliberately require live batch<=0.
    output,counts,*_=run(model,method,config=TLTConfig(bs_threshold=0))
    assert counts['draft']==0 and output['tlt_speculative_workspace_batch']==0
    assert not hasattr(model,'_opd_runtime_cache') and not hasattr(model,'_tlt_tree_workspace')
    assert output['speculative_aal']==0 and output['effective_aal']==1
    names={name for name,dtype in model._opd_attention_workspace.buffers}
    assert not names.intersection({'tree_mask','target_tokens','target_positions','draft_positions'})


@CUDA
@pytest.mark.parametrize('method',['tlt','tlt_opd_reflex'])
@pytest.mark.parametrize('statistical',[False,True])
def test_pending_prefill_outside_mab_events_and_tail_capacity(monkeypatch,method,statistical):
    import helper.tlt_generate as runtime
    from helper.tlt_timing import StreamTimer
    from helper.opd_reflex import OPDReflex
    model=tiny()
    if method=='tlt_opd_reflex':model.enable_opd(8)
    scheduler=TLTScheduler(TLTConfig(bs_threshold=4,warmup_checks=3))
    prefill=runtime.prefill_draft_prefix;start=OPDReflex.start;events=[];allocated=[]
    def rebuild(m,features,ids,padding,**kw):
        assert scheduler.round==3 and scheduler.gate.pending and not scheduler.gate.enabled
        assert not hasattr(m,'_opd_runtime_cache') and not hasattr(m,'_tlt_tree_workspace')
        events.append(('prefill',scheduler.round));return prefill(m,features,ids,padding,**kw)
    def allocate(s,m,batch,*a,**kw):
        assert scheduler.round==3 and batch<=4
        allocated.append((batch,kw['max_nodes']))
        return start(s,m,batch,*a,**kw)
    begin=StreamTimer.begin;end=StreamTimer.end;secs=StreamTimer.seconds
    def begin_event(self):events.append(('begin',id(self),scheduler.round));return begin(self)
    def end_event(self):events.append(('end',id(self),scheduler.round));return end(self)
    def seconds(self):events.append(('seconds',id(self),scheduler.round));return secs(self)
    monkeypatch.setattr(runtime,'prefill_draft_prefix',rebuild)
    monkeypatch.setattr(OPDReflex,'start',allocate)
    monkeypatch.setattr(StreamTimer,'begin',begin_event);monkeypatch.setattr(StreamTimer,'end',end_event);monkeypatch.setattr(StreamTimer,'seconds',seconds)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *a,**k:pytest.fail('global device synchronization'))
    # run() defaults statistical_time=False; call dispatcher directly for flag.
    from types import SimpleNamespace
    torch.manual_seed(42)
    output=runtime.speculative_generate(model,torch.tensor([[0,7,9],[3,5,8]]),torch.tensor([[0,1,1],[1,1,1]]),
        SimpleNamespace(eos_token_id=96),method=method,tlt_scheduler=scheduler,do_sample=True,repeated_generate_nums=2,
        max_length=18,temperature=.8,top_p=.95,statistical_time=statistical,opd_profile=True)
    assert output['tlt_target_only_rounds']==3
    assert output['tlt_sd_transition_count']==1 and output['tlt_transition_draft_prefill_s']>0
    assert output['tlt_speculative_workspace_batch']<=4
    assert output['tlt_verification_capacity']==192
    assert all(row['reward'] is None for row in output['tlt_strategy_trace'][:3])
    if method=='tlt_opd_reflex':
        assert allocated==[(4,192)] and output['opd_overhead_ms']>0
        assert model._opd_runtime_cache[next(iter(model._opd_runtime_cache))].B_fast.count_nonzero()==0
    assert output['no_extra_target_forward']
    # No elapsed-time event wait in target-only rounds when diagnostics off.
    if not statistical:assert not any(e[0]=='seconds' and e[-1]<3 for e in events)


@CUDA
def test_pending_prefill_does_not_consume_target_rng(monkeypatch):
    import helper.tlt_generate as runtime
    prefill=runtime.prefill_draft_prefix
    def observe(*args,**kwargs):
        cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
        out=prefill(*args,**kwargs)
        assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
        return out
    monkeypatch.setattr(runtime,'prefill_draft_prefix',observe)
    run(tiny())


@CUDA
def test_terminal_trigger_round_skips_useless_prefill(monkeypatch):
    import helper.tlt_generate as runtime
    monkeypatch.setattr(runtime,'prefill_draft_prefix',lambda *a,**k:pytest.fail('terminal rollout rebuild'))
    # Trigger round reaches the length limit; pending rebuild must be skipped.
    out,counts,_,_,scheduler=run(tiny(),config=TLTConfig(warmup_checks=14))
    assert scheduler.gate.pending
    assert counts['draft']==0 and out['tlt_sd_transition_count']==0


def test_production_launcher_calibration_default_and_explicit_smoke():
    text=(ROOT/'scripts/run_tlt_opd_reflex.sh').read_text()
    assert 'OPD_REQUIRE_CALIBRATED_PROFILE:-1' in text
    for name in ('helper/tlt_generate.py','helper/tlt_timing.py'):
        tree=ast.parse((ROOT/name).read_text())
        assert not any(isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.cuda.synchronize' for n in ast.walk(tree))


@CUDA
@pytest.mark.parametrize('method',['tlt','tlt_opd_reflex'])
def test_large_batch_shrinks_before_lazy_tail_allocation_and_keeps_pending_a_grad(monkeypatch,method):
    import helper.tlt_generate as runtime
    from helper.opd_reflex import OPDReflex
    from types import SimpleNamespace
    model=tiny()
    if method=='tlt_opd_reflex':
        model.enable_opd(8)
        model.opd_projector_grad_sum.fill_(.25);model.opd_projector_grad_weight.fill_(3.)
        before=model.opd_projector_grad_sum.clone()
    sampled=runtime.sample_target_with_metadata;allocations=[];calls=0
    def force_finishing(logits,**kwargs):
        nonlocal calls
        result=sampled(logits,**kwargs);calls+=1
        # Controlled finishing fixture AFTER the original sampler call. Keeps
        # only two live responses at the first decode round, before threshold.
        if calls==2:result[0][:-2]=96
        return result
    original_start=OPDReflex.start
    def observe(s,m,b,*args,**kwargs):
        allocations.append((b,kwargs['max_nodes']));return original_start(s,m,b,*args,**kwargs)
    monkeypatch.setattr(runtime,'sample_target_with_metadata',force_finishing)
    monkeypatch.setattr(OPDReflex,'start',observe)
    scheduler=TLTScheduler(TLTConfig(bs_threshold=32,warmup_checks=2))
    torch.manual_seed(42)
    out=runtime.speculative_generate(model,torch.tensor([[3,5,7]]*32),torch.ones(32,3,dtype=torch.long),
        SimpleNamespace(eos_token_id=96),method=method,tlt_scheduler=scheduler,do_sample=True,
        repeated_generate_nums=2,max_length=18,temperature=.8,top_p=.95,return_all_draft_input=True)
    assert out['tlt_max_live']==32 and out['tlt_verification_capacity']==32*48
    assert out['tlt_speculative_workspace_batch']==2
    assert len(out['generated_token_ids'])==64
    if method=='tlt_opd_reflex':
        assert allocations==[(2,96)]
        torch.testing.assert_close(before,model.opd_projector_grad_sum,rtol=0,atol=0)
        assert model.opd_projector_grad_weight.item()==3.
