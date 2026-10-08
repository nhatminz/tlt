"""TLT integration for source compatibility, shared sampling and split timings."""
from pathlib import Path
from types import SimpleNamespace
import ast
import json
import pytest
import torch
from test_tlt_fastgrpo import tiny,run,TLTConfig
from scripts.benchmark_tlt_opd import time_generation_and_training,parse_args,canonical_config,config_diff
ROOT=Path(__file__).resolve().parents[1]
CUDA=pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA sampling/timing')


def test_benchmark_phase_intervals_are_adjacent_and_never_double_counted():
    calls=[];values=iter([10.,12.,17.])
    def generate():calls.append('generation');return {'tokens':6}
    def train(output):calls.append('draft_backward_update');assert output['tokens']==6
    def synchronize():calls.append('synchronize')
    output,times=time_generation_and_training(generate,train,synchronize=synchronize,clock=lambda:next(values))
    assert times==dict(generation_wall_s=2.,draft_training_wall_s=5.,combined_wall_s=7.)
    assert calls==['synchronize','generation','synchronize','draft_backward_update','synchronize']
    values=iter([10.,12.]);calls.clear()
    output,times=time_generation_and_training(generate,synchronize=synchronize,clock=lambda:next(values))
    assert times==dict(generation_wall_s=2.,draft_training_wall_s=0.,combined_wall_s=2.)


def test_pair_sampler_mode_is_shared_and_cannot_differ(monkeypatch,tmp_path):
    args=parse_args(['--tiny','--output',str(tmp_path)])
    for mode in ('strict','finite'):
        monkeypatch.setenv('OPD_SAMPLER_MODE',mode)
        a=canonical_config(args,2,42,'tlt',.01,1);b=canonical_config(args,2,42,'tlt_opd_reflex',.01,1)
        assert a['sampler_mode']==b['sampler_mode']==mode
        assert config_diff(a,b)['only_opd_differences']
        b['sampler_mode']='different'
        with pytest.raises(ValueError,match='unfair'):config_diff(a,b)


@CUDA
@pytest.mark.parametrize('method',['tlt','tlt_opd_reflex'])
@pytest.mark.parametrize('stream',[False,True])
def test_tlt_strict_finite_tokens_rng_teacher_feedback_and_acceptance(monkeypatch,method,stream):
    from helper import opd_sampling
    from helper.opd_reflex import OPDReflex
    answers=[];adapters=[];old_clear=OPDReflex.clear
    def capture(s):
        adapters.append(s.B_fast.detach().clone())
        old_clear(s)
    monkeypatch.setattr(OPDReflex,'clear',capture)
    for mode in ('strict','finite'):
        monkeypatch.setattr(opd_sampling,'SAMPLER_MODE',mode)
        model=tiny()
        answer=run(model,method,opd_update_stream=stream,opd_train_projector=True)
        grad=model.opd_projector_grad_sum.clone() if method=='tlt_opd_reflex' else None
        answers.append((answer,grad))
    (a,ga),(b,gb)=answers
    for name in ('generated_token_ids','total_acc_length','total_decoded_token_num','verification_batches',
                 'total_accepted_draft_tokens','total_proposed_draft_tokens','effective_aal','speculative_aal'):
        assert a[0][name]==b[0][name]
    assert torch.equal(a[2],b[2]) and a[1]==b[1]
    for name in ('all_draft_input_states','all_draft_input_ids'):
        for x,y in zip(a[0][name],b[0][name]):torch.testing.assert_close(x,y,rtol=0,atol=0)
    if method=='tlt_opd_reflex':
        # FP32 atomic feedback reductions are not bitwise deterministic, even
        # for repeated strict runs. Bound roundoff by eight eps at tensor scale;
        # probabilities, teacher metadata, sampled tokens and RNG stay exact.
        for x,y in ((adapters[0],adapters[1]),(ga,gb)):
            scale=max(x.abs().max().item(),y.abs().max().item())
            torch.testing.assert_close(x,y,rtol=0,atol=8*torch.finfo(x.dtype).eps*scale)
        for name in ('opd_selected_states','opd_visited_states','opd_frontier_states','opd_updates','opd_kl_sum'):
            assert a[0][name]==b[0][name]


@CUDA
def test_target_only_target_time_counts_every_forward_and_off_adds_no_wait(monkeypatch):
    from helper import tlt_generate
    from helper.tlt_timing import StreamTimer
    from unittest.mock import patch
    events=[];native_begin=StreamTimer.begin;native_end=StreamTimer.end;native_seconds=StreamTimer.seconds
    def begin(s):events.append(('begin',id(s)));native_begin(s)
    def end(s):events.append(('end',id(s)));native_end(s)
    def seconds(s):events.append(('seconds',id(s)));return native_seconds(s)
    monkeypatch.setattr(StreamTimer,'begin',begin);monkeypatch.setattr(StreamTimer,'end',end);monkeypatch.setattr(StreamTimer,'seconds',seconds)
    from helper.tlt_scheduler import TLTScheduler
    def execute(profile):
        model=tiny();torch.manual_seed(42)
        return tlt_generate.speculative_generate(model,torch.tensor([[3,5,7]]),torch.ones(1,3,dtype=torch.long),
            SimpleNamespace(eos_token_id=96),method='tlt',tlt_scheduler=TLTScheduler(TLTConfig(bs_threshold=0)),
            do_sample=True,max_length=12,statistical_time=profile)
    out=execute(True)
    assert out['target_time_cost']>0
    assert sum(e[0]=='seconds' for e in events)==out['target_forwards']
    events.clear();out=execute(False)
    assert out['target_time_cost']==0 and not events


@CUDA
def test_online_benchmark_reports_generation_and_training_separately(tmp_path,monkeypatch):
    from scripts.benchmark_tlt_opd import benchmark
    monkeypatch.setenv('OPD_REQUIRE_CALIBRATED_PROFILE','0')
    monkeypatch.setenv('OPD_PROPOSAL_PROFILE','')
    monkeypatch.setenv('OPD_PROPOSAL_PROFILE_DIR',str(tmp_path/'profiles'))
    monkeypatch.setenv('TLT_SD_WARMUP_CHECKS','2')
    args=parse_args(['--tiny','--output',str(tmp_path),'--batch-sizes','1','--responses','2',
        '--seeds','42','--fast-lrs','0.01','--iterations','2','--warmup','0','--max-length','12',
        '--online-draft','--draft-accumulation-steps','2'])
    report=benchmark(args)
    for row in report['rows']:
        assert row['generation_wall_s']>0 and row['draft_training_wall_s']>0
        assert row['combined_wall_s']==pytest.approx(row['generation_wall_s']+row['draft_training_wall_s'])
        assert row['generation_tokens_per_s']==pytest.approx(row['generated_tokens']/row['generation_wall_s'])
        assert row['combined_tokens_per_s']==pytest.approx(row['generated_tokens']/row['combined_wall_s'])
        assert row['tokens_per_s']==row['generation_tokens_per_s']
        assert row['no_extra_target_forward']
    assert 'draft_training_wall_s' in (tmp_path/'summary.csv').read_text()
