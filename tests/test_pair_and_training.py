import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from scripts.benchmark_tlt_opd import parse_args,canonical_config,config_diff,ordered_methods
from helper.opd_profiles import fingerprint,execution_key,validate_profile,discover_profile
from helper.rollout_metrics import RolloutMetricsWriter
ROOT=Path(__file__).resolve().parents[1]


def test_pair_only_opd_config_changes_and_counterbalance(tmp_path):
    args=parse_args(['--tiny','--output',str(tmp_path),'--seeds','42,43','--batch-sizes','1,2'])
    a=canonical_config(args,2,42,'tlt',.01,1);b=canonical_config(args,2,42,'tlt_opd_reflex',.01,1)
    assert config_diff(a,b)['only_opd_differences']
    assert ordered_methods(42)==['tlt','tlt_opd_reflex'];assert ordered_methods(43)==list(reversed(ordered_methods(42)))
    b['draft_checkpoint']='different';
    with pytest.raises(ValueError,match='unfair'):config_diff(a,b)
    b['draft_checkpoint']=a['draft_checkpoint'];b['training']['lr']=.01
    # Canonical training objects are independent, not accidentally aliased.
    with pytest.raises(ValueError,match='unfair'):config_diff(a,b)


@pytest.mark.parametrize('key',['qwen25_1p5b','qwen25_3b','qwen25_7b','qwen25_14b','qwen3_1p7b','qwen3_4b','llama31_8b'])
def test_paired_model_launchers_share_configs_and_source_paths(tmp_path,key):
    commands=[]
    for name in (f'train_{key}_tlt.sh',f'train_{key}.sh'):
        env=dict(os.environ,DRY_RUN='true',PYTHON_BIN=sys.executable,OUTPUT_ROOT=str(tmp_path),PYTHONDONTWRITEBYTECODE='1')
        result=subprocess.run(['bash',str(ROOT/name)],env=env,text=True,capture_output=True)
        assert result.returncode==0,result.stderr
        line=next(line for line in result.stdout.splitlines() if line.startswith('Command  :'))
        import shlex
        command=shlex.split(line[len('Command  :'):])
        pairs={command[i]:command[i+1] for i in range(len(command)-1) if command[i].startswith('--')}
        commands.append(pairs)
        assert 'SpecNaacl/outputs/pretrain' in pairs['--adapter_path']
        assert pairs['--max_draft_token_length']=='5'
        assert pairs['--max_draft_k']=='8'
        assert pairs['--max_verification_num']=='160'
        assert pairs['--verification_capacity']=='512'
    a,b=commands
    shared=set(a)&set(b)
    excluded={'--method','--version_name','--log_file','--timing_file','--summary_file','--saved_model_dir','--saved_draft_model_dir','--saved_statistics_dir','--checkpoint_dir'}
    assert all(a[name]==b[name] for name in shared-excluded)
    assert a['--method']=='tlt' and b['--method']=='tlt_opd_reflex'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA identity')
def test_profile_is_tlt_native_and_missing_calibrated_fails(tmp_path,monkeypatch):
    hw=fingerprint();assert hw['runtime']=='tlt_fastgrpo_v1' and len(hw['tlt_execution_sha256'])==64
    key=execution_key(hw,97,8,'bf16');source_key={k:v for k,v in key.items() if k not in ('runtime','tlt_execution_sha256')}
    profile={'execution_key':source_key,'records':[{'contexts':1,'trials':[dict(slots=0,sparse=1,fused=2,gemm=3)]}]}
    with pytest.raises(ValueError,match='incompatible'):validate_profile(profile,key)
    monkeypatch.setenv('OPD_REQUIRE_CALIBRATED_PROFILE','1')
    with pytest.raises(ValueError,match='tune_tlt_opd_proposals'):discover_profile(tmp_path,key)


def test_tlt_metrics_capture_and_csv(tmp_path):
    output=dict(total_acc_length=8,total_decoded_token_num=4,total_accepted_draft_tokens=4,total_proposed_draft_tokens=8,
        response_generated_tokens=[8],tlt_target_only_rounds=1,tlt_speculative_rounds=3,tlt_sd_transition_count=1,
        tlt_transition_draft_prefill_s=.01,target_forwards=5,draft_forwards=25,tlt_strategy_trace=['no tensor retention'])
    captured=RolloutMetricsWriter.capture(output);assert 'tlt_strategy_trace' not in captured
    w=RolloutMetricsWriter(tmp_path/'metrics.csv','tlt');w.begin(1,0,1,0);w.finish(captured,grpo_step=1,used_items=1,wall_time_s=1);w.close()
    assert w.state['tlt_sd_transition_count']==1 and 'iter_target_forwards' in (tmp_path/'metrics.csv').read_text()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual entrypoint CUDA')
@pytest.mark.parametrize('method',['tlt','tlt_opd_reflex'])
def test_training_entrypoint_objectives_optimizer_order_and_scheduler_checkpoint(tmp_path,method):
    env=dict(os.environ,TLT_SD_WARMUP_CHECKS='2',OPD_REQUIRE_CALIBRATED_PROFILE='0',OPD_PROPOSAL_PROFILE='',OPD_PROPOSAL_PROFILE_DIR=str(tmp_path/'profiles'),TQDM_DISABLE='1')
    result=subprocess.run([sys.executable,str(ROOT/'tests/tiny_training_runner.py'),str(tmp_path),method,'run','--max_grpo_steps','1'],env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr+'\n'+result.stdout[-4000:]
    events=json.loads((tmp_path/'run/test_events.json').read_text());assert events==['rollout','draft_backward','draft_step','target_step']
    state=torch.load(tmp_path/'run/resume/latest.pt',map_location='cpu',weights_only=False)
    assert 'tlt_scheduler' in state['rank_states'][0]
    assert ('opd_projector' in state['draft_model'])==(method=='tlt_opd_reflex')
    summary=json.loads((tmp_path/'run/summary.json').read_text())
    assert summary['architecture']=='TLT adaptive rollout + FastGRPO drafter'
    assert summary['tlt_cumulative_metrics']['tlt_sd_transition_count']==1


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA pair fixture')
def test_pair_runtime_counterbalances_and_writes_reviewable_outputs(tmp_path):
    env=dict(os.environ,TLT_SD_WARMUP_CHECKS='2',OPD_REQUIRE_CALIBRATED_PROFILE='0',OPD_PROPOSAL_PROFILE='',OPD_PROPOSAL_PROFILE_DIR=str(tmp_path/'profiles'),TLT_STRATEGY_REPLAY='')
    result=subprocess.run([sys.executable,str(ROOT/'scripts/benchmark_tlt_opd.py'),'--tiny','--output',str(tmp_path),
        '--batch-sizes','1,2','--responses','2','--seeds','42,43','--fast-lrs','0.01','--streams','1',
        '--max-length','14','--iterations','1','--warmup','1'],env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    report=json.loads((tmp_path/'report.json').read_text());assert report['fixture'] and report['no_extra_target_forward']
    assert len(report['rows'])==8
    for case in report['cases']:
        assert case['order']==ordered_methods(case['seed']) and case['config_diff']['only_opd_differences']
        rows=[r for r in report['rows'] if r['batch_size']==case['batch'] and r['case_seed']==case['seed']]
        assert [r['method'] for r in rows]==case['order']
        assert len({r['prompt_sha256'] for r in rows})==1
        assert all(r['tlt_sd_transition_count']==1 and r['target_forwards']==1+r['batch_verification_rounds'] for r in rows)
    for name in ('report.json','summary.csv','responses.jsonl','strategy_trace.jsonl','config_diff.json'):assert (tmp_path/name).stat().st_size>0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA resume integration')
@pytest.mark.parametrize('method',['tlt','tlt_opd_reflex'])
def test_training_resume_preserves_scheduler_and_native_optimizer_rng(tmp_path,method):
    env=dict(os.environ,TLT_SD_WARMUP_CHECKS='2',OPD_REQUIRE_CALIBRATED_PROFILE='0',OPD_PROPOSAL_PROFILE='',OPD_PROPOSAL_PROFILE_DIR=str(tmp_path/'profiles'),TQDM_DISABLE='1')
    args=[sys.executable,str(ROOT/'tests/tiny_training_runner.py'),str(tmp_path),method,'run','--max_grpo_steps','1']
    first=subprocess.run(args,env=env,text=True,capture_output=True);assert first.returncode==0,first.stderr
    checkpoint=tmp_path/'run/resume/latest.pt'
    old=torch.load(checkpoint,map_location='cpu',weights_only=False)
    result=subprocess.run(args+['--resume_checkpoint',str(checkpoint)],env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr+'\n'+result.stdout[-4000:]
    assert json.loads((tmp_path/'run/test_restore.json').read_text())['bitwise_restore']
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['rank_states'][0]['tlt_scheduler']['rollout_id']==old['rank_states'][0]['tlt_scheduler']['rollout_id']+1
    assert state['draft_step']==old['draft_step']+1
