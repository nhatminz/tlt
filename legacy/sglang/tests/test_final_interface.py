"""Actual Hydra typing, pre-generation config gate, and parity/report contracts."""
import importlib.util,json,os,subprocess,sys
from pathlib import Path
import pytest,torch
ROOT=Path(__file__).resolve().parents[1]


def args(method='tlt',**values):
    from benchmark import parse_args
    result=parse_args(['--method',method,'--model','/model','--draft','/draft','--dataset','/data','--output','/output'])
    for key,value in values.items():setattr(result,key,value)
    return result


def test_actual_run_rl_hydra_preserves_strategy_strings_and_quoted_paths(tmp_path):
    from omegaconf import OmegaConf
    # Data/model files are not loaded during real upstream Hydra composition.
    env=dict(os.environ,PYTHON_BIN=sys.executable,DRY_RUN='true',MODEL=str(tmp_path/'Model, [001]'),
        DRAFT_EXPORT=str(tmp_path/'Draft, [002]'),RUN_DIR=str(tmp_path/'Run, [003]'),
        RL_DATA=str(tmp_path/'Data, [004].parquet'),EVAL_DATA=str(tmp_path/'Eval, [005].parquet'),
        MAB_CONFIGS='8_4_32,8_4_16,8_4_8',MAB_BUCKETS='1,2,5,21')
    process=subprocess.run(['bash',str(ROOT/'run_rl.sh'),'trainer.total_training_steps=2'],env=env,capture_output=True,text=True,check=True)
    cfg=OmegaConf.create(process.stdout.split('\n',1)[1])
    assert list(cfg.speculative.eagle.mab_configs)==['8_4_32','8_4_16','8_4_8']
    assert all(isinstance(value,str) for value in cfg.speculative.eagle.mab_configs)
    assert list(cfg.speculative.eagle.mab_bs_threshold)==[1,2,5,21]
    assert cfg.actor_rollout_ref.model.path==env['MODEL'] and cfg.speculative.eagle.spec_model_path==env['DRAFT_EXPORT']
    assert cfg.data.train_files==env['RL_DATA'] and cfg.data.val_files==env['EVAL_DATA']
    assert list(cfg.trainer.logger)==['console']


def test_canonical_gate_rejects_any_non_opd_difference():
    from tlt_reflex.benchmark_config import canonical_config,canonical_diff
    left=canonical_config(args());right=canonical_config(args('tlt_opd_reflex'))
    result=canonical_diff(left,right)
    assert result['benchmark_critical_fields_identical']
    assert all(d['path'].startswith('opd.') for d in result['differences'])
    for key,value in [('temperature',.9),('seed',43),('batch_size',4),('warmup',2),('disable_cuda_graph',True),('draft','/different-eagle')]:
        with pytest.raises(ValueError,match='critical'):canonical_diff(left,canonical_config(args('tlt_opd_reflex',**{key:value})))


def test_shell_pair_dumps_and_checks_canonical_configs_before_any_engine(tmp_path):
    env=dict(os.environ,PYTHON_BIN=sys.executable,DRY_RUN='true',PAIR_DIR=str(tmp_path/'pair'),
        BENCH_BATCH_SIZES='1,2',BENCH_SEEDS='42,43')
    subprocess.run(['bash',str(ROOT/'benchmark_tlt_opd_pair.sh')],env=env,capture_output=True,text=True,check=True)
    for batch in (1,2):
        for seed in (42,43):
            path=tmp_path/'pair'/f'b{batch}_s{seed}'
            cfg=json.loads((path/'tlt/canonical_config.json').read_text())
            assert cfg['workload']['sampling_seed']==seed and cfg['workload']['prompt_batch_size']==batch
            assert (path/'tlt_opd/canonical_config.json').is_file()
            diff=json.loads((path/'config_diff.json').read_text())
            assert diff['benchmark_critical_fields_identical'] and all(d['path'].startswith('opd.') for d in diff['differences'])
            assert not (path/'tlt/report.json').exists(),'dry preflight must not invoke an engine'


def test_corrected_logits_and_top16_probabilities_are_validation_gates():
    from tlt_reflex.parity import corrected_logits,compare_payloads
    torch.manual_seed(9);raw=torch.randn(1,37);u=torch.randn(1,8);b=torch.randn(37,8)*.02
    corrected=corrected_logits(raw,u,b);probs,ids=corrected.softmax(-1).topk(16)
    payload=dict(raw_logits=raw,head_input=torch.randn(1,12),u=u,projector=torch.randn(12,8),
        corrected_logits=corrected,top16_ids=ids,top16_probs=probs)
    result=compare_payloads(payload,payload)
    assert result['passed'] and result['max_abs_raw_logits_error']==0 and result['max_abs_corrected_logits_error']==0
    assert not compare_payloads(payload,dict(payload,corrected_logits=corrected+1))['passed']
    assert not compare_payloads(payload,dict(payload,top16_probs=probs+.01))['passed']
    missing=dict(payload);missing.pop('corrected_logits')
    assert not compare_payloads(payload,missing)['passed']


def test_export_trained_override_requires_existing_valid_a_and_preserves_training_record(tmp_path):
    from test_checkpoint_adapter import prepare
    from tlt_reflex.checkpoint import export,load_projector
    from tlt_reflex.experiments import projector_record,experiment_label
    ck,cfg,mapping,target,weights=prepare(tmp_path)
    with pytest.raises(ValueError,match='missing projector'):export(ck,cfg,mapping,target,tmp_path/'missing',projector_provenance='trained')
    a=torch.randn(8,3);weights['opd_projector']=a;torch.save(dict(draft_state_dict=weights,metadata=dict(opd_rank=3)),ck)
    out=export(ck,cfg,mapping,target,tmp_path/'trained',projector_provenance='trained',
        projector_training_dataset='training-only-split',projector_training_steps=20)
    actual,provenance=load_projector(out,8,3);assert torch.equal(a,actual) and provenance=='trained'
    record=projector_record(out)
    assert record['training_dataset']=='training-only-split' and record['training_steps']==20
    assert record['source_checkpoint']==str(ck.resolve())
    assert experiment_label('tlt_opd_reflex',provenance)!=experiment_label('tlt_opd_reflex','head_basis_initialized')
    weights['opd_projector']=torch.randn(7,3);torch.save(dict(draft_state_dict=weights,metadata=dict(opd_rank=3)),ck)
    with pytest.raises(ValueError,match='projector'):export(ck,cfg,mapping,target,tmp_path/'wrong-shape',projector_provenance='trained')


def test_smoke_is_separate_from_official_profile_and_parity_requirements():
    from tlt_reflex.benchmark_config import canonical_config,canonical_diff
    left=canonical_config(args(smoke=True));right=canonical_config(args('tlt_opd_reflex',smoke=True))
    with pytest.raises(ValueError,match='smoke'):canonical_diff(left,right)


def test_grid_aggregate_has_requested_outputs_without_double_counting_responses(tmp_path):
    from tlt_reflex.benchmark_config import canonical_config
    spec=importlib.util.spec_from_file_location('grid_summary',ROOT/'scripts/summarize_tlt_opd.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for seed in (42,43):
        case=tmp_path/f'b1_s{seed}'
        for method,folder,aal,tps in [('tlt','tlt',2.,100.),('tlt_opd_reflex','tlt_opd',2.1,102.)]:
            path=case/folder;path.mkdir(parents=True)
            a=args(method,seed=seed,batch_size=1)
            config=vars(a);canonical=canonical_config(a)
            report=dict(method=method,config=config,engine_config={},canonical_config=canonical,
                verified_aal=aal,tokens_per_s=tps,generation_wall_s=1.,opd_fast_lr=.01,opd_update_stream=1,
                spot_trainer_enabled=False,valid_for_official_comparison=True,artifact_identity={'same':'weights'},
                prompt_token_sha256='same',measured_prompts=1,generated_responses=1,opd_orphan_nodes=0,opd_invalid_contexts=0,
                real_eagle3_parity={'passed':True},opd_projector_experiment='head_basis_initialized')
            (path/'report.json').write_text(json.dumps(report));(path/'responses.jsonl').write_text('{}\n')
        from tlt_reflex.benchmark_config import canonical_diff
        (case/'config_diff.json').write_text(json.dumps(canonical_diff(canonical_config(args(seed=seed)),canonical_config(args('tlt_opd_reflex',seed=seed)))))
        m.summarize(case)
    rows=m.summarize(tmp_path)
    assert len(rows)==4 and len((tmp_path/'responses.jsonl').read_text().splitlines())==4
    for file in ('tlt/report.json','tlt_opd/report.json','comparison.json','summary.csv','config_diff.json'):assert (tmp_path/file).is_file()
    m.summarize(tmp_path)
    assert len((tmp_path/'responses.jsonl').read_text().splitlines())==4
    comparison=json.loads((tmp_path/'comparison.json').read_text())
    assert comparison['projector_experiments']==['head_basis_initialized'] and len(comparison['pairs'])==2
    assert 'peak_allocated_gb' in comparison['pairs'][0]['delta']
