import json,os,subprocess
from pathlib import Path
import pytest,torch
from tlt_reflex.ported.profiles import execution_key
from tlt_reflex.profiles import discover
from test_request_reflex import create,DEVICES
ROOT=Path(__file__).resolve().parents[1]


def profile(key):
    return dict(execution_key=key,records=[dict(contexts=1,trials=[dict(slots=0,sparse=1,fused=2,gemm=3),dict(slots=37,sparse=3,fused=2,gemm=4)])])


def test_reuse_source_profile_and_reject_stale_gpu_hash_dtype(tmp_path,monkeypatch):
    from tlt_reflex import profiles
    hw=dict(gpu='B200',compute_capability=[10,0],torch='test',triton='test',cuda='test',kernel_sha256='tlt')
    source=dict(hw,kernel_sha256='spec')
    monkeypatch.setattr(profiles,'fingerprint',lambda device=None,source=False:dict(hw,kernel_sha256='spec' if source else 'tlt'))
    key=execution_key(source,37,8,'fp32',16);file=tmp_path/'source.json';file.write_text(json.dumps(profile(key)))
    monkeypatch.setenv('OPD_PROPOSAL_PROFILE_DIR',str(tmp_path));monkeypatch.delenv('OPD_PROPOSAL_PROFILE',raising=False)
    selector,path=discover(37,8,torch.float32,16)
    assert path==str(file) and selector is not None
    monkeypatch.setenv('OPD_PROPOSAL_PROFILE',str(file))
    for field,value in [('gpu','3090'),('kernel_sha256','old'),('dtype','torch.bfloat16')]:
        file.write_text(json.dumps(profile(dict(key,**{field:value}))))
        with pytest.raises(ValueError,match='incompatible'):discover(37,8,torch.float32,16)


@pytest.mark.parametrize('wrapper',['train_qwen25_3b.sh','train_qwen25_3b_tlt.sh'])
def test_train_wrappers_select_two_production_modes_and_specnaacl_paths(wrapper):
    # Source shell interface without requiring Hydra, installed CUDA stack or data.
    script=f'''source scripts/launch_common.sh
printf '%s\\n' "$METHOD" "$MODEL" "$DATASET_PATH" "$DRAFT_CHECKPOINT" "$OPD_FAST_LR" "$OPD_UPDATE_STREAM" "$OPD_TRAIN_PROJECTOR" "$OPD_PROPOSAL_PROFILE_DIR"
'''
    mode='tlt' if wrapper.endswith('_tlt.sh') else 'tlt_opd_reflex'
    result=subprocess.run(['bash','-c',script],cwd=ROOT,env=dict(os.environ,METHOD=mode,MODEL_KEY='qwen25_3b'),capture_output=True,text=True,check=True)
    lines=result.stdout.splitlines()
    assert lines[0]==mode and lines[1]=='/workspace/storage-shared/models/Qwen2.5-3B-Instruct'
    assert lines[2].endswith('/data/simplelr_abel_level3to5/train.parquet')
    assert '/SpecNaacl/outputs/pretrain/qwen25_3b/latest_checkpoint' in lines[3]
    assert lines[4:7]==['0.01','1','0']
    assert '/SpecNaacl/outputs/benchmarks/opd_proposals' in lines[7]
    assert mode in (ROOT/wrapper).read_text()


@pytest.mark.parametrize('device',DEVICES)
def test_zero_lr_after_feedback_keeps_b_and_unique_top16_identity(device):
    s=create(device,lr=0,stream=True);slots=torch.tensor([5,1],device=device,dtype=torch.int32)
    raw=torch.randn(2,37,device=device);h=torch.randn(2,12,device=device)
    q,ids=s.propose(raw,h,slots,topk=4);initial=(q.clone(),ids.clone())
    teacher=torch.randn(2,1,74,device=device).softmax(-1)
    path=torch.tensor([[0],[1]],device=device,dtype=torch.int32)
    parents=torch.full((2,1),-1,device=device,dtype=torch.long);ctx=torch.zeros((2,1),device=device,dtype=torch.long)
    s.feedback(teacher,path,slots,parents,ctx)
    actual=s.propose(raw,h,slots,topk=4)
    for a,b in zip(initial,actual):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert not s.B_fast.any() and s.active_count==0


def test_summary_recommends_only_paired_aal_and_throughput_wins(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('summary',ROOT/'scripts/summarize_tlt_opd.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for name,method,aal,tps in [('baseline','tlt',2.,100.),('slow','tlt_opd_reflex',2.5,99.),('fast','tlt_opd_reflex',2.3,102.)]:
        folder=tmp_path/name;folder.mkdir()
        cfg={key:1 for key in module.REQUIRED_CONFIG};cfg.update(batch_size=1,seed=42,profile=False)
        report=dict(method=method,config=cfg,verified_aal=aal,tokens_per_s=tps,opd_fast_lr=.01,opd_update_stream=1,
            engine_config={},artifact_identity={'same':'weights'},prompt_token_sha256='same',measured_prompts=1,generated_responses=1,
            spot_trainer_enabled=False,valid_for_official_comparison=True,opd_orphan_nodes=0,opd_invalid_contexts=0,
            real_eagle3_parity=dict(passed=True),generation_wall_s=100/tps)
        (folder/'report.json').write_text(json.dumps(report));(folder/'responses.jsonl').write_text('{}\n')
    rows=module.summarize(tmp_path)
    assert [r['recommendable'] for r in rows]==[False,True,False]
    for file in ('report.json','summary.csv','responses.jsonl','fastest_observed.env'):assert (tmp_path/file).is_file()
    assert 'export OPD_FAST_LR=0.01' in (tmp_path/'fastest_observed.env').read_text()
