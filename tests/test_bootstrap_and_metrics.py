import importlib.util
import os
from pathlib import Path
import subprocess
import torch
from tlt_reflex.state import RequestReflex
from tlt_reflex.telemetry import Meter

ROOT=Path(__file__).resolve().parents[1]


def test_bootstrap_refuses_wrong_commit_without_resetting_existing_tree(tmp_path):
    tree=tmp_path/'fastrl';tree.mkdir()
    subprocess.run(['git','init','-q',str(tree)],check=True)
    subprocess.run(['git','-C',str(tree),'-c','user.name=Test','-c','user.email=test@localhost',
                    'commit','--allow-empty','-qm','wrong upstream'],check=True)
    head=subprocess.check_output(['git','-C',str(tree),'rev-parse','HEAD'],text=True)
    env=dict(os.environ,UPSTREAM_ROOT=str(tmp_path))
    result=subprocess.run(['bash',str(ROOT/'scripts/bootstrap_upstream.sh')],env=env,capture_output=True,text=True)
    assert result.returncode!=0 and 'upstream HEAD must be' in result.stderr
    assert subprocess.check_output(['git','-C',str(tree),'rev-parse','HEAD'],text=True)==head


def test_memory_metrics_are_host_metadata_and_profiling_off_allocates_no_events(monkeypatch):
    meter=Meter(False)
    state=RequestReflex(7,19,12,torch.arange(19),backend='torch',meter=meter)
    assert meter.reflex_state_memory_mb==7*19*8*4/1e6
    assert meter.reflex_buffer_memory_mb>meter.reflex_state_memory_mb
    def forbidden(*args,**kwargs):raise AssertionError('disabled profiling must not create events/sync')
    monkeypatch.setattr(torch.cuda,'Event',forbidden)
    monkeypatch.setattr(torch.cuda,'synchronize',forbidden)
    with meter.section('proposal_ms'):pass
    meter.verified([1,2],11)
    assert not meter.pending
    assert meter.counters==dict(sequence_verification_rounds=2,accepted_draft_tokens=3,proposed_draft_tokens=22)


def test_effective_config_guard_rejects_spot_trainer_without_importing_gpu_stack():
    from types import SimpleNamespace as NS
    import pytest
    spec=importlib.util.spec_from_file_location('tlt_test_rl',ROOT/'rl.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(ValueError,match='fixed pretrained EAGLE3'):
        module.validate_spot_trainer(NS(speculative=NS(train=NS(enable_drafter_training=True))))
    module.validate_spot_trainer(NS(speculative=NS(train=NS(enable_drafter_training=False))))


def test_v2_inherits_factory_guard_and_unsupported_overlap_is_not_silent(monkeypatch):
    import ast
    import pytest
    from types import SimpleNamespace as NS
    from tlt_reflex.integration import make_reflex
    source=ROOT/'upstream/fastrl/third-party/sglang/python/sglang/srt/speculative/eagle_worker_v2.py'
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='EAGLEWorkerV2')
    assert any(isinstance(n,ast.Name) and n.id=='EAGLEWorker' for n in node.bases)
    init=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='super' for n in ast.walk(init))
    monkeypatch.setenv('TLT_REFLEX_METHOD','tlt_reflex')
    worker=NS(speculative_algorithm=NS(is_eagle3=lambda:True),server_args=NS(disable_overlap_schedule=False))
    with pytest.raises(ValueError,match='EAGLEWorkerV2'):make_reflex(worker,torch.empty(1))


def test_small_bench_batches_reserve_all_upstream_beg_graph_buckets_without_extra_requests():
    from benchmark import engine_request_capacity
    from types import SimpleNamespace as NS
    from test_upstream_hooks import extract,PATCHED
    original=extract(PATCHED/'model_executor/cuda_graph_runner.py','get_batch_sizes_to_capture',
                     dict(require_gathered_buffer=lambda args:False))
    for batch in (1,2,4,8,16,32):
        args=NS(batch_size=batch,responses=1,mab='BEG',mab_configs='8_4_32,8_4_16,8_4_8',mab_buckets='1,2,5,21')
        capacity=engine_request_capacity(args)
        cfg=NS(cuda_graph_bs=list(range(1,capacity+1)),enable_two_batch_overlap=False,enable_torch_compile=False)
        runner=NS(server_args=cfg,req_to_token_pool=NS(size=capacity))
        for lo,hi in [(1,1),(2,4),(5,20),(21,float('inf'))]:
            capture,_=original(runner,lo,hi)
            assert capture,'upstream max(empty graph capture) would fail'
        assert args.batch_size*args.responses==batch,'capacity must not generate extra samples'
    args.mab_configs=''
    assert engine_request_capacity(args)==batch
