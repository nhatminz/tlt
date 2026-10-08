"""Native calibration round-trip and counterbalanced physical run order."""
import importlib.util,json,os,subprocess,sys
from pathlib import Path
import pytest,torch
from tlt_reflex.profiles import fingerprint,validate_native_profile
from tlt_reflex.ported.profiles import execution_key,ProposalProfile
ROOT=Path(__file__).resolve().parents[1]


def tuner():
    spec=importlib.util.spec_from_file_location('tlt_tuner_contract',ROOT/'scripts/tune_tlt_opd_proposals.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_default_shapes_are_unique_effective_buckets_and_loadable(tmp_path):
    t=tuner();shapes=t.effective_shapes(t.DEFAULT_SHAPES)
    assert shapes==[(1,1),(2,1),(4,1),(8,1),(16,1),(32,1),(128,1)]
    key=dict(gpu='fixture',compute_capability=[8,6],vocab=519,rank=8,dtype='torch.float32',
        kernel_sha256='kernel',tlt_execution_sha256='execution')
    payload=dict(profile_kind='tlt_native',execution_key=key,
        benchmark_metadata=dict(cuda_graph=True,active_id_pattern='seeded_sorted_randperm'),
        records=[dict(contexts=b*c,trials=[dict(slots=0,sparse=1.,fused=2.,gemm=3.)]) for b,c in shapes])
    path=tmp_path/'profile.json';t.write_validated_profile(path,payload,key)
    assert ProposalProfile(json.loads(path.read_text())).contexts==[1,2,4,8,16,32,128]
    payload['records'].append(payload['records'][0])
    before=path.read_bytes()
    with pytest.raises(ValueError,match='duplicate'):t.write_validated_profile(path,payload,key)
    assert path.read_bytes()==before,'invalid profile must not overwrite an existing file'
    with pytest.raises(ValueError):t.write_validated_profile(tmp_path/'bad.json',payload,key)
    assert not (tmp_path/'bad.json').exists()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA tuner')
def test_default_cuda_tuner_scattered_ids_parity_and_native_roundtrip(tmp_path):
    t=tuner();key=execution_key(fingerprint(),519,8,'fp32',16)
    first=t.benchmark_configuration(key,t.DEFAULT_SHAPES,[0,16,128,519],3,seed=42)
    path=tmp_path/'native.json';t.write_validated_profile(path,first,key)
    selector=validate_native_profile(json.loads(path.read_text()),key)
    assert selector.contexts==[1,2,4,8,16,32,128]
    trials=[trial for record in first['records'] for trial in record['trials']]
    assert all(trial['bitwise_parity'] for trial in trials)
    partial=first['records'][0]['trials'][1]['active_ids_preview']
    assert partial!=list(range(16)) and partial==sorted(partial)
    repeat=t.benchmark_configuration(key,[(1,1)],[0,16,128,519],1,seed=42)
    assert [v['active_ids_sha256'] for v in first['records'][0]['trials']]==[v['active_ids_sha256'] for v in repeat['records'][0]['trials']]
    assert selector.choose(32,128) in ('sparse','fused','gemm')


@pytest.mark.parametrize('seed,expected',[(42,['tlt','tlt_opd_reflex']),(43,['tlt_opd_reflex','tlt']),
                                        (44,['tlt','tlt_opd_reflex']),(45,['tlt_opd_reflex','tlt'])])
def test_actual_pair_shell_runs_in_seed_counterbalanced_order(tmp_path,seed,expected):
    # Execute the real shell orchestration. Engines are stand-ins; this verifies
    # physical launch order without claiming native throughput/model validation.
    harness=tmp_path/'harness';(harness/'scripts').mkdir(parents=True)
    (harness/'benchmark_pair.sh').write_bytes((ROOT/'benchmark_pair.sh').read_bytes())
    for name in ('pair_order.py','check_tlt_opd_pair_config.py'):
        (harness/'scripts'/name).write_bytes((ROOT/'scripts'/name).read_bytes())
    (harness/'scripts/summarize_tlt_opd.py').write_text('pass\n')
    (harness/'run_benchmark.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${CANONICAL_CONFIG_OUTPUT:-}" ]];then
  "$PYTHON_BIN" "$REAL_ROOT/benchmark.py" --method "$METHOD" --model /model --draft /draft --dataset /data \\
    --output "$OUTPUT_DIR/report.json" --seed "$SEED" --dump-canonical-config "$CANONICAL_CONFIG_OUTPUT"
else
  printf '%s:%s\\n' "$METHOD" "$BENCH_RUN_POSITION" >> "$ORDER_LOG"
fi
''')
    env=dict(os.environ,PYTHON_BIN=sys.executable,PYTHONPATH=str(ROOT),REAL_ROOT=str(ROOT),SEED=str(seed),
        PAIR_DIR=str(tmp_path/'results'),ORDER_LOG=str(tmp_path/'order.log'))
    env.pop('DRY_RUN',None)
    subprocess.run(['bash',str(harness/'benchmark_pair.sh')],env=env,capture_output=True,text=True,check=True)
    lines=(tmp_path/'order.log').read_text().splitlines()
    assert lines==[expected[0]+':1',expected[1]+':2']
    assert json.loads((tmp_path/'results/run_order.json').read_text())['methods']==expected
    diff=json.loads((tmp_path/'results/config_diff.json').read_text())
    assert diff['benchmark_critical_fields_identical'] and all(d['path'].startswith('opd.') for d in diff['differences'])
