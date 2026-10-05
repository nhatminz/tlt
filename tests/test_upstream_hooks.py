"""Execute real upstream Python proposal routines with deterministic model stubs.

This exercises the actual patched/pristine control flow, not an independent
reimplementation of TLT. Native SG kernels/real model engine remain GPU-env
integration checks, separately documented.
"""
import ast
import copy
from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from tlt_reflex.state import RequestReflex

ROOT=Path(__file__).resolve().parents[1]
PATCHED=ROOT/'upstream/fastrl/third-party/sglang/python/sglang/srt'
PRISTINE=ROOT/'upstream/pristine_sglang_python/sglang/srt'


def extract(path,name,namespace,cls=None):
    tree=ast.parse(path.read_text())
    nodes=tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body
    node=copy.deepcopy(next(n for n in nodes if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name))
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    return namespace[name]


def test_upstream_algorithm_source_and_bundled_wheel_provenance():
    spec=importlib.util.spec_from_file_location('audit',ROOT/'scripts/audit_upstream.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.audit()
    assert result['protected_hash_files']==126 and result['pristine_python_files']==680


@pytest.mark.skipif(not torch.cuda.is_available(),reason='upstream select_top_k_tokens explicitly uses CUDA')
@pytest.mark.parametrize('mode',['off','empty_reflex','zero_lr_after_feedback'])
def test_real_draft_routines_tree_inputs_and_rng_match_pristine_upstream(mode):
    # Only the native extension fast_topk is substituted; every implementation
    # under comparison uses the same topk function, actual upstream routines.
    class DraftInput(SimpleNamespace):pass
    namespace=dict(torch=torch,fast_topk=torch.topk,EagleDraftInput=DraftInput)
    namespace['select_top_k_tokens']=extract(PRISTINE/'speculative/spec_utils.py','select_top_k_tokens',namespace)
    namespace['organize_draft_results']=extract(PRISTINE/'speculative/eagle_utils.py','organize_draft_results',namespace)
    mapping=torch.arange(19,device='cuda')*2
    state=RequestReflex(7,19,12,mapping,max_contexts=4)
    state.reset_slots([5,1],allocated=True)
    slots=torch.tensor([5,1],device='cuda',dtype=torch.int32)
    teacher=SimpleNamespace(next_token_logits=torch.randn(2,19,device='cuda'),hidden_states=torch.randn(2,12,device='cuda'))
    if mode=='zero_lr_after_feedback':
        state.lr=0.
        q=state.correct(teacher.next_token_logits,teacher.hidden_states,slots,root=True).softmax(-1)
        state.cache_root(q,slots)
        state.feedback(torch.randn(2,4,38,device='cuda').softmax(-1),
                       torch.tensor([0,4],device='cuda',dtype=torch.int32),slots)
    outputs=[]; cache_moves=[]
    for folder,plugin in [(PRISTINE,None),(PATCHED,None if mode=='off' else state)]:
        forward_count=[]; moves=[]
        def forward(batch,**kwargs):
            forward_count.append(batch.input_ids.clone())
            # Deterministic stand-in for draft NN only; no target forward.
            h=batch.spec_info.hidden_states+batch.input_ids[:,None].float()*.001
            z=h[:,:1]+torch.arange(19,device='cuda')[None]*.02
            return SimpleNamespace(next_token_logits=z,hidden_states=h),None
        worker=SimpleNamespace(_tlt_reflex=plugin,hot_token_id=mapping,topk=4,max_topk=4,
            speculative_num_steps=4,speculative_num_draft_tokens=12,
            server_args=SimpleNamespace(speculative_algorithm='EAGLE3'),
            draft_attn_backend=SimpleNamespace(attn_backends=[None]*4),
            draft_model_runner=SimpleNamespace(forward=forward),
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda:SimpleNamespace(move_kv_cache=lambda a,b:moves.append((a.clone(),b.clone())))),
            _detect_nan_if_needed=lambda x:None)
        capture=extract(folder/'speculative/eagle_worker.py','capture_for_decode',dict(namespace),cls='EAGLEWorker')
        draft=DraftInput()
        before=torch.cuda.get_rng_state().clone()
        if folder==PRISTINE:capture(worker,teacher,draft)
        else:capture(worker,teacher,draft,slots)
        batch=SimpleNamespace(spec_info=draft,batch_size=2,req_pool_indices=slots,
            out_cache_loc=torch.arange(32,device='cuda'),positions=torch.zeros(8,device='cuda',dtype=torch.long))
        routine=extract(folder/'speculative/eagle_worker.py','draft_forward',dict(namespace),cls='EAGLEWorker')
        tree_inputs=routine(worker,batch)
        assert torch.equal(before,torch.cuda.get_rng_state()),'proposal must not consume sampling RNG'
        assert len(forward_count)==3,'no extra draft/target forward'
        outputs.append(tuple(t.clone() for t in tree_inputs));cache_moves.append(moves)
    for a,b in zip(*outputs):assert torch.equal(a,b),'tree parents/selected indices/tokens differ'
    for a,b in zip(*cache_moves):
        for x,y in zip(a,b):assert torch.equal(x,y),'KV cache movement differs'


def test_real_allocator_lifecycle_resets_request_state():
    adapter=SimpleNamespace(create=lambda **kwargs:SimpleNamespace(region=lambda *args:nullcontext()))
    pool_class=extract(PATCHED/'mem_cache/memory_pool.py','ReqToTokenPool',
                       dict(torch=torch,TorchMemorySaverAdapter=adapter,GPU_MEMORY_TYPE_KV_CACHE=0))
    pool=pool_class(7,16,'cpu',False)
    state=RequestReflex(7,19,12,torch.arange(19),backend='torch')
    pool._tlt_reflex=state
    ids=pool.alloc(3)
    assert ids==[0,1,2] and state.live[ids].all()
    state.a[1].fill_(2);pool.free(1)
    assert not state.live[1] and not state.a[1].any()
    pool.free([0,2]);pool.clear()
    assert not state.live.any() and not state.a.any()
    assert pool.alloc(7)==list(range(7))


def test_no_added_sync_or_random_sampling_in_verification_and_graph_hooks():
    for name in ['eagle_worker.py','eagle_info.py','eagle_draft_cuda_graph_runner.py','eagle_draft_extend_cuda_graph_runner.py']:
        old=(PRISTINE/'speculative'/name).read_text()
        new=(PATCHED/'speculative'/name).read_text()
        for token in ['torch.cuda.synchronize(', 'torch.rand(', 'torch.rand_like(', 'tree_speculative_sampling_target_only(']:
            assert new.count(token)==old.count(token)
    verifier=(PATCHED/'speculative/eagle_info.py').read_text()
    assert 'reflex.feedback(teacher, accept_index[:, 0], batch.req_pool_indices' in verifier
    extend=(PATCHED/'speculative/eagle_draft_extend_cuda_graph_runner.py').read_text()
    assert 'root=True' in extend and 'cache_root' in extend and '_tlt_valid_bs.fill_(raw_bs)' in extend
    normal=(PATCHED/'speculative/eagle_draft_cuda_graph_runner.py').read_text()
    assert normal.index('forward_batch = ForwardBatch(')<normal.index('forward_batch._tlt_valid_bs')


@pytest.mark.parametrize('family,cls',[('qwen2','Qwen2ForCausalLM'),('qwen3','Qwen3ForCausalLM'),('llama','LlamaForCausalLM')])
def test_export_feature_layer_ids_use_real_upstream_api_no_double_shift(family,cls):
    from tlt_reflex.checkpoint import validate_config
    dc=dict(architectures=['LlamaForCausalLMEagle3'],num_hidden_layers=1,hidden_size=8,
        vocab_size=19,tie_word_embeddings=False,eagle_config={'eagle_aux_hidden_state_layer_ids':[1,3,6]})
    tc=dict(model_type=family,vocab_size=19,hidden_size=8,num_hidden_layers=10)
    exported=validate_config(dc,tc)
    fake=SimpleNamespace(pp_group=SimpleNamespace(is_last_rank=True),model=SimpleNamespace())
    hook=extract(PATCHED/f'models/{family}.py','set_eagle3_layers_to_capture',{},cls=cls)
    hook(fake,exported['eagle_config']['eagle_aux_hidden_state_layer_ids'])
    assert fake.model.layers_to_capture==[2,4,7]


def test_only_exact_pinned_packages_in_install_lock():
    lines=[s for s in (ROOT/'requirements.txt').read_text().splitlines() if s and not s.startswith('#')]
    assert len(lines)==206
    for line in lines:
        assert '==' in line and '>=' not in line
        assert line.split('==')[0].lower() not in {'asyncio','pathlib','datetime','statistics'}
