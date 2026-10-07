"""Execute real pinned upstream control flow using bounded NN/kernel fixtures."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import nullcontext
import torch
import pytest
from test_request_reflex import create
ROOT=Path(__file__).resolve().parents[1]
PATCHED=ROOT/'upstream/fastrl/third-party/sglang/python/sglang/srt'
PRISTINE=ROOT/'upstream/pristine_sglang_python/sglang/srt'


def extract(path,name,namespace,cls=None):
    tree=ast.parse(path.read_text());nodes=tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body
    node=copy.deepcopy(next(n for n in nodes if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name))
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace);return namespace[name]


def test_upstream_algorithm_source_and_bundled_wheel_provenance():
    spec=importlib.util.spec_from_file_location('audit',ROOT/'scripts/audit_upstream.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert m.audit(require_git=True)['protected_hash_files']==126


@pytest.mark.skipif(not torch.cuda.is_available(),reason='native upstream beam selection uses CUDA')
@pytest.mark.parametrize('mode',['off','zero_lr','active'])
def test_actual_root_and_draft_forward_preserve_upstream_path_rng_forward_counts(mode):
    class DraftInput(NS):pass
    ns=dict(torch=torch,fast_topk=torch.topk,EagleDraftInput=DraftInput)
    ns['select_top_k_tokens']=extract(PRISTINE/'speculative/spec_utils.py','select_top_k_tokens',ns)
    ns['organize_draft_results']=extract(PRISTINE/'speculative/eagle_utils.py','organize_draft_results',ns)
    state=create('cuda',lr=0 if mode=='zero_lr' else .01);slots=torch.tensor([5,1],device='cuda',dtype=torch.int32)
    h=torch.randn(2,12,device='cuda');raw=h@state.head.weight.T
    if mode=='active':
        state.B_fast[0]=10;state.active_ids[0]=0;state.active_count.fill_(1);state.bitmap[0]=1
    outputs=[]
    for folder,plugin in [(PRISTINE,None),(PATCHED,None if mode=='off' else state)]:
        calls=[]
        def forward(batch,**kwargs):
            calls.append(1);head=batch.spec_info.hidden_states+batch.input_ids[:,None]*.001
            tokens=torch.arange(37,device='cuda')[None]
            z=torch.sin(tokens*.73+batch.input_ids[:,None]*.117)+tokens*.035
            return NS(next_token_logits=z,hidden_states=head,opd_head_input=head),None
        worker=NS(_tlt_reflex=plugin,hot_token_id=state.mapping,topk=4,max_topk=4,speculative_num_steps=4,speculative_num_draft_tokens=12,
            server_args=NS(speculative_algorithm='EAGLE3'),draft_attn_backend=NS(attn_backends=[None]*4),draft_model_runner=NS(forward=forward),
            token_to_kv_pool_allocator=NS(get_kvcache=lambda:NS(move_kv_cache=lambda a,b:None)),_detect_nan_if_needed=lambda x:None)
        draft=DraftInput();logits=NS(next_token_logits=raw,hidden_states=h,opd_head_input=h)
        capture=extract(folder/'speculative/eagle_worker.py','capture_for_decode',dict(ns),cls='EAGLEWorker')
        before=torch.cuda.get_rng_state()
        if folder==PRISTINE:capture(worker,logits,draft)
        else:capture(worker,logits,draft,slots)
        batch=NS(spec_info=draft,batch_size=2,req_pool_indices=slots,out_cache_loc=torch.arange(32,device='cuda'),positions=torch.zeros(8,device='cuda',dtype=torch.long))
        routine=extract(folder/'speculative/eagle_worker.py','draft_forward',dict(ns),cls='EAGLEWorker')
        outputs.append(tuple(t.clone() for t in routine(worker,batch)))
        assert len(calls)==3 and torch.equal(before,torch.cuda.get_rng_state())
    if mode!='active':
        for a,b in zip(*outputs):torch.testing.assert_close(a,b,rtol=0,atol=0)
    else:assert not torch.equal(outputs[0][-1],outputs[1][-1])
    if mode!='off':assert state.valid_cache[5,:13].all(),'each forwarded beam must have a head distribution cache'


def test_real_allocator_keeps_shared_b_until_next_empty_pool_epoch():
    adapter=NS(create=lambda **kwargs:NS(region=lambda *a:nullcontext()))
    pool_class=extract(PATCHED/'mem_cache/memory_pool.py','ReqToTokenPool',dict(torch=torch,TorchMemorySaverAdapter=adapter,GPU_MEMORY_TYPE_KV_CACHE=0))
    pool=pool_class(7,16,'cpu',False);state=create();state.clear();pool._tlt_reflex=state
    assert pool.alloc(3)==[0,1,2];state.B_fast.fill_(.2);state.valid_cache[1]=True
    pool.free(1);assert not state.valid_cache[1].any() and torch.all(state.B_fast==.2)
    pool.free([0,2]);pool.alloc(1);assert not state.B_fast.any()
    pool.clear();assert not state.live.any()


def test_no_added_target_forward_rng_or_global_synchronization():
    for name in ['eagle_worker.py','eagle_info.py','eagle_draft_cuda_graph_runner.py','eagle_draft_extend_cuda_graph_runner.py']:
        old=(PRISTINE/'speculative'/name).read_text();new=(PATCHED/'speculative'/name).read_text()
        for token in ('torch.rand(','torch.rand_like(','torch.cuda.synchronize(','target_worker.forward_batch_generation('):
            assert old.count(token)==new.count(token),(name,token)
    for path in (ROOT/'tlt_reflex/kernels.py',ROOT/'tlt_reflex/ported/opd_reflex_kernels.py'):
        source=path.read_text()
        for token in ('.item(','.cpu(','.tolist(','torch.cuda.synchronize('):assert token not in source


def test_tlt_factory_returns_before_importing_state_or_kernels(monkeypatch):
    from tlt_reflex.integration import make_reflex,make_meter
    monkeypatch.setenv('TLT_REFLEX_METHOD','tlt');monkeypatch.delenv('TLT_TRACE',raising=False)
    assert make_reflex(None,None) is None and make_meter() is None


def test_exact_head_input_is_pruned_normalized_operand_not_auxiliary_tensor():
    source=(PATCHED/'layers/logits_processor.py').read_text()
    assert 'opd_head_input = (pruned_states[sample_indices]' in source
    assert source.count('opd_head_input=opd_head_input')==2
    model=(PRISTINE/'models/llama_eagle3.py').read_text()
    assert 'return hidden_states_to_logits, [hidden_states_to_aux]' in model


@pytest.mark.parametrize('mode',['decode','extend'])
def test_execute_logits_processor_exposes_exact_head_operand_preserves_aux(mode):
    class ForwardBatch:pass
    ns=dict(torch=torch,ForwardBatch=ForwardBatch,LogitsProcessorOutput=NS,
        get_global_server_args=lambda:NS(multi_item_scoring_delimiter=None))
    fn=extract(PATCHED/'layers/logits_processor.py','forward',ns,cls='LogitsProcessor')
    # These are normalized head states and distinct auxiliary recurrent states.
    head_inputs=torch.randn(5,12);aux=head_inputs*9+1;head=torch.nn.Linear(12,37,bias=False)
    operands=[]
    def logits(x,*args):operands.append(x);return x@head.weight.T
    processor=NS(expose_opd_head_input=True,debug_tensor_dump_output_folder=None,_get_logits=logits)
    metadata=NS(forward_mode=NS(is_decode_or_idle=lambda:mode=='decode',is_target_verify=lambda:False,
        is_draft_extend_v2=lambda:False,is_extend=lambda:mode=='extend',is_split_prefill=lambda:False),
        extend_return_logprob=False,padded_static_len=-1,extend_seq_lens=torch.tensor([2,3]),
        capture_hidden_mode=NS(need_capture=lambda:True,is_full=lambda:False,is_last=lambda:True))
    result=fn(processor,None,head_inputs,head,metadata,[aux])
    expected=head_inputs if mode=='decode' else head_inputs[[1,4]]
    assert result.opd_head_input.data_ptr()==operands[0].data_ptr()
    torch.testing.assert_close(result.opd_head_input,expected,rtol=0,atol=0)
    torch.testing.assert_close(result.hidden_states,aux if mode=='decode' else aux[[1,4]],rtol=0,atol=0)
    assert not torch.equal(result.hidden_states,result.opd_head_input)
