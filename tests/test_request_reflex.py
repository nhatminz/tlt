"""OPD math, lifecycle, cache and real Triton/CUDA graph contracts."""
from pathlib import Path
from types import SimpleNamespace as NS
import importlib.util
import sys
import torch
import pytest
from tlt_reflex.state import OPDState
from tlt_reflex.ported.reference import initialize_projector,select_states_reference,union_reference
DEVICES=['cpu']+(['cuda'] if torch.cuda.is_available() else [])


def create(device='cpu',v=37,lr=.01,stream=False,mode='auto',dtype=torch.float32):
    torch.manual_seed(17)
    head=torch.nn.Linear(12,v,bias=False,device=device,dtype=dtype)
    state=OPDState(7,head,torch.arange(v,device=device)*2,fast_lr=lr,backend='torch' if device=='cpu' else 'triton',
        max_contexts=13,max_topk=4,max_nodes=12,max_path=5,update_stream=stream,proposal_mode=mode)
    state.reset_slots([5,1,3],allocated=True)
    return state


def tree(device='cpu'):
    # 0->1->3->5 accepted; rejected 2->4->6. Only 2 is one-hop frontier.
    parents=torch.tensor([[-1,0,0,1,2,3,4]]*2,device=device)
    contexts=torch.tensor([[0,1,2,3,4,-1,-1]]*2,device=device)
    path=torch.tensor([[0,1,3,5,-1],[7,8,10,12,-1]],device=device,dtype=torch.int32)
    return parents,contexts,path


@pytest.mark.parametrize('device',DEVICES)
def test_shared_b_and_slot_cache_reorder_free_reuse_epoch(device):
    s=create(device);slots=torch.tensor([5,1],device=device,dtype=torch.int32)
    h=torch.randn(2,12,device=device);raw=torch.nn.functional.linear(h,s.head.weight)
    s.propose(raw,h,slots,topk=4)
    saved=s.head_cache[5,0].clone();s.B_fast.fill_(.1)
    reordered=slots.flip(0);s.propose(raw,h,reordered,topk=4)
    torch.testing.assert_close(s.head_cache[5,0],h[1])
    s.reset_slots(5,allocated=False)
    assert not s.valid_cache[5].any() and not s.head_cache[5].any()
    assert torch.all(s.B_fast==.1),'finishing one request must preserve shared B'
    s.reset_slots(5,allocated=True);assert not s.valid_cache[5].any()
    s.reset_slots([5,1,3],allocated=False);assert torch.all(s.B_fast==.1)
    s.reset_slots(5,allocated=True);assert not s.B_fast.any() and s.epoch==2


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('mode',['auto','sparse','fused','gemm'])
def test_cold_distribution_unique_prefix_and_deterministic_ties(device,mode):
    s=create(device,mode=mode);slots=torch.tensor([5,1],device=device,dtype=torch.int32)
    h=torch.randn(2,12,device=device);raw=torch.randn(2,37,device=device)
    raw_before=raw.clone();q,ids=s.propose(raw,h,slots,topk=4)
    native=raw.softmax(-1).topk(4,-1)
    assert torch.equal(ids,native.indices)
    torch.testing.assert_close(q,native.values,rtol=3e-6,atol=1e-7)
    assert torch.equal(raw,raw_before) and not s.B_fast.any()
    s.propose(torch.zeros_like(raw),h,slots,topk=4)
    assert torch.equal(s.ids_cache[5,0],torch.arange(16,device=device))


def test_selection_frontier_expanded_only_union_tail_and_positive_sentinels():
    p,c,path=tree();local=torch.where(path>=0,path-torch.arange(2)[:,None]*7,-1)
    w,kind=select_states_reference(NS(parents=p,feedback_contexts=c),NS(packed_indices=local))
    assert kind[0].tolist()==[1,1,2,1,0,0,0]
    teacher=torch.tensor([[.2,0,.3,.1,.4]]);draft=torch.tensor([[.1,.2,.3,.15,.25]])
    ids,valid,pu,qu,pt,qt,kl=union_reference(teacher,draft,torch.tensor([[4,2,-1]]),torch.tensor([[0,1,2]]))
    assert valid.tolist()==[[True,True,True,True,False,False]]
    torch.testing.assert_close(pt,torch.tensor([.1]));torch.testing.assert_close(qt,torch.tensor([.15]))
    assert pu[0,1]==0 and valid[0,1],'zero-p draft coordinate remains in union'


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('stream',[False,True])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_feedback_math_target_only_coordinates_and_compact_mass(device,stream,dtype):
    if device=='cpu' and dtype!=torch.float32:pytest.skip('BF16 reconstruction tested on CUDA')
    s=create(device,stream=stream,dtype=dtype);oracle=create('cpu',dtype=torch.float32)
    oracle.head=NS(weight=s.head.weight.cpu(),bias=None);oracle.projector.copy_(s.projector.cpu())
    slots=torch.tensor([5,1],device=device,dtype=torch.int32);hs=[]
    for offset,c in [(0,1),(1,4)]:
        h=torch.randn(2*c,12,device=device,dtype=dtype);raw=torch.nn.functional.linear(h,s.head.weight).float()
        ids=None if offset==0 else torch.tensor([[0,1,4,7],[0,1,4,7]],device=device)
        s.propose(raw,h,slots,topk=4,offset=offset,expanded_ids=ids)
        oracle.propose(raw.cpu(),h.cpu(),slots.cpu(),topk=4,offset=offset,expanded_ids=None if ids is None else ids.cpu())
    p,c,path=tree(device)
    target=torch.randn(2,7,74,device=device).softmax(-1)
    # Zero compact mass and nonfinite selected rows are skipped.
    target[0,1,s.mapping]=0;target[1,2,s.mapping]=float('nan')
    snapshot=target.clone();rng=torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state()
    s.feedback(target,path,slots,p,c);s.wait_for_update()
    oracle.feedback(target.cpu(),path.cpu(),slots.cpu(),p.cpu(),c.cpu())
    torch.testing.assert_close(s.B_fast.cpu(),oracle.B_fast,rtol=.015 if dtype==torch.bfloat16 else 1e-4,atol=3e-5 if dtype==torch.bfloat16 else 3e-7)
    assert torch.equal(rng,torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state())
    torch.testing.assert_close(target,snapshot,equal_nan=True,rtol=0,atol=0)
    assert s.counters[8]==2
    assert s.active_count>0


@pytest.mark.parametrize('device',DEVICES)
def test_actual_expansion_context_mapping_matches_native_parent_table(device):
    s=create(device);slots=torch.tensor([5,1],device=device,dtype=torch.int32)
    h=torch.randn(2,12,device=device);s.propose(h@s.head.weight.T,h,slots,topk=4)
    for offset,expanded in [(1,[0,1,2,3]),(5,[4,6,8,12]),(9,[20,24,30,34])]:
        h=torch.randn(8,12,device=device)
        s.propose(h@s.head.weight.T,h,slots,topk=4,offset=offset,expanded_ids=torch.tensor([expanded]*2,device=device))
    selected=torch.tensor([[0,1,4,6,20,21]]*2,device=device)
    parent_list=torch.tensor([[-1,0,1,2,3,4,6,8,12]]*2,device=device)
    parents,contexts=s.tree_metadata(selected,parent_list,slots,4,4)
    assert parents[0].tolist()==[-1,0,0,1,1,3,3]
    assert contexts[0].tolist()==[0,1,2,5,6,9,-1]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA graph')
def test_graph_replay_adaptive_branches_padding_reorder_and_no_host_sync(monkeypatch):
    s=create('cuda');s.reset_slots(0,allocated=True)
    h=torch.randn(4,12,device='cuda');raw=torch.randn(4,37,device='cuda')
    slots=torch.tensor([5,1,0,0],device='cuda',dtype=torch.int32);limit=torch.tensor(2,device='cuda',dtype=torch.int32)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):s.propose(raw,h,slots,topk=4,valid_bs=limit)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):q,ids=s.propose(raw,h,slots,topk=4,valid_bs=limit)
    forbidden=lambda *a,**k:pytest.fail('global CUDA synchronize in plugin')
    monkeypatch.setattr(torch.cuda,'synchronize',forbidden)
    s.B_fast.fill_(.05);s.active_count.fill_(37);slots[0]=3;limit.fill_(1)
    graph.replay();captured=(q.clone(),ids.clone());eager=s.propose(raw,h,slots,topk=4,valid_bs=limit)
    for a,b in zip(captured,eager):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert s.valid_cache[3,0] and not s.valid_cache[0,0]
    # Move back to sparse with zero active rows; captured graph must still dispatch.
    s.B_fast.zero_();s.active_count.zero_();graph.replay()
    assert s.dispatch_mode==0


def test_fp32_state_and_deterministic_projector_no_rng_draw():
    before=torch.get_rng_state();head=torch.arange(37*12).view(37,12).float()
    a=initialize_projector(12,8,head=head)
    assert torch.equal(before,torch.get_rng_state())
    torch.testing.assert_close(a.T@a,torch.eye(8),atol=1e-6,rtol=1e-6)
    old=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16);s=create()
        for name in ('B_fast','projector','u_cache','q_cache','norm_cache'):assert getattr(s,name).dtype==torch.float32
    finally:torch.set_default_dtype(old)


def test_reference_functions_are_identical_to_read_only_specnaacl():
    source=Path(__file__).resolve().parents[2]/'SpecNaacl/helper/opd_reflex.py'
    import ast
    tree_ast=ast.parse(source.read_text());ns=dict(torch=torch,math=__import__('math'))
    funcs=[n for n in tree_ast.body if isinstance(n,ast.FunctionDef) and n.name in ('initialize_projector','select_states_reference','union_reference')]
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(source),'exec'),ns)
    h=torch.randn(37,12);assert torch.equal(initialize_projector(12,8,head=h),ns['initialize_projector'](12,8,head=h))
    p,c,global_path=tree();path=NS(packed_indices=torch.where(global_path>=0,global_path-torch.arange(2)[:,None]*7,-1));t=NS(parents=p,feedback_contexts=c)
    for x,y in zip(select_states_reference(t,path),ns['select_states_reference'](t,path)):assert torch.equal(x,y)


@pytest.mark.parametrize('device',DEVICES)
def test_root_refresh_handles_shared_b_change_between_scheduler_batches(device):
    s=create(device);slots=torch.tensor([5,1],device=device,dtype=torch.int32)
    head=torch.randn(2,12,device=device);raw=head@s.head.weight.T
    s.propose(raw,head,slots,topk=4)
    s.B_fast[0]=.2*(head[0]@s.projector);s.active_count.fill_(1);s.active_ids[0]=0;s.bitmap[0]=1
    draft=NS();s.refresh_root(draft,slots,4)
    expected=(raw.float()+(head.float()@s.projector)@s.B_fast.T).softmax(-1).topk(4)
    assert torch.equal(draft.topk_index,expected.indices)
    torch.testing.assert_close(draft.topk_p,expected.values,atol=1e-7,rtol=4e-6)


def test_two_feedback_rounds_equal_original_specnaacl_cpu_math():
    import ast
    source=Path(__file__).resolve().parents[2]/'SpecNaacl/helper/opd_reflex.py'
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='OPDReflex')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_feedback_reference')
    ns=dict(torch=torch,select_states_reference=select_states_reference,union_reference=union_reference)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
    s=create();oracle=create();slots=torch.tensor([5,1],dtype=torch.int32)
    p,c,path=tree();local=NS(packed_indices=torch.where(path>=0,path-torch.arange(2)[:,None]*7,-1));t=NS(parents=p,feedback_contexts=c)
    for round in range(2):
        for offset,count in [(0,1),(1,4)]:
            h=torch.randn(2*count,12);raw=h@s.head.weight.T
            for state in (s,oracle):state.propose(raw,h,slots,topk=4,offset=offset)
        target=torch.randn(2,7,74).softmax(-1)
        s.feedback(target,path,slots,p,c)
        oracle.feedback_row_map=slots.long()
        ns['_feedback_reference'](oracle,t,local,target,False)
        torch.testing.assert_close(s.B_fast,oracle.B_fast,rtol=1e-5,atol=1e-7)
        torch.testing.assert_close(s.counters[:13],oracle.counters[:13],rtol=1e-5,atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='original SpecNaacl Triton parity')
@pytest.mark.parametrize('v',[37,519,1025])
def test_proposals_bitwise_equal_original_specnaacl_triton(v):
    from tlt_reflex import kernels
    source=Path(__file__).resolve().parents[2]/'SpecNaacl/helper/opd_reflex_kernels.py'
    original=source.read_text().replace('from helper.tree_kernels import _proposal_merge','from tlt_reflex.ported.merge import _proposal_merge')
    # Triton requires source-backed functions, use temporary copy OUTSIDE SpecNaacl.
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        file=Path(directory)/'original_spec_opd.py';file.write_text(original)
        spec=importlib.util.spec_from_file_location('original_spec_opd',file);module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module;spec.loader.exec_module(module)
        s=create('cuda',v=v,mode='sparse')
        s.score_workspace=torch.empty(s.max_proposal_rows*v,device='cuda')
        raw=torch.randn(2,4,v,device='cuda');u=torch.randn(2,4,8,device='cuda')
        s.prepare_proposal_workspace=lambda *a:None
        for count in (0,1,v//4,v):
            s.B_fast.zero_();s.B_fast[:count]=torch.randn(count,8,device='cuda')*.05
            s.active_count.fill_(count);s.active_ids[:count]=torch.arange(count,device='cuda')
            bits=torch.zeros((v+31)//32,dtype=torch.int32)
            for token in range(count):bits[token//32]|=torch.tensor(1<<(token%32),dtype=torch.int64).to(torch.int32)
            s.bitmap.copy_(bits)
            for mode in ('sparse','fused','gemm'):
                s.proposal_mode=mode
                actual=tuple(t.clone() for t in kernels.propose(s,raw,u))
                outputs=(s.proposal_q[:8*16].view(2,4,16),s.proposal_ids[:8*16].view(2,4,16),s.proposal_norm[:16].view(2,4,2))
                s.selected_proposal_backend=lambda *a:'sparse' if mode=='sparse' else 'dense'
                s.selected_dense_implementation=lambda *a:mode
                module.propose(raw,u,s,16,s.proposal_tiles,outputs,True,False)
                for a,b in zip(actual,outputs):assert torch.equal(a,b),(v,count,mode)
        sys.modules.pop(spec.name,None)
