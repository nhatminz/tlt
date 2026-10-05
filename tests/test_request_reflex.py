from pathlib import Path
import ast
import torch
import pytest
from tlt_reflex.state import RequestReflex

DEVICES=['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])


def create(device='cpu',v=19):
    mapping=torch.arange(v,device=device)*2
    state=RequestReflex(7,v,12,mapping,backend='triton' if device!='cpu' else 'torch',max_contexts=4)
    state.reset_slots([5,1,3],allocated=True)
    return state


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('contexts',[1,4])
def test_zero_state_matches_original_full_logits_probs_topk(device,contexts):
    s=create(device)
    raw=torch.randn(3*contexts,19,device=device)
    raw[0,0]=-0.
    hidden=torch.randn(3*contexts,12,device=device)
    ids=torch.tensor([5,1,3],device=device,dtype=torch.int32)
    corrected=s.correct(raw,hidden,ids,root=contexts==1)
    assert torch.equal(raw,corrected)
    assert torch.equal(raw.signbit(),corrected.signbit())
    a,b=raw.softmax(-1).topk(4,-1),corrected.softmax(-1).topk(4,-1)
    assert torch.equal(a.values,b.values) and torch.equal(a.indices,b.indices)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('greedy',[False,True])
@pytest.mark.parametrize('v',[19,519,16000])
def test_root_feedback_matches_dense_specnaacl_oracle_and_verified_indices_stride(device,greedy,v):
    s=create(device,v)
    torch.manual_seed(17)
    ids=torch.tensor([5,1,3],device=device,dtype=torch.int32)
    raw=torch.randn(3,v,device=device); hidden=torch.randn(3,12,device=device)
    q=s.correct(raw,hidden,ids,root=True).softmax(-1)
    s.cache_root(q,ids)
    psi=s.psi[ids.long()].clone()
    path=torch.tensor([[0,1,-1],[4,5,6],[8,-1,-1]],device=device,dtype=torch.int32)
    roots=path[:,0]  # non-contiguous global target rows, actual verifier API
    teacher=(torch.randint(0,v*2,(3,4),device=device) if greedy
             else torch.randn(3,4,v*2,device=device).softmax(-1))
    p=(s.mapping[None].eq(teacher.reshape(-1)[roots.long(),None]).float() if greedy
       else teacher.reshape(-1,v*2)[roots.long()][:,s.mapping])
    p=p/(p.sum(-1,keepdim=True)+s.eps)
    alpha=torch.minimum(p,q).sum(-1)
    m=(q<p).float(); selected=(m*q).sum(-1,keepdim=True)
    g=q*(selected-m)/(alpha[:,None]+s.eps)
    expected=-s.lr*g[:,:,None]*psi[:,None,:]
    s.feedback(teacher,roots,ids,greedy=greedy)
    torch.testing.assert_close(s.a[ids.long()],expected,rtol=5e-5,atol=2e-7)
    assert not bool(s.cached[ids.long()].any())
    assert s.total_updates==3
    # Reorder/compact batch and add 4 contexts: ownership stays with req slot.
    reordered=ids[[2,0]]
    h=torch.randn(8,12,device=device); z=torch.randn(8,v,device=device)
    actual=s.correct(z,h,reordered)
    features=torch.nn.functional.normalize(h@s.projection,dim=-1,eps=1e-6)
    oracle=z+(s.a[reordered.long()].repeat_interleave(4,0)*features[:,None,:]).sum(-1)
    torch.testing.assert_close(actual,oracle,rtol=5e-5,atol=3e-6)


@pytest.mark.parametrize('device',DEVICES)
def test_slot_free_reuse_does_not_leak_and_padded_rows_cannot_overwrite_slot0(device):
    s=create(device)
    s.a[5].fill_(1.)
    s.reset_slots(5,allocated=False)
    assert not s.live[5] and not s.cached[5] and not bool(s.a[5].any())
    s.reset_slots(5,allocated=True)
    assert s.live[5] and not bool(s.a[5].any())
    s.reset_slots(0,allocated=True)
    slots=torch.tensor([0,1,0,0],device=device,dtype=torch.int32)
    raw=torch.randn(4,19,device=device); h=torch.randn(4,12,device=device)
    limit=torch.tensor(2,device=device,dtype=torch.int32)
    q=s.correct(raw,h,slots,root=True,valid_bs=limit).softmax(-1)
    s.cache_root(q,slots,valid_bs=limit)
    assert torch.equal(s.q[0],q[0])
    assert torch.equal(s.q[1],q[1])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA graph replay')
def test_capture_replay_reads_new_slots_state_valid_batch_and_has_no_global_sync(monkeypatch):
    s=create('cuda:0'); s.reset_slots(0,allocated=True)
    raw=torch.randn(4,19,device='cuda'); hidden=torch.randn(4,12,device='cuda')
    slots=torch.tensor([5,1,0,0],device='cuda',dtype=torch.int32)
    limit=torch.tensor(2,device='cuda',dtype=torch.int32)
    # Warm up Triton on a side stream before capture.
    stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            out=s.correct(raw,hidden,slots,root=True,valid_bs=limit)
            q=out.softmax(-1); s.cache_root(q,slots,valid_bs=limit)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out=s.correct(raw,hidden,slots,root=True,valid_bs=limit)
        q=out.softmax(-1); s.cache_root(q,slots,valid_bs=limit)
    def forbidden(*a,**k): pytest.fail('no global synchronize in production plugin')
    monkeypatch.setattr(torch.cuda,'synchronize',forbidden)
    s.a[3].fill_(.1); slots[0]=3; limit.fill_(1)
    graph.replay()
    expected=s.correct(raw,hidden,slots,root=True,valid_bs=limit).softmax(-1)
    torch.testing.assert_close(q,expected,rtol=0,atol=0)
    assert torch.equal(s.q[3],q[0])


def test_production_plugin_no_collectives_models_loss_log_or_host_transfers():
    source=(Path(__file__).resolve().parents[1]/'tlt_reflex/kernels.py').read_text()
    calls=[n.func for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call)]
    assert not any(isinstance(n,ast.Attribute) and n.attr in {'item','cpu','tolist','synchronize','all_reduce','forward','log'} for n in calls)


def test_fp32_state_is_not_changed_by_global_default_dtype():
    previous=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        s=create()
        for name in ('a','q','psi','root_out','deep_out','mass','stats','root_feature','deep_feature'):
            assert getattr(s,name).dtype==torch.float32
    finally:torch.set_default_dtype(previous)
