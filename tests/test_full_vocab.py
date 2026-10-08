import torch
import pytest
from types import SimpleNamespace
@pytest.mark.skipif(not torch.cuda.is_available(),reason='Triton full-vocabulary tests')
@pytest.mark.parametrize('slots',[0,16,1024])
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16])
def test_full_target_vocab_sparse_fused_gemm_and_zero_B(slots,dtype):
    from helper.opd_reflex import OPDReflex
    from test_opd_reflex import seed
    v,h,r=151936,32,8
    head=torch.nn.Linear(h,v,bias=False,device='cuda',dtype=dtype)
    model=SimpleNamespace(lm_head=head,opd_projector=torch.randn(h,r,device='cuda')*.01)
    state=OPDReflex(rank=r);ids=torch.arange(v,device='cuda')
    state.start(model,1,ids,h,max_contexts=2,max_nodes=2,max_path=2,max_proposal_contexts=2)
    active=torch.arange(slots,device='cuda')*31
    seed(state,active,torch.randn(slots,r,device='cuda')*.01)
    hidden=torch.randn(1,2,h,device='cuda',dtype=dtype)
    raw=head(hidden).detach()
    results=[]
    for backend in ('sparse','fused','gemm'):
        state.proposal_mode='sparse' if backend=='sparse' else 'dense'
        state.dense_implementation='gemm' if backend=='gemm' else 'fused'
        values,tokens,_=state.propose(raw,hidden,16,ids)
        results.append((values.clone(),tokens.clone()))
    for a,b in zip(results,results[1:]):
        assert torch.equal(a[0],b[0]);assert torch.equal(a[1],b[1])
    corrected=raw.float()+state.u_cache[:1,:2]@state.B_fast.t()
    if slots==0:assert torch.equal(corrected,raw.float())
    expected=torch.argsort(corrected,descending=True,stable=True)[...,:16]
    assert torch.equal(results[0][1],expected)
    torch.testing.assert_close(results[0][0],corrected.softmax(-1).gather(-1,expected),rtol=3e-5,atol=1e-7)
