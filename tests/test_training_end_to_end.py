import torch
def assert_nested_equal(a,b,*,projector_moments=False,path=()):
    if torch.is_tensor(a):
        if projector_moments and path==('opd_projector',):
            # Newly executed atomic updates can change A by one FP32 ULP.
            torch.testing.assert_close(a,b,rtol=2*torch.finfo(a.dtype).eps,atol=0)
        elif projector_moments and path[:2]==('state',0) and path[-1] in ('exp_avg','exp_avg_sq'):
            # GPU atomic accumulation order is not deterministic. Only A's
            # newly computed moments admit roundoff. Restoration itself is
            # checked bit for bit in the subprocess before the next rollout.
            # Cancellation makes elementwise relative error unsuitable near
            # zero. Bound atomic roundoff by eight FP32 eps at tensor scale.
            atol=8*torch.finfo(a.dtype).eps*max(a.abs().max().item(),b.abs().max().item())
            torch.testing.assert_close(a,b,rtol=0,atol=atol)
        else:assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_nested_equal(a[k],b[k],projector_moments=projector_moments,path=path+(k,))
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_nested_equal(x,y)
    else:assert a==b
