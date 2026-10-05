"""Slot-indexed SpecNaacl LK projection/correction/update, CUDA-graph safe."""
import triton
import triton.language as tl


@triton.jit
def tlt_reflex_feature(H,R,OUT,HS0,HS1,HIDDEN:tl.constexpr,D:tl.constexpr,BH:tl.constexpr,BD:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    h,d=tl.arange(0,BH),tl.arange(0,BD)
    x=tl.load(H+row*HS0+h*HS1,h<HIDDEN,other=0).to(tl.float32)
    r=tl.load(R+h[:,None]*D+d[None,:],(h[:,None]<HIDDEN)&(d[None,:]<D),other=0)
    projected=tl.sum(x[:,None]*r,axis=0)
    norm=tl.maximum(tl.sqrt(tl.sum(projected*projected,axis=0)),1e-6)
    tl.store(OUT+row*D+d,tl.div_rn(projected,norm),d<D)


@triton.jit
def tlt_reflex_correct(Z,PSI,A,SLOTS,LIVE,OUT,LIMIT,ZS0,ZS1,
                       V:tl.constexpr,D:tl.constexpr,C:tl.constexpr,HAS_LIMIT:tl.constexpr,
                       BV:tl.constexpr,BD:tl.constexpr):
    tile,row=tl.program_id(0),tl.program_id(1).to(tl.int64)
    request=row//C
    live_row=True
    if HAS_LIMIT:
        live_row=request<tl.load(LIMIT)
    slot=tl.load(SLOTS+request).to(tl.int64)
    live=(tl.load(LIVE+slot)!=0)&live_row
    v,d=tile*BV+tl.arange(0,BV),tl.arange(0,BD)
    weight=tl.load(A+(slot*V+v[:,None])*D+d[None,:],live&(v[:,None]<V)&(d[None,:]<D),other=0)
    psi=tl.load(PSI+row*D+d,d<D,other=0)
    raw=tl.load(Z+row*ZS0+v*ZS1,v<V,other=0).to(tl.float32)
    correction=tl.sum(weight*psi[None,:],axis=1)
    score=tl.where(correction==0,raw,raw+correction)  # bitwise zero-state identity
    tl.store(OUT+row*V+v,score,v<V)


@triton.jit
def tlt_reflex_cache(Q,FEATURE,SLOTS,LIVE,CACHED,ROOT_Q,ROOT_PSI,LIMIT,
                     QS0,QS1,V:tl.constexpr,D:tl.constexpr,HAS_LIMIT:tl.constexpr,BV:tl.constexpr,BD:tl.constexpr):
    tile,row=tl.program_id(0),tl.program_id(1).to(tl.int64)
    slot=tl.load(SLOTS+row).to(tl.int64)
    live=tl.load(LIVE+slot)!=0
    if HAS_LIMIT:
        live=live&(row<tl.load(LIMIT))
    v=tile*BV+tl.arange(0,BV)
    values=tl.load(Q+row*QS0+v*QS1,v<V,other=0)
    tl.store(ROOT_Q+slot*V+v,values,live&(v<V))
    if tile==0:
        d=tl.arange(0,BD)
        psi=tl.load(FEATURE+row*D+d,d<D,other=0)
        tl.store(ROOT_PSI+slot*D+d,psi,live&(d<D))
        tl.store(CACHED+slot,True,live)


@triton.jit
def tlt_reflex_reset(A,Q,PSI,LIVE,CACHED,SLOTS,V:tl.constexpr,D:tl.constexpr,
                     ALLOCATED:tl.constexpr,BLOCK:tl.constexpr):
    tile,row=tl.program_id(0),tl.program_id(1)
    slot=tl.load(SLOTS+row).to(tl.int64)
    x=tile*BLOCK+tl.arange(0,BLOCK)
    tl.store(A+slot*V*D+x,0,x<V*D)
    tl.store(Q+slot*V+x,0,x<V)
    tl.store(PSI+slot*D+x,0,x<D)
    if tile==0:
        tl.store(LIVE+slot,ALLOCATED)
        tl.store(CACHED+slot,False)
    # Reset all request-owned state, including cached supervision/feature;
    # done in this lifecycle kernel, never by batch-row compaction.


@triton.jit
def _teacher(T,MAP,ROOT,row,v,TS0,TS1,RS,V:tl.constexpr,GREEDY:tl.constexpr):
    index=tl.load(ROOT+row*RS).to(tl.int64)
    token=tl.load(MAP+v,v<V,other=0).to(tl.int64)
    if GREEDY:
        y=tl.load(T+tl.maximum(index,0)*TS0)
        p=(token==y).to(tl.float32)
    else:
        p=tl.load(T+tl.maximum(index,0)*TS0+token*TS1,v<V,other=0).to(tl.float32)
    return tl.where((v<V)&(index>=0),p,0.)


@triton.jit
def tlt_reflex_mass(T,MAP,ROOT,MASS,TS0,TS1,RS,V:tl.constexpr,TILES:tl.constexpr,GREEDY:tl.constexpr,BV:tl.constexpr):
    tile,row=tl.program_id(0),tl.program_id(1)
    v=tile*BV+tl.arange(0,BV)
    p=_teacher(T,MAP,ROOT,row,v,TS0,TS1,RS,V,GREEDY)
    tl.store(MASS+row*TILES+tile,tl.sum(p,axis=0))


@triton.jit
def tlt_reflex_stats(T,MAP,ROOT,SLOTS,Q,MASS,STATS,TS0,TS1,RS,V:tl.constexpr,TILES:tl.constexpr,
                     EPS:tl.constexpr,GREEDY:tl.constexpr,BV:tl.constexpr,BT:tl.constexpr):
    tile,row=tl.program_id(0),tl.program_id(1).to(tl.int64)
    slot=tl.load(SLOTS+row).to(tl.int64)
    v=tile*BV+tl.arange(0,BV); t=tl.arange(0,BT)
    mass=tl.sum(tl.load(MASS+row*TILES+t,t<TILES,other=0),axis=0)
    p=_teacher(T,MAP,ROOT,row,v,TS0,TS1,RS,V,GREEDY)/(mass+EPS)
    q=tl.load(Q+slot*V+v,v<V,other=0)
    tl.store(STATS+(row*TILES+tile)*2,tl.sum(tl.minimum(q,p),axis=0))
    tl.store(STATS+(row*TILES+tile)*2+1,tl.sum(tl.where(q<p,q,0.),axis=0))


@triton.jit
def tlt_reflex_update(T,MAP,ROOT,SLOTS,Q,PSI,A,LIVE,CACHED,MASS,STATS,UPDATES,TS0,TS1,RS,
                      V:tl.constexpr,D:tl.constexpr,TILES:tl.constexpr,EPS:tl.constexpr,LR:tl.constexpr,
                      DECAY:tl.constexpr,GREEDY:tl.constexpr,BV:tl.constexpr,BD:tl.constexpr,BT:tl.constexpr):
    tile,row=tl.program_id(0),tl.program_id(1).to(tl.int64)
    slot=tl.load(SLOTS+row).to(tl.int64)
    live=(tl.load(LIVE+slot)!=0)&(tl.load(CACHED+slot)!=0)&(tl.load(ROOT+row*RS)>=0)
    if live:
        v,d,t=tile*BV+tl.arange(0,BV),tl.arange(0,BD),tl.arange(0,BT)
        mass=tl.sum(tl.load(MASS+row*TILES+t,t<TILES,other=0),axis=0)
        alpha=tl.sum(tl.load(STATS+(row*TILES+t)*2,t<TILES,other=0),axis=0)
        selected=tl.sum(tl.load(STATS+(row*TILES+t)*2+1,t<TILES,other=0),axis=0)
        p=_teacher(T,MAP,ROOT,row,v,TS0,TS1,RS,V,GREEDY)/(mass+EPS)
        q=tl.load(Q+slot*V+v,v<V,other=0)
        gradient=q*(selected-(q<p).to(tl.float32))/(alpha+EPS)
        psi=tl.load(PSI+slot*D+d,d<D,other=0)
        ptr=A+(slot*V+v[:,None])*D+d[None,:]
        old=tl.load(ptr,(v[:,None]<V)&(d[None,:]<D),other=0)
        if DECAY!=1.:
            old=old*DECAY
        tl.store(ptr,old-LR*gradient[:,None]*psi[None,:],(v[:,None]<V)&(d[None,:]<D))
        if tile==0:
            tl.atomic_add(UPDATES,1)
    # cached flag reset must NOT occur here: other CTAs can still read it.


@triton.jit
def tlt_reflex_consume(CACHED,SLOTS,B:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.arange(0,BLOCK); slot=tl.load(SLOTS+row,row<B,other=0).to(tl.int64)
    tl.store(CACHED+slot,False,row<B)


def feature(hidden,projection,out):
    tlt_reflex_feature[(hidden.shape[0],)](hidden,projection,out,*hidden.stride(),hidden.shape[-1],projection.shape[-1],
        triton.next_power_of_2(hidden.shape[-1]),triton.next_power_of_2(projection.shape[-1]),
        num_warps=8,enable_fp_fusion=False)


def correct(s,raw,psi,slots,contexts,out,limit):
    tlt_reflex_correct[(s.tiles,raw.shape[0])](raw,psi,s.a,slots,s.live,out,slots if limit is None else limit,
        *raw.stride(),s.vocab,s.dim,contexts,limit is not None,256,triton.next_power_of_2(s.dim),
        num_warps=4,enable_fp_fusion=False)


def cache(s,q,slots,limit):
    tlt_reflex_cache[(s.tiles,slots.numel())](q,s.root_feature,slots,s.live,s.cached,s.q,s.psi,
        slots if limit is None else limit,*q.stride(),s.vocab,s.dim,limit is not None,256,
        triton.next_power_of_2(s.dim),num_warps=4,enable_fp_fusion=False)


def reset(s,slots,allocated):
    tlt_reflex_reset[(triton.cdiv(s.vocab*s.dim,256),slots.numel())](s.a,s.q,s.psi,s.live,s.cached,slots,
        s.vocab,s.dim,allocated,256,num_warps=4)


def update(s,target,roots,slots,greedy):
    # View only, no teacher copy/softmax. Original filtered target probs reused.
    t=target.reshape(-1) if greedy else target.reshape(-1,target.shape[-1])
    ts0=t.stride(0); ts1=0 if greedy else t.stride(1)
    rs=roots.stride(0)
    grid=(s.tiles,slots.numel()); bt=triton.next_power_of_2(s.tiles)
    tlt_reflex_mass[grid](t,s.mapping,roots,s.mass,ts0,ts1,rs,s.vocab,s.tiles,greedy,256,num_warps=4)
    tlt_reflex_stats[grid](t,s.mapping,roots,slots,s.q,s.mass,s.stats,ts0,ts1,rs,s.vocab,s.tiles,s.eps,greedy,256,bt,
                         num_warps=4,enable_fp_fusion=False)
    tlt_reflex_update[grid](t,s.mapping,roots,slots,s.q,s.psi,s.a,s.live,s.cached,s.mass,s.stats,s.total_updates,ts0,ts1,rs,
        s.vocab,s.dim,s.tiles,s.eps,s.lr,s.decay,greedy,256,triton.next_power_of_2(s.dim),bt,
        num_warps=4,enable_fp_fusion=False)
    tlt_reflex_consume[(1,)](s.cached,slots,slots.numel(),triton.next_power_of_2(slots.numel()),num_warps=4)
