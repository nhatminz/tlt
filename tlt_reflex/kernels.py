"""Graph-safe slot cache / metadata and device-adaptive OPD proposal dispatch.

Feedback mathematics are the checked SpecNaacl port. Only proposal dispatch and
request ownership differ; no scheduler, verifier, KV cache or RNG lives here.
"""
import torch
import triton
import triton.language as tl
from tlt_reflex.ported import opd_reflex_kernels as opd
from tlt_reflex.ported.merge import _proposal_merge


@triton.jit
def _features(H,A,U,HC,UC,SLOTS,LIVE,VALID,LIMIT,EXPANDED,NODES,
              HS0:tl.constexpr,HS1:tl.constexpr,HIDDEN:tl.constexpr,R:tl.constexpr,
              C:tl.constexpr,CACHE:tl.constexpr,OFFSET:tl.constexpr,HAS_LIMIT:tl.constexpr,
              HAS_NODES:tl.constexpr,BH:tl.constexpr,BR:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);batch=row//C;ctx=OFFSET+row%C
    h,r=tl.arange(0,BH),tl.arange(0,BR)
    x=tl.load(H+row*HS0+h*HS1,h<HIDDEN,other=0)
    a=tl.load(A+h[:,None]*R+r[None,:],(h[:,None]<HIDDEN)&(r[None,:]<R),other=0)
    u=tl.sum(x.to(tl.float32)[:,None]*a,axis=0)
    tl.store(U+row*R+r,u,r<R)
    slot=tl.load(SLOTS+batch).to(tl.int64)
    live=tl.load(LIVE+slot)!=0
    if HAS_LIMIT:live=live&(batch<tl.load(LIMIT))
    cache=slot*CACHE+ctx
    tl.store(HC+cache*HIDDEN+h,x,live&(h<HIDDEN))
    tl.store(UC+cache*R+r,u,live&(r<R))
    tl.store(VALID+cache,True,live)
    if HAS_NODES:
        node=tl.load(NODES+row)
        tl.store(EXPANDED+cache,node,live)


@triton.jit
def _cache(Q,IDS,NORM,QC,IC,NC,SLOTS,LIVE,LIMIT,C:tl.constexpr,CACHE:tl.constexpr,
           OFFSET:tl.constexpr,K:tl.constexpr,HAS_LIMIT:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);batch=row//C;k=tl.arange(0,BK);n=tl.arange(0,2)
    slot=tl.load(SLOTS+batch).to(tl.int64);live=tl.load(LIVE+slot)!=0
    if HAS_LIMIT:live=live&(batch<tl.load(LIMIT))
    cached=slot*CACHE+OFFSET+row%C
    tl.store(QC+cached*K+k,tl.load(Q+row*K+k,k<K,other=0),live&(k<K))
    tl.store(IC+cached*K+k,tl.load(IDS+row*K+k,k<K,other=-1),live&(k<K))
    tl.store(NC+cached*2+n,tl.load(NORM+row*2+n),live)


@triton.jit
def _begin_tree(VALID,EXPANDED,SLOTS,LIVE,LIMIT,CACHE:tl.constexpr,HAS_LIMIT:tl.constexpr,BC:tl.constexpr):
    batch=tl.program_id(0);c=tl.arange(0,BC)
    slot=tl.load(SLOTS+batch).to(tl.int64);live=tl.load(LIVE+slot)!=0
    if HAS_LIMIT:live=live&(batch<tl.load(LIMIT))
    tl.store(VALID+slot*CACHE+c,False,live&(c<CACHE))
    tl.store(EXPANDED+slot*CACHE+c,-1,live&(c<CACHE))


@triton.jit
def _reset(H,U,IDS,Q,NORM,VALID,EXPANDED,LIVE,SLOTS,HIDDEN:tl.constexpr,R:tl.constexpr,
           K:tl.constexpr,CACHE:tl.constexpr,ALLOCATED:tl.constexpr,BLOCK:tl.constexpr):
    x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);slot=tl.load(SLOTS+tl.program_id(1)).to(tl.int64)
    tl.store(H+slot*CACHE*HIDDEN+x,0,x<CACHE*HIDDEN)
    tl.store(U+slot*CACHE*R+x,0,x<CACHE*R)
    tl.store(IDS+slot*CACHE*K+x,-1,x<CACHE*K)
    tl.store(Q+slot*CACHE*K+x,0,x<CACHE*K)
    tl.store(NORM+slot*CACHE*2+x,0,x<CACHE*2)
    tl.store(VALID+slot*CACHE+x,False,x<CACHE)
    tl.store(EXPANDED+slot*CACHE+x,-1,x<CACHE)
    if tl.program_id(0)==0:tl.store(LIVE+slot,ALLOCATED)


@triton.jit
def _dispatch(COUNT,MODE,COUNTERS,THRESHOLD:tl.constexpr,FORCE:tl.constexpr,ROOT:tl.constexpr):
    count=tl.load(COUNT)
    if FORCE<0:mode=tl.where(count<THRESHOLD,0,1)
    else:mode=FORCE
    tl.store(MODE,mode)
    if ROOT and FORCE==2:tl.atomic_add(COUNTERS+17,1.)


@triton.jit
def _metadata(SELECTED,PARENT_LIST,EXPANDED,VALID,SLOTS,PARENTS,CONTEXTS,
              Q:tl.constexpr,PL:tl.constexpr,TOPK:tl.constexpr,CACHE:tl.constexpr,
              USED:tl.constexpr,BQ:tl.constexpr,BC:tl.constexpr):
    batch=tl.program_id(0).to(tl.int64);node=tl.program_id(1)
    choices=tl.arange(0,BQ);c=tl.arange(0,BC)
    slot=tl.load(SLOTS+batch).to(tl.int64)
    if node==0:
        parent=-1;context=tl.where(tl.load(VALID+slot*CACHE),0,-1)
    else:
        full=tl.load(SELECTED+batch*(Q-1)+node-1)
        parent_table=full//TOPK
        if parent_table==0:parent=0
        else:
            parent_full=tl.load(PARENT_LIST+batch*PL+parent_table)
            selected=tl.load(SELECTED+batch*(Q-1)+choices,choices<Q-1,other=-2)
            parent=tl.min(tl.where((choices<Q-1)&(selected==parent_full),choices+1,Q),axis=0)
            parent=tl.where(parent<Q,parent,-2)
        expanded=tl.load(EXPANDED+slot*CACHE+c,c<USED,other=-2)
        valid=tl.load(VALID+slot*CACHE+c,c<USED,other=False)!=0
        context=tl.min(tl.where((c<USED)&valid&(expanded==full),c,CACHE),axis=0)
        context=tl.where(context<CACHE,context,-1)
    tl.store(PARENTS+batch*Q+node,parent);tl.store(CONTEXTS+batch*Q+node,context)


@triton.jit
def _path(IN,OUT,Q:tl.constexpr,WIDTH:tl.constexpr,BW:tl.constexpr):
    batch=tl.program_id(0);j=tl.arange(0,BW)
    index=tl.load(IN+batch*WIDTH+j,j<WIDTH,other=-1)
    tl.store(OUT+batch*WIDTH+j,tl.where(index>=0,index-batch*Q,-1),j<WIDTH)


def reset(s,slots,allocated):
    _reset[(triton.cdiv(s.cache_contexts*s.hidden,256),slots.numel())](
        s.head_cache,s.u_cache,s.ids_cache,s.q_cache,s.norm_cache,s.valid_cache,s.expanded_ids,s.live,slots,
        s.hidden,s.rank,s.topk,s.cache_contexts,allocated,256,num_warps=4)


def begin_tree(s,slots,limit):
    _begin_tree[(slots.numel(),)](s.valid_cache,s.expanded_ids,slots,s.live,slots if limit is None else limit,
        s.cache_contexts,limit is not None,triton.next_power_of_2(s.cache_contexts),num_warps=4)


def feature(s,hidden,slots,contexts,offset,u,limit,nodes):
    _features[(hidden.shape[0],)](hidden,s.projector,u,s.head_cache,s.u_cache,slots,s.live,s.valid_cache,
        slots if limit is None else limit,s.expanded_ids,slots if nodes is None else nodes,
        *hidden.stride(),s.hidden,s.rank,contexts,s.cache_contexts,offset,limit is not None,nodes is not None,
        triton.next_power_of_2(s.hidden),triton.next_power_of_2(s.rank),num_warps=8,enable_fp_fusion=False)


def cache(s,q,ids,norm,slots,c,offset,limit):
    _cache[(q.numel()//s.topk,)](q,ids,norm,s.q_cache,s.ids_cache,s.norm_cache,slots,s.live,
        slots if limit is None else limit,c,s.cache_contexts,offset,s.topk,limit is not None,
        triton.next_power_of_2(s.topk),num_warps=4)


def metadata(s,selected,parent_list,slots,topk,steps):
    b,q=selected.shape[0],selected.shape[1]+1
    parents=s.parents_workspace[:b*q].view(b,q);contexts=s.context_workspace[:b*q].view(b,q)
    _metadata[(b,q)](selected,parent_list,s.expanded_ids,s.valid_cache,slots,parents,contexts,
        q,parent_list.shape[1],topk,s.cache_contexts,1+topk*(steps-1),triton.next_power_of_2(q),
        triton.next_power_of_2(s.cache_contexts),num_warps=4)
    return parents,contexts


def local_path(s,accept_index,q):
    b,w=accept_index.shape;out=s.path_workspace[:b*w].view(b,w)
    _path[(b,)](accept_index,out,q,w,triton.next_power_of_2(w),num_warps=4)
    return out

@triton.jit(do_not_specialize=["ZS0","ZS1","ZS2","C","ENABLED","CAPACITY"])
def _adaptive_scan(Z,SCORES,BITS,SLOTS,CAPACITY,MAX,SUM,VALUES,IDS,U,B,COUNTERS,MODE,ZS0,ZS1,ZS2,
                     C,V:tl.constexpr,K:tl.constexpr,TILES:tl.constexpr,
                     BV:tl.constexpr,ENABLED,R:tl.constexpr,DENSE:tl.constexpr,ROOT:tl.constexpr,SPARSE:tl.constexpr):
    tile,context,batch=tl.program_id(0),tl.program_id(1),tl.program_id(2).to(tl.int64)
    v=tile*BV+tl.arange(0,BV)
    raw=tl.load(Z+batch*ZS0+context*ZS1+v*ZS2,v<V,other=0).to(tl.float32)
    if ENABLED:
        if tl.load(MODE)==1:
            dot=tl.full((BV,),0.,tl.float32)
            for r in tl.static_range(R):
                w=tl.load(B+v*R+r,v<V,other=0)
                u=tl.load(U+(batch*C+context)*R+r)
                dot=dot+w*u
            raw=raw+dot
            if ROOT:
                if (tile==0)&(context==0)&(batch==0):tl.atomic_add(COUNTERS+14,1.);tl.atomic_add(COUNTERS+16,1.)
        else:
            bits=tl.load(BITS+v//32,v<V,other=0)
            active=(v<V)&(((bits>>(v%32))&1)!=0)
            if SPARSE:
                slots=tl.load(SLOTS+v,active,other=0)
                corrected=tl.load(SCORES+(batch*C+context)*CAPACITY+slots,active,other=0)
            else:corrected=tl.load(SCORES+(batch*C+context)*V+v,active,other=0)
            raw=tl.where(active,corrected,raw)
    z=tl.where(v<V,raw,-float('inf'));live=v<V
    maximum=tl.max(z,axis=0)
    # An all-masked tile has mass zero, not exp(-inf - -inf)=NaN.
    # Finite tiles keep exactly the previous subtraction/reduction semantics.
    shift=tl.where(maximum==-float('inf'),0.,maximum)
    # Fixed pairwise FP32 tree: tl.sum alone changes association with Triton's
    # elements-per-thread layout (strided B loads vs contiguous score loads).
    # Pair reductions make strategy switching bitwise stable without another
    # vocabulary pass or global workspace (also supported by Triton 3.1).
    mass=tl.where(live,tl.exp(z-shift),0.)
    tl.static_assert(BV==256)
    for level in tl.static_range(8): # BV=256 in both proposal implementations
        mass=tl.sum(tl.reshape(mass,(BV//(1<<(level+1)),2)),axis=1)
    total=tl.sum(mass,axis=0)
    offset=(batch*C+context)*TILES+tile
    tl.store(MAX+offset,maximum);tl.store(SUM+offset,total)
    for j in range(K):
        score=tl.max(z,axis=0);token=tl.min(tl.where((z==score)&live,v,V),axis=0)
        tl.store(VALUES+offset*K+j,score);tl.store(IDS+offset*K+j,token)
        z=tl.where(v==token,-float('inf'),z);live=live&(v!=token)


def propose(s,raw,u,root=False):
    b,c,v=raw.shape;n=b*c;k=s.topk;tiles=triton.cdiv(v,256)
    out=(s.root_q if root else s.proposal_q)[:n*k].view(b,c,k)
    ids=(s.root_ids if root else s.proposal_ids)[:n*k].view(b,c,k)
    norm=(s.root_norm if root else s.proposal_norm)[:n*2].view(b,c,2)
    pools=[p[:n*tiles*(k if i>=2 else 1)] for i,p in enumerate(s.proposal_tiles)]
    force={'auto':-1,'adaptive':-1,'sparse':0,'fused':1,'gemm':2}[s.proposal_mode]
    _dispatch[(1,)](s.active_count,s.dispatch_mode,s.counters,s.thresholds[n],force,root,num_warps=1)
    _adaptive_sparse[(n,)](raw,u,s.B_fast,s.active_ids,s.active_count,s.sparse_scores,s.active_slots,s.sparse_capacity,
        s.counters,s.dispatch_mode,*raw.stride(),c,v,s.rank,0,2,root,128,
        num_warps=4,enable_fp_fusion=False)
    # Sparse preparation below uses MODE directly, so nonmonotone calibrated
    # dispatch is supported. GEMM is explicit and has a preallocated workspace.
    if force==2:
        opd._dense_gemm[(triton.cdiv(v,128),triton.cdiv(n,4))](raw,u,s.B_fast,s.active_count,s.score_workspace,s.counters,
            *raw.stride(),n,c,v,s.rank,0,1,root,128,4,num_warps=4,enable_fp_fusion=False)
    _adaptive_scan[(tiles,c,b)](raw,s.score_workspace if force==2 else s.sparse_scores,s.bitmap,s.active_slots,s.sparse_capacity,
        *pools,u,s.B_fast,s.counters,s.dispatch_mode,*raw.stride(),c,v,k,tiles,256,True,s.rank,False,root,force!=2,
        num_warps=4,enable_fp_fusion=False)
    _proposal_merge[(n,)](*pools,out,ids,norm,c,v,k,tiles,triton.next_power_of_2(tiles),
        triton.next_power_of_2(tiles*k),num_warps=4,enable_fp_fusion=False)
    return out,ids,norm

@triton.jit(do_not_specialize=["ZS0","ZS1","ZS2","C","THRESHOLD","CAPACITY"])
def _adaptive_sparse(Z,U,B,ACTIVE,COUNT,SCORES,SLOTS,CAPACITY,COUNTERS,DISPATCH,ZS0,ZS1,ZS2,
                   C,V:tl.constexpr,R:tl.constexpr,THRESHOLD,
                   MODE:tl.constexpr,ROOT:tl.constexpr,BS:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);count=tl.load(COUNT)
    use_sparse=tl.load(DISPATCH)==0
    if use_sparse:
        s=tl.arange(0,BS)
        for start in range(0,count,BS):
            ids=tl.load(ACTIVE+start+s,start+s<count,other=0).to(tl.int64)
            dot=tl.full((BS,),0.,tl.float32)
            for r in tl.static_range(R):
                w=tl.load(B+ids*R+r,start+s<count,other=0)
                u=tl.load(U+row*R+r)
                dot=dot+w*u
            raw=tl.load(Z+row//C*ZS0+row%C*ZS1+ids*ZS2,start+s<count,other=0).to(tl.float32)
            tl.store(SCORES+row*CAPACITY+start+s,raw+dot,start+s<count)
            if row==0:tl.store(SLOTS+ids,start+s,start+s<count)
        if ROOT:
            if row==0:tl.atomic_add(COUNTERS+13,1.)


@triton.jit
def _root_inputs(HC,OUT,SLOTS,HIDDEN:tl.constexpr,CACHE:tl.constexpr,BH:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);h=tl.arange(0,BH);slot=tl.load(SLOTS+row).to(tl.int64)
    tl.store(OUT+row*HIDDEN+h,tl.load(HC+slot*CACHE*HIDDEN+h,h<HIDDEN,other=0),h<HIDDEN)


def root_inputs(s,slots):
    out=s.root_head_workspace[:slots.numel()]
    _root_inputs[(slots.numel(),)](s.head_cache,out,slots,s.hidden,s.cache_contexts,
        triton.next_power_of_2(s.hidden),num_warps=4)
    return out
