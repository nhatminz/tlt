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
def _cache(Q,IDS,NORM,QC,IC,NC,SLOTS,LIVE,LIMIT,B_VERSION,ROOT_VERSION,C:tl.constexpr,CACHE:tl.constexpr,
           OFFSET:tl.constexpr,K:tl.constexpr,HAS_LIMIT:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);batch=row//C;k=tl.arange(0,BK);n=tl.arange(0,2)
    slot=tl.load(SLOTS+batch).to(tl.int64);live=tl.load(LIVE+slot)!=0
    if HAS_LIMIT:live=live&(batch<tl.load(LIMIT))
    cached=slot*CACHE+OFFSET+row%C
    tl.store(QC+cached*K+k,tl.load(Q+row*K+k,k<K,other=0),live&(k<K))
    tl.store(IC+cached*K+k,tl.load(IDS+row*K+k,k<K,other=-1),live&(k<K))
    tl.store(NC+cached*2+n,tl.load(NORM+row*2+n),live)
    if OFFSET==0:tl.store(ROOT_VERSION+slot,tl.load(B_VERSION),live)


@triton.jit
def _begin_tree(VALID,EXPANDED,SLOTS,LIVE,LIMIT,CACHE:tl.constexpr,HAS_LIMIT:tl.constexpr,BC:tl.constexpr):
    batch=tl.program_id(0);c=tl.arange(0,BC)
    slot=tl.load(SLOTS+batch).to(tl.int64);live=tl.load(LIVE+slot)!=0
    if HAS_LIMIT:live=live&(batch<tl.load(LIMIT))
    tl.store(VALID+slot*CACHE+c,False,live&(c<CACHE))
    tl.store(EXPANDED+slot*CACHE+c,-1,live&(c<CACHE))


@triton.jit
def _reset(H,U,IDS,Q,NORM,VALID,EXPANDED,LIVE,SLOTS,ROOT_VERSION,HIDDEN:tl.constexpr,R:tl.constexpr,
           K:tl.constexpr,CACHE:tl.constexpr,ALLOCATED:tl.constexpr,BLOCK:tl.constexpr):
    x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);slot=tl.load(SLOTS+tl.program_id(1)).to(tl.int64)
    tl.store(H+slot*CACHE*HIDDEN+x,0,x<CACHE*HIDDEN)
    tl.store(U+slot*CACHE*R+x,0,x<CACHE*R)
    tl.store(IDS+slot*CACHE*K+x,-1,x<CACHE*K)
    tl.store(Q+slot*CACHE*K+x,0,x<CACHE*K)
    tl.store(NORM+slot*CACHE*2+x,0,x<CACHE*2)
    tl.store(VALID+slot*CACHE+x,False,x<CACHE)
    tl.store(EXPANDED+slot*CACHE+x,-1,x<CACHE)
    if tl.program_id(0)==0:
        tl.store(LIVE+slot,ALLOCATED);tl.store(ROOT_VERSION+slot,-1)


@triton.jit
def _dispatch(COUNT,MODE,COUNTERS,POINTS,COSTS,FLAGS,HAS_FLAGS:tl.constexpr,BN:tl.constexpr,
              N:tl.constexpr,P:tl.constexpr,FALLBACK:tl.constexpr,FORCE:tl.constexpr,ROOT:tl.constexpr):
    count=tl.load(COUNT);work=True
    if HAS_FLAGS:
        row=tl.arange(0,BN)
        work=tl.sum((tl.load(FLAGS+row,row<N,other=0)!=0).to(tl.int32),axis=0)>0
    if not work:mode=3  # cached root: no correction backend executed
    elif FORCE>=0:mode=FORCE
    elif P==0:mode=tl.where(count<FALLBACK,0,1)
    else:
        lower=0
        for i in range(P-1):
            lower=tl.where(tl.load(POINTS+i)<=count,i,lower)
        upper=tl.minimum(lower+1,P-1)
        lo=tl.load(POINTS+lower);hi=tl.load(POINTS+upper)
        w=(count-lo).to(tl.float64)/tl.maximum(hi-lo,1).to(tl.float64)
        w=tl.minimum(tl.maximum(w,0.),1.)
        a0=tl.load(COSTS+((N-1)*P+lower)*3);b0=tl.load(COSTS+((N-1)*P+upper)*3)
        a1=tl.load(COSTS+((N-1)*P+lower)*3+1);b1=tl.load(COSTS+((N-1)*P+upper)*3+1)
        a2=tl.load(COSTS+((N-1)*P+lower)*3+2);b2=tl.load(COSTS+((N-1)*P+upper)*3+2)
        c0=a0+(b0-a0)*w;c1=a1+(b1-a1)*w;c2=a2+(b2-a2)*w
        mode=tl.where((c0<=c1)&(c0<=c2),0,tl.where(c1<=c2,1,2))
    tl.store(MODE,mode)
    if ROOT and work:
        tl.atomic_add(COUNTERS+tl.where(mode==0,13,tl.where(mode==1,16,17)),1.)
        if mode!=0:tl.atomic_add(COUNTERS+14,1.)


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
def _path(IN,OUT,Q:tl.constexpr,WIDTH:tl.constexpr,BW:tl.constexpr,ROW_OFFSET:tl.constexpr):
    batch=tl.program_id(0);j=tl.arange(0,BW)
    index=tl.load(IN+batch*WIDTH+j,j<WIDTH,other=-1)
    tl.store(OUT+batch*WIDTH+j,tl.where(index>=0,index-(batch+ROW_OFFSET)*Q,-1),j<WIDTH)


def reset(s,slots,allocated):
    _reset[(triton.cdiv(s.cache_contexts*s.hidden,256),slots.numel())](
        s.head_cache,s.u_cache,s.ids_cache,s.q_cache,s.norm_cache,s.valid_cache,s.expanded_ids,s.live,slots,s.root_B_version,
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
        slots if limit is None else limit,s.B_version,s.root_B_version,c,s.cache_contexts,offset,s.topk,limit is not None,
        triton.next_power_of_2(s.topk),num_warps=4)


def metadata(s,selected,parent_list,slots,topk,steps):
    b,q=selected.shape[0],selected.shape[1]+1
    parents=s.parents_workspace[:b*q].view(b,q);contexts=s.context_workspace[:b*q].view(b,q)
    _metadata[(b,q)](selected,parent_list,s.expanded_ids,s.valid_cache,slots,parents,contexts,
        q,parent_list.shape[1],topk,s.cache_contexts,1+topk*(steps-1),triton.next_power_of_2(q),
        triton.next_power_of_2(s.cache_contexts),num_warps=4)
    return parents,contexts


def local_path(s,accept_index,q,row_offset=0):
    b,w=accept_index.shape;out=s.path_workspace[:b*w].view(b,w)
    _path[(b,)](accept_index,out,q,w,triton.next_power_of_2(w),row_offset,num_warps=4)
    return out

@triton.jit(do_not_specialize=["ZS0","ZS1","ZS2","C","ENABLED","CAPACITY"])
def _adaptive_scan(Z,SCORES,DENSE_SCORES,BITS,SLOTS,CAPACITY,MAX,SUM,VALUES,IDS,U,B,COUNTERS,MODE,FLAGS,HAS_FLAGS:tl.constexpr,ZS0,ZS1,ZS2,
                     C,V:tl.constexpr,K:tl.constexpr,TILES:tl.constexpr,
                     BV:tl.constexpr,ENABLED,R:tl.constexpr,DENSE:tl.constexpr,ROOT:tl.constexpr,SPARSE:tl.constexpr):
    tile,context,batch=tl.program_id(0),tl.program_id(1),tl.program_id(2).to(tl.int64)
    if not HAS_FLAGS or tl.load(FLAGS+batch*C+context)!=0:
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
                if tl.load(MODE)==0:
                    slots=tl.load(SLOTS+v,active,other=0)
                    corrected=tl.load(SCORES+(batch*C+context)*CAPACITY+slots,active,other=0)
                else:corrected=tl.load(DENSE_SCORES+(batch*C+context)*V+v,active,other=0)
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


def propose(s,raw,u,root=False,refresh_mask=None):
    b,c,v=raw.shape;n=b*c;k=s.topk;tiles=triton.cdiv(v,256)
    out=(s.root_q if root else s.proposal_q)[:n*k].view(b,c,k)
    ids=(s.root_ids if root else s.proposal_ids)[:n*k].view(b,c,k)
    norm=(s.root_norm if root else s.proposal_norm)[:n*2].view(b,c,2)
    pools=[p[:n*tiles*(k if i>=2 else 1)] for i,p in enumerate(s.proposal_tiles)]
    force={'auto':-1,'adaptive':-1,'sparse':0,'fused':1,'gemm':2}[s.proposal_mode]
    flags=s.active_count if refresh_mask is None else refresh_mask
    tables=s.dispatch_tables
    _dispatch[(1,)](s.active_count,s.dispatch_mode,s.counters,tables.breakpoints,
        tables.breakpoints if tables.costs is None else tables.costs,flags,refresh_mask is not None,triton.next_power_of_2(n),
        n,tables.points,max(1,v//8),force,root,
        num_warps=1,enable_fp_fusion=False)
    # Exactly one preparation backend does work; unselected kernels are no-ops
    # selected on device, so the same captured graph adapts to current active rows.
    if s.has_sparse:
        _adaptive_sparse[(n,)](raw,u,s.B_fast,s.active_ids,s.active_count,s.sparse_scores,s.active_slots,s.sparse_capacity,
            s.counters,s.dispatch_mode,flags,refresh_mask is not None,*raw.stride(),c,v,s.rank,0,2,False,128,
            num_warps=4,enable_fp_fusion=False)
    if s.has_gemm:
        _adaptive_gemm[(triton.cdiv(v,128),triton.cdiv(n,4))](raw,u,s.B_fast,s.active_count,s.score_workspace,s.counters,
            s.dispatch_mode,flags,refresh_mask is not None,*raw.stride(),n,c,v,s.rank,128,4,
            num_warps=4,enable_fp_fusion=False)
    _adaptive_scan[(tiles,c,b)](raw,s.sparse_scores,s.score_workspace,s.bitmap,s.active_slots,s.sparse_capacity,
        *pools,u,s.B_fast,s.counters,s.dispatch_mode,flags,refresh_mask is not None,*raw.stride(),c,v,k,tiles,256,True,s.rank,False,False,True,
        num_warps=4,enable_fp_fusion=False)
    _masked_merge[(n,)](*pools,out,ids,norm,flags,refresh_mask is not None,c,v,k,tiles,
        triton.next_power_of_2(tiles),triton.next_power_of_2(tiles*k),num_warps=4,enable_fp_fusion=False)
    return out,ids,norm

@triton.jit(do_not_specialize=["ZS0","ZS1","ZS2","C","THRESHOLD","CAPACITY"])
def _adaptive_sparse(Z,U,B,ACTIVE,COUNT,SCORES,SLOTS,CAPACITY,COUNTERS,DISPATCH,FLAGS,HAS_FLAGS:tl.constexpr,ZS0,ZS1,ZS2,
                   C,V:tl.constexpr,R:tl.constexpr,THRESHOLD,
                   MODE:tl.constexpr,ROOT:tl.constexpr,BS:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);count=tl.load(COUNT)
    use_sparse=(tl.load(DISPATCH)==0)
    if HAS_FLAGS:use_sparse=use_sparse&(tl.load(FLAGS+row)!=0)
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
            tl.store(SLOTS+ids,start+s,start+s<count)
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

@triton.jit(do_not_specialize=["CONTEXTS"])
def _masked_merge(MAX, SUM, VALUES, IDS, PROBS, TOP_IDS, NORM, FLAGS,HAS_FLAGS:tl.constexpr,
                    CONTEXTS, VOCAB: tl.constexpr, K: tl.constexpr,
                    TILES: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    if not HAS_FLAGS or tl.load(FLAGS+row)!=0:
        t = tl.arange(0, BT)
        maxima = tl.load(MAX + row * TILES + t, t < TILES, other=-float('inf'))
        maximum = tl.max(maxima, axis=0)
        sums = tl.load(SUM + row * TILES + t, t < TILES, other=0)
        total = tl.sum(sums * tl.exp(maxima - maximum), axis=0)
        tl.store(NORM + row * 2, maximum)
        tl.store(NORM + row * 2 + 1, total)
        candidates = tl.arange(0, BK)
        values = tl.load(VALUES + row * TILES * K + candidates, candidates < TILES * K, other=-float('inf'))
        ids = tl.load(IDS + row * TILES * K + candidates, candidates < TILES * K, other=VOCAB)
        live = (candidates < TILES * K) & (ids < VOCAB)
        for k in range(K):
            value = tl.max(tl.where(live, values, -float('inf')), axis=0)
            index = tl.min(tl.where(live & (values == value), ids, VOCAB), axis=0)
            tl.store(PROBS + row * K + k, tl.div_rn(tl.exp(value - maximum), total))
            tl.store(TOP_IDS + row * K + k, index)
            # -inf is also a legitimate masked score. Keep selection eligibility
            # separately, or exhausted finite candidates are emitted again at q=0.
            live = live & (ids != index)


@triton.jit(do_not_specialize=["ZS0","ZS1","ZS2","N","C"])
def _adaptive_gemm(Z,U,B,COUNT,SCORES,COUNTERS,DISPATCH,FLAGS,HAS_FLAGS:tl.constexpr,ZS0,ZS1,ZS2,
                N,C,V:tl.constexpr,R:tl.constexpr,

                BV:tl.constexpr,BC:tl.constexpr):
    # Tiled rank GEMM U @ B.T; reuse each B tile across BC contexts.
    # The explicit ordered FP32 multiply/add is IDENTICAL to sparse dot. No
    # TF32/FMA/BLAS association drift is allowed merely to switch strategy.
    tile,ct=tl.program_id(0),tl.program_id(1)
    count=tl.load(COUNT)
    use_dense=tl.load(DISPATCH)==2
    if use_dense:
        row=ct*BC+tl.arange(0,BC);v=tile*BV+tl.arange(0,BV)
        live_rows=row<N
        if HAS_FLAGS:live_rows=live_rows&(tl.load(FLAGS+row,row<N,other=0)!=0)
        if tl.sum(live_rows.to(tl.int32),axis=0)>0:
            dot=tl.full((BC,BV),0.,tl.float32)
            for r in tl.static_range(R):
                u=tl.load(U+row*R+r,live_rows,other=0)
                w=tl.load(B+v*R+r,v<V,other=0)
                dot=dot+u[:,None]*w[None,:]
            raw=tl.load(Z+(row//C)[:,None]*ZS0+(row%C)[:,None]*ZS1+v[None,:]*ZS2,
                        live_rows[:,None]&(v[None,:]<V),other=0).to(tl.float32)
            tl.store(SCORES+row[:,None]*V+v[None,:],raw+dot,live_rows[:,None]&(v[None,:]<V))

@triton.jit
def _prepare_root(HC,UC,QC,IC,NC,VALID,EXPANDED,ROOT_VERSION,B_VERSION,SLOTS,
                  H_OUT,U_OUT,Q_OUT,I_OUT,N_OUT,MASK,COUNTERS,
                  HIDDEN:tl.constexpr,R:tl.constexpr,K:tl.constexpr,CACHE:tl.constexpr,
                  BH:tl.constexpr,BR:tl.constexpr,BK:tl.constexpr,BC:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);slot=tl.load(SLOTS+row).to(tl.int64)
    good=tl.load(VALID+slot*CACHE)!=0
    stale=good&(tl.load(ROOT_VERSION+slot)!=tl.load(B_VERSION))
    tl.store(MASK+row,stale)
    h=tl.arange(0,BH);r=tl.arange(0,BR);k=tl.arange(0,BK);c=tl.arange(0,BC);n=tl.arange(0,2)
    # Head/u never change when A is frozen. Only a stale row needs head logits.
    tl.store(H_OUT+row*HIDDEN+h,tl.load(HC+slot*CACHE*HIDDEN+h,h<HIDDEN,other=0),stale&(h<HIDDEN))
    tl.store(U_OUT+row*R+r,tl.load(UC+slot*CACHE*R+r,r<R,other=0),r<R)
    tl.store(Q_OUT+row*K+k,tl.load(QC+slot*CACHE*K+k,k<K,other=0),k<K)
    tl.store(I_OUT+row*K+k,tl.load(IC+slot*CACHE*K+k,k<K,other=-1),k<K)
    tl.store(N_OUT+row*2+n,tl.load(NC+slot*CACHE*2+n))
    tl.store(VALID+slot*CACHE+c,False,(c>0)&(c<CACHE))
    tl.store(EXPANDED+slot*CACHE+c,-1,(c>0)&(c<CACHE))
    tl.atomic_add(COUNTERS+22,stale.to(tl.float64))
    tl.atomic_add(COUNTERS+23,(good&~stale).to(tl.float64))
    tl.atomic_add(COUNTERS+19,(~good).to(tl.float64))


@triton.jit
def _root_gemm(H,HEAD,RAW,MASK,COUNTERS,HIDDEN:tl.constexpr,V:tl.constexpr,
               BV:tl.constexpr,BH:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);tile=tl.program_id(1)
    if tl.load(MASK+row)!=0:
        v=tile*BV+tl.arange(0,BV);h=tl.arange(0,BH)
        # Tensor-core GEMM for native half heads; IEEE FP32 for FP32 fixtures.
        acc=tl.full((16,BV),0.,tl.float32)
        lane=tl.arange(0,16)
        for start in range(tl.cdiv(HIDDEN,BH)):
            hi=start*BH+h
            x=tl.load(H+row*HIDDEN+tl.zeros((16,1),tl.int32)+hi[None,:],(lane[:,None]==0)&(hi[None,:]<HIDDEN),other=0)
            w=tl.load(HEAD+v[None,:]*HIDDEN+hi[:,None],(v[None,:]<V)&(hi[:,None]<HIDDEN),other=0)
            acc=tl.dot(x,w,acc,input_precision='ieee')
        value=tl.sum(tl.where(lane[:,None]==0,acc,0.),axis=0)
        tl.store(RAW+row*V+v,value,v<V)
        if tile==0:tl.atomic_add(COUNTERS+24,1.)


def refresh_root(s,slots,k):
    b=slots.numel();head=s.root_head_workspace[:b];u=s.proposal_u[:b];raw=s.root_logits_workspace[:b]
    _prepare_root[(b,)](s.head_cache,s.u_cache,s.q_cache,s.ids_cache,s.norm_cache,s.valid_cache,s.expanded_ids,
        s.root_B_version,s.B_version,slots,head,u,s.root_q,s.root_ids,s.root_norm,s.root_refresh_mask,s.counters,
        s.hidden,s.rank,s.topk,s.cache_contexts,triton.next_power_of_2(s.hidden),triton.next_power_of_2(s.rank),
        triton.next_power_of_2(s.topk),triton.next_power_of_2(s.cache_contexts),num_warps=4)
    with s.section('opd_root_head_ms'):
        _root_gemm[(b,triton.cdiv(s.vocab,64))](head,s.head.weight,raw,s.root_refresh_mask,s.counters,
            s.hidden,s.vocab,64,32,num_warps=4,enable_fp_fusion=False)
    with s.section('opd_proposal_ms'):
        q,ids,norm=propose(s,raw.view(b,1,s.vocab),u.view(b,1,s.rank),True,s.root_refresh_mask)
        cache(s,q,ids,norm,slots,1,0,None)
    return q.view(b,s.topk)[:,:k],ids.view(b,s.topk)[:,:k]


@triton.jit
def _bump_version(VERSION,CHANGED):
    if tl.load(CHANGED)!=0:tl.atomic_add(VERSION,1)


def bump_version(s):_bump_version[(1,)](s.B_version,s.update_changed,num_warps=1)


@triton.jit
def _validate_tree(PARENTS,CONTEXTS,SLOTS,VALID,EXPANDED,SELECTED,COUNTERS,
                   Q:tl.constexpr,CACHE:tl.constexpr,HAS_SELECTED:tl.constexpr,DEBUG:tl.constexpr,BQ:tl.constexpr):
    batch=tl.program_id(0).to(tl.int64);row=tl.arange(0,BQ)
    slot=tl.load(SLOTS+batch).to(tl.int64)
    parent=tl.load(PARENTS+batch*Q+row,row<Q,other=-2)
    context=tl.load(CONTEXTS+batch*Q+row,row<Q,other=-1)
    orphan=(row>0)&(row<Q)&((parent<0)|(parent>=row))
    invalid=(row<Q)&((context<-1)|(context>=CACHE))
    expanded=context>=0
    cached=tl.load(VALID+slot*CACHE+tl.minimum(tl.maximum(context,0),CACHE-1),row<Q,other=False)!=0
    invalid=invalid|((row<Q)&expanded&~cached)
    if HAS_SELECTED:
        token=tl.load(SELECTED+batch*(Q-1)+tl.maximum(row-1,0),(row>0)&(row<Q),other=-1)
        full=tl.load(EXPANDED+slot*CACHE+tl.minimum(tl.maximum(context,0),CACHE-1),row<Q,other=-1)
        invalid=invalid|((row>0)&(row<Q)&expanded&(full!=token))
    tl.atomic_add(COUNTERS+18,tl.sum(orphan.to(tl.float64)))
    tl.atomic_add(COUNTERS+19,tl.sum(invalid.to(tl.float64)))
    # Never dereference a bad feedback context in production. This masks OPD
    # metadata only; native candidate selection and verifier remain untouched.
    tl.store(CONTEXTS+batch*Q+row,-1,(row<Q)&(orphan|invalid))
    if DEBUG:
        tl.device_assert(tl.sum((orphan|invalid).to(tl.int32))==0,'orphan node or invalid expanded OPD context')


def validate_tree(s,parents,contexts,slots,selected=None):
    b,q=parents.shape
    _validate_tree[(b,)](parents,contexts,slots,s.valid_cache,s.expanded_ids,
        slots if selected is None else selected,s.counters,q,s.cache_contexts,selected is not None,s.debug,
        triton.next_power_of_2(q),num_warps=4,debug=s.debug)


@triton.jit
def _collect_weight(WEIGHT,TOTAL):tl.atomic_add(TOTAL,tl.load(WEIGHT))


@triton.jit
def _apply_gradient(GRAD,B,WEIGHT,BITS,IDS,COUNT,CHANGED,V:tl.constexpr,R:tl.constexpr,LR:tl.constexpr,BV:tl.constexpr,BR:tl.constexpr):
    v=tl.program_id(0)*BV+tl.arange(0,BV);r=tl.arange(0,BR)
    weight=tl.load(WEIGHT);live=(v[:,None]<V)&(r[None,:]<R)&(weight>0)
    g=tl.load(GRAD+v[:,None]*R+r[None,:],live,other=0)
    old=tl.load(B+v[:,None]*R+r[None,:],live,other=0)
    delta=-LR*tl.div_rn(g,tl.where(weight>0,weight,1.))
    tl.store(B+v[:,None]*R+r[None,:],old+delta,live)
    actual=live&((old+delta)!=old)
    if tl.sum(tl.sum(actual.to(tl.int32),axis=1),axis=0)>0:tl.atomic_or(CHANGED,1)
    touched=(v<V)&(tl.sum((delta!=0).to(tl.int32),axis=1)>0)&(weight>0)
    bit=1<<(v%32);before=tl.atomic_or(BITS+v//32,bit,touched)
    first=touched&((before&bit)==0);position=tl.atomic_add(COUNT+tl.zeros((BV,),tl.int32),1,first)
    tl.store(IDS+position,v,first)


def accumulate_feedback(s,q,n):
    opd._update[(min(n,32),)](s.selected_ids,s.selected_count,s.union_ids,s.union_g,s.u_cache,s._chunk_contexts,s.gradient_one,
        s.feedback_gradient,s.gradient_bitmap,s.gradient_ids,s.gradient_count,s.feedback_row_map,
        q,s.cache_contexts,s.topk,s.rank,-1.,triton.next_power_of_2(2*s.topk),triton.next_power_of_2(s.rank),True,
        num_warps=4,enable_fp_fusion=False)
    _collect_weight[(1,)](s.round_weight,s.feedback_total_weight,num_warps=1)


def apply_feedback_gradient(s):
    s.round_weight.copy_(s.feedback_total_weight);s.update_changed.zero_()
    if s.fast_lr>0:
        _apply_gradient[(triton.cdiv(s.vocab,128),)](s.feedback_gradient,s.B_fast,s.round_weight,s.bitmap,s.active_ids,s.active_count,
            s.update_changed,s.vocab,s.rank,s.fast_lr,128,triton.next_power_of_2(s.rank),num_warps=4,enable_fp_fusion=False)
    opd._round_end[(1,)](s.B_fast,s.bitmap,s.active_ids,s.active_count,s.round_weight,s.counters,s.rank,s.fast_lr,
        128,triton.next_power_of_2(s.rank),num_warps=4)
    bump_version(s)

@triton.jit
def _terminal_frontier(PARENTS,PATH,TERMINAL,WEIGHTS,KIND,Q:tl.constexpr,WIDTH:tl.constexpr,BQ:tl.constexpr,BW:tl.constexpr):
    batch=tl.program_id(0)
    if tl.load(TERMINAL+batch)!=0:
        j=tl.arange(0,BW);node=tl.arange(0,BQ)
        path=tl.load(PATH+batch*WIDTH+j,j<WIDTH,other=-1)
        last_slot=tl.max(tl.where(path>=0,j,0),axis=0)
        last=tl.load(PATH+batch*WIDTH+last_slot)
        parent=tl.load(PARENTS+batch*Q+node,node<Q,other=-2)
        kind=tl.load(KIND+batch*Q+node,node<Q,other=0)
        skip=(kind==2)&(parent==last)&(node<Q)
        tl.store(WEIGHTS+batch*Q+node,0.,skip);tl.store(KIND+batch*Q+node,0,skip)


def mask_terminal_frontier(s,tree,path):
    b,q=tree.parents.shape;w=path.packed_indices.shape[1]
    _terminal_frontier[(b,)](tree.parents,path.packed_indices,s.terminal_mask,s.selected_weights,s.selected_kind,
        q,w,triton.next_power_of_2(q),triton.next_power_of_2(w),num_warps=4)
