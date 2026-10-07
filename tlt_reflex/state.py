"""Frozen persistent A, ONE rollout-wave B, slot-owned expanded-state caches."""
from contextlib import nullcontext
from types import SimpleNamespace as NS
import importlib
import math
import torch
from tlt_reflex.ported.reference import initialize_projector,select_states_reference,union_reference

COUNTERS=('opd_state_weight','opd_selected_states','opd_visited_states','opd_frontier_states',
 'opd_kl_sum','opd_union_size_sum','opd_compact_mass_sum','opd_draft_topk_target_mass_sum',
 'opd_invalid_states','opd_updates','opd_active_rows_sum','opd_rounds','opd_nonfinite_kl_states',
 'opd_sparse_rounds','opd_dense_rounds','opd_active_rows_max','opd_fused_rounds','opd_gemm_rounds')


class OPDState:
    def __init__(self,slots,head,mapping,*,projector=None,rank=8,topk=16,fast_lr=.01,
                 max_contexts=29,max_topk=4,max_nodes=48,max_path=9,backend='triton',
                 visited_weight=1.,frontier_weight=1.,update_stream=True,meter=None,
                 proposal_mode='auto',profile=None,projector_provenance='head_basis_initialized'):
        hidden=head.weight.shape[1];v=mapping.numel();device=mapping.device
        if min(slots,hidden,v,max_contexts,max_topk,max_nodes,max_path)<1 or not 1<=rank<=min(64,hidden):
            raise ValueError('invalid OPD dimensions/rank')
        if topk<max_topk or topk<1 or fast_lr<0 or min(visited_weight,frontier_weight)<0:
            raise ValueError('OPD_TOPK must cover all TLT/MAB topk and weights/LR must be nonnegative')
        if mapping.dtype!=torch.long or mapping.ndim!=1:raise ValueError('compact mapping must be long [V]')
        if head.weight.shape[0]<v:raise ValueError('TP-sharded/quantized head unsupported: use TP_SIZE=1')
        if backend not in ('triton','torch') or backend=='torch' and device.type!='cpu':
            raise ValueError('Torch is CPU oracle only; production uses Triton')
        if backend=='triton' and device.type!='cuda':raise ValueError('Triton requires CUDA')
        if proposal_mode not in ('auto','adaptive','sparse','fused','gemm'):raise ValueError('invalid OPD_PROPOSAL_MODE')
        self.slots,self.vocab,self.hidden,self.rank=slots,v,hidden,rank
        self.topk=min(topk,v);self.fast_lr=float(fast_lr);self.cache_contexts=max_contexts
        self.head=NS(weight=head.weight[:v],bias=getattr(head,'bias',None))
        self.mapping=mapping;self.backend=backend;self.meter=meter;self.proposal_mode=proposal_mode
        self.visited_weight,self.frontier_weight=float(visited_weight),float(frontier_weight)
        self.train_projector=False;self.enabled=True;self._ever_updated=False;self.full_vocab_inverse=None
        self.feedback_row_map=None;self.logits_dtype=head.weight.dtype
        self.projector=(initialize_projector(hidden,rank,head=self.head.weight) if projector is None else projector).to(device=device,dtype=torch.float32).detach()
        if self.projector.shape!=(hidden,rank) or not torch.isfinite(self.projector).all():raise ValueError('invalid persistent projector')
        self.projector_provenance=projector_provenance
        self._host_live=set();self.epoch=0;self._update_pending=False
        self.kernels=importlib.import_module('tlt_reflex.kernels') if backend=='triton' else None
        self.feedback_kernels=importlib.import_module('tlt_reflex.ported.opd_reflex_kernels') if backend=='triton' else None
        self.stream=torch.cuda.Stream(device=device) if backend=='triton' and update_stream else None
        self.source_ready=torch.cuda.Event() if self.stream is not None else None
        self.update_done=torch.cuda.Event() if self.stream is not None else None
        self.max_proposal_rows=slots*max_topk;self.max_feedback_rows=slots*max_nodes
        self.max_nodes=max_nodes;self.max_path=max_path
        self.thresholds={n:self._threshold(profile,n) for n in range(1,self.max_proposal_rows+1)}
        self.sparse_capacity=v if proposal_mode=='sparse' else min(v,max(self.thresholds.values()))
        def alloc(shape,dtype=torch.float32,zero=False):
            return (torch.zeros if zero else torch.empty)(shape,device=device,dtype=dtype)
        self.B_fast=alloc((v,rank),zero=True)
        self.bitmap=alloc((v+31)//32,torch.int32,True);self.active_ids=alloc(v,torch.int32)
        self.active_slots=alloc(v,torch.int32);self.active_count=alloc(1,torch.int32,True)
        self.dispatch_mode=alloc(1,torch.int32,True)
        self.live=alloc(slots,torch.bool,True);self.valid_cache=alloc((slots,max_contexts),torch.bool,True)
        self.expanded_ids=alloc((slots,max_contexts),torch.long);self.expanded_ids.fill_(-1)
        self.head_cache=alloc((slots,max_contexts,hidden),self.logits_dtype,True)
        self.u_cache=alloc((slots,max_contexts,rank),zero=True)
        self.ids_cache=alloc((slots,max_contexts,self.topk),torch.long);self.ids_cache.fill_(-1)
        self.q_cache=alloc((slots,max_contexts,self.topk),zero=True);self.norm_cache=alloc((slots,max_contexts,2),zero=True)
        rows=self.max_proposal_rows;k=self.topk;tiles=(v+255)//256
        self.root_head_workspace=alloc((slots,hidden),self.logits_dtype)
        self.root_logits_workspace=alloc((slots,v),self.logits_dtype)
        self.proposal_u=alloc((rows,rank));self.proposal_q=alloc(rows*k)
        self.proposal_ids=alloc(rows*k,torch.long);self.proposal_norm=alloc(rows*2)
        self.root_q=alloc(slots*k);self.root_ids=alloc(slots*k,torch.long);self.root_norm=alloc(slots*2)
        self.sparse_scores=alloc(rows*self.sparse_capacity)
        # Never reserve GEMM scores unless explicitly requested before capture.
        self.score_workspace=alloc(rows*v if proposal_mode=='gemm' else 1)
        self.proposal_tiles=[alloc(rows*tiles*(k if i>=2 else 1),torch.long if i==3 else torch.float32) for i in range(4)] if backend=='triton' else []
        n=self.max_feedback_rows
        self.parents_workspace=alloc(n,torch.long);self.context_workspace=alloc(n,torch.long)
        self.path_workspace=alloc(slots*max_path,torch.long);self.slot_workspace=alloc(slots,torch.long)
        self.selected_weights=alloc(n);self.selected_kind=alloc(n,torch.int32)
        self.selected_ids=alloc(n,torch.int32);self.selected_count=alloc(1,torch.int32)
        self.teacher_p=alloc(n*k);self.teacher_ids=alloc(n*k,torch.long);self.teacher_mass=alloc(n)
        self.teacher_q=alloc(n*k);self.teacher_draft_p=alloc(n*k)
        self.union_ids=alloc(n*2*k,torch.long);self.union_g=alloc(n*2*k)
        self.state_stats=alloc(n*10);self.round_weight=alloc(1)
        self.teacher_tiles=[alloc(n*tiles*(k if i>=2 else 1),torch.long if i==3 else torch.float32) for i in range(4)] if backend=='triton' else []
        self.counters=alloc(len(COUNTERS),torch.float64,True)
        if meter is not None:
            meter.opd=self
            meter.reflex_state_memory_mb=self.B_fast.numel()*4/1e6
            meter.reflex_buffer_memory_mb=sum(t.numel()*t.element_size() for t in self.__dict__.values() if torch.is_tensor(t))/1e6

    def _threshold(self,profile,n):
        if profile is None:return max(1,self.vocab//8)
        slots=sorted({s for values,_ in profile.buckets.values() for s in values})
        return min(slots+[self.vocab+1],key=lambda limit:sum(profile.costs(n,s)[0 if s<limit else 1] for s in slots))

    def section(self,key):return self.meter.section(key) if self.meter else nullcontext()
    def begin(self,key):
        # Ported feedback timer protocol, events on whichever stream executes it.
        return self.meter.begin(key) if self.meter else None
    def end(self,ticket):
        if self.meter:self.meter.end(ticket)

    def wait_for_update(self):
        if self._update_pending:
            with self.section('opd_wait_ms'):torch.cuda.current_stream(self.mapping.device).wait_event(self.update_done)
            self._update_pending=False

    def reset_slots(self,slots,*,allocated=False):
        values=slots if isinstance(slots,list) else [slots]
        if not values:return
        self.wait_for_update()
        if allocated and not self._host_live:
            self.B_fast.zero_();self.bitmap.zero_();self.active_count.zero_();self.epoch+=1
        if allocated:self._host_live.update(values)
        else:self._host_live.difference_update(values)
        ids=torch.tensor(values,device=self.mapping.device,dtype=torch.long)
        if self.kernels:self.kernels.reset(self,ids,allocated)
        else:
            for cache in (self.head_cache,self.u_cache,self.q_cache,self.norm_cache):cache[ids]=0
            self.ids_cache[ids]=-1;self.valid_cache[ids]=False;self.expanded_ids[ids]=-1;self.live[ids]=allocated

    def clear(self):
        self.wait_for_update();self._host_live.clear()
        self.B_fast.zero_();self.bitmap.zero_();self.active_count.zero_();self.live.zero_();self.valid_cache.zero_()
        self.head_cache.zero_();self.u_cache.zero_();self.q_cache.zero_();self.norm_cache.zero_()
        self.ids_cache.fill_(-1);self.expanded_ids.fill_(-1)

    @torch.no_grad()
    def propose(self,raw,head_input,req_slots,*,topk,offset=0,valid_bs=None,expanded_ids=None):
        self.wait_for_update()
        if head_input is None:raise ValueError('exact EAGLE3 lm_head input was not exposed')
        rows,v=raw.shape;b=req_slots.numel()
        if not b:return raw[:0,:topk],req_slots.new_empty((0,topk),dtype=torch.long)
        c=rows//b
        if rows%b or v!=self.vocab or head_input.shape!=(rows,self.hidden) or offset+c>self.cache_contexts or topk>self.topk:
            raise ValueError('OPD proposal shape/context capacity mismatch')
        if offset==0:
            with self.section('opd_cache_reset_ms'):
                if self.kernels:self.kernels.begin_tree(self,req_slots,valid_bs)
                else:self.valid_cache[req_slots.long()]=False;self.expanded_ids[req_slots.long()]=-1
        u=self.proposal_u[:rows]
        with self.section('opd_feature_ms'):
            if self.kernels:self.kernels.feature(self,head_input,req_slots,c,offset,u,valid_bs,expanded_ids)
            else:
                u.copy_(head_input.float()@self.projector)
                count=b if valid_bs is None else int(valid_bs)
                for row in range(count*c):
                    slot=int(req_slots[row//c]);ctx=offset+row%c
                    self.head_cache[slot,ctx]=head_input[row];self.u_cache[slot,ctx]=u[row];self.valid_cache[slot,ctx]=True
                    if expanded_ids is not None:self.expanded_ids[slot,ctx]=expanded_ids.flatten()[row]
        with self.section('opd_proposal_ms'):
            if self.kernels:
                q,ids,norm=self.kernels.propose(self,raw.view(b,c,v),u.view(b,c,self.rank),offset==0)
                self.kernels.cache(self,q,ids,norm,req_slots,c,offset,valid_bs)
            else:
                z=raw.float()+u@self.B_fast.T
                ids=torch.argsort(z,descending=True,stable=True)[:,:self.topk]
                norm=torch.stack((z.amax(-1),(z-z.amax(-1,keepdim=True)).exp().sum(-1)),-1)
                q=z.softmax(-1).gather(1,ids)
                for row in range((b if valid_bs is None else int(valid_bs))*c):
                    slot=int(req_slots[row//c]);ctx=offset+row%c
                    self.ids_cache[slot,ctx]=ids[row];self.q_cache[slot,ctx]=q[row];self.norm_cache[slot,ctx]=norm[row]
                if offset==0:self.counters[13]+=1
            return q.reshape(rows,self.topk)[:,:topk],ids.reshape(rows,self.topk)[:,:topk]

    @torch.no_grad()
    def refresh_root(self,draft_input,slots,max_topk):
        # An extend-root can wait across other scheduler batches while shared B
        # changes. Recompute its head from the cached EXACT operand so the root
        # proposal/norm and target-only q use the same pre-update B. No transformer
        # or target forward; output and head workspaces are preallocated.
        self.wait_for_update()
        with self.section('opd_root_head_ms'):
            if self.kernels:
                head=self.kernels.root_inputs(self,slots)
                raw=self.root_logits_workspace[:slots.numel()]
                torch.matmul(head,self.head.weight.T,out=raw)
            else:
                head=self.head_cache[slots.long(),0].to(self.head.weight.dtype)
                raw=head@self.head.weight.T
        draft_input.topk_p,draft_input.topk_index=self.propose(raw,head,slots,topk=max_topk)

    def tree_metadata(self,selected,parent_list,slots,topk,steps):
        with self.section('opd_context_mapping_ms'):
            return self._tree_metadata(selected,parent_list,slots,topk,steps)

    def _tree_metadata(self,selected,parent_list,slots,topk,steps):
        if self.kernels:return self.kernels.metadata(self,selected,parent_list,slots,topk,steps)
        b,q=selected.shape[0],selected.shape[1]+1
        parents=torch.full((b,q),-1,dtype=torch.long);contexts=torch.full_like(parents,-1)
        for row,slot in enumerate(slots.tolist()):
            contexts[row,0]=0 if self.valid_cache[slot,0] else -1
            for node,full in enumerate(selected[row].tolist(),1):
                table=full//topk
                parents[row,node]=0 if table==0 else int((selected[row]==parent_list[row,table]).nonzero()[0])+1
                match=((self.expanded_ids[slot]==full)&self.valid_cache[slot]).nonzero()
                if match.numel():contexts[row,node]=match[0,0]
        return parents,contexts

    @torch.no_grad()
    def feedback(self,target,accept_index,slots,parents,contexts,*,greedy=False):
        if parents is None or contexts is None:raise ValueError('missing expanded-node OPD metadata')
        self.wait_for_update()
        b,q=parents.shape;n=b*q;k=self.topk
        if n>self.max_feedback_rows:raise ValueError('feedback capacity exceeded')
        self.slot_workspace[:b].copy_(slots);self.feedback_row_map=self.slot_workspace[:b]
        path=NS(packed_indices=self.kernels.local_path(self,accept_index,q) if self.kernels else
                torch.where(accept_index>=0,accept_index-torch.arange(b)[:,None]*q,-1))
        tree=NS(parents=parents,feedback_contexts=contexts)
        # Teacher is consumed on source stream NOW, only compact buffers escape.
        if self.kernels:
            compact=self.feedback_kernels.prepare_compact_teacher(self,tree,path,target,greedy)
        else:compact=self._compact_reference(tree,path,target,greedy)
        if self.stream is not None:
            self.source_ready.record()
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(self.source_ready)
                self.feedback_kernels.feedback(self,tree,path,None,greedy,compact)
                self.update_done.record()
            # Native tree/path allocations remain valid until the async reader.
            for t in (parents,contexts):t.record_stream(self.stream)
            self._update_pending=True
        elif self.kernels:self.feedback_kernels.feedback(self,tree,path,None,greedy,compact)
        else:self._feedback_reference(tree,path,compact)
        self._ever_updated=self._ever_updated or self.fast_lr>0

    def _compact_reference(self,tree,path,target,greedy):
        weights,kind=select_states_reference(tree,path,self.visited_weight,self.frontier_weight)
        b,q=weights.shape;k=self.topk;n=b*q
        self.selected_weights[:n].copy_(weights.flatten());self.selected_kind[:n].copy_(kind.flatten())
        p=(self.mapping[None,None,:]==target[...,None]).float() if greedy else target[...,self.mapping].float()
        mass=p.sum(-1);good=torch.isfinite(mass)&(mass>0)
        normalized=p/torch.where(good,mass,1.)[...,None]
        ids=torch.argsort(normalized,descending=True,stable=True)[...,:k]
        tp=normalized.gather(-1,ids);valid=good[...,None]&(tp>0)
        ids=torch.where(valid,ids,-1);tp=torch.where(valid,tp,0.)
        di=self.ids_cache[self.feedback_row_map[:,None],tree.feedback_contexts.clamp_min(0)]
        pd=p.gather(-1,di.clamp_min(0))
        return tp.reshape(n,k),ids.reshape(n,k),mass.flatten(),pd.reshape(n,k)

    def _feedback_reference(self,tree,path,compact):
        b,q=tree.parents.shape;n=b*q;k=self.topk
        tp,ti,mass,pd=compact;good=torch.isfinite(mass)&(mass>0)
        slot=self.feedback_row_map[:,None];ctx=tree.feedback_contexts.clamp_min(0)
        head=self.head_cache[slot,ctx];u=self.u_cache[slot,ctx].reshape(n,self.rank)
        raw=torch.nn.functional.linear(head.to(self.head.weight.dtype),self.head.weight,self.head.bias).float().reshape(n,self.vocab)
        norm=self.norm_cache[slot,ctx].reshape(n,2)
        draft=(raw+u@self.B_fast.T-norm[:,:1]).exp()/norm[:,1:]
        di=self.ids_cache[slot,ctx].reshape(n,k)
        p=torch.zeros_like(draft);p.scatter_(1,di.clamp_min(0),pd/torch.where(good,mass,1.)[:,None])
        rr=torch.arange(n)[:,None].expand(n,k);positive=ti>=0;p[rr[positive],ti[positive]]=tp[positive]
        p=torch.where(good[:,None],p,0.)
        ids,valid,pu,qu,pt,qt,kl=union_reference(p,draft,ti,di)
        qu[:,:k]=self.q_cache[slot,ctx].reshape(n,k)
        qt=torch.where(valid.sum(-1)==self.vocab,0.,(1-qu.sum(-1)).clamp_min(0.))
        kl=torch.where(pu>0,pu*(pu.log()-qu.log()),0.).sum(-1)+torch.where(pt>0,pt*(pt.log()-qt.log()),0.)
        weights=self.selected_weights[:n];w=torch.where(good,weights,0.);total=w.sum()
        g=torch.where(valid,(qu-pu)*w[:,None],0.)
        if total>0 and self.fast_lr>0:
            delta=torch.zeros_like(self.B_fast)
            delta.index_add_(0,ids.clamp_min(0).flatten(),(g[...,None]*u[:,None,:]).reshape(-1,self.rank))
            self.B_fast.add_(delta,alpha=-self.fast_lr/float(total))
            touched=(self.B_fast.abs().sum(-1)>0).nonzero().flatten()
            active=torch.unique(torch.cat((self.active_ids[:int(self.active_count)].long(),touched)))
            self.active_ids[:active.numel()]=active;self.active_count.fill_(active.numel())
        selected=w>0;kind=self.selected_kind[:n];finite=torch.isfinite(kl)
        stats=torch.stack((total,selected.sum(),(selected&(kind==1)).sum(),(selected&(kind==2)).sum(),
            torch.where(selected&finite,w*kl,0.).sum(),(valid&selected[:,None]).sum(),
            torch.where(selected,mass,0.).sum(),torch.where(selected,pu[:,:k].sum(-1),0.).sum(),
            ((weights>0)&~good).sum(),((total>0)&(self.fast_lr>0)).float(),self.active_count[0].float(),total.new_tensor(1.),
            (selected&~finite).sum()))
        self.counters[:13].add_(stats.double());self.counters[15]=torch.maximum(self.counters[15],self.active_count[0].double())

    def report(self):
        self.wait_for_update()
        # Only explicit server-info RPC: no host reads during generation.
        return dict(zip(COUNTERS,self.counters.cpu().tolist()),opd_epoch=self.epoch,
                    opd_projector_provenance=self.projector_provenance,opd_projector_frozen=True)
