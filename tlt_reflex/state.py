"""FP32 LK fast state keyed by SGLang req_pool_indices, never batch row.

No optimizer/model/collective/target forward. Buffers allocated once before
CUDA-graph capture; lifecycle calls occur at request allocation/free/clear.
"""
from contextlib import nullcontext
import importlib
import torch
import torch.nn.functional as F


class RequestReflex:
    def __init__(self, slots, vocab, hidden, mapping, *, dim=8, lr=.05, decay=0.,
                 seed=42, eps=1e-8, max_contexts=8, backend='triton', meter=None):
        if not 1<=dim<=64 or min(slots,vocab,hidden,max_contexts)<1 or lr<0 or decay<0:
            raise ValueError('invalid Reflex dimensions/LR/decay')
        if mapping.shape!=(vocab,) or mapping.dtype!=torch.long:
            raise ValueError('fixed compact-to-target mapping must have shape[V] and long dtype')
        self.device=mapping.device
        self.slots,self.vocab,self.hidden,self.dim=slots,vocab,hidden,dim
        self.lr,self.decay,self.eps=lr,1-lr*decay,eps
        self.mapping=mapping
        self.meter=meter
        self.backend=backend
        if backend=='triton' and self.device.type!='cuda':
            raise ValueError('production Triton Reflex requires CUDA; torch is a debug backend')
        self.kernels=importlib.import_module('tlt_reflex.kernels') if backend=='triton' else None
        generator=torch.Generator(device=self.device).manual_seed(seed)
        self.projection=torch.randn(hidden,dim,generator=generator,device=self.device,dtype=torch.float32)
        self.projection.mul_(hidden**-.5)
        self.a=torch.zeros(slots,vocab,dim,device=self.device,dtype=torch.float32)
        self.q=torch.zeros(slots,vocab,device=self.device,dtype=torch.float32)
        self.psi=torch.zeros(slots,dim,device=self.device,dtype=torch.float32)
        self.live=torch.zeros(slots,device=self.device,dtype=torch.bool)
        self.cached=torch.zeros_like(self.live)
        self.root_out=torch.empty(slots,vocab,device=self.device,dtype=torch.float32)
        self.deep_out=torch.empty(slots*max_contexts,vocab,device=self.device,dtype=torch.float32)
        self.root_feature=torch.empty(slots,dim,device=self.device,dtype=torch.float32)
        self.deep_feature=torch.empty(slots*max_contexts,dim,device=self.device,dtype=torch.float32)
        self.tiles=(vocab+255)//256
        self.mass=torch.empty(slots,self.tiles,device=self.device,dtype=torch.float32)
        self.stats=torch.empty(slots,self.tiles,2,device=self.device,dtype=torch.float32)
        self.total_updates=torch.zeros((),device=self.device,dtype=torch.int64)

    def section(self,key):
        return self.meter.section(key) if self.meter is not None else nullcontext()

    def reset_slots(self, slots, *, allocated=False):
        # Small host ID list is ALREADY supplied by SGLang allocator. Allocation
        # here is lifecycle-only, not a proposal/update/captured hot path.
        ids=torch.tensor(slots if isinstance(slots,list) else [slots],device=self.device,dtype=torch.long)
        if ids.numel()==0:
            return
        if self.kernels is not None:
            self.kernels.reset(self,ids,allocated)
        else:
            self.a[ids]=0; self.q[ids]=0; self.psi[ids]=0
            self.live[ids]=allocated; self.cached[ids]=False

    def clear(self):
        self.a.zero_(); self.q.zero_(); self.psi.zero_()
        self.live.zero_(); self.cached.zero_()

    @torch.no_grad()
    def correct(self, raw, native_hidden, req_slots, *, root=False, valid_bs=None):
        rows,v=raw.shape
        batch=req_slots.numel()
        if rows==0:
            return raw
        if batch==0 or rows%batch or v!=self.vocab or native_hidden.shape!=(rows,self.hidden):
            raise ValueError('logit/hidden rows must map to SGLang request slots and contexts')
        contexts=rows//batch
        feature_pool=self.root_feature if root else self.deep_feature
        output_pool=self.root_out if root else self.deep_out
        if root and contexts!=1 or rows>output_pool.shape[0]:
            raise ValueError('proposal workspace capacity exceeded')
        features=feature_pool[:rows]
        output=output_pool[:rows]
        with self.section('reflex_feature_ms'):
            if self.kernels is not None:
                self.kernels.feature(native_hidden,self.projection,features)
            else:
                features.copy_(F.normalize(native_hidden.float()@self.projection,dim=-1,eps=1e-6))
        with self.section('reflex_correction_ms'):
            if self.kernels is not None:
                self.kernels.correct(self,raw,features,req_slots,contexts,output,valid_bs)
            else:
                count=batch if valid_bs is None else int(valid_bs)
                for row in range(rows):
                    request=row//contexts; slot=int(req_slots[request])
                    delta=self.a[slot]@features[row] if request<count and self.live[slot] else torch.zeros_like(raw[row])
                    output[row].copy_(torch.where(delta==0,raw[row].float(),raw[row].float()+delta))
        return output

    @torch.no_grad()
    def cache_root(self, probs, req_slots, *, valid_bs=None):
        with self.section('reflex_cache_ms'):
            if self.kernels is not None:
                self.kernels.cache(self,probs,req_slots,valid_bs)
            else:
                count=req_slots.numel() if valid_bs is None else int(valid_bs)
                for row in range(count):
                    slot=int(req_slots[row])
                    if self.live[slot]:
                        self.q[slot].copy_(probs[row]); self.psi[slot].copy_(self.root_feature[row])
                        self.cached[slot]=True

    @torch.no_grad()
    def feedback(self, target, verified_roots, req_slots, *, greedy=False):
        # verified_roots are GLOBAL flattened target-logit indices from the
        # EXISTING verifier accept_index[:,0], before finished-row filtering.
        if req_slots.numel()==0:
            return
        with self.section('reflex_update_ms'):
            if self.kernels is not None:
                self.kernels.update(self,target,verified_roots,req_slots,greedy)
            else:
                for row,slot in enumerate(req_slots.tolist()):
                    index=int(verified_roots[row])
                    if index<0 or not self.live[slot] or not self.cached[slot]:
                        continue
                    p=(self.mapping.eq(target.reshape(-1)[index]).float() if greedy
                       else target.reshape(-1,target.shape[-1])[index].float()[self.mapping])
                    p=p/(p.sum()+self.eps)
                    q=self.q[slot]; alpha=torch.minimum(p,q).sum()
                    m=(q<p).float(); selected=(m*q).sum()
                    gradient=q*(selected-m)/(alpha+self.eps)
                    # Never run mul_(1) at weight_decay=0.
                    if self.decay!=1.:
                        self.a[slot].mul_(self.decay)
                    self.a[slot].add_(gradient[:,None]*self.psi[slot][None],alpha=-self.lr)
                    self.total_updates.add_(1)
                    self.cached[slot]=False
