import torch

def initialize_projector(hidden,rank,seed=42,head=None):
    if not 1<=rank<=64:raise ValueError('OPD rank must be in [1,64]')
    if head is not None:
        ids=torch.linspace(0,head.shape[0]-1,rank,device=head.device).long()
        rows=head.detach().index_select(0,ids).float().t().cpu()
        return torch.linalg.qr(rows,mode='reduced').Q.contiguous()
    raise ValueError('output head required for deterministic OPD initialization')


def select_states_reference(tree,path,visited_weight=1.,frontier_weight=1.):
    """Vectorized CPU oracle, also used for non-CUDA integration tests."""
    b,q=tree.parents.shape;rows=torch.arange(q,device=tree.parents.device)
    visited=(rows[None,:,None]==path.packed_indices[:,None,:]).any(-1)
    parent_visited=(tree.parents[:,:,None]==path.packed_indices[:,None,:]).any(-1)
    frontier=~visited&parent_visited&(rows[None,:]>0)
    expanded=tree.feedback_contexts>=0
    weights=torch.where(visited,float(visited_weight),torch.where(frontier,float(frontier_weight),0.))
    weights=torch.where(expanded,weights,0.)
    kind=torch.where((weights>0)&visited,1,torch.where((weights>0)&frontier,2,0))
    return weights,kind


def union_reference(p,q,target_ids,draft_ids):
    """Top-k union + ONE tail; not a renormalized shortlist loss."""
    k=draft_ids.shape[-1]
    ids=torch.cat((draft_ids,target_ids),-1)
    positive_target=(target_ids>=0)&(p.gather(-1,target_ids.clamp_min(0))>0)
    valid=torch.cat((torch.ones_like(draft_ids,dtype=torch.bool),
                     positive_target&~(target_ids[:,:,None]==draft_ids[:,None,:]).any(-1)),-1)
    pu=torch.where(valid,p.gather(-1,ids.clamp_min(0)),0.);qu=torch.where(valid,q.gather(-1,ids.clamp_min(0)),0.)
    empty_tail=valid.sum(-1)==p.shape[-1]
    pt=torch.where(empty_tail,0.,(1-pu.sum(-1)).clamp_min(0.))
    qt=torch.where(empty_tail,0.,(1-qu.sum(-1)).clamp_min(0.))
    kl=torch.where(pu>0,pu*(pu.log()-qu.log()),0.).sum(-1)
    kl+=torch.where(pt>0,pt*(pt.log()-qt.log()),0.)
    return ids,valid,pu,qu,pt,qt,kl
