"""Draft-only rebuild using the exact shifted FastGRPO input alignment."""
import torch
from helper.opd_attention import AttentionWorkspace
from helper.opd_static_cache import persistent_cache
from helper.fastgrpo_model import CachedDraftModel, CachedDraftAttention


def enable_cached_draft(model):
    # Shared cache arithmetic in both modes; this initializes no OPD state.
    model.draft_model.__class__ = CachedDraftModel
    for layer in model.draft_model.layers:
        layer.self_attn.__class__ = CachedDraftAttention


@torch.inference_mode()
def prefill_draft_prefix(model, features, shifted_ids, padding, *, last_valid=None, batch_capacity=None):
    """features[t]=target(x[t]); shifted_ids[t]=x[t+1], INCLUDING sampled bonus.

    This is exactly SpecNaacl's initial/committed draft prefill alignment.
    Target features are captured by existing target forwards, never recomputed.
    """
    if features.shape[:2] != shifted_ids.shape or padding.shape != shifted_ids.shape:
        raise ValueError('misaligned TLT transition history')
    enable_cached_draft(model)
    batch, length = shifted_ids.shape
    ws = AttentionWorkspace(shifted_ids.device)
    mask = ws.causal('transition', 0, length, batch, model.dtype, padding)
    positions = (~padding).long().cumsum(-1)-1
    positions.masked_fill_(padding, 0)
    cache = persistent_cache(model, '_opd_draft_kv_pool', batch, 1, shifted_ids.device, model.dtype)
    with torch.amp.autocast(str(model.target_model.device),dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16):
        out = model(hidden_states=features.to(model.dtype), input_ids=shifted_ids,
                    attention_mask=mask, position_ids=positions, use_cache=True, past_key_values=cache)
    if last_valid is None:
        last_valid = torch.full((batch,1),length-1,device=shifted_ids.device,dtype=torch.long)
    index = last_valid[:,:,None].expand(-1,-1,features.shape[-1])
    return dict(past_key_values=out['past_key_values'],hidden_states=out['hidden_states'].gather(1,index),
                next_feature_states=out['next_feature_states'].gather(1,index),position_ids=positions)
