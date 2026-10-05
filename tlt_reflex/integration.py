"""Minimal factory, called only by official TLT's EAGLE worker init."""
import os
import torch
from tlt_reflex.telemetry import Meter
from tlt_reflex.state import RequestReflex


def method():
    value=os.environ.get('TLT_REFLEX_METHOD','tlt')
    if value not in ('tlt','tlt_reflex'):
        raise ValueError('TLT_REFLEX_METHOD must be tlt or tlt_reflex')
    return value


def make_meter():
    # Production tlt-only takes the literal original paths when tracing OFF.
    if method()=='tlt' and os.environ.get('TLT_TRACE','0')!='1':
        return None
    return Meter(os.environ.get('REFLEX_PROFILE','0')=='1')


def make_reflex(worker, embed):
    if method()=='tlt':
        return None
    if not worker.speculative_algorithm.is_eagle3():
        raise ValueError('tlt_reflex requires upstream EAGLE3; no fallback to EAGLE or full target vocabulary')
    if not worker.server_args.disable_overlap_schedule:
        raise ValueError('this plugin hooks official non-overlap EAGLEWorker, not EAGLEWorkerV2; set disable_overlap_schedule=True as in FastRL rollout')
    if worker.server_args.enable_dp_attention:
        raise ValueError('DP-attention token-row redistribution requires an explicit mapping adapter; ordinary TP/independent DP workers are supported')
    cfg=worker.model_runner.model_config.hf_config
    vocab=int(cfg.draft_vocab_size or cfg.vocab_size)
    mapping=worker.hot_token_id
    if mapping is None:
        if vocab!=embed.shape[0]:
            raise ValueError('compact EAGLE3 checkpoint must supply d2t mapping')
        mapping=torch.arange(vocab,device=embed.device)
    max_topk=max([worker.max_topk]+[int(c.split('_')[1]) for c in (worker.server_args.speculative_eagle_mab_configs or [])])
    state=RequestReflex(worker.req_to_token_pool.size,vocab,int(cfg.hidden_size),mapping.long(),
        dim=int(os.environ.get('REFLEX_FEATURE_DIM','8')),lr=float(os.environ.get('REFLEX_LR','.05')),
        decay=float(os.environ.get('REFLEX_WEIGHT_DECAY','0')),seed=int(os.environ.get('REFLEX_SEED','42')),
        max_contexts=max_topk,meter=worker._tlt_meter)
    worker.req_to_token_pool._tlt_reflex=state
    return state
