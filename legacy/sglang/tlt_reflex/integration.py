"""Official EAGLE3 factory; TLT returns before importing/allocating OPD."""
import os
from pathlib import Path
import json


def method():
    value=os.environ.get('TLT_REFLEX_METHOD',os.environ.get('METHOD','tlt'))
    if value not in ('tlt','tlt_opd_reflex'):raise ValueError('METHOD must be tlt or tlt_opd_reflex')
    return value


def make_meter():
    if method()=='tlt' and os.environ.get('TLT_TRACE','0')!='1':return None
    from tlt_reflex.telemetry import Meter
    return Meter(os.environ.get('OPD_PROFILE','0')=='1')


def make_reflex(worker,embed):
    if method()=='tlt':return None
    import torch
    from tlt_reflex.state import OPDState
    from tlt_reflex.checkpoint import load_projector
    from tlt_reflex.profiles import discover
    if not worker.speculative_algorithm.is_eagle3():raise ValueError('TLT+OPD requires EAGLE3')
    args=worker.server_args
    if not args.disable_overlap_schedule:raise ValueError('EAGLEWorkerV2 overlap unsupported; use disable_overlap_schedule=True')
    if args.enable_dp_attention:raise ValueError('DP attention row redistribution unsupported')
    if args.tp_size!=1:raise ValueError('OPD sparse head reconstruction requires TP_SIZE=1; TP>1 vocab-sharded head unsupported')
    if os.environ.get('OPD_TRAIN_PROJECTOR','0')!='0':raise ValueError('OPD_TRAIN_PROJECTOR must be 0: no EAGLE3 Spot Trainer; A stays frozen')
    model=worker.draft_model_runner.model;cfg=worker.model_runner.model_config.hf_config
    processor=model.logits_processor
    if processor.use_fp32_lm_head or processor.logit_scale is not None or processor.final_logit_softcapping:
        raise ValueError('OPD supports native unscaled, unquantized EAGLE3 head dtype only')
    if getattr(args,'quantization',None) or getattr(args,'speculative_draft_model_quantization',None):
        raise ValueError('quantized OPD head reconstruction unsupported')
    v=int(cfg.draft_vocab_size or cfg.vocab_size)
    native_width=min(model.lm_head.weight.shape[0],int(cfg.vocab_size))
    if native_width!=v:
        raise ValueError('native EAGLE3 padded head logits do not match compact vocabulary; use a padding-aligned draft vocabulary/export')
    mapping=worker.hot_token_id
    if mapping is None:
        if v!=cfg.vocab_size:raise ValueError('compact EAGLE3 checkpoint requires fixed d2t')
        mapping=torch.arange(v,device=embed.device,dtype=torch.long)
    mapping=mapping.to(device=embed.device,dtype=torch.long)
    configs=[(worker.speculative_num_steps,worker.max_topk,worker.speculative_num_draft_tokens)]
    configs += [tuple(map(int,c.split('_'))) for c in (args.speculative_eagle_mab_configs or [])]
    max_topk=max(c[1] for c in configs)
    contexts=1+max(c[1]*(c[0]-1) for c in configs)
    max_nodes=max(c[2] for c in configs);max_path=max(c[0]+1 for c in configs)
    path=Path(args.speculative_draft_model_path)
    rank=int(os.environ.get('OPD_RANK',getattr(cfg,'opd_rank',8)));topk=int(os.environ.get('OPD_TOPK','16'))
    projector,provenance=load_projector(path,int(cfg.hidden_size),rank)
    from tlt_reflex.checkpoint import require_projector_provenance
    require_projector_provenance(provenance,allow_untrained=os.environ.get('OPD_ALLOW_UNTRAINED_PROJECTOR','0')=='1')
    profile,profile_path=discover(v,rank,model.lm_head.weight.dtype,min(v,topk),embed.device)
    if profile is None and os.environ.get('OPD_REQUIRE_CALIBRATED_PROFILE','0')=='1':
        raise ValueError('official OPD benchmark requires a compatible proposal profile; tune offline with scripts/tune_tlt_opd_proposals.sh before generation')
    mode=os.environ.get('OPD_PROPOSAL_MODE','auto')
    dense=os.environ.get('OPD_DENSE_IMPLEMENTATION','auto')
    if mode=='dense':mode='gemm' if dense=='gemm' else 'fused'
    state=OPDState(worker.req_to_token_pool.size,model.lm_head,mapping,projector=projector,
        rank=rank,topk=topk,fast_lr=float(os.environ.get('OPD_FAST_LR','.01')),
        max_contexts=contexts,max_topk=max_topk,max_nodes=max_nodes,max_path=max_path,
        max_speculative_batch_size=int(os.environ.get('OPD_MAX_SPECULATIVE_BATCH_SIZE',
            args.apdative_speculative_batch_size_threshold or args.max_running_requests or worker.req_to_token_pool.size)),
        debug=os.environ.get('OPD_DEBUG','0')=='1',
        visited_weight=float(os.environ.get('OPD_VISITED_WEIGHT','1')),
        frontier_weight=float(os.environ.get('OPD_FRONTIER_WEIGHT','1')),
        update_stream=os.environ.get('OPD_UPDATE_STREAM','1')=='1',meter=worker._tlt_meter,
        proposal_mode=mode,profile=profile,projector_provenance=provenance)
    print(f'OPD persistent memory MB={state.persistent_memory_mb:.3f}; scratch MB={state.scratch_memory_mb:.3f}; feedback batch capacity={state.max_speculative_batch_size}',flush=True)
    processor.expose_opd_head_input=True
    capture=os.environ.get('OPD_EAGLE3_PARITY_CAPTURE','')
    if capture:
        from tlt_reflex.parity import install_native_recorder
        install_native_recorder(model,state,capture)
    if worker._tlt_meter is not None:
        payload=json.loads(Path(profile_path).read_text()) if profile_path else {}
        worker._tlt_meter.opd_proposal_profile=dict(path=profile_path or None,calibrated=profile is not None,
            execution_key=payload.get('execution_key'),
            selection_policy='measured sparse/fused/GEMM cost interpolation',generation_autotuning=False)
    worker.req_to_token_pool._tlt_reflex=state
    print(f'TLT OPD: rank={rank} TopK={state.topk} contexts={contexts} A={provenance}/frozen; profile={profile_path or "uncalibrated fallback"}',flush=True)
    return state
