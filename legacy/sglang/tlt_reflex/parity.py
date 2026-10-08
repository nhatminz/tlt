"""Offline real EAGLE3 representation validation; never enabled in benchmarks.

The one-shot recorder wraps the actual SGLang draft forward in validation-only
workers. It observes native prefill inputs/output and runs only OPD math (no
additional target/draft transformer forward). Source replay runs in SpecNaacl's
separate Python environment.
"""
import hashlib
import json
from pathlib import Path
from tlt_reflex.checkpoint import sha


def artifact_identity(target,draft):
    target,draft=Path(target).resolve(),Path(draft).resolve()
    files=[target/'config.json',draft/'config.json',draft/'model.safetensors',draft/'opd_projector.pt']
    # Target checkpoint files matter: the shared embedding comes from target.
    files += sorted(target.glob('*.safetensors'))+sorted(target.glob('*.bin'))
    return {str(p):sha(p) for p in files}


def implementation_identity():
    import torch,triton
    root=Path(__file__).resolve().parents[1]
    names=['patches/fastrl_reflex.patch','tlt_reflex/state.py','tlt_reflex/kernels.py','tlt_reflex/parity.py',
           'scripts/validate_tlt_eagle3_parity.py']
    return dict(files={name:sha(root/name) for name in names},torch=torch.__version__,triton=triton.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)


def require_parity_report(path,target,draft):
    if not path:raise ValueError('official OPD benchmark requires OPD_EAGLE3_PARITY_REPORT; run scripts/validate_tlt_eagle3_parity.sh')
    payload=json.loads(Path(path).read_text())
    if payload.get('passed') is not True:raise ValueError('real EAGLE3 representation validation failed')
    if payload.get('artifact_identity')!=artifact_identity(target,draft):raise ValueError('parity report does not match the exact target/EAGLE3/projector artifacts')
    if payload.get('validation_kind')!='real_checkpoint_native_sglang_vs_specnaacl':raise ValueError('fixture parity cannot certify a production checkpoint')
    if payload.get('implementation_identity')!=implementation_identity():raise ValueError('parity report implementation/runtime/GPU differs; regenerate validation')
    sources=payload.get('source_code_identity',{})
    if not sources or any(not Path(path).is_file() or sha(path)!=digest for path,digest in sources.items()):
        raise ValueError('SpecNaacl source differs from the certified representation; regenerate validation')
    if 'max_abs_corrected_logits_error' not in payload or 'max_abs_top16_probability_error' not in payload:
        raise ValueError('legacy parity report lacks corrected logits/probability validation; regenerate it')
    return payload


def corrected_logits(raw,u,adapter):
    """Offline dense observation of the core's ordered FP32 correction equation."""
    import torch
    delta=torch.zeros_like(raw,dtype=torch.float32)
    for rank in range(u.shape[-1]):delta.add_(u[:,rank:rank+1].float()*adapter[:,rank].float()[None,:])
    return raw.float()+delta


def install_native_recorder(model,state,path):
    import torch
    path=Path(path);original=model.forward;captured=False
    def forward(*args,**kwargs):
        nonlocal captured
        batch=kwargs.get('forward_batch',args[2] if len(args)>2 else None)
        # Validation script explicitly disables graphs. Dummy/warmup batches
        # never escape to host, and ordinary workers never install this wrapper.
        eligible=(not captured and not torch.cuda.is_current_stream_capturing() and batch is not None
                  and batch.batch_size==1 and batch.forward_mode.is_extend())
        if eligible:eligible=bool(state.live[batch.req_pool_indices.long()].any())
        inputs=None
        if eligible:
            input_ids=kwargs.get('input_ids',args[0] if args else batch.input_ids)
            positions=kwargs.get('positions',args[1] if len(args)>1 else batch.positions)
            if positions.numel() and int(positions[0])!=0:raise ValueError('parity prefill must start at position 0, no prefix cache/chunk')
            inputs=dict(input_ids=input_ids.detach().cpu(),positions=positions.detach().cpu(),
                aux_inputs=batch.spec_info.hidden_states.detach().cpu())
        output=original(*args,**kwargs)
        if eligible:
            captured=True
            # Fixed, deterministic small active set; no RNG draws for validation.
            v=state.vocab;r=state.rank
            token_ids=torch.tensor(sorted({0,1,v//3,v//2,v-1}),device=state.mapping.device)
            b=torch.zeros_like(state.B_fast)
            b[token_ids]=torch.sin(token_ids.float()[:,None]+torch.arange(r,device=b.device)[None]*.37)*.025
            state.B_fast.copy_(b);state.bitmap.zero_();state.active_ids[:token_ids.numel()].copy_(token_ids)
            state.active_count.fill_(token_ids.numel());state.B_version.add_(1)
            bits=torch.zeros_like(state.bitmap,device='cpu')
            for token in token_ids.cpu().tolist():bits[token//32]|=torch.tensor(1<<(token%32),dtype=torch.int64).to(torch.int32)
            state.bitmap.copy_(bits)
            q,ids=state.propose(output.next_token_logits,output.opd_head_input,batch.req_pool_indices,topk=state.topk)
            slot=batch.req_pool_indices.long()
            payload=dict(**inputs,head_input=output.opd_head_input.detach().cpu(),raw_logits=output.next_token_logits.detach().cpu(),
                u=state.u_cache[slot,0].detach().cpu(),top16_ids=ids.detach().cpu(),top16_probs=q.detach().cpu(),
                projector=state.projector.detach().cpu(),B=b.cpu(),
                corrected_logits=corrected_logits(output.next_token_logits,state.u_cache[slot,0],b).cpu(),
                projector_provenance=state.projector_provenance)
            path.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,path)
        return output
    model.forward=forward


def compare_payloads(native,source,*,head_atol=.02,logits_atol=.05,u_atol=.02,corrected_atol=.05,probability_atol=2e-4,min_top16_agreement=1.):
    import torch
    def error(key):
        if key not in native or key not in source:return float('inf')
        a,b=native[key].float(),source[key].float()
        if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():return float('inf')
        return float((a-b).abs().max())
    errors=dict(max_abs_head_input_error=error('head_input'),max_abs_raw_logits_error=error('raw_logits'),
        max_abs_logits_error=error('raw_logits'),max_abs_u_error=error('u'),max_abs_corrected_logits_error=error('corrected_logits'))
    if native['top16_ids'].shape!=source['top16_ids'].shape:agreement=0.
    else:agreement=float((native['top16_ids']==source['top16_ids']).float().mean())
    errors['top16_agreement']=agreement
    errors['max_abs_top16_probability_error']=error('top16_probs')
    errors['passed']=(errors['max_abs_head_input_error']<=head_atol and errors['max_abs_logits_error']<=logits_atol
        and errors['max_abs_u_error']<=u_atol and errors['max_abs_corrected_logits_error']<=corrected_atol
        and errors['max_abs_top16_probability_error']<=probability_atol and agreement>=min_top16_agreement
        and torch.equal(native['projector'],source['projector']))
    return errors
