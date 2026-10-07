"""Checked SpecForge EAGLE3 -> official SGLang checkpoint layout adapter.

No training/vocabulary rebuilding/random initialization. Source weights/config/
mapping stay in SpecNaacl; export is a separate runtime-format artifact.
"""
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def validate_config(draft,target):
    if draft.get('architectures')!=['LlamaForCausalLMEagle3'] or draft.get('num_hidden_layers')!=1:
        raise ValueError('this adapter requires actual one-layer SpecForge LlamaForCausalLMEagle3')
    if draft.get('fc_norm') or draft.get('norm_output') or draft.get('attention_bias') or draft.get('bias') or draft.get('mlp_bias'):
        raise ValueError('fc_norm/norm_output/attention_bias variants require additional runtime architecture support; do not silently drop them')
    if draft.get('tie_word_embeddings'):
        raise ValueError('this compact-head adapter requires the separately trained lm_head')
    if draft['vocab_size']!=target['vocab_size'] or draft.get('target_hidden_size',draft['hidden_size'])!=target['hidden_size']:
        raise ValueError('draft target vocabulary/feature hidden width mismatches target')
    if target.get('model_type') not in ('llama','qwen2','qwen3'):
        raise ValueError('unsupported target family for this checked EAGLE3 adapter')
    layers=(draft.get('eagle_config') or {}).get('eagle_aux_hidden_state_layer_ids')
    if layers is None:
        n=target['num_hidden_layers']; layers=[1,n//2-1,n-4]
    if len(layers)!=3 or any(not isinstance(i,int) or not 0<=i<target['num_hidden_layers']-1 for i in layers) or sorted(set(layers))!=layers:
        raise ValueError('invalid three SpecForge feature layer indices')
    cfg=dict(draft)
    # The PUBLIC SGLang set_eagle3_layers_to_capture API ALREADY adds +1
    # before setting before-layer capture indices. Keep config layer IDs
    # unchanged; adding +1 here would double-shift teacher supervision.
    cfg['eagle_config']={'eagle_aux_hidden_state_layer_ids':list(layers)}
    cfg['target_hidden_size']=target['hidden_size']
    return cfg


def export(checkpoint,config,mapping,target,output):
    import torch
    from safetensors.torch import load_file,save_file
    checkpoint=Path(checkpoint).resolve(); config=Path(config).resolve()
    source_directory=checkpoint if checkpoint.is_dir() else checkpoint.parent
    side_projector=source_directory/'opd_projector.pt'
    projector_sidecar=side_projector if side_projector.is_file() else None
    mapping=Path(mapping).resolve(); target=Path(target).resolve()
    target_file=target/'config.json'
    dc=json.loads(config.read_text()); tc=json.loads(target_file.read_text())
    cfg=validate_config(dc,tc)
    if checkpoint.is_dir():
        found=[checkpoint/name for name in ('model.safetensors','training_state.pt','pytorch_model.bin') if (checkpoint/name).is_file()]
        if not found:
            latest=sorted(checkpoint.glob('*-latest/training_state.pt'))
            if len(latest)==1:found=latest
        if len(found)!=1:
            # Training exports may include both model.safetensors and state;
            # always prefer the actual HF weight artifact, not optimizer data.
            if found and found[0].name=='model.safetensors': found=found[:1]
            else: raise FileNotFoundError('checkpoint must expose model.safetensors or training_state.pt/pytorch_model.bin')
        checkpoint=found[0]
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    weights=load_file(str(checkpoint)) if checkpoint.suffix=='.safetensors' else torch.load(checkpoint,map_location='cpu',weights_only=False)
    payload=weights
    metadata=payload.get('metadata',payload.get('specforge',{})) if isinstance(payload,dict) else {}
    top_projector=payload.get('opd_projector') if isinstance(payload,dict) else None
    for key in ('draft_state_dict','state_dict','draft_model'):
        if isinstance(weights,dict) and key in weights and isinstance(weights[key],dict):
            weights=weights[key]; break
    if not isinstance(weights,dict) or not all(isinstance(x,torch.Tensor) for x in weights.values()):
        raise ValueError('checkpoint has no actual draft tensor state dictionary')
    # Do not accept silently ignored names: SGLang's upstream loader otherwise
    # leaves unknown/missing parameters randomly initialized without warning.
    allowed=('embed_tokens.weight','fc.weight','norm.weight','lm_head.weight','d2t','t2d',
        'midlayer.input_layernorm.weight','midlayer.hidden_norm.weight','midlayer.post_attention_layernorm.weight',
        'midlayer.self_attn.q_proj.weight','midlayer.self_attn.k_proj.weight','midlayer.self_attn.v_proj.weight',
        'midlayer.self_attn.o_proj.weight','midlayer.mlp.gate_proj.weight','midlayer.mlp.up_proj.weight','midlayer.mlp.down_proj.weight','opd_projector')
    cleaned={}
    for name,value in weights.items():
        for prefix in ('module.','draft_model.','model.'):
            if name.startswith(prefix): name=name[len(prefix):]
        if name.endswith('rotary_emb.inv_freq'):
            continue  # derived RoPE buffer, same config builds it in runtime
        if name not in allowed: raise ValueError('unrecognized checkpoint weight: '+name)
        cleaned[name]=value
    projectors=[value for value in (cleaned.pop('opd_projector',None),top_projector) if value is not None]
    if projector_sidecar is not None:projectors.append(torch.load(projector_sidecar,map_location='cpu',weights_only=True))
    if any(not torch.is_tensor(a) for a in projectors):raise ValueError('invalid checkpoint projector')
    if any(not torch.equal(projectors[0],a) for a in projectors[1:]):raise ValueError('checkpoint projector copies disagree')
    h,v,dv,th=dc['hidden_size'],dc['vocab_size'],dc['draft_vocab_size'],tc['hidden_size']
    hd=dc.get('head_dim',h//dc['num_attention_heads']); qh=hd*dc['num_attention_heads']; kh=hd*dc['num_key_value_heads']; inter=dc['intermediate_size']
    expected={'fc.weight':(h,3*th),'norm.weight':(h,),'lm_head.weight':(dv,h),
        'midlayer.input_layernorm.weight':(h,),'midlayer.hidden_norm.weight':(h,),
        'midlayer.post_attention_layernorm.weight':(h,),
        'midlayer.self_attn.q_proj.weight':(qh,2*h),'midlayer.self_attn.k_proj.weight':(kh,2*h),
        'midlayer.self_attn.v_proj.weight':(kh,2*h),'midlayer.self_attn.o_proj.weight':(h,qh),
        'midlayer.mlp.gate_proj.weight':(inter,h),'midlayer.mlp.up_proj.weight':(inter,h),
        'midlayer.mlp.down_proj.weight':(h,inter)}
    for name,shape in expected.items():
        if name not in cleaned or tuple(cleaned[name].shape)!=shape:
            raise ValueError(f'missing/incompatible pretrained parameter {name}: expected {shape}')
    if 'embed_tokens.weight' in cleaned and cleaned['embed_tokens.weight'].shape!=(v,h):
        raise ValueError('pretrained embedding vocabulary/hidden size mismatch')
    # SGLang shares target embeddings; differing hidden sizes need a different
    # projection/embedding architecture, not a random padding/truncation hack.
    if h!=th: raise ValueError('this shared-target-embedding EAGLE3 path requires draft hidden == target hidden')
    vocab=torch.load(mapping,map_location='cpu',weights_only=True)
    if not isinstance(vocab,dict) or 'd2t' not in vocab or 't2d' not in vocab:
        raise ValueError('fixed SpecForge vocabulary mapping with d2t/t2d required')
    d2t=vocab['d2t'].long(); t2d=vocab['t2d']
    ids=d2t+torch.arange(dv)
    if d2t.shape!=(dv,) or t2d.shape!=(v,) or ids.min()<0 or ids.max()>=v or ids.unique().numel()!=dv:
        raise ValueError('invalid compact vocabulary mapping')
    if not torch.equal(t2d.bool().nonzero().flatten(),ids):
        raise ValueError('t2d and d2t mapping disagree; no vocabulary rebuilding permitted')
    for name in ('d2t','t2d'):
        if name in cleaned and not torch.equal(cleaned[name],vocab[name]):
            raise ValueError('checkpoint mapping differs from fixed pretrained mapping: '+name)
    from tlt_reflex.ported.reference import initialize_projector
    saved_rank=int(projectors[0].shape[1]) if projectors and projectors[0].ndim==2 else None
    rank=int(metadata.get('opd_rank',dc.get('opd_rank',saved_rank or 8)))
    if not 1<=rank<=min(64,h):raise ValueError('invalid OPD projector rank')
    projector=projectors[0].float().contiguous() if projectors else initialize_projector(h,rank,head=cleaned['lm_head.weight'])
    if projector.shape!=(h,rank) or not torch.isfinite(projector).all():raise ValueError('incompatible/nonfinite OPD projector')
    if projectors:
        trained=bool(metadata.get('opd_projector_trained',dc.get('opd_projector_trained',False)))
        provenance=metadata.get('opd_projector_provenance',dc.get('opd_projector_provenance'))
        if provenance is None:
            # Compare against the deterministic initialization; a saved tensor
            # alone is not evidence that its optimizer ever ran.
            basis=initialize_projector(h,rank,head=cleaned['lm_head.weight'])
            provenance='trained' if trained else ('head_basis_initialized' if torch.equal(projector,basis) else 'checkpoint_training_unverified')
        if provenance=='trained':trained=True
    else:trained=False;provenance='head_basis_initialized'
    cfg.update(opd_rank=rank,opd_projector_provenance=provenance,opd_projector_trained=trained)
    # Original SGLang loader accepts model.* midlayer/fc/norm plus lm_head.
    converted={('model.'+name if name not in ('lm_head.weight','d2t','t2d') else name):value.contiguous()
               for name,value in cleaned.items() if name not in ('d2t','t2d')}
    converted.update(d2t=d2t.contiguous(),t2d=t2d.contiguous())
    manifest=dict(source_checkpoint=str(checkpoint),source_config=str(config),source_mapping=str(mapping),
        target_model=str(target),sha256={str(x):sha(x) for x in (checkpoint,config,mapping,target_file)},
        feature_layer_conversion='config IDs unchanged; SGLang set_eagle3_layers_to_capture internally adds +1',
        target_embedding_source='runtime target checkpoint; upstream set_embed',
        vocab_mapping_rebuilt=False,opd_rank=rank,opd_projector_provenance=provenance,
        opd_projector_trained=trained,opd_projector_sha256=hashlib.sha256(projector.numpy().tobytes()).hexdigest())
    if projector_sidecar is not None:manifest['sha256'][str(projector_sidecar)]=sha(projector_sidecar)
    output=Path(output)
    if output.exists():
        marker=output/'conversion.json'
        if marker.is_file():
            previous=json.loads(marker.read_text())
            artifact_hashes=previous.pop('export_sha256',{})
            if previous==manifest and set(artifact_hashes)=={'model.safetensors','config.json','opd_projector.pt'}:
                if all((output/name).is_file() and sha(output/name)==digest for name,digest in artifact_hashes.items()):
                    return output
                raise ValueError('exported checkpoint/config checksum mismatch; refusing corrupted runtime artifact')
        raise FileExistsError('refusing to overwrite a different/incomplete exported checkpoint: '+str(output))
    output.mkdir(parents=True)
    save_file(converted,str(output/'model.safetensors'))
    torch.save(projector,output/'opd_projector.pt')
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    manifest['export_sha256']={name:sha(output/name) for name in ('model.safetensors','config.json','opd_projector.pt')}
    (output/'conversion.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return output


def load_projector(path,hidden,rank):
    """Load checked OPD sidecar; never initialize over an exported projector."""
    import torch
    path=Path(path);cfg=json.loads((path/'config.json').read_text())
    if int(cfg.get('opd_rank',rank))!=rank:raise ValueError('OPD rank differs from exported checkpoint')
    projector_file=path/'opd_projector.pt'
    if not projector_file.is_file():
        raise FileNotFoundError('OPD projector missing; re-export pretrained draft to a NEW DRAFT_EXPORT')
    projector=torch.load(projector_file,map_location='cpu',weights_only=True)
    if projector.shape!=(hidden,rank) or projector.dtype!=torch.float32 or not torch.isfinite(projector).all():
        raise ValueError('invalid frozen persistent OPD projector')
    return projector,cfg.get('opd_projector_provenance','checkpoint_training_unverified')
