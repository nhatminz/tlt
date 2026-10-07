import json
from pathlib import Path
import pytest
import torch
from tlt_reflex.checkpoint import export,validate_config


def config():
    return dict(architectures=['LlamaForCausalLMEagle3'],num_hidden_layers=1,
        hidden_size=8,target_hidden_size=8,vocab_size=19,draft_vocab_size=7,
        num_attention_heads=2,num_key_value_heads=1,intermediate_size=16,
        tie_word_embeddings=False,attention_bias=False,norm_output=False,fc_norm=False,
        eagle_config={'eagle_aux_hidden_state_layer_ids':[1,3,6]})


def test_feature_layers_and_unsupported_normalization_are_not_silently_changed():
    target=dict(vocab_size=19,hidden_size=8,num_hidden_layers=10,model_type='qwen2')
    assert validate_config(config(),target)['eagle_config']['eagle_aux_hidden_state_layer_ids']==[1,3,6]
    for flag in ['fc_norm','norm_output','attention_bias','tie_word_embeddings']:
        with pytest.raises(ValueError):validate_config(dict(config(),**{flag:True}),target)


def prepare(tmp_path):
    pytest.importorskip('safetensors')
    target=tmp_path/'target';target.mkdir()
    (target/'config.json').write_text(json.dumps(dict(vocab_size=19,hidden_size=8,num_hidden_layers=10,model_type='qwen2')))
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(config()))
    shapes={'fc.weight':(8,24),'norm.weight':(8,),'lm_head.weight':(7,8),
        'midlayer.input_layernorm.weight':(8,),'midlayer.hidden_norm.weight':(8,),
        'midlayer.post_attention_layernorm.weight':(8,),'midlayer.self_attn.q_proj.weight':(8,16),
        'midlayer.self_attn.k_proj.weight':(4,16),'midlayer.self_attn.v_proj.weight':(4,16),
        'midlayer.self_attn.o_proj.weight':(8,8),'midlayer.mlp.gate_proj.weight':(16,8),
        'midlayer.mlp.up_proj.weight':(16,8),'midlayer.mlp.down_proj.weight':(8,16)}
    weights={key:torch.randn(shape) for key,shape in shapes.items()}
    checkpoint=tmp_path/'training_state.pt';torch.save(dict(draft_state_dict=weights),checkpoint)
    ids=torch.arange(7)*2;t2d=torch.zeros(19,dtype=torch.bool);t2d[ids]=True
    mapping=tmp_path/'mapping.pt';torch.save(dict(d2t=ids-torch.arange(7),t2d=t2d),mapping)
    return checkpoint,cfg,mapping,target,weights


def test_export_load_names_weights_mapping_and_idempotence(tmp_path):
    pytest.importorskip('safetensors')
    from safetensors.torch import load_file
    ck,cfg,mapping,target,weights=prepare(tmp_path)
    out=export(ck,cfg,mapping,target,tmp_path/'exported')
    actual=load_file(str(out/'model.safetensors'))
    for key,value in weights.items():assert torch.equal(actual[key if key=='lm_head.weight' else 'model.'+key],value)
    assert torch.equal(actual['d2t'],torch.arange(7))
    assert export(ck,cfg,mapping,target,out)==out
    with pytest.raises(FileExistsError):
        (target/'config.json').write_text(json.dumps(dict(vocab_size=19,hidden_size=8,num_hidden_layers=11,model_type='qwen2')))
        export(ck,cfg,mapping,target,out)


@pytest.mark.parametrize('defect',['missing','unknown','mapping'])
def test_incomplete_or_wrong_pretrained_weights_fail_not_random_fallback(tmp_path,defect):
    ck,cfg,mapping,target,weights=prepare(tmp_path)
    if defect=='missing':weights.pop('fc.weight')
    elif defect=='unknown':weights['not_a_real_parameter']=torch.ones(1)
    else:weights['d2t']=torch.zeros(7,dtype=torch.long)
    torch.save(weights,ck)
    with pytest.raises(ValueError):export(ck,cfg,mapping,target,tmp_path/'bad')


def test_corrupted_runtime_export_is_not_reused(tmp_path):
    ck,cfg,mapping,target,_=prepare(tmp_path)
    out=export(ck,cfg,mapping,target,tmp_path/'exported')
    (out/'model.safetensors').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checksum'):export(ck,cfg,mapping,target,out)


def test_trained_projector_export_load_roundtrip_no_overwrite(tmp_path):
    from tlt_reflex.checkpoint import load_projector
    ck,cfg,mapping,target,weights=prepare(tmp_path)
    projector=torch.randn(8,3);weights['opd_projector']=projector
    torch.save(dict(draft_state_dict=weights,opd_projector=projector,metadata=dict(opd_rank=3,opd_projector_trained=True)),ck)
    out=export(ck,cfg,mapping,target,tmp_path/'opd')
    actual,provenance=load_projector(out,8,3)
    assert torch.equal(actual,projector) and provenance=='trained'
    config=json.loads((out/'config.json').read_text());assert config['opd_rank']==3 and config['opd_projector_trained']
    from safetensors.torch import load_file
    assert torch.equal(load_file(str(out/'model.safetensors'))['lm_head.weight'],weights['lm_head.weight'])
    with pytest.raises(ValueError,match='rank'):load_projector(out,8,8)


def test_head_basis_initialized_is_labelled_not_learned(tmp_path):
    from tlt_reflex.checkpoint import load_projector
    from tlt_reflex.ported.reference import initialize_projector
    ck,cfg,mapping,target,weights=prepare(tmp_path);out=export(ck,cfg,mapping,target,tmp_path/'basis')
    a,provenance=load_projector(out,8,8)
    assert provenance=='head_basis_initialized'
    assert torch.equal(a,initialize_projector(8,8,head=weights['lm_head.weight']))
    assert not json.loads((out/'config.json').read_text())['opd_projector_trained']


def test_projector_sidecar_preserved_and_conflicting_copies_rejected(tmp_path):
    ck,cfg,mapping,target,weights=prepare(tmp_path)
    a=torch.randn(8,8);torch.save(a,ck.parent/'opd_projector.pt')
    out=export(ck,cfg,mapping,target,tmp_path/'sidecar')
    assert torch.equal(torch.load(out/'opd_projector.pt',weights_only=True),a)
    weights['opd_projector']=a+1;torch.save(dict(draft_state_dict=weights),ck)
    with pytest.raises(ValueError,match='disagree'):export(ck,cfg,mapping,target,tmp_path/'bad')
