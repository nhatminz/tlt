import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import torch
import pytest
from test_tlt_fastgrpo import tiny
from helper.fastgrpo_training import training_draft_model
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA')
@pytest.mark.parametrize('accumulation',[1,2])
def test_online_draft_loss_gradients_and_update_match_source(accumulation):
    s=(ROOT/'sources/FastGRPO/grpo_speculative.py').read_text();tree=ast.parse(s)
    f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='training_draft_model')
    scope=dict(torch=torch,repeated_generate_nums=2,max_training_token=12,max_training_padding_gap=4,draft_accumulation_steps=accumulation)
    exec(compile(ast.Module(body=[f],type_ignores=[]),'source','exec'),scope)
    model=tiny();other=deepcopy(model)
    gen=torch.Generator(device='cuda').manual_seed(56)
    outputs=dict(all_draft_input_states=[torch.randn(n,32,device='cuda',dtype=torch.bfloat16,generator=gen) for n in [7,9,11,13]],
                 all_draft_input_ids=[torch.randint(0,97,(n,),device='cuda',generator=gen) for n in [7,9,11,13]])
    mask=torch.ones(2,3,dtype=torch.long)
    a=scope['training_draft_model'](model,outputs,mask)
    b=training_draft_model(other,outputs,mask,repeated_generate_nums=2,max_training_token=12,max_training_padding_gap=4,draft_accumulation_steps=accumulation)
    assert a==b
    for x,y in zip(model.draft_model.parameters(),other.draft_model.parameters()):assert torch.equal(x.grad,y.grad)
    for m in (model,other):torch.optim.AdamW(m.draft_model.parameters(),lr=1e-4).step()
    for x,y in zip(model.draft_model.parameters(),other.draft_model.parameters()):assert torch.equal(x,y)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA')
def test_pretrain_loss_and_gradient_match_original_slices_masks_normalization():
    from train_draft import pretrain_loss
    source=(ROOT/'sources/FastGRPO/train_draft.py').read_text()
    start=source.index('        with torch.no_grad():\n            target_outputs=')
    end=source.index('        if torch.isnan(loss)',start)
    import textwrap
    body=textwrap.dedent(source[start:end])
    wrapper='def reference(model,batch):\n    input_ids=batch["input_ids"].cuda()\n    attention_mask=batch["attention_mask"].cuda()\n    loss_mask=batch["loss_mask"].cuda()\n    l1_loss=torch.nn.SmoothL1Loss(reduction="none")\n'+textwrap.indent(body,'    ')+'    return loss1,loss2\n'
    scope={'torch':torch};exec(wrapper,scope)
    a=tiny();b=deepcopy(a)
    batch=dict(input_ids=torch.tensor([[2,3,4,7,8,0,0],[6,8,3,9,11,12,13]]),
        attention_mask=torch.tensor([[1,1,1,1,1,0,0],[1,1,1,1,1,1,1]]),
        loss_mask=torch.tensor([[0,0,1,1,1,0,0],[0,0,0,1,1,1,1]]))
    x=scope['reference'](a,batch);y=pretrain_loss(b,batch)
    for u,v in zip(x,y):assert torch.equal(u,v)
    sum(x).backward();sum(y).backward()
    for u,v in zip(a.draft_model.parameters(),b.draft_model.parameters()):assert torch.equal(u.grad,v.grad)
    for model in (a,b):torch.optim.AdamW(model.parameters(),lr=5e-5).step()
    for u,v in zip(a.draft_model.parameters(),b.draft_model.parameters()):assert torch.equal(u,v)


def test_pretrain_collation_matches_original_sharegpt_mask():
    from helper.pretrain_data import DataCollator
    source=(ROOT/'sources/FastGRPO/train_draft.py').read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='DataCollator')
    scope={'torch':torch,'model_type':'qwen2'}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'source-collator','exec'),scope)
    tokenizer=SimpleNamespace(encode=lambda text,**kw:[ord(c)%97 for c in text],eos_token_id=96)
    batch=[{'conversations':[{'from':'human','value':'What is 2+2?'},{'from':'gpt','value':'4'}]},
           {'conversations':[{'from':'human','value':'Short'},{'from':'gpt','value':'Longer answer here'}]}]
    a=scope['DataCollator'](tokenizer,max_length=200)(batch)
    b=DataCollator(tokenizer,max_length=200)(batch)
    for key in a:assert torch.equal(a[key],b[key])

