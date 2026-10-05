"""Run the real extend-graph proposal body and verifier prefix with NN/kernel stubs.

Native SGLang end-to-end smoke remains separate: these tests prove hook placement,
distribution/metadata reuse and no extra forward, not native kernel equivalence.
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import torch
from tlt_reflex.state import RequestReflex
from test_upstream_hooks import PATCHED,PRISTINE

DEVICES=['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])


def run_once_body(folder,namespace):
    tree=ast.parse((folder/'speculative/eagle_draft_extend_cuda_graph_runner.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    parent=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='capture_one_batch_size')
    child=copy.deepcopy(next(n for n in parent.body if isinstance(n,ast.FunctionDef) and n.name=='run_once'))
    exec(compile(ast.Module(body=[child],type_ignores=[]),'<real draft-extend graph body>','exec'),namespace)
    return namespace['run_once']


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('mode',['off','zero_lr','active'])
def test_real_draft_extend_graph_body_correction_precedes_softmax_topk(device,mode):
    state=RequestReflex(4,19,12,torch.arange(19,device=device),backend='triton' if device!='cpu' else 'torch')
    state.reset_slots([2,0],allocated=True)
    slots=torch.tensor([2,0],device=device,dtype=torch.int32)
    hidden=torch.randn(2,12,device=device)
    raw=torch.arange(19,device=device,dtype=torch.float32)[None].repeat(2,1)*.001
    if mode=='active':
        psi=torch.nn.functional.normalize(hidden@state.projection,dim=-1,eps=1e-6)
        state.a[slots.long(),0]=psi*10
    elif mode=='zero_lr':state.lr=0
    outputs=[]
    for folder,plugin in [(PRISTINE,None),(PATCHED,None if mode=='off' else state)]:
        calls=[]
        def forward(*args):
            calls.append(1)
            return NS(next_token_logits=raw,hidden_states=hidden)
        worker=NS(_tlt_reflex=plugin,draft_model_runner=NS(model=NS(forward=forward)))
        batch=NS(input_ids=None,positions=None,out_cache_loc=None,spec_info=NS(hidden_states=hidden))
        ns=dict(torch=torch,self=NS(eagle_worker=worker,topk=4,_tlt_valid_bs=torch.tensor(2,device=device,dtype=torch.int32)),
            forward_batch=batch,req_pool_indices=slots,set_dp_buffer_len=lambda *args:None,
            global_dp_buffer_len=0,num_tokens=2,fast_topk=torch.topk)
        out=run_once_body(folder,ns)()
        assert len(calls)==1,'hook must not add NN forward'
        outputs.append((out.topk_p.clone(),out.topk_index.clone()))
    if mode=='active':
        assert torch.all(outputs[1][1][:,0]==0), 'active Reflex was bypassed in graph-extend proposal'
        assert not torch.equal(outputs[0][1],outputs[1][1])
        assert state.cached[slots.long()].all()
    else:
        for a,b in zip(*outputs):assert torch.equal(a,b)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual verifier uses CUDA buffers')
@pytest.mark.parametrize('greedy',[True,False])
def test_real_verifier_prefix_reuses_teacher_and_exact_root_indices_without_new_forward(greedy):
    slots=torch.tensor([5,1,3],device='cuda',dtype=torch.int32)
    state=RequestReflex(7,19,12,torch.arange(19,device='cuda')*2)
    state.reset_slots([5,1,3],allocated=True)
    q=state.correct(torch.randn(3,19,device='cuda'),torch.randn(3,12,device='cuda'),slots,root=True).softmax(-1)
    state.cache_root(q,slots)
    raw=torch.randn(12,38,device='cuda');teacher_before=raw.clone()
    roots=torch.tensor([1,8,4],device='cuda',dtype=torch.int32)
    outputs=[];seen=[]
    original_feedback=state.feedback
    def feedback(teacher,indices,requests,**kwargs):
        seen.append((teacher.clone(),indices.clone(),requests.clone()))
        original_feedback(teacher,indices,requests,**kwargs)
    state.feedback=feedback
    for folder,plugin in [(PRISTINE,None),(PATCHED,state)]:
        path=folder/'speculative/eagle_info.py';tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EagleVerifyInput')
        fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='verify'))
        cut=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='unfinished_index' for t in n.targets))
        fn.body=fn.body[:cut]+[ast.parse('return predict.clone(), accept_index.clone()').body[0]]
        sampler_calls=[]
        def kernel(**kwargs):
            sampler_calls.append(1)
            kwargs['predicts'].zero_()
            pred=kwargs.get('target_predict')
            if pred is None:pred=kwargs['target_probs'].argmax(-1)
            kwargs['predicts'][:12].copy_(pred.flatten())
            kwargs['accept_index'][:,0].copy_(roots)
            kwargs['accept_token_num'].fill_(0)
        ns=dict(torch=torch,F=torch.nn.functional,Optional=object,TREE_SPEC_KERNEL_AVAILABLE=True,
            verify_tree_greedy=kernel,tree_speculative_sampling_target_only=kernel,SIMULATE_ACC_LEN=-1,
            top_k_renorm_prob=lambda p,k:p,top_p_renorm_prob=lambda p,k:p,
            get_global_server_args=lambda:NS(speculative_accept_threshold_single=1,speculative_accept_threshold_acc=1))
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns)
        class Sampling(NS):
            def __len__(self):return 3
        sampling=Sampling(is_all_greedy=greedy,has_custom_logit_processor=False,penalizer_orchestrator=NS(is_required=False),
            temperatures=torch.ones(3,1,device='cuda'),top_ks=torch.full((3,),38,device='cuda'),top_ps=torch.ones(3,device='cuda'))
        batch=NS(forward_mode=NS(is_idle=lambda:False),sampling_info=sampling,req_to_token_pool=NS(_tlt_reflex=plugin),req_pool_indices=slots)
        obj=NS(retrive_index=torch.arange(12,device='cuda').view(3,4),draft_token=torch.zeros(12,device='cuda',dtype=torch.long),
            draft_token_num=4,spec_steps=3,retrive_next_token=None,retrive_next_sibling=None)
        torch.cuda.manual_seed(91)
        outputs.append(ns['verify'](obj,batch,NS(next_token_logits=raw),None,1))
        assert len(sampler_calls)==1
        assert torch.equal(raw,teacher_before),'teacher/verifier distribution mutated by Reflex'
    for a,b in zip(*outputs):assert torch.equal(a,b)
    assert len(seen)==1 and torch.equal(seen[0][1],roots) and torch.equal(seen[0][2],slots)
    expected=raw.argmax(-1).view(3,4) if greedy else raw.softmax(-1).view(3,4,38)
    torch.testing.assert_close(seen[0][0],expected,rtol=0,atol=0)
