import ast,copy
from types import SimpleNamespace as NS
import pytest,torch
from test_upstream_hooks import PATCHED,PRISTINE
from test_request_reflex import create,DEVICES


def run_once_body(folder,namespace):
    tree=ast.parse((folder/'speculative/eagle_draft_extend_cuda_graph_runner.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));parent=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='capture_one_batch_size')
    child=copy.deepcopy(next(n for n in parent.body if isinstance(n,ast.FunctionDef) and n.name=='run_once'))
    exec(compile(ast.Module(body=[child],type_ignores=[]),'<actual extend graph body>','exec'),namespace);return namespace['run_once']


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('mode',['off','zero_lr','active'])
def test_actual_extend_graph_body_opd_cache_no_extra_model_forward(device,mode):
    state=create(device);slots=torch.tensor([5,1],device=device,dtype=torch.int32)
    hidden=torch.randn(2,12,device=device);raw=torch.arange(37,device=device)[None].repeat(2,1)*.001
    if mode=='active':
        state.B_fast[0]=10*(hidden[0]@state.projector);state.active_ids[0]=0;state.active_count.fill_(1);state.bitmap[0]=1
    results=[]
    for folder,plugin in [(PRISTINE,None),(PATCHED,None if mode=='off' else state)]:
        calls=[]
        def forward(*args):calls.append(1);return NS(next_token_logits=raw,hidden_states=hidden,opd_head_input=hidden)
        worker=NS(_tlt_reflex=plugin,draft_model_runner=NS(model=NS(forward=forward)))
        batch=NS(input_ids=None,positions=None,out_cache_loc=None,spec_info=NS(hidden_states=hidden))
        ns=dict(torch=torch,self=NS(eagle_worker=worker,topk=4,_tlt_valid_bs=torch.tensor(2,device=device,dtype=torch.int32)),
            forward_batch=batch,req_pool_indices=slots,set_dp_buffer_len=lambda *a:None,global_dp_buffer_len=0,num_tokens=2,fast_topk=torch.topk)
        out=run_once_body(folder,ns)();assert len(calls)==1;results.append((out.topk_p.clone(),out.topk_index.clone()))
    if mode=='active':assert not torch.equal(results[0][1],results[1][1])
    else:
        for a,b in zip(*results):torch.testing.assert_close(a,b,rtol=3e-6,atol=1e-7)
    if mode!='off':assert state.valid_cache[5,0]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual verifier prefix requires CUDA')
@pytest.mark.parametrize('greedy',[True,False])
def test_actual_verifier_teacher_accept_path_rng_reused_before_finish(greedy):
    from pathlib import Path
    slots=torch.tensor([5,1],device='cuda',dtype=torch.int32);state=create('cuda')
    h=torch.randn(2,12,device='cuda');state.propose(h@state.head.weight.T,h,slots,topk=4)
    raw=torch.randn(8,74,device='cuda');seen=[];results=[];rngs=[]
    # Spy consumes exact existing teacher/path; mathematical feedback is tested separately.
    plugin=NS(feedback=lambda teacher,path,req,*metadata,**kwargs:seen.append((teacher.clone(),path.clone(),req.clone())))
    for folder,active in [(PRISTINE,None),(PATCHED,plugin)]:
        path=folder/'speculative/eagle_info.py';tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EagleVerifyInput')
        fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='verify'))
        cut=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='accept_index' and isinstance(n.value,ast.Subscript) for t in n.targets))
        fn.body=fn.body[:cut]+[ast.parse('return predict.clone(), accept_index.clone()').body[0]]
        calls=[]
        def sampler(**kwargs):
            calls.append(1);kwargs['predicts'].zero_();kwargs['accept_index'][:,0].copy_(torch.tensor([0,4],device='cuda'));kwargs['accept_token_num'].zero_()
        ns=dict(torch=torch,F=torch.nn.functional,TREE_SPEC_KERNEL_AVAILABLE=True,verify_tree_greedy=sampler,tree_speculative_sampling_target_only=sampler,
            SIMULATE_ACC_LEN=-1,top_k_renorm_prob=lambda p,k:p,top_p_renorm_prob=lambda p,k:p,
            get_global_server_args=lambda:NS(speculative_accept_threshold_single=1,speculative_accept_threshold_acc=1))
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns)
        class Sampling(NS):
            def __len__(self):return 2
        sampling=Sampling(is_all_greedy=greedy,has_custom_logit_processor=False,penalizer_orchestrator=NS(is_required=False),
            temperatures=torch.ones(2,1,device='cuda'),top_ks=torch.full((2,),74,device='cuda'),top_ps=torch.ones(2,device='cuda'))
        class Req:
            def __init__(self):self.output_ids=[];self.grammar=None;self.spec_verify_ct=0;self.spec_accepted_tokens=0
            def check_finished(self):pass
            def finished(self):return False
        batch=NS(forward_mode=NS(is_idle=lambda:False),sampling_info=sampling,req_to_token_pool=NS(_tlt_reflex=active),req_pool_indices=slots,reqs=[Req(),Req()])
        obj=NS(retrive_index=torch.arange(8,device='cuda').view(2,4),draft_token=torch.zeros(8,device='cuda',dtype=torch.long),draft_token_num=4,spec_steps=3,
            retrive_next_token=None,retrive_next_sibling=None,opd_parents=None,opd_feedback_contexts=None)
        torch.cuda.manual_seed(91);results.append(ns['verify'](obj,batch,NS(next_token_logits=raw),None,1));rngs.append(torch.cuda.get_rng_state())
        assert len(calls)==1
    for a,b in zip(*results):assert torch.equal(a,b)
    assert torch.equal(*rngs) and len(seen)==1
    expected=raw.argmax(-1).view(2,4) if greedy else raw.softmax(-1).view(2,4,74)
    torch.testing.assert_close(seen[0][0],expected,rtol=0,atol=0)
    assert torch.equal(seen[0][2],slots)


def test_normal_graph_calls_same_draft_forward_and_waits_before_replay():
    for name in ('eagle_draft_cuda_graph_runner.py','eagle_draft_extend_cuda_graph_runner.py'):
        source=(PATCHED/'speculative'/name).read_text()
        assert source.index('wait_for_update()',source.index('    def replay('))<source.index('self.graphs[bs].replay()')
    source=(PATCHED/'speculative/eagle_draft_cuda_graph_runner.py').read_text()
    assert 'self.eagle_worker.draft_forward(forward_batch)' in source and 'forward_batch._tlt_valid_bs = self._tlt_valid_bs' in source
