"""Post-finish feedback, bounded scratch, device dispatch and root-version contracts."""
import ast,copy,json
from pathlib import Path
from types import SimpleNamespace as NS
import torch,pytest
from tlt_reflex.state import OPDState,DeferredTreeMetadata
from tlt_reflex.ported.profiles import ProposalProfile
from test_request_reflex import create,DEVICES
from test_upstream_hooks import PATCHED,PRISTINE


@pytest.mark.skipif(not torch.cuda.is_available(),reason='native verifier prefix CUDA')
@pytest.mark.parametrize('stop',['eos','stop','max_length','finished'])
def test_real_verifier_feedback_observes_committed_finish_truncation(stop):
    # Run actual upstream verifier through the finish loop, before flatten/KV.
    slots=torch.tensor([0],device='cuda',dtype=torch.int32)
    head=torch.nn.Linear(12,37,bias=False,device='cuda')
    state=OPDState(1,head,torch.arange(37,device='cuda'),max_contexts=5,max_topk=4,max_nodes=5,max_path=5,update_stream=False)
    state.reset_slots(0,allocated=True)
    h=torch.randn(1,12,device='cuda');state.propose(h@head.weight.T,h,slots,topk=4)
    h=torch.randn(4,12,device='cuda');state.propose(h@head.weight.T,h,slots,topk=4,offset=1,expanded_ids=torch.tensor([[0,1,2,3]],device='cuda'))
    seen=[];feedback=state.feedback
    def observe(teacher,path,slots,parents,contexts,**kw):
        seen.append(path.clone());feedback(teacher,path,slots,parents,contexts,**kw)
    state.feedback=observe
    class Req:
        def __init__(self):self.output_ids=[];self.grammar=None;self.spec_verify_ct=0;self.spec_accepted_tokens=0;self.done=False
        def check_finished(self):
            self.done=(self.output_ids[-1]==9 if stop in ('eos','stop') else len(self.output_ids)>=(1 if stop=='finished' else 2))
        def finished(self):return self.done
    results=[];rng=[]
    for folder,plugin in [(PRISTINE,None),(PATCHED,state)]:
        file=folder/'speculative/eagle_info.py';cls=next(n for n in ast.parse(file.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='EagleVerifyInput')
        fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='verify'))
        cut=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='accept_index' for t in n.targets) and isinstance(n.value,ast.Subscript))
        fn.body=fn.body[:cut]+[ast.parse('return accept_index.clone(), accept_length.clone()').body[0]]
        calls=[]
        def sampler(**kwargs):
            calls.append(1);kwargs['predicts'][:5].copy_(torch.tensor([2,9,7,6,5],device='cuda'))
            kwargs['accept_index'][:].copy_(torch.tensor([[0,1,3,4,-1]],device='cuda'));kwargs['accept_token_num'].fill_(3)
        ns=dict(torch=torch,F=torch.nn.functional,TREE_SPEC_KERNEL_AVAILABLE=True,verify_tree_greedy=sampler,
            tree_speculative_sampling_target_only=sampler,SIMULATE_ACC_LEN=-1)
        mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
        exec(compile(ast.fix_missing_locations(mod),str(file),'exec'),ns)
        class Sampling(NS):
            def __len__(self):return 1
        sampling=Sampling(is_all_greedy=True,has_custom_logit_processor=False,penalizer_orchestrator=NS(is_required=False))
        req=Req();batch=NS(forward_mode=NS(is_idle=lambda:False),sampling_info=sampling,reqs=[req],req_to_token_pool=NS(_tlt_reflex=plugin),req_pool_indices=slots)
        obj=NS(retrive_index=torch.arange(5,device='cuda').view(1,5),draft_token=torch.zeros(5,device='cuda',dtype=torch.long),draft_token_num=5,spec_steps=4,
            retrive_next_token=None,retrive_next_sibling=None,
            opd_parents=torch.tensor([[-1,0,0,1,3]],device='cuda'),opd_feedback_contexts=torch.tensor([[0,1,2,3,4]],device='cuda'))
        raw=torch.randn(5,37,device='cuda');torch.cuda.manual_seed(9)
        results.append(ns['verify'](obj,batch,NS(next_token_logits=raw),None,1));rng.append(torch.cuda.get_rng_state())
        assert req.output_ids==([2] if stop=='finished' else [2,9]) and len(calls)==1
    assert seen[0].tolist()==([[0,-1,-1,-1,-1]] if stop=='finished' else [[0,1,-1,-1,-1]])
    for a,b in zip(*results):assert torch.equal(a,b)
    assert torch.equal(*rng)
    # Only states 0,1 and root rejected sibling2. No continuation frontier after terminal1.
    assert state.counters[1]==(1 if stop=='finished' else 3) and state.counters[2]==(1 if stop=='finished' else 2) and state.counters[3]==(0 if stop=='finished' else 1)


def test_persistent_slot_cache_and_feedback_scratch_have_independent_capacities():
    head=torch.nn.Linear(12,37,bias=False)
    s=OPDState(512,head,torch.arange(37),backend='torch',max_contexts=13,max_topk=4,max_nodes=48,max_path=9,max_speculative_batch_size=32)
    assert s.head_cache.shape[0]==512 and s.valid_cache.shape[0]==512
    assert s.max_feedback_rows==32*48 and s.selected_weights.numel()==32*48
    assert s.parents_workspace.numel()==32*48 and s.path_workspace.numel()==32*9 and s.slot_workspace.numel()==32
    assert s.persistent_memory_mb>0 and s.scratch_memory_mb>0


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('stream',[False,True])
def test_chunked_feedback_normalizes_once_and_reconstructs_against_preupdate_b(device,stream):
    torch.manual_seed(11);head=torch.nn.Linear(12,37,bias=False,device=device)
    states=[OPDState(11,head,torch.arange(37,device=device)*2,backend='torch' if device=='cpu' else 'triton',
        max_contexts=5,max_topk=2,max_nodes=5,max_path=4,max_speculative_batch_size=capacity,update_stream=stream) for capacity in (2,5)]
    slots=torch.tensor([7,3,10,0,5],device=device,dtype=torch.int32)
    for s in states:s.reset_slots([7,3,10,0,5],allocated=True)
    parents=torch.tensor([[-1,0,0,1,2]]*5,device=device);contexts=torch.tensor([[0,1,2,3,4]]*5,device=device)
    path=torch.tensor([[i*5,i*5+1,i*5+3,-1] for i in range(5)],device=device,dtype=torch.int32)
    for round in range(2):
        for offset,c in [(0,1),(1,2),(3,2)]:
            h=torch.randn(5*c,12,device=device);raw=h@head.weight.T
            for s in states:s.propose(raw,h,slots,topk=2,offset=offset)
        teacher=torch.randn(5,5,74,device=device).softmax(-1);teacher[2,1,states[0].mapping]=0
        for s in states:s.feedback(teacher,path,slots,parents,contexts);s.wait_for_update()
        torch.testing.assert_close(states[0].B_fast,states[1].B_fast,rtol=1e-4,atol=2e-7)
        torch.testing.assert_close(states[0].counters[:10],states[1].counters[:10],rtol=1e-5,atol=1e-5)
        assert states[0].B_version==states[1].B_version and states[0].counters[11]==round+1
    assert states[0].counters[21]==2 and states[0].selected_weights.numel()==2*5


@pytest.mark.parametrize('device',DEVICES)
def test_root_versions_reuse_exact_cache_and_only_bump_for_actual_changes(device):
    s=create(device,lr=0,stream=True);slots=torch.tensor([5,1],device=device,dtype=torch.int32)
    h=torch.randn(2,12,device=device);s.propose(h@s.head.weight.T,h,slots,topk=4)
    initial=s.q_cache[slots.long(),0].clone();ptr=s.root_q.data_ptr()
    d=NS();s.refresh_root(d,slots,4)
    torch.testing.assert_close(d.topk_p,initial[:,:4],rtol=0,atol=0)
    assert s.counters[24]==0 and s.B_version==0
    target=torch.randn(2,1,74,device=device).softmax(-1);path=torch.tensor([[0],[1]],device=device,dtype=torch.int32)
    parents=torch.full((2,1),-1,device=device,dtype=torch.long);ctx=torch.zeros((2,1),device=device,dtype=torch.long)
    s.feedback(target,path,slots,parents,ctx);s.refresh_root(d,slots,4)
    assert s.B_version==0 and s.counters[24]==0
    s.fast_lr=.01;s.feedback(target,path,slots,parents,ctx);s.refresh_root(d,slots,4)
    assert s.B_version==1 and s.counters[24]==2 and s.root_q.data_ptr()==ptr
    s.refresh_root(d,slots,4);assert s.counters[24]==2
    assert torch.equal(s.root_B_version[slots.long()],s.B_version.expand(2))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='device profile dispatch graph')
def test_graph_auto_chooses_nonmonotone_sparse_gemm_fused_regions():
    payload=dict(records=[dict(contexts=1,trials=[dict(slots=0,sparse=1,fused=4,gemm=3),dict(slots=16,sparse=5,fused=3,gemm=1),dict(slots=37,sparse=7,fused=1,gemm=4)])])
    profile=ProposalProfile(payload);head=torch.nn.Linear(12,37,bias=False,device='cuda')
    s=OPDState(1,head,torch.arange(37,device='cuda'),profile=profile,max_topk=1,max_contexts=1,max_nodes=1,max_path=1,update_stream=False)
    s.reset_slots(0,allocated=True);slots=torch.tensor([0],device='cuda',dtype=torch.int32)
    h=torch.randn(1,12,device='cuda');raw=h@head.weight.T
    st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):s.propose(raw,h,slots,topk=1)
    torch.cuda.current_stream().wait_stream(st);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):q,ids=s.propose(raw,h,slots,topk=1)
    for count in (0,16,37):
        s.active_count.fill_(count);s.active_ids[:count]=torch.arange(count,device='cuda');s.bitmap.zero_()
        bits=torch.zeros(2,dtype=torch.int32)
        for token in range(count):bits[token//32]|=torch.tensor(1<<(token%32),dtype=torch.int64).to(torch.int32)
        s.bitmap.copy_(bits);s.B_fast.zero_();s.B_fast[:count]=.03
        graph.replay()
        assert int(s.dispatch_mode)==('sparse','fused','gemm').index(profile.choose(1,count))
        expected=(raw+(h@s.projector)@s.B_fast.T).softmax(-1).topk(1)
        assert torch.equal(ids,expected.indices);torch.testing.assert_close(q,expected.values,atol=1e-7,rtol=3e-6)


def test_projector_provenance_gate_requires_explicit_untrained_opt_in():
    from tlt_reflex.checkpoint import require_projector_provenance
    require_projector_provenance('trained')
    with pytest.raises(ValueError,match='OPD_ALLOW_UNTRAINED_PROJECTOR'):require_projector_provenance('head_basis_initialized')
    with pytest.warns(RuntimeWarning,match='NOT trained'):require_projector_provenance('head_basis_initialized',allow_untrained=True)
    with pytest.raises(ValueError):require_projector_provenance('checkpoint_training_unverified',allow_untrained=True)


def test_representation_validation_rejects_wrong_head_inputs_even_if_logits_match():
    from tlt_reflex.parity import compare_payloads
    payload=dict(head_input=torch.randn(1,12),raw_logits=torch.randn(1,37),u=torch.randn(1,8),
        top16_ids=torch.arange(16)[None],top16_probs=torch.ones(1,16)/37,projector=torch.randn(12,8))
    assert compare_payloads(payload,payload)['passed']
    wrong=dict(payload,head_input=payload['head_input']*3)
    assert not compare_payloads(payload,wrong)['passed']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real version-aware root CUDA graph')
def test_root_refresh_graph_handles_mixed_versions_and_has_no_host_sync(monkeypatch):
    s=create('cuda',mode='sparse');slots=torch.tensor([5,1],device='cuda',dtype=torch.int32)
    h=torch.randn(2,12,device='cuda');raw=h@s.head.weight.T;s.propose(raw,h,slots,topk=4)
    d=NS();st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):s.refresh_root(d,slots,4)
    torch.cuda.current_stream().wait_stream(st);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):s.refresh_root(d,slots,4)
    before=s.counters[24].clone();graph.replay();assert s.counters[24]==before
    # Cache the first row with new B; the second row alone is now stale.
    s.B_fast[0]=.2*(h[0]@s.projector);s.B_version.add_(1);s.active_ids[0]=0;s.active_count.fill_(1);s.bitmap[0]=1
    s.propose(raw[:1],h[:1],slots[:1],topk=4)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *a,**k:pytest.fail('global CUDA synchronize in refresh'))
    graph.replay()
    expected=(raw+(h@s.projector)@s.B_fast.T).softmax(-1).topk(4)
    assert torch.equal(d.topk_index,expected.indices)
    torch.testing.assert_close(d.topk_p,expected.values,atol=1e-7,rtol=3e-6)
    assert s.counters[24]==before+1
    graph.replay();assert s.counters[24]==before+1


@pytest.mark.parametrize('device',DEVICES)
def test_zero_gradient_and_invalid_teacher_do_not_advance_b_version(device):
    s=create(device,stream=True);slots=torch.tensor([5,1],device=device,dtype=torch.int32)
    # u=0 exactly, even though a valid teacher produces nonzero q-p.
    h=torch.zeros(2,12,device=device);s.propose(h@s.head.weight.T,h,slots,topk=4)
    path=torch.tensor([[0],[1]],device=device,dtype=torch.int32)
    parents=torch.full((2,1),-1,device=device,dtype=torch.long);contexts=torch.zeros((2,1),device=device,dtype=torch.long)
    target=torch.randn(2,1,74,device=device).softmax(-1)
    s.feedback(target,path,slots,parents,contexts);s.wait_for_update();assert s.B_version==0 and not s.B_fast.any()
    target[...,s.mapping]=0
    s.feedback(target,path,slots,parents,contexts);s.wait_for_update();assert s.B_version==0


@pytest.mark.parametrize('hidden',[256,2048])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='native BF16 head GEMM comparison')
def test_stale_root_head_matches_native_bf16_head(hidden):
    torch.manual_seed(13);head=torch.nn.Linear(hidden,519,bias=False,device='cuda',dtype=torch.bfloat16)
    s=OPDState(2,head,torch.arange(519,device='cuda'),max_contexts=1,max_topk=1,max_nodes=1,max_path=1,update_stream=False)
    s.reset_slots([0,1],allocated=True);slots=torch.tensor([0,1],device='cuda',dtype=torch.int32)
    h=torch.randn(2,hidden,device='cuda',dtype=torch.bfloat16);raw=h@head.weight.T
    s.propose(raw.float(),h,slots,topk=1);s.B_version.add_(1)
    s.refresh_root(NS(),slots,1)
    torch.testing.assert_close(s.root_logits_workspace,raw,rtol=0,atol=.008)


def test_deferred_tree_mapping_is_bounded_and_dynamic_batch_correct():
    head=torch.nn.Linear(12,37,bias=False)
    s=OPDState(5,head,torch.arange(37),backend='torch',max_speculative_batch_size=2,max_contexts=3,max_topk=2,max_nodes=3,max_path=3)
    s.reset_slots([0,1,2,3,4],allocated=True);slots=torch.arange(5,dtype=torch.int32)
    h=torch.randn(5,12);s.propose(h@head.weight.T,h,slots,topk=2)
    h=torch.randn(10,12);s.propose(h@head.weight.T,h,slots,topk=2,offset=1,expanded_ids=torch.tensor([[0,1]]*5))
    parents,contexts=s.tree_metadata(torch.tensor([[0,1]]*5),torch.tensor([[-1,0,1]]*5),slots,2,2)
    assert isinstance(parents,DeferredTreeMetadata) and parents is contexts
    teacher=torch.randn(5,3,37).softmax(-1);path=torch.tensor([[i*3,-1,-1] for i in range(5)],dtype=torch.int32)
    s.feedback(teacher,path,slots,parents,contexts)
    assert s.counters[18]==0 and s.counters[19]==0 and s.counters[21]==1


def test_invalid_context_and_orphan_are_reported_and_debug_fails():
    head=torch.nn.Linear(12,37,bias=False)
    s=OPDState(1,head,torch.arange(37),backend='torch',max_contexts=3,max_topk=1,max_nodes=3,max_path=3,debug=True)
    s.reset_slots(0,allocated=True)
    with pytest.raises(ValueError,match='orphan'):
        s._validate_reference(torch.tensor([[-1,-2,1]]),torch.tensor([[0,1,99]]),torch.tensor([0]))
    assert s.counters[18]>0 and s.counters[19]>0


def test_pair_fairness_rejects_config_prompts_weights_profile_and_context_failures():
    import importlib.util
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('pair_contract',root/'scripts/summarize_tlt_opd.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    cfg={k:1 for k in m.REQUIRED_CONFIG};cfg['profile']=False
    base=dict(config=cfg,engine_config={},artifact_identity={'same':'weights'},prompt_token_sha256='same',measured_prompts=1,
        generated_responses=1,spot_trainer_enabled=False,valid_for_official_comparison=True,opd_orphan_nodes=0,opd_invalid_contexts=0,real_eagle3_parity={'passed':True})
    m.validate_pair(base,base)
    for key,value in [('artifact_identity',{'other':'weights'}),('prompt_token_sha256','different'),('opd_orphan_nodes',1),('opd_invalid_contexts',1)]:
        with pytest.raises(ValueError):m.validate_pair(base,dict(base,**{key:value}))
    with pytest.raises(ValueError):m.validate_pair(base,dict(base,config=dict(cfg,seed=99)))
    with pytest.raises(ValueError):m.validate_pair(base,dict(base,config=dict(cfg,profile=True)))


@pytest.mark.parametrize('device',DEVICES)
def test_production_invalid_feedback_context_is_masked_before_dereference(device):
    s=create(device);slots=torch.tensor([5],device=device,dtype=torch.int32)
    h=torch.randn(1,12,device=device);s.propose(h@s.head.weight.T,h,slots,topk=4)
    parents=torch.tensor([[-1,0,1]],device=device);contexts=torch.tensor([[0,999,-1]],device=device)
    target=torch.randn(1,3,74,device=device).softmax(-1);path=torch.tensor([[0,1,2]],device=device,dtype=torch.int32)
    s.feedback(target,path,slots,parents,contexts);s.wait_for_update()
    assert contexts[0,1]==-1 and s.counters[19]>0
    assert torch.isfinite(s.B_fast).all() and s.counters[1]==1


@pytest.mark.skipif(not torch.cuda.is_available(),reason='one-shot native recorder protocol fixture')
def test_validation_recorder_observes_one_forward_and_real_opd_inputs(tmp_path):
    from tlt_reflex.parity import install_native_recorder
    s=create('cuda');slots=torch.tensor([5],device='cuda',dtype=torch.int32)
    h=torch.randn(1,12,device='cuda');raw=h@s.head.weight.T;calls=[]
    batch=NS(batch_size=1,forward_mode=NS(is_extend=lambda:True),req_pool_indices=slots,
        input_ids=torch.tensor([1,2,3],device='cuda'),positions=torch.arange(3,device='cuda'),spec_info=NS(hidden_states=torch.randn(3,36,device='cuda')))
    def forward(*args):calls.append(1);return NS(next_token_logits=raw,opd_head_input=h)
    model=NS(forward=forward);file=tmp_path/'captured.pt';install_native_recorder(model,s,file)
    before=torch.cuda.get_rng_state();model.forward(batch.input_ids,batch.positions,batch)
    assert len(calls)==1 and torch.equal(before,torch.cuda.get_rng_state())
    payload=torch.load(file,weights_only=True)
    torch.testing.assert_close(payload['head_input'],h.cpu(),atol=0,rtol=0)
    torch.testing.assert_close(payload['u'],(h@s.projector).cpu(),atol=1e-6,rtol=1e-6)
    model.forward(batch.input_ids,batch.positions,batch);assert len(calls)==2
