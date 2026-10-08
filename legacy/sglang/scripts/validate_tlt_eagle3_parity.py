#!/usr/bin/env python3
"""Real checkpoint: native SGLang prefill vs SpecNaacl/SpecForge replay.

Uses separate Python processes/environments. A fixture or missing runtime never
produces a passing production report. Source and target checkpoints stay read-only.
"""
import argparse,json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('target','draft','source-checkpoint','source-config','mapping','report'):p.add_argument('--'+key,required=True)
    p.add_argument('--source-root',default=str(ROOT.parent/'SpecNaacl'))
    p.add_argument('--source-python',default=str(ROOT.parent/'SpecNaacl/.venv/bin/python'))
    p.add_argument('--allow-untrained-projector',action='store_true')
    p.add_argument('--head-atol',type=float,default=.02);p.add_argument('--logits-atol',type=float,default=.05)
    p.add_argument('--u-atol',type=float,default=.02);p.add_argument('--min-top16-agreement',type=float,default=1.)
    p.add_argument('--corrected-atol',type=float,default=.05);p.add_argument('--probability-atol',type=float,default=2e-4)
    p.add_argument('--force',action='store_true');p.add_argument('--stage',choices=['native','source']);p.add_argument('--packet')
    return p.parse_args(argv)


def stage_native(a):
    from tlt_reflex.runtime import configure,require_runtime
    os.environ['OPD_EAGLE3_PARITY_CAPTURE']=a.packet
    os.environ['OPD_REQUIRE_CALIBRATED_PROFILE']='0';os.environ['OPD_FAST_LR']='0';os.environ['OPD_PROFILE']='0';os.environ['OPD_PROPOSAL_MODE']='fused'
    os.environ['OPD_ALLOW_UNTRAINED_PROJECTOR']='1' if a.allow_untrained_projector else '0'
    os.environ.pop('OPD_PROPOSAL_PROFILE',None);configure('tlt_opd_reflex');sg=require_runtime()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.target,local_files_only=True)
    ids=tok.encode('Compute 7 + 5 and explain briefly.')
    engine=sg.Engine(model_path=a.target,speculative_algorithm='EAGLE3',speculative_draft_model_path=a.draft,
        speculative_num_steps=4,speculative_eagle_topk=4,speculative_num_draft_tokens=12,
        speculative_eagle_mab_configs=[],dtype='bfloat16',tp_size=1,disable_overlap_schedule=True,
        disable_cuda_graph=True,disable_radix_cache=True,max_running_requests=1,attention_backend='triton',random_seed=42,
        mem_fraction_static=.6,context_length=512)
    try:engine.generate(input_ids=[ids],sampling_params=dict(temperature=0,max_new_tokens=1))
    finally:engine.shutdown()
    if not Path(a.packet).is_file():raise RuntimeError('native draft prefill was not captured; validation cannot pass')


def stage_source(a):
    import torch
    # Purge inherited TLT module selection; SpecNaacl owns its native dependencies.
    source_root=Path(a.source_root).resolve();sys.path[:0]=[str(source_root),str(source_root/'third_party/SpecForge')]
    from helper.eagle3_specforge import Eagle3FastGRPOAdapter
    from types import SimpleNamespace as NS
    from safetensors import safe_open
    packet=torch.load(a.packet,map_location='cpu',weights_only=True)
    target=Path(a.target);config=json.loads((target/'config.json').read_text())
    embedding=None
    for file in sorted(target.glob('*.safetensors')):
        with safe_open(str(file),framework='pt',device='cpu') as f:
            if 'model.embed_tokens.weight' in f.keys():embedding=f.get_tensor('model.embed_tokens.weight');break
    if embedding is None:raise ValueError('target safetensors embedding not found; no random embedding fallback')
    class Target(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embedding=torch.nn.Embedding.from_pretrained(embedding.to(torch.bfloat16));self.config=NS(**config);self.dtype=torch.bfloat16
        def get_input_embeddings(self):return self.embedding
    adapter=Eagle3FastGRPOAdapter(Target(),a.source_config,a.source_checkpoint,a.mapping,opd_rank=packet['projector'].shape[1]).cuda().eval()
    adapter.load_opd_projector(packet['projector'])
    with torch.inference_mode():
        outputs=adapter(packet['aux_inputs'][None].cuda(),packet['input_ids'][None].cuda(),position_ids=packet['positions'][None].cuda())
        raw,head=adapter.compute_compact_logits_with_inputs(outputs['hidden_states'][:,-1,:])
        # Run the ACTUAL SpecNaacl OPD proposal math from its own environment.
        os.environ.pop('OPD_PROPOSAL_PROFILE',None)
        os.environ['OPD_PROPOSAL_MODE']='dense';os.environ['OPD_DENSE_IMPLEMENTATION']='fused'
        from helper.opd_reflex import OPDReflex
        mapping=adapter.compact_to_target_ids(device='cuda')
        opd=OPDReflex(rank=packet['projector'].shape[1],topk=16,backend='triton')
        opd._validated_tuning=True;opd.start(adapter,1,mapping,head.shape[-1],max_contexts=1,max_nodes=1,max_path=1,max_proposal_contexts=1)
        opd.B_fast.copy_(packet['B'].cuda());active=(opd.B_fast.abs().sum(-1)>0).nonzero().flatten()
        opd.active_ids[:active.numel()]=active;opd.active_count.fill_(active.numel());opd.host_active_count=active.numel();opd._ever_updated=True
        bits=torch.zeros_like(opd.bitmap,device='cpu')
        for token in active.cpu().tolist():bits[token//32]|=torch.tensor(1<<(token%32),dtype=torch.int64).to(torch.int32)
        opd.bitmap.copy_(bits)
        q,ids,_=opd.propose(raw[:,None],head[:,None],16,mapping,head_inputs=head[:,None])
        from tlt_reflex.parity import corrected_logits
        payload=dict(corrected_logits=corrected_logits(raw,opd.u_cache[:,0],opd.B_fast).cpu(),B=packet['B'],head_input=head.cpu(),raw_logits=raw.cpu(),u=opd.u_cache[:,0].cpu(),top16_ids=ids[:,0].cpu(),top16_probs=q[:,0].cpu(),projector=adapter.opd_projector.cpu())
        torch.save(payload,str(Path(a.packet).with_suffix('.source.pt')))


def main(argv=None):
    a=parse(argv)
    if a.stage=='native':return stage_native(a)
    if a.stage=='source':return stage_source(a)
    from tlt_reflex.parity import artifact_identity,require_parity_report,compare_payloads,implementation_identity
    report=Path(a.report);report.parent.mkdir(parents=True,exist_ok=True)
    if report.exists() and not a.force:
        require_parity_report(report,a.target,a.draft);print('Reuse validated real checkpoint report:',report);return
    payload=dict(validation_kind='real_checkpoint_native_sglang_vs_specnaacl',passed=False)
    try:
        # All resources must exist before allocating any model/runtime.
        for path in (a.target,a.draft,a.source_checkpoint,a.source_config,a.mapping,a.source_python):
            if not Path(path).exists():raise FileNotFoundError(path)
        payload['artifact_identity']=artifact_identity(a.target,a.draft)
        payload['implementation_identity']=implementation_identity()
        from tlt_reflex.checkpoint import sha
        source=Path(a.source_root).resolve()
        source_files=[source/'helper'/name for name in ('eagle3_specforge.py','opd_reflex.py','opd_reflex_kernels.py','tree_kernels.py')]
        source_files.append(source/'third_party/SpecForge/specforge/modeling/draft/llama3_eagle.py')
        payload['source_code_identity']={str(path):sha(path) for path in source_files}
        with tempfile.TemporaryDirectory(prefix='tlt-eagle3-parity-') as directory:
            packet=str(Path(directory)/'native.pt');args=sys.argv[1:] if argv is None else list(argv)
            env=dict(os.environ);env['PYTHONDONTWRITEBYTECODE']='1';env.pop('OPD_EAGLE3_PARITY_CAPTURE',None)
            subprocess.run([sys.executable,str(Path(__file__).resolve()),*args,'--stage','native','--packet',packet],env=env,check=True)
            env['PYTHONPATH']=str(Path(a.source_root).resolve())+os.pathsep+str(Path(a.source_root).resolve()/'third_party/SpecForge')
            subprocess.run([a.source_python,str(Path(__file__).resolve()),*args,'--stage','source','--packet',packet],env=env,check=True)
            import torch
            native=torch.load(packet,map_location='cpu',weights_only=True);source=torch.load(str(Path(packet).with_suffix('.source.pt')),map_location='cpu',weights_only=True)
            payload.update(compare_payloads(native,source,head_atol=a.head_atol,logits_atol=a.logits_atol,u_atol=a.u_atol,corrected_atol=a.corrected_atol,probability_atol=a.probability_atol,min_top16_agreement=a.min_top16_agreement))
            payload['projector_provenance']=native['projector_provenance']
            payload['tolerances']=dict(head=a.head_atol,logits=a.logits_atol,u=a.u_atol,corrected_logits=a.corrected_atol,top16_probability=a.probability_atol,top16_agreement=a.min_top16_agreement)
    except Exception as exc:
        payload['error']=f'{type(exc).__name__}: {exc}'
        report.write_text(json.dumps(payload,indent=2)+'\n');raise
    report.write_text(json.dumps(payload,indent=2)+'\n')
    if not payload['passed']:raise SystemExit('real EAGLE3 representation parity failed; benchmark blocked')
    print(json.dumps(payload,indent=2))

if __name__=='__main__':main()
