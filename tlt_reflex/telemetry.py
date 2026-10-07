"""Opt-in CUDA event timing outside graphs, reported via existing info RPC."""
from contextlib import contextmanager
import torch


class Meter:
    def __init__(self, enabled=False):
        self.enabled=enabled
        self.pending=[]
        self.totals={}
        self.counters=dict(sequence_verification_rounds=0,accepted_draft_tokens=0,proposed_draft_tokens=0)
        self.reflex_state_memory_mb=0.
        self.reflex_buffer_memory_mb=0.
    def begin(self,key):
        if self.enabled and not torch.cuda.is_current_stream_capturing():
            event=torch.cuda.Event(enable_timing=True);event.record();return key,event
        return None
    def end(self,ticket):
        if ticket is not None:
            event=torch.cuda.Event(enable_timing=True);event.record()
            self.pending.append((ticket[0],ticket[1],event))
    @contextmanager
    def section(self,key):
        active=self.enabled and not torch.cuda.is_current_stream_capturing()
        if active:
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record()
        yield
        if active:
            b.record(); self.pending.append((key,a,b))
    def verified(self,accepted,proposed):
        # Uses verifier's ALREADY materialized CPU list; no extra transfer.
        self.counters['sequence_verification_rounds']+=len(accepted)
        self.counters['accepted_draft_tokens']+=sum(accepted)
        self.counters['proposed_draft_tokens']+=len(accepted)*proposed
    def report(self):
        # Explicit non-hot-path get_server_info() RPC, never per-token logging.
        for key,a,b in self.pending:
            b.synchronize()
            self.totals[key]=self.totals.get(key,0.)+a.elapsed_time(b)
        self.pending.clear()
        opd=getattr(self,"opd",None)
        opd_report=opd.report() if opd is not None else {}
        self.counters.update({k:v for k,v in opd_report.items() if isinstance(v,(int,float)) and not isinstance(v,bool)})
        return dict(opd_metadata=opd_report,times_ms=self.totals.copy(),counters=self.counters.copy(),
                    reflex_state_memory_mb=self.reflex_state_memory_mb,
                    reflex_buffer_memory_mb=self.reflex_buffer_memory_mb,
                    profile_enabled=self.enabled,graph_inner_times_available=False,
                    gpu_memory=dict(allocated_gb=torch.cuda.memory_allocated()/2**30,
                        reserved_gb=torch.cuda.memory_reserved()/2**30,
                        peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30,
                        peak_reserved_gb=torch.cuda.max_memory_reserved()/2**30))
