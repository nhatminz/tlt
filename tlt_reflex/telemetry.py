"""Opt-in CUDA event timing outside graphs, reported via existing info RPC."""
from contextlib import contextmanager
import torch


class Meter:
    def __init__(self, enabled=False):
        self.enabled=enabled
        self.pending=[]
        self.totals={}
        self.counters=dict(sequence_verification_rounds=0,accepted_draft_tokens=0,proposed_draft_tokens=0)
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
        return dict(times_ms=self.totals.copy(),counters=self.counters.copy(),
                    profile_enabled=self.enabled,graph_inner_times_available=False,
                    gpu_memory=dict(allocated_gb=torch.cuda.memory_allocated()/2**30,
                        reserved_gb=torch.cuda.memory_reserved()/2**30,
                        peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30))
