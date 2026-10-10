"""OPD-OFF tree storage and the same uncorrected proposal arithmetic as OPD-ON.

No projector, B, feedback state or calibration profile is initialized here.
"""
import importlib
import torch
from helper.opd_attention import AttentionWorkspace
from helper.shared_rollout import allocate_tree_buffers


class TreeWorkspace:
    enabled = False
    async_updates = False
    def __init__(self, device, batch, contexts, depth, k, *, vocab=None, proposal_topk=16):
        self.attention_workspace = AttentionWorkspace(device)
        def alloc(shape, dtype=torch.float32): return torch.empty(shape, device=device, dtype=dtype)
        allocate_tree_buffers(self, alloc, batch, contexts, depth+1, k)
        self._kernels = importlib.import_module('helper.tree_kernels') if torch.device(device).type=='cuda' else None
        self._proposal_kernels = (importlib.import_module('helper.opd_reflex_kernels')
                                  if self._kernels is not None else None)
        self._triton = importlib.import_module('triton') if self._kernels is not None else None
        self.proposal_topk = proposal_topk
        self.proposal_batch, self.proposal_contexts = batch, k
        self.proposal_layout = None
        self.B_fast = None
        if vocab is not None: self.allocate_proposals(vocab)
        self.host_sync_count = 0

    def allocate_proposals(self, vocab):
        keep = min(self.proposal_topk, vocab)
        self.proposal_layout = (vocab, keep)
        rows = self.proposal_batch * self.proposal_contexts
        device = self.attention_workspace.device
        self.proposal_values = torch.empty(rows * keep, device=device)
        self.proposal_ids = torch.empty(rows * keep, device=device, dtype=torch.long)
        self.proposal_norm = torch.empty(rows * 2, device=device)
        tiles = (vocab + 255) // 256
        self.proposal_tiles = [torch.empty(rows * tiles * (keep if i >= 2 else 1), device=device,
                                          dtype=torch.long if i == 3 else torch.float32) for i in range(4)]

    def propose(self, logits, hidden, k, vocabulary_ids, **kwargs):
        batch, contexts, vocab = logits.shape
        if self.proposal_layout is None: self.allocate_proposals(vocab)
        expected_vocab, keep = self.proposal_layout
        if vocab != expected_vocab or k > keep or batch > self.proposal_batch or contexts > self.proposal_contexts:
            raise ValueError('uncorrected proposal exceeds allocated vocabulary/K/context envelope')
        values = self.proposal_values[:batch*contexts*keep].view(batch, contexts, keep)
        ids = self.proposal_ids[:batch*contexts*keep].view(batch, contexts, keep)
        norm = self.proposal_norm[:batch*contexts*2].view(batch, contexts, 2)
        if self._proposal_kernels is not None:
            # Reuse the unchanged scan/merge, with correction disabled. Matching
            # FP32 normalization and low-ID ties prevents LR0 trajectory bias.
            kernels = self._proposal_kernels
            tiles = (vocab + 255) // 256
            maxima, sums, scores, indices = [
                pool[:batch*contexts*tiles*(keep if i >= 2 else 1)]
                for i, pool in enumerate(self.proposal_tiles)]
            # ENABLED=False is a runtime branch in the unchanged source kernel.
            # Real scratch pointers satisfy Triton's type checking; no correction
            # pointer is read, rank is0, and ROOT=False disables OPD counters.
            kernels._corrected_scan[(tiles, contexts, batch)](
                logits, values, ids, ids, 0, maxima, sums, scores, indices,
                values, values, norm, *logits.stride(), contexts, vocab, keep,
                tiles, 256, False, 0, True, False, False,
                num_warps=4, enable_fp_fusion=False)
            kernels._proposal_merge[(batch*contexts,)](
                maxima, sums, scores, indices, values, ids, norm, contexts, vocab, keep, tiles,
                self._triton.next_power_of_2(tiles), self._triton.next_power_of_2(tiles*keep),
                num_warps=4, enable_fp_fusion=False)
        else:
            raw = logits.float()
            order = torch.argsort(raw, descending=True, stable=True)[..., :keep]
            ids.copy_(order); values.copy_(raw.softmax(-1).gather(-1, order))
        # Mapped token IDs are owned: the next propose reuses the raw ID scratch.
        return values[..., :k], None, ids[..., :k].clone()

    def begin(self, name): return None
    def end(self, ticket): pass
    def finish(self): return {'opd_backend': 'off'}
    def clear(self): pass
