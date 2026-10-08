"""Tree workspace for OPD-OFF. Contains no projector, B, profile or OPD kernels."""
import importlib
import torch
from helper.opd_attention import AttentionWorkspace
from helper.shared_rollout import allocate_tree_buffers


class TreeWorkspace:
    enabled = False
    async_updates = False
    def __init__(self, device, batch, contexts, depth, k):
        self.attention_workspace = AttentionWorkspace(device)
        def alloc(shape, dtype=torch.float32): return torch.empty(shape, device=device, dtype=dtype)
        allocate_tree_buffers(self, alloc, batch, contexts, depth+1, k)
        self._kernels = importlib.import_module('helper.tree_kernels') if torch.device(device).type=='cuda' else None
        self.host_sync_count = 0

    def propose(self, logits, hidden, k, vocabulary_ids, **kwargs):
        values, ids = torch.topk(logits.softmax(-1), k, dim=-1)
        return values, None, ids

    def begin(self, name): return None
    def end(self, ticket): pass
    def finish(self): return {'opd_backend': 'off'}
    def clear(self): pass
