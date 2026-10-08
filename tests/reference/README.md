This is the byte snapshot of `helper/tlt_generate.py` before the pending-transition,
direct-target-only and lazy-tail fixes. It is imported only by regression tests and
benchmark validation; it is never reachable through the production dispatcher.
Tests run its root-only PackedTree path against the direct path using identical
prefixes, checkpoints, sampler seeds, EOS/compaction and corrected gate schedule.
They compare emitted tokens, CUDA RNG, every target attention mask/KV forward,
and all returned hidden/token training histories.
