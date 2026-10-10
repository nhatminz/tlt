"""Early SD, bounded arm selection, exact source matching, and changing tree layouts."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from helper.fastgrpo_generate import get_adaptive_hyperparameters
from helper.tlt_scheduler import TLTConfig, TLTScheduler
from scripts.benchmark_tlt_opd import validate_opd_experiment

BATCHES = (128, 64, 32, 16, 8, 4, 2, 1)
MODES = ('budget_aware_beg', 'fastgrpo_matched')
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='real CUDA rollout regression')


@pytest.mark.parametrize('batch', BATCHES)
@pytest.mark.parametrize('mode', MODES)
def test_early_transition_and_budget_at_every_live_batch(batch, mode):
    scheduler = TLTScheduler(TLTConfig(bs_threshold=batch, warmup_checks=1, scheduling_mode=mode))
    assert scheduler.start_rollout(batch, 512) == 512
    assert scheduler.select(batch) is None
    assert scheduler.gate.pending and not scheduler.gate.enabled
    scheduler.record([1] * batch, .01)
    scheduler.gate.complete_transition()
    strategy = scheduler.select(batch)
    assert batch * strategy.verification_num <= 512
    assert strategy.total_draft <= strategy.candidate_nodes
    if mode == 'fastgrpo_matched':
        depth, k, draft = get_adaptive_hyperparameters(batch, 512, 5, 8, 160, 3, .75)
        assert (strategy.depth, strategy.k, strategy.total_draft) == (depth, k, draft)
    else:
        assert strategy in scheduler.strategies  # An arm is never silently resized.
    assert scheduler.current['actual_depth'] == strategy.depth
    assert scheduler.current['actual_k'] == strategy.k
    assert scheduler.current['verification_tokens'] == batch * strategy.verification_num


def test_beg_filters_infeasible_preferred_bucket_without_changing_rewards_or_rng():
    cfg = TLTConfig(bs_threshold=128, strategies='5_8_160,5_8_64,4_8_16,3_3_4', buckets=(1, 64, 96, 128))
    scheduler = TLTScheduler(cfg)
    scheduler.start_rollout(128, 512)
    scheduler.gate.check(128); scheduler.gate.complete_transition()
    before = np.random.get_state()
    # Batch64's preferred64-token bucket cannot fit; explicit4-token arm can.
    chosen = scheduler.select(64)
    assert chosen.name == '3_3_4'
    scheduler.record([1] * 64, .01)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert scheduler.trace[-1]['reward'] == 0.  # Preserve upstream initial stable-AAL reward.


def test_no_feasible_arm_fails_without_clamping():
    cfg = TLTConfig(strategies='5_8_160', buckets=(1,))
    scheduler = TLTScheduler(cfg); scheduler.start_rollout(128, 512)
    scheduler.gate.check(128); scheduler.gate.complete_transition()
    with pytest.raises(ValueError, match='no strategy clamping'):
        scheduler.select(128)


def test_beg_exploration_only_visits_explicit_feasible_arms():
    cfg = TLTConfig(strategies='5_8_160,4_8_160,5_8_64,4_8_64,4_8_16,3_8_16,3_3_4,3_2_4', buckets=(1,2,5,21))
    scheduler = TLTScheduler(cfg); scheduler.start_rollout(128, 512)
    scheduler.gate.check(128); scheduler.gate.complete_transition()
    seen = set()
    for _ in range(100):
        arm = scheduler.select(128); seen.add(arm.name)
        assert arm.name in ('3_3_4', '3_2_4')
        scheduler.record([1] * 128, .01)
    assert seen == {'3_3_4', '3_2_4'}


def test_matched_mode_honors_rollout_limits_without_mab_override():
    limits = dict(max_draft_token_length=4, max_draft_k=4, max_verification_num=40,
                  min_draft_token_length=2, draft_token_length_c=.5)
    scheduler = TLTScheduler(TLTConfig(scheduling_mode='fastgrpo_matched'))
    scheduler.start_rollout(128, 512, **limits)
    scheduler.gate.check(128); scheduler.gate.complete_transition()
    for batch in BATCHES:
        arm = scheduler.select(batch)
        expected = get_adaptive_hyperparameters(batch, 512, **limits)
        assert (arm.depth, arm.k, arm.total_draft) == expected


@pytest.mark.parametrize('mode', MODES)
def test_dynamic_strategy_envelope_and_resume(mode):
    cfg = TLTConfig(scheduling_mode=mode)
    scheduler = TLTScheduler(cfg); scheduler.start_rollout(128, 512)
    scheduler.gate.check(128); scheduler.gate.complete_transition()
    for batch in BATCHES:
        arm = scheduler.select(batch)
        assert arm.depth <= scheduler.workspace_depth and arm.k <= scheduler.workspace_k
        scheduler.record([1] * batch, .01)
    restored = TLTScheduler(cfg); restored.load_state_dict(scheduler.state_dict())
    assert restored.select(2) == scheduler.select(2)
    assert restored.capacity == 512


@pytest.mark.parametrize('output', [dict(), dict(tlt_speculative_rounds=1),
    dict(tlt_speculative_rounds=1, opd_feedback_calls=1, opd_selected_states=0)])
def test_empty_opd_experiment_is_rejected(output):
    with pytest.raises(ValueError, match='invalid OPD experiment'):
        validate_opd_experiment(output, 'tlt_opd_reflex')
    validate_opd_experiment(output, 'tlt')


def test_zero_lr_ablation_still_requires_feedback_but_not_updates():
    validate_opd_experiment(dict(tlt_speculative_rounds=1, opd_feedback_calls=1,
                                 opd_selected_states=1, opd_updates=0), 'tlt_opd_reflex')


@CUDA
@pytest.mark.parametrize('batch,k', [(128, 3), (64, 7), (32, 8)])
@pytest.mark.parametrize('ties', [False, True])
def test_uncorrected_proposal_matches_zero_b_in_fp32_probabilities_and_low_id_ties(batch, k, ties):
    from tests.test_tlt_fastgrpo import tiny
    from helper.opd_reflex import OPDReflex
    from helper.tlt_workspace import TreeWorkspace
    model = tiny(); model.enable_opd(8)
    mapping = model.full_vocabulary_ids
    adapter = OPDReflex(8, 16, fast_lr=0, backend='triton')
    adapter.start(model, batch, mapping, 32, max_contexts=33, max_nodes=512, max_path=6, max_proposal_contexts=8)
    plain = TreeWorkspace('cuda', batch, 33, 5, 8, vocab=97, proposal_topk=16)
    logits = torch.zeros(batch, 1, 97, device='cuda', dtype=torch.bfloat16) if ties else torch.randn(
        batch, 1, 194, device='cuda', dtype=torch.bfloat16)[..., ::2]
    hidden = torch.randn(batch, 1, 32, device='cuda', dtype=torch.bfloat16)
    a, _, ids_a = plain.propose(logits, hidden, k, None, root=True)
    b, _, ids_b = adapter.propose(logits, hidden, k, mapping, root=True)
    assert torch.equal(a, b) and torch.equal(ids_a, ids_b)
    assert plain.B_fast is None and not hasattr(plain, 'projector')
    if ties: assert torch.equal(ids_a[0, 0], torch.arange(k, device='cuda'))


@CUDA
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('fast_lr', [0., .01])
def test_changing_depth_k_batch_preserves_workspace_transition_kv_and_opd(monkeypatch, mode, fast_lr):
    from tests.test_tlt_fastgrpo import tiny
    from helper import tlt_generate as runtime, opd_sampling
    native_sample = runtime.sample_target_with_metadata
    native_prefill = runtime.prefill_draft_prefix
    cfg = TLTConfig(bs_threshold=128, warmup_checks=1, scheduling_mode=mode)
    answers = []
    for method in ('tlt', 'tlt_opd_reflex'):
        model = tiny(); scheduler = TLTScheduler(cfg)
        if method == 'tlt_opd_reflex': model.enable_opd(8)
        state = dict(samples=0, target=0, prefill=0)
        caches = []
        def sampled(logits, **kwargs):
            # A finite test distribution finishes exactly half the live rows per
            # speculative round, exercising128->...->1 in a single shared-B rollout.
            state['samples'] += 1
            logits = logits.clone(); logits[..., 96] = -100
            if getattr(scheduler, 'current', {}).get('phase') == 'speculative':
                if logits.shape[0] == 1:
                    logits[0, 0, 96] = 100
                else:
                    logits[1::2, 0, 96] = 100
            return native_sample(logits, **kwargs)
        def prefill(*args, **kwargs):
            state['prefill'] += 1
            before = state['target']
            value = native_prefill(*args, **kwargs)
            assert state['target'] == before  # draft-only transition
            return value
        def observe(module, args, kwargs):
            state['target'] += 1
            # Compare committed, unmasked prefix KV before adding a provisional
            # tree. Unaccepted nodes and padded cache slots are not semantic KV.
            pool = model._opd_target_kv_pool
            prefix = pool.get_seq_length()
            valid = kwargs['attention_mask'][:, 0, 0, :prefix] == 0
            snapshot = []
            if prefix:
                for layer in pool:
                    snapshot.append([x.masked_select(valid[:, None, :, None].expand_as(x)).clone()
                                     for x in layer])
            caches.append(snapshot)
        target_hook = model.target_model.model.layers[0].register_forward_pre_hook(observe, with_kwargs=True)
        monkeypatch.setattr(runtime, 'sample_target_with_metadata', sampled)
        monkeypatch.setattr(runtime, 'prefill_draft_prefix', prefill)
        monkeypatch.setattr(opd_sampling, 'SAMPLER_MODE', 'finite')
        torch.manual_seed(42)
        try:
            out = runtime.speculative_generate(model, torch.tensor([[3, 5, 7]]).expand(128, -1),
                torch.ones(128, 3, dtype=torch.long), SimpleNamespace(eos_token_id=96),
                method=method, tlt_scheduler=scheduler, do_sample=True, max_length=80,
                verification_capacity=512, opd_fast_lr=fast_lr, opd_update_stream=True,
                return_all_draft_input=True)
        finally:
            target_hook.remove()
        spec = [r for r in out['tlt_strategy_trace'] if r['phase'] == 'speculative']
        assert [r['batch_size'] for r in spec] == list(BATCHES)
        assert out['tlt_target_only_rounds'] == out['tlt_sd_transition_count'] == state['prefill'] == 1
        assert state['target'] == 1 + out['batch_verification_rounds']
        assert all(r['verification_tokens'] <= 512 for r in out['tlt_strategy_trace'])
        if method == 'tlt_opd_reflex':
            validate_opd_experiment(out, method)
            assert out['opd_feedback_calls'] == len(spec)
            assert (out['opd_updates'] > 0) == (fast_lr > 0)
            adapter = next(iter(model._opd_runtime_cache.values()))
            assert adapter.max_feedback_rows == 512
        answers.append((out, torch.cuda.get_rng_state(), caches))
    if fast_lr == 0:
        (a, rng_a, kv_a), (b, rng_b, kv_b) = answers
        assert a['generated_token_ids'] == b['generated_token_ids']
        assert torch.equal(rng_a, rng_b)
        for key in ('all_draft_input_ids', 'all_draft_input_states'):
            for x, y in zip(a[key], b[key]): torch.testing.assert_close(x, y, rtol=0, atol=0)
        assert len(kv_a) == len(kv_b)
        for old, new in zip(kv_a, kv_b):
            for a_layer, b_layer in zip(old, new):
                for x, y in zip(a_layer, b_layer): torch.testing.assert_close(x, y, rtol=0, atol=0)
