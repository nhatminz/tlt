"""Pure TLT: early SD, explicit budgeted BEG arms, runtime limits and KV parity."""
import csv
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from helper.rollout_metrics import RolloutMetricsWriter
from helper.tlt_scheduler import TLTConfig, TLTScheduler, DEFAULT_STRATEGIES

BUDGETS = [(256, 2), (128, 4), (64, 8), (32, 16), (16, 32), (8, 64), (4, 128), (2, 160), (1, 160)]
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='real CUDA TLT transition/KV regression')


@pytest.mark.parametrize('batch,verify', BUDGETS)
def test_default_beg_uses_budget_bucket_and_learns_between_legal_arms(batch, verify, monkeypatch):
    import helper.fastgrpo_generate as source
    monkeypatch.setattr(source, 'get_adaptive_hyperparameters', lambda *a, **k: pytest.fail('TLT must retain BEG'))
    scheduler = TLTScheduler(TLTConfig())
    scheduler.start_rollout(batch, 512)
    assert scheduler.gate.threshold == scheduler.max_live == batch
    assert scheduler.select(batch) is None and scheduler.gate.pending
    scheduler.record([1] * batch, .01)
    scheduler.gate.complete_transition()
    seen = set()
    for _ in range(50):
        arm = scheduler.select(batch)
        assert arm.name in DEFAULT_STRATEGIES.split(',')
        assert arm.verification_num == verify and batch * verify <= 512
        assert 3 <= arm.depth <= 5 and arm.k <= 8
        scheduler.record([2] * batch, .01)
        seen.add(arm.name)
    assert len(seen) == 2  # Multiple real arms remain available for BEG adaptation.
    assert scheduler.finish()['speculative_round_ratio'] == pytest.approx(50 / 51)
    row = scheduler.trace[-1]
    assert row['actual_verify_tokens'] == batch * verify
    assert row['actual_verify_tokens_per_response'] == verify
    assert row['selected_mab_strategy'] == row['strategy']


def test_auto_threshold_is_resolved_each_rollout_and_restored(monkeypatch):
    monkeypatch.setenv('TLT_BS_THRESHOLD', 'auto')
    scheduler = TLTScheduler(TLTConfig.from_env())
    for batch in (128, 16, 256):
        scheduler.start_rollout(batch, 512)
        assert scheduler.gate.threshold == batch
        assert scheduler.select(batch) is None
        scheduler.record([1] * batch, .01)
        scheduler.gate.complete_transition()
        restored = TLTScheduler(scheduler.config)
        restored.load_state_dict(scheduler.state_dict())
        assert restored.select(batch) == scheduler.select(batch)


def test_beg_filters_configured_arms_by_max_limits_and_replay_cannot_bypass(tmp_path):
    cfg = TLTConfig(strategies='8_8_160,5_8_160,3_2_4,3_1_2', buckets=(1, 65, 129))
    scheduler = TLTScheduler(cfg)
    scheduler.start_rollout(128, 512, max_draft_token_length=3, max_draft_k=2, max_verification_num=4)
    assert (scheduler.workspace_depth, scheduler.workspace_k, scheduler.workspace_verify) == (3, 2, 4)
    scheduler.gate.check(128); scheduler.gate.complete_transition()
    for batch in (128, 64, 32, 16, 8, 4, 2):
        arm = scheduler.select(batch)
        assert arm.name == '3_2_4'  # Explicit arm, never resized from a larger tree.
        scheduler.record([1] * batch, .01)
    scheduler.replay = [dict(rollout=scheduler.rollout_id, round=scheduler.round,
        batch_size=2, phase='speculative', strategy='5_8_160')]
    with pytest.raises(ValueError, match='within rollout limits'):
        scheduler.select(2)


def test_no_compatible_max_arm_fails_clearly():
    scheduler = TLTScheduler(TLTConfig(strategies='8_8_160', buckets=(1,)))
    scheduler.start_rollout(128, 512)
    scheduler.gate.check(128); scheduler.gate.complete_transition()
    with pytest.raises(ValueError, match='MAX_.*no strategy clamping'):
        scheduler.select(128)


def test_coverage_warning_and_weighted_csv_metrics(tmp_path):
    scheduler = TLTScheduler(TLTConfig(bs_threshold=0))
    scheduler.start_rollout(4, 512)
    scheduler.select(4); scheduler.record([1] * 4, .01)
    with pytest.warns(RuntimeWarning, match='speculative_rounds=0'):
        scheduler.warn_if_low_coverage()
    scheduler = TLTScheduler(TLTConfig(warmup_checks=3))
    scheduler.start_rollout(4, 512)
    for _ in range(3):
        scheduler.select(4); scheduler.record([1] * 4, .01)
    scheduler.gate.complete_transition()
    scheduler.select(2); scheduler.record([2] * 2, .01)
    with pytest.warns(RuntimeWarning, match='speculative_round_ratio=0.250'):
        scheduler.warn_if_low_coverage()
    output = scheduler.finish()
    output.update(total_acc_length=16, total_decoded_token_num=14, response_generated_tokens=[16])
    writer = RolloutMetricsWriter(tmp_path / 'metrics.csv', 'tlt')
    writer.begin(0, 0, 4, 0)
    writer.finish(RolloutMetricsWriter.capture(output), grpo_step=1, used_items=4, wall_time_s=1)
    writer.close()
    with (tmp_path / 'metrics.csv').open() as stream:
        row = next(csv.DictReader(stream))
    assert float(row['speculative_round_ratio']) == .25
    assert float(row['cumulative_speculative_round_ratio']) == .25
    assert float(row['speculative_aal']) == 2
    assert float(row['effective_aal']) == pytest.approx(16 / 14)


def test_tuner_accepts_auto_threshold(monkeypatch):
    from scripts.tune_opd_proposals import parse_args, tail_workload
    monkeypatch.setenv('TLT_BS_THRESHOLD', 'auto')
    args = parse_args([])
    assert args.tlt_bs_threshold is None
    live, k, shapes = tail_workload(16, 8, args.tlt_bs_threshold, args.tlt_strategies)
    assert live == 128 and k == 8 and shapes[-1] == (128 * 8, 1)


@CUDA
@pytest.mark.parametrize('batch,verify', BUDGETS)
def test_pure_tlt_one_target_round_then_sd_with_exact_transition_kv(batch, verify, monkeypatch):
    from tests.test_tlt_fastgrpo import tiny
    from helper import tlt_generate as runtime, opd_sampling
    from helper.modeling_draft import DraftModel
    from helper.opd_reflex import OPDReflex
    monkeypatch.setattr(OPDReflex, '__init__', lambda *a, **k: pytest.fail('pure TLT initialized OPD'))
    monkeypatch.setattr(opd_sampling, 'SAMPLER_MODE', 'finite')
    model = tiny()
    scheduler = TLTScheduler(TLTConfig())
    state = dict(target=0, prefill=0)
    native_sample, native_prefill = runtime.sample_target_with_metadata, runtime.prefill_draft_prefix

    def sample(logits, **kwargs):
        logits = logits.clone(); logits[..., 96] = -100
        if scheduler.round >= 3:
            logits[:, 0, 96] = 100  # Controlled EOS after three SD rounds.
        return native_sample(logits, **kwargs)

    def prefill(m, features, ids, padding, **kwargs):
        state['prefill'] += 1
        assert scheduler.round == 1 and scheduler.gate.pending
        assert ids.shape[0] == batch and ids.shape[1] == 4
        before_target, before_rng = state['target'], torch.cuda.get_rng_state()
        rebuilt = native_prefill(m, features, ids, padding, **kwargs)
        assert state['target'] == before_target and torch.equal(before_rng, torch.cuda.get_rng_state())
        config = deepcopy(m.target_model.config)
        config.num_hidden_layers = 1; config.rope_scaling = None
        with torch.random.fork_rng(devices=[]):
            direct = DraftModel(config).cuda().eval()
        direct.load_state_dict(m.draft_model.state_dict())
        length = ids.shape[1]; minimum = torch.finfo(m.dtype).min
        mask = torch.triu(torch.full((length, length), minimum, device='cuda', dtype=m.dtype), diagonal=1)
        mask = mask[None, None].repeat(batch, 1, 1, 1)
        mask.masked_fill_(padding[:, None, None, :], minimum)
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=m.dtype):
            reference = direct(features, m.embed_tokens(ids), attention_mask=mask,
                position_ids=rebuilt['position_ids'], use_cache=True)
        torch.testing.assert_close(reference['hidden_states'][:, -1:], rebuilt['hidden_states'], rtol=0, atol=0)
        torch.testing.assert_close(reference['next_feature_states'][:, -1:], rebuilt['next_feature_states'], rtol=0, atol=0)
        for expected, actual in zip(reference['past_key_values'], rebuilt['past_key_values']):
            for x, y in zip(expected, actual):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
        return rebuilt

    def count(*args): state['target'] += 1
    hook = model.target_model.model.layers[0].register_forward_pre_hook(count)
    monkeypatch.setattr(runtime, 'sample_target_with_metadata', sample)
    monkeypatch.setattr(runtime, 'prefill_draft_prefix', prefill)
    ids = torch.tensor([[0, 5, 7]]).expand(batch, -1).clone()
    ids[batch // 2:, 0] = 3
    try:
        output = runtime.speculative_generate(model, ids, (ids != 0).long(), SimpleNamespace(eos_token_id=96),
            method='tlt', tlt_scheduler=scheduler, do_sample=True, max_length=40,
            verification_capacity=512, max_draft_token_length=5, max_draft_k=8, max_verification_num=160)
    finally:
        hook.remove()
    assert output['target_only_rounds'] == output['tlt_sd_transition_count'] == state['prefill'] == 1
    assert output['speculative_rounds'] == 3 and output['speculative_round_ratio'] == .75
    assert state['target'] == 1 + output['batch_verification_rounds']
    assert output['no_extra_target_forward'] and output['opd_backend'] == 'off'
    assert not hasattr(model, '_opd_runtime_cache') and model.opd_projector is None
    for row in output['tlt_strategy_trace'][1:]:
        assert row['phase'] == 'speculative' and row['verification_num'] == verify
        assert row['actual_verify_tokens'] == batch * verify <= 512
        assert row['actual_draft_depth'] <= 5 and row['actual_draft_k'] <= 8
        assert row['selected_mab_strategy'] == row['strategy']
