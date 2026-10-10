"""FastRL adaptive tail gating / BEG, with independent scheduler RNG.

Only scheduling is added to FastGRPO. No Spot Trainer or SGLang runtime.
"""
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import json
import os
import warnings
import numpy as np
from helper.tlt_mab import MABConfig, MABGroupManager

# Explicit arms, including multiple choices per budget for BEG learning. The
# second arm is a smaller tree, not a runtime resize of the selected strategy.
DEFAULT_STRATEGIES = ('5_8_160,4_8_160,5_8_128,4_7_128,5_8_64,4_5_64,'
                      '5_8_32,4_4_32,4_8_16,3_4_16,3_7_8,3_4_8,'
                      '3_3_4,3_2_4,3_1_2,4_1_2')
DEFAULT_BUCKETS = (1, 3, 5, 9, 17, 33, 65, 129)


def parse_threshold(value):
    """None/auto resolves separately for each rollout, after response expansion."""
    return None if value is None or str(value).lower() == 'auto' else int(value)


@dataclass(frozen=True)
class Strategy:
    depth: int
    k: int
    verification_num: int

    @classmethod
    def parse(cls, name):
        MABConfig.validate_configs(name)
        value = cls(*MABConfig.parse_config(name))
        if value.verification_num < 2:
            raise ValueError('TLT verification_num must include root and at least one draft token')
        if value.total_draft > value.candidate_nodes:
            raise ValueError(f'{name}: requested {value.total_draft} draft nodes, available {value.candidate_nodes}')
        return value

    @property
    def total_draft(self): return self.verification_num - 1
    @property
    def candidate_nodes(self): return self.k + self.k * self.k * (self.depth - 1)
    @property
    def name(self): return MABConfig.format_config(self.depth, self.k, self.verification_num)


@dataclass(frozen=True)
class TLTConfig:
    bs_threshold: int = None
    warmup_checks: int = 1
    strategies: str = DEFAULT_STRATEGIES
    algorithm: str = 'BEG'
    buckets: tuple = DEFAULT_BUCKETS
    window: int = 1000
    seed: int = 42
    scheduling_mode: str = 'budget_aware_beg'
    min_speculative_round_ratio: float = 0.5

    @classmethod
    def from_env(cls):
        return cls(parse_threshold(os.getenv('TLT_BS_THRESHOLD', 'auto')),
                   int(os.getenv('TLT_SD_WARMUP_CHECKS', '1')),
                   os.getenv('TLT_MAB_CONFIGS', DEFAULT_STRATEGIES),
                   os.getenv('TLT_MAB_ALGORITHM', 'BEG'),
                   tuple(int(x) for x in os.getenv('TLT_MAB_BS_THRESHOLDS', ','.join(map(str, DEFAULT_BUCKETS))).split(',')),
                   int(os.getenv('TLT_MAB_WINDOW_SIZE', '1000')),
                   int(os.getenv('TLT_MAB_SEED', '42')),
                   os.getenv('TLT_SCHEDULING_MODE', 'budget_aware_beg'),
                   float(os.getenv('TLT_MIN_SPECULATIVE_ROUND_RATIO', '0.5')))

    def validate(self):
        if self.scheduling_mode not in ('budget_aware_beg', 'fastgrpo_matched'):
            raise ValueError('TLT_SCHEDULING_MODE must be budget_aware_beg or fastgrpo_matched')
        if (self.bs_threshold is not None and self.bs_threshold < 0) or self.warmup_checks < 1 or self.window < 1:
            raise ValueError('TLT threshold>=0, warmup/window>=1 required')
        if not 0 <= self.min_speculative_round_ratio <= 1:
            raise ValueError('TLT_MIN_SPECULATIVE_ROUND_RATIO must be in [0, 1]')
        if not self.buckets or self.buckets[0] != 1 or any(a >= b for a,b in zip(self.buckets,self.buckets[1:])):
            raise ValueError('TLT MAB buckets must be strictly increasing and start at 1')
        return [Strategy.parse(name) for name in MABConfig.validate_configs(self.strategies)]


class AdaptiveTail:
    def __init__(self, threshold, warmup):
        self.threshold, self.warmup = threshold, warmup
        self.consecutive = 0
        self.enabled = False
        self.pending = False

    def check(self, batch):
        if self.enabled: return True
        if self.pending: return False
        self.consecutive = self.consecutive + 1 if batch <= self.threshold else 0
        self.pending = self.consecutive >= self.warmup
        # FastRL: threshold-triggering batch still decodes target-only.
        return False

    def complete_transition(self):
        if not self.pending: raise ValueError('no pending TLT transition')
        self.pending = False
        self.enabled = True


class TLTScheduler:
    def __init__(self, config=None, *, trace_path=None, replay_path=None):
        self.config = config or TLTConfig.from_env()
        self.strategies = self.config.validate()
        self.manager = (MABGroupManager([s.name for s in self.strategies], self.config.algorithm,
                                       self.config.window, list(self.config.buckets))
                        if self.config.scheduling_mode == 'budget_aware_beg' else None)
        self.rng = np.random.RandomState(self.config.seed)
        self.trace_path = trace_path if trace_path is not None else os.getenv('TLT_STRATEGY_TRACE', '')
        replay_path = replay_path if replay_path is not None else os.getenv('TLT_STRATEGY_REPLAY', '')
        self.replay = [json.loads(line) for line in Path(replay_path).read_text().splitlines() if line.strip()] if replay_path else None
        self.replay_index = 0
        self.rollout_id = -1
        self.trace = []

    @contextmanager
    def private_rng(self):
        # Execute the exact upstream NumPy-based algorithm without touching
        # global sampling RNG. Also preserves reference metric refresh draws.
        previous = np.random.get_state()
        np.random.set_state(self.rng.get_state())
        try: yield
        finally:
            self.rng.set_state(np.random.get_state())
            np.random.set_state(previous)

    def start_rollout(self, batch, verification_capacity=None, vocab=None, proposal_topk=None,
                      *, max_draft_token_length=5, max_draft_k=8, max_verification_num=160,
                      min_draft_token_length=3, draft_token_length_c=0.75):
        self.rollout_id += 1
        self.round = 0
        threshold = batch if self.config.bs_threshold is None else self.config.bs_threshold
        self.gate = AdaptiveTail(threshold, self.config.warmup_checks)
        self.trace = []
        self.transition_count = 0
        self.transition_prefill_s = 0.
        self.max_live = min(batch, threshold)
        self.capacity = 512 if verification_capacity is None else verification_capacity
        if self.capacity < batch or self.capacity < 2:
            raise ValueError('verification_capacity must fit the initial target-only batch and at least two tokens')
        if min(max_draft_token_length, max_draft_k, min_draft_token_length) < 1 or max_verification_num < 2 or draft_token_length_c <= 0:
            raise ValueError('invalid FastGRPO adaptive limits')
        if min_draft_token_length > max_draft_token_length:
            raise ValueError('minimum draft depth exceeds maximum')
        self.adaptive = dict(max_draft_token_length=max_draft_token_length, max_draft_k=max_draft_k,
                             max_verification_num=max_verification_num,
                             min_draft_token_length=min_draft_token_length, draft_token_length_c=draft_token_length_c)
        if self.config.scheduling_mode == 'fastgrpo_matched':
            possible = [self.matched_strategy(b) for b in range(1, self.max_live + 1)
                        if self.capacity // b >= 2]
        else:
            # Keep original arms and upstream MAB state. Only mask arms outside
            # the actual rollout limits; never silently clamp a selected arm.
            possible = [s for s in self.strategies
                        if min_draft_token_length <= s.depth <= max_draft_token_length
                        and s.k <= max_draft_k and s.verification_num <= max_verification_num]
        self.allowed_strategies = possible
        self.workspace_depth = max((s.depth for s in possible), default=max_draft_token_length)
        self.workspace_k = max((s.k for s in possible), default=max_draft_k)
        self.workspace_verify = max((s.verification_num for s in possible), default=max_verification_num)
        if vocab is not None and self.workspace_k > vocab:
            raise ValueError('TLT K exceeds full target vocabulary')
        if proposal_topk is not None and self.workspace_k > proposal_topk:
            raise ValueError('TLT K exceeds OPD_TOPK')
        return self.capacity

    def matched_strategy(self, batch):
        # Same function and inputs as SpecNaacl, without reimplementing its formula.
        from helper.fastgrpo_generate import get_adaptive_hyperparameters
        if self.capacity // batch < 2:
            raise ValueError(f'no feasible speculative strategy for live batch {batch}, capacity {self.capacity}')
        depth, k, draft = get_adaptive_hyperparameters(batch, self.capacity, **self.adaptive)
        return Strategy.parse(f'{depth}_{k}_{draft + 1}')

    def budget_strategy(self, batch):
        valid = [s.name for s in self.allowed_strategies if batch * s.verification_num <= self.capacity]
        if not valid:
            raise ValueError(f'no feasible configured TLT strategy for live batch {batch}, capacity {self.capacity}; '
                             'add an explicit smaller strategy within MAX_* limits; no strategy clamping')
        if len(self.strategies) == 1:
            return valid[0]
        mab = self.manager.mabs[self.manager._get_group(batch)]
        if self.manager.algorithm == 'BEG':
            preferred = next((mab.draft_tokens_groups[tokens]
                              for low, high, tokens in mab.batch_bucket_mapping if low <= batch <= high), valid)
            eligible = [name for name in preferred if name in valid]
            if not eligible:
                # Preserve the nearest smaller predefined bucket, without altering any arm.
                verify = max(Strategy.parse(name).verification_num for name in valid)
                eligible = [name for name in valid if Strategy.parse(name).verification_num == verify]
            return eligible[0] if len(eligible) == 1 else mab._epsilon_greedy_select(eligible)
        if self.manager.algorithm == 'PREDEFINED':
            preferred = mab.select_strategy(batch_size=batch)
            return preferred if preferred in valid else max(valid, key=lambda name: Strategy.parse(name).verification_num)
        return mab.select_strategy(valid)

    def select(self, batch):
        enabled = self.gate.check(batch)
        with self.private_rng():
            name = (self.matched_strategy(batch).name if self.config.scheduling_mode == 'fastgrpo_matched'
                    else self.budget_strategy(batch)) if enabled else None
        expected_name = name
        if self.replay is not None:
            if self.replay_index >= len(self.replay): raise ValueError('TLT strategy replay exhausted')
            row = self.replay[self.replay_index]
            if (row['rollout'], row['round'], row['batch_size'], row['phase']) != (self.rollout_id, self.round, batch, 'speculative' if enabled else 'target_only'):
                raise ValueError('TLT replay live batch/phase/rollout mismatch; cannot claim controlled comparison')
            name = row['strategy']
            self.replay_index += 1
        if (name is None) != (not enabled):
            raise ValueError('TLT replay strategy/phase mismatch')
        if self.config.scheduling_mode == 'fastgrpo_matched' and name != expected_name:
            raise ValueError('TLT replay differs from exact FastGRPO adaptive strategy')
        if self.config.scheduling_mode == 'budget_aware_beg' and name is not None and name not in [s.name for s in self.allowed_strategies]:
            raise ValueError('TLT replay strategy not in configured strategies within rollout limits')
        strategy = Strategy.parse(name) if name else None
        if strategy and batch * strategy.verification_num > self.capacity:
            raise ValueError('selected TLT strategy exceeds allocated verification capacity')
        self.current = dict(rollout=self.rollout_id, round=self.round, batch_size=batch,
                            phase='speculative' if enabled else 'target_only', strategy=name,
                            scheduling_mode=self.config.scheduling_mode,
                            actual_depth=strategy.depth if strategy else 0, actual_k=strategy.k if strategy else 0,
                            actual_draft_depth=strategy.depth if strategy else 0,
                            actual_draft_k=strategy.k if strategy else 0,
                            selected_mab_strategy=name,
                            actual_verify_tokens=batch * (strategy.verification_num if strategy else 1),
                            actual_verify_tokens_per_response=strategy.verification_num if strategy else 1,
                            verification_num=strategy.verification_num if strategy else 1,
                            verification_tokens=batch * (strategy.verification_num if strategy else 1))
        return strategy

    def record(self, accepted_lengths, processing_time):
        if processing_time <= 0: raise ValueError('TLT processing time must be positive')
        row = self.current.copy()
        aal = sum(accepted_lengths) / row['batch_size']
        reward = None
        if row['strategy'] is not None and self.manager is not None:
            with self.private_rng():
                stable = self.manager.get_stable_accept_length(row['strategy']) if len(self.strategies)>1 else aal
                reward = stable * row['batch_size'] / processing_time
                self.manager.record_strategy_metrics(row['batch_size'], row['strategy'], reward, aal)
        row.update(aal=aal, processing_time_s=processing_time, reward=reward)
        row['timing_basis']='cuda_stream_events' if row['strategy'] is not None else 'host_wall_at_eos_scheduling'
        row['transition_prefill_in_reward']=False
        self.trace.append(row)
        if self.trace_path:
            path = Path(self.trace_path); path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as stream: stream.write(json.dumps(row)+'\n')
        self.round += 1

    def finish(self):
        by_strategy = {}
        for row in self.trace:
            key = row['strategy'] or 'target_only'
            group = by_strategy.setdefault(key, dict(rounds=0,time_s=0.,accepted=0.,response_rounds=0))
            group['rounds'] += 1; group['time_s'] += row['processing_time_s']
            group['accepted'] += row['aal'] * row['batch_size'];group['response_rounds'] += row['batch_size']
        for group in by_strategy.values(): group['aal'] = group['accepted']/group['response_rounds']
        response_rounds=sum(r['batch_size'] for r in self.trace)
        accepted=sum(r['aal']*r['batch_size'] for r in self.trace)
        spec=[r for r in self.trace if r['phase']=='speculative']
        spec_rounds=sum(r['batch_size'] for r in spec)
        spec_accepted=sum(r['aal']*r['batch_size'] for r in spec)
        target_rounds = len(self.trace) - len(spec)
        ratio = len(spec) / len(self.trace) if self.trace else 0.
        return dict(target_only_rounds=target_rounds, speculative_rounds=len(spec),
                    speculative_round_ratio=ratio,
                    tlt_target_only_rounds=target_rounds, tlt_speculative_rounds=len(spec),
                    tlt_speculative_round_ratio=ratio,
                    tlt_sd_transition_count=self.transition_count,
                    tlt_transition_draft_prefill_s=self.transition_prefill_s,
                    tlt_strategy_trace=self.trace.copy(),tlt_strategy_metrics=by_strategy,
                    tlt_verification_capacity=self.capacity,
                    tlt_max_live=self.max_live,
                    effective_aal=accepted/response_rounds if response_rounds else 0.,
                    speculative_aal=spec_accepted/spec_rounds if spec_rounds else 0.,
                    tlt_speculative_accepted_length=spec_accepted,tlt_speculative_response_rounds=spec_rounds)

    def warn_if_low_coverage(self):
        metrics = self.finish()
        if not metrics['speculative_rounds'] or metrics['speculative_round_ratio'] < self.config.min_speculative_round_ratio:
            warnings.warn(
                f"TLT SD coverage is low: target_only_rounds={metrics['target_only_rounds']}, "
                f"speculative_rounds={metrics['speculative_rounds']}, "
                f"speculative_round_ratio={metrics['speculative_round_ratio']:.3f}; "
                f"resolved threshold={self.gate.threshold}, warmup={self.config.warmup_checks}. "
                'Use TLT_BS_THRESHOLD=auto and TLT_SD_WARMUP_CHECKS=1; also check EOS/length limits.',
                RuntimeWarning, stacklevel=2)

    def state_dict(self):
        return dict(config=asdict(self.config), manager=copy.deepcopy(self.manager),
                    rng=self.rng.get_state(), rollout_id=self.rollout_id, replay_index=self.replay_index,
                    runtime=copy.deepcopy({name:getattr(self,name) for name in
                        ('gate','round','current','capacity','max_live','trace','transition_count','transition_prefill_s',
                         'adaptive','allowed_strategies','workspace_depth','workspace_k','workspace_verify') if hasattr(self,name)}))

    def load_state_dict(self, state):
        if state['config'] != asdict(self.config): raise ValueError('resume TLT scheduler config mismatch')
        self.manager=copy.deepcopy(state['manager']);self.rng.set_state(state['rng'])
        self.rollout_id=state['rollout_id'];self.replay_index=state['replay_index']
        self.__dict__.update(copy.deepcopy(state.get('runtime',{})))
