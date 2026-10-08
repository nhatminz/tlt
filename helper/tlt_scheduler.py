"""FastRL adaptive tail gating / BEG, with independent scheduler RNG.

Only scheduling is added to FastGRPO. No Spot Trainer or SGLang runtime.
"""
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import json
import os
import numpy as np
from helper.tlt_mab import MABConfig, MABGroupManager

DEFAULT_STRATEGIES = '8_4_48,8_4_32,8_4_16,8_4_8'


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
    bs_threshold: int = 32
    warmup_checks: int = 10
    strategies: str = DEFAULT_STRATEGIES
    algorithm: str = 'BEG'
    buckets: tuple = (1, 2, 5, 21)
    window: int = 1000
    seed: int = 42

    @classmethod
    def from_env(cls):
        return cls(int(os.getenv('TLT_BS_THRESHOLD', '32')),
                   int(os.getenv('TLT_SD_WARMUP_CHECKS', '10')),
                   os.getenv('TLT_MAB_CONFIGS', DEFAULT_STRATEGIES),
                   os.getenv('TLT_MAB_ALGORITHM', 'BEG'),
                   tuple(int(x) for x in os.getenv('TLT_MAB_BS_THRESHOLDS', '1,2,5,21').split(',')),
                   int(os.getenv('TLT_MAB_WINDOW_SIZE', '1000')),
                   int(os.getenv('TLT_MAB_SEED', '42')))

    def validate(self):
        if self.bs_threshold < 0 or self.warmup_checks < 1 or self.window < 1:
            raise ValueError('TLT threshold>=0, warmup/window>=1 required')
        if not self.buckets or self.buckets[0] != 1 or any(a >= b for a,b in zip(self.buckets,self.buckets[1:])):
            raise ValueError('TLT MAB buckets must be strictly increasing and start at 1')
        return [Strategy.parse(name) for name in MABConfig.validate_configs(self.strategies)]


class AdaptiveTail:
    def __init__(self, threshold, warmup):
        self.threshold, self.warmup = threshold, warmup
        self.consecutive = 0
        self.enabled = False

    def check(self, batch):
        if self.enabled: return True
        self.consecutive = self.consecutive + 1 if batch <= self.threshold else 0
        self.enabled = self.consecutive >= self.warmup
        return self.enabled


class TLTScheduler:
    def __init__(self, config=None, *, trace_path=None, replay_path=None):
        self.config = config or TLTConfig.from_env()
        self.strategies = self.config.validate()
        self.manager = MABGroupManager([s.name for s in self.strategies], self.config.algorithm,
                                      self.config.window, list(self.config.buckets))
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

    def start_rollout(self, batch, verification_capacity=None, vocab=None, proposal_topk=None):
        self.rollout_id += 1
        self.round = 0
        self.gate = AdaptiveTail(self.config.bs_threshold, self.config.warmup_checks)
        self.trace = []
        self.transition_count = 0
        self.transition_prefill_s = 0.
        self.capacity = batch * max(s.verification_num for s in self.strategies)
        if verification_capacity is not None and verification_capacity < self.capacity:
            raise ValueError(f'TLT requires verification_capacity>={self.capacity} for batch {batch}; got {verification_capacity}; no strategy clamping')
        if vocab is not None and max(s.k for s in self.strategies) > vocab:
            raise ValueError('TLT K exceeds full target vocabulary')
        if proposal_topk is not None and max(s.k for s in self.strategies) > proposal_topk:
            raise ValueError('TLT K exceeds OPD_TOPK')
        return self.capacity

    def select(self, batch):
        enabled = self.gate.check(batch)
        with self.private_rng():
            name = self.manager.select_strategy(batch) if enabled else None
        if self.replay is not None:
            if self.replay_index >= len(self.replay): raise ValueError('TLT strategy replay exhausted')
            row = self.replay[self.replay_index]
            if (row['rollout'], row['round'], row['batch_size'], row['phase']) != (self.rollout_id, self.round, batch, 'speculative' if enabled else 'target_only'):
                raise ValueError('TLT replay live batch/phase/rollout mismatch; cannot claim controlled comparison')
            name = row['strategy']
            self.replay_index += 1
        if name is not None and name not in [s.name for s in self.strategies]:
            raise ValueError('TLT replay strategy not in configured strategies')
        strategy = Strategy.parse(name) if name else None
        if strategy and batch * strategy.verification_num > self.capacity:
            raise ValueError('selected TLT strategy exceeds allocated verification capacity')
        self.current = dict(rollout=self.rollout_id, round=self.round, batch_size=batch,
                            phase='speculative' if enabled else 'target_only', strategy=name,
                            verification_num=strategy.verification_num if strategy else 1)
        return strategy

    def record(self, accepted_lengths, processing_time):
        if processing_time <= 0: raise ValueError('TLT processing time must be positive')
        row = self.current.copy()
        aal = sum(accepted_lengths) / row['batch_size']
        reward = None
        if row['strategy'] is not None:
            with self.private_rng():
                stable = self.manager.get_stable_accept_length(row['strategy']) if len(self.strategies)>1 else aal
                reward = stable * row['batch_size'] / processing_time
                self.manager.record_strategy_metrics(row['batch_size'], row['strategy'], reward, aal)
        row.update(aal=aal, processing_time_s=processing_time, reward=reward)
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
        return dict(tlt_target_only_rounds=sum(r['phase']=='target_only' for r in self.trace),
                    tlt_speculative_rounds=sum(r['phase']=='speculative' for r in self.trace),
                    tlt_sd_transition_count=self.transition_count,
                    tlt_transition_draft_prefill_s=self.transition_prefill_s,
                    tlt_strategy_trace=self.trace.copy(),tlt_strategy_metrics=by_strategy,
                    tlt_verification_capacity=self.capacity)

    def state_dict(self):
        return dict(config=asdict(self.config), manager=copy.deepcopy(self.manager),
                    rng=self.rng.get_state(), rollout_id=self.rollout_id, replay_index=self.replay_index,
                    runtime=copy.deepcopy({name:getattr(self,name) for name in
                        ('gate','round','current','capacity','trace','transition_count','transition_prefill_s') if hasattr(self,name)}))

    def load_state_dict(self, state):
        if state['config'] != asdict(self.config): raise ValueError('resume TLT scheduler config mismatch')
        self.manager=copy.deepcopy(state['manager']);self.rng.set_state(state['rng'])
        self.rollout_id=state['rollout_id'];self.replay_index=state['replay_index']
        self.__dict__.update(copy.deepcopy(state.get('runtime',{})))
