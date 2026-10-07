import triton
import triton.language as tl

@triton.jit(do_not_specialize=["CONTEXTS"])
def _proposal_merge(MAX, SUM, VALUES, IDS, PROBS, TOP_IDS, NORM,
                    CONTEXTS, VOCAB: tl.constexpr, K: tl.constexpr,
                    TILES: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    t = tl.arange(0, BT)
    maxima = tl.load(MAX + row * TILES + t, t < TILES, other=-float('inf'))
    maximum = tl.max(maxima, axis=0)
    sums = tl.load(SUM + row * TILES + t, t < TILES, other=0)
    total = tl.sum(sums * tl.exp(maxima - maximum), axis=0)
    tl.store(NORM + row * 2, maximum)
    tl.store(NORM + row * 2 + 1, total)
    candidates = tl.arange(0, BK)
    values = tl.load(VALUES + row * TILES * K + candidates, candidates < TILES * K, other=-float('inf'))
    ids = tl.load(IDS + row * TILES * K + candidates, candidates < TILES * K, other=VOCAB)
    live = (candidates < TILES * K) & (ids < VOCAB)
    for k in range(K):
        value = tl.max(tl.where(live, values, -float('inf')), axis=0)
        index = tl.min(tl.where(live & (values == value), ids, VOCAB), axis=0)
        tl.store(PROBS + row * K + k, tl.div_rn(tl.exp(value - maximum), total))
        tl.store(TOP_IDS + row * K + k, index)
        # -inf is also a legitimate masked score. Keep selection eligibility
        # separately, or exhausted finite candidates are emitted again at q=0.
        live = live & (ids != index)

