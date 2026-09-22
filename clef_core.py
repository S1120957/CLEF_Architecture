"""
clef_core.py
============
Canonical implementation of the CLEF metrics and Algorithms A-G.
Every number and figure in the paper is produced from this module.

Complexity (n = block size, k = average read/write-set size, w = workers):
    metrics / reads-from scan ....... O(n k)
    conflict graph .................. O(n k) time, at most 2 n k edges
    Algorithm A (shard-affinity) .... O(n log n)
    Algorithm B (hot detection) ..... O(n k)
    Algorithm C (Model-A scheduler) . O(n log n + n k)   (successor lists)
    Algorithm D (worker placement) .. O(n log w)
    Algorithm E (rebalancing) ....... O(S^2 log S), S = shards
    Algorithm F (goodness) .......... O(n k)
These bounds assume bounded k. Structural Cases 6 and 8 violate that
assumption on purpose (k grows with n).
"""

from __future__ import annotations

import bisect
import heapq
import math
import random
import time
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

from sortedcontainers import SortedList

NUM_SHARDS = 4
NUM_WORKERS = 4
NUM_ACCOUNTS = 10_000
NUM_RESOURCES = 1_000
SENTINEL = 1_000_000_000      # read-only resource that nothing writes


def shard_of(r: int) -> int:
    return r % NUM_SHARDS


class Tx:
    """A transaction with declared read and write sets."""
    __slots__ = ("tx_id", "reads", "writes", "gas", "load_type", "home")

    def __init__(self, tx_id, reads, writes, gas=1, load_type="generic"):
        self.tx_id = tx_id
        self.reads = list(reads)
        self.writes = list(writes)
        self.gas = gas
        self.load_type = load_type
        # S(tx): shard of the first written resource (the transaction's
        # primary object); read-only transactions use their first read.
        self.home = (self.writes[0] if self.writes else self.reads[0]) % NUM_SHARDS

    def home_shard(self) -> int:
        return self.home

    def __repr__(self):
        return (f"Tx{self.tx_id}[{self.load_type}] R={self.reads} "
                f"W={self.writes} @S{self.home}")


# ─────────────────────────────────────────────────────────────────────────
# Workload generators (identical distributions to testCase_algorithms/shared.py,
# sampled with bisect so a 10,000-transaction block is cheap to generate)
# ─────────────────────────────────────────────────────────────────────────

def _normalise(w):
    s = sum(w)
    return [x / s for x in w]


def power_law_weights(n, exponent=1.5):
    return _normalise([1.0 / (i ** exponent) for i in range(1, n + 1)])


def bursty_weights(n):
    w = power_law_weights(n, 2.5)
    w[0] *= 5
    return _normalise(w)


def _cumulative(w):
    out, acc = [], 0.0
    for x in w:
        acc += x
        out.append(acc)
    return out


SENDER_CUM = _cumulative(power_law_weights(NUM_ACCOUNTS, 1.2))
AVG_CUM = _cumulative(power_law_weights(NUM_RESOURCES, 1.5))
BURSTY_CUM = _cumulative(bursty_weights(NUM_RESOURCES))


def _sample(cum, rng):
    i = bisect.bisect_left(cum, rng.random())
    return i if i < len(cum) else len(cum) - 1


def make_p2p(i, rng):
    s = _sample(SENDER_CUM, rng)
    r = rng.randint(0, NUM_ACCOUNTS - 1)
    while r == s:
        r = rng.randint(0, NUM_ACCOUNTS - 1)
    return Tx(i, [s], [s, r], load_type="P2P")


def make_dex(i, bursty, rng):
    pool = _sample(BURSTY_CUM if bursty else AVG_CUM, rng)
    sender = _sample(SENDER_CUM, rng)
    pr = NUM_ACCOUNTS + pool
    return Tx(i, [sender, pr], [sender, pr],
              load_type="DEX_BURSTY" if bursty else "DEX_AVG")


def make_nft(i, rng):
    nft = _sample(AVG_CUM, rng)
    s = _sample(SENDER_CUM, rng)
    rc = rng.randint(0, NUM_ACCOUNTS - 1)
    nr = NUM_ACCOUNTS + nft
    return Tx(i, [nr, s], [nr, rc], load_type="NFT")


def make_mixed(i, rng):
    x = rng.random()
    if x < 0.40:
        return make_p2p(i, rng)
    if x < 0.70:
        return make_dex(i, False, rng)
    if x < 0.85:
        return make_dex(i, True, rng)
    return make_nft(i, rng)


GENERATORS = {
    "P2P": make_p2p,
    "DEX_AVG": lambda i, r: make_dex(i, False, r),
    "DEX_BURSTY": lambda i, r: make_dex(i, True, r),
    "NFT": make_nft,
    "MIXED": make_mixed,
}
WORKLOADS = list(GENERATORS)


def generate_block(n, workload, seed):
    rng = random.Random(seed)
    gen = GENERATORS[workload]
    return [gen(i, rng) for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────
# Metrics: one left-to-right pass over the reads-from relation
# ─────────────────────────────────────────────────────────────────────────

def reads_from_scan(txs: Sequence[Tx]):
    """
    For every read, the source is the last earlier writer of that resource
    (the reads-from relation). A RAW edge i->j is cross-shard iff
    S(tx_i) != S(tx_j) (the project's message definition).

    Returns (msgs, depth, raw_edges, cross_dsts, has_raw_pred) where
    depth[j] = number of cross-shard hops on the longest RAW path into j.
    """
    last_writer: Dict[int, int] = {}
    n = len(txs)
    depth = [0] * n
    has_pred = [False] * n
    msgs = raw_edges = 0
    cross_dsts: List[int] = []
    for j, tx in enumerate(txs):
        hj, d = tx.home, 0
        for r in tx.reads:
            i = last_writer.get(r)
            if i is None:
                continue
            raw_edges += 1
            has_pred[j] = True
            if txs[i].home != hj:
                msgs += 1
                cross_dsts.append(j)
                if depth[i] + 1 > d:
                    d = depth[i] + 1
            elif depth[i] > d:
                d = depth[i]
        depth[j] = d
        for r in tx.writes:
            last_writer[r] = j
    return msgs, depth, raw_edges, cross_dsts, has_pred


def metrics(txs: Sequence[Tx]) -> dict:
    msgs, depth, raw_edges, cross_dsts, has_pred = reads_from_scan(txs)
    rounds = max(depth) if depth else 0
    counts = Counter(depth)
    frontier = [counts.get(k, 0) for k in range(rounds + 1)] if depth else []
    return {
        "msgs": msgs,
        "rounds": rounds,
        "frontier": frontier,
        "F0": frontier[0] if frontier else 0,
        "raw_edges": raw_edges,
        "cross_dsts": cross_dsts,
        "depth": depth,
        "has_pred": has_pred,
    }


def shard_switches(txs: Sequence[Tx]) -> int:
    """Adjacent positions whose home shards differ (shard contiguity)."""
    return sum(1 for a, b in zip(txs, txs[1:]) if a.home != b.home)


def rmw_lower_bound(txs: Sequence[Tx]) -> Tuple[int, int]:
    """
    Proposition 2. A resource r is pure read-modify-write (pure-RMW) if every
    transaction that accesses r both reads and writes it. With s_r the number
    of distinct home shards among its accessors, every permutation of the
    block produces at least sum_r (s_r - 1) cross-shard messages and at least
    max_r (s_r - 1) communication rounds.
    """
    shards: Dict[int, Set[int]] = defaultdict(set)
    pure: Dict[int, bool] = {}
    for tx in txs:
        R, W = set(tx.reads), set(tx.writes)
        for r in R | W:
            pure[r] = pure.get(r, True) and (r in R and r in W)
            shards[r].add(tx.home)
    gaps = [len(shards[r]) - 1 for r in shards if pure[r]]
    return sum(gaps), (max(gaps) if gaps else 0)



def typed_edge_counts(txs):
    """Counts of conflict edges by (kind, is_cross_shard), kind in RAW/WAW/WAR."""
    c = defaultdict(int)
    lw: Dict[int, int] = {}
    lr: Dict[int, List[int]] = defaultdict(list)
    for j, tx in enumerate(txs):
        for r in tx.reads:
            if r in lw:
                c[("RAW", txs[lw[r]].home != tx.home)] += 1
        for r in tx.writes:
            if r in lw:
                c[("WAW", txs[lw[r]].home != tx.home)] += 1
            for p in lr[r]:
                c[("WAR", txs[p].home != tx.home)] += 1
            lw[r] = j
            lr[r] = []
        for r in tx.reads:
            lr[r].append(j)
    return c

# ─────────────────────────────────────────────────────────────────────────
# Full conflict graph (RAW, WAW, WAR on every resource)
# ─────────────────────────────────────────────────────────────────────────

def conflict_graph(txs: Sequence[Tx]):
    """
    Returns (preds, succ). Only the last writer and the readers since the
    last write are linked; every other conflicting pair is ordered
    transitively, so linear extensions of this graph are exactly the
    conflict-equivalent orderings of the block. |edges| <= 2 n k.
    """
    n = len(txs)
    preds: List[Set[int]] = [set() for _ in range(n)]
    last_writer: Dict[int, int] = {}
    last_readers: Dict[int, List[int]] = defaultdict(list)
    for j, tx in enumerate(txs):
        p = preds[j]
        for r in tx.reads:                          # RAW
            i = last_writer.get(r)
            if i is not None:
                p.add(i)
        for r in tx.writes:
            i = last_writer.get(r)
            if i is not None:                        # WAW
                p.add(i)
            lr = last_readers.get(r)
            if lr:                                   # WAR
                p.update(lr)
            last_writer[r] = j
            last_readers[r] = []
        for r in tx.reads:
            last_readers[r].append(j)
        p.discard(j)
    succ: List[List[int]] = [[] for _ in range(n)]
    for j, p in enumerate(preds):
        for i in p:
            succ[i].append(j)
    return preds, succ


def is_linear_extension(original: Sequence[Tx], ordered: Sequence[Tx]) -> bool:
    preds, _ = conflict_graph(original)
    pos = {id(t): k for k, t in enumerate(ordered)}
    return all(pos[id(original[i])] < pos[id(original[j])]
               for j, p in enumerate(preds) for i in p)


# ─────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────

def order_naive(txs):
    return list(txs)


def order_shard_grouped(txs):
    """Stable counting sort by home shard. O(n)."""
    buckets = [[] for _ in range(NUM_SHARDS)]
    for tx in txs:
        buckets[tx.home].append(tx)
    return [tx for b in buckets for tx in b]


# ─────────────────────────────────────────────────────────────────────────
# Algorithm B: hot-resource detection (per-transaction frequency)
# ─────────────────────────────────────────────────────────────────────────

class SlidingWindowHotness:
    """
    eta(r) = fraction of the last W transactions that access r.
    Per-transaction (not per-access) frequency: with per-access counting a
    resource touched by every transaction scores only 1/k, so thresholds
    such as 0.7 could never be reached by transactions with k >= 2 accesses.
    """

    def __init__(self, window=1000):
        self.window = window
        self.win: deque = deque()
        self.count: Dict[int, int] = defaultdict(int)

    def observe(self, tx: Tx):
        rs = set(tx.reads) | set(tx.writes)
        self.win.append(rs)
        for r in rs:
            self.count[r] += 1
        if len(self.win) > self.window:
            for r in self.win.popleft():
                self.count[r] -= 1
                if self.count[r] == 0:
                    del self.count[r]

    def eta(self, r) -> float:
        return self.count.get(r, 0) / max(1, len(self.win))

    def above(self, tau) -> Set[int]:
        m = max(1, len(self.win))
        return {r for r, c in self.count.items() if c / m >= tau}


def algorithm_B(txs, window=1000, tau_h=0.7, tau_e=0.9):
    """
    Detect hot/extreme resources and move avoidable cross-shard readers of
    extreme resources in front of the first writer (Model B).
    Returns (order, hot, extreme, relocated, scores).
    """
    det = SlidingWindowHotness(window)
    for tx in txs:
        det.observe(tx)
    hot, extreme = det.above(tau_h), det.above(tau_e)
    order = list(txs)
    relocated = 0
    rset = {id(t): set(t.reads) for t in txs}
    wset = {id(t): set(t.writes) for t in txs}
    for r in sorted(extreme):
        p = next((k for k, tx in enumerate(order) if r in wset[id(tx)]), None)
        if p is None:
            continue
        movers = [tx for tx in order[p + 1:]
                  if r in rset[id(tx)] and r not in wset[id(tx)]
                  and tx.home != shard_of(r)]
        if movers:
            ids = {id(t) for t in movers}
            rest = [t for t in order[p:] if id(t) not in ids]
            order = order[:p] + movers + rest
            relocated += len(movers)
    scores = {r: det.eta(r) for r in det.count}
    return order, hot, extreme, relocated, scores


# ─────────────────────────────────────────────────────────────────────────
# Algorithm A: shard-aware block construction (Model B)
# ─────────────────────────────────────────────────────────────────────────

def algorithm_A(txs, hot: Optional[Set[int]] = None, max_gas=None, relax=3):
    """
    Avoidable readers of hot resources first, then the remaining
    transactions stably sorted by home shard, within a gas budget.
    Returns (block, n_avoidable).
    """
    hot = hot or set()
    avoidable, rest = [], []
    for tx in txs:
        if (hot and not hot.isdisjoint(tx.reads)
                and hot.isdisjoint(tx.writes)):
            avoidable.append(tx)
        else:
            rest.append(tx)
    candidate = avoidable + order_shard_grouped(rest)
    if max_gas is None:
        return candidate, len(avoidable)
    block, used, deferred = [], 0, []
    for tx in candidate:
        if used + tx.gas <= max_gas:
            block.append(tx)
            used += tx.gas
        else:
            deferred.append(tx)
    for _ in range(relax):
        still = []
        for tx in deferred:
            if used + tx.gas <= max_gas:
                block.append(tx)
                used += tx.gas
            else:
                still.append(tx)
        deferred = still
    return block, len(avoidable)


# ─────────────────────────────────────────────────────────────────────────
# Algorithm C: semantics-preserving (Model A) shard-affinity scheduler
# ─────────────────────────────────────────────────────────────────────────

def algorithm_C(txs):
    """
    Linear extension of the full conflict graph, extracted with a min-heap
    keyed on (home shard, cross-shard depth, hot-avoidability, index).
    Successor lists make the main loop O(n log n + n k).
    By Proposition 1 the output has the same reads-from relation as the
    input, hence identical messages, rounds and frontier.
    """
    n = len(txs)
    if n == 0:
        return []
    preds, succ = conflict_graph(txs)
    in_deg = [len(p) for p in preds]
    depth = [0] * n
    last_sched: Dict[int, int] = {}

    def evaluate(j):
        tx, d, h = txs[j], 0, 1
        for r in tx.reads:
            w = last_sched.get(r)
            if w is None:
                h = 0
                continue
            if txs[w].home != tx.home:
                d = max(d, depth[w] + 1)
            else:
                d = max(d, depth[w])
        return d, h

    heap = []
    for j in range(n):
        if in_deg[j] == 0:
            d, h = evaluate(j)
            heap.append((txs[j].home, d, h, j))
    heapq.heapify(heap)
    out = []
    while heap:
        _, d, _, i = heapq.heappop(heap)
        depth[i] = d
        out.append(txs[i])
        for r in txs[i].writes:
            last_sched[r] = i
        for j in succ[i]:
            in_deg[j] -= 1
            if in_deg[j] == 0:
                dj, hj = evaluate(j)
                heapq.heappush(heap, (txs[j].home, dj, hj, j))
    assert len(out) == n, "conflict graph must be acyclic"
    return out


# ─────────────────────────────────────────────────────────────────────────
# Algorithm D: worker-load-aware placement
# ─────────────────────────────────────────────────────────────────────────

def algorithm_D(txs, n_workers=NUM_WORKERS, preference=None, slack=1.2,
                homes=None):
    """
    Capacity rule: each worker may take at most slack x the block's mean load.
    A transaction goes to the worker hosting its home shard unless that would
    exceed the capacity; otherwise it goes to the least-loaded worker. Remote
    placements are therefore only genuine overflow, independent of the order
    in which the block is scanned. Returns (assignment, worker loads).
    O(n log w). `homes` overrides tx.home (used with finer object shards).
    """
    homes = homes if homes is not None else [tx.home for tx in txs]
    preference = preference or {s: s % n_workers for s in set(homes)}
    cap = slack * sum(tx.gas for tx in txs) / n_workers
    load = [0] * n_workers
    sl = SortedList((0, w) for w in range(n_workers))
    assign = []
    for tx, h in zip(txs, homes):
        pref = preference[h]
        chosen = pref if load[pref] + tx.gas <= cap else sl[0][1]
        sl.remove((load[chosen], chosen))
        load[chosen] += tx.gas
        sl.add((load[chosen], chosen))
        assign.append(chosen)
    return assign, load


def coefficient_of_variation(values) -> float:
    if not values or sum(values) == 0:
        return 0.0
    m = sum(values) / len(values)
    v = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(v) / m


# ─────────────────────────────────────────────────────────────────────────
# Algorithm E: adaptive shard rebalancing (inter-block)
# ─────────────────────────────────────────────────────────────────────────

def mapping_loads(shard_loads, mapping, n_workers):
    """Worker loads that strict shard affinity would produce under mapping."""
    L = [0] * n_workers
    for s, l in shard_loads.items():
        L[mapping[s]] += l
    return L


def algorithm_E(shard_loads, mapping, n_workers=NUM_WORKERS, delta=0.20,
                max_moves=4):
    """
    Adaptive shard rebalancing (corrected rule).
    Input: per-shard loads measured on the last block and the current
    shard-to-worker mapping. While CV of the mapping-induced loads is at
    least delta, move from the most loaded worker the largest shard whose
    load is below the load gap to the least loaded worker. Each move
    strictly lowers the sum of squared worker loads, so the loop
    terminates; at most max_moves migrations per block. O(m (S + w)).
    """
    mapping = dict(mapping)
    L = mapping_loads(shard_loads, mapping, n_workers)
    ops = []
    for _ in range(max_moves):
        if coefficient_of_variation(L) < delta:
            break
        w_max = max(range(n_workers), key=lambda w: L[w])
        w_min = min(range(n_workers), key=lambda w: L[w])
        gap = L[w_max] - L[w_min]
        cands = [s for s, w in mapping.items()
                 if w == w_max and 0 < shard_loads.get(s, 0) < gap]
        if not cands:
            break
        s = max(cands, key=lambda s: shard_loads[s])
        mapping[s] = w_min
        L[w_max] -= shard_loads[s]
        L[w_min] += shard_loads[s]
        ops.append((s, w_max, w_min, shard_loads[s]))
    return mapping, ops, L


def algorithm_E_original(worker_loads, shard_loads, mapping, delta=0.20,
                         max_iter=20):
    """Rule as first specified: move the heaviest shard to the lightest
    worker until CV(worker load) < delta. Kept for comparison."""
    current, s_loads, mapping = list(worker_loads), dict(shard_loads), dict(mapping)
    ops = []
    for _ in range(max_iter):
        if coefficient_of_variation(current) < delta:
            break
        heavy_s, heavy_l = max(s_loads.items(), key=lambda x: x[1])
        light_w = min(range(len(current)), key=lambda w: current[w])
        old_w = mapping[heavy_s]
        if old_w == light_w or heavy_l == 0:
            break
        current[old_w] -= heavy_l
        current[light_w] += heavy_l
        mapping[heavy_s] = light_w
        s_loads[heavy_s] = 0
        ops.append((heavy_s, old_w, light_w, heavy_l))
    return mapping, ops, current


# ─────────────────────────────────────────────────────────────────────────
# Algorithm F: block goodness
# ─────────────────────────────────────────────────────────────────────────

W_DEFAULT = dict(p=0.25, s=0.25, c=0.20, l=0.20, h=0.10)


def algorithm_F(txs, assign, n_workers=NUM_WORKERS, weights=None, m=None):
    """
    G(B) = w_p P + w_s (1 - S/n) + w_c (1 - C/n) + w_l L + w_h H,
    every component clipped to [0, 1].
      P = fraction of transactions with no RAW predecessor
      S = communication rounds, C = cross-shard messages
      L = 1 - CV(worker load)
      H = 1 - (cross-shard RAW edges into maximum-depth txs) / (RAW edges)
    """
    w = weights or W_DEFAULT
    n = len(txs)
    if n == 0:
        return {"G": 0.0}
    m = m or metrics(txs)
    clip = lambda x: max(0.0, min(1.0, x))
    P = sum(1 for x in m["has_pred"] if not x) / n
    S, C = m["rounds"], m["msgs"]
    loads = [0] * n_workers
    for tx, a in zip(txs, assign):
        loads[a] += tx.gas
    L = clip(1.0 - coefficient_of_variation(loads))
    crit = sum(1 for j in m["cross_dsts"] if m["depth"][j] == S) if S > 0 else 0
    H = clip(1.0 - crit / max(1, m["raw_edges"]))
    comps = dict(P=clip(P), S=clip(1 - S / n), C=clip(1 - C / n), L=L, H=H)
    G = (w["p"] * comps["P"] + w["s"] * comps["S"] + w["c"] * comps["C"]
         + w["l"] * comps["L"] + w["h"] * comps["H"])
    return {"G": G, **comps, "loads": loads}


# ─────────────────────────────────────────────────────────────────────────
# Algorithm G: execution-plan generation (the CLEF pipeline, Model B)
# ─────────────────────────────────────────────────────────────────────────

def algorithm_G(txs, params=None, state=None, guard=False):
    """
    B -> A -> C -> D -> F, with E producing the shard-to-worker preference
    for the next block (inter-block feedback). With guard=True the arrival
    order is committed instead whenever it produces fewer cross-shard
    messages (ties broken by fewer rounds). Returns a plan dictionary.
    """
    p = dict(window=1000, tau_h=0.7, tau_e=0.9, max_gas=None, relax=3,
             delta=0.20, n_workers=NUM_WORKERS)
    p.update(params or {})
    state = state or {}
    t0 = time.perf_counter()
    order_b, hot, extreme, relocated, _ = algorithm_B(
        txs, p["window"], p["tau_h"], p["tau_e"])
    block, n_avoid = algorithm_A(order_b, hot, p["max_gas"], p["relax"])
    ordered = algorithm_C(block)
    m = metrics(ordered)
    guarded = False
    if guard:
        m_arr = metrics(txs)
        if (m_arr["msgs"], m_arr["rounds"]) < (m["msgs"], m["rounds"]):
            ordered, m, guarded = list(txs), m_arr, True
    assign, loads = algorithm_D(ordered, p["n_workers"], state.get("mapping"))
    shard_loads = defaultdict(int)
    for tx in ordered:
        shard_loads[tx.home] += tx.gas
    base_map = state.get("mapping") or {s: s % p["n_workers"]
                                        for s in range(NUM_SHARDS)}
    next_map, ops, _ = algorithm_E(dict(shard_loads), base_map,
                                   p["n_workers"], p["delta"])
    good = algorithm_F(ordered, assign, p["n_workers"], m=m)
    elapsed = (time.perf_counter() - t0) * 1000.0
    return {
        "order": ordered, "assign": assign, "metrics": m, "goodness": good,
        "time_ms": elapsed, "n_hot": len(hot), "n_extreme": len(extreme),
        "n_avoidable": n_avoid, "n_relocated": relocated,
        "next_mapping": next_map, "rebalance_ops": ops, "guarded": guarded,
    }


# ─────────────────────────────────────────────────────────────────────────
# Structural cases 0-8: verbatim toy templates and scalable generators
# ─────────────────────────────────────────────────────────────────────────

def _toy(rows, label):
    return [Tx(i, r, w, load_type=label) for i, (r, w) in rows]


TOY_TEMPLATES = {
    0: [(0, ([1], [2])), (1, ([2], [3])), (2, ([3], [4])), (3, ([4], [5]))],
    1: [(0, ([0], [4])), (1, ([8], [12])), (2, ([16], [20])), (3, ([24], [28]))],
    2: [(0, ([8], [0])), (1, ([0], [4])), (2, ([4], [12])), (3, ([12], [16]))],
    3: [(0, ([1], [0])), (1, ([5], [4])), (2, ([0], [9])), (3, ([13], [13])),
        (4, ([9], [2])), (5, ([17], [17]))],
    4: [(0, ([3], [0])), (1, ([0], [5])), (2, ([5], [2])), (3, ([2], [7])),
        (4, ([7], [4])), (5, ([4], [9]))],
    5: [(0, ([3], [0])), (1, ([0], [5])), (2, ([5], [10])), (3, ([10], [15])),
        (4, ([100], [104])), (5, ([101], [105])), (6, ([102], [106]))],
    6: [(0, ([99], [0, 1, 2, 3])), (1, ([0], [5])), (2, ([0, 5], [6])),
        (3, ([0, 5, 6], [7])), (4, ([0, 5, 6, 7], [8]))],
    7: [(0, ([99], [0, 1])), (1, ([0], [0, 5])), (2, ([0, 5], [6])),
        (3, ([1], [0, 7])), (4, ([0, 7], [8]))],
    8: [(5, ([0, 1, 2, 3, 4, 5], [])), (4, ([0, 1, 2, 3, 4], [5])),
        (3, ([0, 1, 2, 3], [4])), (2, ([0, 1, 2], [3])),
        (1, ([0, 1], [2])), (0, ([99], [0, 1]))],
}

CASE_NAMES = {
    0: "Naive chain $a\\to b\\to c\\to d$",
    1: "Intra-shard, independent",
    2: "Intra-shard, dependent chain",
    3: "Inter-shard, partial dependence",
    4: "Inter-shard, full dependence",
    5: "Inter-shard chain + independents",
    6: "Full cross-shard DAG",
    7: "Hot resource (WAW + RAW)",
    8: "Adversarial ordering",
}


def toy_case(c):
    return _toy(TOY_TEMPLATES[c], f"case{c}")


def _tile(c, n, offset=10_000):
    """Disjoint copies of a toy template; offset is a multiple of NUM_SHARDS
    so every copy keeps the template's shard assignment."""
    rows = TOY_TEMPLATES[c]
    out, copy = [], 0
    while len(out) < n:
        for _, (r, w) in rows:
            if len(out) == n:
                break
            out.append(Tx(len(out), [x + offset * copy for x in r],
                          [x + offset * copy for x in w], load_type=f"case{c}"))
        copy += 1
    return out


def scaled_case(c, n):
    """Cases 0,1,2,6,8 generalise parametrically; 3,4,5,7 are tiled."""
    if c == 0:
        return [Tx(i, [i + 1], [i + 2], load_type="case0") for i in range(n)]
    if c == 1:
        return [Tx(i, [8 * i], [8 * i + 4], load_type="case1") for i in range(n)]
    if c == 2:
        txs = [Tx(0, [SENTINEL], [0], load_type="case2")]
        txs += [Tx(i, [4 * (i - 1)], [4 * i], load_type="case2")
                for i in range(1, n)]
        return txs
    if c == 6:
        txs = [Tx(0, [SENTINEL], [0, 1, 2, 3], load_type="case6")]
        for k in range(1, n):
            txs.append(Tx(k, [0] + [j + 4 for j in range(1, k)], [k + 4],
                          load_type="case6"))
        return txs
    if c == 8:
        txs = []
        for k in range(n - 1, 0, -1):
            txs.append(Tx(k, list(range(k + 1)),
                          [] if k == n - 1 else [k + 1], load_type="case8"))
        txs.append(Tx(0, [SENTINEL], [0, 1], load_type="case8"))
        return txs
    return _tile(c, n)


def case6_closed_form(n):
    """Naive messages and rounds of scaled Case 6 without simulation."""
    counts = [len(range(s, n, NUM_SHARDS)) for s in range(NUM_SHARDS)]
    same = sum(c * (c - 1) // 2 for c in counts)
    return n * (n - 1) // 2 - same, n - 1
