"""
Additional replacement/admission policies, reimplemented from their published descriptions (no third-party code is run).

Every policy follows the repository's EvictionPolicy protocol: `on_query` once per request, `on_hit` when a stored entry
is served, `on_insert` after a miss has been inserted, `choose_victim` while the cache is over capacity, `on_evict`.
Admission is expressed the way the repository's TinyLFU-style policies express it: when the newcomer is returned as the
victim it is not admitted.

Policies
  FIFOPolicy        first in, first out (always admits).
  ARCPolicy         Megiddo and Modha (2003), adaptive replacement cache; ghost lists are keyed by the exact request text,
                    as for any classical key-value cache (hits are decided by the semantic index, so a ghost hit means the
                    same text came back after its entry was evicted).
  S3FIFOPolicy      Yang et al. (2023), S3-FIFO: small FIFO (10% of capacity), main FIFO, ghost queue, 2-bit frequency.
  SIEVEPolicy       Zhang et al. (2024), SIEVE: FIFO list, visited bit, a hand that evicts the first unvisited object and
                    keeps its position.
  GDSFPolicy        Cherkasova (1998), Greedy-Dual-Size-Frequency: H = L + frequency * cost / size, L = H of the last victim;
                    cost = 1 (one generation), size = prompt + completion tokens of the entry.
  DACPolicy         Berend et al. (2025), DynamicAdaptiveClimb (arXiv:2511.21235), promotion logic exactly as in their
                    Algorithm 2 with the dynamic resizing disabled (capacity is the experiment's budget, so the size K stays
                    fixed; the resize rule would change the budget being compared).
  SolarPolicy       Sun, Cao and Lam (2026), SOLAR (arXiv:2607.00394), three variants:
                      mode "E"    always admit; eviction by Thompson sampling over per-entry Beta posteriors plus a novelty
                                  bonus;
                      mode "A"    regret-gated admission, heuristic multi-score eviction;
                      mode "full" regret-gated admission and posterior-guided eviction.

Documented adaptations of SOLAR to a response cache (the paper's setting is an agent memory buffer):
  * quality signal: the paper uses a task F1. A response cache has no per-request F1, so qual_t is the cosine between the
    request and the entry served (a hit) and 0 for a miss; delta_t = 1 - qual_t. This uses no label.
  * positive evidence: an entry served on a hit gets alpha += 1; an entry not served for `decay_every` requests gets
    beta += 0.05 once per interval (the paper does not give the interval; it is set to the capacity).
  * modification timing: a new entry exists only on a miss, so the regret gate is evaluated on misses; delta_t is accumulated
    on every request. Before the cache is full every miss is admitted (the gate only has meaning at capacity).
  * the paper does not give gamma, Delta_nov, lambda or tau_0 in the text available; they are tuned on the tuning windows
    only, over the small grid VARIANTS below. SOLAR-A's "heuristic multi-score eviction" is not specified in that text; it is
    implemented as the sum of three deterministic scores, normalised frequency, recency and the novelty bonus (equal weights).
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict

import numpy as np

from .cache import EvictionPolicy

# (lambda, gamma) grid for SOLAR; the index is carried in the experiment config field "h"
SOLAR_VARIANTS = ((0.05, 0.25), (0.2, 0.25), (0.8, 0.25), (0.2, 0.5))


class FIFOPolicy(EvictionPolicy):
    name = "FIFO"

    def __init__(self) -> None:
        self.order: OrderedDict[int, None] = OrderedDict()

    def on_insert(self, entry, vec):
        self.order[entry.eid] = None

    def choose_victim(self, cache, vb):
        return next(iter(self.order))

    def on_evict(self, eid):
        self.order.pop(eid, None)


class ARCPolicy(EvictionPolicy):
    name = "ARC"

    def __init__(self, capacity: int) -> None:
        self.c = int(capacity)
        self.p = 0.0
        self.t1: OrderedDict[int, str] = OrderedDict()     # eid -> key, LRU first
        self.t2: OrderedDict[int, str] = OrderedDict()
        self.b1: OrderedDict[str, None] = OrderedDict()    # ghost keys
        self.b2: OrderedDict[str, None] = OrderedDict()
        self._new: int | None = None
        self._new_in_b2 = False
        self._where: dict[int, int] = {}

    def on_insert(self, entry, vec):
        key = entry.query
        self._new, self._new_in_b2 = entry.eid, False
        if key in self.b1:
            self.p = min(self.c, self.p + max(1.0, len(self.b2) / max(1, len(self.b1))))
            del self.b1[key]
            self.t2[entry.eid] = key
        elif key in self.b2:
            self.p = max(0.0, self.p - max(1.0, len(self.b1) / max(1, len(self.b2))))
            del self.b2[key]
            self.t2[entry.eid] = key
            self._new_in_b2 = True
        else:
            self.t1[entry.eid] = key

    def on_hit(self, entry, vec):
        key = self.t1.pop(entry.eid, None)
        if key is None:
            key = self.t2.pop(entry.eid, entry.query)
        self.t2[entry.eid] = key

    def choose_victim(self, cache, vb):
        n1 = len(self.t1) - (1 if self._new in self.t1 else 0)
        n2 = len(self.t2) - (1 if self._new in self.t2 else 0)
        use_t1 = n1 >= 1 and (n1 > self.p or (self._new_in_b2 and n1 == int(self.p)))
        if not use_t1 and n2 < 1:
            use_t1 = True
        src, ghost = (self.t1, self.b1) if use_t1 else (self.t2, self.b2)
        victim = next(e for e in src if e != self._new)
        ghost[src[victim]] = None
        while len(self.t1) + len(self.b1) > self.c and self.b1:
            self.b1.popitem(last=False)
        while len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) > 2 * self.c and self.b2:
            self.b2.popitem(last=False)
        return victim

    def on_evict(self, eid):
        self.t1.pop(eid, None)
        self.t2.pop(eid, None)


class S3FIFOPolicy(EvictionPolicy):
    name = "S3-FIFO"

    def __init__(self, capacity: int, small_frac: float = 0.1) -> None:
        self.c = int(capacity)
        self.small_cap = max(1, int(round(small_frac * self.c)))
        self.S: OrderedDict[int, str] = OrderedDict()
        self.M: OrderedDict[int, str] = OrderedDict()
        self.G: OrderedDict[str, None] = OrderedDict()
        self.freq: dict[int, int] = defaultdict(int)
        self._new: int | None = None

    def on_insert(self, entry, vec):
        self._new = entry.eid
        self.freq[entry.eid] = 0
        if entry.query in self.G:
            del self.G[entry.query]
            self.M[entry.eid] = entry.query
        else:
            self.S[entry.eid] = entry.query

    def on_hit(self, entry, vec):
        self.freq[entry.eid] = min(3, self.freq[entry.eid] + 1)

    def _evict_main(self):
        for _ in range(4 * max(1, len(self.M)) + 4):
            for e in self.M:
                if e == self._new:
                    continue
                if self.freq[e] > 0:
                    self.freq[e] -= 1
                    self.M.move_to_end(e)
                    break
                return e
            else:
                return None
        return next((e for e in self.M if e != self._new), None)

    def choose_victim(self, cache, vb):
        for _ in range(len(self.S) + len(self.M) + 4):
            small_over = len(self.S) >= self.small_cap or not self.M
            if small_over:
                head = next((e for e in self.S if e != self._new), None)
                if head is not None:
                    if self.freq[head] > 1:
                        self.M[head] = self.S.pop(head)
                        self.freq[head] = 0
                        continue
                    self.G[self.S[head]] = None
                    while len(self.G) > max(1, self.c - self.small_cap):
                        self.G.popitem(last=False)
                    return head
            v = self._evict_main()
            if v is not None:
                return v
            head = next((e for e in self.S if e != self._new), None)
            if head is not None:
                return head
        return self._new

    def on_evict(self, eid):
        self.S.pop(eid, None)
        self.M.pop(eid, None)
        self.freq.pop(eid, None)


class SIEVEPolicy(EvictionPolicy):
    name = "SIEVE"

    def __init__(self) -> None:
        self.nxt: dict[int, int | None] = {}      # towards newer
        self.prv: dict[int, int | None] = {}      # towards older
        self.visited: dict[int, bool] = {}
        self.head: int | None = None              # newest
        self.tail: int | None = None              # oldest
        self.hand: int | None = None
        self._new: int | None = None

    def on_insert(self, entry, vec):
        e = entry.eid
        self._new = e
        self.visited[e] = False
        self.prv[e], self.nxt[e] = self.head, None
        if self.head is not None:
            self.nxt[self.head] = e
        self.head = e
        if self.tail is None:
            self.tail = e

    def on_hit(self, entry, vec):
        self.visited[entry.eid] = True

    def choose_victim(self, cache, vb):
        o = self.hand if self.hand in self.visited else self.tail
        for _ in range(2 * len(self.visited) + 2):
            if o is None:
                o = self.tail
            if o == self._new:
                o = self.nxt[o]
                continue
            if self.visited[o]:
                self.visited[o] = False
                o = self.nxt[o]
                continue
            self.hand = self.nxt[o]
            return o
        return self._new

    def on_evict(self, eid):
        p, n = self.prv.pop(eid, None), self.nxt.pop(eid, None)
        self.visited.pop(eid, None)
        if p is not None:
            self.nxt[p] = n
        else:
            self.tail = n
        if n is not None:
            self.prv[n] = p
        else:
            self.head = p
        if self.hand == eid:
            self.hand = n


class GDSFPolicy(EvictionPolicy):
    name = "GDSF"

    def __init__(self) -> None:
        self.L = 0.0
        self.freq: dict[int, int] = {}
        self.size: dict[int, float] = {}
        self.H: dict[int, float] = {}
        self.seq: dict[int, int] = {}
        self._t = 0

    def _prio(self, e: int) -> float:
        return self.L + self.freq[e] / self.size[e]

    def on_insert(self, entry, vec):
        self._t += 1
        e = entry.eid
        self.freq[e], self.size[e], self.seq[e] = 1, float(max(1, entry.total_tokens)), self._t
        self.H[e] = self._prio(e)

    def on_hit(self, entry, vec):
        e = entry.eid
        self.freq[e] += 1
        self.H[e] = self._prio(e)

    def choose_victim(self, cache, vb):
        v = min(self.H, key=lambda e: (self.H[e], self.seq[e]))
        self.L = self.H[v]
        return v

    def on_evict(self, eid):
        for d in (self.freq, self.size, self.H, self.seq):
            d.pop(eid, None)


class DACPolicy(EvictionPolicy):
    """DynamicAdaptiveClimb with fixed K. cache[1] is the top. A hit at position i climbs actualJump = max(1, min(jump, i-1))
    places; a miss inserts the newcomer at position K - actualJump + 1 (actualJump = max(1, min(K-1, jump))) and the bottom
    entry is evicted. jump starts at K, falls by 1 on a hit (down to -K/2) and rises by 1 on a miss."""
    name = "DynamicAdaptiveClimb"

    def __init__(self, capacity: int) -> None:
        self.K = int(capacity)
        self.jump = self.K
        self.order: list[int] = []                  # top first

    def on_hit(self, entry, vec):
        e = entry.eid
        if self.jump > -self.K // 2:
            self.jump -= 1
        i = self.order.index(e) + 1
        actual = max(1, min(self.jump, i - 1))
        if i > 1:
            self.order.pop(i - 1)
            self.order.insert(i - actual - 1, e)

    def on_insert(self, entry, vec):
        self.jump += 1
        actual = max(1, min(self.K - 1, self.jump))
        pos = max(1, self.K - actual + 1)           # 1-indexed position in the post-eviction list
        self.order.insert(min(pos, len(self.order) + 1) - 1, entry.eid)

    def choose_victim(self, cache, vb):
        return self.order[-1]

    def on_evict(self, eid):
        try:
            self.order.remove(eid)
        except ValueError:
            pass


class SolarPolicy(EvictionPolicy):
    def __init__(self, capacity: int, mode: str = "full", lam: float = 0.2, gamma: float = 0.25, seed: int = 0,
                 alpha_ema: float = 0.1, decay_inc: float = 0.05) -> None:
        assert mode in ("E", "A", "full")
        self.name = {"E": "SOLAR-E", "A": "SOLAR-A", "full": "SOLAR"}[mode]
        self.mode, self.c, self.lam, self.gamma = mode, int(capacity), lam, gamma
        self.alpha_ema, self.decay_inc = alpha_ema, decay_inc
        self.decay_every = max(1, self.c)
        self.delta_nov = max(1, self.c)
        self.rng = np.random.default_rng(seed)
        self.a: dict[int, float] = {}
        self.b: dict[int, float] = {}
        self.born: dict[int, int] = {}
        self.used: dict[int, int] = {}
        self.hits: dict[int, int] = {}
        self.vec: dict[int, np.ndarray] = {}
        self.t = 0
        self.cum = 0.0                       # accumulated regret c
        self.loss_sum, self.loss_n = 1.0, 1  # running mean of delta (the paper's average loss rate L)
        self.tau = float(np.sqrt(2 * lam / 1.0))
        self._served = False
        self._qual = 0.0
        self._new: int | None = None
        self._admit = True
        self.modifications = 0
        self.misses = 0

    # --- bookkeeping per request -------------------------------------------------
    def on_query(self, vec):
        self._finish_request()
        self.t += 1
        self._served, self._qual = False, 0.0
        self._q = vec
        if self.t % self.decay_every == 0:
            for e in self.a:
                if self.t - self.used[e] >= self.decay_every:
                    self.b[e] += self.decay_inc

    def _finish_request(self):
        if self.t == 0:
            return
        delta = 1.0 - (self._qual if self._served else 0.0)
        self.loss_sum += delta
        self.loss_n += 1
        if self._served:                      # a hit: regret accrues, no entry to admit
            self.cum += delta

    def on_hit(self, entry, vec):
        e = entry.eid
        self._served = True
        v = self.vec.get(e)
        self._qual = float(np.clip(np.dot(self._q, v), 0.0, 1.0)) if v is not None else 1.0
        self.a[e] += 1.0
        self.used[e] = self.t
        self.hits[e] += 1

    def on_insert(self, entry, vec):
        e = entry.eid
        self._new = e
        self.misses += 1
        self.a[e], self.b[e], self.born[e], self.used[e], self.hits[e] = 1.0, 1.0, self.t, self.t, 0
        self.vec[e] = np.asarray(vec, dtype=np.float32)
        self.cum += 1.0                       # delta of this miss
        self._admit = True
        if self.mode != "E":
            self._admit = self.cum >= self.tau
            if self._admit:
                self.cum = 0.0
                L = self.loss_sum / self.loss_n
                self.tau = (1 - self.alpha_ema) * self.tau + self.alpha_ema * min(1.0, float(np.sqrt(2 * self.lam / max(L, 1e-6))))
                self.modifications += 1

    # --- eviction ----------------------------------------------------------------
    def choose_victim(self, cache, vb):
        if not self._admit:
            return self._new                  # gate closed: newcomer rejected
        ids = [e for e in self.a if e != self._new]
        if not ids:
            return self._new
        age = np.array([self.t - self.born[e] for e in ids], dtype=np.float64)
        nov = self.gamma / (1.0 + age / self.delta_nov)
        if self.mode == "A":
            h = np.array([self.hits[e] for e in ids], dtype=np.float64)
            rec = 1.0 / (1.0 + (self.t - np.array([self.used[e] for e in ids], dtype=np.float64)) / self.c)
            score = h / (1.0 + h.max()) + rec + nov
        else:
            a = np.array([self.a[e] for e in ids])
            b = np.array([self.b[e] for e in ids])
            score = self.rng.beta(a, b) + nov
        return ids[int(np.argmin(score))]

    def on_evict(self, eid):
        for d in (self.a, self.b, self.born, self.used, self.hits, self.vec):
            d.pop(eid, None)


class SolarOfficialPolicy(EvictionPolicy):
    """SOLAR with the constants and rules of the authors' reference code (github.com/ysunbp/SOLAR, commit 8d01e93), no free parameters.

    Fixed as in their configs and code: lambda_cost 5, Lipschitz prior 2 (initial tau = sqrt(2 lambda / 2) = 2.236), adaptive window 30,
    L_hat = EMA(0.7/0.3) of max(2 x std(last 30 regrets), 0.1), tau = clip(sqrt(2 lambda / L_hat), 0.1, 5), regret accumulated and
    compared with '>' , reset to 0 on a store; bootstrap: store everything while fewer than min(max(K/2, 5), 25) entries are live;
    Thompson sampling with optimistic prior Beta(2, 1), alpha += 1 for a retrieved (served) entry, beta += 0.05 every 5th request for
    the others, score = Beta sample + 0.3 x novelty, novelty = 1 - exp(-d / (2 dim)) with d the mean squared L2 distance to the two
    nearest other entries (computed once, at insertion, because the term is at most ~0.006 for unit vectors and changes a decision only
    in exact ties). SOLAR-A (mode "A") evicts by 0.4 exp(-age/100) + 0.3 access_count/requests + 0.3 diversity, as in solar_a.py.
    Adaptation to a response cache (not in the paper): the regret of a request is 1 for a miss and 1 - cosine(request, served entry)
    for a hit (label-free, no feedback; the harness of the paper uses 1 - F1 of the agent's answer); the regret accumulates at every
    request and the gate is evaluated when a miss produces a new entry. mode "E" always admits.
    """

    def __init__(self, capacity: int, mode: str = "full", dim: int = 768, seed: int = 0):
        assert mode in ("E", "A", "full")
        self.name = {"E": "SOLARo-E", "A": "SOLARo-A", "full": "SOLARo"}[mode]
        self.mode, self.c, self.dim = mode, int(capacity), dim
        self.lam, self.lip = 5.0, 2.0
        self.tau = float(np.sqrt(2 * self.lam / self.lip))
        self.window, self.tmin, self.tmax = 30, 0.1, 5.0
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.regret = 0.0
        self.hist: list[float] = []
        self.L_ema: float | None = None
        self.a: dict[int, float] = {}
        self.b: dict[int, float] = {}
        self.nov: dict[int, float] = {}
        self.vec: dict[int, np.ndarray] = {}
        self.born: dict[int, int] = {}
        self.last: dict[int, int] = {}
        self.cnt: dict[int, int] = {}
        self._cost = 1.0
        self._served: set[int] = set()
        self._new: int | None = None
        self._admit = True
        self.stores = 0

    def _novelty_of(self, v: np.ndarray, others: list[int]) -> float:
        if len(others) < 1:
            return 0.5
        M = np.stack([self.vec[e] for e in others])
        d2 = np.sort(((M - v) ** 2).sum(axis=1))[:2]
        return float(1.0 - np.exp(-d2.mean() / (2 * self.dim)))

    def on_query(self, vec):
        self._finish()
        self.t += 1
        self._q = vec
        self._cost, self._served = 1.0, set()

    def _finish(self):
        """Close the previous request: a hit contributes its regret here (a miss contributed 1 when its entry was produced);
        every 5th request the entries that were not served get beta += 0.05, as in the reference code."""
        if self.t == 0:
            return
        if self._served:
            self.regret += self._cost
            self.hist.append(self._cost)
        if self.t % 5 == 0:
            for e in self.a:
                if e not in self._served:
                    self.b[e] += 0.05

    def on_hit(self, entry, vec):
        e = entry.eid
        v = self.vec.get(e)
        sim = float(np.clip(np.dot(self._q, v), 0.0, 1.0)) if v is not None else 1.0
        self._cost = 1.0 - sim
        self._served.add(e)
        self.a[e] += 1.0
        self.last[e] = self.t
        self.cnt[e] += 1

    def on_insert(self, entry, vec):
        e = entry.eid
        self._new = e
        v = np.asarray(vec, dtype=np.float32)
        self.regret += 1.0
        self.hist.append(1.0)
        others = list(self.a)
        self.a[e], self.b[e], self.born[e], self.last[e], self.cnt[e] = 2.0, 1.0, self.t, self.t, 0
        self.nov[e] = self._novelty_of(v, others)
        self.vec[e] = v
        self._admit = True
        if self.mode != "E":
            if len(self.hist) >= self.window:
                Lh = max(float(np.std(self.hist[-self.window:])) * 2, 0.1)
                self.L_ema = Lh if self.L_ema is None else 0.7 * self.L_ema + 0.3 * Lh
                self.tau = float(np.clip(np.sqrt(2 * self.lam / self.L_ema), self.tmin, self.tmax))
            boot = min(max(self.c // 2, 5), 25)
            live = len(self.a) - 1
            if live < boot:
                self._admit, self.regret = True, 0.0
            elif self.regret > self.tau:
                self._admit, self.regret = True, 0.0
            else:
                self._admit = False
        if self._admit:
            self.stores += 1

    def choose_victim(self, cache, vb):
        if not self._admit:
            return self._new
        ids = [e for e in self.a if e != self._new]
        if not ids:
            return self._new
        if self.mode == "A":
            age = np.array([self.t - self.last[e] for e in ids], dtype=np.float64)
            freq = np.array([self.cnt[e] for e in ids], dtype=np.float64) / max(self.t, 1)
            div = np.array([self.nov[e] for e in ids])
            score = 0.4 * np.exp(-age / 100.0) + 0.3 * freq + 0.3 * div
        else:
            a = np.array([self.a[e] for e in ids])
            b = np.array([self.b[e] for e in ids])
            score = self.rng.beta(a, b) + 0.3 * np.array([self.nov[e] for e in ids])
        return ids[int(np.argmin(score))]

    def on_evict(self, eid):
        for d in (self.a, self.b, self.nov, self.vec, self.born, self.last, self.cnt):
            d.pop(eid, None)
