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
