"""
Cache substrate: CacheBase (query -> response), VectorBase (FAISS index) and
eviction policies.

Dynamic-state note (raised directly in review)
----------------------------------------------
This is a *stateful* cache, not an offline pair scorer. Concretely:

  * The index starts empty. The first occurrence of any query is necessarily a
    miss, no matter how good the similarity model is.
  * A hit is only possible if a semantically equivalent query was inserted
    earlier in the stream *and* has not been evicted since.
  * Every miss mutates the index (insert), and may trigger an eviction, which
    changes what later queries can hit.

Therefore the hit ratio produced by `SemanticCache.run_stream` depends on
arrival order, capacity and eviction policy -- it is a property of the
workload and the cache, not a static function of a labelled pair set. The
`hit_ratio_vs_oracle` field reports how many of the hits that were *possible*
(the query had an equivalent live entry at that moment) were actually taken,
which separates similarity-model error from cold-start and eviction effects.
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .encoders import StageTimer


# --------------------------------------------------------------------------
# Stores
# --------------------------------------------------------------------------
@dataclass
class CacheEntry:
    eid: int
    query: str
    response: str
    prompt_tokens: int
    completion_tokens: int
    created_at: float
    last_access: float
    hits: int = 0
    # VectorQ-style per-entry adaptive threshold state
    tau: float = 0.0
    n_accept: int = 0
    n_reject: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class CacheBase:
    """Key-value store for responses. Redis-shaped API, in-process by default."""

    def __init__(self) -> None:
        self._d: dict[int, CacheEntry] = {}

    def put(self, e: CacheEntry) -> None:
        self._d[e.eid] = e

    def get(self, eid: int) -> CacheEntry | None:
        return self._d.get(eid)

    def delete(self, eid: int) -> None:
        self._d.pop(eid, None)

    def __len__(self) -> int:
        return len(self._d)

    def values(self):
        return self._d.values()


class VectorBase:
    """FAISS inner-product index over L2-normalised embeddings (= cosine).

    Uses IndexFlatIP wrapped in IDMap2 so entries can be removed on eviction
    while keeping stable ids. Falls back to a numpy brute-force scan if FAISS
    is unavailable, with identical semantics.
    """

    def __init__(self, dim: int = C.EMB_DIM, use_faiss: bool = True,
                 kind: str = "flat", hnsw_m: int = 32, ef_search: int = 64):
        """kind="flat" is exact search. kind="hnsw" is approximate (HNSW graph);
        it cannot delete entries, so it is only valid without eviction."""
        self.dim = dim
        self.kind = kind
        self._np_ids: list[int] = []
        self._np_vecs: list[np.ndarray] = []
        self.index = None
        if use_faiss:
            try:
                import faiss
                # Batch-1 searches can stall in FAISS's macOS OpenMP runtime
                # when it joins a multi-threaded pool. Keep FAISS single-threaded
                # without reducing Torch's MPS helper threads.
                faiss.omp_set_num_threads(
                    int(os.environ.get("SEMCACHE_FAISS_THREADS", "1")))
                self._faiss = faiss
                if kind == "hnsw":
                    base = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
                    base.hnsw.efSearch = ef_search
                    self.index = faiss.IndexIDMap2(base)
                elif kind == "flat":
                    self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
                else:
                    raise ValueError(f"unknown index kind {kind!r}")
            except Exception:
                self._faiss = None
        else:
            self._faiss = None

    def add(self, eid: int, vec: np.ndarray) -> None:
        v = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        if self.index is not None:
            self.index.add_with_ids(v, np.array([eid], dtype=np.int64))
        else:
            self._np_ids.append(eid)
            self._np_vecs.append(v[0])

    def remove(self, eid: int) -> None:
        if self.kind != "flat":
            raise RuntimeError(f"{self.kind} index does not support eviction")
        if self.index is not None:
            self.index.remove_ids(np.array([eid], dtype=np.int64))
        else:
            try:
                i = self._np_ids.index(eid)
                self._np_ids.pop(i)
                self._np_vecs.pop(i)
            except ValueError:
                pass

    def search(self, vec: np.ndarray, k: int = 1,
               timer: StageTimer | None = None, stage: str = "vector_search"):
        import contextlib
        ctx = timer.stage(stage) if timer else contextlib.nullcontext()
        with ctx:
            if len(self) == 0:
                return np.array([]), np.array([])
            q = np.asarray(vec, dtype=np.float32).reshape(1, -1)
            kk = min(k, len(self))
            if self.index is not None:
                sims, ids = self.index.search(q, kk)
                return sims[0], ids[0]
            M = np.vstack(self._np_vecs)
            s = M @ q[0]
            order = np.argsort(s)[::-1][:kk]
            return s[order], np.array([self._np_ids[i] for i in order])

    def get_vector(self, eid: int) -> np.ndarray | None:
        if self.index is not None:
            try:
                return self.index.reconstruct(int(eid))
            except Exception:
                return None
        try:
            return self._np_vecs[self._np_ids.index(eid)]
        except ValueError:
            return None

    def __len__(self) -> int:
        if self.index is not None:
            return int(self.index.ntotal)
        return len(self._np_ids)


# --------------------------------------------------------------------------
# Eviction policies
# --------------------------------------------------------------------------
class EvictionPolicy:
    name = "base"

    def on_insert(self, entry: CacheEntry, vec: np.ndarray) -> None: ...
    def on_hit(self, entry: CacheEntry, vec: np.ndarray) -> None: ...
    def on_query(self, vec: np.ndarray) -> None: ...
    def choose_victim(self, cache: CacheBase, vb: VectorBase) -> int: ...
    def on_evict(self, eid: int) -> None: ...


class LRUPolicy(EvictionPolicy):
    name = "LRU"

    def __init__(self) -> None:
        self.order: OrderedDict[int, None] = OrderedDict()

    def on_insert(self, entry, vec):
        self.order[entry.eid] = None

    def on_hit(self, entry, vec):
        self.order.move_to_end(entry.eid)

    def choose_victim(self, cache, vb):
        return next(iter(self.order))

    def on_evict(self, eid):
        self.order.pop(eid, None)


class LFUPolicy(EvictionPolicy):
    name = "LFU"

    def __init__(self) -> None:
        self.freq: dict[int, int] = defaultdict(int)
        self.seq: dict[int, int] = {}
        self._t = 0

    def on_insert(self, entry, vec):
        self._t += 1
        self.freq[entry.eid] = 1
        self.seq[entry.eid] = self._t

    def on_hit(self, entry, vec):
        self.freq[entry.eid] += 1

    def choose_victim(self, cache, vb):
        # ties broken by insertion order (oldest first)
        return min(self.freq, key=lambda e: (self.freq[e], self.seq.get(e, 0)))

    def on_evict(self, eid):
        self.freq.pop(eid, None)
        self.seq.pop(eid, None)


class SemanticDecayPolicy(EvictionPolicy):
    """Score(q,t) = w1*Freshness + w2*Age + w3*Boost.

    Freshness = mean cosine to the N most recent *queries* (hits and misses).
    Age       = exp(-lambda * (t - t0)) with t in stream steps.
    Boost     = mean cosine to the top-K most frequently hit entries.

    All three components are computed from the bi-encoder vectors already in
    VectorBase, so the policy costs one small matrix product at eviction time
    and never invokes the cross-encoder. The eviction cost is measured and
    reported separately in the latency breakdown.
    """
    name = "SemanticDecay"

    def __init__(self, w_fresh: float = C.SD_W_FRESHNESS,
                 w_age: float = C.SD_W_AGE, w_boost: float = C.SD_W_BOOST,
                 lam: float = C.SD_LAMBDA, window: int = C.SD_RECENT_WINDOW,
                 top_k: int = C.SD_FREQ_TOP_K):
        self.w = (w_fresh, w_age, w_boost)
        self.lam, self.window, self.top_k = lam, window, top_k
        self.recent: list[np.ndarray] = []
        self.step = 0
        self.created: dict[int, int] = {}
        self.hits: dict[int, int] = defaultdict(int)
        self.vecs: dict[int, np.ndarray] = {}

    def on_query(self, vec):
        self.step += 1
        self.recent.append(np.asarray(vec, dtype=np.float32))
        if len(self.recent) > self.window:
            self.recent.pop(0)

    def on_insert(self, entry, vec):
        self.created[entry.eid] = self.step
        self.vecs[entry.eid] = np.asarray(vec, dtype=np.float32)

    def on_hit(self, entry, vec):
        self.hits[entry.eid] += 1

    def choose_victim(self, cache, vb):
        ids = list(self.vecs.keys())
        if not ids:
            return next(iter(cache._d))
        M = np.vstack([self.vecs[i] for i in ids])            # (n, d)

        if self.recent:
            R = np.vstack(self.recent)                        # (w, d)
            fresh = (M @ R.T).mean(axis=1)
        else:
            fresh = np.zeros(len(ids), dtype=np.float32)

        age = np.exp(-self.lam * (self.step -
                                  np.array([self.created.get(i, self.step) for i in ids])))

        top = sorted(self.hits.items(), key=lambda kv: -kv[1])[:self.top_k]
        top = [i for i, _ in top if i in self.vecs]
        if top:
            F = np.vstack([self.vecs[i] for i in top])
            boost = (M @ F.T).mean(axis=1)
        else:
            boost = np.zeros(len(ids), dtype=np.float32)

        def z(x):
            s = x.std()
            return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)

        score = self.w[0] * z(fresh) + self.w[1] * z(age) + self.w[2] * z(boost)
        return ids[int(np.argmin(score))]

    def on_evict(self, eid):
        self.vecs.pop(eid, None)
        self.created.pop(eid, None)
        self.hits.pop(eid, None)


class SemanticDecay2Policy(SemanticDecayPolicy):
    """Two-term variant: freshness and boost only, no age term."""
    name = "SemanticDecay2"

    def __init__(self, w_fresh: float = 0.5, w_boost: float = 0.5, **kw):
        super().__init__(w_fresh=w_fresh, w_age=0.0, w_boost=w_boost, **kw)


class SphereLFUPolicy(EvictionPolicy):
    """SphereLFU (Biton & Friedman, 2026, arXiv:2603.03301), following the
    authors' SphereQueryLFU implementation (github.com/dbiton/SemanticCaching).

    Every request spreads one unit of frequency over the cached entries inside
    the hit radius D (L2 distance between unit vectors), in proportion to a
    posterior responsibility

        r_i = (c_i + alpha) exp(-kappa d_i^2 / 2) / sum_j (c_j + alpha) exp(-kappa d_j^2 / 2),

    with kappa = 2 ln(1/eps) / D^2, so the kernel falls to eps at the radius.
    A new entry starts at 1. All counters are halved after every `capacity`
    requests that touched the cache. The victim is the entry with the smallest
    counter (oldest first on ties); the entry being inserted is not a
    candidate, matching the reference, which evicts before inserting.

    D is the L2 equivalent of the bi-encoder hit threshold:
    D = sqrt(2 (1 - cos_threshold)).
    """
    name = "SphereLFU"

    def __init__(self, cos_threshold: float = C.BI_COSINE_THRESHOLD,
                 capacity: int | None = None, alpha: float = 1e-3,
                 eps_at_threshold: float = 0.1):
        self.radius = float(np.sqrt(2.0 * (1.0 - cos_threshold)))
        self.kappa = 2.0 * np.log(1.0 / eps_at_threshold) / self.radius ** 2
        self.alpha = alpha
        self.capacity = capacity
        self.count: dict[int, float] = {}
        self.seq: dict[int, int] = {}
        self._t = 0
        self._since_halving = 0
        self._ids: list[int] = []
        self._slot: dict[int, int] = {}
        self._M = np.zeros((0, 0), dtype=np.float32)
        self._newest: int | None = None

    def _ensure(self, dim: int) -> None:
        if self._M.shape[1] != dim:
            self._M = np.zeros((max(64, len(self._ids)), dim), dtype=np.float32)

    def on_query(self, vec):
        n = len(self._ids)
        if n == 0:
            return
        v = np.asarray(vec, dtype=np.float32)
        d2 = np.maximum(0.0, 2.0 - 2.0 * (self._M[:n] @ v))
        inside = np.flatnonzero(d2 < self.radius ** 2)
        if len(inside) == 0:
            return
        ids = [self._ids[i] for i in inside]
        logw = (np.log(np.array([self.count[e] for e in ids]) + self.alpha)
                - 0.5 * self.kappa * d2[inside])
        w = np.exp(logw - logw.max())
        r = w / w.sum()
        for e, ri in zip(ids, r):
            self.count[e] += float(ri)
        self._since_halving += 1
        if self._since_halving >= (self.capacity or 1000):
            self._since_halving = 0
            for e in self.count:
                self.count[e] *= 0.5

    def on_insert(self, entry, vec):
        v = np.asarray(vec, dtype=np.float32)
        self._ensure(v.shape[0])
        n = len(self._ids)
        if n == self._M.shape[0]:
            self._M = np.vstack([self._M, np.zeros_like(self._M)])
        self._M[n] = v
        self._ids.append(entry.eid)
        self._slot[entry.eid] = n
        self._t += 1
        self.count[entry.eid] = 1.0
        self.seq[entry.eid] = self._t
        self._newest = entry.eid

    def choose_victim(self, cache, vb):
        cands = [e for e in self.count if e != self._newest] or list(self.count)
        return min(cands, key=lambda e: (self.count[e], self.seq[e]))

    def on_evict(self, eid):
        i = self._slot.pop(eid, None)
        if i is not None:
            last = len(self._ids) - 1
            if i != last:
                moved = self._ids[last]
                self._ids[i] = moved
                self._M[i] = self._M[last]
                self._slot[moved] = i
            self._ids.pop()
        self.count.pop(eid, None)
        self.seq.pop(eid, None)


class SemanticAdmissionPolicy(EvictionPolicy):
    """SD-Admit: Semantic Decay redesigned around frequency, semantic demand
    and admission.

    score(e) = hits(e) + beta * demand(e)

    hits(e)    decayed count of e's own hits (the LFU signal);
    demand(e)  decayed count of requests that fell within the hit radius of
               e, hit or miss (how popular e's topic is, not just e).

    Admission (TinyLFU-style): a ring buffer keeps the last `history` request
    vectors with their decayed weights. A new entry's demand is the weight of
    remembered requests within its radius, so it is scored with the same
    evidence as resident entries. If the newcomer has the lowest score, it is
    the victim, i.e. it is not admitted; as in TinyLFU, a tie also rejects
    the newcomer.

    Ties are broken towards entries that have another live entry within the
    radius (their topic stays covered), then towards the oldest.

    Aging: every `period` requests all counts and remembered weights halve,
    so the policy follows shifts in the workload.
    """
    name = "SD-Admit"

    def __init__(self, beta: float = 1.0, cos_radius: float = C.BI_COSINE_THRESHOLD,
                 admission: bool = True, period: int = 1000, history: int = 4000):
        self.beta, self.cos_radius, self.admission = beta, cos_radius, admission
        self.period, self.history = period, history
        self.hits: dict[int, float] = {}
        self.demand: dict[int, float] = {}
        self.seq: dict[int, int] = {}
        self._t = 0
        self._req = 0
        self._ids: list[int] = []
        self._slot: dict[int, int] = {}
        self._M = np.zeros((0, 0), dtype=np.float32)
        self._G = np.zeros((0, 0), dtype=np.float32)    # remembered requests
        self._w = np.zeros(0, dtype=np.float64)
        self._g = 0
        self._newest: int | None = None

    def _ensure(self, dim: int) -> None:
        if self._M.shape[1] != dim:
            self._M = np.zeros((64, dim), dtype=np.float32)
            self._G = np.zeros((self.history, dim), dtype=np.float32)
            self._w = np.zeros(self.history, dtype=np.float64)

    def on_query(self, vec):
        v = np.asarray(vec, dtype=np.float32)
        self._ensure(v.shape[0])
        n = len(self._ids)
        if n:
            inside = np.flatnonzero(self._M[:n] @ v >= self.cos_radius)
            for i in inside:
                self.demand[self._ids[i]] += 1.0
        slot = self._g % self.history
        self._G[slot], self._w[slot] = v, 1.0
        self._g += 1
        self._req += 1
        if self._req % self.period == 0:
            for d in (self.hits, self.demand):
                for e in d:
                    d[e] *= 0.5
            self._w *= 0.5

    def on_hit(self, entry, vec):
        self.hits[entry.eid] += 1.0

    def on_insert(self, entry, vec):
        v = np.asarray(vec, dtype=np.float32)
        self._ensure(v.shape[0])
        n = len(self._ids)
        if n == self._M.shape[0]:
            self._M = np.vstack([self._M, np.zeros_like(self._M)])
        self._M[n] = v
        self._ids.append(entry.eid)
        self._slot[entry.eid] = n
        self._t += 1
        self.seq[entry.eid] = self._t
        self.hits[entry.eid] = 1.0
        m = min(self._g, self.history)
        near = (self._G[:m] @ v) >= self.cos_radius
        self.demand[entry.eid] = float(self._w[:m][near].sum())
        self._newest = entry.eid

    def score(self, e: int) -> float:
        return self.hits[e] + self.beta * self.demand[e]

    def choose_victim(self, cache, vb):
        n = len(self._ids)
        ids = self._ids[:n]
        scores = np.array([self.score(e) for e in ids])
        if not self.admission and self._newest is not None and n > 1:
            scores[self._slot[self._newest]] = np.inf
        low = np.flatnonzero(scores <= scores.min() + 1e-9)
        if len(low) == 1:
            return ids[int(low[0])]
        if self.admission and self._newest is not None and self._slot.get(self._newest) in set(low):
            # TinyLFU rule: a newcomer is admitted only if it beats the victim.
            return self._newest
        sims = self._M[low] @ self._M[:n].T
        sims[np.arange(len(low)), low] = -1.0
        covered = (sims >= self.cos_radius).any(axis=1)
        order = sorted(range(len(low)), key=lambda j: (not covered[j], self.seq[ids[low[j]]]))
        return ids[int(low[order[0]])]

    def on_evict(self, eid):
        i = self._slot.pop(eid, None)
        if i is not None:
            last = len(self._ids) - 1
            if i != last:
                moved = self._ids[last]
                self._ids[i] = moved
                self._M[i] = self._M[last]
                self._slot[moved] = i
            self._ids.pop()
        for d in (self.hits, self.demand, self.seq):
            d.pop(eid, None)


class RiskPricedPolicy(EvictionPolicy):
    """RiskPriced: capacity management that prices false hits.

    Each served hit contributes its expected net value under false-hit cost
    lambda, using the decision layer's confidence p for that hit:

        value(e) = prior + sum_hits decay * (p - lam * (1 - p))

    A confident hit adds value; a low-confidence hit (an entry that keeps
    being matched in an ambiguous neighbourhood) can subtract it.

    * eviction: the entry with the lowest value (oldest on ties);
    * admission (optional): a newcomer starts at `prior` and is admitted only
      if that beats the weakest resident (ties reject, as in TinyLFU);
    * purge (optional): an entry whose value is negative after at least
      `min_hits` hits is removed even when the cache is not full;
    * aging: values decay by half every `period` requests.

    With lam = 0 and p = 1 this reduces to LFU with admission (SD-Admit with
    beta = 0), so the false-hit pricing is the only new ingredient.
    """
    name = "RiskPriced"

    def __init__(self, lam: float = 1.0, prior: float = 1.0, admission: bool = True,
                 purge_negative: bool = True, min_hits: int = 2, period: int = 1000):
        self.lam, self.prior, self.admission = lam, prior, admission
        self.purge_negative, self.min_hits, self.period = purge_negative, min_hits, period
        self.value: dict[int, float] = {}
        self.nhits: dict[int, int] = {}
        self.seq: dict[int, int] = {}
        self._t = 0
        self._req = 0
        self._newest: int | None = None
        self._dirty: list[int] = []

    def on_query(self, vec):
        self._req += 1
        if self._req % self.period == 0:
            for e in self.value:
                self.value[e] *= 0.5

    def on_hit_prob(self, eid: int, p: float) -> None:
        if eid not in self.value:
            return
        p = float(min(max(p, 0.0), 1.0))
        self.value[eid] += p - self.lam * (1.0 - p)
        self.nhits[eid] += 1
        if self.purge_negative and self.nhits[eid] >= self.min_hits and self.value[eid] < 0:
            self._dirty.append(eid)

    def purge(self) -> list[int]:
        out, self._dirty = self._dirty, []
        return [e for e in out if e in self.value and self.value[e] < 0]

    def on_insert(self, entry, vec):
        self._t += 1
        self.value[entry.eid] = self.prior
        self.nhits[entry.eid] = 0
        self.seq[entry.eid] = self._t
        self._newest = entry.eid

    def choose_victim(self, cache, vb):
        ids = list(self.value)
        if not self.admission and self._newest in self.value and len(ids) > 1:
            ids = [e for e in ids if e != self._newest]
        low = min(self.value[e] for e in ids)
        tied = [e for e in ids if self.value[e] <= low + 1e-9]
        if self.admission and self._newest in tied:
            return self._newest
        return min(tied, key=lambda e: self.seq[e])

    def on_evict(self, eid):
        for d in (self.value, self.nhits, self.seq):
            d.pop(eid, None)


class LeCaRPolicy(EvictionPolicy):
    """LeCaR (Vietri et al., HotStorage 2018): online learning over two
    experts, LRU and LFU, adapted to a semantic cache.

    Each eviction follows one expert, drawn with probability given by its
    weight, and the victim is remembered in that expert's history (at most
    `capacity` entries per expert, FIFO). A later request that falls within
    the hit radius of a remembered victim is evidence that evicting it was a
    mistake: the responsible expert's weight is multiplied by
    exp(-lr * d^age), where age is the number of requests since the eviction
    and d = 0.005^(1/capacity), as in the original. In exact-key caches the
    trigger is a request for the evicted key; here it is a request within the
    cosine hit radius of the evicted entry's vector.
    """
    name = "LeCaR"

    def __init__(self, cos_threshold: float = C.BI_COSINE_THRESHOLD,
                 capacity: int | None = None, lr: float = 0.45, seed: int = C.SEED):
        self.radius = cos_threshold
        self.capacity = capacity or 1000
        self.lr = lr
        self.discount = 0.005 ** (1.0 / self.capacity)
        self.w = np.array([0.5, 0.5])                    # [LRU, LFU]
        self.rng = np.random.default_rng(seed)
        self.order: OrderedDict[int, None] = OrderedDict()
        self.freq: dict[int, int] = {}
        self.seq: dict[int, int] = {}
        self.vec: dict[int, np.ndarray] = {}
        self.hist: list[list[tuple[np.ndarray, int]]] = [[], []]
        self._t = 0
        self._req = 0
        self._newest: int | None = None
        self.weight_trace: list[float] = []

    def on_query(self, vec):
        self._req += 1
        v = np.asarray(vec, dtype=np.float32)
        for x in (0, 1):
            h = self.hist[x]
            if not h:
                continue
            sims = np.stack([hv for hv, _ in h]) @ v
            j = int(np.argmax(sims))
            if sims[j] >= self.radius:
                age = self._req - h[j][1]
                self.w[x] *= np.exp(-self.lr * self.discount ** age)
                self.w /= self.w.sum()
                h.pop(j)
        if self._req % 100 == 0:
            self.weight_trace.append(float(self.w[0]))

    def on_insert(self, entry, vec):
        self._t += 1
        self.order[entry.eid] = None
        self.freq[entry.eid] = 1
        self.seq[entry.eid] = self._t
        self.vec[entry.eid] = np.asarray(vec, dtype=np.float32)
        self._newest = entry.eid

    def on_hit(self, entry, vec):
        self.order.move_to_end(entry.eid)
        self.freq[entry.eid] += 1

    def choose_victim(self, cache, vb):
        cands = [e for e in self.order if e != self._newest] or list(self.order)
        expert = int(self.rng.random() >= self.w[0])     # 0 = LRU, 1 = LFU
        if expert == 0:
            victim = cands[0]                            # least recently used
        else:
            victim = min(cands, key=lambda e: (self.freq[e], self.seq[e]))
        h = self.hist[expert]
        h.append((self.vec[victim], self._req))
        if len(h) > self.capacity:
            h.pop(0)
        return victim

    def on_evict(self, eid):
        self.order.pop(eid, None)
        for d in (self.freq, self.seq, self.vec):
            d.pop(eid, None)


class RedundancyRiskDecayPolicy(EvictionPolicy):
    """Semantic Decay, redesigned around decayed frequency, semantic
    redundancy and semantic risk. The victim is the entry with the lowest

        score_i = F_i - alpha * R_i - beta * U_i

    F_i  hit frequency with exponential decay (half-life `half_life` requests;
         None keeps plain counts). A new entry starts at 1, as in LFU.
    R_i  1 when another live entry lies within the cosine radius `radius` and
         has at least the same decayed frequency, i.e. the region of
         embedding space is already covered by an entry that earns as much.
    U_i  mean uncertainty of the hits the entry has served; 0 before its first
         hit. With `accept_threshold` tau, a hit served at score s counts
         (1 - s) / (1 - tau), clipped to [0, 1]: 1 at the acceptance boundary,
         0 at full confidence, so the term is comparable across decision
         layers (meta probability or cosine). Without tau it counts 1 - s.

    Ties go to the oldest entry. The entry being inserted is not a candidate,
    so the policy only evicts and never acts as an admission filter.
    Neighbourhoods are kept incrementally (one similarity row per insert), so
    choosing a victim does not recompute all pairwise similarities.
    """
    name = "SD-RR"

    def __init__(self, alpha: float = 1.0, beta: float = 1.0,
                 half_life: float | None = None, radius: float = C.BI_COSINE_THRESHOLD,
                 accept_threshold: float | None = None):
        self.alpha, self.beta, self.half_life, self.radius = alpha, beta, half_life, radius
        self.tau = accept_threshold
        self.step = 0
        self.F: dict[int, float] = {}
        self.t_last: dict[int, int] = {}
        self.seq: dict[int, int] = {}
        self.u_sum: dict[int, float] = defaultdict(float)
        self.u_n: dict[int, int] = defaultdict(int)
        self.nbrs: dict[int, set[int]] = {}
        self._ids: list[int] = []
        self._slot: dict[int, int] = {}
        self._M = np.zeros((0, 0), dtype=np.float32)
        self._newest: int | None = None

    def _f(self, e: int) -> float:
        if not self.half_life:
            return self.F[e]
        return self.F[e] * 0.5 ** ((self.step - self.t_last[e]) / self.half_life)

    def on_query(self, vec):
        self.step += 1

    def on_insert(self, entry, vec):
        v = np.asarray(vec, dtype=np.float32)
        if self._M.shape[1] != v.shape[0]:
            self._M = np.zeros((64, v.shape[0]), dtype=np.float32)
        n = len(self._ids)
        close = set()
        if n:
            sims = self._M[:n] @ v
            close = {self._ids[i] for i in np.flatnonzero(sims >= self.radius)}
        if n == self._M.shape[0]:
            self._M = np.vstack([self._M, np.zeros_like(self._M)])
        self._M[n] = v
        self._ids.append(entry.eid)
        self._slot[entry.eid] = n
        e = entry.eid
        self.nbrs[e] = close
        for j in close:
            self.nbrs[j].add(e)
        self.F[e], self.t_last[e], self.seq[e] = 1.0, self.step, self.step
        self._newest = e

    def on_hit(self, entry, vec):
        e = entry.eid
        self.F[e] = self._f(e) + 1.0
        self.t_last[e] = self.step

    def on_hit_prob(self, eid, p):
        u = 1.0 - float(p)
        if self.tau is not None and self.tau < 1.0:
            u = min(1.0, max(0.0, u / (1.0 - self.tau)))
        self.u_sum[eid] += u
        self.u_n[eid] += 1

    def score(self, e: int, fcache: dict[int, float]) -> float:
        f = fcache[e]
        r = 1.0 if any(fcache.get(j, -1.0) >= f for j in self.nbrs[e]) else 0.0
        u = self.u_sum[e] / self.u_n[e] if self.u_n[e] else 0.0
        return f - self.alpha * r - self.beta * u

    def choose_victim(self, cache, vb):
        fcache = {e: self._f(e) for e in self.F}
        cands = [e for e in self.F if e != self._newest] or list(self.F)
        return min(cands, key=lambda e: (self.score(e, fcache), self.seq[e]))

    def on_evict(self, eid):
        for j in self.nbrs.pop(eid, set()):
            self.nbrs.get(j, set()).discard(eid)
        i = self._slot.pop(eid, None)
        if i is not None:
            last = len(self._ids) - 1
            if i != last:
                moved = self._ids[last]
                self._ids[i] = moved
                self._M[i] = self._M[last]
                self._slot[moved] = i
            self._ids.pop()
        for d in (self.F, self.t_last, self.seq, self.u_sum, self.u_n):
            d.pop(eid, None)


class SemanticDemandPolicy(EvictionPolicy):
    """Semantic Decay with semantic admission.

    One estimator drives both decisions: the semantic demand of a vector x,
    the aged count of recent requests r with cos(x, r) >= radius, counted over
    all requests, served or not, from a history of recent request embeddings.
    A cached entry starts from the demand its query already had (including
    the request that inserted it) and is credited on every later request
    inside its radius.

    Ageing: "exp" weights a request 0.5 ** (age / half_life); "halve" keeps
    integer counts and halves them (rounding down) every `period` requests,
    as in TinyLFU's reset.

    Eviction removes the resident with the lowest demand (oldest first on
    ties). Admission, for a newly missed query competing with that victim:
      "off"      the newcomer is always cached;
      "compete"  it is cached only if its demand, including the current
                 request, exceeds the victim's (ties go to `tie_admit`);
      "second"   the same, using only the demand it had before the current
                 request, so a first sighting stays out of a full cache until
                 a paraphrase of it arrives.

    With radius close to 1 only identical strings count; with "halve" and
    "compete" this is TinyLFU-style admission with exact counts
    (Einziger et al., 2017).
    """
    name = "SemDemand"

    def __init__(self, radius: float = C.BI_COSINE_THRESHOLD, aging: str = "exp",
                 half_life: float | None = 1000, period: int = 10000,
                 admission: str = "second", tie_admit: bool = False,
                 history: int | None = None):
        if admission is True:
            admission = "compete"
        elif admission is False:
            admission = "off"
        assert aging in ("exp", "halve") and admission in ("off", "compete", "second")
        self.radius, self.aging, self.half_life, self.period = radius, aging, half_life, period
        self.admission, self.tie_admit = admission, tie_admit
        default_w = (4 * half_life if half_life else 10000) if aging == "exp" else 2 * period
        self.W = int(history or default_w)
        self.step = 0
        self.epoch = 0
        self.D: dict[int, float] = {}
        self.t_last: dict[int, int] = {}
        self.seq: dict[int, int] = {}
        self._ids: list[int] = []
        self._slot: dict[int, int] = {}
        self._M = np.zeros((0, 0), dtype=np.float32)
        self._hv: np.ndarray | None = None
        self._ht = np.full(self.W, -1, dtype=np.int64)
        self._he = np.zeros(self.W, dtype=np.int64)
        self._hpos = 0
        self._credited: set[int] = set()
        self._newest: int | None = None
        self._new_prior = 0.0

    # -- ageing ---------------------------------------------------------------
    def _cur(self, e: int) -> float:
        if self.aging == "halve":
            return self.D[e]
        if not self.half_life:
            return self.D[e]
        return float(self.D[e] * 0.5 ** ((self.step - self.t_last[e]) / self.half_life))

    def _credit(self, e: int) -> None:
        self.D[e] = self._cur(e) + 1.0
        self.t_last[e] = self.step
        self._credited.add(e)

    def demand_of(self, v: np.ndarray) -> float:
        """Demand of a vector from the request history (current request included)."""
        live = self._ht >= 0
        if self._hv is None or not live.any():
            return 0.0
        near = (self._hv[live] @ v) >= self.radius
        if self.aging == "halve":
            return float(np.floor((0.5 ** (self.epoch - self._he[live][near])).sum()))
        if not self.half_life:
            return float(near.sum())
        return float((0.5 ** ((self.step - self._ht[live][near]) / self.half_life)).sum())

    # -- events ---------------------------------------------------------------
    def on_query(self, vec):
        self.step += 1
        if self.aging == "halve" and self.step % self.period == 0:
            self.epoch += 1
            for e in self.D:
                self.D[e] = float(np.floor(self.D[e] / 2.0))
        v = np.asarray(vec, dtype=np.float32)
        self._credited = set()
        n = len(self._ids)
        if n:
            for i in np.flatnonzero(self._M[:n] @ v >= self.radius):
                self._credit(self._ids[i])
        if self._hv is None:
            self._hv = np.zeros((self.W, v.shape[0]), dtype=np.float32)
        self._hv[self._hpos] = v
        self._ht[self._hpos] = self.step
        self._he[self._hpos] = self.epoch
        self._hpos = (self._hpos + 1) % self.W

    def on_insert(self, entry, vec):
        v = np.asarray(vec, dtype=np.float32)
        e = entry.eid
        est = max(1.0, self.demand_of(v))            # the inserting request counts once
        self.D[e] = est
        self._new_prior = est - 1.0                  # demand seen before this request
        self.t_last[e], self.seq[e] = self.step, self.step
        if self._M.shape[1] != v.shape[0]:
            self._M = np.zeros((64, v.shape[0]), dtype=np.float32)
        n = len(self._ids)
        if n == self._M.shape[0]:
            self._M = np.vstack([self._M, np.zeros_like(self._M)])
        self._M[n] = v
        self._ids.append(e)
        self._slot[e] = n
        self._newest = e

    def on_hit(self, entry, vec):
        if entry.eid in self.D and entry.eid not in self._credited:
            self._credit(entry.eid)

    def choose_victim(self, cache, vb):
        residents = [e for e in self.D if e != self._newest]
        if not residents:
            return self._newest
        victim = min(residents, key=lambda e: (self._cur(e), self.seq[e]))
        if self.admission != "off" and self._newest in self.D:
            dn = self._new_prior if self.admission == "second" else self._cur(self._newest)
            dv = self._cur(victim)
            if dn < dv or (dn == dv and not self.tie_admit):
                return self._newest
        return victim

    def on_evict(self, eid):
        i = self._slot.pop(eid, None)
        if i is not None:
            last = len(self._ids) - 1
            if i != last:
                moved = self._ids[last]
                self._ids[i] = moved
                self._M[i] = self._M[last]
                self._slot[moved] = i
            self._ids.pop()
        for d in (self.D, self.t_last, self.seq):
            d.pop(eid, None)


class WindowSemanticDemandPolicy(SemanticDemandPolicy):
    """Semantic Decay with a semantic admission window.

    The cache is split into a small LRU window (`window_frac` of capacity)
    and a main segment. Every newly missed query enters the window, so new
    topics are served immediately. When the window overflows, its least
    recently used entry leaves it and competes for the main segment against
    the main entry with the least semantic evidence; it is promoted only if
    its own evidence is strictly larger, and dropped otherwise.

    Semantic evidence is semantic demand (exponentially aged count of recent
    requests inside the cosine radius, including requests that were never
    cached) minus the entry's own inserting request, so a query asked once
    has evidence zero and a query whose paraphrases keep arriving gains it.

    With radius close to 1 only identical strings count, which gives
    W-TinyLFU-style admission with exact counts (Einziger et al., 2017).
    """
    name = "WSemDemand"

    def __init__(self, capacity: int, window_frac: float = 0.05,
                 radius: float = C.BI_COSINE_THRESHOLD, half_life: float | None = 1000,
                 history: int | None = None):
        super().__init__(radius=radius, aging="exp", half_life=half_life,
                         admission="off", history=history)
        self.capacity = capacity
        self.wcap = max(1, int(round(window_frac * capacity)))
        self.window: OrderedDict[int, None] = OrderedDict()
        self.t_ins: dict[int, int] = {}

    def evidence(self, e: int) -> float:
        own = 0.5 ** ((self.step - self.t_ins[e]) / self.half_life) if self.half_life else 1.0
        return max(0.0, self._cur(e) - own)

    def on_insert(self, entry, vec):
        super().on_insert(entry, vec)
        self.t_ins[entry.eid] = self.step
        self.window[entry.eid] = None
        while len(self.window) > self.wcap and len(self.D) <= self.capacity:
            self.window.popitem(last=False)          # room left: move to main unconditionally

    def on_hit(self, entry, vec):
        super().on_hit(entry, vec)
        if entry.eid in self.window:
            self.window.move_to_end(entry.eid)

    def choose_victim(self, cache, vb):
        main = [e for e in self.D if e not in self.window]
        if len(self.window) > self.wcap:
            cand = next(iter(self.window))
            if not main:
                return cand
            victim = min(main, key=lambda e: (self.evidence(e), self.seq[e]))
            if self.evidence(cand) > self.evidence(victim):
                del self.window[cand]                # promoted; the main victim leaves
                return victim
            return cand
        if main:
            return min(main, key=lambda e: (self.evidence(e), self.seq[e]))
        return next(iter(self.window))

    def on_evict(self, eid):
        super().on_evict(eid)
        self.window.pop(eid, None)
        self.t_ins.pop(eid, None)


POLICIES = {"LRU": LRUPolicy, "LFU": LFUPolicy, "SD-RR": RedundancyRiskDecayPolicy,
            "WSemDemand": WindowSemanticDemandPolicy,
            "SemDemand": SemanticDemandPolicy,
            "SemanticDecay": SemanticDecayPolicy,
            "SemanticDecay2": SemanticDecay2Policy, "SphereLFU": SphereLFUPolicy,
            "SD-Admit": SemanticAdmissionPolicy,
            "RiskPriced": RiskPricedPolicy, "LeCaR": LeCaRPolicy}


def make_policy(name: str, **kwargs) -> EvictionPolicy:
    return POLICIES[name](**kwargs)


# --------------------------------------------------------------------------
# Stream-level bookkeeping
# --------------------------------------------------------------------------
RISK_LAMBDAS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)


@dataclass
class StreamStats:
    n: int = 0
    hits: int = 0
    misses: int = 0
    true_hits: int = 0        # hit AND the matched entry was truly equivalent
    false_hits: int = 0       # hit but the matched entry was NOT equivalent
    true_misses: int = 0      # miss and nothing equivalent was live -> correct
    false_misses: int = 0     # miss although an equivalent entry was live
    oracle_hits: int = 0      # an equivalent entry was live at arrival time
    evictions: int = 0
    tokens_served_by_cache: int = 0
    tokens_served_correct: int = 0   # saved by hits on an equivalent entry
    tokens_served_false: int = 0     # "saved" by hits that returned a wrong answer
    tokens_generated: int = 0
    gated: int = 0                   # requests that consulted the cross-encoder gate
    abstained: int = 0               # requests sent to the LLM by the abstention margin
    cross_calls: int = 0             # cross-encoder pairs scored
    latencies_s: list[float] = field(default_factory=list)
    overheads_ms: list[float] = field(default_factory=list)
    hit_latencies_s: list[float] = field(default_factory=list)
    miss_latencies_s: list[float] = field(default_factory=list)
    stage_ms: dict[str, float] = field(default_factory=dict)

    def add_stages(self, snap: dict[str, float]) -> None:
        for k, v in snap.items():
            self.stage_ms[k] = self.stage_ms.get(k, 0.0) + v

    def summary(self) -> dict:
        lat = np.array(self.latencies_s) if self.latencies_s else np.array([0.0])
        ov = np.array(self.overheads_ms) if self.overheads_ms else np.array([0.0])
        hl = np.array(self.hit_latencies_s) if self.hit_latencies_s else np.array([])
        ml = np.array(self.miss_latencies_s) if self.miss_latencies_s else np.array([])
        total_tokens = self.tokens_served_by_cache + self.tokens_generated
        denom = max(1, total_tokens)
        return {
            "n_requests": self.n,
            "hits": self.hits, "misses": self.misses,
            "hit_ratio_pct": 100.0 * self.hits / max(1, self.n),
            "true_hits": self.true_hits, "false_hits": self.false_hits,
            "true_misses": self.true_misses, "false_misses": self.false_misses,
            "precision_of_hits_pct": 100.0 * self.true_hits / max(1, self.hits),
            "oracle_possible_hits": self.oracle_hits,
            "oracle_hit_ratio_pct": 100.0 * self.oracle_hits / max(1, self.n),
            "hit_ratio_vs_oracle_pct": 100.0 * self.true_hits / max(1, self.oracle_hits),
            "evictions": self.evictions,
            "avg_response_time_s": float(lat.mean()),
            "p95_response_time_s": float(np.percentile(lat, 95)),
            "avg_overhead_ms": float(ov.mean()),
            "avg_hit_latency_s": float(hl.mean()) if len(hl) else 0.0,
            "avg_miss_latency_s": float(ml.mean()) if len(ml) else 0.0,
            "tokens_served_by_cache": self.tokens_served_by_cache,
            "tokens_generated": self.tokens_generated,
            "total_tokens_no_cache_equivalent": total_tokens,
            "token_saving_pct": 100.0 * self.tokens_served_by_cache / denom,
            "tokens_served_correct": self.tokens_served_correct,
            "tokens_served_false": self.tokens_served_false,
            "token_saving_correct_only_pct": 100.0 * self.tokens_served_correct / denom,
            "false_discovery_rate_pct": 100.0 * self.false_hits / max(1, self.hits),
            "false_hit_rate_pct": 100.0 * self.false_hits / max(1, self.n),
            # A false hit is charged lambda times the tokens of the call it
            # replaced: lambda = 0 gives the correct-only saving, and the
            # saving reaches zero at lambda = correct tokens / false tokens.
            "risk_adjusted_saving_pct": {
                f"{lam:g}": 100.0 * (self.tokens_served_correct
                                     - lam * self.tokens_served_false) / denom
                for lam in RISK_LAMBDAS},
            "break_even_false_hit_cost": (
                self.tokens_served_correct / self.tokens_served_false
                if self.tokens_served_false else float("inf")),
            "p50_response_time_s": float(np.percentile(lat, 50)),
            "p99_response_time_s": float(np.percentile(lat, 99)),
            "p95_overhead_ms": float(np.percentile(ov, 95)),
            "p99_overhead_ms": float(np.percentile(ov, 99)),
            "gated_pct": 100.0 * self.gated / max(1, self.n),
            "abstained_pct": 100.0 * self.abstained / max(1, self.n),
            "cross_calls_per_request": self.cross_calls / max(1, self.n),
            "stage_ms_per_request": {k: v / max(1, self.n)
                                     for k, v in self.stage_ms.items()},
        }
