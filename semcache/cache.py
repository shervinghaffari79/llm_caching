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

    def __init__(self, dim: int = C.EMB_DIM, use_faiss: bool = True):
        self.dim = dim
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
                self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
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


POLICIES = {"LRU": LRUPolicy, "LFU": LFUPolicy, "SemanticDecay": SemanticDecayPolicy}


def make_policy(name: str) -> EvictionPolicy:
    return POLICIES[name]()


# --------------------------------------------------------------------------
# Stream-level bookkeeping
# --------------------------------------------------------------------------
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
    tokens_generated: int = 0
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
            "token_saving_pct": 100.0 * self.tokens_served_by_cache / max(1, total_tokens),
            "stage_ms_per_request": {k: v / max(1, self.n)
                                     for k, v in self.stage_ms.items()},
        }
