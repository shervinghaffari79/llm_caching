"""
Re-implementations of published semantic-caching systems, evaluated on the
same stream, with the same encoder, on the same hardware.

These are *faithful re-implementations of the published decision logic*, not
the authors' original code. That is stated plainly here and must be stated in
the paper: the original GPTCache/SCALM/VectorQ/ContextCache artefacts differ in
engineering detail (vector store, serialisation, ONNX runtime), and a
comparison that swapped those in as well would confound algorithmic quality
with implementation quality. Holding the substrate fixed and varying only the
decision rule is the comparison that isolates the contribution.

Each baseline is deliberately given its best case:
  * GPTCache uses its own default similarity evaluator semantics.
  * SCALM gets its clustering and frequency-ranked retention.
  * VectorQ is fed *ground-truth* verification feedback, which is strictly
    more informative than the LLM-judge signal it would get in production.
    It is therefore an upper bound on VectorQ, not a handicapped version.
  * ContextCache gets the true session boundaries from the workload
    generator, which a deployed system would have to infer.

References
----------
GPTCache    Bang, F. (2023). NLP-OSS @ EMNLP.  https://aclanthology.org/2023.nlposs-1.24/
SCALM       Li, J. et al. (2024). IEEE/ACM IWQoS.
VectorQ     Schroeder, L. G. (2024). MSc thesis, TUM.
vCache      Schroeder, L. G. et al. (2026). vCache: Verified Semantic Prompt
            Caching. ICLR 2026. arXiv:2502.03771 (v5). Re-implemented from the
            paper and the official algorithm (vcache-project/vCache).
ContextCache Yan et al. (2025).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import config as C
from .cache import (CacheBase, CacheEntry, LRUPolicy, StreamStats, VectorBase,
                    make_policy)
from .encoders import BiEncoder, StageTimer, get_bi
from .llm import LLMBackend
from .workload import Workload


# --------------------------------------------------------------------------
@dataclass
class BaselineSpec:
    key: str
    label: str
    citation: str
    notes: str


BASELINES: dict[str, BaselineSpec] = {
    "gptcache": BaselineSpec(
        "gptcache", "GPTCache (single embedding + fixed distance threshold)",
        "Bang 2023, NLP-OSS",
        "Cosine threshold on one bi-encoder; LRU eviction; no re-ranking."),
    "scalm": BaselineSpec(
        "scalm", "SCALM (cluster + frequency-ranked retention)",
        "Li et al. 2024, IWQoS",
        "Online clustering of cached queries; retention ranked by cluster "
        "value; cluster-aware threshold relaxation for high-value clusters."),
    "vectorq": BaselineSpec(
        "vectorq", "VectorQ (per-entry adaptive thresholds)",
        "Schroeder 2024, TUM",
        "Each entry keeps a Beta posterior over match correctness and adapts "
        "its own threshold; fed ground-truth feedback here."),
    "vcache": BaselineSpec(
        "vcache", "vCache (per-entry sigmoid, error bound delta)",
        "Schroeder et al. 2026, ICLR",
        "Per-entry logistic model of correctness vs. similarity with a "
        "confidence-bounded exploration probability; fed ground-truth "
        "feedback (oracle judge) on explored requests."),
    "contextcache": BaselineSpec(
        "contextcache", "ContextCache (two-stage, context-aware re-ranking)",
        "Yan et al. 2025",
        "Stage 1 recall on the current query; stage 2 re-rank using an "
        "attention-pooled session-context vector."),
}


# --------------------------------------------------------------------------
class BaselineCache:
    """Shared substrate; subclasses override `decide` and optionally
    `on_served` / `choose_victim`."""

    key = "baseline"

    def __init__(self, bi: BiEncoder | None = None, capacity: int | None = None,
                 threshold: float = 0.70, policy_name: str = "LRU"):
        self.bi = bi or get_bi()
        self.capacity = capacity
        self.threshold = threshold
        self.policy = make_policy(policy_name)
        self.policy_name = policy_name
        self.cachebase = CacheBase()
        self.vectorbase = VectorBase()
        self.topic_of: dict[int, int] = {}
        self.live_topics: dict[int, int] = {}
        self._next_id = 0

    # -- hooks -------------------------------------------------------------
    def decide(self, query: str, eq: np.ndarray, sims: np.ndarray,
               ids: np.ndarray, ev, timer: StageTimer) -> tuple[bool, int | None]:
        if len(ids) == 0:
            return False, None
        return (float(sims[0]) >= self.threshold), int(ids[0])

    def on_served(self, hit: bool, eid: int | None, correct: bool,
                  sim: float) -> None: ...

    def on_insert_extra(self, eid: int, vec: np.ndarray, ev) -> None: ...

    def should_insert(self, ev) -> bool:
        """Whether a miss inserts its fresh response (default: always)."""
        return True

    def extra_summary(self, n_requests: int) -> dict:
        """Method-specific counters merged into the run summary."""
        return {}

    # -- machinery ---------------------------------------------------------
    def _insert(self, query: str, vec: np.ndarray, gen, topic_id: int,
                stats: StreamStats, ev) -> None:
        eid = self._next_id
        self._next_id += 1
        now = time.time()
        e = CacheEntry(eid=eid, query=query, response=gen.answer,
                       prompt_tokens=gen.prompt_tokens,
                       completion_tokens=gen.completion_tokens,
                       created_at=now, last_access=now, tau=self.threshold)
        self.cachebase.put(e)
        self.vectorbase.add(eid, vec)
        self.policy.on_insert(e, vec)
        self.topic_of[eid] = topic_id
        self.live_topics[topic_id] = self.live_topics.get(topic_id, 0) + 1
        self.on_insert_extra(eid, vec, ev)

        while self.capacity is not None and len(self.cachebase) > self.capacity:
            victim = self.choose_victim()
            self.cachebase.delete(victim)
            self.vectorbase.remove(victim)
            self.policy.on_evict(victim)
            vt = self.topic_of.pop(victim, None)
            if vt is not None:
                n = self.live_topics.get(vt, 0) - 1
                self.live_topics[vt] = n
                if n <= 0:
                    self.live_topics.pop(vt, None)
            self.on_evict_extra(victim)
            stats.evictions += 1

    def choose_victim(self) -> int:
        return self.policy.choose_victim(self.cachebase, self.vectorbase)

    def on_evict_extra(self, eid: int) -> None: ...

    def run_stream(self, wl: Workload, backend: LLMBackend) -> dict:
        stats = StreamStats()
        timer = StageTimer()
        per_event = []
        for ev in wl.events:
            timer.reset()
            t0 = time.perf_counter()
            eq = self.bi.encode_one(ev.query, timer=timer, stage="bi_encode")
            self.policy.on_query(eq)
            sims, ids = self.vectorbase.search(eq, k=5, timer=timer,
                                               stage="vector_search")
            hit, eid = self.decide(ev.query, eq, sims, ids, ev, timer)
            overhead_s = time.perf_counter() - t0
            stats.add_stages(timer.snapshot_ms())

            oracle = ev.topic_id >= 0 and self.live_topics.get(ev.topic_id, 0) > 0
            sim0 = float(sims[0]) if len(sims) else 0.0

            if hit and eid is not None and self.cachebase.get(eid) is not None:
                entry = self.cachebase.get(eid)
                entry.hits += 1
                entry.last_access = time.time()
                self.policy.on_hit(entry, eq)
                correct = (self.topic_of.get(eid) == ev.topic_id and ev.topic_id >= 0)
                would = backend.pool.get(ev.query)
                saved = would.total_tokens if would else entry.total_tokens
                stats.tokens_served_by_cache += saved
                if correct:
                    stats.tokens_served_correct += saved
                else:
                    stats.tokens_served_false += saved
                total_s = time.perf_counter() - t0
                stats.hits += 1
                stats.true_hits += int(correct)
                stats.false_hits += int(not correct)
                stats.hit_latencies_s.append(total_s)
                self.on_served(True, eid, correct, sim0)
            else:
                gen = backend.generate(ev.query)
                stats.tokens_generated += gen.total_tokens
                if self.should_insert(ev):
                    self._insert(ev.query, eq, gen, ev.topic_id, stats, ev)
                total_s = time.perf_counter() - t0
                if getattr(backend, "virtual_time", False):
                    total_s += gen.latency_s
                correct = None
                stats.misses += 1
                stats.false_misses += int(oracle)
                stats.true_misses += int(not oracle)
                stats.miss_latencies_s.append(total_s)
                self.on_served(False, eid, False, sim0)

            stats.n += 1
            stats.oracle_hits += int(oracle)
            stats.latencies_s.append(total_s)
            stats.overheads_ms.append(overhead_s * 1000.0)
            served = bool(hit and eid is not None and correct is not None)
            per_event.append({"step": ev.step, "hit": served, "oracle": bool(oracle),
                              "correct": correct, "bi_sim": sim0,
                              "is_repeat": ev.is_repeat,
                              "is_hard_negative": ev.is_hard_negative,
                              "overhead_ms": overhead_s * 1000.0,
                              "total_s": total_s})

        from .pipeline import SemanticCache
        out = stats.summary()
        out.update(SemanticCache._subset_ratios(per_event))
        out.update(SemanticCache._quality_adjusted(stats, per_event))
        out["system"] = self.key
        out["label"] = BASELINES[self.key].label if self.key in BASELINES else self.key
        out["threshold"] = self.threshold
        out["virtual_time"] = bool(getattr(backend, "virtual_time", False))
        out.update(self.extra_summary(stats.n))
        self.last_events = per_event
        return out


# --------------------------------------------------------------------------
class GPTCacheBaseline(BaselineCache):
    """GPTCache: one embedding model, one global similarity threshold.

    GPTCache's default `SearchDistanceEvaluation` accepts a candidate when its
    distance falls below a fixed bound; with cosine similarity that is a fixed
    similarity floor. Nothing adapts over the stream's lifetime.
    """
    key = "gptcache"

    def __init__(self, threshold: float = 0.70, **kw):
        kw.setdefault("policy_name", "LRU")
        super().__init__(threshold=threshold, **kw)


# --------------------------------------------------------------------------
class SCALMBaseline(BaselineCache):
    """SCALM: cluster cached queries, rank clusters by accumulated value,
    retain high-value clusters and relax the acceptance threshold inside them.

    Online single-pass clustering (assign to nearest centroid above
    `cluster_tau`, else open a new cluster) stands in for the paper's offline
    hierarchical clustering, so that the policy remains causal on a stream.
    """
    key = "scalm"

    def __init__(self, threshold: float = 0.70, cluster_tau: float = 0.60,
                 relax: float = 0.04, **kw):
        kw.setdefault("policy_name", "LFU")
        super().__init__(threshold=threshold, **kw)
        self.cluster_tau, self.relax = cluster_tau, relax
        self.centroids: list[np.ndarray] = []
        self.cluster_hits: list[float] = []
        self.cluster_size: list[int] = []
        self.entry_cluster: dict[int, int] = {}

    def _assign(self, vec: np.ndarray) -> int:
        if self.centroids:
            M = np.vstack(self.centroids)
            s = M @ vec
            j = int(np.argmax(s))
            if s[j] >= self.cluster_tau:
                n = self.cluster_size[j]
                self.centroids[j] = (self.centroids[j] * n + vec) / (n + 1)
                self.centroids[j] /= max(1e-9, np.linalg.norm(self.centroids[j]))
                self.cluster_size[j] += 1
                return j
        self.centroids.append(vec.copy())
        self.cluster_hits.append(0.0)
        self.cluster_size.append(1)
        return len(self.centroids) - 1

    def on_insert_extra(self, eid, vec, ev):
        self.entry_cluster[eid] = self._assign(vec)

    def decide(self, query, eq, sims, ids, ev, timer):
        if len(ids) == 0:
            return False, None
        with timer.stage("scalm_cluster_lookup"):
            eid = int(ids[0])
            cid = self.entry_cluster.get(eid)
            thr = self.threshold
            if cid is not None and self.cluster_hits:
                # High-value clusters get a relaxed threshold, which is the
                # mechanism behind SCALM's reported hit-ratio gain.
                mx = max(self.cluster_hits) or 1.0
                thr -= self.relax * (self.cluster_hits[cid] / mx)
        return float(sims[0]) >= thr, eid

    def on_served(self, hit, eid, correct, sim):
        if hit and eid is not None:
            cid = self.entry_cluster.get(eid)
            if cid is not None:
                self.cluster_hits[cid] += 1.0

    def choose_victim(self) -> int:
        """Evict from the lowest-value cluster first, oldest entry within it."""
        if not self.entry_cluster:
            return self.policy.choose_victim(self.cachebase, self.vectorbase)
        vals = {e: (self.cluster_hits[c] / max(1, self.cluster_size[c]))
                for e, c in self.entry_cluster.items()
                if self.cachebase.get(e) is not None}
        if not vals:
            return self.policy.choose_victim(self.cachebase, self.vectorbase)
        return min(vals, key=lambda e: (vals[e], self.cachebase.get(e).last_access))

    def on_evict_extra(self, eid):
        self.entry_cluster.pop(eid, None)


# --------------------------------------------------------------------------
class VectorQBaseline(BaselineCache):
    """VectorQ: no global threshold; each cached entry learns its own.

    Every entry holds a Beta(a, b) posterior over "a match at my similarity
    level turned out correct". After each decision involving an entry, the
    posterior is updated with verified feedback and the entry's threshold
    moves toward the similarity level at which its posterior mean clears
    `target_precision`.
    """
    key = "vectorq"

    def __init__(self, threshold: float = 0.70, target_precision: float = 0.90,
                 lr: float = 0.05, tau_min: float = 0.45, tau_max: float = 0.97,
                 **kw):
        kw.setdefault("policy_name", "LRU")
        super().__init__(threshold=threshold, **kw)
        self.target = target_precision
        self.lr, self.tau_min, self.tau_max = lr, tau_min, tau_max
        self.alpha: dict[int, float] = {}
        self.beta: dict[int, float] = {}
        self.tau: dict[int, float] = {}
        self._last: tuple[int, float] | None = None

    def on_insert_extra(self, eid, vec, ev):
        self.alpha[eid], self.beta[eid] = 1.0, 1.0     # uniform prior
        self.tau[eid] = self.threshold

    def decide(self, query, eq, sims, ids, ev, timer):
        if len(ids) == 0:
            return False, None
        with timer.stage("vectorq_threshold"):
            eid = int(ids[0])
            t = self.tau.get(eid, self.threshold)
            hit = float(sims[0]) >= t
        self._last = (eid, float(sims[0]))
        return hit, eid

    def on_served(self, hit, eid, correct, sim):
        if self._last is None:
            return
        eid, sim = self._last
        self._last = None
        if eid not in self.tau:
            return
        if hit:
            # Verified feedback: was this an acceptable substitution?
            self.alpha[eid] += float(correct)
            self.beta[eid] += float(not correct)
            post = self.alpha[eid] / (self.alpha[eid] + self.beta[eid])
            # Below target precision -> tighten; comfortably above -> relax.
            delta = self.lr * (self.target - post)
            self.tau[eid] = float(np.clip(self.tau[eid] + delta,
                                          self.tau_min, self.tau_max))
        else:
            # A near-threshold rejection nudges the threshold down slightly so
            # the entry can be reconsidered next time.
            if sim > self.tau[eid] - 0.05:
                self.tau[eid] = float(np.clip(self.tau[eid] - self.lr * 0.25,
                                              self.tau_min, self.tau_max))

    def on_evict_extra(self, eid):
        for d in (self.alpha, self.beta, self.tau):
            d.pop(eid, None)


# --------------------------------------------------------------------------
# Separation-case variance of t for 6..48 observations, from the official
# vCache implementation (used when a logistic fit is perfectly separated and
# the delta-method variance is undefined).
_VCACHE_VAR_MAP = {
    6: 0.035445, 7: 0.028285, 8: 0.026436, 9: 0.021349, 10: 0.019371,
    11: 0.012615, 12: 0.011433, 13: 0.010228, 14: 0.009963, 15: 0.009253,
    16: 0.011674, 17: 0.013015, 18: 0.010897, 19: 0.011841, 20: 0.013081,
    21: 0.010585, 22: 0.014255, 23: 0.012058, 24: 0.013002, 25: 0.011715,
    26: 0.00839, 27: 0.008839, 28: 0.010628, 29: 0.009899, 30: 0.008033,
    31: 0.00457, 32: 0.007335, 33: 0.008932, 34: 0.00729, 35: 0.007445,
    36: 0.00761, 37: 0.011423, 38: 0.011233, 39: 0.006783, 40: 0.005233,
    41: 0.00872, 42: 0.010005, 43: 0.01199, 44: 0.00977, 45: 0.01891,
    46: 0.01513, 47: 0.02109, 48: 0.01531,
}


class VCacheBaseline(BaselineCache):
    """vCache (Schroeder et al., ICLR 2026): verified semantic caching with a
    user-set error bound delta.

    Every cached entry keeps the observations (s, c) of earlier requests whose
    nearest neighbour it was, where s is the cosine similarity and c whether
    reusing its response would have been correct. A logistic model
    Pr(c=1 | s) = sigmoid(gamma (s - t)) is fitted per entry; with the
    pessimistic threshold t' taken from a (1-eps) confidence band, the request
    is served from cache with probability 1 - tau, where

        tau = min_eps 1 - delta / (1 - (1-eps) sigmoid(gamma (s - t'(eps)))).

    Otherwise vCache calls the LLM ("explore"), records (s, c) for the
    neighbour, and inserts the new prompt only if the neighbour's response was
    not correct. Constants follow the official implementation: an entry is
    always explored until it has 6 observations, eps is searched on a
    50-point grid, the variance of t comes from the delta method (or a lookup
    table when the fit is separable), s is rounded to 3 decimals, and each
    new entry starts with the synthetic observations (0, 0) and (1, 1).

    Correctness feedback c is the stream's equivalence label, i.e. an oracle
    judge. In deployment it comes from an LLM judge or string match, which
    the paper runs off the critical path; `judge_calls` counts those checks so
    their cost can be charged separately.

    insertion="native" follows the paper (insert only when c = 0);
    insertion="always" inserts on every miss, like every other system here.
    """
    key = "vcache"

    def __init__(self, delta: float = 0.02, insertion: str = "native",
                 seed: int = C.SEED, min_obs: int = 6, threshold: float = 0.0,
                 **kw):
        if insertion not in ("native", "always"):
            raise ValueError("insertion must be 'native' or 'always'")
        kw.setdefault("policy_name", "LRU")
        super().__init__(threshold=threshold, **kw)
        from scipy.stats import norm
        self.delta, self.insertion, self.min_obs = delta, insertion, min_obs
        self.rng = np.random.default_rng(seed)
        self.eps = np.linspace(1e-6, 1 - 1e-6, 50)
        self.z = norm.ppf(1 - self.eps)
        self.obs: dict[int, list[tuple[float, int]]] = {}
        self._fit: dict[int, tuple[int, float, float, float] | None] = {}
        self._pending: tuple[int, float, int] | None = None
        self._insert_next = True
        self.n_explore = self.n_exploit = self.n_cold = self.judge_calls = 0

    # -- per-entry model ----------------------------------------------------
    def _params(self, eid: int):
        """(t_hat, gamma, var_t) for an entry, refitted only when its
        observations change (the fit is deterministic)."""
        o = self.obs.get(eid, [])
        cached = self._fit.get(eid)
        if cached is not None and cached[0] == len(o):
            return cached[1:]
        from sklearn.linear_model import LogisticRegression
        s = np.array([x for x, _ in o], dtype=float)
        c = np.array([y for _, y in o], dtype=int)
        X = np.column_stack([np.ones_like(s), s])
        try:
            lr = LogisticRegression(penalty=None, solver="lbfgs", tol=1e-8,
                                    max_iter=1000, fit_intercept=False)
            lr.fit(X, c)
        except ValueError:                     # as in the official code: explore
            self._fit[eid] = (len(o), None, None, None)
            return None, None, None
        b0, g = lr.coef_[0]
        gamma = max(float(g), 1e-6)
        t_hat = float(np.clip(-b0 / gamma, 0.0, 1.0))
        if s[c == 1].min() > s[c == 0].max():
            var_t = _VCACHE_VAR_MAP.get(len(o), _VCACHE_VAR_MAP[48])
        else:
            p = lr.predict_proba(X)[:, 1]
            H = X.T @ ((p * (1 - p))[:, None] * X)
            grad = np.array([-1.0 / gamma, b0 / gamma ** 2])
            try:
                var_t = max(0.0, float(grad @ np.linalg.inv(H) @ grad))
            except np.linalg.LinAlgError:
                var_t = _VCACHE_VAR_MAP[48]
        t_hat, gamma = round(t_hat, 3), round(gamma, 3)
        self._fit[eid] = (len(o), t_hat, gamma, var_t)
        return t_hat, gamma, var_t

    def _tau(self, s: float, eid: int) -> float:
        t_hat, gamma, var_t = self._params(eid)
        if t_hat is None:
            return 1.0
        t_prime = np.clip(t_hat + self.z * np.sqrt(var_t), 0.0, 1.0)
        lik = 1.0 / (1.0 + np.exp(-gamma * (s - t_prime)))
        taus = 1.0 - self.delta / (1.0 - (1.0 - self.eps) * lik)
        return float(np.clip(np.min(taus), 0.0, 1.0))

    # -- hooks --------------------------------------------------------------
    def decide(self, query, eq, sims, ids, ev, timer):
        self._pending = None
        self._insert_next = True
        if len(ids) == 0:
            return False, None
        eid, s = int(ids[0]), round(float(sims[0]), 3)
        with timer.stage("vcache_policy"):
            if len(self.obs.get(eid, [])) < self.min_obs:
                explore = True
                self.n_cold += 1
            else:
                explore = self.rng.uniform() <= self._tau(s, eid)
        if not explore:
            self.n_exploit += 1
            return True, eid
        self.n_explore += 1
        self.judge_calls += 1
        c = int(ev.topic_id >= 0 and self.topic_of.get(eid) == ev.topic_id)
        self._pending = (eid, s, c)
        if self.insertion == "native":
            self._insert_next = (c == 0)
        return False, eid

    def should_insert(self, ev) -> bool:
        return self._insert_next

    def on_served(self, hit, eid, correct, sim):
        if self._pending is None:
            return
        eid, s, c = self._pending
        self._pending = None
        if eid in self.topic_of:              # still cached
            self.obs.setdefault(eid, []).append((s, c))

    def on_insert_extra(self, eid, vec, ev):
        # The official metadata starts every entry with two synthetic
        # observations, one per class, so the logistic fit is always defined.
        self.obs[eid] = [(0.0, 0), (1.0, 1)]

    def on_evict_extra(self, eid):
        self.obs.pop(eid, None)
        self._fit.pop(eid, None)

    def extra_summary(self, n_requests: int) -> dict:
        n = max(1, n_requests)
        return {"delta": self.delta, "insertion": self.insertion,
                "explores": self.n_explore, "exploits": self.n_exploit,
                "cold_start_explores": self.n_cold,
                "judge_calls": self.judge_calls,
                "judge_calls_per_request": self.judge_calls / n,
                "cache_entries": len(self.cachebase)}


# --------------------------------------------------------------------------
class ContextCacheBaseline(BaselineCache):
    """ContextCache: two-stage retrieval with a context-integrated query
    representation.

    Stage 1 recalls candidates with the raw query vector (cheap, high recall).
    Stage 2 re-scores each candidate against a fused representation
        z = normalize(alpha * e_q + (1 - alpha) * ctx)
    where `ctx` is a softmax-attention pooling of the preceding turns of the
    same session. A candidate is accepted only if it clears the threshold on
    the raw query *and* does not lose too much ground under the fused
    representation, which is how the system disambiguates queries that are
    lexically alike but contextually different.
    """
    key = "contextcache"

    def __init__(self, threshold: float = 0.70, alpha: float = 0.75,
                 ctx_tolerance: float = 0.08, temperature: float = 0.1, **kw):
        kw.setdefault("policy_name", "LRU")
        super().__init__(threshold=threshold, **kw)
        self.alpha, self.ctx_tol, self.temp = alpha, ctx_tolerance, temperature
        self.session_hist: dict[int, list[np.ndarray]] = {}
        self.entry_ctx: dict[int, np.ndarray] = {}

    def _context(self, sid: int, eq: np.ndarray) -> np.ndarray | None:
        hist = self.session_hist.get(sid)
        if not hist:
            return None
        H = np.vstack(hist[-8:])
        w = np.exp((H @ eq) / max(1e-6, self.temp))
        w /= w.sum()
        ctx = (w[:, None] * H).sum(axis=0)
        n = np.linalg.norm(ctx)
        return ctx / n if n > 1e-9 else None

    def decide(self, query, eq, sims, ids, ev, timer):
        if len(ids) == 0:
            return False, None
        with timer.stage("contextcache_stage2"):
            ctx = self._context(ev.session_id, eq)
            if ctx is None:
                z = eq
            else:
                z = self.alpha * eq + (1.0 - self.alpha) * ctx
                z /= max(1e-9, np.linalg.norm(z))
            best, best_eid = -1.0, None
            for s, i in zip(sims, ids):
                i = int(i)
                if float(s) < self.threshold:
                    continue
                v = self.vectorbase.get_vector(i)
                if v is None:
                    continue
                ec = self.entry_ctx.get(i)
                zc = v if ec is None else (
                    self.alpha * v + (1.0 - self.alpha) * ec)
                zc = zc / max(1e-9, np.linalg.norm(zc))
                ctx_sim = float(z @ zc)
                if ctx_sim >= float(s) - self.ctx_tol and ctx_sim > best:
                    best, best_eid = ctx_sim, i
        return (best_eid is not None), best_eid

    def on_insert_extra(self, eid, vec, ev):
        ctx = self._context(ev.session_id, vec)
        if ctx is not None:
            self.entry_ctx[eid] = ctx

    def run_stream(self, wl: Workload, backend: LLMBackend) -> dict:
        # session history must be updated as the stream advances
        orig_decide = self.decide

        def decide_and_record(query, eq, sims, ids, ev, timer):
            out = orig_decide(query, eq, sims, ids, ev, timer)
            self.session_hist.setdefault(ev.session_id, []).append(eq)
            return out

        self.decide = decide_and_record          # type: ignore[assignment]
        try:
            return super().run_stream(wl, backend)
        finally:
            self.decide = orig_decide            # type: ignore[assignment]

    def on_evict_extra(self, eid):
        self.entry_ctx.pop(eid, None)


BASELINE_CLASSES = {
    "gptcache": GPTCacheBaseline,
    "scalm": SCALMBaseline,
    "vectorq": VectorQBaseline,
    "contextcache": ContextCacheBaseline,
}

# Baselines whose operating point is not a similarity threshold.
DELTA_BASELINES = {"vcache": VCacheBaseline}
