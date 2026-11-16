"""
The serving pipeline. One implementation, four configurations.

Everything the paper reports about runtime behaviour -- per-component latency,
end-to-end response time, hit ratio, token savings -- comes out of a single
call to `SemanticCache.run_stream`. They are fields of the same measurement,
taken on the same machine, in the same process, on the same query stream, with
batch size 1. That is what makes them arithmetically consistent:

    avg_response_time
        = mean over requests of (overhead + [0 if hit else generation])
        = avg_overhead + miss_ratio * avg_generation_latency

`reporting.consistency` re-derives that identity from the emitted numbers and
fails loudly if the residual is non-trivial. A table pair that cannot satisfy
it -- e.g. a 540 ms cross-encoder overhead reported next to a 311 ms average
response time -- is rejected by construction rather than published.

`avg_overhead` here is the *per request* overhead, not per hit. It is paid on
hits and misses alike, because the similarity check runs before the outcome is
known. The manuscript's latency table reported overhead "per cache hit", which
is a further source of confusion: the cross-encoder cost is incurred on
misses too.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .cache import (CacheBase, CacheEntry, EvictionPolicy, StreamStats,
                    VectorBase, make_policy)
from .encoders import BiEncoder, CrossEncoderWrapper, StageTimer, get_bi, get_cross
from .features import build_online_features
from .llm import Generation, LLMBackend
from .metaclf import TrainedMeta
from .workload import Workload


@dataclass
class ServeResult:
    config_key: str
    stats: dict
    manifest: dict
    per_event: list[dict] = field(default_factory=list)


class SemanticCache:
    """A stateful semantic cache.

    Parameters
    ----------
    cfg           which components are active (see config.CONFIGS)
    meta          trained meta-classifier; required iff cfg.use_meta_clf
    capacity      max live entries; None = unbounded (no eviction pressure)
    policy_name   LRU | LFU | SemanticDecay | SemanticDecay2
    top_k         FAISS candidates considered. Every candidate is scored by the
                  configuration's decision rule and the best accepted one is
                  served; the cross-encoder scores all k pairs in one batched
                  call, so its cost grows with k.
    gate_band     (lo, hi) for configurations with cfg.gated: the meta
                  probability decides alone outside [lo, hi); inside, the
                  cross-encoder decides against cross_threshold.
    abstain_margin  meta configurations only: a probability within this margin
                  of meta_threshold is sent to the LLM instead of being served.
    """

    def __init__(self, cfg: C.ServingConfig,
                 bi: BiEncoder | None = None,
                 cross: CrossEncoderWrapper | None = None,
                 meta: TrainedMeta | None = None,
                 capacity: int | None = None,
                 policy_name: str = "LRU",
                 top_k: int = 1,
                 bi_threshold: float = C.BI_COSINE_THRESHOLD,
                 cross_threshold: float = C.CROSS_THRESHOLD,
                 meta_threshold: float = C.META_PROB_THRESHOLD,
                 gate_band: tuple[float, float] = C.GATE_BAND,
                 abstain_margin: float = 0.0,
                 policy: EvictionPolicy | None = None,
                 log_responses: bool = False,
                 log_density: bool = False,
                 index_kind: str = "flat",
                 ef_search: int = 64):
        self.cfg = cfg
        self.bi = bi or get_bi()
        needs_cross = cfg.use_cross or cfg.gated
        self.cross = cross if cross is not None else (get_cross() if needs_cross else None)
        if cfg.use_meta_clf and meta is None:
            raise ValueError(f"{cfg.key} requires a trained meta-classifier")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.meta = meta.set_inference_threads() if meta is not None else None
        self.capacity = capacity
        self.policy: EvictionPolicy = policy or make_policy(policy_name)
        self.policy_name = self.policy.name
        self.top_k = top_k
        self.bi_threshold = bi_threshold
        self.cross_threshold = cross_threshold
        self.meta_threshold = meta_threshold
        self.gate_band = gate_band
        self.abstain_margin = abstain_margin
        self.log_responses = log_responses
        # Eviction-time redundancy diagnostics (analysis only; never timed).
        self.log_density = log_density
        self._vecs: dict[int, np.ndarray] = {}
        self.eviction_log: list[dict] = []

        if index_kind != "flat" and capacity is not None:
            raise ValueError("approximate indexes cannot evict; use capacity=None")
        self.cachebase = CacheBase()
        self.vectorbase = VectorBase(
            dim=self.bi.model.get_sentence_embedding_dimension(),
            kind=index_kind, ef_search=ef_search)
        self.index_kind = index_kind
        self._next_id = 0
        self.topic_of: dict[int, int] = {}      # eid -> ground-truth topic
        self.live_topics: dict[int, int] = {}   # topic -> count of live entries

    # ---------------------------------------------------------------- utils
    def _insert(self, query: str, vec: np.ndarray, gen: Generation,
                topic_id: int, stats: StreamStats) -> None:
        eid = self._next_id
        self._next_id += 1
        now = time.time()
        e = CacheEntry(eid=eid, query=query, response=gen.answer,
                       prompt_tokens=gen.prompt_tokens,
                       completion_tokens=gen.completion_tokens,
                       created_at=now, last_access=now)
        self.cachebase.put(e)
        self.vectorbase.add(eid, vec)
        self.policy.on_insert(e, vec)
        if self.log_density:
            self._vecs[eid] = np.asarray(vec, dtype=np.float32)
        self.topic_of[eid] = topic_id
        self.live_topics[topic_id] = self.live_topics.get(topic_id, 0) + 1

        while self.capacity is not None and len(self.cachebase) > self.capacity:
            self._evict(self.policy.choose_victim(self.cachebase, self.vectorbase), stats)

    def _evict(self, victim: int, stats: StreamStats) -> None:
        if self.log_density:
            self._record_density(victim)
        self.cachebase.delete(victim)
        self.vectorbase.remove(victim)
        self.policy.on_evict(victim)
        self._vecs.pop(victim, None)
        vt = self.topic_of.pop(victim, None)
        if vt is not None:
            n = self.live_topics.get(vt, 0) - 1
            if n > 0:
                self.live_topics[vt] = n
            else:
                self.live_topics.pop(vt, None)
        stats.evictions += 1

    def _record_density(self, victim: int) -> None:
        """How crowded is the cache when an entry is evicted, and was the
        victim redundant? Neighbours are live entries within the C1 hit radius
        (cosine >= bi_threshold). 'Topic-redundant' means another live entry
        shares the victim's ground-truth topic, so evicting it loses no
        servable answer."""
        ids = list(self._vecs)
        M = np.vstack([self._vecs[i] for i in ids])
        S = M @ M.T
        np.fill_diagonal(S, -1.0)
        nb = (S >= self.bi_threshold).sum(axis=1)
        v = ids.index(victim)
        vt = self.topic_of.get(victim)
        self.eviction_log.append({
            "n_live": len(ids),
            "victim_neighbors": int(nb[v]),
            "mean_neighbors": float(nb.mean()),
            "redundant_entry_pct": 100.0 * float((nb > 0).mean()),
            "victim_topic_redundant": bool(vt is not None
                                           and self.live_topics.get(vt, 0) > 1),
            "victim_hits": int(self.cachebase.get(victim).hits),
        })

    def density_summary(self) -> dict:
        if not self.eviction_log:
            return {}
        L = self.eviction_log
        return {
            "evictions_logged": len(L),
            "mean_victim_neighbors": float(np.mean([e["victim_neighbors"] for e in L])),
            "mean_cache_neighbors": float(np.mean([e["mean_neighbors"] for e in L])),
            "mean_redundant_entry_pct": float(np.mean([e["redundant_entry_pct"] for e in L])),
            "victim_topic_redundant_pct": 100.0 * float(np.mean(
                [e["victim_topic_redundant"] for e in L])),
            "victim_never_hit_pct": 100.0 * float(np.mean(
                [e["victim_hits"] == 0 for e in L])),
        }

    def _oracle_live(self, topic_id: int) -> bool:
        """Was a genuinely equivalent entry live in the cache at this moment?

        This is the ceiling on achievable hits for this request. Comparing the
        realised hit ratio against it separates similarity-model error from
        cold-start misses and eviction-induced misses.
        """
        if topic_id < 0:
            return False
        return self.live_topics.get(topic_id, 0) > 0

    # -------------------------------------------------------------- serving
    def _score_cross(self, query: str, cand_queries: list[str],
                     timer: StageTimer) -> list[float]:
        """Batch-1 for a single pair (the online regime); one batched call for
        several candidates, so top-k re-ranking pays one forward pass."""
        if len(cand_queries) == 1:
            return [self.cross.score_one(query, cand_queries[0], timer=timer,
                                         stage="cross_encode")]
        pairs = [(query, c) for c in cand_queries]
        return [float(v) for v in self.cross.score(
            pairs, batch_size=len(pairs), timer=timer, stage="cross_encode")]

    def _decide(self, query: str, eq: np.ndarray,
                cands: list[tuple[float, int, CacheEntry]],
                timer: StageTimer) -> tuple[int | None, dict]:
        """Apply the configuration's decision rule to the retrieved candidates
        (FAISS order, highest cosine first). Returns the index of the accepted
        candidate, or None for a miss."""
        cfg = self.cfg
        info = {"prob": None, "cross": None, "gated": False,
                "abstained": False, "n_cross": 0}
        cross: list[float | None] = [None] * len(cands)
        if cfg.use_cross:
            cross = self._score_cross(query, [c.query for _, _, c in cands], timer)
            info["n_cross"] = len(cands)

        if cfg.decision == "meta":
            xs = []
            for i, (_, eid, c) in enumerate(cands):
                ec = self.vectorbase.get_vector(eid)
                if ec is None:
                    ec = self.bi.encode_one(c.query)
                xs.append(build_online_features(query, c.query, eq, ec, cross[i], cfg,
                                                timer=timer))
            with timer.stage("meta_infer"):
                probs = [float(p) for p in self.meta.predict_proba(np.vstack(xs))]
            best = int(np.argmax(probs))
            info["prob"] = float(probs[best])
            if cfg.use_cross:
                info["cross"] = cross[best]

            if cfg.gated:
                lo, hi = self.gate_band
                if probs[best] >= hi:
                    return best, info
                band = [i for i, p in enumerate(probs) if lo <= p < hi]
                if not band:
                    return None, info
                info["gated"] = True
                info["n_cross"] = len(band)
                cs = self._score_cross(query, [cands[i][2].query for i in band], timer)
                j = int(np.argmax(cs))
                info["cross"] = cs[j]
                return (band[j] if cs[j] >= self.cross_threshold else None), info

            if (self.abstain_margin > 0
                    and abs(probs[best] - self.meta_threshold) < self.abstain_margin):
                info["abstained"] = True
                return None, info
            return (best if probs[best] >= self.meta_threshold else None), info

        with timer.stage("threshold_decide"):
            ok = [i for i, (s, _, _) in enumerate(cands) if s >= self.bi_threshold]
            if cfg.use_cross:
                ok = [i for i in ok if cross[i] >= self.cross_threshold]
                best = max(ok, key=lambda i: cross[i]) if ok else None
            else:
                best = ok[0] if ok else None
        if cfg.use_cross:
            info["cross"] = cross[best if best is not None else 0]
        return best, info

    def handle(self, query: str, topic_id: int, backend: LLMBackend,
               stats: StreamStats, timer: StageTimer) -> dict:
        timer.reset()
        t0 = time.perf_counter()

        # 1. embed (always: every configuration needs a retrieval key)
        eq = self.bi.encode_one(query, timer=timer, stage="bi_encode")
        self.policy.on_query(eq)

        # 2. retrieve
        sims, ids = self.vectorbase.search(eq, k=self.top_k, timer=timer,
                                           stage="vector_search")
        oracle = self._oracle_live(topic_id)

        cands: list[tuple[float, int, CacheEntry]] = []
        for s, eid in zip(sims, ids):
            eid = int(eid)
            entry = self.cachebase.get(eid) if eid >= 0 else None
            if entry is not None:
                cands.append((float(s), eid, entry))

        # 3. decide
        if cands:
            accepted, info = self._decide(query, eq, cands, timer)
        else:
            accepted, info = None, {"prob": None, "cross": None, "gated": False,
                                    "abstained": False, "n_cross": 0}
        decision = accepted is not None
        matched_eid = cands[accepted][1] if decision else None

        overhead_s = time.perf_counter() - t0
        stats.add_stages(timer.snapshot_ms())

        # 4. serve
        correct = None
        matched_response = None
        if decision:
            entry = self.cachebase.get(matched_eid)
            entry.hits += 1
            entry.last_access = time.time()
            self.policy.on_hit(entry, eq)
            if hasattr(self.policy, "on_hit_prob"):
                # Policies that price false hits are told how confident the
                # decision was (meta-classifier probability; cosine for C1).
                p = info.get("prob")
                self.policy.on_hit_prob(matched_eid, cands[accepted][0] if p is None else p)
            correct = (self.topic_of.get(matched_eid) == topic_id and topic_id >= 0)

            # Tokens avoided = what this query would have cost the LLM.
            would = backend.pool.get(query)
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
            matched_response = entry.response
            out_resp, gen_lat = entry.response, 0.0
        else:
            gen = backend.generate(query)
            stats.tokens_generated += gen.total_tokens
            self._insert(query, eq, gen, topic_id, stats)
            total_s = time.perf_counter() - t0
            if getattr(backend, "virtual_time", False):
                total_s += gen.latency_s
            stats.misses += 1
            stats.false_misses += int(oracle)
            stats.true_misses += int(not oracle)
            stats.miss_latencies_s.append(total_s)
            out_resp, gen_lat = gen.answer, gen.latency_s

        stats.n += 1
        stats.oracle_hits += int(oracle)
        stats.latencies_s.append(total_s)
        stats.overheads_ms.append(overhead_s * 1000.0)
        stats.gated += int(info["gated"])
        stats.abstained += int(info["abstained"])
        stats.cross_calls += info["n_cross"]

        rec = {
            "step": stats.n - 1, "hit": decision, "oracle": bool(oracle),
            "correct": correct,
            "overhead_ms": overhead_s * 1000.0, "total_s": total_s,
            "gen_latency_s": gen_lat, "prob": info["prob"], "cross": info["cross"],
            "bi_sim": cands[0][0] if cands else None,
            "k_retrieved": len(cands), "accepted_rank": accepted,
            "matched_eid": matched_eid,
            "matched_topic": self.topic_of.get(matched_eid) if decision else None,
            "gated": info["gated"], "abstained": info["abstained"],
            "n_cross": info["n_cross"],
            "cache_size": len(self.cachebase),
            "stages_ms": timer.snapshot_ms(),
            "response_chars": len(out_resp or ""),
        }
        if self.log_responses:
            rec["query"] = query
            rec["matched_query"] = cands[accepted][2].query if decision else None
            rec["matched_response"] = matched_response
        if hasattr(self.policy, "purge"):
            # Risk-priced policies may drop entries whose expected value has
            # turned negative, independently of capacity.
            for v in self.policy.purge():
                if self.cachebase.get(v) is not None:
                    self._evict(v, stats)
        return rec

    def run_stream(self, wl: Workload, backend: LLMBackend,
                   keep_events: bool = True) -> ServeResult:
        stats = StreamStats()
        timer = StageTimer()
        per_event: list[dict] = []
        for ev in wl.events:
            rec = self.handle(ev.query, ev.topic_id, backend, stats, timer)
            if keep_events:
                rec.update({"is_repeat": ev.is_repeat, "is_exact": ev.is_exact,
                            "is_hard_negative": ev.is_hard_negative,
                            "topic_id": ev.topic_id})
                per_event.append(rec)

        summary = stats.summary()
        summary.update(self._quality_adjusted(stats, per_event))
        summary.update(self._subset_ratios(per_event))
        summary.update(self.density_summary())
        return ServeResult(
            config_key=self.cfg.key, stats=summary, per_event=per_event,
            manifest={
                "config": self.cfg.key,
                "config_label": self.cfg.label,
                "components": {"bi_encoder": self.cfg.use_bi,
                               "cross_encoder": self.cfg.use_cross,
                               "geometric_features": self.cfg.use_geometric,
                               "text_features": self.cfg.use_text,
                               "lightgbm_meta": self.cfg.use_meta_clf},
                "decision_rule": self.cfg.decision,
                "capacity": self.capacity, "eviction_policy": self.policy_name,
                "top_k": self.top_k,
                "index_kind": self.index_kind, "bi_encoder": self.bi.name,
                "thresholds": {"bi": self.bi_threshold, "cross": self.cross_threshold,
                               "meta": self.meta_threshold},
                "gate_band": list(self.gate_band) if self.cfg.gated else None,
                "abstain_margin": self.abstain_margin,
                "virtual_time": bool(getattr(backend, "virtual_time", False)),
                "llm_pool_misses": int(getattr(backend, "pool_misses", 0)),
                "batch_regime": "online", "batch_size": C.ONLINE_BATCH,
                "hardware": C.HardwareProfile.capture().as_dict(),
                "llm_mode": backend.mode, "llm_model": backend.model,
                "workload": wl.meta,
            })

    # ------------------------------------------------------------- analysis
    @staticmethod
    def _quality_adjusted(stats: StreamStats, per_event: list[dict]) -> dict:
        """Correct-only and risk-adjusted savings are exact per-hit sums in
        StreamStats.summary(); nothing to add here beyond that."""
        return {}

    @staticmethod
    def _subset_ratios(per_event: list[dict]) -> dict:
        """Hit ratio on repeats and correct-miss ratio on distractors -- the
        two quantities the paper's Figure 2 reports, but computed on a live
        stream so they include cold-start and eviction effects."""
        if not per_event:
            return {}
        rep = [e for e in per_event if e["is_repeat"]]
        hn = [e for e in per_event if e["is_hard_negative"]]
        novel = [e for e in per_event if not e["is_repeat"]]
        out = {}
        if rep:
            out["hit_ratio_on_repeats_pct"] = 100.0 * np.mean([e["hit"] for e in rep])
            orc = [e for e in rep if e["oracle"]]
            out["hit_ratio_on_servable_repeats_pct"] = (
                100.0 * np.mean([e["hit"] for e in orc]) if orc else float("nan"))
            out["n_repeats"] = len(rep)
            out["n_servable_repeats"] = len(orc)
        if hn:
            out["correct_miss_ratio_on_hard_negatives_pct"] = (
                100.0 * np.mean([not e["hit"] for e in hn]))
            out["n_hard_negatives"] = len(hn)
        if novel:
            out["correct_miss_ratio_on_novel_pct"] = (
                100.0 * np.mean([not e["hit"] for e in novel]))
        return out


# --------------------------------------------------------------------------
def no_cache_baseline(wl: Workload, backend: LLMBackend) -> dict:
    """Every request goes to the LLM. This is the honest denominator for the
    speed-up factor: it is measured on the same stream, same process, same
    sequential (batch-1) regime as the cached runs."""
    lats, tokens = [], 0
    for ev in wl.events:
        t0 = time.perf_counter()
        g = backend.generate(ev.query)
        dt = time.perf_counter() - t0
        if getattr(backend, "virtual_time", False):
            dt += g.latency_s
        lats.append(dt)
        tokens += g.total_tokens
    v = np.array(lats)
    return {
        "config": C.NO_CACHE_KEY,
        "n_requests": len(v),
        "avg_response_time_s": float(v.mean()),
        "p95_response_time_s": float(np.percentile(v, 95)),
        "total_tokens": int(tokens),
        "token_saving_pct": 0.0,
        "avg_overhead_ms": 0.0,
    }
