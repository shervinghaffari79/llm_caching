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
    policy_name   LRU | LFU | SemanticDecay
    top_k         how many FAISS candidates are re-scored. Only the top-1 is
                  used for the decision when top_k=1; larger values multiply
                  the cross-encoder cost, which is why it is recorded.
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
                 meta_threshold: float = C.META_PROB_THRESHOLD):
        self.cfg = cfg
        self.bi = bi or get_bi()
        self.cross = cross if cross is not None else (get_cross() if cfg.use_cross else None)
        if cfg.use_meta_clf and meta is None:
            raise ValueError(f"{cfg.key} requires a trained meta-classifier")
        self.meta = meta
        self.capacity = capacity
        self.policy: EvictionPolicy = make_policy(policy_name)
        self.policy_name = policy_name
        self.top_k = top_k
        self.bi_threshold = bi_threshold
        self.cross_threshold = cross_threshold
        self.meta_threshold = meta_threshold

        self.cachebase = CacheBase()
        self.vectorbase = VectorBase()
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
        self.topic_of[eid] = topic_id
        self.live_topics[topic_id] = self.live_topics.get(topic_id, 0) + 1

        while self.capacity is not None and len(self.cachebase) > self.capacity:
            victim = self.policy.choose_victim(self.cachebase, self.vectorbase)
            self.cachebase.delete(victim)
            self.vectorbase.remove(victim)
            self.policy.on_evict(victim)
            vt = self.topic_of.pop(victim, None)
            if vt is not None:
                n = self.live_topics.get(vt, 0) - 1
                if n > 0:
                    self.live_topics[vt] = n
                else:
                    self.live_topics.pop(vt, None)
            stats.evictions += 1

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

        decision, matched_eid, cross_s, prob = False, None, None, None
        if len(ids) > 0:
            cand_eid = int(ids[0])
            cand = self.cachebase.get(cand_eid)
            if cand is not None:
                bi_sim = float(sims[0])

                if self.cfg.use_cross:
                    cross_s = self.cross.score_one(query, cand.query, timer=timer,
                                                   stage="cross_encode")

                if self.cfg.decision == "meta":
                    ec = self.vectorbase.get_vector(cand_eid)
                    if ec is None:
                        ec = self.bi.encode_one(cand.query)
                    x = build_online_features(query, cand.query, eq, ec,
                                              cross_s, self.cfg, timer=timer)
                    with timer.stage("meta_infer"):
                        prob = self.meta.predict_proba_one(x)
                    decision = prob >= self.meta_threshold
                else:
                    with timer.stage("threshold_decide"):
                        decision = bi_sim >= self.bi_threshold
                        if self.cfg.use_cross:
                            decision = decision and (cross_s >= self.cross_threshold)

                if decision:
                    matched_eid = cand_eid

        overhead_s = time.perf_counter() - t0
        stats.add_stages(timer.snapshot_ms())

        # 3. serve
        if decision and matched_eid is not None:
            entry = self.cachebase.get(matched_eid)
            entry.hits += 1
            entry.last_access = time.time()
            self.policy.on_hit(entry, eq)
            correct = (self.topic_of.get(matched_eid) == topic_id and topic_id >= 0)

            # Tokens avoided = what this query would have cost the LLM.
            would = backend.pool.get(query)
            saved = would.total_tokens if would else entry.total_tokens
            stats.tokens_served_by_cache += saved

            total_s = time.perf_counter() - t0
            stats.hits += 1
            stats.true_hits += int(correct)
            stats.false_hits += int(not correct)
            stats.hit_latencies_s.append(total_s)
            out_resp, gen_lat = entry.response, 0.0
        else:
            gen = backend.generate(query)
            stats.tokens_generated += gen.total_tokens
            self._insert(query, eq, gen, topic_id, stats)
            total_s = time.perf_counter() - t0
            stats.misses += 1
            stats.false_misses += int(oracle)
            stats.true_misses += int(not oracle)
            stats.miss_latencies_s.append(total_s)
            out_resp, gen_lat = gen.answer, gen.latency_s

        stats.n += 1
        stats.oracle_hits += int(oracle)
        stats.latencies_s.append(total_s)
        stats.overheads_ms.append(overhead_s * 1000.0)

        return {
            "step": stats.n - 1, "hit": bool(decision), "oracle": bool(oracle),
            "overhead_ms": overhead_s * 1000.0, "total_s": total_s,
            "gen_latency_s": gen_lat, "prob": prob, "cross": cross_s,
            "cache_size": len(self.cachebase),
            "stages_ms": timer.snapshot_ms(),
            "response_chars": len(out_resp or ""),
        }

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
                "top_k_rescored": self.top_k,
                "batch_regime": "online", "batch_size": C.ONLINE_BATCH,
                "hardware": C.HardwareProfile.capture().as_dict(),
                "llm_mode": backend.mode, "llm_model": backend.model,
                "workload": wl.meta,
            })

    # ------------------------------------------------------------- analysis
    @staticmethod
    def _quality_adjusted(stats: StreamStats, per_event: list[dict]) -> dict:
        """A false hit serves the wrong answer. Counting its tokens as 'saved'
        overstates the benefit, so we also report savings restricted to
        correct hits."""
        if not per_event:
            return {}
        return {
            "false_hit_rate_pct": 100.0 * stats.false_hits / max(1, stats.n),
            "token_saving_correct_only_pct": 100.0 * (
                stats.tokens_served_by_cache *
                (stats.true_hits / max(1, stats.hits))
            ) / max(1, stats.tokens_served_by_cache + stats.tokens_generated),
        }

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
        lats.append(time.perf_counter() - t0)
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
