"""
Experiment 14 -- decision-path cost under batching, top-k and concurrency
(E5, E6).

All measurements use the serving configurations' own components against a
fixed index of resident questions:

1. Component decomposition per configuration (bi-encoder, FAISS,
   cross-encoder, feature arithmetic, meta-classifier), batch size 1.
2. Cross-encoder per-pair cost as the batch grows (1..32 pairs per call).
3. Top-k re-ranking cost: one cross-encoder call over k candidates, and k
   meta-classifier inferences.
4. Concurrency: c client threads issue full decision paths for C1, C2 and C4;
   we report per-request latency percentiles and throughput.

Run: python -m semcache.experiments.exp14_latency
"""
from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from .. import config as C
from ..cache import VectorBase
from ..data import build_splits
from ..encoders import StageTimer, get_bi, get_cross, warmup
from ..features import build_online_features
from ._common import (add_run_args, banner, hardware_manifest, save_result,
                      train_metas)

BATCHES = (1, 2, 4, 8, 16, 32)
KS = (1, 3, 5, 10)
CLIENTS = (1, 4, 16)
DECISION_KEYS = ("C1_bi_only", "C2_bi_geo_lgb", "C3_bi_cross", "C4_full_stacking")


def _pct(v) -> dict:
    a = np.asarray(v, float)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)),
            "n": int(len(a))}


class Bench:
    def __init__(self, index_size: int, seed: int):
        abl = build_splits().ablation
        self.rng = np.random.default_rng(seed)
        self.bi, self.ce = get_bi(), get_cross()
        self.corpus = abl["question2"].tolist()[:index_size]
        emb = self.bi.encode_cached(self.corpus, f"latency_index_{index_size}")
        self.emb = emb
        self.vb = VectorBase()
        for i, v in enumerate(emb):
            self.vb.add(i, v)
        self.queries = abl["question1"].tolist()
        self.metas = train_metas(seed=seed)

    def sample(self, n):
        idx = self.rng.choice(len(self.queries), size=min(n, len(self.queries)),
                              replace=False)
        return [self.queries[i] for i in idx]

    def decide(self, key: str, q: str, k: int = 1, timer: StageTimer | None = None):
        cfg = C.CONFIGS[key]
        t = timer or StageTimer()
        eq = self.bi.encode_one(q, timer=t, stage="bi_encode")
        sims, ids = self.vb.search(eq, k=k, timer=t, stage="vector_search")
        cands = [int(i) for i in ids]
        cross = [None] * len(cands)
        if cfg.use_cross:
            if len(cands) == 1:
                cross = [self.ce.score_one(q, self.corpus[cands[0]], timer=t)]
            else:
                cross = list(self.ce.score([(q, self.corpus[c]) for c in cands],
                                           batch_size=len(cands), timer=t))
        if cfg.decision == "meta":
            xs = [build_online_features(q, self.corpus[c], eq, self.emb[c], cs,
                                        cfg, timer=t) for c, cs in zip(cands, cross)]
            with t.stage("meta_infer"):
                self.metas[key].predict_proba(np.vstack(xs))
        return t


def decomposition(b: Bench, n: int) -> dict:
    out = {}
    for key in DECISION_KEYS:
        stages: dict[str, list] = {}
        totals = []
        for q in b.sample(n):
            t = StageTimer()
            t0 = time.perf_counter()
            b.decide(key, q, timer=t)
            totals.append((time.perf_counter() - t0) * 1000.0)
            for s, v in t.snapshot_ms().items():
                stages.setdefault(s, []).append(v)
        out[key] = {"total": _pct(totals),
                    "stages": {s: _pct(v) for s, v in stages.items()}}
        print(f"  {key:18s} total {out[key]['total']['mean_ms']:7.2f} ms "
              f"(p95 {out[key]['total']['p95_ms']:.2f})")
    return out


def cross_batching(b: Bench, n_pairs: int) -> dict:
    qs = b.sample(n_pairs)
    pairs = [(q, b.corpus[i % len(b.corpus)]) for i, q in enumerate(qs)]
    out = {}
    for bs in BATCHES:
        t0 = time.perf_counter()
        for i in range(0, len(pairs), bs):
            b.ce.score(pairs[i:i + bs], batch_size=bs)
        per_pair = (time.perf_counter() - t0) * 1000.0 / len(pairs)
        out[str(bs)] = {"ms_per_pair": per_pair, "ms_per_call": per_pair * bs}
        print(f"  cross-encoder batch {bs:2d}: {per_pair:7.2f} ms/pair")
    return out


def topk_cost(b: Bench, n: int) -> dict:
    out = {}
    for key in ("C3_bi_cross", "C4_full_stacking", "C2_bi_geo_lgb"):
        out[key] = {}
        for k in KS:
            totals = []
            for q in b.sample(n):
                t0 = time.perf_counter()
                b.decide(key, q, k=k)
                totals.append((time.perf_counter() - t0) * 1000.0)
            out[key][str(k)] = _pct(totals)
        print(f"  top-k {key}: " + ", ".join(
            f"k={k} {out[key][str(k)]['mean_ms']:.1f}" for k in KS))
    return out


def concurrency(b: Bench, n_per_client: int) -> dict:
    out = {}
    for key in ("C1_bi_only", "C2_bi_geo_lgb", "C4_full_stacking"):
        out[key] = {}
        for c in CLIENTS:
            lats: list[float] = []
            lock = threading.Lock()
            work = [b.sample(n_per_client) for _ in range(c)]

            def client(qs):
                local = []
                for q in qs:
                    t0 = time.perf_counter()
                    b.decide(key, q)
                    local.append((time.perf_counter() - t0) * 1000.0)
                with lock:
                    lats.extend(local)

            th = [threading.Thread(target=client, args=(w,)) for w in work]
            t0 = time.perf_counter()
            for t in th:
                t.start()
            for t in th:
                t.join()
            wall = time.perf_counter() - t0
            out[key][str(c)] = {**_pct(lats), "throughput_rps": len(lats) / wall}
            print(f"  {key:18s} clients={c:2d} p50 {out[key][str(c)]['p50_ms']:.1f} "
                  f"p99 {out[key][str(c)]['p99_ms']:.1f} ms  "
                  f"{out[key][str(c)]['throughput_rps']:.1f} req/s")
    return out


def main(n: int = 300, index_size: int = 2000, seed: int = C.SEED, tag: str = "",
         skip_concurrency: bool = False) -> dict:
    banner("EXP 14 | Decision-path cost: decomposition, batching, top-k, concurrency")
    warmup(include_cross=True)
    b = Bench(index_size, seed)
    for key in DECISION_KEYS:                          # warm every code path
        for q in b.sample(5):
            b.decide(key, q)
    out = {"manifest": hardware_manifest({"index_size": index_size, "n": n}, seed=seed),
           "decomposition": decomposition(b, n),
           "cross_batching": cross_batching(b, n),
           "topk_cost": topk_cost(b, max(50, n // 3))}
    if not skip_concurrency:
        out["concurrency"] = concurrency(b, max(25, n // 6))
    save_result(out, "exp14_latency", tag=tag)
    return out


if __name__ == "__main__":
    ap = add_run_args(argparse.ArgumentParser())
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--index-size", type=int, default=2000)
    ap.add_argument("--skip-concurrency", action="store_true")
    a = ap.parse_args()
    main(a.n, a.index_size, a.seed, a.tag, a.skip_concurrency)
