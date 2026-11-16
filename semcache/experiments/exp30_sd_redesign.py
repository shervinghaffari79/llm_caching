"""
Experiment 30 -- redesigned Semantic Decay (SD-RR): decayed frequency,
semantic redundancy and semantic risk.

    score_i = F_i - alpha * R_i - beta * U_i     (lowest score is evicted)

Protocol, fixed before any evaluation run:

Amendment (made after the first tuning round and before any evaluation run;
the first round is kept as exp30_sd_redesign_tune_m2.json): on the log the
QQP-trained C2 had precision below 50%, so every policy had a negative net hit
ratio and the objective rewarded serving fewer hits. The log is therefore
served by C1 at its in-domain threshold (0.895, fitted in experiment 25 on log
queries 50,000-89,999 whose groups never occur in the first 50,000). Beta
reached the edge of its grid and is extended; QQP tuning uses three tuning
streams (seeds 1042-1044) to reduce selection noise; the risk term measures
uncertainty relative to each decision layer's acceptance threshold; the
redundancy radius is the dataset's acceptance cosine; alpha = 2 is dropped
(identical to 0.5 and 1 in the first round). All runs use the simulated
backend, since decisions do not depend on answers.

Tuning uses only data reserved for tuning:
  * QQP: the tuning streams (seeds 1042-1044, 2,000 requests), capacities 300, 500, 800;
  * SemBenchmarkSearchQueries: queries 100,000-109,999 (a window disjoint from
    the evaluated first 50,000), capacities 500, 1,000, 2,000.
C2 serves QQP at its default threshold; C1 serves the log at 0.895.
Grid: alpha in {0, 0.5, 1}, beta in {0, 1, 2, 4, 8, 16, 32}, half-life in
{none, 4x, 16x capacity} requests. The selected setting maximises the mean
net hit ratio (correct-hit ratio minus false-hit rate, i.e. request-level
risk-adjusted saving at lambda = 1) over the twelve dataset-capacity cells.
Reference policies (LRU, LFU, SphereLFU, Semantic Decay fixed and tuned) are
run on the same tuning cells for orientation only.

Evaluation (later phase, not run here): QQP seeds 42-46 and five
10,000-query windows of the first 50,000 log queries; Holm-corrected paired
tests against LRU, LFU and SphereLFU; frontier control against threshold-tuned
LFU; cache redundancy; recovery after a workload shift; confirmation on fresh
QQP seeds 47-51.

Metrics per run: served-hit ratio, correct-hit ratio, precision, false-hit
rate, net hit ratio, and redundancy (distinct equivalence groups per occupied
slot, sampled every 50 requests once the cache is full).

Run: python -m semcache.experiments.exp30_sd_redesign --phase tune --workers 8
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from .. import config as C
from ..cache import (RedundancyRiskDecayPolicy, SemanticDecayPolicy, SphereLFUPolicy,
                     StreamStats, make_policy)
from ..encoders import StageTimer, warmup
from ..llm import AnswerPool, LLMBackend
from ..workload import StreamEvent, Workload
from ._common import (REVISION_DIR, banner, get_stream, hardware_manifest, make_backend,
                      memoized_encoders, save_result, train_metas)

KEY = "C2_bi_geo_lgb"
import os as _os
LOG = (__import__("pathlib").Path(_os.environ["SEMCACHE_LOG"]) if _os.environ.get("SEMCACHE_LOG")
       else C.DATA_DIR / "external" / "sembenchmark_searchqueries.parquet")
LOG_VECS = C.DATA_DIR / "emb_cache" / "exp30_searchqueries_vectors.npz"
QQP_VECS = C.DATA_DIR / "emb_cache" / "exp24_stream_vectors.npz"
QQP_VECS_EXTRA = C.DATA_DIR / "emb_cache" / "exp30_qqp_vectors.npz"
TUNE = {"qqp": {"seeds": (1042, 1043, 1044), "caps": (300, 500, 800)},
        "log": {"start": 100_000, "n": 10_000, "caps": (500, 1000, 2000)}}
EVAL_LOG_WINDOWS = [(s, 10_000) for s in range(0, 50_000, 10_000)]
ALPHAS, BETAS, HALF_LIVES = (0.0, 0.5, 1.0), (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0), (None, 4, 16)
SERVE = {"qqp": ("C2_bi_geo_lgb", {}, C.META_PROB_THRESHOLD, C.BI_COSINE_THRESHOLD),
         "log": ("C1_bi_only", {"bi_threshold": 0.895}, 0.895, 0.895)}   # key, kwargs, tau, radius
REFS = ("LRU", "LFU", "SphereLFU", "SemanticDecay_fixed", "SemanticDecay_tuned")
SAMPLE_EVERY = 50


def grid() -> list[dict]:
    out = []
    for a, b, h in itertools.product(ALPHAS, BETAS, HALF_LIVES):
        out.append({"kind": "SD-RR", "alpha": a, "beta": b, "half_life_mult": h})
    return out


def build(spec: dict, cap: int, data: str = "qqp"):
    k = spec["kind"]
    if k in ("LRU", "LFU"):
        return make_policy(k)
    if k == "SphereLFU":
        return SphereLFUPolicy(capacity=cap)
    if k == "SemanticDecay_fixed":
        return SemanticDecayPolicy()
    if k == "SemanticDecay_tuned":
        return SemanticDecayPolicy(0.0, 0.0, 1.0)          # experiment 18, tuned on seed 1042
    if k == "SD-RR":
        h = spec["half_life_mult"]
        _, _, tau, radius = SERVE[data]
        return RedundancyRiskDecayPolicy(alpha=spec["alpha"], beta=spec["beta"],
                                         half_life=(h * cap if h else None), radius=radius,
                                         accept_threshold=tau)
    raise ValueError(k)


def log_window(start: int, n: int) -> Workload:
    df = pd.read_parquet(LOG).iloc[start:start + n]
    seen, events = set(), []
    for i, (q, g) in enumerate(zip(df["prompt"], df["id_set"])):
        g = int(g)
        events.append(StreamEvent(step=i, query=str(q), topic_id=g, is_repeat=g in seen,
                                  is_exact=False, is_hard_negative=False, session_id=i, turn=0))
        seen.add(g)
    return Workload(events=events, meta={"source": "SemBenchmarkSearchQueries", "start": start, "n": n})


def ensure_log_vectors(windows) -> None:
    """Encode the log queries of the given windows once and store them, so that
    every worker process makes identical decisions."""
    want = []
    for s, n in windows:
        want += pd.read_parquet(LOG).iloc[s:s + n]["prompt"].astype(str).tolist()
    have = {}
    if LOG_VECS.exists():
        z = np.load(LOG_VECS, allow_pickle=True)
        have = dict(zip(z["queries"].tolist(), z["vectors"]))
    todo = sorted(set(want) - set(have))
    if todo:
        from ..encoders import get_bi
        print(f"  encoding {len(todo)} log queries", flush=True)
        vec = get_bi().model.encode(todo, batch_size=256, normalize_embeddings=True,
                                    show_progress_bar=False, convert_to_numpy=True)
        have.update(zip(todo, vec.astype(np.float32)))
        LOG_VECS.parent.mkdir(parents=True, exist_ok=True)
        np.savez(LOG_VECS, queries=np.array(list(have), dtype=object),
                 vectors=np.stack(list(have.values())))


def ensure_qqp_vectors(seeds) -> None:
    """Encode the tuning streams not covered by the recorded stream vectors."""
    have = _vectors()
    todo = sorted({e.query for s in seeds for e in get_stream(s, 2000).events} - set(have))
    if todo:
        from ..encoders import get_bi
        print(f"  encoding {len(todo)} QQP queries", flush=True)
        old = {}
        if QQP_VECS_EXTRA.exists():
            z = np.load(QQP_VECS_EXTRA, allow_pickle=True)
            old = dict(zip(z["queries"].tolist(), z["vectors"]))
        vec = get_bi().model.encode(todo, batch_size=256, normalize_embeddings=True,
                                    show_progress_bar=False, convert_to_numpy=True)
        old.update(zip(todo, vec.astype(np.float32)))
        np.savez(QQP_VECS_EXTRA, queries=np.array(list(old), dtype=object),
                 vectors=np.stack(list(old.values())))


def _vectors() -> dict:
    out = {}
    for p in (QQP_VECS, QQP_VECS_EXTRA, LOG_VECS):
        if p.exists():
            z = np.load(p, allow_pickle=True)
            out.update(zip(z["queries"].tolist(), z["vectors"]))
    return out


def run_one(wl: Workload, backend, meta, cap: int, policy, key: str = KEY, **kw) -> dict:
    from ..pipeline import SemanticCache
    cache = SemanticCache(C.CONFIGS[key], meta=meta, capacity=cap, policy=policy, **kw)
    stats, timer = StreamStats(), StageTimer()
    red = []
    for i, ev in enumerate(wl.events):
        cache.handle(ev.query, ev.topic_id, backend, stats, timer)
        size = len(cache.cachebase)
        if size >= cap and i % SAMPLE_EVERY == 0:
            red.append(len(cache.live_topics) / size)
    s = stats.summary()
    n = max(1, s["n_requests"])
    correct, false = 100.0 * s["true_hits"] / n, 100.0 * s["false_hits"] / n
    return {"hit_ratio_pct": s["hit_ratio_pct"], "correct_hit_ratio_pct": correct,
            "precision_of_hits_pct": s["precision_of_hits_pct"], "false_hit_rate_pct": false,
            "net_hit_ratio_pct": correct - false, "evictions": s["evictions"],
            "distinct_groups_per_slot": float(np.mean(red)) if red else None}


def _worker(tasks: list[dict]) -> list[dict]:
    warmup()
    vecs = _vectors()
    out = [None] * len(tasks)
    groups: dict[tuple, list[int]] = {}
    for i, t in enumerate(tasks):
        groups.setdefault(t["data"], []).append(i)
    with memoized_encoders() as memo:
        memo._bi_cache.update(vecs)
        for data, idx in groups.items():
            key, kw, _, _ = SERVE[data[0]]
            if data[0] == "qqp":
                seed = data[1]
                wl = get_stream(seed, 2000)
                meta = train_metas(seed=seed, keys=(key,)).get(key)
            else:
                seed = C.SEED
                wl, meta = log_window(data[1], data[2]), None
            for i in idx:
                t = tasks[i]
                backend = LLMBackend(mode="sim", pool=AnswerPool(), virtual_time=True, seed=seed)
                out[i] = run_one(wl, backend, meta, t["cap"], build(t["spec"], t["cap"], data[0]), key, **kw)
    return out


def run_tasks(tasks: list[dict], workers: int) -> list[dict]:
    if workers <= 1:
        return _worker(tasks)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    order = sorted(range(len(tasks)), key=lambda i: tasks[i]["data"])
    size = max(1, len(tasks) // (workers * 4))
    chunks = [order[i:i + size] for i in range(0, len(order), size)]
    out = [None] * len(tasks)
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_worker, [tasks[i] for i in ch]): ch for ch in chunks}
        done = 0
        for f in futs:
            for i, r in zip(futs[f], f.result()):
                out[i] = r
            done += len(futs[f])
            print(f"    {done}/{len(tasks)} runs", flush=True)
    return out


def tune(workers: int, tag: str = "") -> dict:
    banner("EXP 30 | SD-RR tuning (tuning data only)")
    ensure_log_vectors([(TUNE["log"]["start"], TUNE["log"]["n"])])
    ensure_qqp_vectors(TUNE["qqp"]["seeds"])
    for s in TUNE["qqp"]["seeds"]:             # train once here, so workers only load
        train_metas(seed=s, keys=(SERVE["qqp"][0],))
    cells = ([(("qqp", s), c) for s in TUNE["qqp"]["seeds"] for c in TUNE["qqp"]["caps"]]
             + [(("log", TUNE["log"]["start"], TUNE["log"]["n"]), c) for c in TUNE["log"]["caps"]])
    specs = grid() + [{"kind": r} for r in REFS]
    tasks = [{"data": d, "cap": c, "spec": s} for s in specs for d, c in cells]
    got = run_tasks(tasks, workers)
    rows = []
    for k, s in enumerate(specs):
        res = got[k * len(cells):(k + 1) * len(cells)]
        rows.append({"spec": s,
                     "cells": [{"data": d[0], "cap": c, **r} for (d, c), r in zip(cells, res)],
                     "objective_net_hit_ratio_pct": float(np.mean([r["net_hit_ratio_pct"] for r in res])),
                     "qqp_net": float(np.mean([r["net_hit_ratio_pct"] for (d, _), r in zip(cells, res) if d[0] == "qqp"])),
                     "log_net": float(np.mean([r["net_hit_ratio_pct"] for (d, _), r in zip(cells, res) if d[0] == "log"]))})
    sd = [r for r in rows if r["spec"]["kind"] == "SD-RR"]
    best = max(sd, key=lambda r: r["objective_net_hit_ratio_pct"])
    print(f"  selected: {best['spec']}  objective {best['objective_net_hit_ratio_pct']:.2f}")
    for r in sorted(rows, key=lambda r: -r["objective_net_hit_ratio_pct"])[:8] + [r for r in rows if r["spec"]["kind"] in REFS]:
        print(f"    {str(r['spec']):95s} obj {r['objective_net_hit_ratio_pct']:6.2f}  qqp {r['qqp_net']:6.2f}  log {r['log_net']:6.2f}")
    out = {"manifest": hardware_manifest({"phase": "tune", "tune": {k: {kk: (list(v) if isinstance(v, tuple) else v)
                                                                       for kk, v in d.items()} for k, d in TUNE.items()},
                                          "grid": {"alpha": ALPHAS, "beta": BETAS, "half_life_mult": HALF_LIVES},
                                          "serve": {k: [v[0], v[1], v[2], v[3]] for k, v in SERVE.items()},
                                          "amendment": "log served by C1 at 0.895; beta grid extended; three QQP tuning streams; relative uncertainty; see module docstring",
                                          "objective": "mean net hit ratio (correct - false) over the twelve tuning cells"}),
           "rows": rows, "selected": best["spec"]}
    save_result(out, "exp30_sd_redesign_tune", tag=tag or "amended")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("tune",), default="tune")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true", help="two grid points, one capacity per dataset")
    a = ap.parse_args()
    if a.smoke:
        ALPHAS, BETAS, HALF_LIVES = (0.0, 1.0), (0.0, 8.0), (None,)
        TUNE["qqp"]["seeds"], TUNE["qqp"]["caps"], TUNE["log"]["caps"], TUNE["log"]["n"] = (1043,), (300,), (500,), 3000
        REFS = ("LFU",)
        tune(a.workers, tag="smoke")
    else:
        tune(a.workers, tag=a.tag)
