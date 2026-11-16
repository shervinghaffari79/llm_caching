"""
Experiment 20 -- a longer stream for methods that learn per cache entry.

vCache must observe each cached entry several times before it reuses it, so
its hit ratio depends on how often entries are revisited and therefore on
stream length. This experiment serves one 10,000-request stream (same
generator, same mix of repeats, paraphrases and distractors as the
2,000-request streams) with C1, C2, C5, GPTCache and vCache at several error
bounds, and reports quality overall and per block of 1,000 requests, which
shows whether vCache's hit ratio rises as entries accumulate observations.

Encoders are memoised by text; latency is not reported here.

Run: python -m semcache.experiments.exp20_long_stream --n 10000
"""
from __future__ import annotations

import argparse

import numpy as np

from .. import config as C
from ..baselines import BASELINE_CLASSES, VCacheBaseline
from ..encoders import get_bi, warmup
from ._common import (add_run_args, banner, ensure_pool, get_stream,
                      hardware_manifest, make_backend, memoized_encoders,
                      save_result, serve, stream_args, stream_params, train_metas)

C_KEYS = ("C1_bi_only", "C2_bi_geo_lgb", "C5_gated")
DELTAS = (0.01, 0.02, 0.05)
BLOCK = 1000
KEEP = ("hit_ratio_pct", "precision_of_hits_pct", "false_discovery_rate_pct",
        "token_saving_correct_only_pct", "break_even_false_hit_cost",
        "hit_ratio_vs_oracle_pct", "explores", "exploits", "cold_start_explores",
        "judge_calls_per_request", "cache_entries")


def blocks(events: list[dict], size: int = BLOCK) -> list[dict]:
    out = []
    for i in range(0, len(events), size):
        ev = events[i:i + size]
        hits = [e for e in ev if e["hit"]]
        out.append({"start": i, "n": len(ev),
                    "hit_ratio_pct": 100.0 * len(hits) / max(1, len(ev)),
                    "precision_of_hits_pct": (100.0 * float(np.mean([bool(e["correct"]) for e in hits]))
                                              if hits else None)})
    return out


def main(seed: int = C.SEED, n: int = 10000, params: dict | None = None,
         fetch: bool = True, tag: str = "") -> dict:
    banner(f"EXP 20 | Long stream ({n} requests)")
    params = params or {}
    warmup(include_cross=True)
    wl = get_stream(seed, n, **params)
    cov = ensure_pool(wl.queries(), live=fetch)
    print(f"  stream: {wl.meta['n_events']} events, {wl.meta['n_unique_queries']} unique; pool {cov}")
    metas = train_metas(seed=seed)
    results = {}
    with memoized_encoders():
        for key in C_KEYS:
            r = serve(key, wl, make_backend(seed=seed), metas)
            results[key] = {"stats": {k: r.stats.get(k) for k in KEEP},
                            "blocks": blocks(r.per_event)}
        runs = {"gptcache": lambda: BASELINE_CLASSES["gptcache"](
            bi=get_bi(), threshold=C.BI_COSINE_THRESHOLD)}
        for d in DELTAS:
            runs[f"vcache_d{d:g}"] = (lambda d=d: VCacheBaseline(
                bi=get_bi(), delta=d, insertion="native", seed=seed))
        for name, make in runs.items():
            b = make()
            s = b.run_stream(wl, make_backend(seed=seed))
            results[name] = {"stats": {k: s.get(k) for k in KEEP},
                             "blocks": blocks(b.last_events)}
    for name, v in results.items():
        s, bl = v["stats"], v["blocks"]
        print(f"  {name:16s} hit {s['hit_ratio_pct']:5.1f}%  prec {s['precision_of_hits_pct']:5.1f}%  "
              f"first block {bl[0]['hit_ratio_pct']:5.1f}% -> last block {bl[-1]['hit_ratio_pct']:5.1f}%")

    out = {"manifest": hardware_manifest({"n_stream": n, "stream_params": params,
                                          "latency_valid": False, "block": BLOCK,
                                          "deltas": list(DELTAS)}, seed=seed),
           "workload": wl.meta, "pool": cov, "results": results}
    save_result(out, "exp20_long_stream", seed=seed, tag=tag)
    return out


if __name__ == "__main__":
    ap = stream_args(add_run_args(argparse.ArgumentParser()))
    ap.set_defaults(n=10000)
    a = ap.parse_args()
    main(a.seed, a.n, stream_params(a), fetch=not a.no_fetch, tag=a.tag)
