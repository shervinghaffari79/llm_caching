"""
Experiment 18 -- eviction policies under capacity pressure (E7).

* Capacities 200-1000 in steps of 100, five evaluation seeds (independent
  streams), C2 serving.
* Policies: LRU, LFU, SphereLFU (Biton & Friedman 2026), Semantic Decay with
  the fixed weights, Semantic Decay
  with weights tuned on a separate tuning stream, and the two-term variant
  (freshness + boost) tuned the same way. Tuning uses a stream with a seed
  outside the evaluation set and is frozen before evaluation.
* Policies are compared on correct hits (served hits whose cached entry is
  equivalent), since a raw hit ratio rewards serving wrong answers.
* Seed-level paired tests against LRU and LFU at every capacity.
* Eviction-time diagnostics: neighbours within the hit radius of the victim,
  cache-wide redundancy, and whether the victim's topic was still served by
  another live entry.

Encoders are memoised by text; latency is not reported here.

Run: python -m semcache.experiments.exp18_eviction
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np

from .. import config as C
from ..cache import (SemanticDecay2Policy, SemanticDecayPolicy, SphereLFUPolicy,
                     make_policy)
from ..encoders import warmup
from ..stats import paired_seed_test, seed_ci
from ._common import (add_run_args, banner, ensure_pool, get_stream,
                      hardware_manifest, make_backend, memoized_encoders,
                      save_result, serve, stream_args, stream_params, train_metas)

CAPACITIES = tuple(range(200, 1001, 100))
TUNE_CAPACITIES = (300, 500, 800)
EVAL_SEEDS = (42, 43, 44, 45, 46)
TUNE_SEED = 1042
KEY = "C2_bi_geo_lgb"


def _metrics(stats: dict) -> dict:
    n = max(1, stats["n_requests"])
    return {"hit_ratio_pct": stats["hit_ratio_pct"],
            "correct_hit_ratio_pct": 100.0 * stats["true_hits"] / n,
            "precision_of_hits_pct": stats["precision_of_hits_pct"],
            "token_saving_correct_only_pct": stats["token_saving_correct_only_pct"],
            "hit_ratio_vs_oracle_pct": stats["hit_ratio_vs_oracle_pct"],
            "evictions": stats["evictions"],
            "false_hit_rate_pct": 100.0 * stats["false_hits"] / n,
            "risk_adjusted_saving_pct": stats.get("risk_adjusted_saving_pct"),
            "break_even_false_hit_cost": stats.get("break_even_false_hit_cost"),
            **{k: stats.get(k) for k in ("mean_victim_neighbors", "mean_cache_neighbors",
                                         "mean_redundant_entry_pct",
                                         "victim_topic_redundant_pct",
                                         "victim_never_hit_pct")}}


def _run(wl, metas, cap, policy, seed, density=False) -> dict:
    r = serve(KEY, wl, make_backend(seed=seed), metas, keep_events=False,
              capacity=cap, policy=policy, log_density=density)
    return _metrics(r.stats)


def tune(metas, n, params, fetch) -> dict:
    """Grid-search weights on a tuning stream disjoint from evaluation seeds,
    maximising mean correct-hit ratio over TUNE_CAPACITIES."""
    wl = get_stream(TUNE_SEED, n, **params)
    ensure_pool(wl.queries(), live=fetch)
    grid3 = [w for w in itertools.product(np.arange(0.0, 1.01, 0.2), repeat=3)
             if abs(sum(w) - 1.0) < 1e-6]
    grid2 = [(round(a, 1), round(1 - a, 1)) for a in np.arange(0.0, 1.01, 0.1)]
    rows3, rows2 = [], []
    for w in grid3:
        v = np.mean([_run(wl, metas, c, SemanticDecayPolicy(*w), TUNE_SEED)
                     ["correct_hit_ratio_pct"] for c in TUNE_CAPACITIES])
        rows3.append({"w": [float(x) for x in w], "mean_correct_hit_ratio_pct": float(v)})
    for wf, wb in grid2:
        v = np.mean([_run(wl, metas, c, SemanticDecay2Policy(wf, wb), TUNE_SEED)
                     ["correct_hit_ratio_pct"] for c in TUNE_CAPACITIES])
        rows2.append({"w": [wf, wb], "mean_correct_hit_ratio_pct": float(v)})
    best3 = max(rows3, key=lambda r: r["mean_correct_hit_ratio_pct"])
    best2 = max(rows2, key=lambda r: r["mean_correct_hit_ratio_pct"])
    fixed = np.mean([_run(wl, metas, c, SemanticDecayPolicy(), TUNE_SEED)
                     ["correct_hit_ratio_pct"] for c in TUNE_CAPACITIES])
    print(f"  tuning stream seed {TUNE_SEED}: best 3-term {best3['w']} "
          f"({best3['mean_correct_hit_ratio_pct']:.2f}%), best 2-term {best2['w']} "
          f"({best2['mean_correct_hit_ratio_pct']:.2f}%), fixed {fixed:.2f}%")
    return {"tune_seed": TUNE_SEED, "capacities": list(TUNE_CAPACITIES),
            "grid3": rows3, "grid2": rows2, "best3": best3, "best2": best2,
            "fixed_weights_score": float(fixed)}


def main(seeds=EVAL_SEEDS, n: int = 2000, params: dict | None = None,
         fetch: bool = True, tag: str = "", capacities=CAPACITIES) -> dict:
    banner("EXP 18 | Eviction policies across capacities and seeds")
    params = params or {}
    warmup()
    with memoized_encoders():
        tuned = tune(train_metas(seed=TUNE_SEED, keys=(KEY,)), n, params, fetch)
        w3, w2 = tuple(tuned["best3"]["w"]), tuple(tuned["best2"]["w"])
        policies = {
            "LRU": lambda cap: make_policy("LRU"),
            "LFU": lambda cap: make_policy("LFU"),
            "SphereLFU": lambda cap: SphereLFUPolicy(capacity=cap),
            "SemanticDecay_fixed": lambda cap: SemanticDecayPolicy(),
            "SemanticDecay_tuned": lambda cap: SemanticDecayPolicy(*w3),
            "SemanticDecay2_tuned": lambda cap: SemanticDecay2Policy(*w2),
        }
        res: dict = {p: {str(c): {} for c in capacities} for p in policies}
        for seed in seeds:
            wl = get_stream(seed, n, **params)
            ensure_pool(wl.queries(), live=fetch)
            metas = train_metas(seed=seed, keys=(KEY,))
            for cap in capacities:
                for name, mk in policies.items():
                    res[name][str(cap)][seed] = _run(wl, metas, cap, mk(cap), seed,
                                                     density=True)
            print(f"  seed {seed} done")

    summary, tests = {}, {}
    for name in policies:
        summary[name] = {str(c): {m: seed_ci([res[name][str(c)][s][m] for s in seeds])
                                  for m in ("correct_hit_ratio_pct", "hit_ratio_pct",
                                            "precision_of_hits_pct",
                                            "victim_topic_redundant_pct",
                                            "mean_victim_neighbors")}
                         for c in capacities}
    for name in ("SphereLFU", "SemanticDecay_fixed", "SemanticDecay_tuned",
                 "SemanticDecay2_tuned"):
        tests[name] = {}
        for ref in ("LRU", "LFU"):
            tests[name][ref] = {str(c): paired_seed_test(
                [res[name][str(c)][s]["correct_hit_ratio_pct"] for s in seeds],
                [res[ref][str(c)][s]["correct_hit_ratio_pct"] for s in seeds])
                for c in capacities}

    out = {"manifest": hardware_manifest({"n_stream": n, "config": KEY,
                                          "stream_params": params,
                                          "eval_seeds": list(seeds),
                                          "latency_valid": False}),
           "tuning": tuned, "per_seed": res, "summary": summary, "tests": tests}
    save_result(out, "exp18_eviction", tag=tag)
    return out


if __name__ == "__main__":
    ap = stream_args(add_run_args(argparse.ArgumentParser()))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(EVAL_SEEDS))
    ap.add_argument("--capacities", type=int, nargs="+", default=list(CAPACITIES))
    a = ap.parse_args()
    main(a.seeds, a.n, stream_params(a), fetch=not a.no_fetch, tag=a.tag,
         capacities=tuple(a.capacities))
