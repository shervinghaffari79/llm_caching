"""
Experiment 19 -- published baselines across independent seeds (E2, R2.4, R3.3).

Every baseline runs on the same five seed streams as experiment 11, with the
same encoder, index and answer pool, so its numbers line up with C1-C5:

* threshold baselines (GPTCache, SCALM, VectorQ, ContextCache) at the C1
  validation threshold, so all similarity-threshold systems share one
  operating rule;
* vCache (ICLR 2026) at several error bounds delta, with its own insertion
  rule. Its decisions are randomised; the generator is seeded with the stream
  seed, so the five seeds also average over its randomness.

vCache's correctness feedback is the stream's equivalence label (an oracle
judge). Its token saving is also reported with each feedback check charged as
one judge call of `--judge-tokens` tokens (default: the mean judge call
measured in experiment 16), since a deployment pays for that check.

Seed-level paired tests compare each baseline with C1 and C2 using the
per-seed results of experiment 11 (`--c-results`).

Encoders are memoised by text; latency is not reported here.

Run: python -m semcache.experiments.exp19_baselines_seeds
"""
from __future__ import annotations

import argparse
import json

from .. import config as C
from ..baselines import BASELINE_CLASSES, VCacheBaseline
from ..encoders import get_bi, warmup
from ..stats import paired_seed_test, seed_ci
from ._common import (add_run_args, banner, ensure_pool, get_stream,
                      hardware_manifest, make_backend, memoized_encoders,
                      save_result, stream_args, stream_params)

SEEDS = (42, 43, 44, 45, 46)
DELTAS = (0.01, 0.02, 0.05)
METRICS = ("hit_ratio_pct", "precision_of_hits_pct", "false_discovery_rate_pct",
           "false_hit_rate_pct", "token_saving_correct_only_pct",
           "break_even_false_hit_cost", "judge_calls_per_request",
           "token_saving_correct_only_judge_charged_pct")
JUDGE_TOKENS = 780   # mean tokens per judge call in experiment 16 (calibration)


def _row(stats: dict, judge_tokens: int) -> dict:
    out = {k: stats.get(k) for k in METRICS}
    calls = stats.get("judge_calls") or 0
    total = stats.get("total_tokens_no_cache_equivalent") or 0
    if total:
        out["token_saving_correct_only_judge_charged_pct"] = (
            stats["token_saving_correct_only_pct"] - 100.0 * calls * judge_tokens / total)
    else:
        out["token_saving_correct_only_judge_charged_pct"] = None
    for k in ("explores", "exploits", "cold_start_explores", "judge_calls",
              "cache_entries", "threshold", "delta", "insertion"):
        if k in stats:
            out[k] = stats[k]
    return out


def systems(threshold: float):
    out = {name: (lambda cls=cls, seed=None: cls(bi=get_bi(), threshold=threshold))
           for name, cls in BASELINE_CLASSES.items()}
    for d in DELTAS:
        out[f"vcache_d{d:g}"] = (lambda seed, d=d: VCacheBaseline(
            bi=get_bi(), delta=d, insertion="native", seed=seed))
    return out


def main(seeds=SEEDS, n: int = 2000, params: dict | None = None, fetch: bool = True,
         tag: str = "", threshold: float = C.BI_COSINE_THRESHOLD,
         judge_tokens: int = JUDGE_TOKENS, c_results: str | None = None) -> dict:
    banner("EXP 19 | Published baselines across seeds")
    params = params or {}
    warmup()
    per_seed: dict[int, dict] = {}
    with memoized_encoders():
        for seed in seeds:
            wl = get_stream(seed, n, **params)
            ensure_pool(wl.queries(), live=fetch)
            per_seed[seed] = {}
            for name, make in systems(threshold).items():
                s = make(seed=seed).run_stream(wl, make_backend(seed=seed))
                per_seed[seed][name] = _row(s, judge_tokens)
            print(f"  seed {seed}: " + ", ".join(
                f"{k} {v['hit_ratio_pct']:.1f}%/{v['precision_of_hits_pct']:.1f}%"
                for k, v in per_seed[seed].items()))

    names = list(per_seed[seeds[0]])
    across = {k: {m: seed_ci([per_seed[s][k].get(m) for s in seeds]) for m in METRICS}
              for k in names}

    tests = {}
    if c_results:
        cr = json.load(open(c_results, encoding="utf8"))["per_seed"]
        for ref in ("C1_bi_only", "C2_bi_geo_lgb"):
            tests[ref] = {}
            for k in names:
                tests[ref][k] = {m: paired_seed_test(
                    [per_seed[s][k].get(m) for s in seeds],
                    [cr[str(s)]["stats"][ref].get(m) for s in seeds])
                    for m in ("hit_ratio_pct", "precision_of_hits_pct",
                              "token_saving_correct_only_pct")}

    out = {"manifest": hardware_manifest({"n_stream": n, "stream_params": params,
                                          "seeds": list(seeds), "latency_valid": False,
                                          "threshold": threshold, "deltas": list(DELTAS),
                                          "judge_tokens": judge_tokens,
                                          "c_results": c_results}),
           "per_seed": per_seed, "across_seeds": across, "paired_vs_C": tests}
    save_result(out, "exp19_baselines_seeds", tag=tag)
    return out


if __name__ == "__main__":
    ap = stream_args(add_run_args(argparse.ArgumentParser()))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--threshold", type=float, default=C.BI_COSINE_THRESHOLD)
    ap.add_argument("--judge-tokens", type=int, default=JUDGE_TOKENS)
    ap.add_argument("--c-results", default=None,
                    help="experiment 11 result file with per-seed C1/C2 stats")
    a = ap.parse_args()
    main(a.seeds, a.n, stream_params(a), fetch=not a.no_fetch, tag=a.tag,
         threshold=a.threshold, judge_tokens=a.judge_tokens, c_results=a.c_results)
