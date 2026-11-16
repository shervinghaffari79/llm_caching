"""
Experiment 11 -- configurations across independent seeds (E1).

Each seed is an independent QQP-derived stream (generator seed) served by
meta-classifiers fitted with the same seed. For every configuration we record
the serving metrics; for C1 vs C2 we add per-request paired tests within each
seed and a seed-level paired test across seeds, so the choice of a default
rests on stated metrics and their uncertainty.

Run: python -m semcache.experiments.exp11_seeds --seeds 42 43 44 45 46

--real-time sleeps for each miss's recorded generation latency instead of
adding it to the clock. Seeds can then run as separate processes
(--seeds 42 --tag s42) and be combined with --merge FILE [FILE ...].
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .. import config as C
import json

from ..encoders import warmup
from ..stats import mcnemar, paired_bootstrap, paired_seed_test, seed_ci
from ._common import (add_run_args, banner, ensure_pool, get_stream,
                      hardware_manifest, make_backend, save_result, serve,
                      stream_args, stream_params, train_metas)

KEYS = ("C1_bi_only", "C2_bi_geo_lgb", "C3_bi_cross", "C4_full_stacking", "C5_gated")
METRICS = ("hit_ratio_pct", "precision_of_hits_pct", "false_discovery_rate_pct",
           "false_hit_rate_pct", "token_saving_pct", "token_saving_correct_only_pct",
           "correct_miss_ratio_on_hard_negatives_pct", "avg_response_time_s",
           "p95_response_time_s", "avg_overhead_ms", "break_even_false_hit_cost")


def _vectors(per_event):
    served_ok = np.array([bool(e["hit"] and e["correct"]) for e in per_event])
    false_hit = np.array([bool(e["hit"] and not e["correct"]) for e in per_event])
    rt = np.array([e["total_s"] for e in per_event])
    return served_ok, false_hit, rt


def main(seeds=(42, 43, 44, 45, 46), n: int = 2000, params: dict | None = None,
         fetch: bool = True, keys=KEYS, tag: str = "",
         real_time: bool = False) -> dict:
    banner("EXP 11 | Configurations across seeds")
    params = params or {}
    warmup(include_cross=True)
    per_seed: dict[int, dict] = {}
    for seed in seeds:
        wl = get_stream(seed, n, **params)
        cov = ensure_pool(wl.queries(), live=fetch)
        print(f"  seed {seed}: {wl.meta['n_events']} events, "
              f"{wl.meta['hard_negative_rate_pct']:.1f}% distractors, pool {cov}")
        metas = train_metas(seed=seed)
        rows, vecs = {}, {}
        for key in keys:
            res = serve(key, wl, make_backend(virtual_time=not real_time, seed=seed),
                        metas)
            rows[key] = res.stats
            vecs[key] = _vectors(res.per_event)
            s = res.stats
            print(f"    {key:18s} hit {s['hit_ratio_pct']:5.1f}%  "
                  f"prec {s['precision_of_hits_pct']:5.1f}%  "
                  f"correct-only {s['token_saving_correct_only_pct']:5.1f}%  "
                  f"rt {s['avg_response_time_s']:.3f}s")
        a_ok, a_fh, a_rt = vecs["C1_bi_only"]
        b_ok, b_fh, b_rt = vecs["C2_bi_geo_lgb"]
        per_seed[seed] = {
            "workload": wl.meta, "pool": cov, "stats": rows,
            "c1_vs_c2_within_seed": {
                "served_correct": {**paired_bootstrap(a_ok, b_ok), **mcnemar(a_ok, b_ok)},
                "false_hits": {**paired_bootstrap(a_fh, b_fh), **mcnemar(a_fh, b_fh)},
                "response_time_s": paired_bootstrap(a_rt, b_rt),
            },
        }

    return summarise(per_seed, list(seeds), keys, n, params, real_time, tag)


def summarise(per_seed: dict, seeds: list, keys, n, params, real_time, tag,
              extra: dict | None = None) -> dict:
    across = {}
    for key in keys:
        across[key] = {m: seed_ci([per_seed[s]["stats"][key].get(m) for s in seeds])
                       for m in METRICS}
    c1_c2 = {m: paired_seed_test([per_seed[s]["stats"]["C1_bi_only"].get(m) for s in seeds],
                                 [per_seed[s]["stats"]["C2_bi_geo_lgb"].get(m) for s in seeds])
             for m in METRICS}

    out = {"manifest": hardware_manifest({"n_stream": n, "stream_params": params,
                                          "seeds": list(seeds),
                                          "virtual_time": not real_time,
                                          **(extra or {})}),
           "per_seed": per_seed, "across_seeds": across, "c1_vs_c2_across_seeds": c1_c2}
    save_result(out, "exp11_seeds", tag=tag)
    return out


if __name__ == "__main__":
    ap = stream_args(add_run_args(argparse.ArgumentParser()))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--real-time", action="store_true",
                    help="sleep for generation latency instead of virtual time")
    ap.add_argument("--merge", nargs="*", default=None,
                    help="combine per-seed result files into one summary; also "
                         "collects every <tag>_s<seed> file once no seed is running")
    a = ap.parse_args()
    if a.merge is not None:
        import time
        from ._common import REVISION_DIR, out_name
        stem = out_name("exp11_seeds", tag=a.tag)
        while list(REVISION_DIR.glob(f"{stem}_s*.running")):
            time.sleep(60)
        files = sorted(set(a.merge) | {str(p) for p in REVISION_DIR.glob(f"{stem}_s*.json")})
        if not files:
            raise SystemExit("nothing to merge")
        a.merge = files
        runs = [json.load(open(f, encoding="utf8")) for f in a.merge]
        per_seed = {int(s): v for r in runs for s, v in r["per_seed"].items()}
        seeds = sorted(per_seed)
        rt = {not r["manifest"].get("virtual_time", True) for r in runs}
        assert len(rt) == 1, "cannot merge virtual-time and real-time runs"
        summarise(per_seed, seeds, KEYS, a.n, stream_params(a), rt.pop(), a.tag,
                  extra={"merged_from": a.merge,
                         "per_seed_hosts": [r["manifest"]["hardware"].get("cpu")
                                            for r in runs]})
    else:
        # A run that is already in progress or finished under the same name is
        # skipped, so seeds can be started by hand and a queued batch that
        # reaches them later does not repeat them.
        from ._common import REVISION_DIR, out_name
        target = REVISION_DIR / (out_name("exp11_seeds", tag=a.tag) + ".json")
        lock = target.with_suffix(".running")
        if target.exists() or lock.exists():
            print(f"  skipping: {target.name} already exists or is running")
            raise SystemExit
        REVISION_DIR.mkdir(parents=True, exist_ok=True)
        lock.write_text(str(os.getpid()))
        try:
            main(a.seeds, a.n, stream_params(a), fetch=not a.no_fetch, tag=a.tag,
                 real_time=a.real_time)
        finally:
            lock.unlink(missing_ok=True)
