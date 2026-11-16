"""
Experiment 15 -- sensitivity to workload shape (E8).

One factor at a time around the default stream: Zipf exponent, repeat
(duplicate) rate and hard-negative rate, each over several seeds. For every
setting we serve C1-C4 and report decision quality and savings, so the
comparison between configurations can be read across workload regimes rather
than at one operating point.

Run: python -m semcache.experiments.exp15_workload_sensitivity
"""
from __future__ import annotations

import argparse

from .. import config as C
from ..encoders import warmup
from ..stats import seed_ci
from ._common import (add_run_args, banner, ensure_pool, get_stream,
                      hardware_manifest, make_backend, save_result, serve,
                      stream_args, stream_params, train_metas)

FACTORS = {
    "zipf_s": (0.8, 1.1, 1.4),
    "duplicate_rate": (0.3, 0.5, 0.7),
    "hard_negative_rate": (0.10, 0.35, 0.60),
}
KEYS = ("C1_bi_only", "C2_bi_geo_lgb", "C3_bi_cross", "C4_full_stacking")
METRICS = ("hit_ratio_pct", "precision_of_hits_pct", "false_discovery_rate_pct",
           "token_saving_correct_only_pct", "correct_miss_ratio_on_hard_negatives_pct",
           "hit_ratio_on_servable_repeats_pct", "break_even_false_hit_cost",
           "avg_response_time_s")


def settings():
    seen = set()
    for factor, values in FACTORS.items():
        for v in values:
            p = {factor: v}
            key = tuple(sorted(p.items()))
            if v == {"zipf_s": C.STREAM_ZIPF_S, "duplicate_rate": C.STREAM_DUPLICATE_RATE,
                     "hard_negative_rate": 0.35}[factor]:
                p, key = {}, ()
            if key not in seen:
                seen.add(key)
                yield factor, v, p


def main(seeds=(42, 43, 44), n: int = 2000, base: dict | None = None,
         fetch: bool = True, keys=KEYS, tag: str = "") -> dict:
    banner("EXP 15 | Workload sensitivity")
    base = base or {}
    warmup(include_cross=True)
    results = []
    for factor, value, p in settings():
        per_seed = {}
        for seed in seeds:
            wl = get_stream(seed, n, **base, **p)
            cov = ensure_pool(wl.queries(), live=fetch)
            metas = train_metas(seed=seed)
            per_seed[seed] = {"workload": wl.meta, "pool": cov, "stats": {}}
            for key in keys:
                r = serve(key, wl, make_backend(seed=seed), metas, keep_events=False)
                per_seed[seed]["stats"][key] = {m: r.stats.get(m) for m in METRICS}
        summary = {key: {m: seed_ci([per_seed[s]["stats"][key][m] for s in seeds])
                         for m in METRICS} for key in keys}
        results.append({"factor": factor, "value": value, "params": {**base, **p},
                        "per_seed": per_seed, "summary": summary})
        line = "  ".join(f"{k.split('_')[0]} {summary[k]['precision_of_hits_pct']['mean']:.1f}%"
                         for k in keys)
        print(f"  {factor}={value}: precision {line}")

    out = {"manifest": hardware_manifest({"n_stream": n, "seeds": list(seeds),
                                          "base_params": base, "virtual_time": True}),
           "factors": {k: list(v) for k, v in FACTORS.items()}, "results": results}
    save_result(out, "exp15_workload_sensitivity", tag=tag)
    return out


if __name__ == "__main__":
    ap = stream_args(add_run_args(argparse.ArgumentParser()))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    a = ap.parse_args()
    main(a.seeds, a.n, stream_params(a), fetch=not a.no_fetch, tag=a.tag)
