"""
Experiment 12 -- operating curves, matched comparisons and false-hit pricing
(E2, E3).

Each decision threshold is swept for C1-C4 and for the four published-rule
baselines on the same stream; vCache is swept over its error bound delta,
with its own insertion rule and with insert-on-every-miss. Because the cache is stateful, every point is a
full re-run of the stream, not a re-thresholding of logged scores. From the
curves we report:

* precision, correct-only saving and false-hit rate against served-hit ratio;
* every system interpolated to common served-hit ratios and to common
  precision levels, so systems are compared at matched operating points
  rather than at their individual default thresholds;
* risk-adjusted saving, where a false hit costs lambda times the call it
  replaced, the best threshold for each lambda, and the break-even lambda.

Encoders are memoised by text during the sweep; latency is not reported here.

Run: python -m semcache.experiments.exp12_threshold_risk
"""
from __future__ import annotations

import argparse

import numpy as np

from .. import config as C
from ..baselines import BASELINE_CLASSES, DELTA_BASELINES
from ..cache import RISK_LAMBDAS
from ..encoders import get_bi, warmup
from ._common import (add_run_args, banner, ensure_pool, get_stream,
                      hardware_manifest, make_backend, memoized_encoders,
                      save_result, serve, stream_args, stream_params, train_metas)

BI_GRID = [round(x, 3) for x in np.arange(0.60, 0.951, 0.025)] + [0.96, 0.97, 0.98, 0.99]
PROB_GRID = [round(x, 2) for x in np.arange(0.10, 0.951, 0.05)] + [0.97, 0.98, 0.99, 0.995]
# vCache's operating point is its error bound delta, not a threshold.
DELTA_GRID = (0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12)
KEEP = ("hit_ratio_pct", "precision_of_hits_pct", "false_discovery_rate_pct",
        "false_hit_rate_pct", "token_saving_pct", "token_saving_correct_only_pct",
        "risk_adjusted_saving_pct", "break_even_false_hit_cost",
        "correct_miss_ratio_on_hard_negatives_pct", "hit_ratio_on_servable_repeats_pct",
        "judge_calls_per_request", "total_tokens_no_cache_equivalent")
HIT_TARGETS = (45.0, 50.0, 55.0, 60.0)
PREC_TARGETS = (85.0, 90.0, 95.0)


def _point(stats: dict, thr: float) -> dict:
    return {"threshold": thr, **{k: stats.get(k) for k in KEEP}}


def _interp(curve: list[dict], x_key: str, y_key: str, x: float):
    """Linear interpolation of y at x along a curve; None outside its range."""
    pts = sorted((p[x_key], p[y_key]) for p in curve
                 if p.get(x_key) is not None and p.get(y_key) is not None)
    xs = [a for a, _ in pts]
    if not xs or x < min(xs) or x > max(xs):
        return None
    return float(np.interp(x, xs, [b for _, b in pts]))


def _matched(curves: dict) -> dict:
    out = {"at_hit_ratio": {}, "at_precision": {}}
    for h in HIT_TARGETS:
        out["at_hit_ratio"][f"{h:g}"] = {
            sys: {"precision_of_hits_pct": _interp(cv, "hit_ratio_pct", "precision_of_hits_pct", h),
                  "token_saving_correct_only_pct": _interp(cv, "hit_ratio_pct", "token_saving_correct_only_pct", h)}
            for sys, cv in curves.items()}
    for pr in PREC_TARGETS:
        out["at_precision"][f"{pr:g}"] = {
            sys: {"hit_ratio_pct": _interp(cv, "precision_of_hits_pct", "hit_ratio_pct", pr),
                  "token_saving_correct_only_pct": _interp(cv, "precision_of_hits_pct", "token_saving_correct_only_pct", pr)}
            for sys, cv in curves.items()}
    return out


def _best_per_lambda(curves: dict) -> dict:
    out = {}
    for lam in RISK_LAMBDAS:
        k = f"{lam:g}"
        out[k] = {}
        for sys, cv in curves.items():
            best = max(cv, key=lambda p: p["risk_adjusted_saving_pct"][k])
            out[k][sys] = {"threshold": best["threshold"],
                           "risk_adjusted_saving_pct": best["risk_adjusted_saving_pct"][k],
                           "hit_ratio_pct": best["hit_ratio_pct"],
                           "precision_of_hits_pct": best["precision_of_hits_pct"]}
    return out


def main(seed: int = C.SEED, n: int = 2000, params: dict | None = None,
         fetch: bool = True, tag: str = "", quick: bool = False,
         configs_only: bool = False) -> dict:
    banner("EXP 12 | Threshold sweeps, matched comparisons, false-hit pricing")
    params = params or {}
    warmup(include_cross=True)
    wl = get_stream(seed, n, **params)
    cov = ensure_pool(wl.queries(), live=fetch)
    metas = train_metas(seed=seed)
    bi_grid = BI_GRID[::3] if quick else BI_GRID
    prob_grid = PROB_GRID[::3] if quick else PROB_GRID

    sweeps = {
        "C1_bi_only": ("bi_threshold", bi_grid),
        "C2_bi_geo_lgb": ("meta_threshold", prob_grid),
        "C3_bi_cross": ("cross_threshold", prob_grid),
        "C4_full_stacking": ("meta_threshold", prob_grid),
    }
    curves: dict[str, list[dict]] = {}
    with memoized_encoders():
        for key, (arg, grid) in sweeps.items():
            curves[key] = []
            for thr in grid:
                r = serve(key, wl, make_backend(seed=seed), metas,
                          keep_events=False, **{arg: thr})
                curves[key].append(_point(r.stats, thr))
            print(f"  {key}: {len(grid)} points")
        for name, cls in ({} if configs_only else BASELINE_CLASSES).items():
            curves[name] = []
            for thr in bi_grid:
                s = cls(bi=get_bi(), threshold=thr).run_stream(
                    wl, make_backend(seed=seed))
                curves[name].append(_point(s, thr))
            print(f"  {name}: {len(bi_grid)} points")
        delta_grid = DELTA_GRID[::2] if quick else DELTA_GRID
        for name, cls in ({} if configs_only else DELTA_BASELINES).items():
            for ins in ("native", "always"):
                sys_key = name if ins == "native" else f"{name}_insert_always"
                curves[sys_key] = []
                for d in delta_grid:
                    s = cls(bi=get_bi(), delta=d, insertion=ins, seed=seed).run_stream(
                        wl, make_backend(seed=seed))
                    curves[sys_key].append(_point(s, d))
                print(f"  {sys_key}: {len(delta_grid)} points (delta)")

    out = {"manifest": hardware_manifest({"n_stream": n, "stream_params": params,
                                          "latency_valid": False,
                                          "risk_lambdas": list(RISK_LAMBDAS),
                                          "bi_grid": bi_grid, "prob_grid": prob_grid,
                                          "configs_only": configs_only}, seed=seed),
           "workload": wl.meta, "pool": cov, "curves": curves,
           "matched": _matched(curves), "best_per_lambda": _best_per_lambda(curves)}
    save_result(out, "exp12_threshold_risk", seed=seed, tag=tag)
    return out


if __name__ == "__main__":
    ap = stream_args(add_run_args(argparse.ArgumentParser()))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--configs-only", action="store_true",
                    help="sweep C1-C4 only (no published baselines or vCache)")
    a = ap.parse_args()
    main(a.seed, a.n, stream_params(a), fetch=not a.no_fetch, tag=a.tag, quick=a.quick,
         configs_only=a.configs_only)
