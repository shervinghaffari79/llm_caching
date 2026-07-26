"""
Experiment 7 -- eviction policies, including the weight sensitivity study the
conclusion promises.

Two additions over the manuscript's version:

  * the comparison runs on the dynamic stream, so capacity pressure is real
    and evictions actually change what later queries can hit;
  * the Semantic Decay weights (w1, w2, w3) are swept rather than fixed at the
    single grid-search optimum, because a policy whose reported advantage
    depends on weights tuned on the same workload it is evaluated on is a
    preliminary result, not a validated one.

Run: python -m semcache.experiments.exp7_eviction
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np

from .. import config as C
from ..cache import SemanticDecayPolicy
from ..data import build_splits
from ..llm import AnswerPool, LLMBackend
from ..pipeline import SemanticCache
from ..reporting import fig_eviction, save_json
from ..workload import build_workloads
from ._common import banner, hardware_manifest
from .exp2_end_to_end import train_metas


def _run(cfg, meta, wl, backend, capacity, policy, weights=None) -> dict:
    cache = SemanticCache(cfg, meta=meta, capacity=capacity, policy_name=policy)
    if weights is not None and isinstance(cache.policy, SemanticDecayPolicy):
        cache.policy.w = weights
    return cache.run_stream(wl, backend, keep_events=False).stats


def main(n_stream: int = 2000, config_key: str = "C2_bi_geo_lgb",
         llm_mode: str = "sim", sweep: bool = True) -> dict:
    banner("EXP 7 | Eviction policies on a dynamic stream + weight sensitivity")
    splits = build_splits()
    wl = build_workloads(splits.stream_pool, lengths=(n_stream,))[n_stream]
    backend = LLMBackend(mode=llm_mode, pool=AnswerPool())
    metas = train_metas()
    cfg = C.CONFIGS[config_key]
    meta = metas.get(config_key)
    print(f"  configuration: {cfg.label}")
    print(f"  stream: {n_stream} events, {wl.meta['n_topics']} topics")

    curves: dict[str, dict[int, float]] = {}
    detail: dict[str, dict[str, dict]] = {}
    for policy in ("SemanticDecay", "LRU", "LFU"):
        curves[policy], detail[policy] = {}, {}
        for cap in C.CACHE_SIZES_EVICTION:
            s = _run(cfg, meta, wl, backend, cap, policy)
            curves[policy][cap] = s["hit_ratio_pct"]
            detail[policy][str(cap)] = s
            print(f"    {policy:14s} cap={cap:4d} hit={s['hit_ratio_pct']:5.1f}% "
                  f"precision={s['precision_of_hits_pct']:5.1f}% "
                  f"evictions={s['evictions']:4d} "
                  f"overhead={s['avg_overhead_ms']:.2f} ms")

    rel = {}
    for cap in C.CACHE_SIZES_EVICTION:
        sd = curves["SemanticDecay"][cap]
        rel[str(cap)] = {
            "vs_LRU_relative_pct": 100.0 * (sd - curves["LRU"][cap]) / max(1e-9, curves["LRU"][cap]),
            "vs_LFU_relative_pct": 100.0 * (sd - curves["LFU"][cap]) / max(1e-9, curves["LFU"][cap]),
            "vs_LRU_absolute_pp": sd - curves["LRU"][cap],
            "vs_LFU_absolute_pp": sd - curves["LFU"][cap],
        }
    print("\n  Semantic Decay relative improvement:")
    for cap, v in rel.items():
        print(f"    cap={cap:>4s}  vs LRU {v['vs_LRU_relative_pct']:+.1f}% "
              f"({v['vs_LRU_absolute_pp']:+.1f} pp)  "
              f"vs LFU {v['vs_LFU_relative_pct']:+.1f}% "
              f"({v['vs_LFU_absolute_pp']:+.1f} pp)")

    # -- weight sensitivity ------------------------------------------------
    sens = {}
    if sweep:
        print("\n  Weight sensitivity (cap=500), simplex grid over (w1,w2,w3):")
        grid = [w for w in itertools.product(np.arange(0.0, 1.01, 0.2), repeat=3)
                if abs(sum(w) - 1.0) < 1e-6]
        rows = []
        for w in grid:
            s = _run(cfg, meta, wl, backend, 500, "SemanticDecay", weights=tuple(w))
            rows.append({"w_freshness": float(w[0]), "w_age": float(w[1]),
                         "w_boost": float(w[2]),
                         "hit_ratio_pct": s["hit_ratio_pct"],
                         "precision_of_hits_pct": s["precision_of_hits_pct"]})
        rows.sort(key=lambda r: -r["hit_ratio_pct"])
        hits = np.array([r["hit_ratio_pct"] for r in rows])
        paper = _run(cfg, meta, wl, backend, 500, "SemanticDecay",
                     weights=(C.SD_W_FRESHNESS, C.SD_W_AGE, C.SD_W_BOOST))
        sens = {
            "grid_points": len(rows),
            "best": rows[0], "worst": rows[-1],
            "spread_pp": float(hits.max() - hits.min()),
            "std_pp": float(hits.std(ddof=1)),
            "paper_weights": {"w1": C.SD_W_FRESHNESS, "w2": C.SD_W_AGE,
                              "w3": C.SD_W_BOOST,
                              "hit_ratio_pct": paper["hit_ratio_pct"]},
            "lru_reference_pct": curves["LRU"][500],
            "fraction_of_grid_beating_lru": float(
                np.mean(hits > curves["LRU"][500])),
            "all": rows,
        }
        print(f"    best  {rows[0]} ")
        print(f"    worst {rows[-1]}")
        print(f"    spread across the simplex: {sens['spread_pp']:.1f} pp "
              f"(std {sens['std_pp']:.2f})")
        print(f"    fraction of weight settings that still beat LRU: "
              f"{sens['fraction_of_grid_beating_lru']*100:.0f}%")

    fig_eviction(curves)
    out = {
        "manifest": hardware_manifest({"config": config_key, "n_stream": n_stream,
                                       "llm_mode": llm_mode}),
        "curves": curves,
        "detail": detail,
        "relative_improvement": rel,
        "weight_sensitivity": sens,
        "caveat": ("The Semantic Decay weights were tuned on this workload "
                   "family. The sensitivity sweep reports how much of the "
                   "advantage survives away from the tuned point; a policy "
                   "that only wins at its tuned weights should be presented "
                   "as preliminary."),
    }
    save_json(out, "exp7_eviction")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--config", default="C2_bi_geo_lgb", choices=list(C.CONFIGS))
    ap.add_argument("--llm-mode", choices=["replay", "sim", "live"], default="sim")
    ap.add_argument("--no-sweep", action="store_true")
    a = ap.parse_args()
    main(a.n, a.config, a.llm_mode, sweep=not a.no_sweep)
