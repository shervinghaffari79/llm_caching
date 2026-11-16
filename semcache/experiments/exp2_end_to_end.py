"""
Experiment 2 -- end-to-end response time and token savings, per configuration.

This is the experiment that makes the manuscript's central claim checkable.
It runs the *same* query stream through four explicitly named configurations
and one no-cache baseline, in one process, at batch size 1, and reports for
each of them: overhead per request, hit ratio, average response time, speed-up
and token savings -- plus the residual of

    avg_response_time = avg_overhead + miss_ratio * avg_generation_latency

so a reader can verify that the response-time number and the latency number
describe the same system.

Phases
------
A. prefetch  -- answer every distinct query once with the real LLM, recording
                latency and token usage (see `semcache.llm` for why).
B. calibrate -- a strictly sequential, unbatched sample of API calls; this is
                the no-cache baseline latency.
C. serve     -- replay the stream through each configuration.
D. verify    -- optionally re-run one configuration against the live API and
                confirm the replayed average matches within noise.

Run:
    python -m semcache.experiments.exp2_end_to_end --n 2000
    python -m semcache.experiments.exp2_end_to_end --n 500 --llm-mode sim
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .. import config as C
from ..data import build_splits
from ..encoders import warmup
from ..llm import AnswerPool, LLMBackend
from ..metaclf import fit_final
from ..pipeline import SemanticCache, no_cache_baseline
from ..reporting import (check_run_consistency, fig_response_time,
                         fig_token_savings, save_json, save_latex,
                         table_end_to_end)
from ..workload import build_workloads
from ._common import (META_GROUPS, banner, encode_frame, hardware_manifest,
                      save_result, train_metas)

def main(n_stream: int = 2000, llm_mode: str = "replay", capacity: int | None = None,
         verify_live: int = 0, prefetch: bool = True, seed: int = C.SEED,
         stream_kw: dict | None = None, virtual_time: bool = False,
         tag: str = "", legacy_output: bool = False) -> dict:
    banner("EXP 2 | End-to-end response time and token savings by configuration")
    hw = C.HardwareProfile.capture()
    print(f"  device={hw.device}  llm={C.LLM_MODEL} "
          f"(reasoning_effort={C.LLM_REASONING_EFFORT})  mode={llm_mode}")

    splits = build_splits()
    wl = build_workloads(splits.stream_pool, lengths=(n_stream,), seed=seed,
                         **(stream_kw or {}))[n_stream]
    print(f"  stream: {wl.meta['n_events']} events, "
          f"{wl.meta['n_unique_queries']} unique, "
          f"{wl.meta['repeat_rate_pct']:.1f}% repeats "
          f"({wl.meta['paraphrase_repeat_rate_pct']:.1f}% paraphrase), "
          f"{wl.meta['hard_negative_rate_pct']:.1f}% near-duplicate distractors")

    pool = AnswerPool()
    backend = LLMBackend(mode=llm_mode, pool=pool, seed=seed,
                         virtual_time=virtual_time)
    has_key = bool(os.environ.get("OPENAI_API_KEY"))

    # ---- Phase A -------------------------------------------------------
    calib = {}
    if llm_mode in ("replay", "live") and prefetch and has_key:
        uq = wl.unique_queries()
        print(f"\n  [A] prefetching {len(uq)} unique queries "
              f"(pool already holds {len(pool)})")
        print("     ", backend.prefetch(uq))
        # ---- Phase B ---------------------------------------------------
        print("  [B] sequential latency calibration (no-cache baseline)")
        calib = LLMBackend(mode="live", pool=pool).latency_calibration(uq, n=25)
        print("     ", calib)

    # ---- Phase C -------------------------------------------------------
    warmup(include_cross=True)
    metas = train_metas(seed=seed)

    rows, raw = [], {}
    for key in (*C.CONFIG_ORDER, "C5_gated"):
        cfg = C.CONFIGS[key]
        print(f"\n  [C] serving {n_stream} requests | {cfg.label}")
        cache = SemanticCache(cfg, meta=metas.get(key), capacity=capacity,
                              policy_name="SemanticDecay" if capacity else "LRU")
        res = cache.run_stream(wl, backend)
        s = res.stats
        s["consistency"] = check_run_consistency(s)
        s["config"] = key
        s["label"] = cfg.label
        s["short"] = cfg.paper_name or key
        s["components_active"] = res.manifest["components"]
        rows.append(s)
        raw[key] = {"stats": s, "manifest": res.manifest}
        print(f"      hit ratio      {s['hit_ratio_pct']:.1f}%  "
              f"(precision of hits {s['precision_of_hits_pct']:.1f}%)")
        print(f"      overhead/req   {s['avg_overhead_ms']:.2f} ms")
        print(f"      avg response   {s['avg_response_time_s']:.3f} s")
        print(f"      token saving   {s['token_saving_pct']:.1f}% "
              f"(correct hits only {s.get('token_saving_correct_only_pct', float('nan')):.1f}%)")
        print(f"      identity residual {s['consistency']['residual_ms']:+.1f} ms "
              f"-> {'CONSISTENT' if s['consistency']['consistent'] else 'INCONSISTENT'}")

    print(f"\n  [C] no-cache baseline over the same {n_stream} requests")
    nc = no_cache_baseline(wl, backend)
    print(f"      avg response   {nc['avg_response_time_s']:.3f} s")

    for r in rows:
        r["speedup_vs_no_cache"] = (nc["avg_response_time_s"] /
                                    max(1e-9, r["avg_response_time_s"]))

    # ---- Phase D -------------------------------------------------------
    live_check = {}
    if verify_live > 0:
        print(f"\n  [D] live-API verification on {verify_live} requests "
              f"(config C2_bi_geo_lgb)")
        sub = type(wl)(events=wl.events[:verify_live], meta=dict(wl.meta))
        live_backend = LLMBackend(mode="live", pool=pool)
        cache = SemanticCache(C.CONFIGS["C2_bi_geo_lgb"],
                              meta=metas["C2_bi_geo_lgb"], capacity=capacity)
        live = cache.run_stream(sub, live_backend)
        replayed = [r for r in rows if r["config"] == "C2_bi_geo_lgb"][0]
        live_check = {
            "n": verify_live,
            "live_avg_response_s": live.stats["avg_response_time_s"],
            "replayed_avg_response_s": replayed["avg_response_time_s"],
            "abs_diff_s": abs(live.stats["avg_response_time_s"] -
                              replayed["avg_response_time_s"]),
            "live_avg_overhead_ms": live.stats["avg_overhead_ms"],
            "replayed_avg_overhead_ms": replayed["avg_overhead_ms"],
        }
        print("     ", live_check)

    out = {
        "manifest": hardware_manifest({
            "batch_regime": "online", "batch_size": C.ONLINE_BATCH,
            "llm_model": C.LLM_MODEL, "llm_mode": llm_mode,
            "reasoning_effort": C.LLM_REASONING_EFFORT,
            "capacity": capacity, "n_stream": n_stream,
            "virtual_time": virtual_time, "stream_params": stream_kw or {},
            "llm_pool_misses": backend.pool_misses}, seed=seed),
        "workload": wl.meta,
        "llm_pool": {"size": len(pool), **pool.latency_stats(),
                     **pool.token_stats()},
        "sequential_calibration": calib,
        "no_cache": nc,
        "configs": rows,
        "live_verification": live_check,
        "component_activation_table": {
            k: {"bi_encoder": C.CONFIGS[k].use_bi,
                "cross_encoder": C.CONFIGS[k].use_cross,
                "geometric_features": C.CONFIGS[k].use_geometric,
                "text_features": C.CONFIGS[k].use_text,
                "lightgbm_meta_classifier": C.CONFIGS[k].use_meta_clf}
            for k in C.CONFIG_ORDER},
        "raw": raw,
    }
    save_result(out, "exp2_end_to_end", seed=seed, tag=tag)
    if legacy_output:
        save_json(out, "exp2_end_to_end")
        save_latex(table_end_to_end(rows, nc), "table_end_to_end")
        fig_response_time(rows, nc)
        fig_token_savings(rows)

    banner("EXP 2 | summary")
    print(f"  {'configuration':32s} {'overhead':>10s} {'avg resp':>10s} "
          f"{'speedup':>8s} {'tok save':>9s}")
    for r in rows:
        print(f"  {r['short']:32s} {r['avg_overhead_ms']:8.1f}ms "
              f"{r['avg_response_time_s']:9.3f}s "
              f"{r['speedup_vs_no_cache']:7.2f}x "
              f"{r['token_saving_pct']:8.1f}%")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--llm-mode", choices=["replay", "live", "sim"], default="replay")
    ap.add_argument("--capacity", type=int, default=None)
    ap.add_argument("--verify-live", type=int, default=0)
    ap.add_argument("--no-prefetch", action="store_true")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--recycled-hn", action="store_true")
    ap.add_argument("--virtual-time", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--legacy-output", action="store_true",
                    help="also overwrite results/raw/exp2_end_to_end.json and its table")
    a = ap.parse_args()
    main(a.n, a.llm_mode, a.capacity, a.verify_live, prefetch=not a.no_prefetch,
         seed=a.seed, stream_kw={"fresh_hard_negatives": False} if a.recycled_hn else {},
         virtual_time=a.virtual_time, tag=a.tag, legacy_output=a.legacy_output)
