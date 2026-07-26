"""
Experiment 5 -- cache hit/miss behaviour as a *dynamic* property of a stream.

The manuscript's Figure 2 reports hit ratio on duplicates and miss ratio on
non-duplicates at 500/1000/1500/2000 "samples". Read as a pair-classification
score that is just recall and specificity under another name, and it cannot
move with cache size, eviction policy or arrival order -- none of which appear
in a static pair score.

This experiment replaces it with a measurement on a live cache:

  * the index starts empty, so a query's first occurrence is always a miss no
    matter how good the model is;
  * a repeat can only hit if its earlier equivalent is still resident;
  * every miss inserts, and insertion may evict, changing what later queries
    can hit.

Three quantities are therefore reported, and they are not interchangeable:

  hit_ratio_pct                    realised hits / all requests
  oracle_hit_ratio_pct             requests for which an equivalent entry was
                                   resident at arrival -- the ceiling
  hit_ratio_on_servable_repeats    realised / achievable, i.e. the similarity
                                   model's own accuracy with cold-start and
                                   eviction effects factored out

The stream lengths are nested prefixes of one stream, so the curves show cache
warm-up rather than four unrelated samples.

Run: python -m semcache.experiments.exp5_dynamic_cache
"""
from __future__ import annotations

import argparse

import numpy as np

from .. import config as C
from ..data import build_splits
from ..llm import AnswerPool, LLMBackend
from ..pipeline import SemanticCache
from ..reporting import fig_dynamic_hit_ratio, save_json
from ..workload import build_workloads
from ._common import banner, hardware_manifest
from .exp2_end_to_end import train_metas


def main(config_key: str = "C2_bi_geo_lgb", llm_mode: str = "sim",
         capacities=(None, 1500, 2000)) -> dict:
    banner("EXP 5 | Dynamic cache behaviour over a live query stream")
    splits = build_splits()
    wls = build_workloads(splits.stream_pool, lengths=C.STREAM_LENGTHS)
    longest = max(C.STREAM_LENGTHS)
    print(f"  stream family: nested prefixes of one {longest}-event stream")
    print(f"  {wls[longest].meta['n_topics']} topics, "
          f"top-10 topics account for "
          f"{wls[longest].meta['top10_topic_share_pct']:.1f}% of traffic")

    pool = AnswerPool()
    backend = LLMBackend(mode=llm_mode, pool=pool)
    metas = train_metas()
    cfg = C.CONFIGS[config_key]
    print(f"  configuration under test: {cfg.label}")

    by_capacity: dict[str, dict] = {}
    for cap in capacities:
        per_length: dict[int, dict] = {}
        for L in sorted(C.STREAM_LENGTHS):
            cache = SemanticCache(cfg, meta=metas.get(config_key), capacity=cap,
                                  policy_name="SemanticDecay" if cap else "LRU")
            res = cache.run_stream(wls[L], backend)
            per_length[L] = res.stats
            s = res.stats
            print(f"    cap={str(cap):>5s} L={L:5d} | hit {s['hit_ratio_pct']:5.1f}% "
                  f"| oracle ceiling {s['oracle_hit_ratio_pct']:5.1f}% "
                  f"| of servable repeats {s.get('hit_ratio_on_servable_repeats_pct', float('nan')):5.1f}% "
                  f"| correct-miss on distractors "
                  f"{s.get('correct_miss_ratio_on_hard_negatives_pct', float('nan')):5.1f}% "
                  f"| evictions {s['evictions']}")
        by_capacity[str(cap)] = per_length

    unbounded = by_capacity[str(capacities[0])]
    fig_dynamic_hit_ratio(unbounded)

    # Show that the metric really is dynamic: same model, different capacity
    # and arrival order give different hit ratios.
    sensitivity = {
        "capacity_effect": {
            cap: by_capacity[cap][longest]["hit_ratio_pct"] for cap in by_capacity},
        "warmup_effect": {
            str(L): unbounded[L]["hit_ratio_pct"] for L in sorted(unbounded)},
    }
    print("\n  Evidence the hit ratio is dynamic, not a static pair score:")
    print(f"    same model, capacity varied  -> {sensitivity['capacity_effect']}")
    print(f"    same model, stream length    -> {sensitivity['warmup_effect']}")

    out = {
        "manifest": hardware_manifest({"config": config_key,
                                       "llm_mode": llm_mode,
                                       "capacities": [str(c) for c in capacities]}),
        "workload": wls[longest].meta,
        "by_capacity": by_capacity,
        "sensitivity": sensitivity,
        "metric_definitions": {
            "hit_ratio_pct": "realised hits / total requests on the stream",
            "oracle_hit_ratio_pct": ("fraction of requests for which a "
                                     "semantically equivalent entry was "
                                     "resident in the cache at arrival time; "
                                     "the achievable ceiling"),
            "hit_ratio_on_servable_repeats_pct": ("realised hits divided by "
                                                  "achievable hits; isolates "
                                                  "similarity-model error"),
            "correct_miss_ratio_on_hard_negatives_pct": ("fraction of injected "
                                                         "near-duplicate "
                                                         "distractors correctly "
                                                         "sent to the LLM"),
            "precision_of_hits_pct": ("fraction of served hits whose cached "
                                      "answer was for a genuinely equivalent "
                                      "query; a false hit returns a wrong "
                                      "answer, which no pair-level metric on "
                                      "a static file can express"),
        },
    }
    save_json(out, "exp5_dynamic_cache")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="C2_bi_geo_lgb", choices=list(C.CONFIGS))
    ap.add_argument("--llm-mode", choices=["replay", "sim", "live"], default="sim")
    a = ap.parse_args()
    main(a.config, a.llm_mode)
