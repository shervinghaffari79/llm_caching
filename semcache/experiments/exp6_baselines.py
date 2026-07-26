"""
Experiment 6 -- comparison against published semantic-caching systems.

The manuscript compares only against individual encoders and against
non-semantic eviction policies, which leaves open whether the stacking
ensemble beats an actual semantic cache. This experiment puts GPTCache,
SCALM, VectorQ and ContextCache on the same stream, same bi-encoder, same
FAISS index, same hardware, batch size 1, and reports the same metric set as
our own configurations.

On ContextCache specifically (asked directly in review): yes, it can be used
as a baseline, with one caveat that must be stated. ContextCache's advantage
is defined over *multi-turn* traffic where the meaning of a query depends on
the preceding turns. QQP pairs carry no dialogue context, so the workload
generator synthesises sessions and the baseline is given the true session
boundaries. That makes this a fair test of whether context-aware re-ranking
helps on QQP-derived traffic, but it is not a test of ContextCache on the
conversational data it was designed for, and the paper should say so rather
than claim a general win.

Run: python -m semcache.experiments.exp6_baselines
"""
from __future__ import annotations

import argparse

from .. import config as C
from ..baselines import BASELINE_CLASSES, BASELINES
from ..data import build_splits
from ..llm import AnswerPool, LLMBackend
from ..pipeline import SemanticCache
from ..reporting import (fig_baselines, save_json, save_latex, table_baselines)
from ..workload import build_workloads
from ._common import banner, hardware_manifest
from .exp2_end_to_end import train_metas


def main(n_stream: int = 2000, llm_mode: str = "replay",
         capacity: int | None = 2000, threshold: float = 0.70) -> dict:
    banner("EXP 6 | Comparison with GPTCache / SCALM / VectorQ / ContextCache")
    splits = build_splits()
    wl = build_workloads(splits.stream_pool, lengths=(n_stream,))[n_stream]

    pool = AnswerPool()
    backend = LLMBackend(mode=llm_mode, pool=pool)
    if llm_mode == "replay":
        missing = [q for q in wl.unique_queries() if q not in pool]
        if missing:
            print(f"  prefetching {len(missing)} queries not yet in the pool")
            print("   ", backend.prefetch(missing))

    rows = []

    # -- published systems -------------------------------------------------
    for key, cls in BASELINE_CLASSES.items():
        spec = BASELINES[key]
        print(f"\n  {spec.label}\n    ({spec.citation}) -- {spec.notes}")
        sys_ = cls(capacity=capacity, threshold=threshold)
        s = sys_.run_stream(wl, backend)
        s["short"] = key
        s["citation"] = spec.citation
        s["implementation_note"] = spec.notes
        rows.append(s)
        print(f"    hit {s['hit_ratio_pct']:5.1f}% | precision of hits "
              f"{s['precision_of_hits_pct']:5.1f}% | correct-miss on distractors "
              f"{s.get('correct_miss_ratio_on_hard_negatives_pct', float('nan')):5.1f}% "
              f"| tokens saved {s['token_saving_pct']:5.1f}% "
              f"| overhead {s['avg_overhead_ms']:.2f} ms")

    # -- our configurations ------------------------------------------------
    metas = train_metas()
    for key in ("C1_bi_only", "C2_bi_geo_lgb", "C4_full_stacking"):
        cfg = C.CONFIGS[key]
        print(f"\n  Ours: {cfg.label}")
        cache = SemanticCache(cfg, meta=metas.get(key), capacity=capacity,
                              policy_name="SemanticDecay")
        res = cache.run_stream(wl, backend)
        s = res.stats
        s["label"] = f"Ours -- {cfg.label}"
        s["short"] = key
        s["system"] = key
        rows.append(s)
        print(f"    hit {s['hit_ratio_pct']:5.1f}% | precision of hits "
              f"{s['precision_of_hits_pct']:5.1f}% | correct-miss on distractors "
              f"{s.get('correct_miss_ratio_on_hard_negatives_pct', float('nan')):5.1f}% "
              f"| tokens saved {s['token_saving_pct']:5.1f}% "
              f"| overhead {s['avg_overhead_ms']:.2f} ms")

    out = {
        "manifest": hardware_manifest({
            "n_stream": n_stream, "capacity": capacity,
            "shared_bi_encoder": C.BI_ENCODER,
            "shared_threshold_for_baselines": threshold,
            "batch_regime": "online", "llm_mode": llm_mode}),
        "workload": wl.meta,
        "systems": rows,
        "fairness_statement": (
            "All systems share one bi-encoder, one FAISS index type, one "
            "cache substrate and one query stream, and differ only in the "
            "acceptance rule and retention logic. They are re-implementations "
            "of the published logic, not the authors' code; differences in "
            "vector store or runtime would otherwise be attributed to the "
            "algorithm. VectorQ is given ground-truth verification feedback, "
            "so its numbers are an upper bound on its real behaviour."),
        "contextcache_caveat": (
            "QQP has no dialogue context. Sessions are synthesised by the "
            "workload generator and ContextCache is given the true session "
            "boundaries. The comparison is valid on this traffic but does not "
            "evaluate ContextCache on the multi-turn data it targets."),
    }
    save_json(out, "exp6_baselines")
    save_latex(table_baselines(rows), "table_baseline_comparison")
    fig_baselines(rows)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--llm-mode", choices=["replay", "sim", "live"], default="replay")
    ap.add_argument("--capacity", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=0.70)
    a = ap.parse_args()
    main(a.n, a.llm_mode, a.capacity, a.threshold)
