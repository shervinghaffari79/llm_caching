"""
Experiment 22 -- a real query log: SemBenchmarkSearchQueries.

The stream is the first N queries (default 50,000) of vCache/SemBenchmarkSearch
Queries (Apache-2.0; real search-engine queries from a filtered ORCAS subset),
in the dataset's order. Two queries are equivalent when they share an
`id_set` (built by the dataset authors with an LLM judge deciding whether two
queries yield the same response). No query string repeats, so every
repeat is a rephrasing, and repeats become more frequent further into the log.

All systems use the same encoder and exact index as elsewhere. C1's threshold
and C2's meta-classifier were tuned on QQP and are applied here unchanged
(out-of-domain); C1 and C2 are also swept over their thresholds, since the
QQP-tuned operating point need not transfer. vCache adapts per entry and needs
no tuning. Hit decisions use only the queries and labels; responses are not
part of the dataset, so the LLM backend is simulated and token counts are
nominal: report hit ratio, precision and false-hit statistics, not token
savings or latency.

Download (only the needed columns) to data/external/sembenchmark_searchqueries.parquet.

Run: python -m semcache.experiments.exp22_searchqueries --n 50000
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import config as C
from ..baselines import VCacheBaseline
from ..encoders import get_bi, warmup
from ..llm import AnswerPool, LLMBackend
from ..workload import StreamEvent, Workload
from ._common import (add_run_args, banner, hardware_manifest, memoized_encoders,
                      save_result, serve, train_metas)

DATA = C.DATA_DIR / "external" / "sembenchmark_searchqueries.parquet"
SOURCE = "https://huggingface.co/datasets/vCache/SemBenchmarkSearchQueries"
BI_GRID = (0.70, C.BI_COSINE_THRESHOLD, 0.75, 0.80, 0.85, 0.90)
META_GRID = (0.3, C.META_PROB_THRESHOLD, 0.7, 0.9)
DELTAS = (0.01, 0.02, 0.05)
BLOCK = 5000
KEEP = ("hit_ratio_pct", "precision_of_hits_pct", "false_discovery_rate_pct",
        "false_hit_rate_pct", "hit_ratio_vs_oracle_pct", "oracle_hit_ratio_pct",
        "explores", "exploits", "cold_start_explores", "judge_calls_per_request",
        "cache_entries")


def load_stream(n: int) -> Workload:
    df = pd.read_parquet(DATA).iloc[:n]
    seen: set[int] = set()
    events = []
    for i, (q, g) in enumerate(zip(df["prompt"], df["id_set"])):
        g = int(g)
        events.append(StreamEvent(step=i, query=str(q), topic_id=g,
                                  is_repeat=g in seen, is_exact=False,
                                  is_hard_negative=False, session_id=i, turn=0))
        seen.add(g)
    rep = sum(e.is_repeat for e in events)
    meta = {"source": SOURCE, "n_events": len(events),
            "n_unique_queries": int(df["prompt"].nunique()),
            "n_topics": int(df["id_set"].nunique()),
            "repeat_rate_pct": 100.0 * rep / max(1, len(events)),
            "exact_repeat_rate_pct": 0.0,
            "paraphrase_repeat_rate_pct": 100.0 * rep / max(1, len(events)),
            "hard_negative_rate_pct": None, "order": "dataset order"}
    return Workload(events=events, meta=meta)


def blocks(events: list[dict], size: int = BLOCK) -> list[dict]:
    out = []
    for i in range(0, len(events), size):
        ev = events[i:i + size]
        hits = [e for e in ev if e["hit"]]
        out.append({"start": i, "n": len(ev),
                    "hit_ratio_pct": 100.0 * len(hits) / max(1, len(ev)),
                    "precision_of_hits_pct": (100.0 * float(np.mean([bool(e["correct"]) for e in hits]))
                                              if hits else None),
                    "repeat_rate_pct": None})
    return out


def _backend(seed):
    return LLMBackend(mode="sim", pool=AnswerPool(), virtual_time=True, seed=seed)


def main(n: int = 50000, seed: int = C.SEED, tag: str = "") -> dict:
    banner(f"EXP 22 | SemBenchmarkSearchQueries, first {n} queries")
    warmup(include_cross=True)
    wl = load_stream(n)
    print(f"  {wl.meta['n_events']} queries, {wl.meta['n_topics']} equivalence groups, "
          f"{wl.meta['repeat_rate_pct']:.1f}% rephrased repeats")
    metas = train_metas(seed=seed)
    rep = np.array([e.is_repeat for e in wl.events])
    res: dict[str, dict] = {}

    def record(name, stats, events):
        bl = blocks(events)
        for b in bl:
            b["repeat_rate_pct"] = 100.0 * float(rep[b["start"]:b["start"] + b["n"]].mean())
        res[name] = {"stats": {k: stats.get(k) for k in KEEP}, "blocks": bl}
        print(f"  {name:22s} hit {stats['hit_ratio_pct']:5.1f}%  "
              f"prec {stats['precision_of_hits_pct']:5.1f}%  "
              f"(last block {bl[-1]['hit_ratio_pct']:.1f}%)", flush=True)

    with memoized_encoders() as memo:
        memo.prefill(wl.queries())
        for thr in BI_GRID:
            r = serve("C1_bi_only", wl, _backend(seed), metas, bi_threshold=thr)
            record(f"C1_t{thr:g}", r.stats, r.per_event)
        for thr in META_GRID:
            r = serve("C2_bi_geo_lgb", wl, _backend(seed), metas, meta_threshold=thr)
            record(f"C2_p{thr:g}", r.stats, r.per_event)
        r = serve("C5_gated", wl, _backend(seed), metas)
        record("C5_gated", r.stats, r.per_event)
        for d in DELTAS:
            b = VCacheBaseline(bi=get_bi(), delta=d, insertion="native", seed=seed)
            s = b.run_stream(wl, _backend(seed))
            record(f"vcache_d{d:g}", s, b.last_events)

    out = {"manifest": hardware_manifest({"dataset": SOURCE, "n_stream": n,
                                          "latency_valid": False,
                                          "token_counts_valid": False,
                                          "bi_grid": list(BI_GRID),
                                          "meta_grid": list(META_GRID),
                                          "deltas": list(DELTAS), "block": BLOCK},
                                         seed=seed),
           "workload": wl.meta, "results": res}
    save_result(out, "exp22_searchqueries", tag=tag or f"n{n}")
    return out


if __name__ == "__main__":
    ap = add_run_args(argparse.ArgumentParser())
    ap.add_argument("--n", type=int, default=50000)
    a = ap.parse_args()
    main(a.n, a.seed, a.tag)
