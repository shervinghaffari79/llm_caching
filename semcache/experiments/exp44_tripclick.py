"""
Experiment 44 -- the QQP-trained decision layer on a second, strictly
chronological log: TripClick (health-search click log, 2013-2020).

Protocol, fixed before any run:
  * Request = one search (session, normalised query); time = first click;
    class = first-clicked document (scripts/build_tripclick_requests.py).
    A hit is correct when the cached request's class equals the new one's.
    The label is a proxy: equal first clicks do not prove equal intent, and
    different clicks do not prove different intent.
  * Stream = the n most recent requests ending at the last request of the log
    (default 40,000), in time order, served by an unbounded cache that starts
    empty. The decision layer is trained on QQP only, so nothing in the
    stream or its history is seen during training.
  * C1 over cosine thresholds and C2 over probability thresholds, exactly as
    exp40. Hit ratio, precision and per-request flags are recorded; blocks of
    requests show cold start.
  * Nothing is generated, replayed or simulated on a miss (NullBackend), so
    only label-based metrics are reported.

Run: python -m semcache.experiments.exp44_tripclick --data external/tripclick_requests.parquet --start -40000 --n 40000
The data are licensed for non-commercial research and are not redistributed.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import config as C
from ..encoders import warmup
from ..llm import NullBackend
from ._common import banner, hardware_manifest, memoized_encoders, save_result, serve, train_metas
from . import exp40_chronological_transfer as E40
from .exp22_searchqueries import KEEP, blocks


def main(data: str, start: int, n: int, seed: int = C.SEED, tag: str = "") -> dict:
    path = C.DATA_DIR / data
    total = len(pd.read_parquet(path, columns=["id_set"]))
    start = start + total if start < 0 else start
    banner(f"EXP 44 | TripClick, requests {start}-{start + n - 1} of {total}")
    warmup(include_cross=False)
    E40.DATA = path
    wl = E40.load_stream(start, n)
    wl.meta.update({"source": "TripClick (Trip Database click log), non-commercial research use",
                    "order": "time of first click", "total_requests": total,
                    "equivalence": "same first-clicked document (proxy label)"})
    print("  " + str({k: wl.meta[k] for k in ("created_first", "created_last", "repeat_within_window_pct",
                                              "repeat_given_full_history_pct")}), flush=True)
    from ..encoders import get_bi
    qs = sorted(set(wl.queries()))
    print(f"  encoding {len(qs)} distinct queries", flush=True)
    enc = get_bi().model.encode(qs, batch_size=512, normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    vecs = dict(zip(qs, enc))
    metas = train_metas(seed=seed)
    res = {}

    def record(name, stats, events):
        res[name] = {"stats": {k: stats.get(k) for k in KEEP}, "blocks": blocks(events),
                     "hit": [int(bool(e["hit"])) for e in events],
                     "correct": [int(bool(e["correct"])) if e["hit"] else 0 for e in events]}
        c = sum(res[name]["correct"]) / len(events) * 100
        print(f"  {name:12s} hit {stats['hit_ratio_pct']:5.2f}%  correct-hit {c:5.2f}%  "
              f"prec {stats['precision_of_hits_pct']:5.1f}%", flush=True)

    with memoized_encoders() as memo:
        memo._bi_cache.update(vecs)
        for thr in E40.BI_GRID:
            r = serve("C1_bi_only", wl, NullBackend(), metas, bi_threshold=thr)
            record(f"C1_t{thr:g}", r.stats, r.per_event)
        for thr in E40.META_GRID:
            r = serve("C2_bi_geo_lgb", wl, NullBackend(), metas, meta_threshold=thr)
            record(f"C2_p{thr:g}", r.stats, r.per_event)
    out = {"manifest": hardware_manifest({"start": start, "n": n, "total_requests": total, "latency_valid": False,
                                          "generation": "none", "bi_grid": list(E40.BI_GRID),
                                          "meta_grid": list(E40.META_GRID)}, seed=seed),
           "workload": wl.meta, "results": res}
    save_result(out, "exp44_tripclick", tag=tag or f"s{start}_n{n}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="external/tripclick_requests.parquet")
    ap.add_argument("--start", type=int, default=-40000, help="negative counts from the end of the log")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(a.data, a.start, a.n, tag=a.tag)
