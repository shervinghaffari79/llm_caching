"""
Experiment 40 -- the QQP-trained decision layer on a chronological Stack
Exchange stream (gaming.stackexchange.com, one question title per request, in
creation order, 2009-2024; scripts/build_stackexchange_log.py).

Equivalence: two questions are equivalent when a human closed one as a
duplicate of the other, so a correct hit means the cached question's answers
resolve the request. This is stricter than title paraphrase, and a near
paraphrase that nobody linked counts as a false hit. Cosine similarity of the
labelled duplicate pairs is low (median 0.48 in the oldest 20,000 questions),
so this experiment measures how much of the repetition an off-the-shelf
bi-encoder cache can serve at all, before any admission policy is considered.

Protocol, fixed before the run: stream = rows [start, start + n) in creation
order (default the most recent 40,000 questions, rows 60,000-99,999), served by
an unbounded cache that starts empty; QQP-trained models only; C1 over cosine
thresholds and C2 over probability thresholds; hit ratio, precision and the
per-request flags are recorded. The share of requests that repeat an earlier
class (the ceiling for any policy) is reported in the workload metadata.

Run: python -m semcache.experiments.exp40_chronological_transfer --start 60000 --n 40000
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import config as C
from ..encoders import warmup
from ..workload import StreamEvent, Workload
from ._common import banner, hardware_manifest, memoized_encoders, save_result, serve, train_metas
from .exp22_searchqueries import KEEP, _backend, blocks

DATA = C.DATA_DIR / "external" / "stackexchange_gaming.parquet"
VECS = C.DATA_DIR / "external" / "gaming_vectors.npz"
BI_GRID = (0.60, 0.70, C.BI_COSINE_THRESHOLD, 0.80, 0.85, 0.895, 0.93, 0.95)
META_GRID = (0.1, 0.3, C.META_PROB_THRESHOLD, 0.7, 0.9, 0.97)


def load_stream(start: int, n: int) -> Workload:
    full = pd.read_parquet(DATA)
    hist = set(int(g) for g in full.id_set.iloc[:start])      # classes seen before the window
    df = full.iloc[start:start + n]
    seen_g, seen_q, events, rep_hist = set(), set(), [], 0
    for i, (q, g) in enumerate(zip(df["prompt"].astype(str), df["id_set"])):
        g = int(g)
        rep_hist += g in hist or g in seen_g
        events.append(StreamEvent(step=i, query=q, topic_id=g, is_repeat=g in seen_g, is_exact=q in seen_q,
                                  is_hard_negative=False, session_id=i, turn=0))
        seen_g.add(g)
        seen_q.add(q)
    rep = sum(e.is_repeat for e in events)
    meta = {"source": "gaming.stackexchange.com data dump (CC BY-SA)", "n_events": len(events),
            "start": start, "created_first": str(df.created.iloc[0]) if "created" in df else None,
            "created_last": str(df.created.iloc[-1]) if "created" in df else None,
            "n_topics": len(seen_g), "repeat_within_window_pct": 100.0 * rep / len(events),
            "repeat_given_full_history_pct": 100.0 * rep_hist / len(events), "order": "creation time"}
    return Workload(events=events, meta=meta)


def main(start: int = 60000, n: int = 40000, seed: int = C.SEED, tag: str = "") -> dict:
    banner(f"EXP 40 | gaming.stackexchange.com, rows {start}-{start + n - 1}")
    warmup(include_cross=False)
    wl = load_stream(start, n)
    print("  " + str({k: wl.meta[k] for k in ("created_first", "created_last", "repeat_within_window_pct")}), flush=True)
    z = np.load(VECS, allow_pickle=True)
    vecs = dict(zip(z["queries"].tolist(), z["vectors"]))
    missing = sorted(set(wl.queries()) - set(vecs))
    if missing:                                   # encode what the stored file lacks, with the same encoder
        from ..encoders import get_bi
        print(f"  encoding {len(missing)} queries missing from the vector file", flush=True)
        enc = get_bi().model.encode(missing, batch_size=256, normalize_embeddings=True,
                                    show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        vecs.update(zip(missing, enc))
    metas = train_metas(seed=seed)
    res = {}

    def record(name, stats, events):
        bl = blocks(events)
        res[name] = {"stats": {k: stats.get(k) for k in KEEP}, "blocks": bl,
                     "hit": [int(bool(e["hit"])) for e in events],
                     "correct": [int(bool(e["correct"])) if e["hit"] else 0 for e in events]}
        c = sum(res[name]["correct"]) / len(events) * 100
        print(f"  {name:12s} hit {stats['hit_ratio_pct']:5.2f}%  correct-hit {c:5.2f}%  "
              f"prec {stats['precision_of_hits_pct']:5.1f}%", flush=True)

    with memoized_encoders() as memo:
        memo._bi_cache.update({q: vecs[q] for q in wl.queries()})
        for thr in BI_GRID:
            r = serve("C1_bi_only", wl, _backend(seed), metas, bi_threshold=thr)
            record(f"C1_t{thr:g}", r.stats, r.per_event)
        for thr in META_GRID:
            r = serve("C2_bi_geo_lgb", wl, _backend(seed), metas, meta_threshold=thr)
            record(f"C2_p{thr:g}", r.stats, r.per_event)
    out = {"manifest": hardware_manifest({"start": start, "n": n, "latency_valid": False,
                                          "bi_grid": list(BI_GRID), "meta_grid": list(META_GRID)}, seed=seed),
           "workload": wl.meta, "results": res}
    save_result(out, "exp40_chronological_transfer", tag=tag or f"s{start}_n{n}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=60000)
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--data", default=None, help="parquet with prompt, id_set (stream order); default the gaming stream")
    ap.add_argument("--vecs", default=None, help="npz with queries, vectors")
    a = ap.parse_args()
    if a.data:
        DATA = C.DATA_DIR / a.data
    if a.vecs:
        VECS = C.DATA_DIR / a.vecs
    main(a.start, a.n, tag=a.tag)
