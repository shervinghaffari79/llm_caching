"""
Experiment 39 -- ReuseGate under a change of traffic distribution: evaluation
on topics that neither the reference traffic nor the training windows contain.

Motivation: the real log is one sample from one domain, and ReuseGate takes
neighbour statistics from reference traffic. If that reference only helps for
topics it covers, the policy would not survive a change in what users ask.
This experiment holds the topics of the evaluation windows out entirely.

Construction (seeded, fixed before the run): the 150,000 log queries are
clustered into 30 coarse topics by MiniBatchKMeans (k = 30, seed 0) on the
all-mpnet-base-v2 vectors. Topics are shuffled (seed 0) and assigned to side A
until A holds at least 75,000 queries; the remaining topics form side B. The
stream keeps the log's original row order within each side. Side A supplies

    reference corpus  A rows      0 - 24,999
    tuning windows    A rows 25,000 - 74,999  (five windows of 10,000)

and the first 50,000 queries of side B supply the five evaluation windows. No
topic occurs on both sides. Coarse clustering can cut an equivalence class
across a topic boundary, so side-B rows whose class also occurs in the side-A
rows used for reference and tuning are dropped before the windows are formed;
the evaluation windows then share no class with the reference or tuning data
(neighbours across topic boundaries remain possible and are what the test
measures). Everything else is experiment 37: same features, label-free target,
LightGBM settings, half-life, threshold grid, capacities, selection rule, and
the same tests, with the reference size at 25,000 (all of A's reference part)
and at 0 (cold start).

Pre-set tests (paired t-tests over the five windows, one Holm family):
  H1d  Learned[match] - W-TinyLFU at lambda = 0, 1, 2
  H2d  Learned[match] - LRU and - LFU at lambda = 1
Results are reported whatever their sign.

Run: python -m semcache.experiments.exp39_topic_shift_reusegate --workers 30
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from .. import config as C

ORIG = C.DATA_DIR / "external" / "sembenchmark_searchqueries.parquet"
SPLIT = C.DATA_DIR / "external" / "sembenchmark_searchqueries_topicsplit.parquet"
os.environ["SEMCACHE_LOG"] = str(SPLIT)          # before exp30 is imported; spawned workers inherit it

from . import exp30_sd_redesign as E30            # noqa: E402
from . import exp37_causal_reusegate as E37       # noqa: E402

K, SIDE_A, SIDE_B = 30, 75_000, 50_000


def build_split() -> dict:
    from sklearn.cluster import MiniBatchKMeans
    df = pd.read_parquet(ORIG)
    z = np.load(E30.LOG_VECS, allow_pickle=True)
    vec = dict(zip(z["queries"].tolist(), z["vectors"]))
    miss = [q for q in df["prompt"].astype(str) if q not in vec]
    if miss:
        raise RuntimeError(f"{len(miss)} log queries have no stored vector; run experiment 36/37 first")
    X = np.stack([vec[q] for q in df["prompt"].astype(str)]).astype(np.float32)
    lab = MiniBatchKMeans(K, random_state=0, n_init=3, batch_size=4096).fit_predict(X)
    order = np.random.default_rng(0).permutation(K)
    sizes = np.bincount(lab, minlength=K)
    a, n = [], 0
    for c in order:
        if n >= SIDE_A:
            break
        a.append(int(c))
        n += int(sizes[c])
    b = [int(c) for c in range(K) if c not in a]
    ia = np.flatnonzero(np.isin(lab, a))
    ib = np.flatnonzero(np.isin(lab, b))
    gids = df["id_set"].to_numpy()
    in_a = set(gids[ia[:SIDE_A]].tolist())
    n_b_before = len(ib)
    ib = ib[~np.isin(gids[ib], list(in_a))]       # a class cut by the topic boundary stays on side A only
    if len(ia) < SIDE_A or len(ib) < SIDE_B:
        raise RuntimeError(f"split too small: A {len(ia)}, B {len(ib)}")
    keep = np.concatenate([ia[:SIDE_A], ib[:SIDE_B]])
    out = df.iloc[keep][["prompt", "id_set"]].reset_index(drop=True)
    out.to_parquet(SPLIT, index=False)
    ga, gb = set(out.id_set[:SIDE_A]), set(out.id_set[SIDE_A:])
    info = {"k": K, "topics_A": a, "topics_B": b, "rows_A_total": int(len(ia)), "rows_B_total": int(len(ib)), "rows_B_dropped_shared_class": int(n_b_before - len(ib)),
            "rows_used": [SIDE_A, SIDE_B], "classes_shared_between_sides": len(ga & gb)}
    print("  topic split:", json.dumps(info), flush=True)
    (C.RAW_DIR / "revision").mkdir(parents=True, exist_ok=True)
    (C.RAW_DIR / "revision" / "exp39_topic_split.json").write_text(json.dumps(info, indent=1), encoding="utf8")
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="topic")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--with-group", action="store_true")
    ap.add_argument("--causal-tuning", action="store_true")
    ap.add_argument("--base-suffix", default=None)
    a = ap.parse_args()
    info = build_split()
    E37.CAUSAL_TUNING = a.causal_tuning
    E37.WITH_GROUP = a.with_group
    E37.BASE_SFX = a.base_suffix
    E37.REF = (0, 25_000)
    E37.TUNE = tuple((s, 10_000) for s in range(25_000, 75_000, 10_000))
    E37.EVAL = tuple((s, 10_000) for s in range(75_000, 125_000, 10_000))
    E37.REF_SIZES = (0, 25_000)
    if a.smoke:
        E37.REF = (0, 2_000)
        E37.TUNE = ((2_000, 1_500), (3_500, 1_500))
        E37.EVAL = ((75_000, 1_500), (76_500, 1_500))
        E37.CAPS, E37.TAUS, E37.LAMBDAS, E37.REF_SIZES = (200,), (0.895, 0.94), (0, 1), (0, 2_000)
        E37.BASE = ("LRU", "WTinyLFU")
        a.tag = "topic_smoke"
    E37.main(a.workers, a.tag)
