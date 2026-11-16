"""
Experiment 43 -- ReuseGate under grouped (topic-held-out) k-fold over the whole log.

Experiment 39 held out one half of the topics once. Here the 150,000 queries
are clustered into 30 coarse topics (MiniBatchKMeans, seed 0, on the
all-mpnet-base-v2 vectors), the clusters are split into 3 groups of nearly equal
size (greedy balancing), and each group is held out in turn, so every query is
evaluated exactly once on topics that its reference and tuning data do not
contain:

  fold f   evaluation = the queries of group f, with every query whose
           equivalence class also occurs in the other two groups removed
           (coarse clusters can cut a class);
           reference + tuning = the other two groups, in the original row
           order: reference = their first 25,000 queries, tuning = all
           following full windows of 10,000.

The experiment-37 protocol applies unchanged with causal tuning, reference
sizes 0 and 25,000, the same features, label-free target, LightGBM settings,
half-life, threshold grid, capacities, selection rule and tests. Evaluation
windows are all full windows of 10,000 that remain; no evaluation row is used
in any selection. Rows were used in earlier design experiments, so the folds
are not fresh data, but the held-out topics are unseen by the fold's training,
reference and tuning data.

Run: python -m semcache.experiments.exp43_topic_kfold --fold 0 --workers 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from .. import config as C

ORIG = C.DATA_DIR / "external" / "sembenchmark_searchqueries.parquet"
VECS = C.DATA_DIR / "emb_cache" / "exp30_searchqueries_vectors.npz"
K, GROUPS, REF_N, WIN = 30, 3, 25_000, 10_000


def _fold() -> int:
    return int(sys.argv[sys.argv.index("--fold") + 1]) if "--fold" in sys.argv else 0


FOLD = _fold()
PATH = C.DATA_DIR / "external" / f"sembenchmark_searchqueries_topicfold{FOLD}.parquet"
os.environ["SEMCACHE_LOG"] = str(PATH)             # before exp30 is imported; spawned workers inherit it

from . import exp37_causal_reusegate as E37       # noqa: E402


def build() -> dict:
    from sklearn.cluster import MiniBatchKMeans
    df = pd.read_parquet(ORIG)
    z = np.load(VECS, allow_pickle=True)
    vec = dict(zip(z["queries"].tolist(), z["vectors"]))
    X = np.stack([vec[q] for q in df["prompt"].astype(str)]).astype(np.float32)
    lab = MiniBatchKMeans(K, random_state=0, n_init=3, batch_size=4096).fit_predict(X)
    sizes = np.bincount(lab, minlength=K)
    grp = np.zeros(K, dtype=int)
    load = [0] * GROUPS
    for c in np.argsort(-sizes):                    # greedy: biggest cluster to the lightest group
        g = int(np.argmin(load))
        grp[c] = g
        load[g] += int(sizes[c])
    row_g = grp[lab]
    a = np.flatnonzero(row_g != FOLD)
    b = np.flatnonzero(row_g == FOLD)
    gid = df["id_set"].to_numpy()
    shared = np.isin(gid[b], np.unique(gid[a]))
    out = pd.concat([df.iloc[a], df.iloc[b[~shared]]], ignore_index=True)[["prompt", "id_set"]]
    out.to_parquet(PATH, index=False)
    info = {"fold": FOLD, "clusters_per_group": np.bincount(grp, minlength=GROUPS).tolist(),
            "rows_train_side": int(len(a)), "rows_heldout_group": int(len(b)),
            "rows_dropped_shared_class": int(shared.sum()), "rows_eval_side": int((~shared).sum()),
            "classes_shared_between_sides": int(len(set(out.id_set[:len(a)]) & set(out.id_set[len(a):])))}
    print("  topic fold:", json.dumps(info), flush=True)
    (C.RAW_DIR / "revision").mkdir(parents=True, exist_ok=True)
    (C.RAW_DIR / "revision" / f"exp43_fold{FOLD}_split.json").write_text(json.dumps(info, indent=1), encoding="utf8")
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--with-group", action="store_true")
    a = ap.parse_args()
    info = build()
    n_a = info["rows_train_side"]
    E37.CAUSAL_TUNING = True
    E37.WITH_GROUP = a.with_group
    E37.REF = (0, REF_N)
    E37.TUNE = tuple((s, WIN) for s in range(REF_N, n_a - WIN + 1, WIN))
    full, rest = divmod(info["rows_eval_side"], WIN)
    E37.EVAL = tuple((n_a + i * WIN, WIN) for i in range(full)) + (((n_a + full * WIN, rest),) if rest >= WIN // 2 else ())
    E37.REF_SIZES = (0, REF_N)
    print(f"  tuning windows {len(E37.TUNE)}, evaluation windows {len(E37.EVAL)}", flush=True)
    tag = f"topicfold{a.fold}"
    if a.smoke:
        E37.REF = (0, 2000)
        E37.TUNE = ((2000, 1500), (3500, 1500), (5000, 1500))
        E37.EVAL = ((n_a, 1500), (n_a + 1500, 1500))
        E37.CAPS, E37.TAUS, E37.LAMBDAS, E37.REF_SIZES = (200,), (0.895, 0.94), (0, 1), (0, 2000)
        E37.BASE = ("LRU", "WTinyLFU")
        tag += "_smoke"
    E37.main(a.workers, tag)
