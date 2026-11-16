"""
Experiment 25 -- the decision layer retrained on the target domain
(SemBenchmarkSearchQueries).

Experiment 22 applies C1's threshold and C2's meta-classifier, both fitted on
QQP, to a real search-query log. Here every configuration is adapted to the
log under the same rule, using data disjoint from the test stream:

* test stream: the first N queries of the log (as in experiment 22);
* training data: queries after position N whose equivalence group (id_set)
  never occurs in the test stream. Each training query is paired with its
  5 nearest training queries under the bi-encoder, and a pair is positive when
  both share an id_set, so the training pairs follow the candidate
  distribution a cache sees;
* C1: threshold re-tuned on those pairs (F1-maximising);
* C2 and C4: meta-classifiers refitted on those pairs (C5 uses the refitted
  C2 model with the same gate band as elsewhere).

The test stream is then served with each adapted configuration. `--systems`
selects a subset so the work can be split across processes; `--merge` joins
the parts.

Encoders are memoised by text; latency and token counts are not reported.

Run: python -m semcache.experiments.exp25_searchqueries_retrain --systems C1 C2
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from .. import config as C
from ..encoders import get_bi, warmup
from ..metaclf import fit_final
from ._common import (META_GROUPS, MODEL_DIR, REVISION_DIR, add_run_args, banner,
                      encode_frame, hardware_manifest, memoized_encoders,
                      save_result, serve)
from .exp22_searchqueries import BLOCK, DATA, KEEP, META_GRID, SOURCE, _backend, blocks, load_stream

K_NEIGHBOURS = 5
MAX_TRAIN_QUERIES = int(__import__("os").environ.get("SQ_MAX_TRAIN", "40000"))


def training_pairs(n_test: int, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    test_sets = set(df["id_set"].iloc[:n_test])
    tr = df.iloc[n_test:]
    tr = tr[~tr["id_set"].isin(test_sets)]
    if len(tr) > MAX_TRAIN_QUERIES:
        tr = tr.sample(MAX_TRAIN_QUERIES, random_state=seed)
    q = tr["prompt"].astype(str).tolist()
    g = tr["id_set"].to_numpy()
    emb = get_bi().encode(q)
    import faiss
    faiss.omp_set_num_threads(int(__import__("os").environ.get("SEMCACHE_FAISS_THREADS", "8")))
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    _, nn = index.search(emb, K_NEIGHBOURS + 1)
    seen, rows = set(), []
    for i in range(len(q)):
        for j in nn[i]:
            if j < 0 or j == i:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            rows.append((q[i], q[j], int(g[i] == g[j])))
    out = pd.DataFrame(rows, columns=["question1", "question2", "is_duplicate"])
    print(f"  training: {len(q)} queries from {tr['id_set'].nunique()} groups disjoint from "
          f"the test stream -> {len(out)} pairs, {100 * out.is_duplicate.mean():.1f}% positive",
          flush=True)
    return out


def fit(n_test: int, seed: int, with_c4: bool = True) -> dict:
    tag = f"sq_n{n_test}_seed{seed}"
    keys = ("C2_bi_geo_lgb", "C4_full_stacking") if with_c4 else ("C2_bi_geo_lgb",)
    paths = {k: MODEL_DIR / f"meta_{k}_{tag}.joblib" for k in keys}
    thr_path = MODEL_DIR / f"c1_threshold_{tag}.json"
    if all(p.exists() for p in paths.values()) and thr_path.exists():
        metas = {k: joblib.load(p).set_inference_threads() for k, p in paths.items()}
        info = json.loads(thr_path.read_text())
    else:
        pairs = training_pairs(n_test, seed)
        enc = encode_frame(pairs, f"searchq_train_{tag}", with_cross=with_c4)
        cos = enc.bundle.matrix(("bi_cosine",))[:, 0]
        grid = np.round(np.arange(0.50, 0.99, 0.005), 3)
        f1 = []
        for t in grid:
            pred = cos >= t
            tp = float(np.sum(pred & (enc.y == 1)))
            f1.append(2 * tp / max(1.0, pred.sum() + enc.y.sum()))
        thr = float(grid[int(np.argmax(f1))])
        metas = {}
        for k in paths:
            metas[k] = fit_final(enc.bundle.matrix(META_GROUPS[k]), enc.y, META_GROUPS[k], seed=seed)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(metas[k], paths[k])
            metas[k].set_inference_threads()
        info = {"c1_threshold": thr, "c1_train_f1": float(max(f1)), "n_pairs": int(len(pairs)),
                "positive_pct": float(100 * pairs.is_duplicate.mean())}
        thr_path.write_text(json.dumps(info))
    metas["C5_gated"] = metas["C2_bi_geo_lgb"]
    print(f"  adapted C1 threshold {info['c1_threshold']} (train F1 {info['c1_train_f1']:.3f})", flush=True)
    return {"metas": metas, "info": info}


def main(n: int = 50000, seed: int = C.SEED, systems=("C1", "C2", "C4", "C5"),
         tag: str = "", fit_only: bool = False, bi_grid=(), meta_grid=META_GRID) -> dict:
    banner(f"EXP 25 | SearchQueries with the decision layer retrained in-domain ({','.join(systems)})")
    with_c4 = "C4" in systems
    warmup(include_cross=with_c4 or "C5" in systems)
    fitted = fit(n, seed, with_c4=with_c4)
    if fit_only:
        return fitted["info"]
    metas = fitted["metas"]
    wl = load_stream(n)
    rep = np.array([e.is_repeat for e in wl.events])
    res = {}

    def record(name, r):
        bl = blocks(r.per_event)
        for b in bl:
            b["repeat_rate_pct"] = 100.0 * float(rep[b["start"]:b["start"] + b["n"]].mean())
        res[name] = {"stats": {k: r.stats.get(k) for k in KEEP}, "blocks": bl}
        print(f"  {name:24s} hit {r.stats['hit_ratio_pct']:5.1f}%  prec {r.stats['precision_of_hits_pct']:5.1f}%",
              flush=True)

    with memoized_encoders() as memo:
        memo.prefill(wl.queries())
        if "C1" in systems:
            t = fitted["info"]["c1_threshold"]
            record(f"C1_adapted_t{t:g}", serve("C1_bi_only", wl, _backend(seed), metas, bi_threshold=t))
        for t in bi_grid:
            record(f"C1_t{t:g}", serve("C1_bi_only", wl, _backend(seed), metas, bi_threshold=t))
        if "C2" in systems:
            for p in meta_grid:
                record(f"C2_adapted_p{p:g}", serve("C2_bi_geo_lgb", wl, _backend(seed), metas,
                                                   meta_threshold=p))
        if "C4" in systems:
            record("C4_adapted", serve("C4_full_stacking", wl, _backend(seed), metas))
        if "C5" in systems:
            record("C5_adapted", serve("C5_gated", wl, _backend(seed), metas))

    out = {"manifest": hardware_manifest({"dataset": SOURCE, "n_stream": n, "systems": list(systems),
                                          "latency_valid": False, "token_counts_valid": False,
                                          "training": fitted["info"], "k_neighbours": K_NEIGHBOURS,
                                          "bi_grid": list(bi_grid), "meta_grid": list(meta_grid),
                                          "max_train_queries": MAX_TRAIN_QUERIES, "block": BLOCK},
                                         seed=seed),
           "workload": wl.meta, "results": res}
    save_result(out, "exp25_searchqueries_retrain", tag=tag or f"n{n}_{'-'.join(systems)}")
    return out


def merge(files: list[str], tag: str) -> dict:
    parts = [json.load(open(f, encoding="utf8")) for f in files]
    out = {"manifest": {**parts[0]["manifest"], "merged_from": files},
           "workload": parts[0]["workload"],
           "results": {k: v for p in parts for k, v in p["results"].items()}}
    save_result(out, "exp25_searchqueries_retrain", tag=tag)
    return out


if __name__ == "__main__":
    ap = add_run_args(argparse.ArgumentParser())
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--systems", nargs="+", default=["C1", "C2", "C4", "C5"])
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--merge", nargs="*", default=None)
    ap.add_argument("--bi-grid", type=float, nargs="*", default=[],
                    help="extra C1 thresholds to serve (QQP-independent sweep)")
    ap.add_argument("--meta-grid", type=float, nargs="*", default=list(META_GRID))
    a = ap.parse_args()
    if a.merge is not None:
        merge(a.merge, a.tag or "merged")
    else:
        main(a.n, a.seed, tuple(a.systems), a.tag, a.fit_only, tuple(a.bi_grid), tuple(a.meta_grid))
