"""
Experiment 51 -- vCache benchmark datasets: SemBenchmarkCombo and SemBenchmarkClassification
(docs/PREREGISTRATION_round2b.md, E8).

Combo (27,500 prompts) = 3,500 LmArena prompts that have no match (one prompt per class) + 24,000 classification queries,
whose ID_Set groups prompts with the same expected response. A hit served for an LmArena prompt is a guaranteed
false hit; a hit on a classification query is correct when the ID_Set matches.

Template hypothesis (checked, not assumed): a shared instruction template inflates the cosine between short prompts.
  * longest common character prefix and suffix per source dataset, and the share of prompt characters they cover;
  * cosine of random non-equivalent pairs, raw vs with the common prefix/suffix stripped;
  * C1 and C2 (QQP-trained) false-hit rate on the no-match prompts at matched hit ratio on the matchable prompts.
Classification (45,000 rows, no class column) is used for the template diagnostics only.

Run: python -m semcache.experiments.exp51_combo --data external/sembenchmark_combo.parquet --cls external/sembenchmark_classification.parquet
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from .. import config as C
from ..encoders import get_bi
from ..llm import NullBackend
from ..workload import StreamEvent, Workload
from ._common import banner, hardware_manifest, memoized_encoders, save_result, serve, train_metas

C1_GRID = (0.70, 0.76, 0.82, 0.86, 0.90, 0.93, 0.96, 0.98)
C2_GRID = (0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98)
NO_MATCH = "lm arena"


def encode(texts):
    return get_bi().model.encode(list(texts), batch_size=128, normalize_embeddings=True, show_progress_bar=False,
                                 convert_to_numpy=True).astype(np.float32)


def template_edges(texts, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    s = [texts[i] for i in rng.choice(len(texts), min(n, len(texts)), replace=False)]
    pre = os.path.commonprefix(s)
    suf = os.path.commonprefix([t[::-1] for t in s])[::-1]
    return pre, suf


def strip(t, pre, suf):
    t = t[len(pre):] if pre and t.startswith(pre) else t
    return t[:-len(suf)] if suf and t.endswith(suf) else t


def cos_table(V, ids, ds, rng, n_pairs=200_000):
    i, j = rng.integers(0, len(V), n_pairs), rng.integers(0, len(V), n_pairs)
    keep = (i != j) & (ids[i] != ids[j])
    i, j = i[keep], j[keep]
    c = np.einsum("ij,ij->i", V[i], V[j])
    same_ds = ds[i] == ds[j]
    q = lambda x: {"mean": float(x.mean()), "p50": float(np.quantile(x, 0.5)), "p95": float(np.quantile(x, 0.95)),
                   "share_ge_0.726": float((x >= 0.726).mean()), "share_ge_0.895": float((x >= 0.895).mean())}
    return {"within_dataset": q(c[same_ds]) if same_ds.any() else None, "across_datasets": q(c[~same_ds])}


def curve(df, order, wl, thr_grid, key, kw, metas):
    rows = []
    mask_nm = (df.dataset_name.to_numpy()[order] == NO_MATCH)
    for thr in thr_grid:
        r = serve(key, wl, NullBackend(), metas, **{kw: thr})
        ev = r.per_event
        hit = np.array([bool(e["hit"]) for e in ev])
        cor = np.array([bool(e["hit"] and e["correct"]) for e in ev])
        rows.append({"threshold": thr, "hit_ratio_matchable": float(hit[~mask_nm].mean()),
                     "precision_matchable": float(cor[~mask_nm].sum() / max(1, hit[~mask_nm].sum())),
                     "false_hit_rate_no_match": float(hit[mask_nm].mean()), "hits_no_match": int(hit[mask_nm].sum()),
                     "n_no_match": int(mask_nm.sum())})
        print(f"  {key} thr {thr:.2f}: matchable hit {rows[-1]['hit_ratio_matchable']:.3f} prec "
              f"{rows[-1]['precision_matchable']:.3f} | no-match false hits {rows[-1]['false_hit_rate_no_match']:.3f}", flush=True)
    return rows


def at_matched(rows_a, rows_b, grid=(0.3, 0.5, 0.7, 0.9)):
    xa = np.array([r["hit_ratio_matchable"] for r in rows_a])
    xb = np.array([r["hit_ratio_matchable"] for r in rows_b])
    out = {}
    for g in grid:
        f = lambda rows, xs: (float(np.interp(g, np.sort(xs), np.array([r["false_hit_rate_no_match"] for r in rows])[np.argsort(xs)]))
                              if xs.min() <= g <= xs.max() else None)
        out[str(g)] = {"C1": f(rows_a, xa), "C2": f(rows_b, xb)}
    return out


def main(data: str, cls: str, seed: int = C.SEED, tag: str = "") -> dict:
    banner("EXP 51 | vCache Combo / Classification")
    t0 = time.time()
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(C.DATA_DIR / data)
    prompts = df.prompt.astype(str).tolist()
    ids = pd.factorize(df.ID_Set)[0]
    ds = df.dataset_name.to_numpy()
    V = encode(prompts)
    print(f"  combo: {len(df)} prompts, {len(np.unique(ids))} classes, encoded [{time.time() - t0:.0f}s]", flush=True)
    out = {"preregistration": "docs/PREREGISTRATION_round2b.md section E8", "combo_rows": int(len(df)),
           "combo_classes": int(len(np.unique(ids))), "template": {}}

    # ---- template diagnostics ---------------------------------------------------------------------
    stripped = list(prompts)
    for d in np.unique(ds):
        idx = np.flatnonzero(ds == d)
        pre, suf = template_edges([prompts[i] for i in idx])
        for i in idx:
            stripped[i] = strip(prompts[i], pre, suf)
        share = float(np.mean([(len(pre) + len(suf)) / max(1, len(prompts[i])) for i in idx]))
        out["template"][d] = {"n": int(len(idx)), "prefix_chars": len(pre), "suffix_chars": len(suf),
                              "template_share_of_prompt": share, "mean_prompt_chars": float(np.mean([len(prompts[i]) for i in idx])),
                              "prefix_example": pre[:160], "suffix_example": suf[:160]}
        print(f"  template {d}: prefix {len(pre)} chars, suffix {len(suf)} chars, share {share:.2f}", flush=True)
    Vs = encode(stripped)
    out["cosine_non_equivalent_pairs"] = {"raw": cos_table(V, ids, ds, rng), "template_stripped": cos_table(Vs, ids, ds, rng)}
    print("  cosine raw vs stripped:", {k: v["across_datasets"]["share_ge_0.726"] for k, v in out["cosine_non_equivalent_pairs"].items()},
          flush=True)
    if cls:
        dc = pd.read_parquet(C.DATA_DIR / cls)
        pc = dc.prompt.astype(str).tolist()
        out["classification_template"] = {}
        for d in dc.dataset_name.unique():
            sub = [pc[i] for i in np.flatnonzero(dc.dataset_name.to_numpy() == d)]
            pre, suf = template_edges(sub)
            out["classification_template"][d] = {"n": len(sub), "prefix_chars": len(pre), "suffix_chars": len(suf),
                                                 "template_share_of_prompt": float(np.mean([(len(pre) + len(suf)) / max(1, len(t)) for t in sub]))}

    # ---- stream: false hits on the no-match prompts at matched hit ratio on the matchable ones ----------
    order = rng.permutation(len(df))
    seen, events = set(), []
    for step, i in enumerate(order):
        events.append(StreamEvent(step=step, query=prompts[i], topic_id=int(ids[i]), is_repeat=int(ids[i]) in seen,
                                  is_exact=False, is_hard_negative=False, session_id=step, turn=0))
        seen.add(int(ids[i]))
    wl = Workload(events=events, meta={"source": "vCache/SemBenchmarkCombo", "n_events": len(events), "order": "random, seed 42"})
    metas = train_metas(seed=seed, keys=("C2_bi_geo_lgb",))
    with memoized_encoders() as memo:
        memo._bi_cache.update({prompts[i]: V[i] for i in range(len(prompts))})
        c1 = curve(df, order, wl, C1_GRID, "C1_bi_only", "bi_threshold", None)
        c2 = curve(df, order, wl, C2_GRID, "C2_bi_geo_lgb", "meta_threshold", metas)
    out["stream"] = {"C1": c1, "C2": c2, "false_hit_rate_no_match_at_matched_matchable_hit_ratio": at_matched(c1, c2)}
    out["manifest"] = hardware_manifest({"generation": "none", "latency_valid": False}, seed=seed)
    save_result(out, "exp51_combo", tag=tag)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="external/sembenchmark_combo.parquet")
    ap.add_argument("--cls", default="external/sembenchmark_classification.parquet")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(a.data, a.cls, tag=a.tag)
