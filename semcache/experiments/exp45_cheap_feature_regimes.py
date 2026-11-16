"""
Experiment 45 -- where do the cheap features (C2) beat the bare bi-encoder (C1)?

Question: is there a regime in which C2 (cosine + L1/L2 distance + element-wise
product and |difference| + six text features, LightGBM) exceeds C1 (cosine
only) by a large margin at the same recall?

Protocol, fixed before any run. The family of comparisons is small and closed:

  R1  encoder strength: five encoders from weak to strong, full training data.
  R2  labelled-data budget: training rows capped at 500, 2,000, 8,000 and all,
      for the weakest and the default encoder.
  R3  confusable region: AUC restricted to the pairs whose cosine lies in the
      top quintile of the encoder's own scores, where C1 has to decide.

Data: the 30,000-pair QQP meta set, group-disjoint 5-fold CV (a question
component never spans train and validation). C1 needs no training and its score
is the cosine. C2 is fitted inside each fold only. Both are scored on the same
pooled out-of-fold pairs.

Metrics (threshold-free, so no operating point is tuned on validation labels):
ROC-AUC, average precision, and precision at 80% and 90% recall read off the
pooled out-of-fold curve. The difference C2 - C1 carries a 95% bootstrap CI
over question components. A gap is called large when its CI excludes 0 and the
point estimate is at least 5 points; the number of regimes examined is reported
with every result, so a large gap among many is not presented alone.

Run: python -m semcache.experiments.exp45_cheap_feature_regimes
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .. import config as C
from ..data import build_splits, make_group_folds
from ..metaclf import cross_validate
from ._common import banner, encode_frame, hardware_manifest, save_result

ENCODERS = ("sentence-transformers/paraphrase-MiniLM-L3-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L12-v2",
            "BAAI/bge-small-en-v1.5",
            "sentence-transformers/all-mpnet-base-v2")
CAPS = (500, 2000, 8000, None)
CAP_ENCODERS = (ENCODERS[0], ENCODERS[-1])
C2_GROUPS = ("bi_cosine", "distance", "elementwise", "text")
B = 200


def p_at_recall(y, s, r):
    p, rec, _ = precision_recall_curve(y, s)
    ok = rec >= r
    return float(p[ok].max()) if ok.any() else float("nan")


def metrics(y, s, band=None):
    out = {"auc": roc_auc_score(y, s), "ap": average_precision_score(y, s),
           "p_at_r80": p_at_recall(y, s, 0.80), "p_at_r90": p_at_recall(y, s, 0.90)}
    if band is not None and len(np.unique(y[band])) == 2:
        out["auc_top_quintile_cosine"] = roc_auc_score(y[band], s[band])
    return out


def cap_folds(folds, groups, cap, seed):
    if cap is None:
        return folds
    rng = np.random.default_rng(seed)
    out = []
    for tr, va in folds:
        if cap >= len(tr):
            out.append((tr, va)); continue
        df = pd.DataFrame({"g": groups[tr], "i": tr})
        order = rng.permutation(df.g.unique())
        sizes = df.groupby("g").size().reindex(order).cumsum()
        keep = set(order[: int((sizes <= cap).sum()) or 1])
        out.append((df.i[df.g.isin(keep)].to_numpy(), va))
    return out


def compare(y, groups, s1, s2, seed):
    band = s1 >= np.quantile(s1, 0.80)
    m1, m2 = metrics(y, s1, band), metrics(y, s2, band)
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    rows_of = pd.Series(np.arange(len(y))).groupby(groups).apply(lambda s: s.to_numpy())
    diffs = {k: [] for k in m1}
    for _ in range(B):
        idx = np.concatenate([rows_of[g] for g in rng.choice(ug, len(ug))])
        bb = band[idx]
        a, b = metrics(y[idx], s1[idx], bb), metrics(y[idx], s2[idx], bb)
        for k in diffs:
            if k in a and k in b:
                diffs[k].append(b[k] - a[k])
    out = {}
    for k in m1:
        d = np.array(diffs[k])
        out[k] = {"C1": m1[k], "C2": m2[k], "delta": m2[k] - m1[k],
                  "ci": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}
    return out


def main(seed: int = C.SEED, tag: str = "") -> dict:
    banner("EXP 45 | where do the cheap features beat the bare bi-encoder?")
    meta = build_splits().meta
    results, n_regimes = {}, 0
    for enc_name in ENCODERS:
        short = enc_name.split("/")[-1]
        t0 = time.time()
        enc = encode_frame(meta, f"meta30000_{short}", with_cross=False, bi_name=enc_name)
        y, groups = enc.y, np.asarray(enc.groups)
        s1 = enc.bundle.matrix(("bi_cosine",))[:, 0]
        X2 = enc.bundle.matrix(C2_GROUPS)
        folds = make_group_folds(y, groups, C.CV_FOLDS_META)
        for cap in (CAPS if enc_name in CAP_ENCODERS else (None,)):
            cv = cross_validate(X2, y, cap_folds(folds, groups, cap, seed), "LightGBM",
                                name=f"{short}_cap{cap}", seed=seed, fold_scheme="stratified_group_kfold")
            n_regimes += 1
            r = compare(y, groups, s1, cv.oof_prob, seed)
            key = f"{short}|train_cap={cap}"
            results[key] = r
            print(f"  {key:62s} AUC {r['auc']['C1']:.3f}->{r['auc']['C2']:.3f}  "
                  f"P@R90 {r['p_at_r90']['C1']:.3f}->{r['p_at_r90']['C2']:.3f} "
                  f"({r['p_at_r90']['delta']:+.3f} [{r['p_at_r90']['ci'][0]:+.3f},{r['p_at_r90']['ci'][1]:+.3f}])  "
                  f"top-quintile AUC {r['auc_top_quintile_cosine']['C1']:.3f}->{r['auc_top_quintile_cosine']['C2']:.3f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    out = {"manifest": hardware_manifest({"encoders": list(ENCODERS), "caps": list(CAPS), "bootstrap": B,
                                          "regimes_examined": n_regimes, "large_gap_rule": "CI excludes 0 and delta >= 0.05"},
                                         seed=seed),
           "results": results}
    save_result(out, "exp45_cheap_feature_regimes", tag=tag)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    main(tag=ap.parse_args().tag)
