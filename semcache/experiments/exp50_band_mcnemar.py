"""
Experiment 50 -- band-conditioned diagnostic, decision-level paired comparison and error taxonomy
(docs/PREREGISTRATION_round2b.md, E6).

QQP meta set (30,000 pairs), group-disjoint 5-fold out-of-fold scores for C1 (cosine), C2 and C2-diff.
  * Band: AUC and precision only for pairs whose cosine lies within +/-0.05 of 0.726 (the QQP default) and of 0.895
    (the search-log default).
  * Decision level at 90% matched recall: each system's threshold accepts 90% of true duplicates; then a paired
    comparison with C1: errors fixed (C1 wrong, system right), errors added (C1 right, system wrong), exact McNemar
    p-value, split into false hits (non-duplicates accepted) and misses (duplicates rejected).
  * Taxonomy: each pair gets one category by priority (number mismatch, negation difference, capitalised-token
    difference, question-type difference, length ratio below 0.5, other); fixed and added errors are counted per category.

Run: python -m semcache.experiments.exp50_band_mcnemar
"""
from __future__ import annotations

import argparse
import re
import time

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from .. import config as C
from ..data import build_splits, make_group_folds
from ..metaclf import make_pipeline
from ._common import banner, encode_frame, hardware_manifest, save_result
from .exp48_targeted_features import NEG, NUM, TOK, WH
from .exp49_slices import CAP, SYSTEMS, feats, thr_for_recall

BANDS = ((0.726, "QQP default 0.726"), (0.895, "search-log default 0.895"))
CATS = ("number", "negation", "capitalised", "question_type", "length_ratio", "other")


def category(a: str, b: str) -> str:
    if set(NUM.findall(a)) != set(NUM.findall(b)):
        return "number"
    ta, tb = set(TOK.findall(a.lower())), set(TOK.findall(b.lower()))
    if bool(ta & NEG or "n't" in a.lower()) != bool(tb & NEG or "n't" in b.lower()):
        return "negation"
    ca, cb = set(CAP.findall(" ".join(a.split()[1:]))), set(CAP.findall(" ".join(b.split()[1:])))
    if ca != cb:
        return "capitalised"
    wa = WH.get(a.lower().split()[0], 99) if a.split() else 99
    wb = WH.get(b.lower().split()[0], 99) if b.split() else 99
    if wa != wb:
        return "question_type"
    if min(len(a), len(b)) / max(1, max(len(a), len(b))) < 0.5:
        return "length_ratio"
    return "other"


def main(seed: int = C.SEED, tag: str = "") -> dict:
    banner("EXP 50 | band diagnostic, McNemar comparison and error taxonomy")
    t0 = time.time()
    meta = build_splits().meta
    enc = encode_frame(meta, "meta30000", with_cross=False)
    y, groups = enc.y, np.asarray(enc.groups)
    qa, qb = meta.question1.tolist(), meta.question2.tolist()
    X2, X2d = feats(qa, qb, enc.e1, enc.e2)
    cos = X2[:, 0]
    folds = make_group_folds(y, groups, C.CV_FOLDS_META)
    sc = {s: np.full(len(y), np.nan) for s in SYSTEMS}
    sc["C1"] = cos.copy()
    for f, (tr, va) in enumerate(folds):
        m2, md = make_pipeline("LightGBM", seed=seed + f), make_pipeline("LightGBM", seed=seed + f)
        m2.fit(X2[tr], y[tr])
        md.fit(X2d[tr], y[tr])
        sc["C2"][va] = m2.predict_proba(X2[va])[:, 1]
        sc["C2-diff"][va] = md.predict_proba(X2d[va])[:, 1]
        print(f"  fold {f} [{time.time() - t0:.0f}s]", flush=True)

    out = {"preregistration": "docs/PREREGISTRATION_round2b.md section E6", "pairs": int(len(y)),
           "positive_rate": float(y.mean())}
    # ---- bands ---------------------------------------------------------------------------------------
    thr = {s: thr_for_recall(sc[s][y == 1], 0.90) for s in SYSTEMS}
    dec = {s: sc[s] >= thr[s] for s in SYSTEMS}
    out["thresholds_at_recall_0.90"] = thr
    out["bands"] = {}
    for centre, label in BANDS:
        band = np.abs(cos - centre) <= 0.05
        yb = y[band]
        row = {"pairs_in_band": int(band.sum()), "positive_rate": float(yb.mean()) if band.any() else None}
        for s in SYSTEMS:
            acc = dec[s][band]
            row[s] = {"auc": float(roc_auc_score(yb, sc[s][band])) if len(np.unique(yb)) == 2 else None,
                      "accept_rate": float(acc.mean()), "precision": float(yb[acc].mean()) if acc.any() else None}
        out["bands"][label] = row
        print(f"  band {label}: n={band.sum()} AUC " + " ".join(f"{s} {row[s]['auc']:.3f}" for s in SYSTEMS), flush=True)

    # ---- paired decisions and taxonomy ------------------------------------------------------------
    err = {s: dec[s] != (y == 1) for s in SYSTEMS}
    cat = np.array([category(a, b) for a, b in zip(qa, qb)])
    out["mcnemar_at_recall_0.90"], out["taxonomy"] = {}, {}
    out["total_errors"] = {s: int(err[s].sum()) for s in SYSTEMS}
    for s in ("C2", "C2-diff"):
        fixed, added = err["C1"] & ~err[s], ~err["C1"] & err[s]
        res = {"fixed": int(fixed.sum()), "added": int(added.sum()),
               "mcnemar_p": float(binomtest(int(fixed.sum()), int(fixed.sum() + added.sum()), 0.5).pvalue)
               if fixed.sum() + added.sum() else 1.0}
        for name, mask in (("false_hits", y == 0), ("misses", y == 1)):
            f, a = int((fixed & mask).sum()), int((added & mask).sum())
            res[name] = {"fixed": f, "added": a,
                         "mcnemar_p": float(binomtest(f, f + a, 0.5).pvalue) if f + a else 1.0}
        out["mcnemar_at_recall_0.90"][s] = res
        out["taxonomy"][s] = {c: {"pairs": int((cat == c).sum()), "C1_errors": int((err["C1"] & (cat == c)).sum()),
                                  "fixed": int((fixed & (cat == c)).sum()), "added": int((added & (cat == c)).sum())}
                              for c in CATS}
        print(f"  {s}: fixed {res['fixed']} added {res['added']} McNemar p {res['mcnemar_p']:.3g}", flush=True)
    out["manifest"] = hardware_manifest({"recall": 0.90}, seed=seed)
    save_result(out, "exp50_band_mcnemar", tag=tag)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    main(tag=ap.parse_args().tag)
