"""
Experiment 4 -- meta-classifier tables (basic vs extended features) recomputed
with fold-internal normalisation and under both fold schemes.

Two things are established here:

1. **Normalisation is fitted inside each training fold.** The scaler is a step
   of an sklearn Pipeline that is constructed and fitted inside the fold loop,
   so mu and sigma never see validation rows. The magnitude of the difference
   against the incorrect protocol is measured rather than asserted.

2. **Fold scheme matters more than normalisation.** Because QQP is a graph,
   `StratifiedKFold` lets the same question appear in training and validation
   through different pairs. Both schemes are reported so the paper can quote
   the group-disjoint number as its headline and show the gap.

Run: python -m semcache.experiments.exp4_meta_classifier
"""
from __future__ import annotations

import argparse

import numpy as np

from .. import config as C
from ..data import build_splits, make_group_folds, make_stratified_folds
from ..metaclf import (MODEL_FACTORIES, cross_validate, leak_check_experiment,
                       tune_threshold)
from ..reporting import esc, pm, save_json, save_latex
from ._common import banner, encode_frame, hardware_manifest

BASIC = ("bi_cosine", "cross_score")
EXTENDED = ("bi_cosine", "cross_score", "distance", "elementwise", "text")


def table_meta(rows, caption: str, label: str) -> str:
    head = (f"\\begin{{table}}[htbp]\n\\centering\n\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n\\resizebox{{\\textwidth}}{{!}}{{\n"
            "\\begin{tabular}{lccccc}\n\\toprule\n"
            "Classifier & Accuracy & Precision & Recall & F1 & AUC\\\\\n"
            "\\midrule\n")
    body = ""
    for r in rows:
        body += (f"{esc(r['classifier'])} & {pm(r['accuracy_mean'], r['accuracy_std'])} & "
                 f"{pm(r['precision_mean'], r['precision_std'])} & "
                 f"{pm(r['recall_mean'], r['recall_std'])} & "
                 f"{pm(r['f1_mean'], r['f1_std'])} & "
                 f"{pm(r['auc_mean'], r['auc_std'])}\\\\\n")
    return head + body + "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"


def run_block(enc, folds, scheme: str, feature_sets: dict) -> dict:
    out = {}
    for fs_name, groups in feature_sets.items():
        X = enc.bundle.matrix(groups)
        rows = []
        for clf in MODEL_FACTORIES:
            r = cross_validate(X, enc.y, folds, clf, name=f"{clf}/{fs_name}",
                               fold_scheme=scheme)
            a = r.agg()
            a["classifier"] = clf
            a["feature_set"] = fs_name
            a["n_features"] = int(X.shape[1])
            rows.append(a)
            print(f"    {scheme:26s} {fs_name:8s} {clf:14s} "
                  f"F1={a['f1_mean']:.4f} +/- {a['f1_std']:.4f}  "
                  f"AUC={a['auc_mean']:.4f}")
        out[fs_name] = rows
    return out


def baseline_single_models(enc, folds) -> dict:
    """Per-fold threshold-tuned single-model baselines, for the improvement
    percentages the abstract quotes."""
    out = {}
    for name, score in (("bi_encoder_cosine", enc.bundle.blocks["bi_cosine"]),
                        ("cross_encoder_score", enc.bundle.blocks["cross_score"])):
        f1s, thrs, accs = [], [], []
        from sklearn.metrics import accuracy_score, f1_score
        for tr, va in folds:
            t, _ = tune_threshold(enc.y[tr], score[tr])
            pred = (score[va] >= t).astype(int)
            f1s.append(f1_score(enc.y[va], pred, zero_division=0))
            accs.append(accuracy_score(enc.y[va], pred))
            thrs.append(t)
        out[name] = {"f1_mean": float(np.mean(f1s)),
                     "f1_std": float(np.std(f1s, ddof=1)),
                     "accuracy_mean": float(np.mean(accs)),
                     "threshold_mean": float(np.mean(thrs)),
                     "threshold_std": float(np.std(thrs, ddof=1))}
    return out


def main(n: int | None = None) -> dict:
    banner("EXP 4 | Meta-classifier tables with fold-internal normalisation")
    splits = build_splits()
    df = splits.meta if n is None else splits.meta.iloc[:n]
    print(f"  {len(df)} balanced QQP pairs, {df.component.nunique()} components")

    enc = encode_frame(df, f"meta{len(df)}", with_cross=True)
    strat = make_stratified_folds(enc.y, C.CV_FOLDS_META)
    grp = make_group_folds(enc.y, enc.groups, C.CV_FOLDS_META)
    fsets = {"basic": BASIC, "extended": EXTENDED}

    print("\n  Cross-validating:")
    res_strat = run_block(enc, strat, "stratified_kfold", fsets)
    res_grp = run_block(enc, grp, "stratified_group_kfold", fsets)

    print("\n  Single-model baselines (per-fold tuned thresholds, group folds):")
    singles = baseline_single_models(enc, grp)
    for k, v in singles.items():
        print(f"    {k:22s} F1={v['f1_mean']:.4f} +/- {v['f1_std']:.4f} "
              f"(threshold {v['threshold_mean']:.3f})")

    print("\n  Normalisation protocol check on the extended feature set:")
    leak = leak_check_experiment(enc.bundle.matrix(EXTENDED), enc.y, grp)
    print(f"    scaler inside fold        F1={leak['scaler_inside_fold']['f1_mean']:.5f}")
    print(f"    scaler fitted before split F1={leak['scaler_fitted_before_split']['f1_mean']:.5f}")
    print(f"    delta = {leak['delta_f1']:+.5f}")

    best_grp = max(res_grp["extended"], key=lambda r: r["f1_mean"])
    best_strat = max(res_strat["extended"], key=lambda r: r["f1_mean"])
    print(f"\n  Best extended-feature model:")
    print(f"    stratified folds  {best_strat['classifier']:12s} F1={best_strat['f1_mean']:.4f}")
    print(f"    group folds       {best_grp['classifier']:12s} F1={best_grp['f1_mean']:.4f}")
    print(f"    leakage-driven inflation: {best_strat['f1_mean']-best_grp['f1_mean']:+.4f} F1")

    improvements = {
        "vs_best_bi_encoder_pct": 100.0 * (best_grp["f1_mean"] -
                                           singles["bi_encoder_cosine"]["f1_mean"]),
        "vs_best_cross_encoder_pct": 100.0 * (best_grp["f1_mean"] -
                                              singles["cross_encoder_score"]["f1_mean"]),
        "extended_vs_basic_pct": 100.0 * (
            best_grp["f1_mean"] -
            max(r["f1_mean"] for r in res_grp["basic"])),
        "note": ("Percentage-point differences in F1, computed under "
                 "group-disjoint folds. The manuscript quotes +4.2 / +5.2 / "
                 "+3.3; these are the leakage-controlled equivalents."),
    }
    print(f"\n  Improvement over baselines (group folds, percentage points):")
    for k, v in improvements.items():
        if isinstance(v, float):
            print(f"    {k:28s} {v:+.2f}")

    out = {
        "manifest": hardware_manifest({"n_pairs": len(df),
                                       "cv_folds": C.CV_FOLDS_META}),
        "stratified_kfold": res_strat,
        "stratified_group_kfold": res_grp,
        "single_model_baselines": singles,
        "normalisation_leak_check": leak,
        "leakage_inflation_f1": best_strat["f1_mean"] - best_grp["f1_mean"],
        "improvements": improvements,
        "normalisation_statement": (
            "For every result in this file, z-score normalisation is applied "
            "by a StandardScaler that is the first step of an sklearn "
            "Pipeline. The Pipeline is instantiated and fitted inside the "
            "fold loop on the training indices only, then used to transform "
            "the validation indices. Statistics from validation folds "
            "therefore never enter training."),
    }
    save_json(out, "exp4_meta_classifier")
    save_latex(table_meta(res_grp["basic"],
                          "Meta-classifier performance with basic features "
                          "(bi-encoder cosine + cross-encoder score). 5-fold "
                          "group-disjoint CV on 30,000 balanced QQP pairs; "
                          "normalisation fitted inside each training fold.",
                          "tab:meta_classifier_basic"),
               "table_meta_classifier_basic")
    save_latex(table_meta(res_grp["extended"],
                          "Meta-classifier performance with extended features. "
                          "5-fold group-disjoint CV on 30,000 balanced QQP "
                          "pairs; normalisation fitted inside each training "
                          "fold.",
                          "tab:meta_classifier_extended"),
               "table_meta_classifier_extended")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    main(a.n)
