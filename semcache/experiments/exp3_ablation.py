"""
Experiment 3 -- the feature-group ablation, recomputed on the subset it claims
to be computed on.

The published Table 7 reports Full Model F1 = 0.8716 +/- 0.004 and
Full - Cross-Encoder = 0.8709 +/- 0.004 for a 5-fold CV on a 3,000-pair,
60/40-imbalanced, held-out sample -- identical to three decimals, standard
deviation included, to the Table 4 result from the 30,000-pair balanced
training set. Two different samples of different size and different class
balance do not produce matching means *and* matching standard deviations.

This script recomputes the ablation from scratch on the actual ablation
subset and, in the same run, recomputes the 30k figure, so the two can be
printed next to each other. Whatever the outcome, the difference between them
is then a measured quantity.

Every configuration is fitted with an sklearn Pipeline whose StandardScaler is
fitted inside the fold, so no validation statistics reach the training data.

Run: python -m semcache.experiments.exp3_ablation
"""
from __future__ import annotations

import argparse

import numpy as np

from .. import config as C
from ..data import build_splits, make_group_folds, make_stratified_folds
from ..features import GROUP_SIZES
from ..metaclf import (bootstrap_f1_delta, cross_validate, group_importance,
                       fit_final, leak_check_experiment, paired_fold_test)
from ..reporting import (save_json, save_latex, table_ablation_incremental,
                         table_ablation_isolation)
from ._common import banner, encode_frame, hardware_manifest

GROUP_LABELS = {
    "bi_cosine": "Bi-Encoder Cosine",
    "cross_score": "Cross-Encoder Score",
    "distance": "Distance Metrics (L2, L1)",
    "elementwise": "Element-wise Operations (product, |diff|)",
    "text": "Text Features",
}

INCREMENTAL = [
    ("Bi + Cross", ("bi_cosine", "cross_score")),
    ("+ Distance Metrics", ("bi_cosine", "cross_score", "distance")),
    ("+ Element-wise Ops", ("bi_cosine", "cross_score", "elementwise")),
    ("+ Text Features", ("bi_cosine", "cross_score", "text")),
    ("Full Model", ("bi_cosine", "cross_score", "distance", "elementwise", "text")),
    ("Full Model - Cross-Encoder", ("bi_cosine", "distance", "elementwise", "text")),
    ("Bi-Encoder Cosine only", ("bi_cosine",)),
]


def _latency_lookup() -> dict:
    """Marginal per-request cost of each group, from exp1 if it has run."""
    try:
        from ..reporting import load_json
        d = load_json("exp1_latency_breakdown")
        s = d["online_stages_ms"]
        return {
            "bi_cosine": s.get("feat_bi_cosine", {}).get("mean_ms", 0.0),
            "cross_score": s.get("cross_encode", {}).get("mean_ms", None),
            "distance": s.get("feat_distance", {}).get("mean_ms", 0.0),
            "elementwise": s.get("feat_elementwise", {}).get("mean_ms", 0.0),
            "text": s.get("feat_text", {}).get("mean_ms", 0.0),
            "meta_infer": s.get("meta_infer", {}).get("mean_ms", 0.0),
            "bi_encode": s.get("bi_encode", {}).get("mean_ms", 0.0),
            "vector_search": s.get("vector_search", {}).get("mean_ms", 0.0),
        }
    except Exception:
        return {}


def run_suite(enc, folds, scheme: str, model: str = "LightGBM") -> dict:
    """Isolation + incremental ablation under one fold scheme."""
    isolation, incremental, cv_objects = [], [], {}
    lat = _latency_lookup()

    for g in ("bi_cosine", "cross_score", "distance", "elementwise", "text"):
        X = enc.bundle.matrix((g,))
        r = cross_validate(X, enc.y, folds, model, name=g, fold_scheme=scheme)
        a = r.agg()
        isolation.append({
            "group": g, "group_label": GROUP_LABELS[g],
            "n_features": GROUP_SIZES[g],
            "f1_mean": a["f1_mean"], "f1_std": a["f1_std"],
            "auc_mean": a["auc_mean"], "auc_std": a["auc_std"],
            "accuracy_mean": a["accuracy_mean"],
            "marginal_cost_ms": lat.get(g),
        })

    for label, groups in INCREMENTAL:
        X = enc.bundle.matrix(groups)
        r = cross_validate(X, enc.y, folds, model, name=label, fold_scheme=scheme)
        cv_objects[label] = r
        a = r.agg()
        over = None
        if lat:
            over = (lat.get("bi_encode", 0.0) + lat.get("vector_search", 0.0)
                    + sum(lat.get(g, 0.0) or 0.0 for g in groups)
                    + lat.get("meta_infer", 0.0))
        incremental.append({
            "label": label, "groups": list(groups),
            "n_features": int(X.shape[1]),
            "f1_mean": a["f1_mean"], "f1_std": a["f1_std"],
            "auc_mean": a["auc_mean"], "auc_std": a["auc_std"],
            "accuracy_mean": a["accuracy_mean"],
            "precision_mean": a["precision_mean"], "recall_mean": a["recall_mean"],
            "overhead_ms": over,
        })

    base = next(r for r in incremental if r["label"] == "Bi + Cross")
    for r in incremental:
        r["delta_f1"] = r["f1_mean"] - base["f1_mean"]

    # The decisive test: does removing the cross-encoder change anything?
    full = cv_objects["Full Model"]
    nox = cv_objects["Full Model - Cross-Encoder"]
    cross_test = paired_fold_test(full, nox)
    cross_test["interpretation"] = (
        "Positive mean_delta_f1 means the cross-encoder helps. If the "
        "bootstrap 95% CI on the out-of-fold F1 difference contains zero, the "
        "cross-encoder's contribution is not distinguishable from noise on "
        "this data, and its ~500 ms cost buys nothing measurable.")

    return {"isolation": isolation, "incremental": incremental,
            "cross_encoder_removal_test": cross_test,
            "fold_scheme": scheme,
            "_cv": cv_objects}


def main(model: str = "LightGBM") -> dict:
    banner("EXP 3 | Feature-group ablation on the held-out 3,000-pair subset")
    splits = build_splits()
    abl = splits.ablation
    print(f"  ablation subset: {len(abl)} pairs, "
          f"{abl.is_duplicate.mean()*100:.0f}% duplicates, "
          f"{abl.component.nunique()} components, "
          f"0 shared with the 30k meta set "
          f"(verified: {splits.provenance['component_overlap_meta_ablation']} "
          f"component overlaps)")

    enc = encode_frame(abl, "ablation3000", with_cross=True)

    strat = make_stratified_folds(enc.y, C.CV_FOLDS_META)
    grp = make_group_folds(enc.y, enc.groups, C.CV_FOLDS_META)

    res_strat = run_suite(enc, strat, "stratified_kfold", model)
    res_grp = run_suite(enc, grp, "stratified_group_kfold", model)

    print("\n  Feature groups in isolation (group-disjoint folds):")
    for r in res_grp["isolation"]:
        print(f"    {r['group_label']:44s} n={r['n_features']:5d} "
              f"F1={r['f1_mean']:.4f} +/- {r['f1_std']:.4f}")

    print("\n  Incremental (group-disjoint folds):")
    for r in res_grp["incremental"]:
        print(f"    {r['label']:32s} F1={r['f1_mean']:.4f} +/- {r['f1_std']:.4f} "
              f"(dF1 {r['delta_f1']:+.4f})")

    ct = res_grp["cross_encoder_removal_test"]
    print(f"\n  Cross-encoder removal: dF1 = {ct['mean_delta_f1']:+.4f}, "
          f"bootstrap 95% CI [{ct.get('bootstrap_ci95_low', float('nan')):+.4f}, "
          f"{ct.get('bootstrap_ci95_high', float('nan')):+.4f}], "
          f"paired t p={ct.get('paired_t_p', float('nan')):.3f}")

    # ---- balanced re-check ---------------------------------------------
    print("\n  Balanced 1500/1500 re-check on a disjoint sample of the same pool")
    enc_bal = encode_frame(splits.ablation_bal, "ablationbal3000", with_cross=True)
    grp_bal = make_group_folds(enc_bal.y, enc_bal.groups, C.CV_FOLDS_META)
    res_bal = run_suite(enc_bal, grp_bal, "stratified_group_kfold_balanced", model)
    for r in res_bal["isolation"]:
        print(f"    {r['group_label']:44s} F1={r['f1_mean']:.4f}")

    # ---- the 30k reference, recomputed ---------------------------------
    print("\n  Recomputing the 30k meta-classifier reference for comparison")
    enc30 = encode_frame(splits.meta, "meta30000", with_cross=True)
    folds30 = make_group_folds(enc30.y, enc30.groups, C.CV_FOLDS_META)
    X30 = enc30.bundle.matrix(("bi_cosine", "cross_score", "distance",
                               "elementwise", "text"))
    r30 = cross_validate(X30, enc30.y, folds30, model, name="30k_full",
                         fold_scheme="stratified_group_kfold").agg()
    full_abl = next(r for r in res_grp["incremental"] if r["label"] == "Full Model")
    print(f"    30k balanced   F1 = {r30['f1_mean']:.4f} +/- {r30['f1_std']:.4f}")
    print(f"    3k  60/40      F1 = {full_abl['f1_mean']:.4f} +/- {full_abl['f1_std']:.4f}")
    print(f"    difference     {full_abl['f1_mean'] - r30['f1_mean']:+.4f}")

    # ---- normalisation-leak check --------------------------------------
    print("\n  Normalisation protocol check (scaler inside fold vs before split)")
    leak = leak_check_experiment(enc.bundle.matrix(
        ("bi_cosine", "cross_score", "distance", "elementwise", "text")),
        enc.y, grp, model)
    print(f"    dF1 = {leak['delta_f1']:+.5f}, dAUC = {leak['delta_auc']:+.5f}")

    # ---- group importances ---------------------------------------------
    Xfull = enc.bundle.matrix(("bi_cosine", "cross_score", "distance",
                               "elementwise", "text"))
    names = enc.bundle.names(("bi_cosine", "cross_score", "distance",
                              "elementwise", "text"))
    trained = fit_final(Xfull, enc.y, ("bi_cosine", "cross_score", "distance",
                                       "elementwise", "text"), model)
    gimp = group_importance(trained, names)
    print("\n  Meta-classifier split-gain importance by group:")
    for g, v in gimp.items():
        print(f"    {g:14s} {v['importance_pct']:6.2f}%")

    for d in (res_strat, res_grp, res_bal):
        d.pop("_cv", None)

    out = {
        "manifest": hardware_manifest({"model": model,
                                       "subset": "held-out 3,000 QQP pairs (60/40)",
                                       "cv_folds": C.CV_FOLDS_META}),
        "subset_description": {
            "n_pairs": int(len(abl)),
            "duplicate_ratio": float(abl.is_duplicate.mean()),
            "n_components": int(abl.component.nunique()),
            "disjoint_from_meta_set": splits.provenance,
        },
        "stratified_kfold": res_strat,
        "stratified_group_kfold": res_grp,
        "balanced_recheck": res_bal,
        "reference_30k_recomputed": r30,
        "delta_ablation_vs_30k_f1": full_abl["f1_mean"] - r30["f1_mean"],
        "normalisation_leak_check": leak,
        "group_importance": gimp,
        "published_table7_for_comparison": {
            "full_model_f1": "0.8716 +/- 0.004",
            "full_minus_cross_f1": "0.8709 +/- 0.004",
            "note": ("Identical to the Table 4 30k result including the "
                     "standard deviation. The values recomputed here on the "
                     "actual 3k 60/40 subset are what Table 7 should report."),
        },
    }
    save_json(out, "exp3_ablation")
    save_latex(table_ablation_isolation(res_grp["isolation"]),
               "table_ablation_isolation")
    save_latex(table_ablation_incremental(res_grp["incremental"]),
               "table_ablation_incremental")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LightGBM",
                    choices=["LightGBM", "XGBoost", "RandomForest"])
    a = ap.parse_args()
    main(a.model)
