"""
Meta-classifier training and cross-validation with **fold-internal**
normalisation.

This module exists mainly to make one guarantee auditable:

    mu and sigma for z-scoring are estimated on the training portion of each
    fold ONLY, and then applied to that fold's validation portion.

The manuscript's wording ("computed from training data and applied
consistently across all sets") is ambiguous about whether the statistics were
fitted before or after splitting. Fitting before splitting leaks the
validation distribution into training. Here the scaler is a step *inside* an
sklearn `Pipeline`, and the pipeline is fitted inside the fold loop, so the
leak is structurally impossible rather than merely intended to be absent.

`leak_check_experiment()` quantifies what the difference is worth on this
data, so the paper can report the magnitude instead of just claiming
correctness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C


# --------------------------------------------------------------------------
# Model factories
# --------------------------------------------------------------------------
def make_lightgbm(seed: int = C.SEED, **kw):
    import lightgbm as lgb
    params = dict(n_estimators=400, learning_rate=0.05, num_leaves=63,
                  min_child_samples=20, subsample=0.9, subsample_freq=1,
                  colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
                  random_state=seed, n_jobs=-1, verbose=-1)
    params.update(kw)
    return lgb.LGBMClassifier(**params)


def make_xgboost(seed: int = C.SEED, **kw):
    import xgboost as xgb
    params = dict(n_estimators=400, learning_rate=0.05, max_depth=6,
                  subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=seed, n_jobs=-1, tree_method="hist",
                  eval_metric="logloss")
    params.update(kw)
    return xgb.XGBClassifier(**params)


def make_random_forest(seed: int = C.SEED, **kw):
    params = dict(n_estimators=400, min_samples_leaf=2, n_jobs=-1,
                  random_state=seed, oob_score=True, bootstrap=True)
    params.update(kw)
    return RandomForestClassifier(**params)


MODEL_FACTORIES: dict[str, Callable[..., object]] = {
    "LightGBM": make_lightgbm,
    "XGBoost": make_xgboost,
    "RandomForest": make_random_forest,
}


def make_pipeline(model_name: str, seed: int = C.SEED, scale: bool = True):
    """Scaler + classifier as one estimator.

    Tree ensembles are scale-invariant, so the scaler does not change their
    predictions -- it is kept because the manuscript specifies z-scoring, and
    keeping it inside the pipeline is what makes the no-leakage claim true for
    every classifier including the scale-sensitive ones a reader might swap in.
    """
    est = MODEL_FACTORIES[model_name](seed=seed)
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", est))
    return Pipeline(steps)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                           threshold: float = C.META_PROB_THRESHOLD) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "threshold": threshold,
    }


def tune_threshold(y_true: np.ndarray, score: np.ndarray,
                   n_grid: int = 101) -> tuple[float, float]:
    """Grid-search the F1-maximising threshold, as the manuscript describes.

    Called only on a *validation* fold, and the returned threshold is applied
    only to that same fold -- matching the paper's per-fold protocol.
    """
    lo, hi = float(np.min(score)), float(np.max(score))
    grid = np.linspace(lo, hi, n_grid) if hi > lo else np.array([lo])
    best_t, best_f1 = float(grid[0]), -1.0
    for t in grid:
        f1 = f1_score(y_true, (score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t, best_f1


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------
@dataclass
class CVResult:
    name: str
    folds: list[dict] = field(default_factory=list)
    oof_prob: np.ndarray | None = None
    oof_true: np.ndarray | None = None
    n_features: int = 0
    scaler_fitted_inside_fold: bool = True
    fold_scheme: str = "stratified"
    extra: dict = field(default_factory=dict)

    def agg(self) -> dict:
        keys = [k for k in self.folds[0] if isinstance(self.folds[0][k], (int, float))]
        out = {"name": self.name, "n_features": self.n_features,
               "n_folds": len(self.folds), "fold_scheme": self.fold_scheme,
               "scaler_fitted_inside_fold": self.scaler_fitted_inside_fold}
        for k in keys:
            vals = np.array([f[k] for f in self.folds], dtype=float)
            out[f"{k}_mean"] = float(np.nanmean(vals))
            out[f"{k}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.update(self.extra)
        return out

    def per_fold_f1(self) -> np.ndarray:
        return np.array([f["f1"] for f in self.folds], dtype=float)


def cross_validate(X: np.ndarray, y: np.ndarray, folds: Sequence[tuple],
                   model_name: str = "LightGBM", name: str = "",
                   seed: int = C.SEED, scale: bool = True,
                   fold_scheme: str = "stratified",
                   leak_scaler: bool = False) -> CVResult:
    """Run CV, fitting the scaler inside each fold.

    `leak_scaler=True` deliberately reproduces the *wrong* protocol (fit the
    scaler on all rows before splitting) so the two can be compared. It is
    used only by `leak_check_experiment`.
    """
    res = CVResult(name=name or model_name, n_features=X.shape[1],
                   fold_scheme=fold_scheme,
                   scaler_fitted_inside_fold=not leak_scaler)
    oof_prob = np.full(len(y), np.nan, dtype=float)

    X_pre = X
    if leak_scaler and scale:
        X_pre = StandardScaler().fit_transform(X)      # <-- the leak

    for i, (tr, va) in enumerate(folds):
        pipe = make_pipeline(model_name, seed=seed + i,
                             scale=scale and not leak_scaler)
        pipe.fit(X_pre[tr], y[tr])
        prob = pipe.predict_proba(X_pre[va])[:, 1]
        oof_prob[va] = prob
        m = classification_metrics(y[va], prob)
        m["fold"] = i
        m["n_train"], m["n_val"] = len(tr), len(va)
        res.folds.append(m)

    res.oof_prob, res.oof_true = oof_prob, y
    return res


def leak_check_experiment(X: np.ndarray, y: np.ndarray, folds: Sequence[tuple],
                          model_name: str = "LightGBM") -> dict:
    """Quantify the normalisation-leak question raised in review."""
    clean = cross_validate(X, y, folds, model_name, name="scaler_in_fold",
                           leak_scaler=False)
    leaked = cross_validate(X, y, folds, model_name, name="scaler_on_all_data",
                            leak_scaler=True)
    a, b = clean.agg(), leaked.agg()
    return {
        "scaler_inside_fold": a,
        "scaler_fitted_before_split": b,
        "delta_f1": b["f1_mean"] - a["f1_mean"],
        "delta_auc": b["auc_mean"] - a["auc_mean"],
        "note": ("Tree ensembles are invariant to monotone per-feature affine "
                 "transforms, so the measured delta is expected to be ~0. The "
                 "protocol still matters for scale-sensitive meta-classifiers "
                 "and is enforced structurally via sklearn Pipeline."),
    }


# --------------------------------------------------------------------------
# Paired significance testing between two CV configurations
# --------------------------------------------------------------------------
def paired_fold_test(a: CVResult, b: CVResult) -> dict:
    """Paired t-test + Wilcoxon over matched folds, plus a bootstrap CI on the
    out-of-fold F1 difference.

    Used to answer "is the cross-encoder's contribution distinguishable from
    zero?" with a statistic rather than an eyeballed third decimal place.
    """
    from scipy import stats
    fa, fb = a.per_fold_f1(), b.per_fold_f1()
    d = fa - fb
    out = {"mean_delta_f1": float(d.mean()),
           "std_delta_f1": float(d.std(ddof=1)) if len(d) > 1 else 0.0}
    if len(d) > 1 and np.any(d != 0):
        t, p = stats.ttest_rel(fa, fb)
        out["paired_t"], out["paired_t_p"] = float(t), float(p)
        try:
            w, pw = stats.wilcoxon(fa, fb)
            out["wilcoxon_p"] = float(pw)
        except ValueError:
            out["wilcoxon_p"] = float("nan")
    else:
        out["paired_t"] = out["paired_t_p"] = out["wilcoxon_p"] = float("nan")

    if a.oof_prob is not None and b.oof_prob is not None:
        out.update(bootstrap_f1_delta(a.oof_true, a.oof_prob, b.oof_prob))
    return out


def bootstrap_f1_delta(y: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray,
                       n_boot: int = 2000, seed: int = C.SEED,
                       threshold: float = C.META_PROB_THRESHOLD) -> dict:
    rng = np.random.default_rng(seed)
    ok = ~(np.isnan(prob_a) | np.isnan(prob_b))
    y, pa, pb = y[ok], prob_a[ok], prob_b[ok]
    ya, yb = (pa >= threshold).astype(int), (pb >= threshold).astype(int)
    deltas = np.empty(n_boot)
    n = len(y)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = (f1_score(y[idx], ya[idx], zero_division=0) -
                     f1_score(y[idx], yb[idx], zero_division=0))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "oof_f1_a": float(f1_score(y, ya, zero_division=0)),
        "oof_f1_b": float(f1_score(y, yb, zero_division=0)),
        "bootstrap_delta_mean": float(deltas.mean()),
        "bootstrap_ci95_low": float(lo),
        "bootstrap_ci95_high": float(hi),
        "bootstrap_ci_includes_zero": bool(lo <= 0.0 <= hi),
    }


# --------------------------------------------------------------------------
# Final (non-CV) fit used to power the live serving pipeline
# --------------------------------------------------------------------------
@dataclass
class TrainedMeta:
    pipeline: object
    groups: tuple[str, ...]
    n_features: int
    train_rows: int
    threshold: float = C.META_PROB_THRESHOLD

    def predict_proba_one(self, x: np.ndarray) -> float:
        return float(self.pipeline.predict_proba(x)[0, 1])


def fit_final(X: np.ndarray, y: np.ndarray, groups: tuple[str, ...],
              model_name: str = "LightGBM", seed: int = C.SEED) -> TrainedMeta:
    pipe = make_pipeline(model_name, seed=seed)
    pipe.fit(X, y)
    return TrainedMeta(pipeline=pipe, groups=tuple(groups),
                       n_features=X.shape[1], train_rows=len(y))


def feature_importance(trained: TrainedMeta, names: Sequence[str],
                       top_k: int = 40) -> list[dict]:
    clf = trained.pipeline.named_steps["clf"]
    imp = getattr(clf, "feature_importances_", None)
    if imp is None:
        return []
    order = np.argsort(imp)[::-1][:top_k]
    total = float(imp.sum()) or 1.0
    return [{"feature": names[i] if i < len(names) else f"f{i}",
             "importance": float(imp[i]),
             "importance_pct": 100.0 * float(imp[i]) / total} for i in order]


def group_importance(trained: TrainedMeta, names: Sequence[str]) -> dict:
    """Aggregate split-gain importance up to the feature-group level."""
    clf = trained.pipeline.named_steps["clf"]
    imp = getattr(clf, "feature_importances_", None)
    if imp is None:
        return {}

    def group_of(n: str) -> str:
        if n.startswith(("prod_", "absdiff_")):
            return "elementwise"
        if n in ("euclidean", "manhattan"):
            return "distance"
        if n in ("len_chars_1", "len_chars_2", "len_ratio_chars",
                 "len_ratio_words", "jaccard_words", "edit_ratio"):
            return "text"
        return n

    agg: dict[str, float] = {}
    for n, v in zip(names, imp):
        agg[group_of(n)] = agg.get(group_of(n), 0.0) + float(v)
    total = sum(agg.values()) or 1.0
    return {k: {"importance": v, "importance_pct": 100.0 * v / total}
            for k, v in sorted(agg.items(), key=lambda kv: -kv[1])}
