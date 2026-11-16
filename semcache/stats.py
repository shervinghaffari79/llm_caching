"""
Uncertainty for the serving comparisons.

Two sources of variation are reported separately:

* across seeds   -- each seed is an independent stream (and LightGBM seed);
                    `seed_ci` gives a t-interval over the per-seed values.
* within a run   -- paired per-request outcomes of two systems on the same
                    stream; `paired_bootstrap` resamples requests, and
                    `mcnemar` tests disagreement on paired binary outcomes.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st


def seed_ci(values, conf: float = 0.95) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    n = len(v)
    if n == 0:
        return {"n": 0}
    mean = float(v.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "sd": 0.0, "ci_low": mean, "ci_high": mean}
    sd = float(v.std(ddof=1))
    half = float(st.t.ppf(0.5 + conf / 2, n - 1) * sd / np.sqrt(n))
    return {"n": n, "mean": mean, "sd": sd, "ci_low": mean - half,
            "ci_high": mean + half}


def paired_seed_test(a, b) -> dict:
    """Seed-level paired comparison of two systems (same seeds, same order)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out = {"n": int(len(d)), **{f"delta_{k}": v for k, v in seed_ci(d).items()
                                 if k != "n"}}
    if len(d) >= 2 and np.any(d != 0):
        out["t_p"] = float(st.ttest_rel(a, b).pvalue)
        if len(d) >= 5:
            out["wilcoxon_p"] = float(st.wilcoxon(a, b).pvalue)
    return out


def paired_bootstrap(a, b, n_boot: int = 10_000, seed: int = 0,
                     stat=np.mean) -> dict:
    """Bootstrap over paired units (requests) of stat(a) - stat(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    deltas = stat(a[idx], axis=1) - stat(b[idx], axis=1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    obs = float(stat(a) - stat(b))
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {"delta": obs, "ci_low": float(lo), "ci_high": float(hi),
            "p_boot": float(min(1.0, p)),
            "ci_includes_zero": bool(lo <= 0 <= hi)}


def mcnemar(a, b) -> dict:
    """Exact McNemar test on paired binary outcomes (e.g. served-correctly)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01 = int(np.sum(~a & b))
    n10 = int(np.sum(a & ~b))
    n = n01 + n10
    p = float(st.binomtest(min(n01, n10), n, 0.5).pvalue) if n else 1.0
    return {"a_only": n10, "b_only": n01, "p": p}


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjustment for one family of tests.

    `pvals` maps a test name to its p-value (None entries are skipped);
    returns the adjusted p-values under the same names."""
    items = sorted(((p, k) for k, p in pvals.items() if p is not None), key=lambda t: t[0])
    m, out, running = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
