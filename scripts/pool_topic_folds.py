"""Pool the per-fold paired tests of exp43 (topic-held-out 3-fold) over folds x windows.

Each fold file stores n, mean and SD of the per-window differences, so the pooled mean, SD and
t-test over all windows follow exactly from the sum-of-squares decomposition. Holm correction is
applied over the five false-hit prices within each variant.
Usage: python scripts/pool_topic_folds.py <out.json> <fold0.json> <fold1.json> <fold2.json>
"""
import json, sys
import numpy as np
from scipy import stats

import os
LAMS, BASE, REF = (0, 1, 2, 5, 10), "WTinyLFU", int(os.environ.get("POOL_REF", 25000))


def pooled(files, key):
    ts = [json.load(open(f))["tests"][key] for f in files]
    n = np.array([t["n"] for t in ts]); m = np.array([t["delta_mean"] for t in ts]); sd = np.array([t["delta_sd"] for t in ts])
    N = n.sum(); M = (n * m).sum() / N
    var = (((n - 1) * sd ** 2).sum() + (n * (m - M) ** 2).sum()) / (N - 1)
    se = np.sqrt(var / N); p = 2 * stats.t.sf(abs(M / se), N - 1); h = stats.t.ppf(0.975, N - 1) * se
    return {"n_windows": int(N), "mean": float(M), "ci_low": float(M - h), "ci_high": float(M + h), "p": float(p)}


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def main(out, files):
    res = {}
    for variant, name in (("match", "ReuseGate"), ("group", "ReuseGate-R")):
        rows = [pooled(files, f"R{REF}|lambda{l}|Learned[{variant}]-{BASE}") for l in LAMS]
        for r, h in zip(rows, holm([r["p"] for r in rows])): r["holm_p"] = float(h)
        res[name] = {f"lambda{l}": r for l, r in zip(LAMS, rows)}
    json.dump({"comparison": f"vs {BASE}, reference {REF}", "folds": list(files), "results": res}, open(out, "w"), indent=1)
    for name, d in res.items():
        print(name, " | ".join(f"l{k[6:]}: {v['mean']:+.2f} (Holm {v['holm_p']:.3f})" for k, v in d.items()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
