"""Tables for the extra-baseline comparison (round 2c). Usage:
python scripts/tables_extra_baselines.py out.md "<label>=<file1>[,<file2>...]" ...
Windows of the files of one label are pooled (e.g. three LmArena rotations). Score = correct-hit % - lambda x false-hit %,
taken per evaluation window at each policy's tuned configuration (mean over cache capacities). Delta = ReuseGate minus policy,
t-interval and paired t-test over windows; Holm over the baselines inside each (variant, lambda) family. The portfolio bound
is the per-window maximum over GDSF, S3-FIFO, SIEVE, DynamicAdaptiveClimb and W-TinyLFU (SCION's portfolio members available here),
i.e. what a perfect per-window selector among them could reach; it is an upper bound, not AUTO."""
import json
import sys

import numpy as np
from scipy import stats

PORTFOLIO = ("GDSF", "S3FIFO", "SIEVE", "DynamicAdaptiveClimb", "WTinyLFU")
LAMS = ("0", "1", "2", "5", "10")


def load(files):
    pw = {}
    for f in files:
        d = json.load(open(f))
        R = max(d["per_window_scores"], key=int)
        for lam, fams in d["per_window_scores"][R].items():
            for fam, v in fams.items():
                pw.setdefault(lam, {}).setdefault(fam, []).extend(v)
    return pw


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def ttest(a, b):
    d = np.array(a) - np.array(b)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return d.mean(), (d.mean(), d.mean()), 1.0 if d.mean() == 0 else 0.0
    se = d.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return d.mean(), (d.mean() - h, d.mean() + h), float(2 * stats.t.sf(abs(d.mean() / se), n - 1))


out = ["# Round 2c: ReuseGate against further replacement and admission policies", "",
       "Definitions are in `scripts/tables_extra_baselines.py`. All baselines are reimplemented from their published descriptions; "
       "no third-party code was run. SOLAR's settings the paper does not give were tuned on the tuning windows only (grid in "
       "`semcache/policies_extra.py`). DynamicAdaptiveClimb runs with its resizing disabled so that every policy has the same capacity budget. "
       "Score = correct-hit % minus lambda x false-hit %, mean over cache capacities, per evaluation window.", ""]
for spec in sys.argv[2:]:
    label, files = spec.split("=", 1)
    pw = load(files.split(","))
    n = len(pw["0"]["LRU"])
    base = [f for f in pw["0"] if not f.startswith("Learned")]
    out += [f"## {label} ({n} evaluation windows)", "", "### Mean score per policy", "",
            "| policy | " + " | ".join(f"λ={l}" for l in LAMS) + " |", "|---|" + "---|" * len(LAMS)]
    for f in ["Learned[match]", "Learned[group]"] + base:
        out.append(f"| {f} | " + " | ".join(f"{np.mean(pw[l][f]):.2f}" for l in LAMS) + " |")
    port = {l: np.max([pw[l][f] for f in PORTFOLIO], axis=0) for l in LAMS}
    allb = {l: np.max([pw[l][f] for f in base], axis=0) for l in LAMS}
    out.append("| *best-in-hindsight over GDSF, S3-FIFO, SIEVE, DynamicAdaptiveClimb, W-TinyLFU (bound)* | " + " | ".join(f"{np.mean(port[l]):.2f}" for l in LAMS) + " |")
    out.append("| *best-in-hindsight over all baselines (bound)* | " + " | ".join(f"{np.mean(allb[l]):.2f}" for l in LAMS) + " |")
    for var in ("Learned[match]", "Learned[group]"):
        name = "ReuseGate (label-free)" if "match" in var else "ReuseGate-R (uses earlier group labels)"
        out += ["", f"### {name} minus each policy: delta [95% CI] (Holm-adjusted p within the lambda column)", "",
                "| policy | " + " | ".join(f"λ={l}" for l in LAMS) + " |", "|---|" + "---|" * len(LAMS)]
        cell = {}
        for l in LAMS:
            rows = [(b, *ttest(pw[l][var], pw[l][b])) for b in base]
            adj = holm([r[3] for r in rows])
            for (b, m, ci, p), a in zip(rows, adj):
                cell[(b, l)] = f"{m:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] p={a:.3f}"
            for key, arr in (("PORTFOLIO bound", port[l]), ("ALL-baselines bound", allb[l])):
                m, ci, p = ttest(pw[l][var], arr)
                cell[(key, l)] = f"{m:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] p={p:.3f}"
        for b in base + ["PORTFOLIO bound", "ALL-baselines bound"]:
            out.append(f"| {b} | " + " | ".join(cell[(b, l)] for l in LAMS) + " |")
    out.append("")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
print("\n".join(out[:60]))
