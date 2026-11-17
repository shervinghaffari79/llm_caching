"""
Aggregate exp46 (audit-budgeted decision rules) under the pre-registered rules in docs/PREREGISTRATION_round2b.md, E1.

Rule (a): at budgets 1,000 and 10,000, precision of C2-in minus C1 at matched hit ratio, read from the threshold curves
          (C1 grid vs C2-in grid, linear interpolation over the overlapping hit-ratio range), one value per window,
          95% t-interval over windows. Favourable if the gain is >= 2 points with the interval above 0.
Rule (b): |realised precision - target| of C2-in vs C1-cal over (window, budget, draw, target) cells where both are
          feasible; paired difference, 95% bootstrap CI. Favourable if C2-in's absolute error is smaller with CI above 0.
Also reported: infeasible rates, realised precision and hit ratio per system, budget and target.
Windows still missing cells are reported as incomplete; nothing is imputed.

Usage: python scripts/aggregate_exp46.py <out.md> <partial_w0.json> [<partial_w1.json> ...]
"""
import json
import sys

import numpy as np
from scipy import stats

BUDGETS_RULE_A = (1000, 10000)


def interp_gap(c2, c1, n=30):
    a = sorted((r["hit_ratio_pct"], r["precision_of_hits_pct"]) for r in c1 if r.get("precision_of_hits_pct") is not None)
    b = sorted((r["hit_ratio_pct"], r["precision_of_hits_pct"]) for r in c2 if r.get("precision_of_hits_pct") is not None)
    if len(a) < 2 or len(b) < 2:
        return None
    lo, hi = max(a[0][0], b[0][0]), min(a[-1][0], b[-1][0])
    if lo >= hi:
        return None
    xs = np.linspace(lo, hi, n)
    ya = np.interp(xs, [p[0] for p in a], [p[1] for p in a])
    yb = np.interp(xs, [p[0] for p in b], [p[1] for p in b])
    return float((yb - ya).mean()), (float(lo), float(hi))


def main(out, files):
    rows_all, curves_all, wins = [], [], {}
    for f in files:
        d = json.load(open(f, encoding="utf8"))
        w = int(f.split("_w")[-1].split(".")[0].split("_")[0])
        wins[w] = {"cells_done": len(d["done"])}
        for r in d["rows"]:
            rows_all.append({**r, "window": w})
        for c in d["curves"]:
            curves_all.append({**c, "window": w})
    lines = ["# E1 aggregate (pre-registered rules)", "", f"Windows: {sorted(wins)}; cells done: " +
             ", ".join(f"w{w}={v['cells_done']}" for w, v in sorted(wins.items())), ""]
    # ---- rule (a)
    lines += ["## Rule (a): precision gain of C2-in over C1 at matched hit ratio", "",
              "| budget | window | mean gap (points) | overlap hit ratio (%) |", "|---|---|---|---|"]
    gaps = {b: [] for b in BUDGETS_RULE_A}
    for b in BUDGETS_RULE_A:
        for w in sorted(wins):
            c1 = [c for c in curves_all if c["window"] == w and c["system"] == "C1"]
            c2 = [c for c in curves_all if c["window"] == w and c["system"] == "C2-in" and c["budget"] == b]
            g = interp_gap(c2, c1) if c2 else None
            if g:
                gaps[b].append(g[0])
                lines.append(f"| {b} | {w} | {g[0]:+.2f} | {g[1][0]:.0f}-{g[1][1]:.0f} |")
            else:
                lines.append(f"| {b} | {w} | not available yet | |")
    lines += ["", "| budget | windows | mean gap | 95% t-interval | decision (>= 2 points, interval above 0) |", "|---|---|---|---|---|"]
    for b in BUDGETS_RULE_A:
        g = np.array(gaps[b])
        if len(g) >= 2:
            m, se = g.mean(), g.std(ddof=1) / np.sqrt(len(g))
            h = stats.t.ppf(0.975, len(g) - 1) * se
            ok = (m >= 2.0) and (m - h > 0)
            lines.append(f"| {b} | {len(g)} | {m:+.2f} | [{m - h:+.2f}, {m + h:+.2f}] | {'FAVOURABLE' if ok else 'not favourable'}"
                         f"{' (incomplete: fewer than 4 windows)' if len(g) < 4 else ''} |")
        else:
            lines.append(f"| {b} | {len(g)} | - | - | not enough windows |")
    # ---- rule (b) and descriptive table
    lines += ["", "## Rule (b) and realised precision at audit-chosen thresholds", "",
              "| target | budget | system | feasible/total | realised precision (%) | hit ratio (%) |", "|---|---|---|---|---|---|"]
    amend_flag = lambda r: bool(r.get("amendment"))
    for amend in (False, True):
        for p in sorted({r["p_star"] for r in rows_all if r["p_star"] is not None and amend_flag(r) == amend}):
            for b in sorted({r["budget"] for r in rows_all if r["budget"] is not None}):
                for s in ("C1-cal", "C2-in", "C2-qqp-cal"):
                    sel = [r for r in rows_all if r["system"] == s and r["p_star"] == p and r["budget"] == b and amend_flag(r) == amend]
                    if not sel:
                        continue
                    ok = [r for r in sel if not r.get("infeasible")]
                    prec = np.mean([r["precision_of_hits_pct"] for r in ok]) if ok else float("nan")
                    hit = np.mean([r["hit_ratio_pct"] for r in ok]) if ok else float("nan")
                    lines.append(f"| {p:.2f}{' (amendment)' if amend else ''} | {b} | {s} | {len(ok)}/{len(sel)} | {prec:.1f} | {hit:.1f} |")
    lines += ["", "### Paired absolute target error, C2-in vs C1-cal (cells where both are feasible)", "",
              "| target | cells | mean |err| C1-cal | mean |err| C2-in | difference (C1 - C2) | 95% bootstrap CI | decision |", "|---|---|---|---|---|---|---|"]
    rng = np.random.default_rng(0)
    for amend in (False, True):
        for p in sorted({r["p_star"] for r in rows_all if r["p_star"] is not None and amend_flag(r) == amend}):
            key = lambda r: (r["window"], r["budget"], r["draw"])
            a = {key(r): r for r in rows_all if r["system"] == "C1-cal" and r["p_star"] == p and amend_flag(r) == amend and not r.get("infeasible")}
            c = {key(r): r for r in rows_all if r["system"] == "C2-in" and r["p_star"] == p and amend_flag(r) == amend and not r.get("infeasible")}
            both = sorted(set(a) & set(c))
            if len(both) < 5:
                lines.append(f"| {p:.2f}{' (amendment)' if amend else ''} | {len(both)} | - | - | - | - | too few paired cells |")
                continue
            e1 = np.array([abs(a[k]["precision_of_hits_pct"] - 100 * p) for k in both])
            e2 = np.array([abs(c[k]["precision_of_hits_pct"] - 100 * p) for k in both])
            d = e1 - e2
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            lines.append(f"| {p:.2f}{' (amendment)' if amend else ''} | {len(both)} | {e1.mean():.2f} | {e2.mean():.2f} | {d.mean():+.2f} | "
                         f"[{lo:+.2f}, {hi:+.2f}] | {'FAVOURABLE' if lo > 0 else 'not favourable'} |")
    open(out, "w", encoding="utf8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
