"""
Experiment 27 -- operating curves across seeds (aggregates experiment 12).

Experiment 12, run per seed with `--configs-only --tag ext`, sweeps the
decision threshold of C1-C4 over an extended grid. This script combines the
five seeds:

* precision and correct-only saving of each configuration at matched
  served-hit ratios (interpolated along each seed's curve), with seed-level
  paired tests against C1 and Holm adjustment across all such tests;
* the best risk-adjusted saving per false-hit cost lambda when each
  configuration's threshold is chosen for that lambda, paired tests against
  C1, and whether the optimum lies on the edge of the swept grid;
* the gain from choosing the threshold for lambda over the default
  threshold used in experiment 11 (same streams).

Run: python -m semcache.experiments.exp27_operating_curves_seeds --pattern "exp12_threshold_risk_*_seed{seed}_ext.json"
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np

from ..stats import holm, paired_seed_test, seed_ci
from ._common import REVISION_DIR, hardware_manifest, save_result

CFG = ("C1_bi_only", "C2_bi_geo_lgb", "C3_bi_cross", "C4_full_stacking")
SEEDS = (42, 43, 44, 45, 46)
HIT_TARGETS = (40.0, 45.0, 50.0, 55.0)
LAMBDAS = ("0", "1", "2", "5", "10")


def interp(curve: list[dict], x_key: str, y_key: str, x: float):
    pts = sorted((p[x_key], p[y_key]) for p in curve if p.get(x_key) is not None and p.get(y_key) is not None)
    xs = [a for a, _ in pts]
    if not xs or x < min(xs) or x > max(xs):
        return None
    return float(np.interp(x, xs, [b for _, b in pts]))


def main(pattern: str, exp11: str, seeds=SEEDS, tag: str = "") -> dict:
    files = {s: glob.glob(str(REVISION_DIR / pattern.format(seed=s))) for s in seeds}
    missing = [s for s, f in files.items() if not f]
    if missing:
        raise SystemExit(f"missing exp12 results for seeds {missing}")
    runs = {s: json.load(open(f[0], encoding="utf8")) for s, f in files.items()}
    grid_top = {k: max(p["threshold"] for p in runs[seeds[0]]["curves"][k]) for k in CFG}

    # ---- matched served-hit ratio ------------------------------------------
    matched, tests = {}, {}
    for h in HIT_TARGETS:
        H = f"{h:g}"
        matched[H] = {}
        for k in CFG:
            prec = [interp(runs[s]["curves"][k], "hit_ratio_pct", "precision_of_hits_pct", h) for s in seeds]
            sav = [interp(runs[s]["curves"][k], "hit_ratio_pct", "token_saving_correct_only_pct", h) for s in seeds]
            matched[H][k] = {"precision_of_hits_pct": seed_ci([v for v in prec if v is not None]),
                             "token_saving_correct_only_pct": seed_ci([v for v in sav if v is not None]),
                             "n_seeds_in_range": int(sum(v is not None for v in prec)),
                             "per_seed_precision": prec}
        for k in CFG[1:]:
            a = matched[H][k]["per_seed_precision"]
            b = matched[H]["C1_bi_only"]["per_seed_precision"]
            pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
            if len(pairs) >= 3:
                tests[f"precision|{H}|{k}-C1"] = paired_seed_test([x for x, _ in pairs], [y for _, y in pairs])

    # ---- best risk-adjusted saving per lambda --------------------------------
    best = {}
    for lam in LAMBDAS:
        best[lam] = {}
        for k in CFG:
            vals, thr, edge = [], [], 0
            for s in seeds:
                curve = runs[s]["curves"][k]
                b = max(curve, key=lambda p: p["risk_adjusted_saving_pct"][lam])
                vals.append(b["risk_adjusted_saving_pct"][lam])
                thr.append(b["threshold"])
                edge += int(b["threshold"] >= grid_top[k])
            best[lam][k] = {"risk_adjusted_saving_pct": seed_ci(vals), "per_seed": vals,
                            "thresholds": thr, "seeds_at_grid_edge": edge}
        for k in CFG[1:]:
            tests[f"best_ras|{lam}|{k}-C1"] = paired_seed_test(best[lam][k]["per_seed"],
                                                               best[lam]["C1_bi_only"]["per_seed"])

    # ---- gain from choosing the threshold for lambda --------------------------
    gain = {}
    if exp11:
        e11 = json.load(open(exp11, encoding="utf8"))["per_seed"]
        for lam in LAMBDAS:
            gain[lam] = {}
            for k in CFG:
                default = [e11[str(s)]["stats"][k]["risk_adjusted_saving_pct"][lam] for s in seeds]
                tuned = best[lam][k]["per_seed"]
                gain[lam][k] = {"default": seed_ci(default), "tuned": seed_ci(tuned),
                                "gain": paired_seed_test(tuned, default)}

    adj = holm({k: v.get("t_p") for k, v in tests.items()})
    for k, v in tests.items():
        v["holm_p"] = adj.get(k)

    for H in matched:
        line = "  ".join(f"{k[:2]} {matched[H][k]['precision_of_hits_pct'].get('mean', float('nan')):.2f}"
                         f"({matched[H][k]['n_seeds_in_range']})" for k in CFG)
        print(f"  precision at {H}% hits: {line}")
    for lam in LAMBDAS:
        line = "  ".join(f"{k[:2]} {best[lam][k]['risk_adjusted_saving_pct']['mean']:.2f}"
                         f"(edge {best[lam][k]['seeds_at_grid_edge']})" for k in CFG)
        print(f"  best RAS at lambda {lam}: {line}")

    out = {"manifest": hardware_manifest({"seeds": list(seeds), "pattern": pattern, "exp11": exp11,
                                          "hit_targets": list(HIT_TARGETS), "grid_top": grid_top,
                                          "source_hosts": [runs[s]["manifest"]["hardware"].get("cpu") for s in seeds]}),
           "matched": matched, "best_per_lambda": best, "default_vs_tuned": gain, "tests": tests}
    save_result(out, "exp27_operating_curves_seeds", tag=tag)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="exp12_threshold_risk_*_seed{seed}_ext.json")
    ap.add_argument("--exp11", default=str(REVISION_DIR / "exp11_seeds_geforcertx4090_real.json"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(a.pattern, a.exp11, tag=a.tag)
