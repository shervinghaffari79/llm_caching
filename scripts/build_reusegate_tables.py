"""Build the ReuseGate tables and the numbers quoted in the text from the saved results.

Inputs: results/raw/revision/exp37_causal_reusegate_cpu_{causaltune,rot1,rot2}.json (block rotations;
causaltune is rotation 0) and ..._topicfold{0,1,2}.json (topic-held-out folds). Pooled tests are paired
t-tests over all windows, computed exactly from the per-layout (n, mean, sd) of the window-level
differences, with Holm correction over the five pre-set comparisons.

Run: python scripts/build_reusegate_tables.py > results/revision/reusegate_tables.tex
"""
import json
import math

from scipy import stats

R = "results/raw/revision/exp37_causal_reusegate_cpu_"
ROT = {"0": "causaltune", "1": "rot1", "2": "rot2"}
FOLD = {str(i): f"topicfold{i}" for i in range(3)}
LAMS = ("0", "1", "2", "5", "10")


def load(names):
    return {k: json.load(open(R + v + ".json")) for k, v in names.items()}


def pool(items):
    n = sum(i["n"] for i in items)
    m = sum(i["n"] * i["mean"] for i in items) / n
    ss = sum((i["n"] - 1) * i["sd"] ** 2 + i["n"] * (i["mean"] - m) ** 2 for i in items)
    sd = math.sqrt(ss / (n - 1))
    return n, m, 2 * stats.t.sf(abs(m / (sd / math.sqrt(n))), n - 1)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        out[i] = min(1.0, run)
    return out


def delta(d, key):
    x = d["tests"][key]
    return {"n": len(d["manifest"]["layout"]["eval"]), "mean": x["delta_mean"], "sd": x["delta_sd"]}


def pooled_tests(layouts, ref, fam, baselines_by_lam):
    rows = []
    for lam, b in baselines_by_lam:
        per = [delta(d, f"R{ref}|lambda{lam}|{fam}-{b}") for d in layouts.values()]
        rows.append((lam, b, per) + pool(per))
    return rows


def mean_over(layouts, ref, fam, lam, metric="score"):
    return sum(d["by_reference_size"][ref]["table"][lam][fam][metric]["mean"] for d in layouts.values()) / len(layouts)


def fmt_p(p):
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


def main():
    rot, fold = load(ROT), load(FOLD)
    fams = [("LRU", "LRU"), ("LFU", "LFU"), ("SphereLFU", "SphereLFU"), ("LeCaR", "LeCaR"),
            ("WTinyLFU", "W-TinyLFU"), ("Learned[match]", "ReuseGate"), ("Learned[group]", "ReuseGate-R")]
    qqp = {"LRU": "45.11 & 2.53 & \\textbf{46.48}", "LFU": "46.00 & 2.53 & 45.53",
           "SphereLFU": "45.74 & 2.52 & 46.16", "LeCaR": "45.13 & 2.53 & \\textbf{46.48}",
           "WTinyLFU": "\\textbf{46.94} & 2.44 & 43.30", "Learned[match]": "46.81 & \\textbf{2.31} & 43.02",
           "Learned[group]": "-- & -- & --"}
    print("% ---- Table: ReuseGate against baselines (log columns computed from the rotation results) ----")
    best = {lam: max(mean_over(rot, "50000", f, lam) for f, _ in fams
                     if f in rot["0"]["by_reference_size"]["50000"]["table"][lam]) for lam in LAMS}
    for f, name in fams:
        if f not in rot["0"]["by_reference_size"]["50000"]["table"]["0"]:
            continue
        cells = []
        for lam in LAMS:
            v = mean_over(rot, "50000", f, lam)
            s = f"{v:.2f}".replace("-", "$-$")
            cells.append(f"\\textbf{{{s}}}" if abs(v - best[lam]) < 1e-9 else s)
        print(f"{name} & " + " & ".join(cells) + f" & {qqp[f]} \\\\")
    print("\n% ---- Pooled tests, rotations, reference 50,000 and cold start ----")
    for ref in ("50000", "0"):
        for fam in ("Learned[match]", "Learned[group]"):
            if fam == "Learned[group]" and ref == "0":
                continue
            pl = [(0, "WTinyLFU"), (1, "WTinyLFU"), (2, "WTinyLFU"), (5, "WTinyLFU"), (10, "WTinyLFU")]
            rows = pooled_tests(rot, ref, fam, pl)
            adj = holm([r[5] for r in rows])
            for r, a in zip(rows, adj):
                print(f"rot ref={ref} {fam} lam={r[0]} vs {r[1]}: per-rotation "
                      f"{[round(p['mean'], 2) for p in r[2]]} pooled n={r[3]} {r[4]:+.2f} p={r[5]:.2g} holm={a:.2g}")
        for lam, b in ((1, "LRU"), (1, "LFU"), (1, "SphereLFU")):
            r = pooled_tests(rot, ref, "Learned[match]", [(lam, b)])[0]
            print(f"rot ref={ref} Learned[match] lam={lam} vs {b}: pooled {r[4]:+.2f} p={r[5]:.2g}")
    print("\n% ---- Pooled tests, topic folds ----")
    for ref in ("25000", "0"):
        rows = pooled_tests(fold, ref, "Learned[match]", [(l, "WTinyLFU") for l in (0, 1, 2, 5, 10)])
        adj = holm([r[5] for r in rows])
        for r, a in zip(rows, adj):
            print(f"fold ref={ref} lam={r[0]} vs WTinyLFU: per-fold {[round(p['mean'], 2) for p in r[2]]} "
                  f"pooled n={r[3]} {r[4]:+.2f} p={r[5]:.2g} holm={a:.2g}")
        for lam, b in ((1, "LRU"), (1, "LFU")):
            r = pooled_tests(fold, ref, "Learned[match]", [(lam, b)])[0]
            print(f"fold ref={ref} lam={lam} vs {b}: pooled {r[4]:+.2f} p={r[5]:.2g}")
    print("\n% ---- AUC ----")
    for name, ls, ref in (("rot", rot, "50000"), ("rot cold", rot, "0"), ("fold", fold, "25000"), ("fold cold", fold, "0")):
        a = [sum(d["by_reference_size"][ref]["auc"]) / len(d["by_reference_size"][ref]["auc"]) for d in ls.values()]
        print(name, [round(x, 3) for x in a])


if __name__ == "__main__":
    main()
