"""Merge E7 partial/final files (Amendment 2 design) and render tables. Usage: python scripts/tables_e7.py out.md partial1.json [partial2.json ...]"""
import json
import sys

atk, ctl, extra = {}, {}, {}
for f in sys.argv[2:]:
    d = json.load(open(f))
    atk.update(d["attacks"] if "attacks" in d else d["results"]["attacks"])
    ctl.update(d["controls"] if "controls" in d else d["results"]["controls"])
pp = lambda r: f"{r['rate']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"
dd = lambda r: f"{100 * r['delta_vs_C1']:+.0f} [{100 * r['delta_ci'][0]:+.0f}, {100 * r['delta_ci'][1]:+.0f}]"
L = ["# E7 results: key-collision attack, non-adaptive, reduced CPU design (Amendment 2)", "",
     "40 victims, 60 GCG steps, 48 candidates per step. Hit rate = share of victims whose planted entry is accepted by the decision rule "
     "(95% bootstrap CI over victims). Thresholds accept 90% / 80% of held-out true duplicates. 'Delta' = system minus C1 in points "
     "(95% CI). A system is reported as more robust only if Delta <= -20 points with the CI excluding 0 (pre-registered).", ""]
for r in ("0.9", "0.8"):
    L += [f"## Recall {int(float(r) * 100)}%", "",
          "| configuration | mean cosine | C1 | C2 | Delta C2 | C2-diff | Delta C2-diff | verdict (C2 / C2-diff) |", "|---|---|---|---|---|---|---|---|"]
    rows = [(k, v) for k, v in ctl.items()] + [(k, v) for k, v in sorted(atk.items())]
    for k, v in rows:
        if r not in v:
            continue
        x = v[r]
        def verdict(s):
            lo, hi = x[s]["delta_ci"]
            return "more robust" if x[s]["delta_vs_C1"] <= -0.20 and hi < 0 else "no advantage claimed"
        lab = ("control: " if k in ctl else "attack: ") + k
        L.append(f"| {lab} | {v['mean_cosine']:.2f} | {pp(x['C1'])} | {pp(x['C2'])} | {dd(x['C2'])} | {pp(x['C2-diff'])} | {dd(x['C2-diff'])} | {verdict('C2')} / {verdict('C2-diff')} |")
    L.append("")
L += ["## Example planted entries", ""]
for k, v in sorted(atk.items()):
    for e in v.get("examples", [])[:1]:
        L.append(f"- {k}: `{e}`")
L.append("")
open(sys.argv[1], "w").write("\n".join(L))
print("\n".join(L))
