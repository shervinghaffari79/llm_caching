"""Render E5 (slices) and E6 (band, McNemar, taxonomy) result tables as markdown. Usage: python scripts/tables_e5_e6.py out.md"""
import json
import sys

R = "results/raw/revision/"
e5 = json.load(open(R + "exp49_slices_cpu.json"))
e6 = json.load(open(R + "exp50_band_mcnemar_cpu.json"))
L = []
pp = lambda x: f"{100 * x:.1f}"
L += ["# Round 2b, E5 and E6 results (pre-registered in docs/PREREGISTRATION_round2b.md)", "",
      "All numbers come from `exp49_slices_cpu.json` and `exp50_band_mcnemar_cpu.json`. Models are trained on QQP only; "
      "thresholds are set so each system accepts the same share (90% or 80%) of true duplicate pairs. "
      "FAR = false-acceptance rate on pairs that are no longer equivalent. Delta = system minus C1 (points); "
      "95% bootstrap CI over source pairs. A slice 'differs' if |Delta| >= 5 points with the CI excluding 0 "
      "(no multiplicity correction was applied to the CIs below; where the CI is close to 0 treat the call as tentative).", ""]
for r in ("0.9", "0.8"):
    L += [f"## E5-A. QQP perturbation slices, matched recall {int(float(r) * 100)}%", "",
          "| slice | source pairs | FAR C1 | FAR C2 | Delta C2 [95% CI] | FAR C2-diff | Delta C2-diff [95% CI] | verdict C2 / C2-diff |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in e5["partA_qqp"].items():
        s = v[r]
        def verdict(sys_):
            d, (lo, hi) = s["delta"][sys_], s["ci"][sys_]
            if abs(d) >= 0.05 and (lo > 0 or hi < 0):
                return "better" if d < 0 else "worse"
            return "tie"
        if k == "control":
            L.append(f"| control (label-preserving; value = share still ACCEPTED, higher is better) | {s['n']} | {pp(s['far']['C1'])} | {pp(s['far']['C2'])} | "
                     f"{100 * s['delta']['C2']:+.1f} [{100 * s['ci']['C2'][0]:+.1f}, {100 * s['ci']['C2'][1]:+.1f}] | {pp(s['far']['C2-diff'])} | "
                     f"{100 * s['delta']['C2-diff']:+.1f} [{100 * s['ci']['C2-diff'][0]:+.1f}, {100 * s['ci']['C2-diff'][1]:+.1f}] | - |")
            continue
        L.append(f"| {k} | {s['n']} | {pp(s['far']['C1'])} | {pp(s['far']['C2'])} | {100 * s['delta']['C2']:+.1f} "
                 f"[{100 * s['ci']['C2'][0]:+.1f}, {100 * s['ci']['C2'][1]:+.1f}] | {pp(s['far']['C2-diff'])} | "
                 f"{100 * s['delta']['C2-diff']:+.1f} [{100 * s['ci']['C2-diff'][0]:+.1f}, {100 * s['ci']['C2-diff'][1]:+.1f}] | "
                 f"{verdict('C2')} / {verdict('C2-diff')} |")
    L.append("")
h = e5["partB_external"]["HEROS"]
L += [f"## E5-B. HEROS (zero-shot, base = {h['base']}), FAR of pairs labelled non-equivalent", "",
      "| subset | R=90% C1 | C2 | C2-diff | R=80% C1 | C2 | C2-diff |", "|---|---|---|---|---|---|---|"]
for k, v in h["slices"].items():
    L.append(f"| {k} | " + " | ".join(pp(v[r][s]) for r in ("0.9", "0.8") for s in ("C1", "C2", "C2-diff")) + " |")
p = e5["partB_external"]["PAWS"]
L += ["", f"## E5-C. PAWS ({p['n']} pairs, {100 * p['positive_rate']:.0f}% paraphrase)", "",
      "| system | AUC | FAR at 90% recall | FAR at 80% recall |", "|---|---|---|---|"]
for s in ("C1", "C2", "C2-diff"):
    L.append(f"| {s} | {p['auc'][s]:.3f} | {pp(p['far_at_recall']['0.9'][s])} | {pp(p['far_at_recall']['0.8'][s])} |")
L += ["", "SemAntoNeg: not run (source not located); not claimed.", ""]
L += ["## E6-A. Band diagnostic", "", f"QQP meta set, {e6['pairs']} pairs, 5-fold group-disjoint out-of-fold scores. Accept = threshold at matched 90% recall on the whole set.", "",
      "| band | pairs | duplicate share | system | AUC | accept rate | precision of accepted |", "|---|---|---|---|---|---|---|"]
for lab, r in e6["bands"].items():
    for s in ("C1", "C2", "C2-diff"):
        v = r[s]
        L.append(f"| {lab} | {r['pairs_in_band']} | {100 * r['positive_rate']:.0f}% | {s} | {v['auc']:.3f} | {100 * v['accept_rate']:.0f}% | {100 * v['precision']:.1f}% |")
L += ["", "## E6-B. Paired decisions at 90% matched recall (C1 vs system)", "",
      f"Total errors: {e6['total_errors']}.", "",
      "| system | C1 errors fixed | errors added | net | exact McNemar p | false hits fixed / added | misses fixed / added |", "|---|---|---|---|---|---|---|"]
for s, r in e6["mcnemar_at_recall_0.90"].items():
    L.append(f"| {s} | {r['fixed']} | {r['added']} | {r['fixed'] - r['added']:+d} | {r['mcnemar_p']:.1e} | {r['false_hits']['fixed']} / {r['false_hits']['added']} | {r['misses']['fixed']} / {r['misses']['added']} |")
for s in ("C2", "C2-diff"):
    L += ["", f"## E6-C. Error taxonomy, {s} vs C1", "", "| detector category | pairs | C1 errors | fixed | added | net |", "|---|---|---|---|---|---|"]
    for c, v in e6["taxonomy"][s].items():
        L.append(f"| {c} | {v['pairs']} | {v['C1_errors']} | {v['fixed']} | {v['added']} | {v['fixed'] - v['added']:+d} |")
L.append("")
open(sys.argv[1], "w").write("\n".join(L))
print("\n".join(L))
