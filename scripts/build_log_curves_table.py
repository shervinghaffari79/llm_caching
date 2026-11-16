"""Real-log operating curves over the whole log (three 50,000-query blocks) -> LaTeX table rows and quoted numbers."""
import json

B = [("exp40_chronological_transfer_cpu_log_s0.json", "0--49,999"),
     ("exp40_chronological_transfer_cpu_log_s50000.json", "50,000--99,999"),
     ("exp40_chronological_transfer_cpu_log_s100000.json", "100,000--149,999")]
D = [json.load(open("results/raw/revision/" + f)) for f, _ in B]
C1 = [("C1_t0.7", "cosine $\\geq 0.70$"), ("C1_t0.726", "cosine $\\geq 0.726$ (default)"), ("C1_t0.8", "cosine $\\geq 0.80$"),
      ("C1_t0.85", "cosine $\\geq 0.85$"), ("C1_t0.895", "cosine $\\geq 0.895$"), ("C1_t0.95", "cosine $\\geq 0.95$")]
C2 = [("C2_p0.3", "$p \\geq 0.3$"), ("C2_p0.5", "$p \\geq 0.5$ (default)"), ("C2_p0.7", "$p \\geq 0.7$"),
      ("C2_p0.9", "$p \\geq 0.9$"), ("C2_p0.97", "$p \\geq 0.97$")]


def row(key, label, first):
    cells = []
    for d in D:
        s = d["results"][key]["stats"]
        cells += [f"{s['hit_ratio_pct']:.1f}\\%", f"{s['precision_of_hits_pct']:.1f}\\%"]
    return f"{first} & {label} & " + " & ".join(cells) + " \\\\"


print(row("C1_t0.7", C1[0][1], "C1: Bi-encoder only"))
for k, l in C1[1:]:
    print(row(k, l, ""))
print("\\midrule")
print(row("C2_p0.3", C2[0][1], "C2: Cheap-feature stacking"))
for k, l in C2[1:]:
    print(row(k, l, ""))
print()
for (f, name), d in zip(B, D):
    w = d["workload"]
    c1, c2 = d["results"]["C1_t0.726"]["stats"], d["results"]["C2_p0.5"]["stats"]
    print(name, "topics", w["n_topics"], "repeat %.1f" % w["repeat_within_window_pct"],
          "| false-hit share of requests C1 %.1f C2 %.1f" % (c1["hit_ratio_pct"] * (1 - c1["precision_of_hits_pct"] / 100),
                                                               c2["hit_ratio_pct"] * (1 - c2["precision_of_hits_pct"] / 100)),
          "| correct-hit C1 %.1f C2 %.1f" % (c1["hit_ratio_pct"] * c1["precision_of_hits_pct"] / 100,
                                              c2["hit_ratio_pct"] * c2["precision_of_hits_pct"] / 100))
