"""
Generate the manuscript's result tables (LaTeX) from the revision results.

    python scripts/build_manuscript_tables.py

Writes results/revision/manuscript_tables.tex. Every number is read from
results/raw/revision/*.json. Each table is preceded by a comment block giving
its placement in Manuscript-2026-08-09.tex (section, anchor, new or
replacing an existing label) and the reviewer comments it serves. Style
follows the manuscript: table/table*, caption above, booktabs, arrows in
headers, best values in bold.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "revision"
OUT = ROOT / "results" / "revision" / "manuscript_tables.tex"
sys.path.insert(0, str(ROOT))

CFG = ("C1_bi_only", "C2_bi_geo_lgb", "C3_bi_cross", "C4_full_stacking", "C5_gated")
NAME = {"C1_bi_only": "C1: Bi-encoder only", "C2_bi_geo_lgb": "C2: Cheap-feature stacking",
        "C3_bi_cross": "C3: Cross-encoder re-ranking", "C4_full_stacking": "C4: Full stacking",
        "C5_gated": "C5: Gated verification"}
BASE = {"gptcache": "GPTCache", "scalm": "SCALM", "vectorq": "VectorQ", "contextcache": "ContextCache"}


def load(n):
    return json.load(open(RAW / n, encoding="utf8"))


def mn(x):
    return x.get("mean") if isinstance(x, dict) else x


def num(x, nd=1, unit=""):
    x = mn(x)
    if x is None:
        return "--"
    return f"{x:.{nd}f}{unit}"


def pct(x, nd=1):
    return num(x, nd, r"\%")


def ci(x, nd=1, unit=r"\%"):
    if not isinstance(x, dict) or "ci_low" not in x:
        return num(x, nd, unit)
    return f"{x['mean']:.{nd}f}{unit} [{x['ci_low']:.{nd}f}, {x['ci_high']:.{nd}f}]"


def bold_best(rows, cols, values, higher, decimals=None):
    """Bold the best cell in each column index of `cols`, comparing values as
    displayed (rounded to `decimals`, default 1), so displayed ties are all bold.
    `values[r][c]` holds the raw number for row r, column c (or None)."""
    decimals = decimals or [1] * len(cols)
    for c, hi, nd in zip(cols, higher, decimals):
        vals = [(round(values[r][c], nd), r) for r in range(len(rows)) if values[r][c] is not None]
        if not vals:
            continue
        best = (max if hi else min)(v for v, _ in vals)
        for v, r in vals:
            if v == best:
                rows[r][c] = r"\textbf{" + rows[r][c] + "}"
    return rows


def tab(label, caption, head, rows, star=True, resize=True, placement="", serves="", colspec=None):
    env = "table*" if star else "table"
    colspec = colspec or ("l" + "c" * (len(head) - 1))
    body = "\n".join(r"\midrule" if r and r[0] == "__mid__" else " & ".join(str(c) for c in r) + r" \\"
                     for r in rows)
    lines = [f"% ---------------------------------------------------------------------------",
             f"% {label}",
             f"% Placement: {placement}",
             f"% Serves:    {serves}",
             f"% ---------------------------------------------------------------------------",
             rf"\begin{{{env}}}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}"]
    tabular = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(head) + r" \\", r"\midrule", body,
               r"\bottomrule", r"\end{tabular}"]
    if resize:
        lines += [r"\resizebox{\textwidth}{!}{" if star else r"\resizebox{\columnwidth}{!}{"] + tabular + ["}"]
    else:
        lines += [r"\footnotesize"] + tabular
    lines += [rf"\end{{{env}}}", ""]
    return "\n".join(lines)


def mid(n_cols):
    return ["__mid__"] + [""] * (n_cols - 1)


# =========================================================================
def t_end_to_end():
    e = load("exp2_end_to_end_geforcertx4090_seed42_real.json")
    nc = e["no_cache"]
    rows, vals = [], []
    rows.append(["No cache", "--", "--", "--", num(nc["avg_response_time_s"], 3, " s"),
                 num(nc["p95_response_time_s"], 2, " s"), r"1.00$\times$", r"0.0\%", r"0.0\%"])
    vals.append([None] * 9)
    for r in e["configs"]:
        rows.append([NAME[r["config"]], num(r["avg_overhead_ms"], 2, " ms"), pct(r["hit_ratio_pct"]),
                     pct(r["precision_of_hits_pct"]), num(r["avg_response_time_s"], 3, " s"),
                     num(r["p95_response_time_s"], 2, " s"), num(r["speedup_vs_no_cache"], 2, r"$\times$"),
                     pct(r["token_saving_pct"]), pct(r["token_saving_correct_only_pct"])])
        vals.append([None, r["avg_overhead_ms"], r["hit_ratio_pct"], r["precision_of_hits_pct"],
                     r["avg_response_time_s"], r["p95_response_time_s"], r["speedup_vs_no_cache"],
                     r["token_saving_pct"], r["token_saving_correct_only_pct"]])
    rows = bold_best(rows, [1, 2, 3, 4, 5, 6, 7, 8], vals, [False, True, True, False, False, True, True, True],
                     [2, 1, 1, 3, 2, 2, 1, 1])
    return tab("tab:end_to_end",
               "End-to-end results on the 2,000-request QQP-derived stream (seed 42) on an RTX 4090 host. Every miss "
               "waits for the recorded generation latency of the served model. Decision overhead covers work up to the "
               "hit-or-miss commitment and excludes generation and insertion. Headline token saving counts every served "
               "hit; correct-only saving counts tokens reused by true hits only.",
               ["Configuration", r"Decision overhead ($\downarrow$)", r"Served-hit ratio ($\uparrow$)",
                r"Hit precision ($\uparrow$)", r"Avg.\ response ($\downarrow$)", r"p95 response ($\downarrow$)",
                r"Speed-up ($\uparrow$)", r"Token saving ($\uparrow$)", r"Correct-only saving ($\uparrow$)"], rows,
               placement="Sec. Stateful Semantic Caching Performance; REPLACES the existing tab:end_to_end. Update the "
                         "two paragraphs after it.",
               serves="R1.3 (latency), R1.2 (correct-only saving), R2.3")


def t_seeds():
    d = load("exp11_seeds_geforcertx4090_real.json")["across_seeds"]
    rows, vals = [], []
    for k in CFG:
        a = d[k]
        rows.append([NAME[k], ci(a["hit_ratio_pct"]), ci(a["precision_of_hits_pct"]),
                     ci(a["token_saving_correct_only_pct"]), num(a["avg_overhead_ms"], 2, " ms"),
                     num(a["avg_response_time_s"], 3, " s"), num(a["break_even_false_hit_cost"], 1)])
        vals.append([None, mn(a["hit_ratio_pct"]), mn(a["precision_of_hits_pct"]), mn(a["token_saving_correct_only_pct"]),
                     mn(a["avg_overhead_ms"]), mn(a["avg_response_time_s"]), mn(a["break_even_false_hit_cost"])])
    rows = bold_best(rows, [1, 2, 3, 4, 5, 6], vals, [True, True, True, False, False, True], [1, 1, 1, 2, 3, 1])
    return tab("tab:seed_variation",
               "Configurations across five independent stream seeds (42--46): mean with 95\\% $t$-interval across seeds. "
               r"Break-even $\lambda$ is the false-hit cost, in units of one generation, at which risk-adjusted saving "
               "reaches zero.",
               ["Configuration", r"Served-hit ratio ($\uparrow$)", r"Hit precision ($\uparrow$)",
                r"Correct-only saving ($\uparrow$)", r"Decision overhead ($\downarrow$)", r"Avg.\ response ($\downarrow$)",
                r"Break-even $\lambda$ ($\uparrow$)"], rows,
               placement="Sec. Stateful Semantic Caching Performance, directly after tab:end_to_end. NEW.",
               serves="R1.1, R1.5")


def t_paired():
    c = load("exp11_seeds_geforcertx4090_real.json")["c1_vs_c2_across_seeds"]
    lab = {"hit_ratio_pct": r"Served-hit ratio (pp)", "precision_of_hits_pct": r"Hit precision (pp)",
           "token_saving_correct_only_pct": r"Correct-only saving (pp)", "avg_overhead_ms": r"Decision overhead (ms)",
           "avg_response_time_s": r"Avg.\ response (s)", "break_even_false_hit_cost": r"Break-even $\lambda$"}
    rows = []
    for k, nm in lab.items():
        v = c[k]
        nd = 3 if k == "avg_response_time_s" else 2
        p = v.get("t_p")
        rows.append([nm, f"{v['delta_mean']:+.{nd}f}", f"[{v['delta_ci_low']:+.{nd}f}, {v['delta_ci_high']:+.{nd}f}]",
                     ("$<$0.001" if p is not None and p < 1e-3 else f"{p:.3f}") if p is not None else "--"])
    return tab("tab:c1_c2_paired",
               "Seed-level paired comparison of C1 and C2 on the five streams of Table~\\ref{tab:seed_variation}: "
               "difference C1 $-$ C2, 95\\% interval, and paired $t$-test.",
               ["Measure", r"$\Delta$ (C1 $-$ C2)", "95\\% interval", "$p$"], rows, star=False, resize=False,
               placement="Sec. Stateful Semantic Caching Performance, after tab:seed_variation. NEW.",
               serves="R1.1")


def t_risk():
    e = load("exp2_end_to_end_geforcertx4090_seed42_real.json")
    lams = ("0", "1", "2", "5", "10")
    rows, vals = [], []
    for r in e["configs"]:
        ras = r["risk_adjusted_saving_pct"]
        rows.append([NAME[r["config"]], pct(r["false_discovery_rate_pct"]), pct(r["false_hit_rate_pct"], 2),
                     *[pct(ras[l]) for l in lams], num(r["break_even_false_hit_cost"], 1)])
        vals.append([None, r["false_discovery_rate_pct"], r["false_hit_rate_pct"], *[ras[l] for l in lams],
                     r["break_even_false_hit_cost"]])
    rows = bold_best(rows, list(range(1, 9)), vals, [False, False, True, True, True, True, True, True],
                     [1, 2, 1, 1, 1, 1, 1, 1])
    return tab("tab:risk_adjusted",
               "False hits and risk-adjusted saving on the seed-42 stream. The false-discovery rate is the share of "
               "served hits that are false; the false-hit rate is their share of all requests. Risk-adjusted saving "
               r"charges each false hit $\lambda$ times the generation it replaced, so $\lambda=0$ equals correct-only saving.",
               ["Configuration", r"FDR ($\downarrow$)", r"False-hit rate ($\downarrow$)", r"$\lambda=0$", r"$\lambda=1$",
                r"$\lambda=2$", r"$\lambda=5$", r"$\lambda=10$", r"Break-even $\lambda$ ($\uparrow$)"], rows,
               placement="NEW subsection 'Pricing False Hits' after the Latency Breakdown subsubsection of Sec. Stateful "
                         "Semantic Caching Performance. Define risk-adjusted saving and break-even lambda in "
                         "Sec. Evaluation Metrics and Statistical Significance.",
               serves="R1.2")


def _e27():
    import glob as _g
    return json.load(open(sorted(_g.glob(str(RAW / "exp27_operating_curves_seeds_*.json")))[-1], encoding="utf8"))


def t_matched():
    d = _e27()
    tests = d["tests"]
    rows = []
    for h, row in d["matched"].items():
        cells, vals = [], []
        for k in CFG[:4]:
            c = row[k]["precision_of_hits_pct"]
            ok = row[k]["n_seeds_in_range"] == 5
            vals.append(round(c["mean"], 2) if ok else None)
            cells.append(f"{c['mean']:.2f}\\% $\\pm$ {c['sd']:.2f}" if ok else "--")
        if vals[0] is None or sum(v is not None for v in vals) < 2:   # comparisons are against C1
            continue
        best = max(v for v in vals if v is not None)
        cells = [(r"\textbf{" + c + "}") if v is not None and v == best else c for c, v in zip(cells, vals)]
        rows.append([f"{float(h):.0f}\\%", *cells])
    worst = min((v["holm_p"] for k, v in tests.items() if k.startswith("precision") and v.get("holm_p") is not None),
                default=1.0)
    return tab("tab:matched_precision",
               "Hit precision at matched served-hit ratio, mean $\\pm$ standard deviation over seeds 42--46, "
               "interpolated along each seed's full threshold sweep. Dashes mark ratios outside a configuration's "
               f"swept range on some seed. No difference from C1 is significant (smallest Holm-adjusted $p={worst:.2f}$).",
               [r"Served-hit ratio", "C1", "C2", "C3", "C4"], rows, star=False, resize=True,
               placement="NEW subsection 'Pricing False Hits' (with tab:risk_adjusted and tab:best_per_lambda); "
                         "or Sec. Comparison with Published Semantic Caches next to the operating-curve figure (F3).",
               serves="R1.1, R1.2, R3.3")


def t_best_lambda():
    d = _e27()
    b, g = d["best_per_lambda"], d["default_vs_tuned"]
    rows = []
    for lam in ("0", "1", "2", "5", "10"):
        vals = [round(b[lam][k]["risk_adjusted_saving_pct"]["mean"], 1) for k in CFG[:4]]
        best = max(vals)
        cells = []
        for k, v in zip(CFG[:4], vals):
            dflt = g[lam][k]["default"]["mean"]
            s_ = f"{v:.1f}\\% ({dflt:.1f}\\%)"
            cells.append(r"\textbf{" + s_ + "}" if v == best else s_)
        rows.append([f"$\\lambda={lam}$", *cells])
    return tab("tab:best_per_lambda",
               "Risk-adjusted saving when each configuration's decision threshold is chosen for the false-hit cost "
               "$\\lambda$, mean over seeds 42--46; the value at the configuration's default threshold is in "
               "parentheses. No difference between configurations at a chosen threshold is significant "
               "(Holm-adjusted paired $t$-tests against C1).",
               ["False-hit cost", "C1", "C2", "C3", "C4"], rows, star=False, resize=True,
               placement="NEW subsection 'Pricing False Hits', after tab:matched_precision.",
               serves="R1.1, R1.2")


def t_latency():
    d = load("exp1_latency_breakdown_geforcertx4090.json")["per_config"]
    rows = [[NAME.get(r["config"], r["config"]), num(r["bi_encode_ms"], 2, " ms"), num(r["vector_search_ms"], 2, " ms"),
             num(r["cross_encode_ms"], 2, " ms"), num(r["features_meta_ms"], 2, " ms"),
             num(r["total_overhead_ms"], 2, " ms"), num(r["overhead_vs_bi_only_x"], 2, r"$\times$")] for r in d]
    vals = [[None, None, None, None, None, r["total_overhead_ms"], None] for r in d]
    rows = bold_best(rows, [5], vals, [False], [2])
    return tab("tab:latency_breakdown",
               "Fixed-index component microbenchmark: 300 batch-size-1 queries against a 2,000-vector index on an RTX "
               "4090 host (AMD Ryzen 9 3900X). Generation, insertion, and index growth are excluded.",
               ["Configuration", r"Bi-encoder ($\downarrow$)", r"FAISS ($\downarrow$)", r"Cross-encoder ($\downarrow$)",
                r"Features + LightGBM ($\downarrow$)", r"Total ($\downarrow$)", r"Relative to C1"], rows,
               placement="Sec. Latency Breakdown (\\label{subsubsec:latency_breakdown}); REPLACES the existing "
                         "tab:latency_breakdown. Update the surrounding numbers (cross-encoder 11.50 ms, features+LightGBM "
                         "2.93 ms) and the cost columns of tab:ablation_isolation and tab:cheap_vs_expensive.",
               serves="R1.3, R2.3")


def t_latency_dist():
    e = load("exp14_latency_geforcertx4090.json")
    d, cc = e["decomposition"], e["concurrency"]
    rows = []
    for k in CFG[:4]:
        t = d[k]["total"]
        c = cc.get(k, {})
        rows.append([NAME[k], num(t["mean_ms"], 2, " ms"), num(t["p50_ms"], 2, " ms"), num(t["p95_ms"], 2, " ms"),
                     num(t["p99_ms"], 2, " ms"),
                     *[num(c[n]["p95_ms"], 1, " ms") if n in c else "--" for n in ("4", "16")],
                     *[num(c[n]["throughput_rps"], 0) if n in c else "--" for n in ("1", "16")]])
    return tab("tab:latency_distribution",
               "Distribution of decision-path cost for single requests (batch size 1, 300 queries) and under concurrent "
               "clients sharing one GPU. Throughput is in requests per second.",
               ["Configuration", "Mean", "p50", "p95", "p99", "p95, 4 clients", "p95, 16 clients",
                "Throughput, 1 client", "Throughput, 16 clients"], rows,
               placement="Sec. Latency Breakdown, after tab:latency_breakdown. NEW.",
               serves="R1.3")


def t_batching():
    cb = load("exp14_latency_geforcertx4090.json")["cross_batching"]
    rows = [["Cost per pair", *[num(v["ms_per_pair"], 2, " ms") for v in cb.values()]]]
    return tab("tab:cross_batching",
               "Cross-encoder cost per pair by batch size on the RTX 4090 host.",
               ["Batch size", *cb.keys()], rows, star=False, resize=False,
               placement="Sec. Latency Breakdown, replacing the paragraph on offline batching with this table. NEW.",
               serves="R1.3")


def t_topk():
    d = load("exp13_topk_gate_geforcertx4090_seed42.json")["topk"]
    rows, prev = [], None
    for r in d:
        if prev and r["config"] != prev:
            rows.append(mid(6))
        prev = r["config"]
        rows.append([NAME[r["config"]] if r["k"] == 1 else "", r["k"], pct(r["hit_ratio_pct"]),
                     pct(r["precision_of_hits_pct"]), pct(r["token_saving_correct_only_pct"]),
                     num(r["avg_overhead_ms"], 2, " ms")])
    return tab("tab:topk",
               "Scoring the $k$ nearest cached entries and serving the best accepted candidate (seed 42). The "
               "candidates of one request are scored in a single batched call.",
               ["Configuration", "$k$", r"Served-hit ratio ($\uparrow$)", r"Hit precision ($\uparrow$)",
                r"Correct-only saving ($\uparrow$)", r"Decision overhead ($\downarrow$)"], rows, star=False,
               placement="NEW subsection 'Top-k Retrieval and Selective Verification' in Results (after Latency "
                         "Breakdown). Describe top-k scoring in Sec. Query Processing.",
               serves="R3.1, R3.5")


def t_gate():
    d = load("exp13_topk_gate_geforcertx4090_seed42.json")
    c2 = d["c2_reference"]
    rows = [["C2 (no verification)", "--", num(c2.get("cross_calls_per_request", 0), 3), pct(c2["hit_ratio_pct"]),
             pct(c2["precision_of_hits_pct"]), num(c2["avg_overhead_ms"], 2, " ms")], mid(6)]
    for g in d["gating"]:
        b = g["band"]
        rows.append(["C5: gated cross-encoder", f"[{b['lo']:.2f}, {b['hi']:.2f}] ({100 * b['share_target']:.0f}\\%)",
                     num(g["cross_calls_per_request"], 3), pct(g["hit_ratio_pct"]), pct(g["precision_of_hits_pct"]),
                     num(g["avg_overhead_ms"], 2, " ms")])
    rows.append(mid(6))
    for a in d["abstention"]:
        rows.append(["C2 + abstention", f"$\\pm${a['margin']:g} around 0.5", "0.000", pct(a["hit_ratio_pct"]),
                     pct(a["precision_of_hits_pct"]), num(a["avg_overhead_ms"], 2, " ms")])
    return tab("tab:gating",
               "Selective verification (seed 42). C5 calls the cross-encoder only when the C2 probability falls inside "
               "an uncertainty band, chosen on out-of-fold training predictions to cover the stated share of training "
               "pairs; abstention instead treats such requests as misses.",
               ["Variant", "Band", "Cross-encoder calls per request", r"Served-hit ratio ($\uparrow$)",
                r"Hit precision ($\uparrow$)", r"Decision overhead ($\downarrow$)"], rows,
               placement="NEW subsection 'Top-k Retrieval and Selective Verification', after tab:topk. Add C5 to "
                         "tab:component_activation and Sec. Evaluated Cache Configurations.",
               serves="R3.1, R1.2, R3.5")


def t_workload():
    import semcache.config as C
    d = load("exp15_workload_sensitivity_geforcertx4090.json")["results"]
    lab = {"zipf_s": "Zipf exponent", "duplicate_rate": "Duplicate rate", "hard_negative_rate": "Hard-negative rate"}
    rows, prev = [], None
    for r in d:
        if prev and r["factor"] != prev:
            rows.append(mid(6))
        rows.append([lab[r["factor"]] if r["factor"] != prev else "", f"{r['value']:g}",
                     *[f"{mn(r['summary'][k]['hit_ratio_pct']):.1f} / {mn(r['summary'][k]['precision_of_hits_pct']):.1f}"
                       for k in CFG[:4]]])
        prev = r["factor"]
    return tab("tab:workload_sensitivity",
               "Sensitivity to workload composition: served-hit ratio / hit precision (\\%), mean over three seeds. One "
               f"generator parameter varies at a time from the default stream (Zipf exponent {C.STREAM_ZIPF_S:g}, "
               f"duplicate rate {C.STREAM_DUPLICATE_RATE:g}, hard-negative rate 0.35).",
               ["Parameter", "Value", "C1", "C2", "C3", "C4"], rows,
               placement="NEW subsubsection 'Workload Sensitivity' at the end of Sec. Stateful Semantic Caching "
                         "Performance.",
               serves="R1.5")


def t_baselines():
    b = load("exp19_baselines_seeds_m2.json")["across_seeds"]
    c = load("exp11_seeds_geforcertx4090_real.json")["across_seeds"]
    rows, vals = [], []
    for k, nm in BASE.items():
        a = b[k]
        rows.append([nm, ci(a["hit_ratio_pct"]), ci(a["precision_of_hits_pct"]), pct(a["token_saving_correct_only_pct"]),
                     num(a["break_even_false_hit_cost"], 1), "--"])
        vals.append([None, mn(a["hit_ratio_pct"]), mn(a["precision_of_hits_pct"]), mn(a["token_saving_correct_only_pct"]),
                     mn(a["break_even_false_hit_cost"]), None])
    for d in ("0.01", "0.02", "0.05"):
        a = b[f"vcache_d{d}"]
        rows.append([f"vCache ($\\delta={d}$)", ci(a["hit_ratio_pct"]), ci(a["precision_of_hits_pct"]),
                     pct(a["token_saving_correct_only_pct"]), num(a["break_even_false_hit_cost"], 1),
                     num(a["judge_calls_per_request"], 2)])
        vals.append([None, mn(a["hit_ratio_pct"]), mn(a["precision_of_hits_pct"]), mn(a["token_saving_correct_only_pct"]),
                     mn(a["break_even_false_hit_cost"]), None])
    rows.append(mid(6))
    vals.append([None] * 6)
    for k in CFG:
        a = c[k]
        rows.append([NAME[k], ci(a["hit_ratio_pct"]), ci(a["precision_of_hits_pct"]), pct(a["token_saving_correct_only_pct"]),
                     num(a["break_even_false_hit_cost"], 1), "--"])
        vals.append([None, mn(a["hit_ratio_pct"]), mn(a["precision_of_hits_pct"]), mn(a["token_saving_correct_only_pct"]),
                     mn(a["break_even_false_hit_cost"]), None])
    rows = bold_best(rows, [1, 2, 3, 4], vals, [True, True, True, True])
    return tab("tab:baseline_comparison",
               "Decision quality on the five seed streams (mean, with 95\\% interval across seeds for hit ratio and "
               "precision). Published systems are decision-rule reimplementations on the same bi-encoder, exact FAISS "
               "index, cache substrate and streams, so they differ only in their acceptance and retention rules. "
               "Threshold rules use the validation threshold of C1 (0.726). vCache uses its own insertion rule and "
               "receives the stream's equivalence label on every explored request; verification calls count these checks.",
               ["System", r"Served-hit ratio ($\uparrow$)", r"Hit precision ($\uparrow$)", r"Correct-only saving ($\uparrow$)",
                r"Break-even $\lambda$ ($\uparrow$)", "Verification calls per request"], rows,
               placement="Sec. Comparison with Published Semantic Caches (\\label{subsec:baseline_comparison}); REPLACES "
                         "the existing tab:baseline_comparison. Add vCache (ICLR 2026) to the paragraph listing the systems.",
               serves="R2.4, R3.3, R3.5")


def t_baseline_timing():
    rows = []
    for k, nm in [*BASE.items(), ("vcache_d0.02", r"vCache ($\delta=0.02$)")]:
        f = RAW / f"exp21_baseline_timing_geforcertx4090_seed42_{k}.json"
        s = json.load(open(f))["stats"]
        rows.append([nm, pct(s["hit_ratio_pct"]), pct(s["precision_of_hits_pct"]), num(s["avg_overhead_ms"], 2, " ms"),
                     num(s["avg_response_time_s"], 3, " s")])
    rows.append(mid(5))
    hc = load("exp14_latency_geforcertx4090_host_check.json")["decomposition"]
    for k in CFG[:4]:
        rows.append([NAME[k], "--", "--", num(hc[k]["total"]["mean_ms"], 2, " ms"), "--"])
    return tab("tab:baseline_timing",
               "Published systems served with real generation latency on the seed-42 stream, measured on one RTX 4090 "
               "host. The lower rows give the decision-path cost of C1--C4 on the same host (fixed-index benchmark, "
               "batch size 1) for reference.",
               ["System", r"Served-hit ratio ($\uparrow$)", r"Hit precision ($\uparrow$)", r"Decision overhead ($\downarrow$)",
                r"Avg.\ response ($\downarrow$)"], rows, star=False,
               placement="Sec. Comparison with Published Semantic Caches, after tab:baseline_comparison. NEW.",
               serves="R2.4, R1.3")


def t_long_stream():
    d = load("exp20_long_stream_m2_seed42.json")["results"]
    lab = {"C1_bi_only": NAME["C1_bi_only"], "C2_bi_geo_lgb": NAME["C2_bi_geo_lgb"], "C5_gated": NAME["C5_gated"],
           "gptcache": "GPTCache", "vcache_d0.01": r"vCache ($\delta=0.01$)", "vcache_d0.02": r"vCache ($\delta=0.02$)",
           "vcache_d0.05": r"vCache ($\delta=0.05$)"}
    rows = []
    for k, nm in lab.items():
        v = d[k]
        bl = v["blocks"]
        rows.append([nm, pct(v["stats"]["hit_ratio_pct"]), pct(v["stats"]["precision_of_hits_pct"]),
                     pct(bl[0]["hit_ratio_pct"]), pct(bl[len(bl) // 2]["hit_ratio_pct"]), pct(bl[-1]["hit_ratio_pct"])])
    return tab("tab:long_stream",
               "A 10,000-request stream from the same generator (seed 42): overall served-hit ratio and precision, and "
               "the served-hit ratio in the first, middle and last block of 1,000 requests.",
               ["System", r"Served-hit ratio ($\uparrow$)", r"Hit precision ($\uparrow$)", "Requests 1--1,000",
                "5,001--6,000", "9,001--10,000"], rows, star=False,
               placement="Sec. Comparison with Published Semantic Caches, after tab:baseline_timing. NEW.",
               serves="R2.4, R1.5")


def t_eviction():
    e18, e23, e24 = load("exp18_eviction_m2.json"), load("exp23_eviction_headroom_m2.json"), load("exp24_sd_admit_cpu.json")
    caps = ["200", "300", "400", "500", "600", "800", "1000"]
    t = e24["tests"]
    holm = load("holm_adjusted.json")["exp24 admission vs references"]["holm"]
    rows = [["Clairvoyant reference", *[num(e23["summary"][c]["Clairvoyant"], 2) for c in caps]], mid(len(caps) + 1)]
    for pol, nm in (("LRU", "LRU"), ("LFU", "LFU"), ("SphereLFU", "SphereLFU"),
                    ("SemanticDecay_fixed", "Semantic Decay, default weights"),
                    ("SemanticDecay_tuned", "Semantic Decay, tuned weights")):
        rows.append([nm, *[num(e18["summary"][pol][c]["correct_hit_ratio_pct"], 2) for c in caps]])
    rows.append(["Frequency with aging, no admission", *[num(e24["summary"][c]["SD-Admit_no_admission"], 2) for c in caps]])
    cells = []
    for c in caps:
        mine = round(mn(e24["summary"][c]["SD-Admit"]), 2)
        others = [round(mn(e18["summary"][p][c]["correct_hit_ratio_pct"]), 2) for p in
                  ("LRU", "LFU", "SphereLFU", "SemanticDecay_fixed", "SemanticDecay_tuned")]
        others.append(round(mn(e24["summary"][c]["SD-Admit_no_admission"]), 2))
        v = f"{mine:.2f}"
        sig = all(holm.get(f"SD-Admit|{r}|{c}", 1.0) < 0.05 for r in ("LFU", "SphereLFU"))
        cells.append((r"\textbf{" + v + "}" if mine > max(others) else v) + (r"$^\dagger$" if sig else ""))
    rows.append(["Frequency with aging and admission", *cells])
    return tab("tab:eviction_capacity",
               "Correct-hit ratio (\\%) under capacity limits with C2 serving, mean over seeds 42--46. Tuned weights "
               "and the admission policy's parameters were selected on a separate stream (seed 1042) and frozen. The "
               "clairvoyant reference knows the whole stream and evicts the entry whose topic recurs latest. "
               r"$^\dagger$Higher than both LFU and SphereLFU (paired $t$-test over seeds, Holm-adjusted $p<0.05$).",
               ["Policy", *[f"{c}" for c in caps]], rows,
               placement="Sec. Eviction Policy Evaluation (\\label{subsubsec:eviction_policy_impact}); REPLACES "
                         "tab:semantic_decay_sensitivity. Keep tab:semantic_decay_weights (definition); add the tuned "
                         "weights (0, 0, 1) to the text. Row names depend on the eviction framing decision.",
               serves="R1.4, R3.2")


def t_admission_risk():
    e26 = load("exp26_risk_priced_cpu.json")["summary"]
    rows, prev = [], None
    for lam in ("1", "5", "10"):
        if prev:
            rows.append(mid(6))
        prev = lam
        for c in ("200", "400", "800"):
            s = e26[lam][c]
            vals = [round(mn(s[n]["risk_adjusted_saving_pct"]), 1) for n in ("LFU", "SphereLFU", "SD-Admit", f"RiskPriced@{lam}")]
            best = max(vals)
            rows.append([f"$\\lambda={lam}$" if c == "200" else "", c,
                         *[(r"\textbf{" + f"{v:.1f}" + "}") if v == best else f"{v:.1f}" for v in vals]])
    return tab("tab:admission_risk",
               "Risk-adjusted saving (\\%) under capacity limits, mean over seeds 42--46. The risk-priced policy values "
               "each served hit by its expected net saving, $p-\\lambda(1-p)$, where $p$ is the decision layer's "
               "confidence; all policy parameters were selected on the separate tuning stream for each $\\lambda$.",
               ["False-hit cost", "Capacity", "LFU", "SphereLFU", "Frequency + admission", "Risk-priced"], rows, star=False,
               resize=False,
               placement="Sec. Eviction Policy Evaluation, after tab:eviction_capacity. NEW. Keep or drop the "
                         "risk-priced column depending on the eviction framing.",
               serves="R1.4, R1.2")


def t_admission_precision():
    e26 = load("exp26_risk_priced_cpu.json")["summary"]["1"]
    rows = []
    for c in ("200", "400", "800"):
        s = e26[c]
        rows.append([c, pct(s["LFU"]["false_hit_rate_pct"], 2), pct(s["SD-Admit"]["false_hit_rate_pct"], 2),
                     pct(s["LFU"]["precision_of_hits_pct"]), pct(s["SD-Admit"]["precision_of_hits_pct"])])
    return tab("tab:admission_precision",
               "False-hit rate and hit precision with and without admission control (C2 serving, mean over seeds 42--46).",
               ["Capacity", "False-hit rate, LFU", "False-hit rate, with admission", "Precision, LFU",
                "Precision, with admission"], rows, star=False,
               placement="Sec. Eviction Policy Evaluation, after tab:admission_risk. NEW.",
               serves="R1.4, R1.2")


def t_density():
    e18 = load("exp18_eviction_m2.json")
    pols = (("LRU", "LRU"), ("LFU", "LFU"), ("SphereLFU", "SphereLFU"), ("SemanticDecay_fixed", "Semantic Decay, default"),
            ("SemanticDecay_tuned", "Semantic Decay, tuned"))
    rows = []
    for pol, nm in pols:
        s = e18["summary"][pol]["400"]
        never = np.mean([v["victim_never_hit_pct"] for v in e18["per_seed"][pol]["400"].values()
                         if v.get("victim_never_hit_pct") is not None])
        rows.append([nm, num(s["mean_victim_neighbors"], 3), pct(s["victim_topic_redundant_pct"]), pct(never)])
    return tab("tab:eviction_density",
               "Cache density at eviction time (capacity 400, C2 serving, mean over seeds 42--46): live entries within "
               "the hit radius of the victim, evictions whose topic remains served by another live entry, and victims "
               "that were never hit.",
               ["Policy", "Neighbours of victim", "Topic still covered", "Never hit"], rows, star=False, resize=False,
               placement="Sec. Eviction Policy Evaluation, after tab:admission_precision. NEW.",
               serves="R3.4")


def t_searchqueries():
    e22 = load("exp22_searchqueries_m2_n50000.json")["results"]
    s1 = load("exp25_searchqueries_retrain_m2_sweep_C1.json")["results"]
    s2 = load("exp25_searchqueries_retrain_m2_sweep_C2.json")["results"]
    rows = []
    for k in ("C1_t0.726", "C1_t0.8", "C1_t0.9"):
        v = e22[k]["stats"]
        rows.append([NAME["C1_bi_only"] if k == "C1_t0.726" else "", f"cosine $\\geq$ {k.split('_t')[1]}",
                     pct(v["hit_ratio_pct"]), pct(v["precision_of_hits_pct"])])
    for k in ("C1_t0.95", "C1_t0.98"):
        v = s1[k]["stats"]
        rows.append(["", f"cosine $\\geq$ {k.split('_t')[1]}", pct(v["hit_ratio_pct"]), pct(v["precision_of_hits_pct"])])
    rows.append(mid(4))
    for k in ("C2_p0.5", "C2_p0.9"):
        v = e22[k]["stats"]
        rows.append([r"C2, trained on QQP" if k == "C2_p0.5" else "", f"$p \\geq$ {k.split('_p')[1]}",
                     pct(v["hit_ratio_pct"]), pct(v["precision_of_hits_pct"])])
    for k in ("C2_adapted_p0.05", "C2_adapted_p0.1", "C2_adapted_p0.2"):
        v = s2[k]["stats"]
        rows.append([r"C2, trained in-domain" if k == "C2_adapted_p0.05" else "", f"$p \\geq$ {k.split('_p')[1]}",
                     pct(v["hit_ratio_pct"]), pct(v["precision_of_hits_pct"])])
    v = e22["C5_gated"]["stats"]
    rows.append([NAME["C5_gated"], "QQP models", pct(v["hit_ratio_pct"]), pct(v["precision_of_hits_pct"])])
    rows.append(mid(4))
    for d in ("0.01", "0.05"):
        v = e22[f"vcache_d{d}"]["stats"]
        rows.append(["vCache" if d == "0.01" else "", f"$\\delta={d}$", pct(v["hit_ratio_pct"]), pct(v["precision_of_hits_pct"])])
    return tab("tab:searchqueries",
               "Real query log: the first 50,000 queries of SemBenchmarkSearchQueries (ORCAS search queries; 24,192 "
               "equivalence groups; 51.6\\% of queries rephrase an earlier one and none repeats verbatim), served in log "
               "order. In-domain models are trained on later queries whose groups never occur in the evaluated prefix.",
               ["Configuration", "Operating point", r"Served-hit ratio ($\uparrow$)", r"Hit precision ($\uparrow$)"], rows,
               star=False,
               placement="NEW subsection 'Real Query Log' in Sec. Cross-Dataset Validation (after the STS tables). "
                         "Describe the dataset in Sec. Datasets and Preprocessing.",
               serves="R1.5, R1.1, R2.4")


def t_searchqueries_matched():
    e22 = load("exp22_searchqueries_m2_n50000.json")["results"]
    s1 = load("exp25_searchqueries_retrain_m2_sweep_C1.json")["results"]
    s2 = load("exp25_searchqueries_retrain_m2_sweep_C2.json")["results"]
    c1 = sorted([(v["stats"]["hit_ratio_pct"], v["stats"]["precision_of_hits_pct"]) for k, v in e22.items() if k.startswith("C1_")]
                + [(v["stats"]["hit_ratio_pct"], v["stats"]["precision_of_hits_pct"]) for v in s1.values()])
    H, P = [a for a, _ in c1], [b for _, b in c1]
    rows = []
    for k, v in sorted(s2.items(), key=lambda kv: kv[1]["stats"]["hit_ratio_pct"]):
        h, pr = v["stats"]["hit_ratio_pct"], v["stats"]["precision_of_hits_pct"]
        if not (min(H) <= h <= max(H)):
            continue
        ref = float(np.interp(h, H, P))
        rows.append([f"$p \\geq$ {k.split('_p')[1]}", pct(h), pct(pr), pct(ref), f"{pr - ref:+.1f}"])
    return tab("tab:searchqueries_matched",
               "In-domain C2 against C1 at the same served-hit ratio on the real query log; C1 precision is "
               "interpolated along its threshold sweep.",
               ["C2 operating point", "Served-hit ratio", "C2 precision", "C1 precision", r"$\Delta$ (pp)"], rows,
               star=False, resize=False,
               placement="NEW subsection 'Real Query Log', after tab:searchqueries.",
               serves="R1.1, R1.5")


def t_audit():
    d = load("exp16_answer_audit_m2_seed42.json")
    rows = []
    for k, v in d["per_config"].items():
        s = v["strata"]
        rows.append([NAME[k], v["hits"], v["verbatim_hits"], pct(v["label_precision_pct"]), pct(v["judge_precision_pct"]),
                     f"{num(s['label_correct']['substitutable_pct'])}\\% ({s['label_correct']['n_judged']})",
                     f"{num(s['label_false']['substitutable_pct'])}\\% ({s['label_false']['n_judged']})"])
    vals = [[None, None, None, None, v["judge_precision_pct"], None, None] for v in d["per_config"].values()]
    rows = bold_best(rows, [4], vals, [True], [1])
    a = d["agreement"]
    return tab("tab:answer_audit",
               "Answer-level audit of served hits (seed 42). An LLM judge reads the user's question, the cached answer "
               "and a freshly generated reference answer, and decides whether the cached answer is acceptable. Hits are "
               "sampled per label stratum (judged pairs in parentheses); verbatim repeats are counted as acceptable. "
               "Judge-based precision reweights the strata to each configuration's label mix. Agreement between judge "
               f"and labels: {a['agreement_pct']:.1f}\\% (Cohen's $\\kappa={a['cohen_kappa']:.2f}$, {a['n']} pairs).",
               ["Configuration", "Served hits", "Verbatim", r"Label precision", r"Judge precision ($\uparrow$)",
                "Acceptable, label-correct", "Acceptable, label-false"], rows,
               placement="NEW subsection 'Answer-Level Audit' in Results (after Comparison with Published Semantic "
                         "Caches). State the judge model, prompt and the 256-token answer cap in Sec. Evaluation Framework.",
               serves="R3.7, R3.5, R1.5")


def t_encoders():
    d = load("exp17_encoder_ann_geforcertx4090_seed42.json")
    rows, prev = [], None
    for r in d["encoders"]:
        enc = r["encoder"].split("/")[-1]
        if prev and enc != prev:
            rows.append(mid(6))
        rows.append([enc if enc != prev else "", NAME[r["config"]], f"{r['bi_threshold']:g}", pct(r["hit_ratio_pct"]),
                     pct(r["precision_of_hits_pct"]), num(r["avg_overhead_ms"], 2, " ms")])
        prev = enc
    return tab("tab:encoders",
               "Bi-encoder sensitivity (seed 42). Each encoder uses its own validation threshold; the C2 "
               "meta-classifier is refitted on that encoder's features.",
               ["Bi-encoder", "Configuration", "Threshold", r"Served-hit ratio ($\uparrow$)", r"Hit precision ($\uparrow$)",
                r"Decision overhead ($\downarrow$)"], rows, star=False,
               placement="NEW subsection 'Encoder and Index Sensitivity' in Results (after Answer-Level Audit).",
               serves="R3.5")


def t_index():
    d = load("exp17_encoder_ann_geforcertx4090_seed42.json")["index"]
    rows = [[NAME[r["config"]], "Exact (flat)" if r["index"] == "flat" else "HNSW", r["ef_search"] or "--",
             pct(r["hit_ratio_pct"], 2), pct(r["precision_of_hits_pct"], 2),
             num(r["stage_ms_per_request"]["vector_search"], 3, " ms")] for r in d]
    return tab("tab:index",
               "Exact against approximate nearest-neighbour search (HNSW, $M=32$) on an unbounded cache (seed 42).",
               ["Configuration", "Index", "efSearch", r"Served-hit ratio", r"Hit precision", r"Search time ($\downarrow$)"],
               rows, star=False,
               placement="NEW subsection 'Encoder and Index Sensitivity', after tab:encoders.",
               serves="R3.5")


TABLES = [t_end_to_end, t_seeds, t_paired, t_workload, t_latency, t_latency_dist, t_batching, t_risk, t_matched,
          t_best_lambda, t_topk, t_gate, t_baselines, t_baseline_timing, t_long_stream, t_audit, t_encoders, t_index,
          t_eviction, t_admission_risk, t_admission_precision, t_density, t_searchqueries, t_searchqueries_matched]


def main():
    header = ("% Result tables for the revised manuscript, generated by scripts/build_manuscript_tables.py\n"
              "% from results/raw/revision/*.json. Requires booktabs (already loaded by the manuscript).\n"
              "% Order follows the manuscript's Results section.\n\n")
    OUT.write_text(header + "\n".join(fn() for fn in TABLES), encoding="utf8")
    print("wrote", OUT, len(TABLES), "tables")


if __name__ == "__main__":
    main()
