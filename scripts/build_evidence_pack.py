"""
Build the revision evidence pack from the result files.

    python scripts/build_evidence_pack.py

Writes results/revision/COMMENT_MAP.md (one section per reviewer comment,
every number read from results/raw/revision/*.json) and
results/revision/figures/*.png. Re-running after new results regenerates
everything; nothing in the pack is typed by hand except the prose.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "revision"
OUT = ROOT / "results" / "revision"
FIG = OUT / "figures"
CFG = ("C1_bi_only", "C2_bi_geo_lgb", "C3_bi_cross", "C4_full_stacking", "C5_gated")
SHORT = {"C1_bi_only": "C1", "C2_bi_geo_lgb": "C2", "C3_bi_cross": "C3",
         "C4_full_stacking": "C4", "C5_gated": "C5"}


def load(name: str) -> dict:
    return json.load(open(RAW / name, encoding="utf8"))


def m(x):
    return x.get("mean") if isinstance(x, dict) else x


def f(x, nd=1, pct=False):
    x = m(x)
    if x is None:
        return "–"
    s = f"{x:.{nd}f}"
    return s + ("%" if pct else "")


def ci(x, nd=1):
    if not isinstance(x, dict) or "ci_low" not in x:
        return f(x, nd)
    return f"{x['mean']:.{nd}f} [{x['ci_low']:.{nd}f}, {x['ci_high']:.{nd}f}]"


def p(t: dict) -> str:
    v = t.get("t_p")
    return "–" if v is None else (f"{v:.1e}" if v < 1e-3 else f"{v:.3f}")


def table(head: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------- figures
def figures() -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    made = {}

    # F1 eviction: correct-hit ratio vs capacity
    e18, e23, e24 = load("exp18_eviction_m2.json"), load("exp23_eviction_headroom_m2.json"), load("exp24_sd_admit_cpu.json")
    caps = sorted(int(c) for c in e18["summary"]["LRU"])
    fig, ax = plt.subplots(figsize=(6.4, 4))
    series = [("Clairvoyant", [m(e23["summary"][str(c)]["Clairvoyant"]) for c in caps], "k--"),
              ("LFU + admission (SD-Admit)", [m(e24["summary"][str(c)]["SD-Admit"]) for c in caps], "C3-o")]
    for pol, st in (("LFU", "C0-s"), ("SphereLFU", "C2-^"), ("SemanticDecay_tuned", "C1-v"), ("LRU", "C7-x")):
        series.append((pol.replace("_tuned", " (tuned)"),
                       [m(e18["summary"][pol][str(c)]["correct_hit_ratio_pct"]) for c in caps], st))
    for lab, ys, st in series:
        ax.plot(caps, ys, st, label=lab, ms=4)
    ax.set_xlabel("cache capacity (entries)")
    ax.set_ylabel("correct-hit ratio (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "F1_eviction_capacity.png", dpi=160)
    plt.close(fig)
    made["F1"] = "figures/F1_eviction_capacity.png"

    # F2 risk-adjusted saving at lambda=10
    e26 = load("exp26_risk_priced_cpu.json")
    caps26 = sorted(int(c) for c in e26["summary"]["10"])
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for nm, st in (("LFU", "C0-s"), ("SphereLFU", "C2-^"), ("SD-Admit", "C3-o"), ("RiskPriced@10", "C4-d"), ("LRU", "C7-x")):
        ax.plot(caps26, [m(e26["summary"]["10"][str(c)][nm]["risk_adjusted_saving_pct"]) for c in caps26],
                st, label=nm.replace("SD-Admit", "LFU + admission"), ms=4)
    ax.set_xlabel("cache capacity (entries)")
    ax.set_ylabel("risk-adjusted saving at λ = 10 (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "F2_risk_adjusted_lambda10.png", dpi=160)
    plt.close(fig)
    made["F2"] = "figures/F2_risk_adjusted_lambda10.png"

    # F3 QQP operating curves
    e12 = load("exp12_threshold_risk_m2_seed42.json")["curves"]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for k, st in (("C1_bi_only", "C0-"), ("C2_bi_geo_lgb", "C3-"), ("C3_bi_cross", "C2-"), ("C4_full_stacking", "C1-"),
                  ("gptcache", "C7:"), ("vectorq", "C8:"), ("vcache", "C4-")):
        pts = sorted((q["hit_ratio_pct"], q["precision_of_hits_pct"]) for q in e12[k] if q["precision_of_hits_pct"])
        ax.plot([a for a, _ in pts], [b for _, b in pts], st, marker=".", label=SHORT.get(k, k))
    ax.set_xlabel("served-hit ratio (%)")
    ax.set_ylabel("precision of hits (%)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "F3_qqp_operating_curves.png", dpi=160)
    plt.close(fig)
    made["F3"] = "figures/F3_qqp_operating_curves.png"

    # F4 SearchQueries operating curves
    e22 = load("exp22_searchqueries_m2_n50000.json")["results"]
    s1 = load("exp25_searchqueries_retrain_m2_sweep_C1.json")["results"]
    s2 = load("exp25_searchqueries_retrain_m2_sweep_C2.json")["results"]
    c1 = sorted([(v["stats"]["hit_ratio_pct"], v["stats"]["precision_of_hits_pct"]) for k, v in e22.items() if k.startswith("C1_")]
                + [(v["stats"]["hit_ratio_pct"], v["stats"]["precision_of_hits_pct"]) for v in s1.values()])
    c2q = sorted((v["stats"]["hit_ratio_pct"], v["stats"]["precision_of_hits_pct"]) for k, v in e22.items() if k.startswith("C2_"))
    c2r = sorted((v["stats"]["hit_ratio_pct"], v["stats"]["precision_of_hits_pct"]) for v in s2.values())
    vc = [(v["stats"]["hit_ratio_pct"], v["stats"]["precision_of_hits_pct"]) for k, v in e22.items() if k.startswith("vcache")]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(*zip(*c1), "C0-o", ms=3, label="C1 (threshold sweep)")
    ax.plot(*zip(*c2q), "C3--s", ms=3, label="C2 trained on QQP")
    ax.plot(*zip(*c2r), "C3-^", ms=3, label="C2 retrained in-domain")
    ax.plot(*zip(*vc), "C4D", ms=5, label="vCache (δ = 0.01–0.05)")
    ax.set_xlabel("served-hit ratio (%)")
    ax.set_ylabel("precision of hits (%)")
    ax.set_title("SemBenchmarkSearchQueries, first 50,000 queries", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "F4_searchqueries_curves.png", dpi=160)
    plt.close(fig)
    made["F4"] = "figures/F4_searchqueries_curves.png"

    # F5 decision-path cost by stage
    e14 = load("exp14_latency_geforcertx4090.json")["decomposition"]
    keys = [k for k in CFG if k in e14]
    stages = sorted({s for k in keys for s in e14[k]["stages"]})
    fig, ax = plt.subplots(figsize=(6.4, 4))
    bottom = np.zeros(len(keys))
    for s in stages:
        v = np.array([e14[k]["stages"].get(s, {}).get("mean_ms", 0.0) for k in keys])
        if v.sum() < 0.05:
            continue
        ax.bar([SHORT[k] for k in keys], v, bottom=bottom, label=s)
        bottom += v
    ax.set_ylabel("mean decision cost per request (ms)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "F5_latency_stages.png", dpi=160)
    plt.close(fig)
    made["F5"] = "figures/F5_latency_stages.png"
    return made


# ---------------------------------------------------------------- sections
def s_platform() -> str:
    rows = []
    for fn in sorted(glob.glob(str(RAW / "*.json"))):
        name = Path(fn).name
        try:
            hw = json.load(open(fn))["manifest"].get("hardware", {})
        except Exception:
            continue
        rows.append([f"`{name}`", hw.get("cpu") or "–", hw.get("host_cpu") or "–", hw.get("device") or "–"])
    return table(["result file", "accelerator / device name", "host CPU", "torch device"], rows)


def s_labels() -> str:
    d = load("exp10_label_fix_geforcertx4090_seed42.json")["configs"]
    rows = []
    for k, v in d.items():
        a, b, c = v["shipped_labels"], v["corrected_labels"], v["fresh_distractor_stream"]
        rows.append([SHORT[k], f(a["precision_of_hits_pct"]), f(b["precision_of_hits_pct"]), f(c["precision_of_hits_pct"]),
                     f(a["token_saving_correct_only_pct"]), f(b["token_saving_correct_only_pct"]),
                     f(a["break_even_false_hit_cost"]), f(b["break_even_false_hit_cost"]), v["identical_decisions"]])
    return table(["config", "precision, shipped labels", "precision, corrected", "precision, fresh-distractor stream",
                  "correct-only saving, shipped", "corrected", "break-even λ, shipped", "corrected", "same decisions"], rows)


def s_r11() -> str:
    d = load("exp11_seeds_geforcertx4090_real.json")
    a = d["across_seeds"]
    rows = [[SHORT[k], ci(a[k]["hit_ratio_pct"]), ci(a[k]["precision_of_hits_pct"]),
             ci(a[k]["token_saving_correct_only_pct"]), f(a[k]["avg_overhead_ms"], 2),
             f(a[k]["avg_response_time_s"], 3), f(a[k]["break_even_false_hit_cost"])] for k in CFG if k in a]
    t1 = table(["config", "hit ratio % [95% CI]", "precision %", "correct-only saving %", "overhead ms",
                "response s", "break-even λ"], rows)
    c = d["c1_vs_c2_across_seeds"]
    t2 = table(["metric (C1 − C2)", "Δ mean", "95% CI", "paired t p"],
               [[k, f(v.get("delta_mean"), 2), f"[{v.get('delta_ci_low', 0):.2f}, {v.get('delta_ci_high', 0):.2f}]", p(v)]
                for k, v in c.items() if k in ("hit_ratio_pct", "precision_of_hits_pct", "false_discovery_rate_pct",
                                               "token_saving_correct_only_pct", "avg_overhead_ms", "break_even_false_hit_cost")])
    e27 = json.load(open(sorted(glob.glob(str(RAW / "exp27_operating_curves_seeds_*.json")))[-1]))
    mrow = []
    for h, row in e27["matched"].items():
        if row["C1_bi_only"]["n_seeds_in_range"] < 5 or sum(row[k]["n_seeds_in_range"] == 5 for k in CFG[:4]) < 2:
            continue
        mrow.append([h, *[(f"{row[k]['precision_of_hits_pct']['mean']:.2f} ± {row[k]['precision_of_hits_pct']['sd']:.2f}"
                           if row[k]["n_seeds_in_range"] == 5 else "–") for k in CFG[:4]]])
    t4 = table(["served-hit ratio %", *[SHORT[k] for k in CFG[:4]]], mrow)
    tt = table(["test (config − C1)", "Δ", "95% CI", "p", "Holm p"],
               [[k, f(v.get("delta_mean"), 2), f"[{v.get('delta_ci_low', 0):.2f}, {v.get('delta_ci_high', 0):.2f}]",
                 p(v), f(v.get("holm_p"), 3)] for k, v in e27["tests"].items()])
    g = e27["default_vs_tuned"]
    tg = table(["λ", *[f"{SHORT[k]} default → tuned" for k in CFG[:4]]],
               [[lam, *[f"{g[lam][k]['default']['mean']:.1f} → {g[lam][k]['tuned']['mean']:.1f}" for k in CFG[:4]]]
                for lam in ("0", "1", "2", "5", "10")])
    t1 = t1 + ("\n\nPrecision (%) at the same served-hit ratio, mean ± sd over seeds 42–46, from full threshold sweeps "
               "on an extended grid (exp12 per seed, aggregated by exp27). All four configurations lie on one "
               "precision–coverage curve (Figure F3):\n\n" + t4 +
               "\n\nPaired tests against C1 (matched precision and best risk-adjusted saving per λ), Holm-adjusted "
               "across all of them:\n\n" + tt +
               "\n\nRisk-adjusted saving (%) at each configuration's default threshold and at the threshold chosen "
               "for λ (mean of 5 seeds):\n\n" + tg)
    return t1 + "\n\nC1 vs C2 at default thresholds, seed-level paired test (5 seeds):\n\n" + t2


def s_r12() -> str:
    e2 = load("exp2_end_to_end_geforcertx4090_seed42_real.json")
    rows = []
    for r in e2["configs"]:
        ras = r["risk_adjusted_saving_pct"]
        rows.append([SHORT[r["config"]], f(r["false_discovery_rate_pct"]), f(r["false_hit_rate_pct"], 2),
                     *[f(ras[l]) for l in ("0", "1", "2", "5", "10")], f(r["break_even_false_hit_cost"])])
    return table(["config", "FDR %", "false-hit rate %", "RAS λ=0", "λ=1", "λ=2", "λ=5", "λ=10", "break-even λ"], rows)


def s_r13() -> str:
    e14 = load("exp14_latency_geforcertx4090.json")
    d = e14["decomposition"]
    t1 = table(["config", "mean ms", "p50", "p95", "p99"],
               [[SHORT[k], f(d[k]["total"]["mean_ms"], 2), f(d[k]["total"]["p50_ms"], 2), f(d[k]["total"]["p95_ms"], 2),
                 f(d[k]["total"]["p99_ms"], 2)] for k in CFG if k in d])
    cb = e14["cross_batching"]
    t2 = table(["cross-encoder batch", *cb.keys()], [["ms per pair", *[f(v["ms_per_pair"], 2) for v in cb.values()]]])
    tk = e14["topk_cost"]
    t3 = table(["config", *[f"k={k}" for k in next(iter(tk.values()))]],
               [[SHORT[c], *[f(v["mean_ms"], 1) for v in tk[c].values()]] for c in tk])
    cc = e14["concurrency"]
    t4 = table(["config", "clients", "mean ms", "p95", "p99", "throughput rps"],
               [[SHORT[c], n, f(v["mean_ms"], 1), f(v["p95_ms"], 1), f(v["p99_ms"], 1), f(v["throughput_rps"], 0)]
                for c in cc for n, v in cc[c].items()])
    e2 = load("exp2_end_to_end_geforcertx4090_seed42_real.json")
    t5 = table(["config", "hit %", "precision %", "overhead ms", "avg response s", "p95 s", "speed-up vs no cache", "identity residual ms"],
               [[SHORT[r["config"]], f(r["hit_ratio_pct"]), f(r["precision_of_hits_pct"]), f(r["avg_overhead_ms"], 2),
                 f(r["avg_response_time_s"], 3), f(r["p95_response_time_s"], 2), f(r["speedup_vs_no_cache"], 2),
                 f(r["consistency"]["residual_ms"], 1)] for r in e2["configs"]])
    nc = e2["no_cache"]
    return ("Decision-path cost (exp14, batch 1, 300 queries):\n\n" + t1 + "\n\nCross-encoder batching:\n\n" + t2 +
            "\n\nTop-k cost (mean ms per request):\n\n" + t3 + "\n\nConcurrent clients:\n\n" + t4 +
            f"\n\nEnd-to-end with real generation waits (exp2, 2,000 requests, seed 42; no cache: "
            f"{nc['avg_response_time_s']:.3f} s average):\n\n" + t5)


def s_r23() -> str:
    d = load("exp1_latency_breakdown_geforcertx4090.json")["per_config"]
    return table(["config", "bi-encode ms", "vector search ms", "cross-encode ms", "features + meta ms", "total ms",
                  "× C1", "cross-encoder share %"],
                 [[SHORT.get(r["config"], r["config"]), f(r["bi_encode_ms"], 2), f(r["vector_search_ms"], 2),
                   f(r["cross_encode_ms"], 2), f(r["features_meta_ms"], 2), f(r["total_overhead_ms"], 2),
                   f(r["overhead_vs_bi_only_x"], 2), f(r["cross_encoder_share_pct"])] for r in d])


def s_eviction() -> str:
    e18, e23, e24, e26 = (load("exp18_eviction_m2.json"), load("exp23_eviction_headroom_m2.json"),
                          load("exp24_sd_admit_cpu.json"), load("exp26_risk_priced_cpu.json"))
    caps = ["200", "300", "400", "500", "600", "700", "800", "900", "1000"]
    pols = ["LRU", "LFU", "SphereLFU", "SemanticDecay_fixed", "SemanticDecay_tuned", "SemanticDecay2_tuned"]
    rows = [["Clairvoyant (exp23)", *[f(e23["summary"][c]["Clairvoyant"], 2) for c in caps]]]
    rows += [[pol, *[f(e18["summary"][pol][c]["correct_hit_ratio_pct"], 2) for c in caps]] for pol in pols]
    rows += [[v + " (exp24)", *[f(e24["summary"][c][v], 2) for c in caps]]
             for v in ("SD-Admit", "SD-Admit_no_admission", "SD-Admit_no_demand")]
    t1 = table(["policy", *caps], rows)
    t2 = table(["", *caps],
               [["headroom over best online (pts)", *[f(e23["headroom"][c]["headroom_over_best_online"].get("delta_mean"), 2) for c in caps]],
                ["best online policy", *[e23["headroom"][c]["best_online_policy"] for c in caps]],
                ["SD-Admit share of headroom captured (%)", *[f(e24["summary"][c].get("headroom_captured_pct"), 0) for c in caps]]])
    t3 = table(["SD-Admit vs", *caps],
               [[ref, *[f"{e24['tests'][c]['SD-Admit'][ref].get('delta_mean', 0):+.2f} (p {p(e24['tests'][c]['SD-Admit'][ref])})" for c in caps]]
                for ref in ("LFU", "SphereLFU", "SemanticDecay_tuned")])
    t4 = table(["Semantic Decay (tuned) vs", *caps],
               [[ref, *[f"{e18['tests']['SemanticDecay_tuned'][ref][c].get('delta_mean', 0):+.2f} (p {p(e18['tests']['SemanticDecay_tuned'][ref][c])})" for c in caps]]
                for ref in ("LRU", "LFU")])
    dens = table(["policy", "victim neighbours within hit radius", "victim topic still covered %", "victim never hit %"],
                 [[pol, *[f(e18["summary"][pol]["400"].get(k), 3) for k in ("mean_victim_neighbors", "victim_topic_redundant_pct")],
                   f(float(np.mean([v.get("victim_never_hit_pct") for v in e18["per_seed"][pol]["400"].values()
                                    if v.get("victim_never_hit_pct") is not None])), 1)] for pol in pols])
    tun = e24["tuning"]["best"]
    lam_rows = []
    for L in ("1", "2", "5", "10"):
        for c in ("200", "400", "800"):
            s = e26["summary"][L][c]
            lam_rows.append([L, c, *[f(s[n]["risk_adjusted_saving_pct"], 2) for n in ("LFU", "SphereLFU", "SD-Admit", f"RiskPriced@{L}")],
                             f(s["LFU"]["false_hit_rate_pct"], 2), f(s["SD-Admit"]["false_hit_rate_pct"], 2),
                             f(s["LFU"]["precision_of_hits_pct"], 1), f(s["SD-Admit"]["precision_of_hits_pct"], 1)])
    t5 = table(["λ", "capacity", "RAS LFU", "SphereLFU", "LFU + admission", "RiskPriced", "false-hit % LFU",
                "false-hit % LFU + admission", "precision LFU", "precision LFU + admission"], lam_rows)
    return ("Correct-hit ratio (%) by capacity, C2 serving, mean of seeds 42–46:\n\n" + t1 +
            "\n\nHeadroom and what admission captures:\n\n" + t2 +
            f"\n\nSD-Admit as frozen from the tuning stream: β = {tun['beta']}, radius {tun['cos_radius']}, aging every "
            f"{tun['period_mult']}× capacity requests, admission {tun['admission']} (β = 0 switches neighbourhood demand off; "
            "all radii tie). Paired tests on correct-hit ratio (5 seeds):\n\n" + t3 +
            "\n\nSemantic Decay (tuned weights) vs LRU and LFU:\n\n" + t4 +
            "\n\nEviction-time density at capacity 400 (exp18):\n\n" + dens +
            "\n\nPricing false hits (exp26): risk-adjusted saving (RAS, %) and false-hit rates, mean of 5 seeds:\n\n" + t5)


def s_r15() -> str:
    d = load("exp15_workload_sensitivity_geforcertx4090.json")["results"]
    rows = [[r["factor"], r["value"], *[f"{f(r['summary'][k]['hit_ratio_pct'])} / {f(r['summary'][k]['precision_of_hits_pct'])}"
                                        for k in CFG[:4]]] for r in d]
    t1 = table(["factor", "value", *[f"{SHORT[k]} hit / prec %" for k in CFG[:4]]], rows)
    e20 = load("exp20_long_stream_m2_seed42.json")["results"]
    t2 = table(["system", "hit %", "precision %", "hit % by 1,000-request block"],
               [[k, f(v["stats"]["hit_ratio_pct"]), f(v["stats"]["precision_of_hits_pct"]),
                 " ".join(f"{b['hit_ratio_pct']:.0f}" for b in v["blocks"])] for k, v in e20.items()])
    e22 = load("exp22_searchqueries_m2_n50000.json")
    t3 = table(["system", "hit %", "precision %", "hit % by 5,000-query block"],
               [[k, f(v["stats"]["hit_ratio_pct"]), f(v["stats"]["precision_of_hits_pct"]),
                 " ".join(f"{b['hit_ratio_pct']:.0f}" for b in v["blocks"])] for k, v in e22["results"].items()])
    w = e22["workload"]
    return ("Workload sensitivity (3 seeds per value, one factor at a time):\n\n" + t1 +
            "\n\nLong QQP stream (10,000 requests, seed 42, exp20):\n\n" + t2 +
            f"\n\nReal query log: SemBenchmarkSearchQueries, first {w['n_events']:,} queries "
            f"({w['n_topics']:,} equivalence groups, {w['repeat_rate_pct']:.1f}% rephrased repeats, no exact repeats), "
            "QQP-tuned thresholds and models (exp22):\n\n" + t3)


def s_domain() -> str:
    e25 = load("exp25_searchqueries_retrain_geforcertx3060_n50000.json")
    tr = e25["manifest"]["training"]
    e22 = load("exp22_searchqueries_m2_n50000.json")["results"]
    s1 = load("exp25_searchqueries_retrain_m2_sweep_C1.json")["results"]
    s2 = load("exp25_searchqueries_retrain_m2_sweep_C2.json")["results"]
    c1 = sorted([(v["stats"]["hit_ratio_pct"], v["stats"]["precision_of_hits_pct"]) for k, v in e22.items() if k.startswith("C1_")]
                + [(v["stats"]["hit_ratio_pct"], v["stats"]["precision_of_hits_pct"]) for v in s1.values()])
    H, P = [a for a, _ in c1], [b for _, b in c1]
    rows = []
    for k, v in sorted(s2.items(), key=lambda kv: kv[1]["stats"]["hit_ratio_pct"]):
        h, pr = v["stats"]["hit_ratio_pct"], v["stats"]["precision_of_hits_pct"]
        if min(H) <= h <= max(H):
            ref = float(np.interp(h, H, P))
            rows.append([k, f(h), f(pr), f(ref), f"{pr - ref:+.1f}"])
        else:
            rows.append([k, f(h), f(pr), "outside C1 range", "–"])
    other = table(["system (retrained in-domain, exp25)", "hit %", "precision %"],
                  [[k, f(v["stats"]["hit_ratio_pct"]), f(v["stats"]["precision_of_hits_pct"])] for k, v in e25["results"].items()])
    note = ("\n\nThe server run (RTX 3060, table above) and the local sweep (Apple M2, table below) use the same "
            "training pairs; their bi-encoder embeddings were computed on different hardware, so the same operating "
            "point can differ by up to about 0.4 points.")
    return (f"Training data: {tr['n_pairs']:,} candidate pairs from queries whose groups never occur in the test stream "
            f"({tr['positive_pct']:.1f}% equivalent). C1 threshold re-tuned on them: {tr['c1_threshold']} "
            f"(training F1 {tr['c1_train_f1']:.3f}).\n\n" + other + note +
            "\n\nRetrained C2 against C1 at the same served-hit ratio (C1 precision interpolated on its threshold curve):\n\n" +
            table(["C2 operating point", "hit %", "C2 precision %", "C1 precision %", "Δ"], rows))


def s_r31() -> str:
    d = load("exp13_topk_gate_geforcertx4090_seed42.json")
    t1 = table(["config", "k", "hit %", "precision %", "overhead ms"],
               [[SHORT[r["config"]], r["k"], f(r["hit_ratio_pct"]), f(r["precision_of_hits_pct"]), f(r["avg_overhead_ms"], 2)]
                for r in d["topk"]])
    t2 = table(["gate band (out-of-fold share)", "hit %", "precision %", "cross-encoder calls / request", "overhead ms"],
               [[f"{g['band']['share_target']:.2f} ({g['band']['lo']:.2f}–{g['band']['hi']:.2f})", f(g["hit_ratio_pct"]),
                 f(g["precision_of_hits_pct"]), f(g["cross_calls_per_request"], 3), f(g["avg_overhead_ms"], 2)] for g in d["gating"]])
    t3 = table(["abstention margin", "hit %", "precision %", "abstained %"],
               [[a["margin"], f(a["hit_ratio_pct"]), f(a["precision_of_hits_pct"]), f(a["abstained_pct"], 2)] for a in d["abstention"]])
    c2 = d["c2_reference"]
    return (t1 + "\n\nSelective cross-encoder gate (C5), bands chosen on out-of-fold probabilities; C2 reference: "
            f"hit {c2['hit_ratio_pct']:.1f}%, precision {c2['precision_of_hits_pct']:.1f}%. Empty band reproduces C2: "
            f"{d['gate_sanity']['reproduces_c2']}.\n\n" + t2 + "\n\nAbstention near the decision threshold:\n\n" + t3)


def s_baselines() -> str:
    d = load("exp19_baselines_seeds_m2.json")
    a, t = d["across_seeds"], d["paired_vs_C"]
    rows = [[k, ci(a[k]["hit_ratio_pct"]), ci(a[k]["precision_of_hits_pct"]), f(a[k]["token_saving_correct_only_pct"]),
             f(a[k]["break_even_false_hit_cost"]), f(a[k]["judge_calls_per_request"], 2),
             f"{t['C2_bi_geo_lgb'][k]['precision_of_hits_pct'].get('delta_mean', 0):+.2f} (p {p(t['C2_bi_geo_lgb'][k]['precision_of_hits_pct'])})"]
            for k in a]
    t1 = table(["system", "hit %", "precision %", "correct-only saving %", "break-even λ", "judge calls / request",
                "Δ precision vs C2"], rows)
    timing = []
    for fn in sorted(glob.glob(str(RAW / "exp21_baseline_timing_*.json"))):
        s = json.load(open(fn))["stats"]
        timing.append([Path(fn).stem.split("seed42_")[-1], f(s["hit_ratio_pct"]), f(s["precision_of_hits_pct"]),
                       f(s["avg_overhead_ms"], 2), f(s["avg_response_time_s"], 3), s["consistency"]["consistent"]])
    hc = load("exp14_latency_geforcertx4090_host_check.json")["decomposition"]
    t2 = table(["system", "hit %", "precision %", "overhead ms", "avg response s", "identity holds"], timing)
    t3 = table(["config (same host, exp14 reference)", "mean decision ms"],
               [[SHORT[k], f(hc[k]["total"]["mean_ms"], 2)] for k in CFG if k in hc])
    e12 = load("exp12_threshold_risk_m2_seed42.json")["matched"]["at_hit_ratio"]
    t4 = table(["served-hit ratio", *e12[next(iter(e12))].keys()],
               [[h, *[f(v["precision_of_hits_pct"]) for v in row.values()]] for h, row in e12.items()])
    return ("Baselines on the five exp11 seeds (threshold baselines at the C1 validation threshold; vCache native insertion, "
            "oracle-judge feedback) (exp19):\n\n" + t1 +
            "\n\nPrecision at matched served-hit ratio, threshold sweeps (exp12, seed 42):\n\n" + t4 +
            "\n\nTiming with real waits (exp21, seed 42):\n\n" + t2 + "\n\n" + t3)


def s_r35() -> str:
    d = load("exp17_encoder_ann_geforcertx4090_seed42.json")
    t1 = table(["encoder", "config", "threshold", "hit %", "precision %", "overhead ms"],
               [[r["encoder"].split("/")[-1], SHORT[r["config"]], r["bi_threshold"], f(r["hit_ratio_pct"]),
                 f(r["precision_of_hits_pct"]), f(r["avg_overhead_ms"], 2)] for r in d["encoders"]])
    t2 = table(["index", "efSearch", "config", "hit %", "precision %", "search ms"],
               [[r["index"], r["ef_search"] or "–", SHORT[r["config"]], f(r["hit_ratio_pct"], 2),
                 f(r["precision_of_hits_pct"], 2), f(r["stage_ms_per_request"]["vector_search"], 3)] for r in d["index"]])
    return t1 + "\n\n" + t2


def s_r37() -> str:
    d = load("exp16_answer_audit_m2_seed42.json")
    t1 = table(["config", "hits", "verbatim", "label precision %", "judge precision %",
                "judged substitutable: label-correct", "label-false"],
               [[SHORT[k], v["hits"], v["verbatim_hits"], f(v["label_precision_pct"]), f(v["judge_precision_pct"]),
                 f"{f(v['strata']['label_correct']['substitutable_pct'])} (n={v['strata']['label_correct']['n_judged']})",
                 f"{f(v['strata']['label_false']['substitutable_pct'])} (n={v['strata']['label_false']['n_judged']})"]
                for k, v in d["per_config"].items()])
    a = d["agreement"]
    cal = load("exp16_rejudge_m2_seed42_calib_v2.json")
    return (t1 + f"\n\nJudge–label agreement on {a['n']} judged pairs: {a['agreement_pct']:.1f}%, Cohen's κ = "
            f"{a['cohen_kappa']:.2f}; label-false but substitutable: {a['label_false_but_substitutable']}; "
            f"label-correct but not substitutable: {a['label_correct_but_not_substitutable']}. Judge: "
            f"{d['manifest'].get('judge_model')}, prompt v{d['manifest'].get('judge_prompt_version', 2)}; "
            f"judge tokens {d['judge_tokens']['prompt']:,} in / {d['judge_tokens']['completion']:,} out.\n\n"
            f"Prompt calibration on 30 pairs: agreement {cal['agreement_previous_prompt']['agreement_pct']:.1f}% → "
            f"{cal['agreement']['agreement_pct']:.1f}%, κ {cal['agreement_previous_prompt']['cohen_kappa']:.2f} → "
            f"{cal['agreement']['cohen_kappa']:.2f} after telling the judge that both answers share a 256-token cap "
            "and that the reference is a guide, not a checklist.")


# ---------------------------------------------------------------- document
def main() -> None:
    figs = figures()
    doc = f"""# Revision evidence pack

Every table below is generated by `scripts/build_evidence_pack.py` from the result files in
`results/raw/revision/`. Re-run the script after any new result. Figures are in `results/revision/figures/`.

**Protocol common to all runs.** Labels use the corrected stream generator (one stable label per distinct
question; distractors drawn from never-emitted strings). QQP streams: 2,000 requests, seeds 42–46 for
evaluation, seed 1042 reserved for tuning. Served LLM gpt-5.4-nano, answers capped at 256 tokens. Response
times come from replay with real generation waits; quality-only sweeps use memoised encoders and report no
latency. C2 serving uses single-threaded LightGBM inference and batched candidate scoring.

**Status by comment**

| Comment | Topic | Evidence | Status |
|---|---|---|---|
| R1.1 | Why C2, not C1 | exp11, exp12, exp16, exp22, exp25 | evidence complete; framing decision needed |
| R1.2 | Price of false hits | exp2, exp11, exp12, exp13 | complete |
| R1.3 | Latency generality | exp1, exp2, exp14, exp21 | complete |
| R1.4, R3.2, R3.4 | Semantic Decay | exp18, exp23, exp24, exp26 | evidence complete; framing decision needed |
| R1.5 | Synthetic workload | exp15, exp20, exp22, exp25 | complete |
| R2.1 | Related work | literature checks (§9) | writing |
| R2.2 | Figure 1 | – | writing |
| R2.3 | Full-mode overhead | exp1, exp14 | complete |
| R2.4 | 2026 baselines | exp12, exp19, exp20, exp21, exp22 | complete (RAC optional) |
| R2.5 | Language | – | writing |
| R3.1 | Top-k and gating | exp13, exp14 | complete |
| R3.3 | Baseline protocol | exp12, exp19 | complete |
| R3.5 | Additional experiments | exp13, exp16, exp17, exp19 | complete |
| R3.7 | Correctness beyond labels | exp16 | complete |

## 0. Corrections that change reported numbers (disclose)

Stream labels: shipped vs corrected labels on the same decisions, and the fresh-distractor stream (exp10, seed 42):

{s_labels()}

Other disclosures: answers are capped at 256 tokens (both cached and fresh answers; token savings are
computed on capped answers); 88 of 3,586 answers for the 10,000-request stream failed to fetch and fall back to
simulated answers (hit decisions unaffected); SD-Admit's tie rule and slow-aging setting were added after an
inspection run on evaluation seed 42 (final settings chosen on the tuning stream only).

## 1. R1.1 — Why C2 rather than C1

Five seeds, real generation waits (exp11):

{s_r11()}

Summary of the evidence: at their default thresholds C2 is significantly more precise than C1 (and has the
highest answer-level precision in the judge audit, §7), but at equal served-hit ratio the configurations are
indistinguishable on QQP across five seeds, no configuration beats C1 once each threshold is chosen for λ (no
difference survives Holm adjustment), threshold choice is worth up to 35 points at λ = 10, and on a real
search-query log C2 never beats a re-thresholded C1, even after in-domain retraining (§5). What C2 adds is a
well-placed default operating point, not a better decision boundary.

## 2. R1.2 — Pricing false hits

Seed 42, real waits (exp2). RAS = risk-adjusted saving, where each false hit costs λ times the call it replaced:

{s_r12()}

## 3. R1.3 / R2.3 — Latency and its components

{s_r13()}

![decision cost by stage]({figs['F5']})

Per-component breakdown (exp1, batch 1):

{s_r23()}

## 4. R1.4 / R3.2 / R3.4 — Eviction and admission

{s_eviction()}

![eviction by capacity]({figs['F1']})

![risk-adjusted saving at λ = 10]({figs['F2']})

## 5. R1.5 — Workload realism, long streams and a real log

{s_r15()}

Decision layer retrained on the real log (exp25):

{s_domain()}

![SearchQueries operating curves]({figs['F4']})

## 6. R2.4 / R3.3 — Baselines, including 2026 methods

{s_baselines()}

![QQP operating curves]({figs['F3']})

## 7. R3.7 — Answer substitutability beyond labels

{s_r37()}

## 8. R3.1 / R3.5 — Top-k, gating, encoder and index sensitivity

{s_r31()}

Encoders and approximate search (exp17, seed 42):

{s_r35()}

## 9. R2.1 — Literature identified for related work

| Work | Venue / date | Role |
|---|---|---|
| vCache (Schroeder et al.) | ICLR 2026; arXiv 2502.03771 | implemented baseline |
| SphereLFU (Biton & Friedman) | arXiv 2603.03301, Feb 2026 | implemented eviction baseline |
| Which Eviction Policy Should an LLM Cache Use? (Kulkarni et al.) | arXiv 2608.20280, Aug 2026 | eviction finding (LFU hard to beat); cite |
| RAC: Relation-Aware Cache Replacement (Wu et al.) | arXiv 2602.21547, Feb 2026 | eviction; cite or add as baseline |
| Continuous Semantic Caching (Atalar et al.) | arXiv 2604.20021, Apr 2026 | cost-learned content selection; cite |
| SOLAR (Sun, Cao & Lam) | arXiv 2607.00394, Jul 2026 | admission for agent memory; cite |
| TinyLFU (Einziger, Friedman & Manes) | ACM TOS 2017 | admission mechanism; cite |
| MVR-cache (Noshad et al.) | ICML 2026; arXiv 2605.24914 | cite (reduces to vCache on short queries) |
| LaCache (Liang et al.) | arXiv 2608.01718, Aug 2026 | cite (LLM decode per candidate hit) |
| Krites (Singh et al.) | arXiv 2602.13165, Feb 2026 | cite (needs a static tier) |
| SemBenchmarkSearchQueries (vCache project) | Hugging Face, Apache-2.0 | real-log evaluation data |

## 10. Platforms

Rented servers (Clore.ai) and the files each produced. Two servers share the Ryzen 9 3900X, so the server is
named here: timings are compared only within one server.

| server | hardware | files |
|---|---|---|
| #110175 | RTX 4090, Intel i7-12700KF | exp10, exp15 |
| #90937 | RTX 4090, AMD Ryzen 9 3900X | exp1, exp2 (real waits), exp11 (real waits), exp13, exp14, exp17 |
| #118280 | RTX 4090, AMD Ryzen 9 3900X | exp21 (baseline timing), exp14 host check |
| #119906 | RTX 3060, Intel i5-11600K | exp24 (CPU), exp25 (server run) |
| #113590 | RTX 5060 Ti, AMD Ryzen 9 5950X | exp25 sweeps (replicate), exp26 (CPU) |
| local | Apple M2 | exp12, exp16, exp18, exp19, exp20, exp22, exp23, exp25 sweeps |

Quality-only experiments (marked latency-invalid in their manifests) give the same decisions on any host up to
embedding round-off; exp24 and exp26 reuse embeddings recorded on the M2 so that their decisions match the
M2 runs of exp18 and exp23 exactly.

{s_platform()}
"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "COMMENT_MAP.md").write_text(doc, encoding="utf8")
    print("wrote", OUT / "COMMENT_MAP.md", "and", len(figs), "figures")


if __name__ == "__main__":
    main()
