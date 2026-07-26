"""
Table/figure emission and — most importantly — the arithmetic consistency
checker.

`check_run_consistency` re-derives the average response time from the same
run's overhead, miss ratio and generation latency. If a set of numbers cannot
satisfy

    avg_response_time  ==  avg_overhead + miss_ratio * avg_generation_latency

then those numbers did not come from one experiment, and the checker says so.
Running it over the manuscript's published values is what turns "these numbers
cannot both be correct" from a reviewer's assertion into a quantity:
`audit_manuscript_claims` reports the smallest average response time that a
539.80 ms overhead could possibly produce.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from . import config as C


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
def _default(o: Any):
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def save_json(obj: Any, name: str, directory: Path | None = None) -> Path:
    directory = directory or C.RAW_DIR
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / (name if name.endswith(".json") else f"{name}.json")
    p.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf8")
    print(f"  wrote {p}")
    return p


def load_json(name: str, directory: Path | None = None) -> Any:
    directory = directory or C.RAW_DIR
    p = directory / (name if name.endswith(".json") else f"{name}.json")
    return json.loads(p.read_text(encoding="utf8"))


def save_latex(body: str, name: str) -> Path:
    p = C.TABLES_DIR / (name if name.endswith(".tex") else f"{name}.tex")
    p.write_text(body, encoding="utf8")
    print(f"  wrote {p}")
    return p


def ms(x: float | None, nd: int = 2) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) \
        else f"{x:.{nd}f} ms"


def pm(mean: float, std: float, nd: int = 4) -> str:
    return f"{mean:.{nd}f} $\\pm$ {std:.{nd}f}"


def esc(s: str) -> str:
    return (str(s).replace("_", r"\_").replace("&", r"\&")
            .replace("%", r"\%").replace("#", r"\#"))


# --------------------------------------------------------------------------
# Consistency checking
# --------------------------------------------------------------------------
def check_run_consistency(stats: dict, tol_ms: float = 25.0) -> dict:
    """Verify avg_response = avg_overhead + miss_ratio * avg_generation."""
    n = stats["n_requests"]
    miss_ratio = stats["misses"] / max(1, n)
    avg_overhead_s = stats["avg_overhead_ms"] / 1000.0
    # Average generation latency, recovered from the miss-path latencies.
    avg_miss = stats.get("avg_miss_latency_s", 0.0)
    avg_gen = max(0.0, avg_miss - avg_overhead_s)
    predicted = avg_overhead_s + miss_ratio * avg_gen
    actual = stats["avg_response_time_s"]
    resid_ms = (actual - predicted) * 1000.0
    return {
        "n_requests": n,
        "miss_ratio": miss_ratio,
        "avg_overhead_ms": stats["avg_overhead_ms"],
        "avg_generation_latency_s": avg_gen,
        "predicted_avg_response_s": predicted,
        "measured_avg_response_s": actual,
        "residual_ms": resid_ms,
        "consistent": bool(abs(resid_ms) <= tol_ms),
        "tolerance_ms": tol_ms,
    }


def min_possible_avg_response(overhead_ms: float, hit_ratio_pct: float,
                              gen_latency_s: float) -> dict:
    """Lower bound on average response time given a per-request overhead.

    The overhead is paid on *every* request, so the average can never fall
    below it. This is the arithmetic the reviewer applied by hand.
    """
    miss = 1.0 - hit_ratio_pct / 100.0
    lower = overhead_ms / 1000.0
    expected = lower + miss * gen_latency_s
    return {
        "overhead_ms": overhead_ms,
        "hit_ratio_pct": hit_ratio_pct,
        "assumed_generation_latency_s": gen_latency_s,
        "hard_lower_bound_s": lower,
        "expected_avg_response_s": expected,
    }


def audit_manuscript_claims(measured_gen_latency_s: float = 0.854) -> dict:
    """Apply the checker to the numbers as published, so the rebuttal can
    state the size of the contradiction rather than merely acknowledging it."""
    published = {
        "full_stacking_overhead_ms": 539.80,
        "cross_encoder_alone_ms": 520.76,
        "bi_encoder_only_overhead_ms": 13.44,
        "reported_avg_response_with_cache_s": 0.311,
        "reported_avg_response_no_cache_s": 0.854,
        "reported_hit_ratio_pct": 93.3,
        "reported_speedup": 2.7,
    }
    full = min_possible_avg_response(539.80, 93.3, measured_gen_latency_s)
    bi = min_possible_avg_response(13.44, 93.3, measured_gen_latency_s)
    return {
        "published": published,
        "if_full_stacking_was_used": full,
        "if_bi_encoder_only_was_used": bi,
        "verdict": (
            "The published 0.311 s average is below the published 539.80 ms "
            "per-request overhead of the full stacking configuration, and "
            "below the 520.76 ms cross-encoder term alone. No hit ratio can "
            "reconcile them: overhead is paid on every request, so 0.5398 s "
            "is a hard floor for that configuration. The 0.311 s figure is "
            "only attainable in a configuration whose overhead is small "
            "relative to the generation latency, i.e. bi-encoder-only "
            "(13.44 ms) or bi-encoder + cheap features. The response-time "
            "and token-savings claims must therefore be relabelled to the "
            "configuration that actually produced them."),
        "recommended_relabelling": {
            "figure_response_time": "C1_bi_only / C2_bi_geo_lgb",
            "figure_token_savings": "same configuration as response time",
            "table_latency_breakdown": "all four configurations, batch size 1",
        },
    }


# --------------------------------------------------------------------------
# LaTeX tables
# --------------------------------------------------------------------------
def table_latency_breakdown(rows: Sequence[dict], hardware: dict,
                            regime: str) -> str:
    head = (
        "\\begin{table*}[htbp]\n\\centering\n"
        "\\caption{Per-request overhead breakdown on QQP, measured at batch "
        "size 1 (online serving regime) on %s. Overhead is paid on every "
        "request, hits and misses alike, because the similarity decision "
        "precedes the outcome. LLM generation time is excluded.}\n"
        "\\label{tab:latency_breakdown}\n"
        "\\resizebox{\\textwidth}{!}{\n"
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        "\\textbf{Configuration} & \\textbf{Bi-Encoder} & \\textbf{FAISS "
        "Search} & \\textbf{Cross-Encoder} & \\textbf{Features + LightGBM} & "
        "\\textbf{Total Overhead} \\\\\n\\midrule\n"
        % esc(f"{hardware.get('device','?')} ({hardware.get('cpu','?')})")
    )
    body = ""
    for r in rows:
        body += (f"{esc(r['label'])} & {ms(r.get('bi_encode_ms'))} & "
                 f"{ms(r.get('vector_search_ms'))} & "
                 f"{ms(r.get('cross_encode_ms'))} & "
                 f"{ms(r.get('features_meta_ms'))} & "
                 f"\\textbf{{{ms(r.get('total_overhead_ms'))}}} \\\\\n")
    return head + body + "\\bottomrule\n\\end{tabular}\n}\n\\end{table*}\n"


def table_end_to_end(rows: Sequence[dict], no_cache: dict) -> str:
    head = (
        "\\begin{table*}[htbp]\n\\centering\n"
        "\\caption{End-to-end serving results on the QQP-derived query stream. "
        "Every column comes from the same run of the same configuration, so "
        "the overhead, response time and token savings are mutually "
        "consistent (residual of the identity "
        "$\\bar{T}=\\bar{o}+(1-h)\\bar{g}$ reported in the last column).}\n"
        "\\label{tab:end_to_end}\n"
        "\\resizebox{\\textwidth}{!}{\n"
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        "\\textbf{Configuration} & \\textbf{Overhead/req} & \\textbf{Hit "
        "Ratio} & \\textbf{Avg Response} & \\textbf{Speed-up} & "
        "\\textbf{Token Saving} & \\textbf{Residual} \\\\\n\\midrule\n")
    base = no_cache["avg_response_time_s"]
    body = (f"{esc('No cache (direct LLM)')} & -- & -- & "
            f"{base:.3f} s & 1.00$\\times$ & 0.0\\% & -- \\\\\n")
    for r in rows:
        body += (f"{esc(r['label'])} & {ms(r['avg_overhead_ms'])} & "
                 f"{r['hit_ratio_pct']:.1f}\\% & "
                 f"{r['avg_response_time_s']:.3f} s & "
                 f"{base / max(1e-9, r['avg_response_time_s']):.2f}$\\times$ & "
                 f"{r['token_saving_pct']:.1f}\\% & "
                 f"{r['consistency']['residual_ms']:+.1f} ms \\\\\n")
    return head + body + "\\bottomrule\n\\end{tabular}\n}\n\\end{table*}\n"


def table_ablation_isolation(rows: Sequence[dict]) -> str:
    head = ("\\begin{table*}[htbp]\n\\centering\n"
            "\\caption{Feature groups in isolation. 5-fold stratified, "
            "group-disjoint cross-validation on the held-out 3,000-pair QQP "
            "subset (60/40 duplicate ratio). Normalisation is fitted inside "
            "each training fold. Cost is the marginal per-request cost of "
            "producing the group at batch size 1.}\n"
            "\\label{tab:ablation_isolation}\n"
            "\\begin{tabular}{lccc}\n\\toprule\n"
            "\\textbf{Feature Group} & \\textbf{\\# Features} & \\textbf{F1} & "
            "\\textbf{Marginal Cost} \\\\\n\\midrule\n")
    body = ""
    for r in rows:
        body += (f"{esc(r['group_label'])} & {r['n_features']} & "
                 f"{pm(r['f1_mean'], r['f1_std'], 3)} & "
                 f"{ms(r.get('marginal_cost_ms'), 2)} \\\\\n")
    return head + body + "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"


def table_ablation_incremental(rows: Sequence[dict]) -> str:
    head = ("\\begin{table*}[htbp]\n\\centering\n"
            "\\caption{Incremental feature addition, measured on the held-out "
            "3,000-pair ablation subset (60/40). $\\Delta$F1 is relative to "
            "the bi-encoder + cross-encoder baseline; the 95\\% CI is a "
            "bootstrap over out-of-fold predictions.}\n"
            "\\label{tab:ablation_incremental}\n"
            "\\begin{tabular}{lcccc}\n\\toprule\n"
            "\\textbf{Configuration} & \\textbf{F1} & "
            "\\textbf{$\\Delta$F1 vs Bi+Cross} & \\textbf{95\\% CI} & "
            "\\textbf{Overhead} \\\\\n\\midrule\n")
    body = ""
    for r in rows:
        ci = r.get("ci95")
        ci_s = "--" if not ci else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        d = r.get("delta_f1")
        d_s = "--" if d is None else f"{d:+.4f}"
        body += (f"{esc(r['label'])} & {pm(r['f1_mean'], r['f1_std'], 4)} & "
                 f"{d_s} & {ci_s} & {ms(r.get('overhead_ms'), 2)} \\\\\n")
    return head + body + "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"


def table_baselines(rows: Sequence[dict]) -> str:
    head = ("\\begin{table*}[htbp]\n\\centering\n"
            "\\caption{Comparison against published semantic-caching systems "
            "on the same QQP-derived stream, same bi-encoder, same hardware, "
            "batch size 1. All systems are faithful re-implementations of the "
            "published decision logic sharing one cache substrate, so the "
            "comparison isolates the decision rule.}\n"
            "\\label{tab:baseline_comparison}\n"
            "\\resizebox{\\textwidth}{!}{\n"
            "\\begin{tabular}{lcccccc}\n\\toprule\n"
            "\\textbf{System} & \\textbf{Hit Ratio} & \\textbf{Hit Precision} "
            "& \\textbf{Correct Miss (distractors)} & \\textbf{Token Saving} & "
            "\\textbf{Overhead/req} & \\textbf{Avg Response} \\\\\n\\midrule\n")
    body = ""
    for r in rows:
        body += (f"{esc(r['label'])} & {r['hit_ratio_pct']:.1f}\\% & "
                 f"{r['precision_of_hits_pct']:.1f}\\% & "
                 f"{r.get('correct_miss_ratio_on_hard_negatives_pct', float('nan')):.1f}\\% & "
                 f"{r['token_saving_pct']:.1f}\\% & "
                 f"{ms(r['avg_overhead_ms'])} & "
                 f"{r['avg_response_time_s']:.3f} s \\\\\n")
    return head + body + "\\bottomrule\n\\end{tabular}\n}\n\\end{table*}\n"


def table_cheap_vs_expensive(rows: Sequence[dict]) -> str:
    head = ("\\begin{table*}[htbp]\n\\centering\n"
            "\\caption{Accuracy per unit of latency. \\emph{Cheap} features "
            "are arithmetic over bi-encoder vectors the pipeline has already "
            "computed; the \\emph{expensive} encoder is the cross-encoder, "
            "which requires a second pairwise forward pass and cannot reuse "
            "them.}\n"
            "\\label{tab:cheap_vs_expensive}\n"
            "\\begin{tabular}{llccc}\n\\toprule\n"
            "\\textbf{Component} & \\textbf{Class} & \\textbf{$\\Delta$F1} & "
            "\\textbf{Added Latency} & \\textbf{$\\Delta$F1 per 100 ms} "
            "\\\\\n\\midrule\n")
    body = ""
    for r in rows:
        eff = r.get("f1_per_100ms")
        eff_s = "--" if eff is None or not np.isfinite(eff) else f"{eff:+.4f}"
        body += (f"{esc(r['component'])} & {esc(r['class'])} & "
                 f"{r['delta_f1']:+.4f} & {ms(r['added_ms'], 2)} & "
                 f"{eff_s} \\\\\n")
    return head + body + "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10,
                         "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def fig_dynamic_hit_ratio(per_length: dict[int, dict], out: str = "fig2_dynamic_hit_ratio.png"):
    plt = _plt()
    lens = sorted(per_length)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(lens, [per_length[l]["hit_ratio_on_repeats_pct"] for l in lens],
               "o-", label="Hit ratio on repeated queries")
    ax[0].plot(lens, [per_length[l]["hit_ratio_on_servable_repeats_pct"] for l in lens],
               "s--", label="Hit ratio on servable repeats (oracle-conditioned)")
    ax[0].plot(lens, [per_length[l]["correct_miss_ratio_on_hard_negatives_pct"] for l in lens],
               "^-", label="Correct miss on near-duplicate distractors")
    ax[0].set_xlabel("Stream length (requests processed)")
    ax[0].set_ylabel("%")
    ax[0].set_title("Cache behaviour as the stream advances")
    ax[0].legend(fontsize=8)
    ax[1].plot(lens, [per_length[l]["hit_ratio_pct"] for l in lens], "o-",
               label="Overall hit ratio")
    ax[1].plot(lens, [per_length[l]["oracle_hit_ratio_pct"] for l in lens], "k--",
               label="Oracle ceiling (equivalent entry live)")
    ax[1].set_xlabel("Stream length (requests processed)")
    ax[1].set_ylabel("%")
    ax[1].set_title("Realised vs achievable hit ratio")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    p = C.FIGURES_DIR / out
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def fig_response_time(rows: Sequence[dict], no_cache: dict,
                      out: str = "fig3_response_time_by_config.png"):
    plt = _plt()
    labels = ["No cache"] + [r["short"] for r in rows]
    vals = [no_cache["avg_response_time_s"]] + [r["avg_response_time_s"] for r in rows]
    over = [0.0] + [r["avg_overhead_ms"] / 1000.0 for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, vals, color="#4C72B0", label="Average response time")
    ax.bar(x, over, color="#DD8452", label="of which: cache overhead")
    for i, v in enumerate(vals):
        sp = no_cache["avg_response_time_s"] / max(1e-9, v)
        ax.text(i, v + 0.02, f"{v:.3f}s\n{sp:.2f}x", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("seconds")
    ax.set_title("Average response time by cache configuration (batch size 1)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = C.FIGURES_DIR / out
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def fig_token_savings(rows: Sequence[dict], out: str = "fig8_token_savings.png"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(rows))
    w = 0.38
    ax.bar(x - w / 2, [r["token_saving_pct"] for r in rows], w,
           label="Token saving (all hits)")
    ax.bar(x + w / 2, [r.get("token_saving_correct_only_pct", np.nan) for r in rows],
           w, label="Token saving (correct hits only)")
    ax.set_xticks(x)
    ax.set_xticklabels([r["short"] for r in rows], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("%")
    ax.set_title("Token savings by configuration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = C.FIGURES_DIR / out
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def fig_pareto(points: Sequence[dict], out: str = "fig9_latency_accuracy_pareto.png"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for p_ in points:
        ax.scatter(p_["overhead_ms"], p_["f1"], s=70)
        ax.annotate(p_["label"], (p_["overhead_ms"], p_["f1"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Per-request overhead (ms, log scale, batch size 1)")
    ax.set_ylabel("F1 on held-out QQP ablation subset")
    ax.set_title("Latency-accuracy trade-off")
    fig.tight_layout()
    p = C.FIGURES_DIR / out
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def fig_eviction(results: dict[str, dict[int, float]],
                 out: str = "fig7_eviction_policies.png"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, series in results.items():
        sizes = sorted(series)
        ax.plot(sizes, [series[s] for s in sizes], "o-", label=name)
    ax.set_xlabel("Cache capacity (entries)")
    ax.set_ylabel("Hit ratio (%)")
    ax.set_title("Eviction policy comparison on a dynamic stream")
    ax.legend()
    fig.tight_layout()
    p = C.FIGURES_DIR / out
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def fig_baselines(rows: Sequence[dict], out: str = "fig10_baseline_comparison.png"):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    names = [r["short"] for r in rows]
    x = np.arange(len(names))
    ax[0].bar(x - 0.2, [r["hit_ratio_pct"] for r in rows], 0.4, label="Hit ratio")
    ax[0].bar(x + 0.2, [r["precision_of_hits_pct"] for r in rows], 0.4,
              label="Precision of hits")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax[0].set_ylabel("%")
    ax[0].set_title("Hit quality")
    ax[0].legend(fontsize=8)
    ax[1].scatter([r["avg_overhead_ms"] for r in rows],
                  [r["token_saving_pct"] for r in rows], s=70)
    for r in rows:
        ax[1].annotate(r["short"], (r["avg_overhead_ms"], r["token_saving_pct"]),
                       textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Overhead per request (ms, log)")
    ax[1].set_ylabel("Token saving (%)")
    ax[1].set_title("Cost vs benefit")
    fig.tight_layout()
    p = C.FIGURES_DIR / out
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p
