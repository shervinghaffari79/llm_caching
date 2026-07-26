"""
Experiment 8 -- does the ablation actually support the title?

"When Cheap Features Beat Expensive Encoders" is a quantitative claim, so it
needs a quantitative table. This script joins the measured per-group latency
(exp1) to the measured per-group F1 contribution (exp3) and reports, for each
component, the F1 it buys per 100 ms it costs.

Definitions, stated so the title is not doing work the paper never does:

  cheap feature      A value computed by arithmetic over bi-encoder vectors
                     that the retrieval step has already produced: the cosine
                     itself, L2/L1 distances, the element-wise product and
                     absolute difference, and surface text statistics. Their
                     marginal cost is a vector op, tens of microseconds.
                     They are *not* encoder-free -- they consume bi-encoder
                     output. The claim is that no *second, pairwise* encoder
                     is needed, not that no encoder is needed.

  expensive encoder  The cross-encoder. It requires an additional forward pass
                     over the concatenated pair with quadratic attention, and
                     it cannot reuse the cached embedding of either side,
                     because its representation is defined jointly on the pair.
                     Its cost is per candidate, so it also scales with the
                     number of candidates re-ranked.

The title is supported iff the cheap groups deliver more F1 per unit latency
*and* the expensive one adds no F1 that survives a significance test. Both
conditions are checked here and the verdict is written into the output.

Run: python -m semcache.experiments.exp8_cheap_vs_expensive
"""
from __future__ import annotations

import numpy as np

from .. import config as C
from ..features import CHEAP_GROUPS, EXPENSIVE_GROUPS
from ..reporting import (fig_pareto, load_json, save_json, save_latex,
                         table_cheap_vs_expensive)
from ._common import banner, hardware_manifest


def main() -> dict:
    banner("EXP 8 | Cheap features vs expensive encoder: cost-benefit accounting")
    try:
        lat = load_json("exp1_latency_breakdown")
        abl = load_json("exp3_ablation")
    except FileNotFoundError as e:
        raise SystemExit("run exp1_latency_breakdown and exp3_ablation first") from e

    stages = lat["online_stages_ms"]
    inc = {r["label"]: r for r in abl["stratified_group_kfold"]["incremental"]}
    iso = {r["group"]: r for r in abl["stratified_group_kfold"]["isolation"]}
    ct = abl["stratified_group_kfold"]["cross_encoder_removal_test"]

    def sm(k):
        return stages.get(k, {}).get("mean_ms", 0.0)

    base = inc["Bi + Cross"]["f1_mean"]
    nocross = inc["Full Model - Cross-Encoder"]["f1_mean"]
    full = inc["Full Model"]["f1_mean"]
    bi_only = inc["Bi-Encoder Cosine only"]["f1_mean"]

    rows = [
        {"component": "Element-wise ops (product, |diff|)", "class": "cheap",
         "delta_f1": inc["+ Element-wise Ops"]["f1_mean"] - base,
         "added_ms": sm("feat_elementwise")},
        {"component": "Distance metrics (L2, L1)", "class": "cheap",
         "delta_f1": inc["+ Distance Metrics"]["f1_mean"] - base,
         "added_ms": sm("feat_distance")},
        {"component": "Text features", "class": "cheap",
         "delta_f1": inc["+ Text Features"]["f1_mean"] - base,
         "added_ms": sm("feat_text")},
        {"component": "LightGBM meta-classifier inference", "class": "cheap",
         "delta_f1": float("nan"), "added_ms": sm("meta_infer")},
        {"component": "Cross-encoder (stsb-roberta-large)", "class": "expensive",
         "delta_f1": full - nocross,
         "added_ms": sm("cross_encode")},
    ]
    for r in rows:
        r["f1_per_100ms"] = (r["delta_f1"] / r["added_ms"] * 100.0
                             if r["added_ms"] and np.isfinite(r["delta_f1"])
                             else float("nan"))

    print(f"  {'component':42s} {'class':10s} {'dF1':>9s} {'ms':>10s} {'dF1/100ms':>11s}")
    for r in rows:
        print(f"  {r['component']:42s} {r['class']:10s} {r['delta_f1']:+9.4f} "
              f"{r['added_ms']:10.3f} {r['f1_per_100ms']:+11.4f}")

    cheap_total_ms = sum(sm(k) for k in ("feat_bi_cosine", "feat_distance",
                                         "feat_elementwise", "feat_text",
                                         "meta_infer"))
    cheap_gain = nocross - bi_only
    cross_gain = full - nocross
    cross_ms = sm("cross_encode")

    verdict = {
        "cheap_features_gain_f1": cheap_gain,
        "cheap_features_cost_ms": cheap_total_ms,
        "cheap_f1_per_100ms": cheap_gain / max(1e-9, cheap_total_ms) * 100.0,
        "cross_encoder_gain_f1": cross_gain,
        "cross_encoder_cost_ms": cross_ms,
        "cross_f1_per_100ms": cross_gain / max(1e-9, cross_ms) * 100.0,
        "efficiency_ratio": ((cheap_gain / max(1e-9, cheap_total_ms)) /
                             max(1e-12, (cross_gain / max(1e-9, cross_ms)))
                             if cross_gain > 0 else float("inf")),
        "cross_encoder_ci95": [ct.get("bootstrap_ci95_low"),
                               ct.get("bootstrap_ci95_high")],
        "cross_encoder_ci_includes_zero": ct.get("bootstrap_ci_includes_zero"),
        "cross_encoder_paired_t_p": ct.get("paired_t_p"),
    }
    verdict["title_supported"] = bool(
        verdict["cheap_f1_per_100ms"] > verdict["cross_f1_per_100ms"]
        and (ct.get("bootstrap_ci_includes_zero") or cross_gain <= 0.001))
    verdict["statement"] = (
        f"Cheap, bi-encoder-derived features add {cheap_gain:+.4f} F1 over the "
        f"bare cosine for {cheap_total_ms:.2f} ms per request. The "
        f"cross-encoder adds {cross_gain:+.4f} F1 for {cross_ms:.1f} ms. "
        + ("Its 95% bootstrap CI includes zero, so on this data its "
           "contribution is not distinguishable from noise. "
           if ct.get("bootstrap_ci_includes_zero") else
           "Its contribution is statistically detectable but small. ")
        + ("The title's claim is supported on a per-millisecond basis and "
           "should be defended in those terms in the text, not left to the "
           "title alone."
           if verdict["title_supported"] else
           "The title overstates the result as measured here and should be "
           "softened."))

    print(f"\n  cheap: {cheap_gain:+.4f} F1 for {cheap_total_ms:.2f} ms "
          f"({verdict['cheap_f1_per_100ms']:+.4f} F1/100ms)")
    print(f"  cross: {cross_gain:+.4f} F1 for {cross_ms:.1f} ms "
          f"({verdict['cross_f1_per_100ms']:+.4f} F1/100ms)")
    print(f"  efficiency ratio: {verdict['efficiency_ratio']:.0f}x")
    print(f"\n  title supported by the measurements: {verdict['title_supported']}")

    pareto = []
    for key in C.CONFIG_ORDER:
        cfg = C.CONFIGS[key]
        lab = {"C1_bi_only": "Bi-Encoder Cosine only",
               "C2_bi_geo_lgb": "Full Model - Cross-Encoder",
               "C3_bi_cross": "Bi + Cross",
               "C4_full_stacking": "Full Model"}[key]
        r = next(x for x in lat["per_config"] if x["config"] == key)
        pareto.append({"label": cfg.paper_name or key,
                       "overhead_ms": r["total_overhead_ms"],
                       "f1": inc[lab]["f1_mean"], "config": key})
    fig_pareto(pareto)

    out = {
        "manifest": hardware_manifest(),
        "definitions": {
            "cheap_features": list(CHEAP_GROUPS),
            "expensive_encoder": list(EXPENSIVE_GROUPS),
            "cheap_definition": (
                "arithmetic over bi-encoder vectors already computed by the "
                "retrieval step; marginal cost is a vector operation. Not "
                "encoder-free: they consume bi-encoder output."),
            "expensive_definition": (
                "a second forward pass with quadratic attention over the "
                "concatenated pair, per candidate, unable to reuse either "
                "side's cached embedding."),
        },
        "rows": rows,
        "verdict": verdict,
        "pareto": pareto,
        "group_isolation_f1": {g: iso[g]["f1_mean"] for g in iso},
        "why_keep_the_cross_encoder": (
            "The measurements say it can be dropped for latency-sensitive "
            "deployments. Reasons to keep it as an optional stage, which "
            "should be argued explicitly rather than left implicit: (1) it is "
            "the only component that models token-level interaction between "
            "the two queries, so it degrades differently from the "
            "embedding-derived features under distribution shift; (2) on the "
            "STS regression formulation the ensemble uses its continuous "
            "score, where a threshold-free signal matters more than in binary "
            "cache decisions; (3) in an offline batching regime its amortised "
            "cost falls by an order of magnitude, so the trade-off differs "
            "for batch re-validation of a cache versus online serving. None "
            "of these justify keeping it on the online path by default, and "
            "the paper should recommend the cross-encoder-free configuration "
            "as the deployment default."),
    }
    save_json(out, "exp8_cheap_vs_expensive")
    save_latex(table_cheap_vs_expensive(rows), "table_cheap_vs_expensive")
    return out


if __name__ == "__main__":
    main()
