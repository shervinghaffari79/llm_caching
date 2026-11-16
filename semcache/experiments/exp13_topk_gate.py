"""
Experiment 13 -- top-k retrieval, selective cross-encoder gating, abstention
(E4).

* Top-k: C1-C4 with k in {1, 3, 5, 10}. Each candidate is scored by the
  configuration's rule and the best accepted one is served; the cross-encoder
  scores all k pairs in one batched call.
* Gating: C2's meta-classifier decides alone outside a probability band and
  defers to the cross-encoder inside it. Bands are chosen on out-of-fold
  probabilities of the 30k-pair meta set (group-disjoint folds), never on the
  stream: each band is the central interval holding a target share of OOF
  pairs, and we report the share of OOF errors it captures.
* Abstention: C2 sends requests whose probability lies within a margin of the
  threshold to the LLM instead of serving them.

Latency is measured (no memoisation), with generation added in virtual time.

Run: python -m semcache.experiments.exp13_topk_gate
"""
from __future__ import annotations

import argparse

import numpy as np

from .. import config as C
from ..data import build_splits, make_group_folds
from ..encoders import warmup
from ..metaclf import cross_validate
from ._common import (META_GROUPS, add_run_args, banner, encode_frame,
                      ensure_pool, get_stream, hardware_manifest, make_backend,
                      save_result, serve, stream_args, stream_params, train_metas)

KS = (1, 3, 5, 10)
BAND_SHARES = (0.05, 0.10, 0.20, 0.35)
MARGINS = (0.05, 0.10, 0.20)
KEEP = ("hit_ratio_pct", "precision_of_hits_pct", "false_discovery_rate_pct",
        "token_saving_correct_only_pct", "break_even_false_hit_cost",
        "correct_miss_ratio_on_hard_negatives_pct", "hit_ratio_on_servable_repeats_pct",
        "avg_overhead_ms", "p95_overhead_ms", "p99_overhead_ms",
        "avg_response_time_s", "gated_pct", "abstained_pct",
        "cross_calls_per_request", "stage_ms_per_request")


def _row(stats: dict, **extra) -> dict:
    return {**extra, **{k: stats.get(k) for k in KEEP}}


def gate_bands_from_oof(seed: int) -> list[dict]:
    """Central probability intervals of C2's OOF predictions."""
    splits = build_splits()
    enc = encode_frame(splits.meta, f"meta{len(splits.meta)}", with_cross=True)
    X = enc.bundle.matrix(META_GROUPS["C2_bi_geo_lgb"])
    folds = make_group_folds(enc.y, enc.groups, C.CV_FOLDS_META)
    cv = cross_validate(X, enc.y, folds, "LightGBM", seed=seed)
    p, y = cv.oof_prob, cv.oof_true
    err = (p >= C.META_PROB_THRESHOLD) != (y == 1)
    order = np.argsort(np.abs(p - C.META_PROB_THRESHOLD))
    bands = []
    for share in BAND_SHARES:
        idx = order[: max(1, int(share * len(p)))]
        half = float(np.abs(p[idx] - C.META_PROB_THRESHOLD).max())
        lo, hi = C.META_PROB_THRESHOLD - half, C.META_PROB_THRESHOLD + half
        inside = (p >= lo) & (p < hi)
        bands.append({"share_target": share, "lo": lo, "hi": hi,
                      "oof_share_inside": float(inside.mean()),
                      "oof_errors_captured_pct": 100.0 * float(err[inside].sum() / max(1, err.sum())),
                      "oof_error_rate_inside_pct": 100.0 * float(err[inside].mean()) if inside.any() else None})
    return bands


def main(seed: int = C.SEED, n: int = 2000, params: dict | None = None,
         fetch: bool = True, tag: str = "") -> dict:
    banner("EXP 13 | Top-k, selective gating, abstention")
    params = params or {}
    warmup(include_cross=True)
    wl = get_stream(seed, n, **params)
    cov = ensure_pool(wl.queries(), live=fetch)
    metas = train_metas(seed=seed)

    topk = []
    for key in ("C1_bi_only", "C2_bi_geo_lgb", "C3_bi_cross", "C4_full_stacking"):
        for k in KS:
            r = serve(key, wl, make_backend(seed=seed), metas, keep_events=False, top_k=k)
            topk.append(_row(r.stats, config=key, k=k))
            print(f"  {key} k={k}: prec {r.stats['precision_of_hits_pct']:.1f}% "
                  f"overhead {r.stats['avg_overhead_ms']:.1f} ms")

    bands = gate_bands_from_oof(seed)
    gating = []
    for b in bands:
        r = serve("C5_gated", wl, make_backend(seed=seed), metas, keep_events=False,
                  gate_band=(b["lo"], b["hi"]))
        gating.append(_row(r.stats, band=b))
        print(f"  gate {b['lo']:.3f}-{b['hi']:.3f}: gated {r.stats['gated_pct']:.1f}% "
              f"prec {r.stats['precision_of_hits_pct']:.1f}%")
    # sanity: an empty band must reproduce C2 exactly
    r_empty = serve("C5_gated", wl, make_backend(seed=seed), metas, keep_events=False,
                    gate_band=(C.META_PROB_THRESHOLD, C.META_PROB_THRESHOLD))
    r_c2 = serve("C2_bi_geo_lgb", wl, make_backend(seed=seed), metas, keep_events=False)
    sanity = {"empty_band_hits": r_empty.stats["hits"], "c2_hits": r_c2.stats["hits"],
              "empty_band_true_hits": r_empty.stats["true_hits"],
              "c2_true_hits": r_c2.stats["true_hits"]}
    sanity["reproduces_c2"] = (sanity["empty_band_hits"] == sanity["c2_hits"]
                               and sanity["empty_band_true_hits"] == sanity["c2_true_hits"])

    abstain = []
    for m in MARGINS:
        r = serve("C2_bi_geo_lgb", wl, make_backend(seed=seed), metas,
                  keep_events=False, abstain_margin=m)
        abstain.append(_row(r.stats, margin=m))

    out = {"manifest": hardware_manifest({"n_stream": n, "stream_params": params,
                                          "virtual_time": True}, seed=seed),
           "workload": wl.meta, "pool": cov, "topk": topk, "gate_bands": bands,
           "gating": gating, "gate_sanity": sanity, "abstention": abstain,
           "c2_reference": _row(r_c2.stats, config="C2_bi_geo_lgb")}
    save_result(out, "exp13_topk_gate", seed=seed, tag=tag)
    return out


if __name__ == "__main__":
    ap = stream_args(add_run_args(argparse.ArgumentParser()))
    a = ap.parse_args()
    main(a.seed, a.n, stream_params(a), fetch=not a.no_fetch, tag=a.tag)
