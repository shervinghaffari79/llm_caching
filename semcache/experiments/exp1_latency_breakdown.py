"""
Experiment 1 -- per-component latency, re-measured under the *same* hardware
and batching conditions as the end-to-end test.

Two regimes are reported side by side:

  online  (batch size 1)  -- what a live cache actually pays. A query must be
                            answered before the next one is known, so no
                            batching is possible. Every number used in the
                            end-to-end comparison comes from this regime.
  offline (batch size 32) -- what an evaluation script pays when it scores a
                            whole labelled file. Per-item cost is far lower,
                            especially for the cross-encoder.

Reporting an offline-amortised cross-encoder cost next to an online end-to-end
average (or vice versa) is the mechanism that produces impossible table pairs,
so both are emitted and the regime is printed on every row.

Run: python -m semcache.experiments.exp1_latency_breakdown [--n 300]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .. import config as C
from ..cache import VectorBase
from ..data import build_splits
from ..encoders import StageTimer, get_bi, get_cross, warmup
from ..features import (distance_features, elementwise_features,
                        text_features_one)
from ..metaclf import fit_final
from ..reporting import save_json, save_latex, table_latency_breakdown
from ._common import banner, encode_frame, hardware_manifest, save_result


def _percentiles(v: list[float]) -> dict:
    a = np.array(v, dtype=float)
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)),
            "std_ms": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n": int(len(a))}


def measure_online(n_queries: int, index_size: int, seed: int = C.SEED) -> dict:
    """Time each stage individually, batch size 1, on a warm process."""
    splits = build_splits()
    abl = splits.ablation
    rng = np.random.default_rng(seed)

    bi = get_bi()
    ce = get_cross()
    warmup(include_cross=True)

    # Build a realistic index to search against.
    corpus = abl["question2"].tolist()[:index_size]
    corpus_emb = bi.encode_cached(corpus, f"latency_index_{index_size}")
    vb = VectorBase()
    for i, v in enumerate(corpus_emb):
        vb.add(i, v)

    queries = abl["question1"].tolist()
    idx = rng.choice(len(queries), size=min(n_queries, len(queries)), replace=False)
    queries = [queries[i] for i in idx]

    stages: dict[str, list[float]] = {k: [] for k in
                                      ("bi_encode", "vector_search", "cross_encode",
                                       "feat_distance", "feat_elementwise",
                                       "feat_text", "feat_bi_cosine", "meta_infer")}

    # A trained meta-classifier is needed to time inference realistically.
    enc = encode_frame(abl.iloc[:1500], "latency_meta_fit", with_cross=True)
    X = enc.bundle.matrix(("bi_cosine", "cross_score", "distance",
                           "elementwise", "text"))
    meta = fit_final(X, enc.y, ("bi_cosine", "cross_score", "distance",
                                "elementwise", "text")).set_inference_threads()

    for q in queries:
        t = StageTimer()
        eq = bi.encode_one(q, timer=t, stage="bi_encode")
        sims, ids = vb.search(eq, k=1, timer=t, stage="vector_search")
        cand = corpus[int(ids[0])]
        ec = corpus_emb[int(ids[0])]

        cs = ce.score_one(q, cand, timer=t, stage="cross_encode")

        with t.stage("feat_bi_cosine"):
            _ = float(np.dot(eq, ec))
        with t.stage("feat_distance"):
            _ = distance_features(eq, ec)
        with t.stage("feat_elementwise"):
            _ = elementwise_features(eq, ec)
        with t.stage("feat_text"):
            _ = text_features_one(q, cand)

        x = np.concatenate([
            np.array([float(np.dot(eq, ec))], dtype=np.float32),
            np.array([cs], dtype=np.float32),
            distance_features(eq, ec),
            elementwise_features(eq, ec),
            text_features_one(q, cand)]).reshape(1, -1)
        with t.stage("meta_infer"):
            meta.predict_proba_one(x)

        snap = t.snapshot_ms()
        for k in stages:
            if k in snap:
                stages[k].append(snap[k])

    return {k: _percentiles(v) for k, v in stages.items() if v}


def measure_offline(n: int, batch: int = C.OFFLINE_BATCH) -> dict:
    """Amortised per-item cost when the whole workload is known in advance."""
    splits = build_splits()
    abl = splits.ablation.iloc[:n]
    q1, q2 = abl["question1"].tolist(), abl["question2"].tolist()
    bi, ce = get_bi(), get_cross()
    warmup()

    t0 = time.perf_counter()
    bi.encode(q1, batch_size=batch)
    bi.encode(q2, batch_size=batch)
    bi_ms = (time.perf_counter() - t0) / len(q1) * 1000.0

    t0 = time.perf_counter()
    ce.score(list(zip(q1, q2)), batch_size=batch)
    ce_ms = (time.perf_counter() - t0) / len(q1) * 1000.0

    return {"batch_size": batch, "n_pairs": len(q1),
            "bi_encode_ms_per_pair": bi_ms, "cross_encode_ms_per_pair": ce_ms}


def compose_configs(online: dict) -> list[dict]:
    """Sum the measured stages into per-configuration totals."""
    def m(k):
        return online.get(k, {}).get("mean_ms", 0.0)

    bi, search, cross = m("bi_encode"), m("vector_search"), m("cross_encode")
    feats_geo = m("feat_distance") + m("feat_elementwise")
    feats_text = m("feat_text")
    feats_cos = m("feat_bi_cosine")
    lgb = m("meta_infer")

    rows = []
    for key in C.CONFIG_ORDER:
        cfg = C.CONFIGS[key]
        fm = 0.0
        if cfg.use_meta_clf:
            fm = feats_cos + lgb + (feats_geo if cfg.use_geometric else 0.0) \
                 + (feats_text if cfg.use_text else 0.0)
        total = bi + search + (cross if cfg.use_cross else 0.0) + fm
        rows.append({
            "config": key, "label": cfg.label, "short": cfg.paper_name or key,
            "bi_encode_ms": bi, "vector_search_ms": search,
            "cross_encode_ms": cross if cfg.use_cross else None,
            "features_meta_ms": fm if fm else None,
            "total_overhead_ms": total,
            "components": {"bi_encoder": cfg.use_bi, "cross_encoder": cfg.use_cross,
                           "geometric": cfg.use_geometric, "text": cfg.use_text,
                           "lightgbm": cfg.use_meta_clf},
        })
    base = rows[0]["total_overhead_ms"]
    for r in rows:
        r["overhead_vs_bi_only_x"] = r["total_overhead_ms"] / max(1e-9, base)
        r["cross_encoder_share_pct"] = (
            100.0 * (r["cross_encode_ms"] or 0.0) / max(1e-9, r["total_overhead_ms"]))
    return rows


def main(n_queries: int = 300, index_size: int = 2000) -> dict:
    banner("EXP 1 | Per-component latency (online batch=1 vs offline batch=32)")
    hw = C.HardwareProfile.capture()
    print(f"  device={hw.device}  torch_threads={hw.torch_threads}  {hw.cpu}")

    online = measure_online(n_queries, index_size)
    print("\n  Online (batch size 1) per-stage mean:")
    for k, v in sorted(online.items(), key=lambda kv: -kv[1]["mean_ms"]):
        print(f"    {k:18s} {v['mean_ms']:9.3f} ms  (p95 {v['p95_ms']:.3f})")

    offline = measure_offline(min(1000, C.N_ABLATION))
    print(f"\n  Offline (batch size {offline['batch_size']}) per-pair: "
          f"bi {offline['bi_encode_ms_per_pair']:.2f} ms, "
          f"cross {offline['cross_encode_ms_per_pair']:.2f} ms")
    print(f"  -> batching speeds the cross-encoder up by "
          f"{online['cross_encode']['mean_ms'] / max(1e-9, offline['cross_encode_ms_per_pair']):.1f}x, "
          f"which is why an offline-measured cost must never be quoted next to "
          f"an online serving average.")

    rows = compose_configs(online)
    print("\n  Per-configuration overhead (online, per request):")
    for r in rows:
        print(f"    {r['short']:28s} {r['total_overhead_ms']:8.2f} ms"
              f"   cross share {r['cross_encoder_share_pct']:.1f}%")

    out = {
        "manifest": hardware_manifest({"batch_regime": "online",
                                       "index_size": index_size,
                                       "n_queries": n_queries}),
        "online_stages_ms": online,
        "offline_amortised": offline,
        "batching_speedup_cross": (
            online["cross_encode"]["mean_ms"] /
            max(1e-9, offline["cross_encode_ms_per_pair"])),
        "per_config": rows,
        "note": ("Overhead is per *request*, not per hit: the similarity "
                 "decision runs before the hit/miss outcome is known, so it "
                 "is paid on misses too."),
    }
    save_json(out, "exp1_latency_breakdown")
    save_result(out, "exp1_latency_breakdown")
    save_latex(table_latency_breakdown(rows, hw.as_dict(), "online"),
               "table_latency_breakdown")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--index-size", type=int, default=2000)
    a = ap.parse_args()
    main(a.n, a.index_size)
