"""
Experiment 53 -- cost of the decision and of false hits (docs/PREREGISTRATION_round2b.md, E9).

Part 1, CPU timing per request (batch 1, single thread unless stated): bi-encoder encode, exact vector search over
1k/10k/100k cached entries, feature construction for one candidate, and C2 inference through (a) the scikit-learn
pipeline and (b) an optimised path (scaler folded into one numpy expression, LightGBM Booster.predict with one thread
on a contiguous float32 row). The two outputs must agree to 1e-6.
Part 2, break-even: from the Combo stream results of exp51 (false-hit and correct-hit rates of C1 and C2 at a matched
hit ratio on the matchable prompts), net benefit per 1,000 requests for false-hit prices lambda in
{0, 1, 2, 5, 10, 20, 50}. Benefit of a correct hit = one generation avoided (g seconds, the mean recorded latency of
the real answers in the repository's answer pool); a false hit costs lambda * g; the decision overhead is paid on every
request. Zero-benefit traffic = the 3,500 no-match prompts alone. Nothing is simulated.

Run: python -m semcache.experiments.exp53_cost --combo results/raw/revision/exp51_combo_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .. import config as C
from ..features import build_bundle
from ..metaclf import make_pipeline
from ._common import banner, hardware_manifest, save_result

LAMBDAS = (0, 1, 2, 5, 10, 20, 50)
GROUPS = ("bi_cosine", "distance", "elementwise", "text")


def tm(fn, n=200, warm=20):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    a = np.array(t)
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95))}


def generation_latency() -> float:
    lat = [json.loads(l)["latency_s"] for l in open(C.LLM_CACHE_DIR / "pool_gpt-5.4-nano.jsonl", encoding="utf8")
           if l.strip()]
    return float(np.mean(lat))


def matched_point(rows, target):
    xs = np.array([r["hit_ratio_matchable"] for r in rows])
    o = np.argsort(xs)
    f = lambda key: float(np.interp(target, xs[o], np.array([r[key] for r in rows])[o]))
    return {k: f(k) for k in ("hit_ratio_matchable", "precision_matchable", "false_hit_rate_no_match")}


def main(combo: str, seed: int = C.SEED, tag: str = "") -> dict:
    banner("EXP 53 | decision overhead and break-even false-hit price")
    import torch
    torch.set_num_threads(1)
    from ..data import build_splits
    from ..encoders import get_bi
    bi = get_bi()                      # run on a CPU-only runtime: the encoder then defaults to the CPU
    q = "How do I reset my router to factory settings without losing my saved wifi passwords?"
    c = "What is the best way to reset a router to its factory defaults?"
    out = {"timing": {}}
    out["timing"]["bi_encode (mpnet, batch 1, 1 thread)"] = tm(lambda: bi.model.encode([q], normalize_embeddings=True,
                                                                                       convert_to_numpy=True), n=60, warm=5)
    rng = np.random.default_rng(seed)
    eq = bi.model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)[0]
    ec = bi.model.encode([c], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)[0]
    for n in (1000, 10000, 100000):
        M = rng.normal(size=(n, eq.shape[0])).astype(np.float32)
        M /= np.linalg.norm(M, axis=1, keepdims=True)
        out["timing"][f"vector_search (exact, {n} entries)"] = tm(lambda: int((M @ eq).argmax()), n=100, warm=10)
    out["timing"]["features (1 candidate)"] = tm(lambda: build_bundle([q], [c], eq[None], ec[None], None, groups=GROUPS)
                                                  .matrix(GROUPS))
    # C2 inference: standard vs optimised, on a model fitted to real pairs
    sp = build_splits().meta.iloc[:8000]
    e = get_bi().model.encode(sp.question1.tolist() + sp.question2.tolist(), batch_size=128, normalize_embeddings=True,
                              convert_to_numpy=True).astype(np.float32)
    n = len(sp)
    X = build_bundle(sp.question1.tolist(), sp.question2.tolist(), e[:n], e[n:], None, groups=GROUPS).matrix(GROUPS)
    pipe = make_pipeline("LightGBM", seed=seed)
    pipe.fit(X, sp.is_duplicate.to_numpy())
    x1 = X[:1].copy()
    sc = pipe.named_steps["scaler"]
    mu, inv = sc.mean_.astype(np.float32), (1.0 / sc.scale_).astype(np.float32)
    booster = pipe.named_steps["clf"].booster_

    def fast(x):
        return booster.predict(np.ascontiguousarray((x - mu) * inv, dtype=np.float32), num_threads=1)

    out["timing"]["C2 inference, sklearn pipeline"] = tm(lambda: pipe.predict_proba(x1)[:, 1])
    out["timing"]["C2 inference, optimised booster"] = tm(lambda: fast(x1))
    out["optimised_matches_pipeline_max_abs_diff"] = float(np.abs(pipe.predict_proba(X[:500])[:, 1] - fast(X[:500])).max())
    print("  timing:", {k: round(v["mean_ms"], 3) for k, v in out["timing"].items()}, flush=True)

    # ---- break-even -------------------------------------------------------------------------------
    g = generation_latency()
    d = json.load(open(combo, encoding="utf8"))["stream"]
    over = {"C1": out["timing"]["bi_encode (mpnet, batch 1, 1 thread)"]["mean_ms"] / 1000 +
            out["timing"]["vector_search (exact, 10000 entries)"]["mean_ms"] / 1000}
    over["C2 (sklearn)"] = over["C1"] + (out["timing"]["features (1 candidate)"]["mean_ms"] +
                                         out["timing"]["C2 inference, sklearn pipeline"]["mean_ms"]) / 1000
    over["C2 (optimised)"] = over["C1"] + (out["timing"]["features (1 candidate)"]["mean_ms"] +
                                           out["timing"]["C2 inference, optimised booster"]["mean_ms"]) / 1000
    n_match, n_nm = 24000, 3500
    rows = {}
    for target in (0.3, 0.5, 0.7):
        for sysname, key, ov in (("C1", "C1", over["C1"]), ("C2 (sklearn)", "C2", over["C2 (sklearn)"]),
                                 ("C2 (optimised)", "C2", over["C2 (optimised)"])):
            p = matched_point(d[key], target)
            hit, prec, fh = p["hit_ratio_matchable"], p["precision_matchable"], p["false_hit_rate_no_match"]
            mix = lambda a, b: (n_match * a + n_nm * b) / (n_match + n_nm)
            p_correct, p_false = mix(hit * prec, 0.0), mix(hit * (1 - prec), fh)
            row = {"matchable_hit_ratio": hit, "precision_matchable": prec, "false_hit_rate_no_match": fh,
                   "overhead_s_per_request": ov, "p_correct_hit": p_correct, "p_false_hit": p_false,
                   "break_even_lambda": float((p_correct * g - ov) / (p_false * g)) if p_false > 0 else None}
            for lam in LAMBDAS:
                row[f"net_s_per_1000_mixed_lambda{lam}"] = 1000 * (p_correct * g - lam * p_false * g - ov)
                row[f"net_s_per_1000_nomatch_lambda{lam}"] = 1000 * (-lam * fh * g - ov)
            rows[f"{sysname} @ matchable hit {target}"] = row
    out["generation_latency_s"] = g
    out["break_even"] = rows
    out["manifest"] = hardware_manifest({"threads": 1, "combo_results": combo}, seed=seed)
    save_result(out, "exp53_cost", tag=tag)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(a.combo, tag=a.tag)
