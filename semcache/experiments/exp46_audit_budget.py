"""
Experiment 46 -- audit-budgeted decision rules: a calibrated bi-encoder (C1) against a C2 trained on the
same small labelled audit sample from the target log.

Protocol: docs/PREREGISTRATION_round2b.md, section E1 (fixed before this run).

  * Log: SemBenchmarkSearchQueries in dataset order. Audit pool = rows 0-39,999 (never served). Evaluation
    windows = rows 60,000-139,999 in four windows of 20,000, empty cache at the start of each.
  * Audit sample: nearest earlier neighbour (cosine >= 0.6) of each audit-pool row, labelled by shared
    id_set; B pairs drawn at random; three draws per budget.
  * C1-cal: lowest cosine threshold whose empirical precision on all B audit pairs is >= P* (>= 30 accepted).
  * C2-in: LightGBM on the cheap features, fitted on 80% of the B pairs; probability threshold from the other
    20% by the same rule.
  * C2-qqp: the QQP-trained C2, with the same audit rule applied to its scores (transfer baseline).
  * Nothing is generated, replayed or simulated on a miss (NullBackend): hit ratio and precision only.

Run: python -m semcache.experiments.exp46_audit_budget --window 0
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from .. import config as C
from ..features import build_bundle
from ..llm import NullBackend
from ..metaclf import fit_final
from . import exp40_chronological_transfer as E40
from ..reporting import save_json
from ._common import (REVISION_DIR, banner, hardware_manifest, memoized_encoders, save_result, serve,
                      train_metas)

DATA = C.DATA_DIR / "external" / "sembenchmark_searchqueries.parquet"
VECS = C.DATA_DIR / "emb_cache" / "exp30_searchqueries_vectors.npz"
AUDIT_ROWS = 40_000
WINDOWS = tuple((s, 20_000) for s in range(60_000, 140_000, 20_000))
BUDGETS = (250, 500, 1000, 2000, 5000, 10000)
DRAWS = 3
P_STARS = (0.90, 0.95)
MIN_ACCEPTED = 30
CAND_FLOOR = 0.6
CURVE_BUDGETS = (1000, 10000)
C1_GRID = (0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.96)
C2_GRID = (0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98)
GROUPS = ("bi_cosine", "distance", "elementwise", "text")
KEYS = ("hit_ratio_pct", "precision_of_hits_pct", "false_discovery_rate_pct", "false_hit_rate_pct", "cache_entries")


def pick_threshold(score: np.ndarray, y: np.ndarray, p_star: float) -> float | None:
    """Lowest threshold whose empirical precision over the scored pairs is >= p_star with >= MIN_ACCEPTED accepted."""
    order = np.argsort(-score, kind="stable")
    s, yy = score[order], y[order].astype(float)
    n = np.arange(1, len(yy) + 1)
    prec = np.cumsum(yy) / n
    ok = np.flatnonzero((prec >= p_star) & (n >= MIN_ACCEPTED))
    return float(s[ok.max()]) if len(ok) else None


def audit_pool(prompts: np.ndarray, ids: np.ndarray, V: np.ndarray) -> pd.DataFrame:
    """For each audit-pool row, its nearest earlier row (cosine) and whether they share a class."""
    n = len(V)
    nn = np.full(n, -1)
    sim = np.full(n, -1.0, dtype=np.float32)
    for a in range(0, n, 2048):
        b = min(n, a + 2048)
        S = V[a:b] @ V[:b].T
        S[np.arange(b)[None, :] >= np.arange(a, b)[:, None]] = -1.0       # only earlier rows
        j = S.argmax(axis=1)
        nn[a:b], sim[a:b] = j, S[np.arange(b - a), j]
    df = pd.DataFrame({"i": np.arange(n), "j": nn, "cos": sim})
    df = df[(df.j >= 0) & (df.cos >= CAND_FLOOR)].reset_index(drop=True)
    df["y"] = (ids[df.i.to_numpy()] == ids[df.j.to_numpy()]).astype(int)
    return df


def features(prompts, V, pairs: pd.DataFrame) -> np.ndarray:
    i, j = pairs.i.to_numpy(), pairs.j.to_numpy()
    b = build_bundle([prompts[k] for k in i], [prompts[k] for k in j], V[i], V[j], None, groups=GROUPS)
    return b.matrix(GROUPS)


def main(window: int, seed: int = C.SEED, tag: str = "") -> dict:
    banner(f"EXP 46 | audit-budgeted decision rules, window {window}")
    t0 = time.time()
    df = pd.read_parquet(DATA)
    prompts = df.prompt.astype(str).to_numpy()
    ids = df.id_set.to_numpy()
    z = np.load(VECS, allow_pickle=True)
    vecs = dict(zip(z["queries"].tolist(), z["vectors"]))
    V = np.stack([vecs[q] for q in prompts[:AUDIT_ROWS]]).astype(np.float32)
    pool = audit_pool(prompts, ids, V)
    print(f"  audit pool: {len(pool)} candidate pairs, {100 * pool.y.mean():.1f}% same class "
          f"[{time.time() - t0:.0f}s]", flush=True)

    start, n = WINDOWS[window]
    E40.DATA = DATA
    wl = E40.load_stream(start, n)
    qqp = train_metas(seed=seed, keys=("C2_bi_geo_lgb",))["C2_bi_geo_lgb"]
    rows, curves, done = [], [], set()
    part = REVISION_DIR / f"exp46_partial_w{window}.json"
    if part.exists():                                   # resume after a lost runtime
        import json
        d = json.loads(part.read_text(encoding="utf8"))
        rows, curves, done = d["rows"], d["curves"], {tuple(x) for x in d["done"]}
        print(f"  resumed: {len(done)} (budget, draw) cells already done", flush=True)

    def checkpoint():
        save_json({"rows": rows, "curves": curves, "done": sorted(done, key=str)}, part.name, directory=REVISION_DIR)

    def record(system, budget, draw, p_star, thr, stats, extra=None):
        r = {"system": system, "budget": budget, "draw": draw, "p_star": p_star, "threshold": thr,
             **{k: stats.get(k) for k in KEYS}, **(extra or {})}
        rows.append(r)
        return r

    with memoized_encoders() as memo:
        memo._bi_cache.update({q: vecs[q] for q in wl.queries()})
        if ("base", 0) not in done:
            for thr in C1_GRID:                                   # budget-free C1 curve
                r = serve("C1_bi_only", wl, NullBackend(), None, bi_threshold=thr)
                curves.append({"system": "C1", "budget": None, "threshold": thr, **{k: r.stats.get(k) for k in KEYS}})
            r = serve("C2_bi_geo_lgb", wl, NullBackend(), {"C2_bi_geo_lgb": qqp}, meta_threshold=C.META_PROB_THRESHOLD)
            record("C2-qqp-default", None, None, None, C.META_PROB_THRESHOLD, r.stats)
            done.add(("base", 0)); checkpoint()
        print(f"  C1 grid + QQP default done [{time.time() - t0:.0f}s]", flush=True)

        for budget in BUDGETS:
            for draw in range(DRAWS):
                if (budget, draw) in done:
                    continue
                rng = np.random.default_rng(seed * 1000 + budget + draw)
                sel = pool.iloc[rng.choice(len(pool), size=min(budget, len(pool)), replace=False)].reset_index(drop=True)
                X, y = features(prompts, V, sel), sel.y.to_numpy()
                perm = rng.permutation(len(sel))
                cut = int(0.8 * len(sel))
                tr, ho = perm[:cut], perm[cut:]
                c2 = fit_final(X[tr], y[tr], GROUPS, seed=seed + draw) if len(np.unique(y[tr])) == 2 else None
                s_c2 = c2.pipeline.predict_proba(X[ho])[:, 1] if c2 is not None else None
                s_qqp = qqp.pipeline.predict_proba(X)[:, 1]
                for p_star in P_STARS:
                    jobs = [("C1-cal", pick_threshold(sel.cos.to_numpy(), y, p_star), "C1_bi_only", None),
                            ("C2-in", pick_threshold(s_c2, y[ho], p_star) if c2 is not None else None, "C2_bi_geo_lgb", c2),
                            ("C2-qqp-cal", pick_threshold(s_qqp, y, p_star), "C2_bi_geo_lgb", qqp)]
                    for system, thr, key, model in jobs:
                        if thr is None:
                            record(system, budget, draw, p_star, None, {}, {"infeasible": True})
                            continue
                        kw = {"bi_threshold": thr} if key == "C1_bi_only" else {"meta_threshold": thr}
                        r = serve(key, wl, NullBackend(), {key: model} if model is not None else None, **kw)
                        record(system, budget, draw, p_star, thr, r.stats, {"audit_pairs": int(len(sel)),
                                                                          "audit_pos_rate": float(y.mean())})
                if budget in CURVE_BUDGETS and draw == 0 and c2 is not None:
                    for thr in C2_GRID:
                        r = serve("C2_bi_geo_lgb", wl, NullBackend(), {"C2_bi_geo_lgb": c2}, meta_threshold=thr)
                        curves.append({"system": "C2-in", "budget": budget, "threshold": thr,
                                       **{k: r.stats.get(k) for k in KEYS}})
                done.add((budget, draw)); checkpoint()
                print(f"  budget {budget} draw {draw} done [{time.time() - t0:.0f}s]", flush=True)

    out = {"manifest": hardware_manifest({"window": window, "start": start, "n": n, "audit_rows": AUDIT_ROWS,
                                          "budgets": list(BUDGETS), "draws": DRAWS, "p_stars": list(P_STARS),
                                          "generation": "none", "latency_valid": False}, seed=seed),
           "audit_pool": {"pairs": int(len(pool)), "positive_rate": float(pool.y.mean())},
           "workload": wl.meta, "rows": rows, "curves": curves}
    save_result(out, "exp46_audit_budget", tag=tag or f"w{window}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(a.window, tag=a.tag)
