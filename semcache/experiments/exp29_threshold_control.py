"""
Experiment 29 -- does eviction add precision beyond the decision threshold?

Experiment 18 found that Semantic Decay with weights tuned on the tuning
stream (seed 1042, frozen at w = (0, 0, 1), i.e. the boost term only) serves
fewer false hits than LRU, LFU and SphereLFU at small capacities, at the cost
of some correct hits. A higher decision threshold also trades correct hits for
fewer false hits, so the question is whether the eviction policy moves the
operating frontier or only moves along it.

Protocol, C2 serving, capacities CAPS, thresholds THRESHOLDS (full re-runs):
* frontier: for each seed, the correct-hit ratio of each policy is
  interpolated at matched false-hit rates along its threshold curve;
* primary endpoint (fixed before the run): for each capacity and false-hit
  cost lambda, the threshold of every policy is chosen on the tuning stream
  (seed 1042) by token risk-adjusted saving and evaluated on seeds 42-46;
  Semantic Decay is compared with each reference by seed-level paired tests,
  Holm-adjusted over all such comparisons;
* secondary: the best risk-adjusted saving when each seed's threshold is
  chosen in hindsight (an upper bound for every policy alike);
* mechanism: under LFU at the default threshold, the false-hit rate of hits
  grouped by the served entry's boost score (mean cosine to the most-hit
  entries) at the moment of the hit.

Encoders are memoised by text (optionally from recorded embeddings);
latency is not reported here.

Run: python -m semcache.experiments.exp29_threshold_control --workers 8 --vectors FILE
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .. import config as C
from ..cache import SemanticDecayPolicy, SphereLFUPolicy, make_policy
from ..encoders import warmup
from ..stats import holm, paired_seed_test, seed_ci
from ._common import (REVISION_DIR, banner, get_stream, hardware_manifest, make_backend,
                      memoized_encoders, save_result, serve, train_metas)
from .exp18_eviction import EVAL_SEEDS, KEY, TUNE_SEED, _metrics

CAPS = (200, 300, 400, 600)
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95,
              0.97, 0.98, 0.99, 0.995)
LAMBDAS = (1.0, 2.0, 5.0, 10.0)
FH_TARGETS = (0.5, 1.0, 1.5, 2.0)
SD_W = (0.0, 0.0, 1.0)            # experiment 18 tuning on seed 1042, frozen
POLICIES = ("LRU", "LFU", "SphereLFU", "SemanticDecay_tuned")
REFS = POLICIES[:-1]
SD = POLICIES[-1]


def build(name: str, cap: int):
    if name in ("LRU", "LFU"):
        return make_policy(name)
    if name == "SphereLFU":
        return SphereLFUPolicy(capacity=cap)
    if name == SD:
        return SemanticDecayPolicy(*SD_W)
    raise ValueError(name)


def _one(wl, metas, t: dict) -> dict:
    r = serve(KEY, wl, make_backend(seed=t["seed"]), metas, keep_events=False,
              capacity=t["cap"], policy=build(t["policy"], t["cap"]),
              meta_threshold=t["thr"])
    return _metrics(r.stats)


def _worker(tasks: list[dict], n: int, vectors_path: str | None) -> list[dict]:
    warmup()
    vecs = {}
    if vectors_path:
        z = np.load(vectors_path, allow_pickle=True)
        vecs = dict(zip(z["queries"].tolist(), z["vectors"]))
    out: list[dict | None] = [None] * len(tasks)
    by_seed: dict[int, list[int]] = {}
    for i, t in enumerate(tasks):
        by_seed.setdefault(t["seed"], []).append(i)
    with memoized_encoders() as memo:
        memo._bi_cache.update(vecs)
        for seed, idx in by_seed.items():
            wl = get_stream(seed, n)
            metas = train_metas(seed=seed, keys=(KEY,))
            for i in idx:
                out[i] = _one(wl, metas, tasks[i])
    return out


def runner(workers: int, vectors_path: str | None):
    def run(tasks: list[dict], n: int) -> list[dict]:
        if workers <= 1:
            return _worker(tasks, n, vectors_path)
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        order = sorted(range(len(tasks)), key=lambda i: tasks[i]["seed"])
        size = max(1, len(tasks) // (workers * 3))
        chunks = [order[i:i + size] for i in range(0, len(order), size)]
        out: list[dict | None] = [None] * len(tasks)
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(_worker, [tasks[i] for i in ch], n, vectors_path): ch for ch in chunks}
            done = 0
            for f in futs:
                for i, r in zip(futs[f], f.result()):
                    out[i] = r
                done += len(futs[f])
                print(f"    {done}/{len(tasks)} runs", flush=True)
        return out
    return run


# ------------------------------------------------------------------ mechanism
class BoostProbe:
    """Wraps an eviction policy and records, at every served hit, the served
    entry's boost score as Semantic Decay defines it (mean cosine to the
    top-K most-hit live entries), computed before the hit is counted."""

    def __init__(self, inner, top_k: int = C.SD_FREQ_TOP_K):
        self.inner, self.top_k = inner, top_k
        self.name = getattr(inner, "name", "probe")
        self.vecs: dict[int, np.ndarray] = {}
        self.hits: dict[int, int] = {}
        self.boost_at_hit: list[float] = []

    def __getattr__(self, k):                      # hooks the inner policy lacks stay absent
        return getattr(self.inner, k)

    def on_query(self, vec):
        self.inner.on_query(vec)

    def on_insert(self, entry, vec):
        self.vecs[entry.eid] = np.asarray(vec, dtype=np.float32)
        self.inner.on_insert(entry, vec)

    def on_hit(self, entry, vec):
        top = [i for i, _ in sorted(self.hits.items(), key=lambda kv: -kv[1])[:self.top_k]
               if i in self.vecs]
        v = self.vecs.get(entry.eid)
        self.boost_at_hit.append(float((np.vstack([self.vecs[i] for i in top]) @ v).mean())
                                 if top and v is not None else float("nan"))
        self.hits[entry.eid] = self.hits.get(entry.eid, 0) + 1
        self.inner.on_hit(entry, vec)

    def on_evict(self, eid):
        self.vecs.pop(eid, None)
        self.hits.pop(eid, None)
        self.inner.on_evict(eid)

    def choose_victim(self, cache, vb):
        return self.inner.choose_victim(cache, vb)


def mechanism(seeds, n: int, vectors_path: str | None, cap: int = 200) -> dict:
    warmup()
    vecs = {}
    if vectors_path:
        z = np.load(vectors_path, allow_pickle=True)
        vecs = dict(zip(z["queries"].tolist(), z["vectors"]))
    per_seed = {}
    with memoized_encoders() as memo:
        memo._bi_cache.update(vecs)
        for seed in seeds:
            wl = get_stream(seed, n)
            metas = train_metas(seed=seed, keys=(KEY,))
            probe = BoostProbe(make_policy("LFU"))
            r = serve(KEY, wl, make_backend(seed=seed), metas, keep_events=True,
                      capacity=cap, policy=probe)
            hits = [e for e in r.per_event if e["hit"]]
            assert len(hits) == len(probe.boost_at_hit)
            b = np.array(probe.boost_at_hit)
            false = np.array([not e["correct"] for e in hits])
            ok = ~np.isnan(b)
            qs = np.quantile(b[ok], [0.25, 0.5, 0.75])
            band = np.digitize(b[ok], qs)
            per_seed[str(seed)] = {
                "n_hits": int(ok.sum()),
                "false_hit_pct_by_boost_quartile": [float(100 * false[ok][band == q].mean())
                                                    for q in range(4)],
                "mean_boost_false": float(b[ok][false[ok]].mean()) if false[ok].any() else None,
                "mean_boost_correct": float(b[ok][~false[ok]].mean())}
    q = np.array([v["false_hit_pct_by_boost_quartile"] for v in per_seed.values()])
    return {"capacity": cap, "policy": "LFU", "threshold": C.META_PROB_THRESHOLD,
            "per_seed": per_seed,
            "false_hit_pct_by_boost_quartile": [seed_ci(list(q[:, j])) for j in range(4)],
            "lowest_vs_highest_quartile": paired_seed_test(list(q[:, 0]), list(q[:, 3]))}


# ------------------------------------------------------------------ analysis
def ras(r: dict, lam: float) -> float:
    return r["risk_adjusted_saving_pct"][f"{lam:g}"]


def interp_correct(rows: list[dict], fh: float):
    pts = sorted((r["false_hit_rate_pct"], r["correct_hit_ratio_pct"]) for r in rows)
    xs = [a for a, _ in pts]
    if fh < xs[0] or fh > xs[-1]:
        return None
    return float(np.interp(fh, xs, [b for _, b in pts]))


def main(seeds=EVAL_SEEDS, n: int = 2000, tag: str = "", workers: int = 1,
         vectors: str | None = None) -> dict:
    banner("EXP 29 | Eviction versus decision threshold: frontier control")
    run = runner(workers, vectors)
    all_seeds = (TUNE_SEED, *seeds)
    tasks = [{"seed": s, "cap": c, "policy": p, "thr": t}
             for s in all_seeds for c in CAPS for p in POLICIES for t in THRESHOLDS]
    got = run(tasks, n)
    R: dict = {p: {str(c): {str(s): [] for s in all_seeds} for c in CAPS} for p in POLICIES}
    for t, r in zip(tasks, got):
        R[t["policy"]][str(t["cap"])][str(t["seed"])].append({"threshold": t["thr"], **r})

    S = [str(s) for s in seeds]
    tests_primary, tests_hind, tests_front = {}, {}, {}
    primary, hindsight, frontier = {}, {}, {}
    for c in map(str, CAPS):
        primary[c], hindsight[c], frontier[c] = {}, {}, {}
        for lam in LAMBDAS:
            L = f"{lam:g}"
            primary[c][L], hindsight[c][L] = {}, {}
            for p in POLICIES:
                tune_rows = R[p][c][str(TUNE_SEED)]
                thr = max(tune_rows, key=lambda r: ras(r, lam))["threshold"]
                vals = [ras(next(r for r in R[p][c][s] if r["threshold"] == thr), lam) for s in S]
                best = [max(ras(r, lam) for r in R[p][c][s]) for s in S]
                primary[c][L][p] = {"threshold": thr, "per_seed": vals, "ras": seed_ci(vals)}
                hindsight[c][L][p] = {"per_seed": best, "ras": seed_ci(best)}
            for ref in REFS:
                tests_primary[f"{c}|{L}|{ref}"] = paired_seed_test(
                    primary[c][L][SD]["per_seed"], primary[c][L][ref]["per_seed"])
                tests_hind[f"{c}|{L}|{ref}"] = paired_seed_test(
                    hindsight[c][L][SD]["per_seed"], hindsight[c][L][ref]["per_seed"])
        for fh in FH_TARGETS:
            F = f"{fh:g}"
            frontier[c][F] = {p: [interp_correct(R[p][c][s], fh) for s in S] for p in POLICIES}
            for ref in REFS:
                pairs = [(a, b) for a, b in zip(frontier[c][F][SD], frontier[c][F][ref])
                         if a is not None and b is not None]
                if len(pairs) >= 3:
                    tests_front[f"{c}|{F}|{ref}"] = paired_seed_test(
                        [a for a, _ in pairs], [b for _, b in pairs])
    for tests in (tests_primary, tests_hind, tests_front):
        adj = holm({k: v.get("t_p") for k, v in tests.items()})
        for k, v in tests.items():
            v["holm_p"] = adj.get(k)

    print("  primary (threshold chosen on seed 1042), SD minus reference, risk-adjusted saving:")
    for k, v in tests_primary.items():
        print(f"    cap|lambda|ref {k}: {v['delta_mean']:+.2f}  holm p {v['holm_p']:.3g}")
    print("  frontier (correct hits at matched false-hit rate), SD minus reference:")
    for k, v in tests_front.items():
        print(f"    cap|fh|ref {k}: {v['delta_mean']:+.2f}  holm p {v['holm_p']:.3g}")

    mech = mechanism(seeds, n, vectors)
    print("  mechanism, LFU cap 200: false-hit % by boost quartile (low->high):",
          [round(x["mean"], 2) for x in mech["false_hit_pct_by_boost_quartile"]])

    out = {"manifest": hardware_manifest({"seeds": list(seeds), "tune_seed": TUNE_SEED, "n": n,
                                          "caps": list(CAPS), "thresholds": list(THRESHOLDS),
                                          "lambdas": list(LAMBDAS), "sd_weights": list(SD_W),
                                          "primary_endpoint": "tuned-threshold RAS, SD vs each reference, Holm"}),
           "curves": R, "primary": primary, "hindsight": hindsight, "frontier": frontier,
           "tests": {"primary": tests_primary, "hindsight": tests_hind, "frontier": tests_front},
           "mechanism": mech}
    save_result(out, "exp29_threshold_control", tag=tag)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--vectors", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true", help="one seed, two thresholds, cap 200")
    a = ap.parse_args()
    exp18 = REVISION_DIR / "exp18_eviction_m2.json"
    if exp18.exists():
        assert tuple(json.load(open(exp18))["tuning"]["best3"]["w"]) == SD_W
    if a.smoke:
        CAPS, THRESHOLDS = (200,), (0.5, 0.9)
        main(seeds=(42, 43, 44), n=a.n, tag="smoke", workers=a.workers, vectors=a.vectors)
    else:
        main(n=a.n, tag=a.tag, workers=a.workers, vectors=a.vectors)
