"""
Experiment 26 -- RiskPriced: capacity management that prices false hits.

Every published eviction or admission policy values a cached entry by how
often it is hit. In a semantic cache an entry can also be harmful: one that
keeps being matched in an ambiguous neighbourhood produces false hits.
RiskPriced values each served hit at p - lambda (1 - p), where p is the
decision layer's confidence for that hit and lambda the cost of a false hit,
evicts the lowest-valued entry, optionally gates admission, and optionally
purges entries whose value turns negative.

Protocol (as in experiments 18 and 24), separately for each false-hit cost
lambda in LAMBDAS (the operator knows its cost, and the policy is told it):
* a grid is scored on the tuning stream (seed 1042) at the tuning
  capacities by risk-adjusted saving at lambda; every row is reported;
* the best setting is frozen and evaluated on seeds 42-46 at capacities
  200-1000 with C2 serving, with ablations (no purge, no admission);
* the reference policies (LRU, LFU, SphereLFU, SD-Admit as frozen in
  experiment 24) are re-run in the same environment, and seed-level paired
  tests compare RiskPriced with each at the same lambda.

Encoders are memoised by text (optionally from recorded embeddings);
latency is not reported here.

Run: python -m semcache.experiments.exp26_risk_priced --workers 10 --vectors FILE
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np

from .. import config as C
from ..cache import (RiskPricedPolicy, SemanticAdmissionPolicy, SphereLFUPolicy,
                     make_policy)
from ..encoders import warmup
from ..stats import paired_seed_test, seed_ci
from ._common import (add_run_args, banner, get_stream, hardware_manifest,
                      memoized_encoders, save_result, train_metas)
from .exp18_eviction import CAPACITIES, EVAL_SEEDS, KEY, TUNE_CAPACITIES, TUNE_SEED, _run

LAMBDAS = (1.0, 2.0, 5.0, 10.0)
PRIORS = (0.5, 1.0, 2.0)
ADMISSION = (True, False)
PURGE = (True, False)
PERIOD_MULT = (4, 16)
SD_ADMIT = {"beta": 0.0, "cos_radius": 0.65, "period_mult": 16, "admission": True}  # exp24 frozen
REFS = ("LRU", "LFU", "SphereLFU", "SD-Admit")


def build(spec: dict, cap: int):
    k = spec["kind"]
    if k in ("LRU", "LFU"):
        return make_policy(k)
    if k == "SphereLFU":
        return SphereLFUPolicy(capacity=cap)
    if k == "SD-Admit":
        return SemanticAdmissionPolicy(beta=SD_ADMIT["beta"], cos_radius=SD_ADMIT["cos_radius"],
                                       admission=SD_ADMIT["admission"],
                                       period=SD_ADMIT["period_mult"] * cap)
    if k == "RiskPriced":
        return RiskPricedPolicy(lam=spec["lam"], prior=spec["prior"], admission=spec["admission"],
                                purge_negative=spec["purge"], period=spec["period_mult"] * cap)
    raise ValueError(k)


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
                t = tasks[i]
                out[i] = _run(wl, metas, t["cap"], build(t["spec"], t["cap"]), seed)
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


def ras(r: dict, lam: float) -> float:
    return r["risk_adjusted_saving_pct"][f"{lam:g}"]


def main(seeds=EVAL_SEEDS, n: int = 2000, capacities=CAPACITIES, tag: str = "",
         run=None, extra: dict | None = None) -> dict:
    banner("EXP 26 | RiskPriced: eviction and admission that price false hits")
    run = run or runner(1, None)

    # ---- tuning, per lambda, on the tuning stream only --------------------
    grid = [{"kind": "RiskPriced", "prior": p, "admission": a, "purge": g, "period_mult": m}
            for p, a, g, m in itertools.product(PRIORS, ADMISSION, PURGE, PERIOD_MULT)]
    tasks = [{"seed": TUNE_SEED, "cap": c, "spec": {**g, "lam": lam}}
             for lam in LAMBDAS for g in grid for c in TUNE_CAPACITIES]
    got = run(tasks, n)
    tuning, best = {}, {}
    k = 0
    for lam in LAMBDAS:
        rows = []
        for g in grid:
            rs = got[k:k + len(TUNE_CAPACITIES)]
            k += len(TUNE_CAPACITIES)
            rows.append({**g, "lam": lam, "per_capacity": [ras(r, lam) for r in rs],
                         "mean_risk_adjusted_saving_pct": float(np.mean([ras(r, lam) for r in rs]))})
        b = max(rows, key=lambda x: x["mean_risk_adjusted_saving_pct"])
        tuning[f"{lam:g}"] = {"grid": rows, "best": b}
        best[lam] = {kk: b[kk] for kk in ("kind", "prior", "admission", "purge", "period_mult", "lam")}
        print(f"  lambda {lam:g}: best {best[lam]} ({b['mean_risk_adjusted_saving_pct']:.2f}%)", flush=True)

    # ---- evaluation ------------------------------------------------------
    specs = {r: {"kind": r} for r in REFS}
    for lam in LAMBDAS:
        specs[f"RiskPriced@{lam:g}"] = best[lam]
        specs[f"RiskPriced@{lam:g}_no_purge"] = {**best[lam], "purge": False}
        specs[f"RiskPriced@{lam:g}_no_admission"] = {**best[lam], "admission": False}
    tasks = [{"seed": s, "cap": c, "spec": sp, "name": name}
             for s in seeds for c in capacities for name, sp in specs.items()]
    got = run(tasks, n)
    res = {name: {str(c): {} for c in capacities} for name in specs}
    for t, r in zip(tasks, got):
        res[t["name"]][str(t["cap"])][t["seed"]] = r

    summary, tests = {}, {}
    for lam in LAMBDAS:
        L = f"{lam:g}"
        summary[L], tests[L] = {}, {}
        names = [*REFS, f"RiskPriced@{L}", f"RiskPriced@{L}_no_purge", f"RiskPriced@{L}_no_admission"]
        for c in map(str, capacities):
            summary[L][c] = {nm: {"risk_adjusted_saving_pct": seed_ci([ras(res[nm][c][s], lam) for s in seeds]),
                                  "correct_hit_ratio_pct": seed_ci([res[nm][c][s]["correct_hit_ratio_pct"] for s in seeds]),
                                  "precision_of_hits_pct": seed_ci([res[nm][c][s]["precision_of_hits_pct"] for s in seeds]),
                                  "false_hit_rate_pct": seed_ci([res[nm][c][s]["false_hit_rate_pct"] for s in seeds])}
                             for nm in names}
            me = [ras(res[f"RiskPriced@{L}"][c][s], lam) for s in seeds]
            tests[L][c] = {ref: paired_seed_test(me, [ras(res[ref][c][s], lam) for s in seeds])
                           for ref in REFS}
        line = "  ".join(f"{c}:{tests[L][c]['SD-Admit'].get('delta_mean', 0):+.2f}"
                         f"{'*' if (tests[L][c]['SD-Admit'].get('t_p') or 1) < 0.05 else ''}"
                         for c in map(str, capacities))
        print(f"  lambda {L}: RiskPriced vs SD-Admit (risk-adjusted saving) {line}", flush=True)

    out = {"manifest": hardware_manifest({"n_stream": n, "config": KEY, "eval_seeds": list(seeds),
                                          "latency_valid": False, "lambdas": list(LAMBDAS),
                                          "grid": {"prior": list(PRIORS), "admission": list(ADMISSION),
                                                   "purge": list(PURGE), "period_mult": list(PERIOD_MULT)},
                                          "sd_admit": SD_ADMIT, **(extra or {})}),
           "tuning": tuning, "specs": specs, "per_seed": res, "summary": summary, "tests": tests}
    save_result(out, "exp26_risk_priced", tag=tag)
    return out


if __name__ == "__main__":
    ap = add_run_args(argparse.ArgumentParser())
    ap.add_argument("--seeds", type=int, nargs="+", default=list(EVAL_SEEDS))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--vectors", default=None)
    a = ap.parse_args()
    main(a.seeds, a.n, tag=a.tag, run=runner(a.workers, a.vectors),
         extra={"workers": a.workers, "vectors": a.vectors})
