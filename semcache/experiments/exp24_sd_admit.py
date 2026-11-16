"""
Experiment 24 -- SD-Admit: Semantic Decay with frequency, semantic demand and
admission.

Protocol (as in experiment 18):
* every setting on a grid is scored on the tuning stream (seed 1042, outside
  the evaluation seeds) at the tuning capacities; all grid rows are reported;
* the best setting is frozen and evaluated on seeds 42-46 at capacities
  200-1000 with C2 serving;
* two ablations isolate the parts: the frozen setting without admission, and
  the frozen setting with beta = 0 (own hits only, i.e. LFU with admission);
* seed-level paired tests against LRU, LFU, SphereLFU and tuned Semantic Decay
  (read from experiment 18's output, same streams and serving), and the share
  of the clairvoyant headroom (experiment 23) that is captured.

Encoders are memoised by text; latency is not reported here.

Run: python -m semcache.experiments.exp24_sd_admit
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

from .. import config as C
from ..cache import SemanticAdmissionPolicy
from ..encoders import warmup
from ..stats import paired_seed_test, seed_ci
from ._common import (add_run_args, banner, get_stream, hardware_manifest,
                      memoized_encoders, save_result, train_metas)
from .exp18_eviction import CAPACITIES, EVAL_SEEDS, KEY, TUNE_CAPACITIES, TUNE_SEED, _run

BETAS = (0.0, 0.25, 0.5, 1.0, 2.0)
RADII = (0.65, C.BI_COSINE_THRESHOLD, 0.80)
PERIOD_MULT = (1, 4, 16)
ADMISSION = (True, False)
REFS = ("LRU", "LFU", "SphereLFU", "SemanticDecay_tuned")


def make(cfg: dict, cap: int) -> SemanticAdmissionPolicy:
    return SemanticAdmissionPolicy(beta=cfg["beta"], cos_radius=cfg["cos_radius"],
                                   admission=cfg["admission"],
                                   period=cfg["period_mult"] * cap)


def grid() -> list[dict]:
    return [{"beta": b, "cos_radius": r, "period_mult": pm, "admission": adm}
            for b, r, pm, adm in itertools.product(BETAS, RADII, PERIOD_MULT, ADMISSION)]


def run_tasks(tasks: list[dict], n: int = 2000) -> list[dict]:
    """Run (seed, cap, cfg) tasks; streams and models are loaded once per seed.
    Any runner with this signature can replace it (e.g. one that fans the
    tasks out over remote workers)."""
    out: list[dict | None] = [None] * len(tasks)
    by_seed: dict[int, list[int]] = {}
    for i, t in enumerate(tasks):
        by_seed.setdefault(t["seed"], []).append(i)
    with memoized_encoders():
        for seed, idx in by_seed.items():
            wl = get_stream(seed, n)
            metas = train_metas(seed=seed, keys=(KEY,))
            for i in idx:
                t = tasks[i]
                out[i] = _run(wl, metas, t["cap"], make(t["cfg"], t["cap"]), seed)
    return out


def _load_vectors(path: str | None) -> dict:
    if not path:
        return {}
    z = np.load(path, allow_pickle=True)
    return dict(zip(z["queries"].tolist(), z["vectors"]))


def _worker(tasks: list[dict], n: int, vectors_path: str | None) -> list[dict]:
    """Runs in a separate process: memoised encoders pre-filled with the
    recorded stream embeddings, so every process makes identical decisions."""
    warmup()
    vecs = _load_vectors(vectors_path)
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
                out[i] = _run(wl, metas, t["cap"], make(t["cfg"], t["cap"]), seed)
    return out


def parallel_runner(workers: int, vectors_path: str | None):
    """Splits tasks into seed-homogeneous chunks over `workers` processes."""
    def run(tasks: list[dict], n: int) -> list[dict]:
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


def tune(n: int, runner=run_tasks) -> dict:
    cfgs = grid()
    tasks = [{"seed": TUNE_SEED, "cap": c, "cfg": cfg} for cfg in cfgs for c in TUNE_CAPACITIES]
    got = runner(tasks, n)
    rows = []
    for k, cfg in enumerate(cfgs):
        vals = [got[k * len(TUNE_CAPACITIES) + j]["correct_hit_ratio_pct"]
                for j in range(len(TUNE_CAPACITIES))]
        rows.append({**cfg, "per_capacity": vals, "mean_correct_hit_ratio_pct": float(np.mean(vals))})
    best = max(rows, key=lambda x: x["mean_correct_hit_ratio_pct"])
    print(f"  tuning ({len(rows)} settings): best {best}", flush=True)
    return {"tune_seed": TUNE_SEED, "capacities": list(TUNE_CAPACITIES),
            "grid": rows, "best": best}


def main(exp18_file: str, exp23_file: str | None, seeds=EVAL_SEEDS, n: int = 2000,
         capacities=CAPACITIES, tag: str = "", runner=run_tasks, extra: dict | None = None) -> dict:
    banner("EXP 24 | SD-Admit: frequency + semantic demand + admission")
    warmup()
    tuned = tune(n, runner)          # completes before any evaluation run
    b = {k: tuned["best"][k] for k in ("beta", "cos_radius", "period_mult", "admission")}
    variants = {"SD-Admit": b,
                "SD-Admit_no_admission": {**b, "admission": False},
                "SD-Admit_no_demand": {**b, "beta": 0.0}}
    tasks = [{"seed": s, "cap": c, "cfg": cfg, "variant": v}
             for s in seeds for c in capacities for v, cfg in variants.items()]
    got = runner(tasks, n)
    res = {v: {str(c): {} for c in capacities} for v in variants}
    for t, r in zip(tasks, got):
        res[t["variant"]][str(t["cap"])][t["seed"]] = r
    print(f"  evaluation: {len(tasks)} runs done", flush=True)

    online = json.load(open(exp18_file, encoding="utf8"))["per_seed"]
    oracle = json.load(open(exp23_file, encoding="utf8"))["per_seed"] if exp23_file else None
    m = "correct_hit_ratio_pct"
    summary, tests = {}, {}
    for c in map(str, capacities):
        summary[c] = {v: seed_ci([res[v][c][s][m] for s in seeds]) for v in variants}
        for ref in REFS:
            summary[c][ref] = seed_ci([online[ref][c][str(s)][m] for s in seeds])
        if oracle:
            summary[c]["Clairvoyant"] = seed_ci([oracle[c][str(s)][m] for s in seeds])
        tests[c] = {}
        for v in variants:
            tests[c][v] = {ref: paired_seed_test([res[v][c][s][m] for s in seeds],
                                                 [online[ref][c][str(s)][m] for s in seeds])
                           for ref in REFS}
        if oracle:
            lfu, orc = summary[c]["LFU"]["mean"], summary[c]["Clairvoyant"]["mean"]
            gap = orc - lfu
            summary[c]["headroom_captured_pct"] = (100.0 * (summary[c]["SD-Admit"]["mean"] - lfu) / gap
                                                   if gap > 0.25 else None)
        t = tests[c]["SD-Admit"]
        print(f"  capacity {c:>4s}: SD-Admit {summary[c]['SD-Admit']['mean']:.2f}%  "
              + "  ".join(f"vs {r} {t[r].get('delta_mean', 0):+.2f}"
                          f"{'*' if (t[r].get('t_p') or 1) < 0.05 else ''}" for r in REFS))

    out = {"manifest": hardware_manifest({"n_stream": n, "config": KEY, "eval_seeds": list(seeds),
                                          "latency_valid": False, "online_results": exp18_file,
                                          "clairvoyant_results": exp23_file,
                                          **(extra or {}),
                                          "grid": {"beta": list(BETAS), "cos_radius": list(RADII),
                                                   "period_mult": list(PERIOD_MULT),
                                                   "admission": list(ADMISSION)}}),
           "tuning": tuned, "variants": variants, "per_seed": res,
           "summary": summary, "tests": tests}
    save_result(out, "exp24_sd_admit", tag=tag)
    return out


if __name__ == "__main__":
    ap = add_run_args(argparse.ArgumentParser())
    rd = C.ROOT / "results/raw/revision"
    ap.add_argument("--exp18", default=str(rd / "exp18_eviction_m2.json"))
    ap.add_argument("--exp23", default=str(rd / "exp23_eviction_headroom_m2.json"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(EVAL_SEEDS))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=0,
                    help="run the simulations in this many processes")
    ap.add_argument("--vectors", default=None,
                    help=".npz of recorded stream embeddings (queries, vectors)")
    a = ap.parse_args()
    runner = parallel_runner(a.workers, a.vectors) if a.workers else run_tasks
    main(a.exp18, a.exp23, a.seeds, a.n, tag=a.tag, runner=runner,
         extra={"workers": a.workers, "vectors": a.vectors})
