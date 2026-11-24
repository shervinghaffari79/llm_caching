"""
Experiment 31 -- Semantic Decay with semantic admission (SemDemand).

One estimator, the decayed count of recent requests inside a cosine radius
(semantic demand), decides both which resident to evict and whether a missed
query is cached at all. Exact-key TinyLFU-style admission is the same policy
with the radius at 0.9999, so it is tuned and evaluated under the same budget.

Protocol, fixed before any run of this experiment:

Tuning (this phase) uses only data reserved for tuning:
  * QQP tuning streams, seeds 1042-1044 (2,000 requests), capacities 300, 500, 800;
  * shifted tuning streams 1042->1043, 1043->1044, 1044->1042 (first 1,000
    requests of each), capacities 300 and 500, so that the selected setting
    must also cope with a change of popular topics;
  * SemBenchmarkSearchQueries queries 100,000-109,999, capacities 500, 1,000, 2,000.
QQP and shifted streams are served by C2 at its default threshold; the log by
C1 at its in-domain threshold 0.895 (experiment 30, second round).
Grids. Ageing: exponential with half-life {0.5, 1, 2, 4} x capacity, or
TinyLFU-style integer halving every {1, 4, 10} x capacity requests.
Admission: off; "compete" (the newcomer's demand including the current
request must exceed the victim's; ties rejected); "second" (only the demand it
had before the current request counts, so first sightings stay out of a full
cache). SemDemand: radius at the acceptance cosine or 0.05 above it, all
admission modes. TinyLFU (radius 0.9999): admission "compete" or "second".
History length: four half-lives or two halving periods, at most 16 x capacity.
(A smoke run of an earlier draft with exponential ageing only showed that
admission without the "second" rule never rejects anything once residents
decay; that draft was replaced before any tuning run.)
Selection: for SemDemand and for TinyLFU separately, the setting with the
highest mean net hit ratio (correct-hit ratio minus false-hit rate) over all
eighteen tuning cells. Reference policies (LRU, LFU, SphereLFU, LeCaR,
Semantic Decay with fixed and tuned weights) run on the same cells.

Evaluation (later phase, not run here): fresh QQP seeds 47-51 at capacities
200-1000, five 10,000-query windows of the first 50,000 log queries, and the
five shifted evaluation streams of experiment 28; Holm-corrected paired tests
against the best of LFU, SphereLFU, LeCaR and TinyLFU. Success requires a
significant correct-hit gain on the fresh QQP seeds, a significant gain over
TinyLFU on the log, and no significant loss to LRU after a shift.

Run: python -m semcache.experiments.exp31_semantic_admission --workers 30
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

from .. import config as C
from ..cache import (LeCaRPolicy, SemanticDecayPolicy, SemanticDemandPolicy, SphereLFUPolicy,
                     StreamStats, make_policy)
from ..encoders import StageTimer, warmup
from ..llm import AnswerPool, LLMBackend
from ._common import get_stream, hardware_manifest, banner, memoized_encoders, save_result, train_metas
from .exp30_sd_redesign import _vectors, ensure_log_vectors, ensure_qqp_vectors, log_window

SERVE = {"qqp": ("C2_bi_geo_lgb", {}, C.BI_COSINE_THRESHOLD),
         "shift": ("C2_bi_geo_lgb", {}, C.BI_COSINE_THRESHOLD),
         "log": ("C1_bi_only", {"bi_threshold": 0.895}, 0.895)}     # key, kwargs, acceptance cosine
TUNE = {"qqp": {"seeds": (1042, 1043, 1044), "caps": (300, 500, 800)},
        "shift": {"pairs": ((1042, 1043), (1043, 1044), (1044, 1042)), "caps": (300, 500)},
        "log": {"start": 100_000, "n": 10_000, "caps": (500, 1000, 2000)}}
AGINGS = [("exp", h) for h in (0.5, 1, 2, 4)] + [("halve", m) for m in (1, 4, 10)]
ADMISSIONS = ("off", "compete", "second")
RADIUS_OFFSETS = (0.0, 0.05)
EXACT = 0.9999
REFS = ("LRU", "LFU", "SphereLFU", "LeCaR", "SemanticDecay_fixed", "SemanticDecay_tuned")
HALF, RECOVERY, SAMPLE_EVERY = 1000, 200, 50
CKPTS = (1000, 2500, 5000, 10000)


def specs() -> list[dict]:
    out = []
    for (ag, m), adm, off in itertools.product(AGINGS, ADMISSIONS, RADIUS_OFFSETS):
        out.append({"kind": "SemDemand", "aging": ag, "mult": m, "admission": adm, "radius_offset": off})
    for (ag, m), adm in itertools.product(AGINGS, ("compete", "second")):
        out.append({"kind": "TinyLFU", "aging": ag, "mult": m, "admission": adm})
    return out + [{"kind": r} for r in REFS]


def _demand(spec: dict, cap: int, radius: float) -> SemanticDemandPolicy:
    ag, m = spec["aging"], spec["mult"]
    w = int(min(4 * m * cap if ag == "exp" else 2 * m * cap, 16 * cap))
    return SemanticDemandPolicy(radius=radius, aging=ag,
                                half_life=(m * cap if ag == "exp" else None),
                                period=(int(m * cap) if ag == "halve" else 10 ** 9),
                                admission=spec["admission"], history=w)


def build(spec: dict, cap: int, data: str, seed: int):
    k = spec["kind"]
    acc = SERVE[data][2]
    if k in ("LRU", "LFU"):
        return make_policy(k)
    if k == "SphereLFU":
        return SphereLFUPolicy(cos_threshold=acc, capacity=cap)
    if k == "LeCaR":
        return LeCaRPolicy(cos_threshold=acc, capacity=cap, seed=seed)
    if k == "SemanticDecay_fixed":
        return SemanticDecayPolicy()
    if k == "SemanticDecay_tuned":
        return SemanticDecayPolicy(0.0, 0.0, 1.0)
    if k == "SemDemand":
        return _demand(spec, cap, min(0.999, acc + spec["radius_offset"]))
    if k == "TinyLFU":
        return _demand(spec, cap, EXACT)
    raise ValueError(k)


def run_one(wl, meta, cap: int, policy, key: str, seed: int, **kw) -> dict:
    from ..pipeline import SemanticCache
    cache = SemanticCache(C.CONFIGS[key], meta=meta, capacity=cap, policy=policy, **kw)
    backend = LLMBackend(mode="sim", pool=AnswerPool(), virtual_time=True, seed=seed)
    stats, timer = StreamStats(), StageTimer()
    red, corr, fals = [], [], []
    admitted = misses = 0                       # newcomers kept after the eviction step / all misses
    adm_full = miss_full = 0                    # the same, counted only for misses that arrive at a full cache
    ckpt: dict = {}                             # cumulative counts at request checkpoints (cold start from an empty cache)
    for i, ev in enumerate(wl.events):
        was_full = len(cache.cachebase) >= cap
        rec = cache.handle(ev.query, ev.topic_id, backend, stats, timer)
        corr.append(bool(rec["hit"] and rec["correct"]))
        fals.append(bool(rec["hit"] and not rec["correct"]))
        if not rec["hit"]:
            misses += 1
            kept = int(cache.cachebase.get(cache._next_id - 1) is not None)
            admitted += kept
            if was_full:
                miss_full += 1
                adm_full += kept
        if (i + 1) in CKPTS:
            ckpt[str(i + 1)] = {"correct": int(sum(corr)), "false": int(sum(fals)), "admitted": admitted}
        size = len(cache.cachebase)
        if size >= cap and i % SAMPLE_EVERY == 0:
            red.append(len(cache.live_topics) / size)
    s = stats.summary()
    n = max(1, s["n_requests"])
    c, f = 100.0 * float(np.mean(corr)), 100.0 * float(np.mean(fals))
    out = {"hit_ratio_pct": s["hit_ratio_pct"], "correct_hit_ratio_pct": c, "false_hit_rate_pct": f,
           "precision_of_hits_pct": s["precision_of_hits_pct"], "net_hit_ratio_pct": c - f,
           "evictions": s["evictions"], "cache_size_end": len(cache.cachebase),
           "distinct_groups_per_slot": float(np.mean(red)) if red else None,
           "admitted": admitted, "misses": misses, "admitted_per_1000_requests": 1000.0 * admitted / n,
           "admitted_full": adm_full, "misses_full": miss_full,
           "admitted_per_1000_requests_full": 1000.0 * adm_full / n,
           "ckpt": ckpt}
    if wl.meta.get("kind") == "shift":
        out.update({"second_half_correct_pct": 100.0 * float(np.mean(corr[HALF:])),
                    "recovery_correct_pct": 100.0 * float(np.mean(corr[HALF:HALF + RECOVERY])),
                    "second_half_false_pct": 100.0 * float(np.mean(fals[HALF:]))})
    return out


def _worker(tasks: list[dict]) -> list[dict]:
    from .exp28_shift_lecar import shifted_stream
    warmup()
    vecs = _vectors()
    out = [None] * len(tasks)
    groups: dict[tuple, list[int]] = {}
    for i, t in enumerate(tasks):
        groups.setdefault(tuple(t["data"]), []).append(i)
    with memoized_encoders() as memo:
        memo._bi_cache.update(vecs)
        for data, idx in groups.items():
            kind = data[0]
            key, kw, _ = SERVE[kind]
            if kind == "qqp":
                seed, wl = data[1], get_stream(data[1], 2000)
            elif kind == "shift":
                seed, wl = data[1], shifted_stream(data[1], data[2], 2000)
            else:
                seed, wl = C.SEED, log_window(data[1], data[2])
            meta = train_metas(seed=seed, keys=(key,)).get(key) if kind != "log" else None
            for i in idx:
                t = tasks[i]
                out[i] = run_one(wl, meta, t["cap"], build(t["spec"], t["cap"], kind, seed), key, seed, **kw)
    return out


def _task_key(t: dict) -> str:
    return json.dumps([t["data"], t["cap"], t["spec"]], sort_keys=True)


def run_tasks(tasks: list[dict], workers: int, checkpoint=None) -> list[dict]:
    """Runs the tasks, appending every finished result to `checkpoint` (JSON
    lines) so that an interrupted run resumes where it stopped."""
    out = [None] * len(tasks)
    done_keys = {}
    if checkpoint is not None and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done_keys[rec["key"]] = rec["result"]
    todo = []
    for i, t in enumerate(tasks):
        k = _task_key(t)
        if k in done_keys:
            out[i] = done_keys[k]
        else:
            todo.append(i)
    print(f"    {len(tasks) - len(todo)} runs restored from checkpoint, {len(todo)} to run", flush=True)

    def record(idx, results):
        for i, r in zip(idx, results):
            out[i] = r
        if checkpoint is not None:
            with open(checkpoint, "a", encoding="utf8") as fh:
                for i, r in zip(idx, results):
                    fh.write(json.dumps({"key": _task_key(tasks[i]), "result": r}) + "\n")

    if workers <= 1:
        for i in todo:
            record([i], _worker([tasks[i]]))
        return out
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    by_data: dict[tuple, list[int]] = {}
    for i in todo:
        by_data.setdefault(tuple(tasks[i]["data"]), []).append(i)
    size = max(1, len(todo) // (workers * 3))
    chunks = [idx[j:j + size] for idx in by_data.values() for j in range(0, len(idx), size)]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_worker, [tasks[i] for i in ch]): ch for ch in chunks}
        done = len(tasks) - len(todo)
        for f in as_completed(futs):
            record(futs[f], f.result())
            done += len(futs[f])
            print(f"    {done}/{len(tasks)} runs", flush=True)
    return out


def tune(workers: int, tag: str = "") -> dict:
    banner("EXP 31 | SemDemand tuning (tuning data only)")
    ensure_log_vectors([(TUNE["log"]["start"], TUNE["log"]["n"])])
    seeds = sorted({*TUNE["qqp"]["seeds"], *(s for p in TUNE["shift"]["pairs"] for s in p)})
    ensure_qqp_vectors(seeds)
    for s in seeds:                                     # train once here, so workers only load
        train_metas(seed=s, keys=(SERVE["qqp"][0],))
    cells = ([(("qqp", s), c) for s in TUNE["qqp"]["seeds"] for c in TUNE["qqp"]["caps"]]
             + [(("shift", a, b), c) for a, b in TUNE["shift"]["pairs"] for c in TUNE["shift"]["caps"]]
             + [(("log", TUNE["log"]["start"], TUNE["log"]["n"]), c) for c in TUNE["log"]["caps"]])
    sp = specs()
    tasks = [{"data": list(d), "cap": c, "spec": s} for s in sp for d, c in cells]
    from ._common import REVISION_DIR
    ckpt = REVISION_DIR / f"exp31_semantic_admission_tune_checkpoint{('_' + tag) if tag else ''}.jsonl"
    got = run_tasks(tasks, workers, checkpoint=ckpt)
    rows = []
    for k, s in enumerate(sp):
        res = got[k * len(cells):(k + 1) * len(cells)]
        cell_rows = [{"data": d[0], "stream": list(d[1:]), "cap": c, **r} for (d, c), r in zip(cells, res)]
        mean = lambda kind, m: float(np.mean([r[m] for r in cell_rows if r["data"] == kind and r.get(m) is not None]))
        rows.append({"spec": s, "cells": cell_rows,
                     "objective_net_hit_ratio_pct": float(np.mean([r["net_hit_ratio_pct"] for r in cell_rows])),
                     **{f"{kind}_net": mean(kind, "net_hit_ratio_pct") for kind in ("qqp", "shift", "log")},
                     **{f"{kind}_correct": mean(kind, "correct_hit_ratio_pct") for kind in ("qqp", "shift", "log")},
                     **{f"{kind}_false": mean(kind, "false_hit_rate_pct") for kind in ("qqp", "shift", "log")},
                     "shift_second_half_correct": mean("shift", "second_half_correct_pct"),
                     "shift_recovery_correct": mean("shift", "recovery_correct_pct")})
    best = {k: max((r for r in rows if r["spec"]["kind"] == k), key=lambda r: r["objective_net_hit_ratio_pct"])
            for k in ("SemDemand", "TinyLFU")}
    show = sorted(rows, key=lambda r: -r["objective_net_hit_ratio_pct"])
    print("  top settings and references (objective | net qqp/shift/log | correct qqp/shift/log | false qqp/shift/log | shift 2nd half / recovery):")
    for r in show[:6] + [r for r in rows if r["spec"]["kind"] in REFS] + [best["TinyLFU"]]:
        print(f"    {str(r['spec']):100s} {r['objective_net_hit_ratio_pct']:6.2f} | "
              f"{r['qqp_net']:5.2f} {r['shift_net']:5.2f} {r['log_net']:5.2f} | "
              f"{r['qqp_correct']:5.2f} {r['shift_correct']:5.2f} {r['log_correct']:5.2f} | "
              f"{r['qqp_false']:4.2f} {r['shift_false']:4.2f} {r['log_false']:4.2f} | "
              f"{r['shift_second_half_correct']:5.2f} {r['shift_recovery_correct']:5.2f}", flush=True)
    print(f"  selected SemDemand: {best['SemDemand']['spec']}")
    print(f"  selected TinyLFU:   {best['TinyLFU']['spec']}")
    out = {"manifest": hardware_manifest({"phase": "tune",
                                          "tune": {k: {kk: (list(map(list, v)) if kk == "pairs" else
                                                            list(v) if isinstance(v, tuple) else v)
                                                       for kk, v in d.items()} for k, d in TUNE.items()},
                                          "grid": {"agings": AGINGS, "admissions": ADMISSIONS,
                                                   "radius_offset": RADIUS_OFFSETS, "exact_radius": EXACT},
                                          "serve": {k: [v[0], v[1], v[2]] for k, v in SERVE.items()},
                                          "objective": "mean net hit ratio over the eighteen tuning cells"}),
           "rows": rows, "selected": {k: v["spec"] for k, v in best.items()}}
    save_result(out, "exp31_semantic_admission_tune", tag=tag)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true", help="tiny grid on a few short cells")
    a = ap.parse_args()
    if a.smoke:
        AGINGS, RADIUS_OFFSETS = [("exp", 1), ("halve", 4)], (0.0,)
        TUNE["qqp"]["seeds"], TUNE["qqp"]["caps"] = (1043,), (300,)
        TUNE["shift"]["pairs"], TUNE["shift"]["caps"] = ((1042, 1043),), (300,)
        TUNE["log"]["n"], TUNE["log"]["caps"] = 3000, (500,)
        REFS = ("LFU",)
        tune(a.workers, tag="smoke")
    else:
        tune(a.workers, tag=a.tag)
