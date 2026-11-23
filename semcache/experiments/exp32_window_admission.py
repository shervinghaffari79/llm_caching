"""
Experiment 32 -- Semantic Decay with a semantic admission window (WSemDemand).

New entries enter a small LRU window; an entry leaving the window is promoted
to the main segment only if its semantic evidence (aged count of other
requests inside its cosine radius, cached or not) exceeds that of the main
entry with the least evidence. W-TinyLFU-style exact-key admission is the same
policy with the radius at 0.9999 and is tuned under the same budget.

Protocol, fixed before any run of this experiment (`--phase all` runs both
phases in one job; the selection rule and the success criteria below are
applied mechanically):

Tuning: the eighteen tuning cells of experiment 31 (QQP seeds 1042-1044,
shifted tuning streams 1042->1043, 1043->1044, 1044->1042, log queries
100,000-109,999). Grid for both policies: half-life {1, 2, 4, 8} x capacity,
window {0.01, 0.05, 0.1, 0.2, 0.3} of capacity; WSemDemand also radius at the
acceptance cosine or 0.05 above it. Selection: highest mean net hit ratio
(correct-hit ratio minus false-hit rate) over the eighteen cells, separately
for each policy.

Evaluation (data never used before): fresh QQP streams, seeds 47-51,
capacities 200-1000 in steps of 100; the five 10,000-query windows of log
queries 0-49,999, capacities 500, 1,000, 2,000; the five shifted evaluation
streams of experiment 28 (42->43, ..., 46->42), capacities 200, 400, 600, 800.
Policies: selected WSemDemand and W-TinyLFU, LRU, LFU, SphereLFU, LeCaR,
second-sighting admission without a window (best "second" setting of
experiment 31; ablation), and Semantic Decay with its original weights.
Primary endpoints, per stream averaged over that dataset's capacities,
paired t-tests with Holm adjustment over all primary comparisons:
  (i)   QQP correct-hit ratio, WSemDemand minus each of LFU, SphereLFU, LeCaR,
        W-TinyLFU;
  (ii)  log correct-hit ratio, WSemDemand minus W-TinyLFU;
  (iii) shifted streams, second-half correct-hit ratio, WSemDemand minus LRU.
Success: (i) significantly positive against the best of the four baselines,
(ii) significantly positive, and (iii) not significantly negative.
False-hit rate, precision and net hit ratio are reported for every cell.

Run: python -m semcache.experiments.exp32_window_admission --phase all --workers 24
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

from .. import config as C
from ..cache import (LeCaRPolicy, SemanticDecayPolicy, SemanticDemandPolicy, SphereLFUPolicy,
                     WindowSemanticDemandPolicy, make_policy)
from ..encoders import warmup
from ..stats import holm, paired_seed_test, seed_ci
from ._common import REVISION_DIR, banner, get_stream, hardware_manifest, memoized_encoders, save_result, train_metas
from .exp30_sd_redesign import _vectors, ensure_log_vectors, ensure_qqp_vectors, log_window
from .exp31_semantic_admission import SERVE, TUNE, run_one

EXACT = 0.9999
HALF_LIVES = (1, 2, 4, 8)
WINDOWS = (0.01, 0.05, 0.1, 0.2, 0.3)
RADIUS_OFFSETS = (0.0, 0.05)
EVAL = {"qqp": {"seeds": (47, 48, 49, 50, 51), "caps": tuple(range(200, 1001, 100))},
        "log": {"windows": tuple((s, 10_000) for s in range(0, 50_000, 10_000)), "caps": (500, 1000, 2000)},
        "shift": {"pairs": ((42, 43), (43, 44), (44, 45), (45, 46), (46, 42)), "caps": (200, 400, 600, 800)}}
SECOND_NO_WINDOW = {"kind": "SemDemand", "aging": "exp", "mult": 4, "admission": "second",
                    "radius_offset": 0.05}        # best "second" setting of experiment 31 tuning
BASELINES = ("LFU", "SphereLFU", "LeCaR", "WTinyLFU")


def tune_specs() -> list[dict]:
    out = [{"kind": "WSemDemand", "mult": h, "window": w, "radius_offset": r}
           for h, w, r in itertools.product(HALF_LIVES, WINDOWS, RADIUS_OFFSETS)]
    out += [{"kind": "WTinyLFU", "mult": h, "window": w} for h, w in itertools.product(HALF_LIVES, WINDOWS)]
    return out + [{"kind": k} for k in ("LRU", "LFU", "SphereLFU", "LeCaR")]


EXTRA_KINDS = ("FIFO", "ARC", "S3FIFO", "SIEVE", "GDSF", "DynamicAdaptiveClimb", "SOLAR-E", "SOLAR-A", "SOLAR", "SOLARo-E", "SOLARo-A", "SOLARo")


def build(spec: dict, cap: int, data: str, seed: int):
    k, acc = spec["kind"], SERVE[data][2]
    if k in ("LRU", "LFU"):
        return make_policy(k)
    if k == "SphereLFU":
        return SphereLFUPolicy(cos_threshold=acc, capacity=cap)
    if k == "LeCaR":
        return LeCaRPolicy(cos_threshold=acc, capacity=cap, seed=seed)
    if k == "SemanticDecay_fixed":
        return SemanticDecayPolicy()
    if k in ("WSemDemand", "WTinyLFU"):
        radius = EXACT if k == "WTinyLFU" else min(0.999, acc + spec["radius_offset"])
        h = spec["mult"] * cap
        return WindowSemanticDemandPolicy(capacity=cap, window_frac=spec["window"], radius=radius,
                                          half_life=h, history=int(min(4 * h, 16 * cap)))
    if k in EXTRA_KINDS:
        from ..policies_extra import (ARCPolicy, DACPolicy, FIFOPolicy, GDSFPolicy, SolarOfficialPolicy, S3FIFOPolicy, SIEVEPolicy, SolarPolicy,
                                      SOLAR_VARIANTS)
        if k == "FIFO":
            return FIFOPolicy()
        if k == "ARC":
            return ARCPolicy(cap)
        if k == "S3FIFO":
            return S3FIFOPolicy(cap)
        if k == "SIEVE":
            return SIEVEPolicy()
        if k == "GDSF":
            return GDSFPolicy()
        if k == "DynamicAdaptiveClimb":
            return DACPolicy(cap)
        if k.startswith("SOLARo"):
            return SolarOfficialPolicy(cap, {"SOLARo-E": "E", "SOLARo-A": "A", "SOLARo": "full"}[k], seed=seed)
        lam, gam = SOLAR_VARIANTS[int(spec.get("variant") or 0)]
        return SolarPolicy(cap, {"SOLAR-E": "E", "SOLAR-A": "A", "SOLAR": "full"}[k], lam=lam, gamma=gam, seed=seed)
    if k == "SemDemand":
        h = spec["mult"] * cap
        return SemanticDemandPolicy(radius=min(0.999, acc + spec["radius_offset"]), aging="exp",
                                    half_life=h, admission=spec["admission"],
                                    history=int(min(4 * h, 16 * cap)))
    raise ValueError(k)


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


def _key(t: dict) -> str:
    return json.dumps([t["data"], t["cap"], t["spec"]], sort_keys=True)


def run_tasks(tasks: list[dict], workers: int, checkpoint) -> list[dict]:
    out, done = [None] * len(tasks), {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r["result"]
    todo = [i for i, t in enumerate(tasks) if _key(t) not in done]
    for i, t in enumerate(tasks):
        if _key(t) in done:
            out[i] = done[_key(t)]
    print(f"    {len(tasks) - len(todo)} runs restored, {len(todo)} to run", flush=True)

    def record(idx, res):
        with open(checkpoint, "a", encoding="utf8") as fh:
            for i, r in zip(idx, res):
                out[i] = r
                fh.write(json.dumps({"key": _key(tasks[i]), "result": r}) + "\n")

    if not todo:
        return out
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
        n = len(tasks) - len(todo)
        for f in as_completed(futs):
            record(futs[f], f.result())
            n += len(futs[f])
            print(f"    {n}/{len(tasks)} runs", flush=True)
    return out


def _prepare(seeds) -> None:
    ensure_qqp_vectors(sorted(seeds))
    for s in sorted(seeds):
        train_metas(seed=s, keys=(SERVE["qqp"][0],))


# ------------------------------------------------------------------ tuning
def tune(workers: int, tag: str) -> dict:
    banner("EXP 32 | WSemDemand tuning (tuning data only)")
    ensure_log_vectors([(TUNE["log"]["start"], TUNE["log"]["n"])])
    _prepare({*TUNE["qqp"]["seeds"], *(s for p in TUNE["shift"]["pairs"] for s in p)})
    cells = ([(("qqp", s), c) for s in TUNE["qqp"]["seeds"] for c in TUNE["qqp"]["caps"]]
             + [(("shift", a, b), c) for a, b in TUNE["shift"]["pairs"] for c in TUNE["shift"]["caps"]]
             + [(("log", TUNE["log"]["start"], TUNE["log"]["n"]), c) for c in TUNE["log"]["caps"]])
    sp = tune_specs()
    tasks = [{"data": list(d), "cap": c, "spec": s} for s in sp for d, c in cells]
    got = run_tasks(tasks, workers, REVISION_DIR / f"exp32_tune_checkpoint{('_' + tag) if tag else ''}.jsonl")
    rows = []
    for k, s in enumerate(sp):
        res = got[k * len(cells):(k + 1) * len(cells)]
        cr = [{"data": d[0], "stream": list(d[1:]), "cap": c, **r} for (d, c), r in zip(cells, res)]
        m = lambda kind, key: float(np.mean([r[key] for r in cr if r["data"] == kind and r.get(key) is not None]))
        rows.append({"spec": s, "cells": cr,
                     "objective_net_hit_ratio_pct": float(np.mean([r["net_hit_ratio_pct"] for r in cr])),
                     **{f"{kd}_{q}": m(kd, f"{q}_hit_ratio_pct" if q != "false" else "false_hit_rate_pct")
                        for kd in ("qqp", "shift", "log") for q in ("net", "correct", "false")},
                     "shift_second_half_correct": m("shift", "second_half_correct_pct"),
                     "shift_recovery_correct": m("shift", "recovery_correct_pct")})
    best = {k: max((r for r in rows if r["spec"]["kind"] == k), key=lambda r: r["objective_net_hit_ratio_pct"])
            for k in ("WSemDemand", "WTinyLFU")}
    for r in sorted(rows, key=lambda r: -r["objective_net_hit_ratio_pct"])[:8] + \
            [r for r in rows if r["spec"]["kind"] in ("LRU", "LFU", "SphereLFU", "LeCaR")] + [best["WTinyLFU"]]:
        print(f"    {str(r['spec']):80s} {r['objective_net_hit_ratio_pct']:6.2f} | qqp {r['qqp_correct']:5.2f}/{r['qqp_false']:4.2f} "
              f"shift2 {r['shift_second_half_correct']:5.2f} log {r['log_net']:5.2f}", flush=True)
    sel = {k: v["spec"] for k, v in best.items()}
    print(f"  selected: {sel}", flush=True)
    out = {"manifest": hardware_manifest({"phase": "tune", "grid": {"half_life_mult": HALF_LIVES, "window": WINDOWS,
                                                                    "radius_offset": RADIUS_OFFSETS}}),
           "rows": rows, "selected": sel}
    save_result(out, "exp32_window_admission_tune", tag=tag)
    return out


# ------------------------------------------------------------------ evaluation
def evaluate(selected: dict, workers: int, tag: str) -> dict:
    banner("EXP 32 | Evaluation on fresh data")
    ensure_log_vectors(list(EVAL["log"]["windows"]))
    _prepare({*EVAL["qqp"]["seeds"], *(s for p in EVAL["shift"]["pairs"] for s in p)})
    pols = {"WSemDemand": selected["WSemDemand"], "WTinyLFU": selected["WTinyLFU"],
            "LRU": {"kind": "LRU"}, "LFU": {"kind": "LFU"}, "SphereLFU": {"kind": "SphereLFU"},
            "LeCaR": {"kind": "LeCaR"}, "SecondNoWindow": SECOND_NO_WINDOW,
            "SemanticDecay_fixed": {"kind": "SemanticDecay_fixed"}}
    cells = ([(("qqp", s), c) for s in EVAL["qqp"]["seeds"] for c in EVAL["qqp"]["caps"]]
             + [(("log", a, n), c) for a, n in EVAL["log"]["windows"] for c in EVAL["log"]["caps"]]
             + [(("shift", a, b), c) for a, b in EVAL["shift"]["pairs"] for c in EVAL["shift"]["caps"]])
    tasks = [{"data": list(d), "cap": c, "spec": s, "name": n} for n, s in pols.items() for d, c in cells]
    got = run_tasks(tasks, workers, REVISION_DIR / f"exp32_eval_checkpoint{('_' + tag) if tag else ''}.jsonl")
    res: dict = {n: {} for n in pols}
    for t, r in zip(tasks, got):
        res[t["name"]].setdefault(json.dumps(t["data"]), {})[str(t["cap"])] = r

    def per_stream(name, kind, metric):
        return [float(np.mean([v[metric] for v in caps.values()]))
                for d, caps in sorted(res[name].items()) if json.loads(d)[0] == kind]

    summary = {n: {kind: {m: seed_ci(per_stream(n, kind, m))
                          for m in ("correct_hit_ratio_pct", "false_hit_rate_pct", "net_hit_ratio_pct",
                                    "precision_of_hits_pct")
                          + (("second_half_correct_pct", "recovery_correct_pct", "second_half_false_pct")
                             if kind == "shift" else ())}
                   for kind in ("qqp", "log", "shift")} for n in pols}
    primary = {}
    for b in BASELINES:
        primary[f"qqp_correct|WSemDemand-{b}"] = paired_seed_test(per_stream("WSemDemand", "qqp", "correct_hit_ratio_pct"),
                                                                  per_stream(b, "qqp", "correct_hit_ratio_pct"))
    primary["log_correct|WSemDemand-WTinyLFU"] = paired_seed_test(per_stream("WSemDemand", "log", "correct_hit_ratio_pct"),
                                                                  per_stream("WTinyLFU", "log", "correct_hit_ratio_pct"))
    primary["shift_second_half|WSemDemand-LRU"] = paired_seed_test(per_stream("WSemDemand", "shift", "second_half_correct_pct"),
                                                                   per_stream("LRU", "shift", "second_half_correct_pct"))
    adj = holm({k: v.get("t_p") for k, v in primary.items()})
    for k, v in primary.items():
        v["holm_p"] = adj.get(k)
    best_b = max(BASELINES, key=lambda b: summary[b]["qqp"]["correct_hit_ratio_pct"]["mean"])
    t1 = primary[f"qqp_correct|WSemDemand-{best_b}"]
    t2 = primary["log_correct|WSemDemand-WTinyLFU"]
    t3 = primary["shift_second_half|WSemDemand-LRU"]
    sig = lambda t: t.get("holm_p") is not None and t["holm_p"] < 0.05
    criteria = {"i_qqp_gain_over_best_baseline": bool(sig(t1) and t1["delta_mean"] > 0), "best_baseline": best_b,
                "ii_log_gain_over_wtinylfu": bool(sig(t2) and t2["delta_mean"] > 0),
                "iii_no_significant_shift_loss_to_lru": not bool(sig(t3) and t3["delta_mean"] < 0)}
    criteria["success"] = all(v for k, v in criteria.items() if k != "best_baseline")

    # secondary: per-capacity tests on QQP and the log, WSemDemand minus each baseline
    secondary = {}
    for kind, caps in (("qqp", EVAL["qqp"]["caps"]), ("log", EVAL["log"]["caps"])):
        for b in BASELINES + ("LRU",):
            for c in map(str, caps):
                for m in ("correct_hit_ratio_pct", "false_hit_rate_pct"):
                    a = [caps_[c][m] for d, caps_ in sorted(res["WSemDemand"].items()) if json.loads(d)[0] == kind]
                    o = [caps_[c][m] for d, caps_ in sorted(res[b].items()) if json.loads(d)[0] == kind]
                    secondary[f"{kind}|{c}|{m}|WSemDemand-{b}"] = paired_seed_test(a, o)
    adj2 = holm({k: v.get("t_p") for k, v in secondary.items()})
    for k, v in secondary.items():
        v["holm_p"] = adj2.get(k)

    for n in pols:
        s = summary[n]
        print(f"  {n:20s} qqp corr {s['qqp']['correct_hit_ratio_pct']['mean']:5.2f} false {s['qqp']['false_hit_rate_pct']['mean']:4.2f} | "
              f"log corr {s['log']['correct_hit_ratio_pct']['mean']:5.2f} false {s['log']['false_hit_rate_pct']['mean']:4.2f} | "
              f"shift 2nd {s['shift']['second_half_correct_pct']['mean']:5.2f} rec {s['shift']['recovery_correct_pct']['mean']:5.2f}", flush=True)
    for k, v in primary.items():
        print(f"  primary {k}: {v['delta_mean']:+.2f}  holm p {v['holm_p']:.4f}", flush=True)
    print(f"  criteria: {criteria}", flush=True)
    out = {"manifest": hardware_manifest({"phase": "eval", "eval": {k: {kk: (list(map(list, vv)) if kk in ("windows", "pairs")
                                                                          else list(vv)) for kk, vv in d.items()}
                                                                  for k, d in EVAL.items()},
                                          "selected": selected, "second_no_window": SECOND_NO_WINDOW}),
           "results": res, "summary": summary, "primary": primary, "criteria": criteria, "secondary": secondary}
    save_result(out, "exp32_window_admission_eval", tag=tag)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("tune", "eval", "all"), default="all")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        HALF_LIVES, WINDOWS, RADIUS_OFFSETS = (2,), (0.1,), (0.0,)
        TUNE["qqp"]["seeds"], TUNE["qqp"]["caps"] = (1043,), (300,)
        TUNE["shift"]["pairs"], TUNE["shift"]["caps"] = ((1042, 1043),), (300,)
        TUNE["log"]["n"], TUNE["log"]["caps"] = 3000, (500,)
        EVAL["qqp"]["seeds"], EVAL["qqp"]["caps"] = (47, 48, 49), (300,)
        EVAL["log"]["windows"], EVAL["log"]["caps"] = ((0, 3000), (10000, 3000), (20000, 3000)), (500,)
        EVAL["shift"]["pairs"], EVAL["shift"]["caps"] = ((42, 43), (43, 44), (44, 45)), (300,)
        a.tag = "smoke"
    sel = None
    if a.phase in ("tune", "all"):
        sel = tune(a.workers, a.tag)["selected"]
    if a.phase in ("eval", "all"):
        if sel is None:
            from ._common import out_name
            sel = json.load(open(REVISION_DIR / (out_name("exp32_window_admission_tune", tag=a.tag) + ".json")))["selected"]
        evaluate(sel, a.workers, a.tag)
