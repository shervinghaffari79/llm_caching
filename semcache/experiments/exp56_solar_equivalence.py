"""
Experiment 56 -- equivalence check of our SOLAR ports against the authors' code (round 2c).

Three implementations of the SOLAR policies (full, A = admission only, E = eviction only) serve the same stream through
the same cache:

  authors  the authors' unmodified agents (github.com/ysunbp/SOLAR, commit 8d01e93), driven through
           semcache.authors_solar.AuthorsSolarPolicy
  v1       SolarOfficialPolicy, the port used in the round 2c tables
  v2       SolarPortV2, the port rewritten to follow the authors' control flow call for call

Every call the cache makes to the policy is logged. For each request the decision tuple is (served entry, inserted
entry, evicted entries). Reported per run: the index of the first request at which two implementations decide
differently, the number of requests with identical decisions, and the number of stores, hits and evictions of each.
Identical decision logs for all requests mean that the two implementations are equivalent on that stream (same seeds for
the random draws, same cache, same vectors). No claim is made beyond the streams run here.

Run: python -m semcache.experiments.exp56_solar_equivalence --workers 12
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np

from ..encoders import warmup
from ._common import banner, hardware_manifest, memoized_encoders, save_result
from .exp30_sd_redesign import _vectors, ensure_log_vectors, log_window
from .exp31_semantic_admission import SERVE, run_one

KINDS = ("authors", "v1", "v2")
MODES = ("full", "A", "E")


def build(kind: str, cap: int, mode: str):
    from ..authors_solar import AuthorsSolarPolicy, SolarPortV2
    from ..policies_extra import SolarOfficialPolicy
    if kind == "authors":
        return AuthorsSolarPolicy(cap, mode, seed=0)
    if kind == "v2":
        return SolarPortV2(cap, mode, seed=0)
    return SolarOfficialPolicy(cap, mode, seed=0)


def decisions(log):
    out, cur = [], None
    for ev in log:
        if ev[0] == "q":
            if cur is not None:
                out.append(cur)
            cur = {"hit": None, "ins": None, "victims": []}
        elif ev[0] == "hit":
            cur["hit"] = ev[1]
        elif ev[0] == "ins":
            cur["ins"] = ev[1]
        elif ev[0] == "victim":
            cur["victims"].append(ev[1])
    if cur is not None:
        out.append(cur)
    return [(d["hit"], d["ins"], tuple(d["victims"])) for d in out]


def counts(dec):
    hits = sum(d[0] is not None for d in dec)
    stores = sum(d[1] is not None and d[1] not in d[2] for d in dec)
    inserted = sum(d[1] is not None for d in dec)
    evict = sum(len([v for v in d[2] if v != d[1]]) for d in dec)
    return {"requests": len(dec), "hits": hits, "misses": inserted, "stores": stores, "evictions_of_old_entries": evict}


def compare(a, b):
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), None)
    det = None if first is None else {"a": [list(map(lambda x: x if not isinstance(x, tuple) else list(x), a[i])) for i in range(max(0, first - 1), min(n, first + 2))],
                                      "b": [list(map(lambda x: x if not isinstance(x, tuple) else list(x), b[i])) for i in range(max(0, first - 1), min(n, first + 2))]}
    return {"first_divergence_request": first, "detail": det, "identical_requests": int(sum(a[i] == b[i] for i in range(n))),
            "requests_compared": n, "identical_logs": first is None and len(a) == len(b)}


def _task(t: dict) -> dict:
    from ..authors_solar import Recorder
    warmup()
    vecs = _vectors()
    key, kw, _ = SERVE["log"]
    out = {"task": t, "kinds": {}, "pairs": {}}
    decs = {}
    with memoized_encoders() as memo:
        memo._bi_cache.update(vecs)
        wl = log_window(t["start"], t["n"])
        for kind in KINDS:
            pol = Recorder(build(kind, t["cap"], t["mode"]))
            r = run_one(wl, None, t["cap"], pol, key, 0, **kw)
            decs[kind] = decisions(pol.log)
            out["kinds"][kind] = {**counts(decs[kind]), "correct_hit_ratio_pct": r["correct_hit_ratio_pct"],
                                  "false_hit_rate_pct": r["false_hit_rate_pct"], "cache_size_end": r["cache_size_end"]}
    out["pairs"]["authors_vs_v1"] = compare(decs["authors"], decs["v1"])
    out["pairs"]["authors_vs_v2"] = compare(decs["authors"], decs["v2"])
    return out


def main(workers: int, tag: str, windows, caps, n: int) -> dict:
    banner("EXP 56 | SOLAR equivalence check")
    ensure_log_vectors([(s, n) for s in windows])
    tasks = [{"start": s, "n": n, "cap": c, "mode": m} for s, c, m in itertools.product(windows, caps, MODES)
             if not (m == "E" and c > 500)]       # the authors' eviction scans the index once per entry: too slow for E at 1,000
    tasks.append({"start": windows[0], "n": 3000, "cap": 1000, "mode": "E"})    # E at 1,000 on a shorter window
    if workers <= 1:
        res = [_task(t) for t in tasks]
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        res = [None] * len(tasks)
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(_task, t): i for i, t in enumerate(tasks)}
            for f in as_completed(futs):
                res[futs[f]] = f.result()
                print(f"    {sum(r is not None for r in res)}/{len(tasks)} runs", flush=True)
    for r in res:
        t = r["task"]
        print(f"  start {t['start']} cap {t['cap']} mode {t['mode']}: "
              + "  ".join(f"{k}: stores {v['stores']} hits {v['hits']}" for k, v in r["kinds"].items())
              + f" | authors~v1 first diff {r['pairs']['authors_vs_v1']['first_divergence_request']}"
              + f" | authors~v2 first diff {r['pairs']['authors_vs_v2']['first_divergence_request']}", flush=True)
    out = {"manifest": hardware_manifest({"windows": list(windows), "caps": list(caps), "n": n, "modes": list(MODES)}),
           "runs": res}
    save_result(out, "exp56_solar_equivalence", tag=tag)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.smoke:
        main(a.workers, "smoke", (100_000,), (200,), 1500)
    else:
        main(a.workers, a.tag, (100_000, 110_000), (250, 1000), 5000)
