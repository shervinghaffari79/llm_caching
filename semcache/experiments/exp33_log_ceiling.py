"""
Experiment 33 -- how much room does the real query log leave for any eviction
or admission policy? (go/no-go test for further semantic policy work)

On SemBenchmarkSearchQueries no query repeats verbatim; every reuse is a
paraphrase, so exact-key policies (W-TinyLFU) carry no frequency information
and any gain above them must come from semantic information. Offline
(clairvoyant) policies that know the rest of the stream give a reference for
what a policy could still gain:

  OracleTopic     Belady on groups (experiment 23): evict the entry whose
                  group is requested again furthest in the future.
  OracleServable  Belady on servable reuse: an entry's next use is the next
                  request of its group whose cosine with the entry's key
                  reaches the acceptance threshold; requests that another live
                  entry of the group could also serve do not count.
  OracleNet       OracleServable, but entries that would be matched by a
                  request of another group before their next use (a false hit
                  under the threshold) are evicted first.
The newcomer is itself a candidate, so every oracle also decides admission.
Optimal offline semantic eviction is NP-hard, so these are heuristics; the
best of them is an achievable offline policy and a conservative estimate of
the ceiling.

Protocol, fixed before any run of this experiment:
Data: log queries 100,000-149,999 in five 10,000-query windows (tuning data;
the evaluation windows 0-49,999 are not used). Serving: C1 at its in-domain
threshold 0.895 (experiments 30-32). Capacities 500, 1,000, 2,000.
Online references: LRU, LFU, SphereLFU, and W-TinyLFU and WSemDemand with the
settings selected in experiment 32. An unbounded cache gives the ceiling of
any capacity policy.
Primary quantity: per window, averaged over the capacities, the net hit ratio
(correct-hit ratio minus false-hit rate) of the best oracle minus that of
W-TinyLFU; mean and 95% t-interval over the five windows.
Decision rule: further semantic policy work is pursued only if this headroom
is at least 1.0 percentage point; otherwise the line is closed.

Run: python -m semcache.experiments.exp33_log_ceiling --workers 16
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
from collections import defaultdict

import numpy as np

from .. import config as C
from ..cache import EvictionPolicy
from ..encoders import warmup
from ..stats import paired_seed_test, seed_ci
from ._common import REVISION_DIR, banner, hardware_manifest, memoized_encoders, save_result
from .exp23_eviction_headroom import ClairvoyantTopicPolicy
from .exp30_sd_redesign import _vectors, ensure_log_vectors, log_window
from .exp31_semantic_admission import SERVE, run_one
from .exp32_window_admission import build

WINDOWS = tuple((s, 10_000) for s in range(100_000, 150_000, 10_000))
CAPS = (500, 1000, 2000)
TAU = SERVE["log"][2]
SELECTED = {"WSemDemand": {"kind": "WSemDemand", "mult": 4, "window": 0.3, "radius_offset": 0.05},
            "WTinyLFU": {"kind": "WTinyLFU", "mult": 2, "window": 0.3}}      # experiment 32 selection
ONLINE = {"LRU": {"kind": "LRU"}, "LFU": {"kind": "LFU"}, "SphereLFU": {"kind": "SphereLFU"}, **SELECTED}
ORACLES = ("OracleTopic", "OracleServable", "OracleNet")
INF = float("inf")


def future_pairs(vecs: np.ndarray, topics: list[int], tau: float, block: int = 1000):
    """For each request i, the later requests j with cosine(i, j) >= tau, split
    into requests of the same group (servable) and of other groups (risk)."""
    n = len(vecs)
    same, other = [[] for _ in range(n)], [[] for _ in range(n)]
    for a in range(0, n, block):
        rows, cols = np.nonzero(vecs[a:a + block] @ vecs.T >= tau)
        for r, c in zip(rows.tolist(), cols.tolist()):
            i = a + r
            if c > i:
                (same if topics[c] == topics[i] else other)[i].append(c)
    return same, other


class ServableOracle(EvictionPolicy):
    """Belady on servable reuse; with risk=True, entries that would serve a
    request of another group before their next use are evicted first."""

    def __init__(self, topics: list[int], same, other, risk: bool = False):
        self.name = "OracleNet" if risk else "OracleServable"
        self.topics, self.same, self.other, self.risk = topics, same, other, risk
        self.same_set: dict[int, set] = {}
        self.step = -1
        self.born: dict[int, int] = {}
        self.live: dict[int, set] = defaultdict(set)      # group -> live entry ids

    def on_query(self, vec):
        self.step += 1

    def on_insert(self, entry, vec):
        self.born[entry.eid] = self.step
        self.live[self.topics[self.step]].add(entry.eid)

    def _servable(self, i: int) -> set:
        if i not in self.same_set:
            self.same_set[i] = set(self.same[i])
        return self.same_set[i]

    def _next_use(self, e: int) -> float:
        i = self.born[e]
        steps = self.same[i]
        k = bisect.bisect_right(steps, self.step)
        if k == len(steps):
            return INF
        mates = [self.born[f] for f in self.live[self.topics[i]] if f != e]
        for j in steps[k:]:
            if not any(j in self._servable(m) for m in mates):
                return j
        return INF

    def choose_victim(self, cache, vb):
        def key(e):
            nxt = self._next_use(e)
            if self.risk:
                risk = self.other[self.born[e]]
                n_wrong = bisect.bisect_left(risk, nxt) - bisect.bisect_right(risk, self.step)
                if n_wrong > 0:
                    return (2, n_wrong, -self.born[e])
            if nxt == INF:
                return (1, 0, -self.born[e])
            return (0, nxt, -self.born[e])
        return max(self.born, key=key)

    def on_evict(self, eid):
        i = self.born.pop(eid, None)
        if i is not None:
            self.live[self.topics[i]].discard(eid)


def _policy(name: str, cap: int, wl, pairs):
    if name == "OracleTopic":
        return ClairvoyantTopicPolicy(wl)
    if name in ("OracleServable", "OracleNet"):
        topics = [e.topic_id for e in wl.events]
        return ServableOracle(topics, *pairs, risk=(name == "OracleNet"))
    return build(ONLINE[name], cap, "log", C.SEED)


def _worker(tasks: list[dict]) -> list[dict]:
    warmup()
    vecs = _vectors()
    key, kw, _ = SERVE["log"]
    out = [None] * len(tasks)
    by_window: dict[tuple, list[int]] = defaultdict(list)
    for i, t in enumerate(tasks):
        by_window[tuple(t["window"])].append(i)
    with memoized_encoders() as memo:
        memo._bi_cache.update(vecs)
        for (start, n), idx in by_window.items():
            wl = log_window(start, n)
            pairs = None
            if any(tasks[i]["policy"] in ("OracleServable", "OracleNet") for i in idx):
                V = np.stack([vecs[e.query] for e in wl.events]).astype(np.float32)
                pairs = future_pairs(V, [e.topic_id for e in wl.events], TAU)
            for i in idx:
                t = tasks[i]
                cap = t["cap"] or n                      # None = unbounded (never full)
                pol = _policy(t["policy"], cap, wl, pairs)
                out[i] = run_one(wl, None, cap, pol, key, C.SEED, **kw)
    return out


def _key(t: dict) -> str:
    return json.dumps([t["window"], t["cap"], t["policy"]])


def run_tasks(tasks: list[dict], workers: int, checkpoint) -> list[dict]:
    done = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r["result"]
    out = [done.get(_key(t)) for t in tasks]
    todo = [i for i, r in enumerate(out) if r is None]
    print(f"    {len(tasks) - len(todo)} runs restored, {len(todo)} to run", flush=True)

    def record(idx, res):
        with open(checkpoint, "a", encoding="utf8") as fh:
            for i, r in zip(idx, res):
                out[i] = r
                fh.write(json.dumps({"key": _key(tasks[i]), "result": r}) + "\n")

    if workers <= 1:
        for i in todo:
            record([i], _worker([tasks[i]]))
        return out
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    # one task per job: the oracles are slow and the windows few
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_worker, [tasks[i]]): [i] for i in todo}
        n = len(tasks) - len(todo)
        for f in as_completed(futs):
            record(futs[f], f.result())
            n += 1
            print(f"    {n}/{len(tasks)} runs  ({tasks[futs[f][0]]['policy']}, cap {tasks[futs[f][0]]['cap']})",
                  flush=True)
    return out


def main(workers: int, tag: str = "") -> dict:
    banner("EXP 33 | Ceiling test on the real query log (tuning windows)")
    try:
        import torch
        torch.set_num_threads(max(1, os.cpu_count() or 1))   # encoding only; workers stay single-threaded
    except Exception:
        pass
    ensure_log_vectors(list(WINDOWS))
    names = list(ONLINE) + list(ORACLES)
    tasks = [{"window": list(w), "cap": c, "policy": p} for w in WINDOWS for c in CAPS for p in names]
    tasks += [{"window": list(w), "cap": None, "policy": "LRU"} for w in WINDOWS]
    got = run_tasks(tasks, workers, REVISION_DIR / f"exp33_checkpoint{('_' + tag) if tag else ''}.jsonl")

    res: dict = defaultdict(lambda: defaultdict(dict))         # policy -> window -> cap -> result
    for t, r in zip(tasks, got):
        name = "Unbounded" if t["cap"] is None else t["policy"]
        res[name][str(t["window"][0])][str(t["cap"])] = r
    wins = [str(w[0]) for w in WINDOWS]

    def per_window(name, metric, caps=CAPS):
        return [float(np.mean([res[name][w][str(c)][metric] for c in caps])) for w in wins]

    def best_oracle(metric, caps=CAPS):
        return [float(np.mean([max(res[o][w][str(c)][metric] for o in ORACLES) for c in caps])) for w in wins]

    metrics = ("correct_hit_ratio_pct", "false_hit_rate_pct", "net_hit_ratio_pct", "precision_of_hits_pct")
    summary = {n: {m: seed_ci(per_window(n, m)) for m in metrics} for n in names}
    summary["Unbounded"] = {m: seed_ci([res["Unbounded"][w]["None"][m] for w in wins]) for m in metrics}
    summary["BestOracle"] = {m: seed_ci(best_oracle(m)) for m in ("correct_hit_ratio_pct", "net_hit_ratio_pct")}
    per_cap = {str(c): {n: {m: seed_ci(per_window(n, m, (c,))) for m in metrics} for n in names}
               for c in CAPS}
    for c in CAPS:
        per_cap[str(c)]["BestOracle"] = {m: seed_ci(best_oracle(m, (c,)))
                                         for m in ("correct_hit_ratio_pct", "net_hit_ratio_pct")}

    head_net = paired_seed_test(best_oracle("net_hit_ratio_pct"), per_window("WTinyLFU", "net_hit_ratio_pct"))
    head_corr = paired_seed_test(best_oracle("correct_hit_ratio_pct"), per_window("WTinyLFU", "correct_hit_ratio_pct"))
    wsem_net = paired_seed_test(per_window("WSemDemand", "net_hit_ratio_pct"), per_window("WTinyLFU", "net_hit_ratio_pct"))
    per_cap_head = {str(c): paired_seed_test(best_oracle("net_hit_ratio_pct", (c,)),
                                             per_window("WTinyLFU", "net_hit_ratio_pct", (c,))) for c in CAPS}
    share = wsem_net["delta_mean"] / head_net["delta_mean"] if head_net["delta_mean"] else None
    decision = {"primary_net_headroom_over_wtinylfu_pts": head_net["delta_mean"],
                "ci": [head_net.get("delta_ci_low"), head_net.get("delta_ci_high")],
                "threshold_pts": 1.0, "pursue": bool(head_net["delta_mean"] >= 1.0),
                "share_of_headroom_taken_by_wsemdemand": share,
                # Of the net headroom, the part from extra correct hits; the rest is
                # from avoided false hits, which a stricter threshold also buys (exp29).
                "correct_hit_component_pts": head_corr["delta_mean"],
                "false_hit_component_pts": head_net["delta_mean"] - head_corr["delta_mean"]}

    for n in names + ["Unbounded"]:
        s = summary[n]
        print(f"  {n:16s} correct {s['correct_hit_ratio_pct']['mean']:6.2f}  false {s['false_hit_rate_pct']['mean']:5.2f}  "
              f"net {s['net_hit_ratio_pct']['mean']:6.2f}  precision {s['precision_of_hits_pct']['mean']:5.1f}", flush=True)
    for c in CAPS:
        print(f"  cap {c:5d}: best oracle net {per_cap[str(c)]['BestOracle']['net_hit_ratio_pct']['mean']:5.2f}  "
              f"W-TinyLFU net {per_cap[str(c)]['WTinyLFU']['net_hit_ratio_pct']['mean']:5.2f}  "
              f"headroom {per_cap_head[str(c)]['delta_mean']:+.2f}", flush=True)
    print(f"  net headroom over W-TinyLFU {head_net['delta_mean']:+.2f} pts "
          f"[{head_net.get('delta_ci_low', float('nan')):.2f}, {head_net.get('delta_ci_high', float('nan')):.2f}]; "
          f"correct-hit headroom {head_corr['delta_mean']:+.2f}; WSemDemand takes {wsem_net['delta_mean']:+.2f}", flush=True)
    print(f"  decision: {decision}", flush=True)
    out = {"manifest": hardware_manifest({"windows": [list(w) for w in WINDOWS], "caps": list(CAPS), "tau": TAU,
                                          "selected": SELECTED, "oracles": list(ORACLES)}),
           "results": json.loads(json.dumps(res)), "summary": summary, "per_capacity": per_cap,
           "headroom": {"net_over_wtinylfu": head_net, "correct_over_wtinylfu": head_corr,
                        "wsemdemand_net_over_wtinylfu": wsem_net, "net_over_wtinylfu_per_capacity": per_cap_head},
           "decision": decision}
    save_result(out, "exp33_log_ceiling", tag=tag)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        WINDOWS = ((100_000, 1500), (101_500, 1500))
        CAPS = (200,)
        a.tag = "smoke"
    main(a.workers, a.tag)
