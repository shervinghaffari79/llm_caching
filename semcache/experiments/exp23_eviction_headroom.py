"""
Experiment 23 -- how much room is there above LFU? (eviction headroom)

A clairvoyant policy that knows the whole stream evicts the entry whose topic
is requested again furthest in the future (never = infinitely far). When
several live entries share a topic, one of them suffices to serve it, so all
but the oldest are treated as never needed again. The new entry is itself a
candidate, so the policy may also decline to keep it (admission).

Optimal offline semantic eviction is NP-hard (Biton & Friedman 2026), so this
is a clairvoyant heuristic, not a proven bound. It is the standard
Belady-style reference: the gap between it and the best online policy
estimates how much any online policy could still gain.

Same serving configuration (C2), seeds and capacities as experiment 18;
the online policies' results are read from experiment 18's output.

Run: python -m semcache.experiments.exp23_eviction_headroom --exp18 FILE
"""
from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict

from .. import config as C
from ..cache import EvictionPolicy
from ..encoders import warmup
from ..stats import paired_seed_test, seed_ci
from ._common import (add_run_args, banner, get_stream, hardware_manifest,
                      memoized_encoders, save_result, train_metas)
from .exp18_eviction import CAPACITIES, EVAL_SEEDS, _run


class ClairvoyantTopicPolicy(EvictionPolicy):
    """Belady on topics: evict the entry whose topic recurs latest."""
    name = "Clairvoyant"

    def __init__(self, wl):
        self.topics = [e.topic_id for e in wl.events]
        self.future: dict[int, list[int]] = defaultdict(list)
        for i, t in enumerate(self.topics):
            self.future[t].append(i)
        self.step = -1
        self.topic_of: dict[int, int] = {}
        self.seq: dict[int, int] = {}

    def on_query(self, vec):
        self.step += 1

    def on_insert(self, entry, vec):
        self.topic_of[entry.eid] = self.topics[self.step]
        self.seq[entry.eid] = len(self.seq)

    def _next_use(self, topic: int) -> float:
        steps = self.future.get(topic, [])
        i = bisect.bisect_right(steps, self.step)
        return steps[i] if i < len(steps) else float("inf")

    def choose_victim(self, cache, vb):
        keeper: dict[int, int] = {}          # topic -> oldest live entry
        for e in sorted(self.topic_of, key=self.seq.get):
            keeper.setdefault(self.topic_of[e], e)

        def key(e):
            t = self.topic_of[e]
            nxt = self._next_use(t) if keeper[t] == e else float("inf")
            return (nxt, -self.seq[e])       # farthest first; newest on ties
        return max(self.topic_of, key=key)

    def on_evict(self, eid):
        self.topic_of.pop(eid, None)
        self.seq.pop(eid, None)


def main(exp18_file: str, seeds=EVAL_SEEDS, n: int = 2000, capacities=CAPACITIES,
         tag: str = "") -> dict:
    banner("EXP 23 | Eviction headroom: clairvoyant reference vs online policies")
    warmup()
    online = json.load(open(exp18_file, encoding="utf8"))
    res = {str(c): {} for c in capacities}
    with memoized_encoders():
        for seed in seeds:
            wl = get_stream(seed, n)
            metas = train_metas(seed=seed, keys=("C2_bi_geo_lgb",))
            for cap in capacities:
                res[str(cap)][seed] = _run(wl, metas, cap, ClairvoyantTopicPolicy(wl), seed)
            print(f"  seed {seed} done", flush=True)

    per_seed_online = online["per_seed"]
    summary, gaps = {}, {}
    for c in capacities:
        c = str(c)
        orc = [res[c][s]["correct_hit_ratio_pct"] for s in seeds]
        summary[c] = {"Clairvoyant": seed_ci(orc)}
        best_online = None
        for pol, by_cap in per_seed_online.items():
            vals = [by_cap[c][str(s)]["correct_hit_ratio_pct"] for s in seeds]
            summary[c][pol] = seed_ci(vals)
            if best_online is None or summary[c][pol]["mean"] > summary[c][best_online]["mean"]:
                best_online = pol
        ref = [per_seed_online[best_online][c][str(s)]["correct_hit_ratio_pct"] for s in seeds]
        lfu = [per_seed_online["LFU"][c][str(s)]["correct_hit_ratio_pct"] for s in seeds]
        gaps[c] = {"best_online_policy": best_online,
                   "headroom_over_best_online": paired_seed_test(orc, ref),
                   "headroom_over_LFU": paired_seed_test(orc, lfu)}
        print(f"  capacity {c:>4s}: clairvoyant {summary[c]['Clairvoyant']['mean']:.2f}%  "
              f"best online {best_online} {summary[c][best_online]['mean']:.2f}%  "
              f"headroom {gaps[c]['headroom_over_best_online']['delta_mean']:+.2f} pts")

    out = {"manifest": hardware_manifest({"n_stream": n, "config": "C2_bi_geo_lgb",
                                          "eval_seeds": list(seeds), "latency_valid": False,
                                          "online_results": exp18_file}),
           "per_seed": res, "summary": summary, "headroom": gaps}
    save_result(out, "exp23_eviction_headroom", tag=tag)
    return out


if __name__ == "__main__":
    ap = add_run_args(argparse.ArgumentParser())
    ap.add_argument("--exp18", default=str(C.ROOT / "results/raw/revision/exp18_eviction_m2.json"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(EVAL_SEEDS))
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    main(a.exp18, a.seeds, a.n, tag=a.tag)
