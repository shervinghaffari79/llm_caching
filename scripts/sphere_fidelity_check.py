"""
Fidelity check of SphereLFUPolicy against the authors' SphereQueryLFU
(github.com/dbiton/SemanticCaching, src/caches/cache.py; Biton & Friedman 2026).

The authors' file is third-party code and is NOT part of this repository: pass
the path to a local copy. Only its `Cache` and `SphereQueryLFU` classes are
executed, with numpy and heapq as the sole dependencies; FAISS is replaced by an
exact flat L2 index with the same range_search contract (squared distance,
strict radius), so both implementations see identical float32 inputs.

Both are driven by the same request stream, one request at a time (batch 1):
  * hit flag per request,
  * identifiers evicted, in order,
  * final counters.
Usage: python scripts/sphere_fidelity_check.py <authors cache.py> [out.json]
"""
from __future__ import annotations

import heapq
import json
import os
import re
import sys
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semcache.cache import SphereLFUPolicy  # noqa: E402


class FlatL2:
    """IndexIDMap(IndexFlatL2) stand-in: exact squared-L2, strict radius, insertion order."""

    def __init__(self):
        self.ids, self.X = [], []

    def add_with_ids(self, embeds, ids):
        for e, i in zip(embeds, ids):
            self.X.append(np.asarray(e, dtype=np.float32)); self.ids.append(int(i))

    def remove_ids(self, ids):
        drop = {int(i) for i in ids}
        keep = [k for k, i in enumerate(self.ids) if i not in drop]
        self.X = [self.X[k] for k in keep]; self.ids = [self.ids[k] for k in keep]

    def range_search(self, embeds, radius_squared):
        lims, d2s, out = [0], [], []
        M = np.stack(self.X) if self.X else np.zeros((0, embeds.shape[1]), np.float32)
        for q in embeds:
            d2 = ((M - q.astype(np.float32)) ** 2).sum(axis=1).astype(np.float32) if len(M) else np.zeros(0, np.float32)
            sel = np.flatnonzero(d2 < np.float32(radius_squared))
            d2s.extend(d2[sel].tolist()); out.extend(self.ids[k] for k in sel)
            lims.append(len(out))
        return np.array(lims), np.array(d2s, dtype=np.float32), np.array(out, dtype=np.int64)


def load_authors(path):
    src = open(path, encoding="utf8").read()
    a, b = src.index("class Cache:"), src.index("class SphereQueryLFU")
    end = src.index("\nclass ", b + 10)
    ns = {"np": np, "heapq": heapq, "List": List}
    exec(src[a:end], ns)
    return ns["SphereQueryLFU"]


def stream(n, dim, n_clusters, sigma, zipf_s, seed):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_clusters, dim)); centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    p = 1.0 / np.arange(1, n_clusters + 1) ** zipf_s; p /= p.sum()
    c = rng.choice(n_clusters, size=n, p=p)
    # per-request noise scale varies so some requests land near the radius
    sc = sigma * rng.uniform(0.5, 1.6, size=n)[:, None]
    X = centres[c] + sc * rng.normal(size=(n, dim)) / np.sqrt(dim)
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def run(Auth, X, cos_thr, cap):
    D = float(np.sqrt(2.0 * (1.0 - cos_thr)))
    A = Auth(D); A.initialize(cap, FlatL2())
    ours = SphereLFUPolicy(cos_threshold=cos_thr, capacity=cap)
    ha, ho, ev_a, ev_o = [], [], [], []

    class E:                                   # minimal entry for on_insert
        def __init__(self, eid): self.eid = eid

    for i, v in enumerate(X):
        hits, ev = A.cache(v[None, :], [i])
        ha.append(bool(hits[0] > 0)); ev_a += [int(e) for e in ev]
        n = len(ours._ids)
        inside = False
        if n:
            d2 = np.maximum(0.0, 2.0 - 2.0 * (ours._M[:n] @ v))
            inside = bool((d2 < ours.radius ** 2).any())
        ours.on_query(v)
        ho.append(inside)
        if not inside:                         # a miss inserts, then evicts over capacity
            ours.on_insert(E(i), v)
            if len(ours.count) > cap:
                vic = ours.choose_victim(None, None); ours.on_evict(vic); ev_o.append(int(vic))
    ca = {int(k): float(v) for k, v in A.items.items()}
    co = {int(k): float(v) for k, v in ours.count.items()}
    same_keys = set(ca) == set(co)
    max_diff = max((abs(ca[k] - co[k]) for k in set(ca) & set(co)), default=0.0)
    return {"requests": len(X), "hits_authors": int(sum(ha)), "hits_ours": int(sum(ho)),
            "hit_flag_mismatches": int(sum(x != y for x, y in zip(ha, ho))),
            "evictions_authors": len(ev_a), "evictions_ours": len(ev_o),
            "eviction_sequence_identical": ev_a == ev_o,
            "first_eviction_mismatch": next((k for k, (x, y) in enumerate(zip(ev_a, ev_o)) if x != y), None),
            "final_cache_ids_identical": same_keys, "max_abs_counter_diff": max_diff}


def main(path, out=None):
    Auth = load_authors(path)
    res = {}
    for cos_thr in (0.726, 0.895):
        for cap in (50, 200, 1000):
            for seed in (0, 1):
                X = stream(6000, 64, 400, sigma=0.9, zipf_s=1.1, seed=seed)
                key = f"cos{cos_thr}_cap{cap}_seed{seed}"
                res[key] = run(Auth, X, cos_thr, cap)
                r = res[key]
                print(f"{key:26s} hits {r['hits_authors']:5d}/{r['hits_ours']:5d}  hit-flag mismatches {r['hit_flag_mismatches']}  "
                      f"evictions {r['evictions_authors']}/{r['evictions_ours']} identical={r['eviction_sequence_identical']}  "
                      f"max|counter diff| {r['max_abs_counter_diff']:.2e}", flush=True)
    if out:
        json.dump(res, open(out, "w"), indent=1)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
