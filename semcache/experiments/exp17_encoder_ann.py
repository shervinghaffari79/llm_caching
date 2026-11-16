"""
Experiment 17 -- sensitivity to the bi-encoder and to approximate search (E10).

Encoders: three sentence-transformers from the individual-model evaluation,
each at its own F1-maximising validation threshold (Table 2), serving C1 and
C2 (C2's meta-classifier is refitted on each encoder's features).

Index: exact inner-product search against HNSW at several efSearch values,
for C1 and C2 on the default encoder. HNSW cannot delete entries, so these runs
are capacity-unbounded, which matches the end-to-end setting.

Run: python -m semcache.experiments.exp17_encoder_ann
"""
from __future__ import annotations

import argparse

from .. import config as C
from ..encoders import get_bi, warmup
from ._common import (add_run_args, banner, ensure_pool, get_stream,
                      hardware_manifest, make_backend, save_result, serve,
                      stream_args, stream_params, train_metas)

# name -> F1-maximising validation threshold from the individual-model table
ENCODERS = {
    "sentence-transformers/all-mpnet-base-v2": 0.726,
    "sentence-transformers/all-distilroberta-v1": 0.712,
    "sentence-transformers/all-MiniLM-L6-v2": 0.712,
}
EF_SEARCH = (8, 16, 64)
KEEP = ("hit_ratio_pct", "precision_of_hits_pct", "false_discovery_rate_pct",
        "token_saving_correct_only_pct", "correct_miss_ratio_on_hard_negatives_pct",
        "hit_ratio_on_servable_repeats_pct", "avg_overhead_ms", "p95_overhead_ms",
        "avg_response_time_s", "stage_ms_per_request")


def main(seed: int = C.SEED, n: int = 2000, params: dict | None = None,
         fetch: bool = True, tag: str = "") -> dict:
    banner("EXP 17 | Encoder and approximate-index sensitivity")
    params = params or {}
    warmup()
    wl = get_stream(seed, n, **params)
    cov = ensure_pool(wl.queries(), live=fetch)

    encoders = []
    for name, thr in ENCODERS.items():
        bi = get_bi(name)
        for _ in range(3):
            bi.encode_one("warmup query for the semantic cache")
        metas = train_metas(seed=seed, keys=("C2_bi_geo_lgb",), bi_name=name)
        for key in ("C1_bi_only", "C2_bi_geo_lgb"):
            r = serve(key, wl, make_backend(seed=seed), metas, keep_events=False,
                      bi=bi, bi_threshold=thr)
            encoders.append({"encoder": name, "bi_threshold": thr, "config": key,
                             **{k: r.stats.get(k) for k in KEEP}})
            print(f"  {name.split('/')[-1]:22s} {key}: "
                  f"prec {r.stats['precision_of_hits_pct']:.1f}% "
                  f"hit {r.stats['hit_ratio_pct']:.1f}%")

    metas = train_metas(seed=seed, keys=("C2_bi_geo_lgb",))
    index = []
    for key in ("C1_bi_only", "C2_bi_geo_lgb"):
        r = serve(key, wl, make_backend(seed=seed), metas, keep_events=False)
        index.append({"index": "flat", "ef_search": None, "config": key,
                      **{k: r.stats.get(k) for k in KEEP}})
        for ef in EF_SEARCH:
            r = serve(key, wl, make_backend(seed=seed), metas, keep_events=False,
                      index_kind="hnsw", ef_search=ef)
            index.append({"index": "hnsw", "ef_search": ef, "config": key,
                          **{k: r.stats.get(k) for k in KEEP}})
        print(f"  index sweep done for {key}")

    out = {"manifest": hardware_manifest({"n_stream": n, "stream_params": params,
                                          "virtual_time": True}, seed=seed),
           "workload": wl.meta, "pool": cov, "encoders": encoders, "index": index}
    save_result(out, "exp17_encoder_ann", seed=seed, tag=tag)
    return out


if __name__ == "__main__":
    ap = stream_args(add_run_args(argparse.ArgumentParser()))
    a = ap.parse_args()
    main(a.seed, a.n, stream_params(a), fetch=not a.no_fetch, tag=a.tag)
