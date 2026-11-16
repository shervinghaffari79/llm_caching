"""
Experiment 21 -- end-to-end timing of the published baselines.

Serves the seed-42 2,000-request stream with one baseline, sleeping for each
miss's recorded generation latency (real waits, as in experiment 2), so the
baselines' decision overhead and response times come from the same host and
protocol as C1-C5. Threshold baselines use the C1 validation threshold;
vCache uses its own insertion rule and the given error bound.

Run one system per process:
    python -m semcache.experiments.exp21_baseline_timing --system gptcache
    python -m semcache.experiments.exp21_baseline_timing --system vcache --delta 0.02
"""
from __future__ import annotations

import argparse

from .. import config as C
from ..baselines import BASELINE_CLASSES, VCacheBaseline
from ..encoders import get_bi, warmup
from ..reporting import check_run_consistency
from ._common import (add_run_args, banner, ensure_pool, get_stream,
                      hardware_manifest, make_backend, save_result, stream_args,
                      stream_params)


def main(system: str, seed: int = C.SEED, n: int = 2000, params: dict | None = None,
         fetch: bool = False, delta: float = 0.02, tag: str = "",
         threshold: float = C.BI_COSINE_THRESHOLD) -> dict:
    banner(f"EXP 21 | Baseline timing with real waits: {system}")
    params = params or {}
    warmup()
    bi = get_bi()
    for _ in range(3):
        bi.encode_one("warmup query for the semantic cache")
    wl = get_stream(seed, n, **params)
    cov = ensure_pool(wl.queries(), live=fetch)
    if system == "vcache":
        cache = VCacheBaseline(bi=bi, delta=delta, insertion="native", seed=seed)
    else:
        cache = BASELINE_CLASSES[system](bi=bi, threshold=threshold)
    s = cache.run_stream(wl, make_backend(virtual_time=False, seed=seed))
    s["consistency"] = check_run_consistency(s)
    print(f"  hit {s['hit_ratio_pct']:.1f}%  prec {s['precision_of_hits_pct']:.1f}%  "
          f"overhead {s['avg_overhead_ms']:.2f} ms  response {s['avg_response_time_s']:.3f} s  "
          f"consistent {s['consistency']['consistent']}")
    name = system if system != "vcache" else f"vcache_d{delta:g}"
    out = {"manifest": hardware_manifest({"n_stream": n, "stream_params": params,
                                          "virtual_time": False, "system": name,
                                          "threshold": threshold}, seed=seed),
           "workload": wl.meta, "pool": cov, "stats": s}
    save_result(out, "exp21_baseline_timing", seed=seed, tag=tag or name)
    return out


if __name__ == "__main__":
    ap = stream_args(add_run_args(argparse.ArgumentParser()))
    ap.add_argument("--system", required=True,
                    choices=[*BASELINE_CLASSES, "vcache"])
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--threshold", type=float, default=C.BI_COSINE_THRESHOLD)
    a = ap.parse_args()
    main(a.system, a.seed, a.n, stream_params(a), fetch=not a.no_fetch,
         delta=a.delta, tag=a.tag, threshold=a.threshold)
