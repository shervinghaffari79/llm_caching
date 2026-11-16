"""
Experiment 10 -- effect of the ground-truth labelling on reported quality.

The shipped stream gave each injected distractor a fresh singleton label at
every occurrence, so a verbatim recurrence of an earlier distractor counted as
non-equivalent to its own cached copy. The current generator assigns one label
per distinct question. This experiment serves C1-C4 once on the shipped query
sequence and scores the identical decisions under both label sets, then once
on the fresh-distractor stream used by the revision experiments.

Run: python -m semcache.experiments.exp10_label_fix
"""
from __future__ import annotations

import argparse

from .. import config as C
from ..encoders import warmup
from ..workload import Workload
from ._common import (add_run_args, banner, ensure_pool, get_stream,
                      hardware_manifest, make_backend, save_result, serve,
                      train_metas)

KEYS = ("C1_bi_only", "C2_bi_geo_lgb", "C3_bi_cross", "C4_full_stacking")
KEEP = ("hits", "true_hits", "false_hits", "hit_ratio_pct", "precision_of_hits_pct",
        "false_discovery_rate_pct", "token_saving_pct", "token_saving_correct_only_pct",
        "correct_miss_ratio_on_hard_negatives_pct", "n_hard_negatives",
        "hit_ratio_on_servable_repeats_pct", "oracle_hit_ratio_pct",
        "break_even_false_hit_cost")


def main(seed: int = C.SEED, tag: str = "", fetch: bool = True) -> dict:
    banner("EXP 10 | Labelling: shipped vs corrected, and fresh distractors")
    warmup(include_cross=True)
    metas = train_metas(seed=seed)
    shipped = Workload.load(C.DATA_DIR / "workloads" / f"stream_2000_seed{seed}.json")
    relabelled = get_stream(seed, 2000, fresh_hard_negatives=False)
    assert shipped.queries() == relabelled.queries(), "query sequences differ"
    fresh = get_stream(seed, 2000)
    ensure_pool(fresh.queries(), live=fetch)

    rows = {}
    for key in KEYS:
        a = serve(key, shipped, make_backend(seed=seed), metas)
        b = serve(key, relabelled, make_backend(seed=seed), metas)
        same = [e["hit"] for e in a.per_event] == [e["hit"] for e in b.per_event]
        c = serve(key, fresh, make_backend(seed=seed), metas, keep_events=False)
        rows[key] = {"shipped_labels": {k: a.stats.get(k) for k in KEEP},
                     "corrected_labels": {k: b.stats.get(k) for k in KEEP},
                     "fresh_distractor_stream": {k: c.stats.get(k) for k in KEEP},
                     "identical_decisions": same}
        print(f"  {key:18s} precision {a.stats['precision_of_hits_pct']:5.1f}% -> "
              f"{b.stats['precision_of_hits_pct']:5.1f}% (same decisions: {same}); "
              f"fresh stream {c.stats['precision_of_hits_pct']:5.1f}%")

    out = {"manifest": hardware_manifest({"virtual_time": True}, seed=seed),
           "workloads": {"shipped": shipped.meta, "relabelled": relabelled.meta,
                         "fresh": fresh.meta},
           "configs": rows}
    save_result(out, "exp10_label_fix", seed=seed, tag=tag)
    return out


if __name__ == "__main__":
    ap = add_run_args(argparse.ArgumentParser())
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    main(a.seed, a.tag, fetch=not a.no_fetch)
