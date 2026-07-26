"""
Experiment 0 -- data audit.

Answers, with counts:
  * how many duplicated / mirrored / self / conflicting rows QQP contains,
  * how badly a naive StratifiedKFold leaks questions between folds,
  * whether the 3,000-pair ablation subset is genuinely independent of the
    30,000-pair meta-classifier set.

Run: python -m semcache.experiments.exp0_data_audit
"""
from __future__ import annotations

import numpy as np

from .. import config as C
from ..data import (build_splits, fold_overlap_stats, make_group_folds,
                    make_stratified_folds, save_audit)
from ..reporting import save_json
from ._common import banner


def main() -> dict:
    banner("EXP 0 | QQP duplication and leakage audit")
    splits = build_splits()
    save_audit(splits)

    print("Row-level cleaning:")
    for k, v in splits.audit.as_dict().items():
        print(f"  {k:42s} {v}")

    print("\nComponent-level independence of the ablation subset:")
    for k, v in splits.provenance.items():
        print(f"  {k:42s} {v}")

    # How much does the fold scheme matter?
    meta = splits.meta
    y = meta["is_duplicate"].to_numpy()
    strat = make_stratified_folds(y, C.CV_FOLDS_META)
    grp = make_group_folds(y, meta["component"].to_numpy(), C.CV_FOLDS_META)

    strat_stats = fold_overlap_stats(meta, strat)
    grp_stats = fold_overlap_stats(meta, grp)

    def mean(rows, key):
        return float(np.mean([r[key] for r in rows]))

    print("\nFold-scheme comparison on the 30k meta set:")
    print(f"  StratifiedKFold        : {mean(strat_stats,'shared_questions'):.0f} "
          f"questions shared train/val per fold "
          f"({mean(strat_stats,'pct_val_questions_seen'):.1f}% of val questions "
          f"already seen in training)")
    print(f"  StratifiedGroupKFold   : {mean(grp_stats,'shared_questions'):.0f} "
          f"questions shared "
          f"({mean(grp_stats,'pct_val_questions_seen'):.1f}%)")

    out = {
        "audit": splits.audit.as_dict(),
        "provenance": splits.provenance,
        "sizes": {"meta": len(splits.meta), "ablation": len(splits.ablation),
                  "ablation_balanced": len(splits.ablation_bal),
                  "stream_pool": len(splits.stream_pool)},
        "fold_schemes": {
            "stratified_kfold": {
                "per_fold": strat_stats,
                "mean_shared_questions": mean(strat_stats, "shared_questions"),
                "mean_pct_val_questions_seen": mean(strat_stats, "pct_val_questions_seen"),
            },
            "stratified_group_kfold": {
                "per_fold": grp_stats,
                "mean_shared_questions": mean(grp_stats, "shared_questions"),
                "mean_pct_val_questions_seen": mean(grp_stats, "pct_val_questions_seen"),
            },
        },
        "interpretation": (
            "QQP is a graph, not a list of independent pairs. Under plain "
            "StratifiedKFold a large share of validation questions have "
            "already been seen during training in another pair, so validation "
            "F1 partly measures memorisation of individual questions. "
            "StratifiedGroupKFold over connected components removes that "
            "channel; every headline number in this study is reported under "
            "both schemes so the size of the effect is visible."),
    }
    save_json(out, "exp0_data_audit")
    return out


if __name__ == "__main__":
    main()
