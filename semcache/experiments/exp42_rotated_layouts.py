"""
Experiment 42 -- ReuseGate over the whole real log by block rotation (k = 3).

The log has no timestamps, so its row order is arbitrary; any past-only layout
is therefore an equally valid causal protocol. Experiment 37 used one layout:
reference = rows 0-49,999 (block 1), tuning = block 2, evaluation = block 3.
To evaluate every row of the log exactly once, with reference, tuning and
evaluation data disjoint and ordered in the same past-only way, the two other
rotations are run here:

  rotation 1   reference block 2, tuning block 3, evaluation block 1
  rotation 2   reference block 3, tuning block 1, evaluation block 2

(rotation 0 is experiment 37 with causal tuning). Each rotation is a rewritten
copy of the log in which the blocks appear in that order; the experiment-37
protocol then applies unchanged, with causal tuning (each tuning window is
scored by a model fitted on earlier tuning windows only, and the first tuning
window only trains), reference sizes 0 and 50,000, the same features, label-free
target, LightGBM settings, half-life, threshold grid, capacities, selection rule
and tests. No hyperparameter is chosen on an evaluation block of its own
rotation. Rows of every block were used in earlier design experiments, so none
of the rotations is fresh data; the rotation shows that the result does not
depend on which block of the arbitrary row order plays which role.

Run: python -m semcache.experiments.exp42_rotated_layouts --rotation 1 --workers 30
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from .. import config as C

ORIG = C.DATA_DIR / "external" / "sembenchmark_searchqueries.parquet"
BLOCK = 50_000


def _rotation() -> int:
    return int(sys.argv[sys.argv.index("--rotation") + 1]) if "--rotation" in sys.argv else 0


ROT = _rotation()
PATH = C.DATA_DIR / "external" / f"sembenchmark_searchqueries_rot{ROT}.parquet"
if ROT:
    os.environ["SEMCACHE_LOG"] = str(PATH)          # before exp30 is imported; spawned workers inherit it

from . import exp37_causal_reusegate as E37       # noqa: E402


def build() -> None:
    df = pd.read_parquet(ORIG)
    parts = [df.iloc[i * BLOCK:(i + 1) * BLOCK] for i in range(3)]
    out = pd.concat([parts[(ROT + j) % 3] for j in range(3)], ignore_index=True)
    out.to_parquet(PATH, index=False)
    print(f"  rotation {ROT}: blocks in order {[(ROT + j) % 3 + 1 for j in range(3)]}, {len(out)} rows", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotation", type=int, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--with-group", action="store_true")
    ap.add_argument("--lecar", action="store_true")
    a = ap.parse_args()
    build()
    E37.CAUSAL_TUNING = True
    E37.WITH_GROUP = a.with_group
    if a.lecar:
        E37.BASE = E37.BASE + ("LeCaR",)
    E37.REF_SIZES = (0, 50_000)
    tag = f"rot{a.rotation}"
    if a.smoke:
        E37.REF = (0, 3000)
        E37.TUNE = ((3000, 1500), (4500, 1500), (6000, 1500))
        E37.EVAL = ((100_000, 1500), (101_500, 1500))
        E37.CAPS, E37.TAUS, E37.LAMBDAS, E37.REF_SIZES = (200,), (0.895, 0.94), (0, 1), (0, 3000)
        E37.BASE = ("LRU", "WTinyLFU")
        tag += "_smoke"
    E37.main(a.workers, tag)
