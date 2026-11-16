"""
Experiment 47 -- ReuseGate on further domains (docs/PREREGISTRATION_round2b.md, section E2).

Same causal layout and statistics as exp37 / exp42 / exp43: reference traffic strictly earlier than tuning
traffic, tuning strictly earlier than evaluation, each tuning window scored by a model fitted on earlier windows
only, Holm correction over lambda in {0, 1, 2, 5, 10}.

  lmarena    SemBenchmarkLmArena (LLM prompts, 63,795 rows used, labelled by ID_Set). Three block rotations;
             in each, block 1 = reference, block 2 = tuning (three windows), block 3 = evaluation (three
             windows). The dataset has no timestamps, so dataset order stands in for time, as for the search log.
  tripclick  the final 150,000 requests of TripClick in time order: reference rows 0-49,999, tuning
             50,000-99,999, evaluation 100,000-149,999 (strictly chronological). Licensed for non-commercial
             research; data, vectors and models are not redistributed.

Nothing is generated, replayed or simulated. Run:
  python -m semcache.experiments.exp47_reusegate_domains --domain lmarena --rotation 0 --workers 8 --with-group
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from .. import config as C

EXT = C.DATA_DIR / "external"


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


DOMAIN, ROT = _arg("--domain", "lmarena"), int(_arg("--rotation", 0))
PATH = EXT / f"reusegate_{DOMAIN}_rot{ROT}.parquet"
os.environ["SEMCACHE_LOG"] = str(PATH)               # before exp30 is imported; spawned workers inherit it

from . import exp37_causal_reusegate as E37          # noqa: E402


def load_domain() -> pd.DataFrame:
    if DOMAIN == "lmarena":
        src = EXT / "sembenchmark_lmarena.parquet"
        if not src.exists():
            from datasets import load_dataset
            d = next(iter(load_dataset("vCache/SemBenchmarkLmArena").values())).to_pandas()
            d.rename(columns={"ID_Set": "id_set"})[["prompt", "id_set"]].to_parquet(src, index=False)
        return pd.read_parquet(src)[["prompt", "id_set"]]
    return pd.read_parquet(EXT / "tripclick_requests.parquet")[["prompt", "id_set"]].iloc[-150_000:].reset_index(drop=True)


def build():
    df = load_domain()
    if DOMAIN == "lmarena":
        n = len(df) // 3
        parts = [df.iloc[i * n:(i + 1) * n] for i in range(3)]
        out = pd.concat([parts[(ROT + j) % 3] for j in range(3)], ignore_index=True)
        info = {"block_rows": n, "block_order": [(ROT + j) % 3 + 1 for j in range(3)]}
    else:
        out, info = df, {"rows": len(df)}
    out.to_parquet(PATH, index=False)
    print(f"  {DOMAIN} rotation {ROT}: {len(out)} rows {info}", flush=True)
    return len(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=("lmarena", "tripclick"), required=True)
    ap.add_argument("--rotation", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--with-group", action="store_true")
    a = ap.parse_args()
    total = build()
    E37.CAUSAL_TUNING = True
    E37.WITH_GROUP = a.with_group
    if DOMAIN == "lmarena":
        blk = total // 3
        w = blk // 3
        E37.REF = (0, blk)
        E37.TUNE = tuple((blk + k * w, w) for k in range(3))
        E37.EVAL = tuple((2 * blk + k * w, w) for k in range(3))
        E37.REF_SIZES = (0, blk)
        E37.CAPS = (250, 500, 1000, 2000)
    else:
        E37.REF = (0, 50_000)
        E37.TUNE = tuple((50_000 + k * 10_000, 10_000) for k in range(5))
        E37.EVAL = tuple((100_000 + k * 10_000, 10_000) for k in range(5))
        E37.REF_SIZES = (0, 50_000)
    tag = f"{DOMAIN}_rot{a.rotation}" if DOMAIN == "lmarena" else "tripclick"
    if a.smoke:
        E37.REF = (0, 2000)
        E37.TUNE = ((2000, 1500), (3500, 1500), (5000, 1500))
        E37.EVAL = ((6500, 1500), (8000, 1500))
        E37.CAPS, E37.TAUS, E37.LAMBDAS, E37.REF_SIZES = (200,), (0.895, 0.94), (0, 1), (0, 2000)
        E37.BASE = ("LRU", "WTinyLFU")
        tag += "_smoke"
    E37.main(a.workers, tag)
