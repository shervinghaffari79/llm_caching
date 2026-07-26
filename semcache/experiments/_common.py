"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config as C
from ..encoders import get_bi, get_cross, warmup
from ..features import FeatureBundle, build_bundle


@dataclass
class Encoded:
    df: pd.DataFrame
    e1: np.ndarray
    e2: np.ndarray
    cross: np.ndarray | None
    bundle: FeatureBundle
    y: np.ndarray
    groups: np.ndarray


def encode_frame(df: pd.DataFrame, tag: str, with_cross: bool = True) -> Encoded:
    """Bulk feature preparation.

    This runs in the *offline* batching regime and is explicitly NOT the
    source of any latency number: it exists so that the classification
    experiments do not re-run encoders. All timing comes from
    `exp1_latency_breakdown` and `exp2_end_to_end`, which use batch size 1.
    """
    q1 = df["question1"].tolist()
    q2 = df["question2"].tolist()
    bi = get_bi()
    t0 = time.perf_counter()
    e1 = bi.encode_cached(q1, f"{tag}_q1")
    e2 = bi.encode_cached(q2, f"{tag}_q2")
    print(f"  bi-encoded {len(df)} pairs in {time.perf_counter()-t0:.1f}s "
          f"(offline batching, not used for latency reporting)")

    cross = None
    if with_cross:
        ce = get_cross()
        t0 = time.perf_counter()
        cross = ce.score_cached(list(zip(q1, q2)), f"{tag}_ce")
        print(f"  cross-encoded {len(df)} pairs in {time.perf_counter()-t0:.1f}s")

    bundle = build_bundle(q1, q2, e1, e2, cross)
    y = df["is_duplicate"].to_numpy().astype(int)
    groups = df["component"].to_numpy() if "component" in df else np.arange(len(df))
    return Encoded(df=df, e1=e1, e2=e2, cross=cross, bundle=bundle, y=y,
                   groups=groups)


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


def hardware_manifest(extra: dict | None = None) -> dict:
    hw = C.HardwareProfile.capture().as_dict()
    out = {"hardware": hw, "seed": C.SEED, "bi_encoder": C.BI_ENCODER,
           "cross_encoder": C.CROSS_ENCODER}
    if extra:
        out.update(extra)
    return out
