"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

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


def encode_frame(df: pd.DataFrame, tag: str, with_cross: bool = True,
                 bi_name: str | None = None) -> Encoded:
    """Bulk feature preparation.

    This runs in the *offline* batching regime and is explicitly NOT the
    source of any latency number: it exists so that the classification
    experiments do not re-run encoders. All timing comes from
    `exp1_latency_breakdown` and `exp2_end_to_end`, which use batch size 1.
    """
    q1 = df["question1"].tolist()
    q2 = df["question2"].tolist()
    bi = get_bi(bi_name or C.BI_ENCODER)
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

    from ..features import GROUPS
    groups = GROUPS if cross is not None else tuple(g for g in GROUPS if g != "cross_score")
    bundle = build_bundle(q1, q2, e1, e2, cross, groups=groups)
    y = df["is_duplicate"].to_numpy().astype(int)
    groups = df["component"].to_numpy() if "component" in df else np.arange(len(df))
    return Encoded(df=df, e1=e1, e2=e2, cross=cross, bundle=bundle, y=y,
                   groups=groups)


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


def hardware_manifest(extra: dict | None = None, seed: int | None = None) -> dict:
    hw = C.HardwareProfile.capture().as_dict()
    out = {"hardware": hw, "seed": C.SEED if seed is None else seed,
           "bi_encoder": C.BI_ENCODER, "cross_encoder": C.CROSS_ENCODER,
           "meta_infer_threads": C.META_INFER_THREADS}
    if extra:
        out.update(extra)
    return out


# --------------------------------------------------------------------------
# Run identity: platform and seed tags keep result files from overwriting
# each other when the same experiment runs on the M2 and on the A10, or
# under several seeds.
# --------------------------------------------------------------------------
def platform_tag() -> str:
    hw = C.HardwareProfile.capture()
    if hw.device == "cuda":
        return "".join(ch for ch in hw.cpu.lower().replace("nvidia", "")
                       if ch.isalnum()) or "cuda"
    return {"mps": "m2"}.get(hw.device, hw.device)


def out_name(base: str, seed: int | None = None, tag: str = "") -> str:
    parts = [base, platform_tag()]
    if seed is not None:
        parts.append(f"seed{seed}")
    if tag:
        parts.append(tag)
    return "_".join(parts)


def add_run_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--seed", type=int, default=C.SEED,
                    help="stream and meta-classifier seed")
    ap.add_argument("--tag", default="", help="suffix for output files")
    return ap


# --------------------------------------------------------------------------
# Meta-classifiers, persisted so every experiment process serves the same
# trained model instead of refitting it.
# --------------------------------------------------------------------------
META_GROUPS = {
    "C2_bi_geo_lgb": ("bi_cosine", "distance", "elementwise", "text"),
    "C4_full_stacking": ("bi_cosine", "cross_score", "distance", "elementwise", "text"),
}
META_GROUPS["C5_gated"] = META_GROUPS["C2_bi_geo_lgb"]
MODEL_DIR = C.DATA_DIR / "models"


def train_metas(n_train: int | None = None, seed: int = C.SEED,
                keys: tuple[str, ...] | None = None,
                bi_name: str | None = None) -> dict:
    """Train (or load) the meta-classifiers the serving configurations need.

    Trained on the 30,000-pair meta set, which is component-disjoint from the
    stream pool, so nothing in the stream was seen during training. `seed`
    sets the LightGBM seed; the training rows are fixed by the split seed.
    """
    import joblib
    from ..data import build_splits
    from ..metaclf import fit_final

    keys = keys or tuple(META_GROUPS)
    splits = build_splits()
    df = splits.meta if n_train is None else splits.meta.iloc[:n_train]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out, enc = {}, None
    for key in keys:
        groups = META_GROUPS[key]
        enc_tag = "" if not bi_name or bi_name == C.BI_ENCODER else \
            "_" + bi_name.split("/")[-1]
        path = MODEL_DIR / f"meta_{'-'.join(groups)}_n{len(df)}_seed{seed}{enc_tag}.joblib"
        if path.exists():
            out[key] = joblib.load(path).set_inference_threads()
            continue
        if enc is None:
            # the cross-encoder score is needed only by configurations whose feature groups include it;
            # C1/C2 never use it, so skipping it changes no feature and no model
            need_cross = any("cross_score" in META_GROUPS[k] for k in keys)
            enc = encode_frame(df, f"meta{len(df)}", with_cross=need_cross, bi_name=bi_name)
        X = enc.bundle.matrix(groups)
        out[key] = fit_final(X, enc.y, groups, seed=seed)
        joblib.dump(out[key], path)
        out[key].set_inference_threads()
        print(f"  trained meta for {key}: {X.shape[1]} features, {len(enc.y)} rows")
    return out


# --------------------------------------------------------------------------
# Shared serving helpers for the revision experiments
# --------------------------------------------------------------------------
EVAL_STREAM_SEEDS = (42, 43, 44, 45, 46)
_DISJOINT: dict = {}


def disjoint_pool():
    """The QQP stream pool without any question component that occurs in an
    evaluation stream (seeds 42-46).

    A component is the connected set of questions linked by duplicate labels;
    removing the whole component removes the evaluation questions, their
    paraphrases, and every non-duplicate pair that mentions any of them. Streams
    drawn from this pool therefore share no question and no label class with
    the evaluation streams, which are unchanged."""
    if "pool" in _DISJOINT:
        return _DISJOINT["pool"]
    from ..data import UnionFind, build_splits, qid_of
    from ..workload import build_workloads
    pool = build_splits().stream_pool
    ev = {qid_of(e.query) for s in EVAL_STREAM_SEEDS
          for e in build_workloads(pool, lengths=(2000,), seed=s)[2000].events}
    pos = pool[pool.is_duplicate == 1]
    uf = UnionFind()
    for q1, q2 in zip(pos.question1, pos.question2):
        uf.union(qid_of(q1), qid_of(q2))
    members = {}
    for q1, q2 in zip(pos.question1, pos.question2):
        for q in (q1, q2):
            members.setdefault(uf.find(qid_of(q)), set()).add(qid_of(q))
    bad_roots = {uf.find(q) for q in ev if q in uf.parent}
    bad = set().union(*(members[r] for r in bad_roots if r in members)) | ev
    keep = ~(pool.question1.map(qid_of).isin(bad) | pool.question2.map(qid_of).isin(bad))
    _DISJOINT.update(pool=pool[keep].reset_index(drop=True), n_bad=len(bad), n_rows_removed=int((~keep).sum()))
    return _DISJOINT["pool"]


def get_stream(seed: int = C.SEED, n: int = 2000, **params):
    """One stream of `n` events; `params` override workload.STREAM_DEFAULTS.

    With SEMCACHE_DISJOINT_QQP=1, every stream of seed >= 1000 (tuning and
    reference streams) is drawn from `disjoint_pool()`; evaluation streams
    (seeds 42-46) are drawn from the full pool exactly as before."""
    import os
    from ..data import build_splits
    from ..workload import build_workloads
    if os.environ.get("SEMCACHE_DISJOINT_QQP") == "1" and seed >= 1000:
        return build_workloads(disjoint_pool(), lengths=(n,), seed=seed,
                               cache_dir=C.DATA_DIR / "workloads_disjoint", **params)[n]
    return build_workloads(build_splits().stream_pool, lengths=(n,), seed=seed,
                           **params)[n]


def ensure_pool(queries, pool=None, live: bool = True) -> dict:
    """Make sure every query has a recorded answer, latency and token count.

    Missing queries are fetched from the API when a key is available, so
    replay runs never fall back to simulated answers. Returns coverage."""
    import os
    from ..llm import AnswerPool, LLMBackend
    pool = pool or AnswerPool()
    missing = [q for q in dict.fromkeys(queries) if q not in pool]
    fetched = {}
    if missing and live and os.environ.get("OPENAI_API_KEY"):
        fetched = LLMBackend(mode="live", pool=pool).prefetch(missing)
    still = sum(1 for q in dict.fromkeys(queries) if q not in pool)
    return {"unique": len(set(queries)), "missing_before": len(missing),
            "missing_after": still, **fetched}


def make_backend(virtual_time: bool = True, pool=None, seed: int = C.SEED):
    from ..llm import AnswerPool, LLMBackend
    return LLMBackend(mode="replay", pool=pool or AnswerPool(),
                      virtual_time=virtual_time, seed=seed)


def serve(key: str, wl, backend, metas: dict | None = None,
          keep_events: bool = True, **cache_kw):
    """Run one configuration over a stream; returns the ServeResult."""
    from ..pipeline import SemanticCache
    cfg = C.CONFIGS[key]
    cache = SemanticCache(cfg, meta=(metas or {}).get(key), **cache_kw)
    res = cache.run_stream(wl, backend, keep_events=keep_events)
    res.stats["config"] = key
    return res


REVISION_DIR = C.RAW_DIR / "revision"


def save_result(obj: dict, base: str, seed: int | None = None, tag: str = "") -> Path:
    """Revision results live in results/raw/revision/, tagged by platform,
    seed and an optional label, so nothing overwrites the shipped runs."""
    from ..reporting import save_json
    return save_json(obj, out_name(base, seed, tag), directory=REVISION_DIR)


def stream_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--n", type=int, default=2000, help="stream length")
    ap.add_argument("--recycled-hn", action="store_true",
                    help="allow distractors to recur verbatim (the original "
                         "generator); the default draws fresh distractors")
    ap.add_argument("--no-fetch", action="store_true",
                    help="do not call the API for queries missing from the pool")
    return ap


def stream_params(a) -> dict:
    return {"fresh_hard_negatives": False} if getattr(a, "recycled_hn", False) else {}


class memoized_encoders:
    """Context manager: cache bi-encoder embeddings and cross-encoder scores by
    text for the duration of a sweep. Decisions are unchanged (the same text
    gives the same vector), but stage timings become meaningless, so sweeps
    that use this must not report latency."""

    def __enter__(self):
        from ..encoders import get_bi, get_cross
        self.bi, self.ce = get_bi(), get_cross()
        self._orig = (self.bi.encode_one, self.ce.score_one)
        bi_cache, ce_cache = {}, {}
        orig_bi, orig_ce = self._orig

        def encode_one(text, timer=None, stage="bi_encode"):
            if text not in bi_cache:
                bi_cache[text] = orig_bi(text)
            return bi_cache[text]

        def score_one(a, b, timer=None, stage="cross_encode"):
            if (a, b) not in ce_cache:
                ce_cache[(a, b)] = orig_ce(a, b)
            return ce_cache[(a, b)]

        self.bi.encode_one, self.ce.score_one = encode_one, score_one
        self._bi_cache = bi_cache
        return self

    def prefill(self, texts) -> None:
        """Batch-encode texts into the memo (same vectors, much faster than
        one call per text)."""
        todo = [t for t in dict.fromkeys(texts) if t not in self._bi_cache]
        if todo:
            for t, v in zip(todo, self.bi.encode(todo)):
                self._bi_cache[t] = v

    def __exit__(self, *exc):
        self.bi.encode_one, self.ce.score_one = self._orig
        return False
