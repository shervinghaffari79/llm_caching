"""
Feature extraction, organised into the exact groups the ablation reports.

The paper's title claims cheap features beat an expensive encoder. To support
that claim the code must be able to say, per group: how many features it
contributes, what it costs in milliseconds, and what it buys in F1. This
module owns the first two; `experiments/exp3_ablation.py` owns the third.

Group taxonomy (`GROUPS`):
  bi_cosine    1 feature     cosine similarity of the two bi-encoder vectors
  cross_score  1 feature     sigmoid-squashed cross-encoder score
  distance     2 features    L2 and L1 distance between embeddings
  elementwise  2*d features  e1*e2 and |e1-e2|  (d = 768 -> 1536 features)
  text         6 features    length stats, length ratio, Jaccard, edit distance

"Cheap" is defined operationally, not rhetorically: a group is cheap if its
marginal cost is a vector op over embeddings the pipeline has already
computed. `distance`, `elementwise` and `text` are cheap in that sense --
they add microseconds on top of an embedding lookup. `cross_score` is
expensive because it requires a second, quadratic-attention forward pass over
the concatenated pair, which cannot reuse anything.

Note that `bi_cosine`, `distance` and `elementwise` all *depend on* the
bi-encoder. The claim is therefore not "no encoder needed" but "no *second*,
pairwise encoder needed": everything after the bi-encoder forward pass is
arithmetic on vectors already in memory.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np

from . import config as C
from .encoders import StageTimer

GROUPS = ("bi_cosine", "cross_score", "distance", "elementwise", "text")

CHEAP_GROUPS = ("bi_cosine", "distance", "elementwise", "text")
EXPENSIVE_GROUPS = ("cross_score",)

GROUP_SIZES = {
    "bi_cosine": 1,
    "cross_score": 1,
    "distance": 2,
    "elementwise": 2 * C.EMB_DIM,
    "text": 6,
}


# --------------------------------------------------------------------------
# Text features
# --------------------------------------------------------------------------
def _edit_ratio(a: str, b: str) -> float:
    """Normalised character edit distance in [0,1] (0 = identical)."""
    try:
        import Levenshtein
        d = Levenshtein.distance(a, b)
    except Exception:                       # pragma: no cover - fallback
        import difflib
        return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()
    return d / max(1, max(len(a), len(b)))


def text_features_one(a: str, b: str) -> np.ndarray:
    la, lb = len(a), len(b)
    wa, wb = a.split(), b.split()
    sa, sb = set(w.lower() for w in wa), set(w.lower() for w in wb)
    union = len(sa | sb)
    return np.array([
        la, lb,
        min(la, lb) / max(1, max(la, lb)),               # length ratio
        min(len(wa), len(wb)) / max(1, max(len(wa), len(wb))),
        (len(sa & sb) / union) if union else 0.0,        # Jaccard word overlap
        _edit_ratio(a, b),                               # normalised edit distance
    ], dtype=np.float32)


def text_features_batch(q1: list[str], q2: list[str]) -> np.ndarray:
    return np.vstack([text_features_one(a, b) for a, b in zip(q1, q2)])


# --------------------------------------------------------------------------
# Embedding-derived features
# --------------------------------------------------------------------------
def cosine(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """Row-wise cosine; embeddings arrive L2-normalised from the encoder."""
    if e1.ndim == 1:
        return np.array([float(np.dot(e1, e2))], dtype=np.float32)
    return np.einsum("ij,ij->i", e1, e2).astype(np.float32)


def distance_features(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    d = e1 - e2
    if e1.ndim == 1:
        return np.array([np.linalg.norm(d, 2), np.abs(d).sum()], dtype=np.float32)
    return np.stack([np.linalg.norm(d, 2, axis=1), np.abs(d).sum(axis=1)],
                    axis=1).astype(np.float32)


def elementwise_features(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    if e1.ndim == 1:
        return np.concatenate([e1 * e2, np.abs(e1 - e2)]).astype(np.float32)
    return np.concatenate([e1 * e2, np.abs(e1 - e2)], axis=1).astype(np.float32)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
@dataclass
class FeatureBundle:
    """Per-group feature blocks kept separate so the ablation can select
    groups without recomputing anything."""
    blocks: dict[str, np.ndarray]

    def matrix(self, groups: tuple[str, ...] | list[str]) -> np.ndarray:
        parts = [self.blocks[g] for g in groups if g in self.blocks]
        if not parts:
            raise ValueError("no feature groups selected")
        parts = [p.reshape(len(p), -1) if p.ndim > 1 else p.reshape(-1, 1)
                 for p in parts]
        return np.hstack(parts).astype(np.float32)

    def names(self, groups: tuple[str, ...] | list[str]) -> list[str]:
        out: list[str] = []
        for g in groups:
            if g not in self.blocks:
                continue
            blk = self.blocks[g]
            width = 1 if blk.ndim == 1 else blk.shape[1]
            if g == "text":
                out += ["len_chars_1", "len_chars_2", "len_ratio_chars",
                        "len_ratio_words", "jaccard_words", "edit_ratio"]
            elif g == "distance":
                out += ["euclidean", "manhattan"]
            elif g == "elementwise":
                out += [f"prod_{i}" for i in range(C.EMB_DIM)] + \
                       [f"absdiff_{i}" for i in range(C.EMB_DIM)]
            elif width == 1:
                out.append(g)
            else:
                out += [f"{g}_{i}" for i in range(width)]
        return out

    @property
    def n_rows(self) -> int:
        return len(next(iter(self.blocks.values())))


def build_bundle(q1: list[str], q2: list[str],
                 e1: np.ndarray, e2: np.ndarray,
                 cross: np.ndarray | None,
                 groups: tuple[str, ...] = GROUPS) -> FeatureBundle:
    blocks: dict[str, np.ndarray] = {}
    if "bi_cosine" in groups:
        blocks["bi_cosine"] = cosine(e1, e2)
    if "cross_score" in groups:
        if cross is None:
            raise ValueError("cross_score requested but no cross-encoder scores given")
        blocks["cross_score"] = np.asarray(cross, dtype=np.float32).reshape(-1)
    if "distance" in groups:
        blocks["distance"] = distance_features(e1, e2)
    if "elementwise" in groups:
        blocks["elementwise"] = elementwise_features(e1, e2)
    if "text" in groups:
        blocks["text"] = text_features_batch(q1, q2)
    return FeatureBundle(blocks)


# --------------------------------------------------------------------------
# Online (single-pair) feature construction, with per-group timing
# --------------------------------------------------------------------------
def build_online_features(q: str, c: str, eq: np.ndarray, ec: np.ndarray,
                          cross_score: float | None,
                          cfg: "C.ServingConfig",
                          timer: StageTimer | None = None) -> np.ndarray:
    """Build one feature row in the order the trained model expects.

    Group order must match `cfg.feature_groups`, which is also the order used
    at training time -- both derive from the same `ServingConfig`.
    """
    parts: list[np.ndarray] = []
    ctx = (lambda n: timer.stage(n)) if timer else (lambda n: contextlib.nullcontext())

    if cfg.use_bi:
        with ctx("feat_bi_cosine"):
            parts.append(np.array([float(np.dot(eq, ec))], dtype=np.float32))
    if cfg.use_cross:
        with ctx("feat_cross"):
            parts.append(np.array([float(cross_score)], dtype=np.float32))
    if cfg.use_geometric:
        with ctx("feat_distance"):
            parts.append(distance_features(eq, ec))
        with ctx("feat_elementwise"):
            parts.append(elementwise_features(eq, ec))
    if cfg.use_text:
        with ctx("feat_text"):
            parts.append(text_features_one(q, c))
    return np.concatenate(parts).astype(np.float32).reshape(1, -1)


def groups_for_config(cfg: "C.ServingConfig") -> tuple[str, ...]:
    return cfg.feature_groups
