"""
Encoder wrappers with per-call timing instrumentation.

The important property here is that the *same* wrapper objects, with the same
`batch_regime`, are used by the latency breakdown and by the end-to-end serving
loop. There is no separate "fast path" used for one table and a slow path for
another. `Timer` accumulates wall-clock per stage into whatever dict the caller
passes, so a single serving call yields both its answer and its own latency
decomposition.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import config as C


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
class StageTimer:
    """Accumulates per-stage wall-clock in seconds."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    @contextlib.contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.totals[name] = self.totals.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1

    def ms(self, name: str) -> float:
        return self.totals.get(name, 0.0) * 1000.0

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()

    def snapshot_ms(self) -> dict[str, float]:
        return {k: v * 1000.0 for k, v in self.totals.items()}


def _progress_ok() -> bool:
    """Progress bars only when a human is watching; they corrupt captured logs."""
    if os.environ.get("SEMCACHE_PROGRESS") == "1":
        return True
    return sys.stderr.isatty()


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_torch() -> None:
    torch.set_grad_enabled(False)
    if C.TORCH_THREADS:
        torch.set_num_threads(C.TORCH_THREADS)


# --------------------------------------------------------------------------
# Bi-encoder
# --------------------------------------------------------------------------
class BiEncoder:
    def __init__(self, name: str = C.BI_ENCODER, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        configure_torch()
        self.name = name
        self.device = device or _default_device()
        self.model = SentenceTransformer(name, device=self.device)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int = C.OFFLINE_BATCH,
               timer: StageTimer | None = None, stage: str = "bi_encode",
               show_progress: bool = False) -> np.ndarray:
        ctx = timer.stage(stage) if timer else contextlib.nullcontext()
        with ctx:
            emb = self.model.encode(
                texts, batch_size=batch_size, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=show_progress)
        return emb.astype(np.float32)

    def encode_one(self, text: str, timer: StageTimer | None = None,
                   stage: str = "bi_encode") -> np.ndarray:
        """Online path: batch size 1, exactly what a live cache must do."""
        return self.encode([text], batch_size=C.ONLINE_BATCH, timer=timer,
                           stage=stage)[0]

    # -- disk-cached bulk encoding (offline feature prep only, never timed) --
    def encode_cached(self, texts: list[str], tag: str,
                      batch_size: int = C.OFFLINE_BATCH) -> np.ndarray:
        key = hashlib.blake2b(
            (self.name + "|" + tag + "|" + str(len(texts))).encode(), digest_size=12
        ).hexdigest()
        path = C.EMB_CACHE_DIR / f"{key}.npy"
        if path.exists():
            arr = np.load(path)
            if arr.shape[0] == len(texts):
                return arr
        arr = self.encode(texts, batch_size=batch_size,
                          show_progress=_progress_ok())
        np.save(path, arr)
        return arr


# --------------------------------------------------------------------------
# Cross-encoder
# --------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class CrossEncoderWrapper:
    def __init__(self, name: str = C.CROSS_ENCODER, device: str | None = None):
        from sentence_transformers import CrossEncoder
        configure_torch()
        self.name = name
        self.device = device or _default_device()
        self.model = CrossEncoder(name, device=self.device)
        self.model.model.eval()

    def score(self, pairs: list[tuple[str, str]], batch_size: int = C.OFFLINE_BATCH,
              timer: StageTimer | None = None, stage: str = "cross_encode",
              show_progress: bool = False) -> np.ndarray:
        ctx = timer.stage(stage) if timer else contextlib.nullcontext()
        with ctx:
            raw = self.model.predict(list(pairs), batch_size=batch_size,
                                     show_progress_bar=show_progress,
                                     convert_to_numpy=True)
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        # stsb heads emit a similarity in ~[0,5] or a logit depending on the
        # checkpoint; squash to [0,1] the same way the manuscript describes.
        return raw if (raw.min() >= 0.0 and raw.max() <= 1.0) else _sigmoid(raw)

    def score_one(self, a: str, b: str, timer: StageTimer | None = None,
                  stage: str = "cross_encode") -> float:
        return float(self.score([(a, b)], batch_size=C.ONLINE_BATCH,
                                timer=timer, stage=stage)[0])

    def score_cached(self, pairs: list[tuple[str, str]], tag: str,
                     batch_size: int = C.OFFLINE_BATCH) -> np.ndarray:
        key = hashlib.blake2b(
            (self.name + "|" + tag + "|" + str(len(pairs))).encode(), digest_size=12
        ).hexdigest()
        path = C.EMB_CACHE_DIR / f"ce_{key}.npy"
        if path.exists():
            arr = np.load(path)
            if arr.shape[0] == len(pairs):
                return arr
        arr = self.score(pairs, batch_size=batch_size,
                         show_progress=_progress_ok())
        np.save(path, arr)
        return arr


# --------------------------------------------------------------------------
# Lazy singletons (models are large; load once per process)
# --------------------------------------------------------------------------
_BI: dict[str, BiEncoder] = {}
_CE: dict[str, CrossEncoderWrapper] = {}


def get_bi(name: str = C.BI_ENCODER) -> BiEncoder:
    if name not in _BI:
        _BI[name] = BiEncoder(name)
    return _BI[name]


def get_cross(name: str = C.CROSS_ENCODER) -> CrossEncoderWrapper:
    if name not in _CE:
        _CE[name] = CrossEncoderWrapper(name)
    return _CE[name]


def warmup() -> None:
    """Run a few throwaway inferences so the first timed call is not paying
    lazy-init / kernel-autotune costs. Called by every timing experiment."""
    bi = get_bi()
    for _ in range(3):
        bi.encode_one("warmup query for the semantic cache")
    for name in list(_CE):
        for _ in range(2):
            _CE[name].score_one("warmup query", "warm up query")
