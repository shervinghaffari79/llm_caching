"""
LLM backend for the end-to-end response-time and token-savings experiments.

Design decision that matters for the reviewer's consistency question
----------------------------------------------------------------------
Comparing four cache configurations on the same query stream requires the
*LLM side* to be identical across the four runs; otherwise a difference in
average response time could come from API jitter rather than from the cache.
So the experiment runs in two phases:

  Phase A (`prefetch`)  -- every distinct query in the stream is answered once
                           by the real model. Wall-clock latency, prompt
                           tokens and completion tokens are recorded per query
                           and persisted.
  Phase B (`serve`)     -- each configuration replays the stream. A cache miss
                           reproduces that query's *recorded* generation
                           latency (by sleeping) and its recorded token counts.

Replay makes the LLM contribution a fixed constant across configurations, so
any difference in the reported averages is attributable to the cache pipeline.
`LLMBackend(mode="live")` runs the real API inside the serving loop instead,
and `experiments/exp2_end_to_end.py` uses it to verify that replayed and live
averages agree within noise -- so replay is a variance-reduction technique,
not a substitute for measurement.

The API key is read from the environment (`OPENAI_API_KEY`). It is never
written to the repository.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from . import config as C


# --------------------------------------------------------------------------
# Token counting
# --------------------------------------------------------------------------
_ENC = None


def _encoder():
    global _ENC
    if _ENC is None:
        import tiktoken
        try:
            _ENC = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            _ENC = tiktoken.get_encoding("o200k_base")
    return _ENC


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text or ""))


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------
@dataclass
class Generation:
    query: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    model: str
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class NullBackend:
    """Backend for experiments whose metrics are cache decisions against a
    ground-truth label (hit ratio, precision, coverage). A miss returns an
    empty answer with zero tokens and zero latency, so nothing is generated,
    replayed or simulated, and no token or latency figure derived from it may
    be reported."""

    mode = "none"
    model = "none"
    virtual_time = True
    sleep = False

    def __init__(self):
        self.pool = AnswerPool()
        self.pool_misses = 0
        self.calls = 0

    def generate(self, query: str) -> "Generation":
        self.calls += 1
        return Generation(query=query, answer="", prompt_tokens=0,
                          completion_tokens=0, latency_s=0.0, model="none")


class AnswerPool:
    """Disk-persisted map: query -> Generation. Keyed by exact query text."""

    def __init__(self, path: Path | None = None):
        self.path = path or (C.LLM_CACHE_DIR / f"pool_{C.LLM_MODEL.replace(chr(47), chr(95))}.jsonl")
        self._d: dict[str, Generation] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf8").splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    self._d[r["query"]] = Generation(**r)
                except Exception:
                    continue

    def __contains__(self, q: str) -> bool:
        return q in self._d

    def __len__(self) -> int:
        return len(self._d)

    def get(self, q: str) -> Generation | None:
        return self._d.get(q)

    def add(self, g: Generation) -> None:
        with self._lock:
            self._d[g.query] = g
            with open(self.path, "a", encoding="utf8") as fh:
                fh.write(json.dumps(asdict(g), ensure_ascii=False) + "\n")

    def latency_stats(self) -> dict:
        v = np.array([g.latency_s for g in self._d.values() if g.error is None])
        if not len(v):
            return {}
        return {"n": int(len(v)), "mean_s": float(v.mean()),
                "median_s": float(np.median(v)), "p95_s": float(np.percentile(v, 95)),
                "std_s": float(v.std(ddof=1)) if len(v) > 1 else 0.0}

    def token_stats(self) -> dict:
        p = np.array([g.prompt_tokens for g in self._d.values()])
        c = np.array([g.completion_tokens for g in self._d.values()])
        if not len(p):
            return {}
        return {"mean_prompt_tokens": float(p.mean()),
                "mean_completion_tokens": float(c.mean()),
                "mean_total_tokens": float((p + c).mean())}


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------
class LLMBackend:
    """
    mode="live"    -- call the API inside the serving loop (true wall clock)
    mode="replay"  -- reproduce the recorded latency/tokens from the pool
    mode="sim"     -- no API at all; synthesise latency from a fitted lognormal
                      (used for smoke tests and for machines without a key)

    virtual_time   replay/sim only. Generation latency is returned but not
                   slept; the serving loop adds it to the measured wall-clock
                   of the request. Cache-side work is still timed for real, so
                   response time = measured overhead and bookkeeping + recorded
                   generation latency, without waiting ~3 s per miss.
    """

    def __init__(self, mode: str = "replay", pool: AnswerPool | None = None,
                 model: str = C.LLM_MODEL, sim_mean_s: float = 0.854,
                 sim_sigma: float = 0.35, seed: int = C.SEED,
                 virtual_time: bool = False):
        if virtual_time and mode == "live":
            raise ValueError("virtual_time applies to replay and sim only")
        self.virtual_time = virtual_time
        self.pool_misses = 0
        self.mode = mode
        self.model = model
        self.pool = pool or AnswerPool()
        self._client = None
        self._rng = np.random.default_rng(seed)
        self.sim_mean_s, self.sim_sigma = sim_mean_s, sim_sigma
        self.calls = 0

    # -- API ---------------------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Export it (or put it in .env) "
                    "before running live/prefetch modes.")
            self._client = OpenAI(api_key=key, timeout=C.LLM_TIMEOUT_S,
                                  max_retries=C.LLM_MAX_RETRIES)
        return self._client

    def _call_api(self, query: str) -> Generation:
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": C.LLM_SYSTEM_PROMPT},
                      {"role": "user", "content": query}],
            max_completion_tokens=C.LLM_MAX_COMPLETION_TOKENS,
        )
        if C.LLM_REASONING_EFFORT:
            kwargs["reasoning_effort"] = C.LLM_REASONING_EFFORT
        t0 = time.perf_counter()
        try:
            r = self.client.chat.completions.create(**kwargs)
            dt = time.perf_counter() - t0
            u = r.usage
            return Generation(query=query, answer=r.choices[0].message.content or "",
                              prompt_tokens=int(u.prompt_tokens),
                              completion_tokens=int(u.completion_tokens),
                              latency_s=dt, model=self.model)
        except Exception as e:                              # pragma: no cover
            dt = time.perf_counter() - t0
            return Generation(query=query, answer="", prompt_tokens=0,
                              completion_tokens=0, latency_s=dt,
                              model=self.model, error=f"{type(e).__name__}: {e}")

    # -- Serving-loop entry point -----------------------------------------
    def generate(self, query: str) -> Generation:
        """Called on a cache miss. Blocking, and timed by the caller."""
        self.calls += 1
        if self.mode == "live":
            g = self._call_api(query)
            if g.error is None:
                self.pool.add(g)
            return g

        if self.mode == "replay":
            g = self.pool.get(query)
            if g is None:
                # Not prefetched (should not happen for a prefetched stream):
                # fall back to simulation rather than silently reporting 0 ms.
                self.pool_misses += 1
                return self._simulate(query)
            if not self.virtual_time:
                time.sleep(g.latency_s)             # reproduce the real cost
            return g

        return self._simulate(query)

    def _simulate(self, query: str) -> Generation:
        stats = self.pool.latency_stats()
        mean = stats.get("mean_s", self.sim_mean_s)
        lat = float(self._rng.lognormal(np.log(max(mean, 1e-3)), self.sim_sigma))
        lat = min(lat, 10.0)
        if not self.virtual_time:
            time.sleep(lat)
        pt = count_tokens(C.LLM_SYSTEM_PROMPT) + count_tokens(query)
        return Generation(query=query, answer=f"[simulated answer] {query[:60]}",
                          prompt_tokens=pt, completion_tokens=64,
                          latency_s=lat, model=f"{self.model}:sim")

    # -- Phase A ----------------------------------------------------------
    def prefetch(self, queries: list[str], workers: int = C.LLM_PREFETCH_WORKERS,
                 progress: bool = True) -> dict:
        """Answer every distinct query once, concurrently, and persist.

        Concurrency here is safe for the experiment's validity because these
        latencies are only used as *replay* values; the per-request latency of
        a concurrent batch is measured per request, and `latency_calibration`
        below cross-checks it against strictly sequential calls.
        """
        todo = [q for q in dict.fromkeys(queries) if q not in self.pool]
        if not todo:
            return {"prefetched": 0, "already": len(set(queries))}
        done, errors = 0, 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self._call_api, q): q for q in todo}
            for fut in as_completed(futs):
                g = fut.result()
                if g.error is None:
                    self.pool.add(g)
                else:
                    errors += 1
                done += 1
                if progress and done % 25 == 0:
                    print(f"  prefetch {done}/{len(todo)} (errors={errors})",
                          flush=True)
        return {"prefetched": done, "errors": errors, "pool_size": len(self.pool)}

    def latency_calibration(self, queries: list[str], n: int = 25) -> dict:
        """Sequential, unbatched API calls -- the honest no-cache baseline.

        Reported next to the concurrent prefetch stats so the paper can state
        which number the no-cache bar in the response-time figure uses.
        """
        lats = []
        for q in queries[:n]:
            g = self._call_api(q)
            if g.error is None:
                lats.append(g.latency_s)
                self.pool.add(g)
        v = np.array(lats)
        if not len(v):
            return {}
        return {"n_sequential": int(len(v)), "mean_s": float(v.mean()),
                "median_s": float(np.median(v)),
                "std_s": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "p95_s": float(np.percentile(v, 95))}
