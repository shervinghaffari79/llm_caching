"""
Equivalence check for the SOLAR ports (round 2c).

AuthorsSolarPolicy drives the authors' unmodified agents (github.com/ysunbp/SOLAR, commit 8d01e93, Apache-2.0)
as the replacement/admission policy of the repository's cache.  Only the model-loading constructor of
`EmbedderAgent` is replaced (it would load an embedder and an LLM client that the policy never uses) and `_embed`
returns the vector that the cache already computed; every admission, eviction and retrieval-update method that runs is
the authors' code.

SolarPortV2 is our port written to follow the authors' control flow call for call (same candidate schedule, same
retrieval updates, same lazy deletion and compaction, same random draws), so that decision logs can be compared.

Adaptation shared by both (not in the paper): in a response cache the feedback score of a request is 1 - cosine(request,
served entry) for a hit and 1 for a miss (label-free).  The authors' experience-mode function accumulates the regret and
may store in the same call; on a hit a cache cannot store, so for a hit the policy executes the lines of the authors'
function that precede the store decision (candidate count, regret, history, threshold adaptation) and then stops.
"""
from __future__ import annotations

import io
import os
import sys
import tarfile
import urllib.request

import numpy as np

from .cache import EvictionPolicy

COMMIT = "8d01e9340eb3a9ab77761366e57a29c53dafefe7"
BASE = "/root/solar_authors_src"
_LOADED: dict = {}


def load_authors():
    if _LOADED:
        return _LOADED
    root = None
    if os.path.isdir(BASE):
        for d in os.listdir(BASE):
            if os.path.isdir(os.path.join(BASE, d, "src")):
                root = os.path.join(BASE, d)
    if root is None:
        url = f"https://codeload.github.com/ysunbp/SOLAR/tar.gz/{COMMIT}"
        data = urllib.request.urlopen(url, timeout=180).read()
        os.makedirs(BASE, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            tf.extractall(BASE)
        root = next(os.path.join(BASE, d) for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d, "src")))
    sys.path.insert(0, root)
    import faiss
    try:
        from src.agent import embedder as E
    except Exception as exc:                    # the package __init__ may import evaluators/LLM clients that the policy never uses
        print("  authors' package import failed, stubbing src.llms:", repr(exc)[:200], flush=True)
        import types
        stub = types.ModuleType("src.llms")
        stub.LlmFactory = type("LlmFactory", (), {"create": staticmethod(lambda **k: None)})
        sys.modules["src.llms"] = stub
        from src.agent import embedder as E

    def _headless_init(self, config):           # EmbedderAgent.__init__ without the embedder and the LLM client
        self.config = config
        self.index = faiss.IndexFlatL2(config.embedding_dim)
        self.metadata = []

    E.EmbedderAgent.__init__ = _headless_init
    from src.agent.solar import SolarAgent, SolarAgentConfig
    from src.agent.solar_a import SolarAAgent, SolarAAgentConfig
    from src.agent.solar_e import SolarEAgent, SolarEAgentConfig
    _LOADED.update(root=root, SolarAgent=SolarAgent, SolarAgentConfig=SolarAgentConfig, SolarAAgent=SolarAAgent,
                   SolarAAgentConfig=SolarAAgentConfig, SolarEAgent=SolarEAgent, SolarEAgentConfig=SolarEAgentConfig)
    return _LOADED


def _active_ids(ag) -> list:
    return [m["doc_id"] for m in ag.metadata if not m.get("deleted", False)]


class AuthorsSolarPolicy(EvictionPolicy):
    """The authors' SolarAgent ("full"), SolarAAgent ("A") or SolarEAgent ("E") as the policy of our cache."""

    def __init__(self, capacity: int, mode: str = "full", dim: int = 768, seed: int = 0, retrieve_k: int = 10):
        assert mode in ("E", "A", "full")
        L = load_authors()
        cls, cfg = {"full": (L["SolarAgent"], L["SolarAgentConfig"]), "A": (L["SolarAAgent"], L["SolarAAgentConfig"]),
                    "E": (L["SolarEAgent"], L["SolarEAgentConfig"])}[mode]

        class Headless(cls):
            _cur = None

            def _embed(self, text):             # the cache has already embedded the request
                return self._cur

        np.random.seed(seed)                    # the authors draw with the global generator
        self.name = {"E": "SOLARa-E", "A": "SOLARa-A", "full": "SOLARa"}[mode]
        self.mode, self.k = mode, retrieve_k
        self.ag = Headless(cfg(capacity=int(capacity), embedding_dim=dim, embedder_provider="huggingface"))
        self._vec: dict = {}
        self._q = None
        self._new = None
        self._victim = None

    def _gate_prefix(self, cost: float) -> None:
        """Lines of add_memory_experience_gated that precede the store decision (used for a hit)."""
        ag = self.ag
        ag.total_candidates += 1
        if self.mode == "E":
            return
        ag.accumulated_regret += cost
        ag.novelty_history.append(cost)
        if ag.solar_config.threshold_mode == "adaptive":
            ag._adapt_threshold()

    def on_query(self, vec):
        self._q = np.asarray(vec, dtype=np.float32)
        self.ag._cur = self._q
        self.ag.retrieve_memory("q", k=self.k)      # authors' retrieval: top-k, updates the per-entry statistics

    def on_hit(self, entry, vec):
        v = self._vec.get(entry.eid)
        sim = float(np.clip(np.dot(self._q, v), 0.0, 1.0)) if v is not None else 1.0
        self._gate_prefix(1.0 - sim)

    def on_insert(self, entry, vec):
        ag, e = self.ag, entry.eid
        v = np.asarray(vec, dtype=np.float32)
        ag._cur = v
        cap = ag.solar_config.capacity if self.mode != "E" else ag.thompson_config.capacity
        room = ag._active_count < cap                # the cache keeps the entry whatever the gate says while it has room
        before = set(_active_ids(ag))
        if self.mode == "E":
            stored = ag.add_memory_thompson(str(e), doc_id=e)
        else:
            stored = ag.add_memory_experience_gated(str(e), 0.0, doc_id=e)
            if not stored and room:                  # the lines of the authors' store block, executed because the slot is free
                ag.metadata.append({"doc_id": e, "content": str(e)})
                ag.index.add(np.array([v]))
                ag._active_count += 1
                idx = len(ag.metadata) - 1
                if self.mode == "A":
                    ag._access_counts[idx] = 0
                    ag._last_accessed[idx] = ag.total_candidates
                else:
                    ag._init_entry_stats(idx)
                stored = True
        after = set(_active_ids(ag))
        self._new = e
        if stored:
            self._vec[e] = v
            gone = before - after
            self._victim = next(iter(gone)) if gone else None
        else:
            self._victim = e

    def choose_victim(self, cache, vb):
        if self._victim is None:
            raise RuntimeError("desync: the cache is over capacity but the agent evicted nothing")
        v, self._victim = self._victim, None
        return v

    def on_evict(self, eid):
        self._vec.pop(eid, None)


class SolarPortV2(EvictionPolicy):
    """Our port following the authors' code call for call (see the module docstring)."""

    def __init__(self, capacity: int, mode: str = "full", dim: int = 768, seed: int = 0, retrieve_k: int = 10):
        assert mode in ("E", "A", "full")
        self.name = {"E": "SOLARv2-E", "A": "SOLARv2-A", "full": "SOLARv2"}[mode]
        self.mode, self.cap, self.dim, self.k = mode, int(capacity), dim, retrieve_k
        np.random.seed(seed)
        self.lam, self.lip, self.window, self.tmin, self.tmax = 5.0, 2.0, 30, 0.1, 5.0
        self.threshold = float(np.sqrt(2 * self.lam / self.lip))
        self.regret, self.hist, self.L_ema = 0.0, [], None
        self.cands = 0
        self.eids: list = []                    # authors' metadata: doc id per slot (lazy deletion until compaction)
        self.dead: list = []
        self.V = np.zeros((self.cap + 128, dim), dtype=np.float32)    # authors' FAISS flat index, one row per slot
        self.nn = np.full((self.cap + 128, 3), np.inf, dtype=np.float32)   # per slot: the 3 smallest squared distances to all slots, itself first
        self.slot_of: dict = {}                 # doc id -> slot
        self.m = 0
        self.active = 0
        self.evictions = 0
        self.alpha, self.beta, self.last, self.cnt = {}, {}, {}, {}   # keyed by slot position, remapped at compaction
        self.rstep = 0
        self._q = None
        self._victim = None
        self._new = None

    # ---- index helpers (squared L2 over all slots, tombstones included, as IndexFlatL2)
    def _first_k(self, v, k):
        d = ((self.V[:self.m] - v) ** 2).sum(axis=1)
        order = np.argsort(d, kind="stable")[:k]
        return order, d[order]

    def _pair_d2(self):
        X = self.V[:self.m]
        sq = (X * X).sum(axis=1)
        D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        np.maximum(D, 0.0, out=D)
        np.fill_diagonal(D, 0.0)
        return D

    # ---- retrieval statistics (authors' retrieve_memory)
    def on_query(self, vec):
        self._q = np.asarray(vec, dtype=np.float32)
        if self.m == 0:
            return
        order, _ = self._first_k(self._q, min(self.k * 2, self.m))
        if not any(not self.dead[int(i)] for i in order):      # authors' retrieve_memory returned [] -> no update
            return
        rs = set()
        for i in order:
            i = int(i)
            if self.dead[i]:
                continue
            if self.mode == "A":                # solar_a.py: all active entries of the 2k searched, no cut at k
                self.cnt[i] = self.cnt.get(i, 0) + 1
                self.last[i] = self.cands
            else:                               # solar.py / solar_e.py: the k nearest active entries
                rs.add(i)
                self.alpha[i] = self.alpha.get(i, 1.0) + 1.0
                self.last[i] = self.cands
                if len(rs) >= self.k:
                    break
        if self.mode != "A":
            self.rstep += 1
            if self.rstep % 5 == 0:
                for i in range(self.m):
                    if not self.dead[i] and i not in rs:
                        self.beta[i] = self.beta.get(i, 1.0) + 0.05

    # ---- gate
    def _adapt(self):
        if len(self.hist) < self.window:
            return
        new_L = max(np.std(self.hist[-self.window:]) * 2, 0.1)
        self.L_ema = new_L if self.L_ema is None else 0.7 * self.L_ema + 0.3 * new_L
        self.threshold = float(np.clip(np.sqrt(2 * self.lam / self.L_ema), self.tmin, self.tmax))

    def _prefix(self, cost):
        self.cands += 1
        if self.mode == "E":
            return
        self.regret += cost
        self.hist.append(cost)
        self._adapt()

    def on_hit(self, entry, vec):
        i = self.slot_of.get(entry.eid)
        sim = float(np.clip(np.dot(self._q, self.V[i]), 0.0, 1.0)) if i is not None and not self.dead[i] else 1.0
        self._prefix(1.0 - sim)

    def on_insert(self, entry, vec):
        e = entry.eid
        v = np.asarray(vec, dtype=np.float32)
        self._new = e
        self._victim = None
        self._prefix(1.0)
        store = True
        if self.mode != "E":
            boot = min(max(self.cap // 2, 5), 25)
            store = self.active < boot or self.regret > self.threshold
        room = self.active < self.cap
        if not store and not room:
            self._victim = e
            return
        if store and self.mode != "E":
            self.regret = 0.0
        while self.active >= self.cap:
            before = self.active
            (self._evict_ts if self.mode != "A" else self._evict_a)()
            if self.active >= before:
                break
        m = self.m
        if m > 0:                                # neighbour rows: merge the distance to the new vector (tombstones stay neighbours until compaction)
            d = ((self.V[:m] - v) ** 2).sum(axis=1)
            cand = np.concatenate([self.nn[:m], d[:, None]], axis=1)
            cand.sort(axis=1)
            self.nn[:m] = cand[:, :3]
            two = np.sort(d)[:2]
            self.nn[m] = np.inf
            self.nn[m, 0] = 0.0
            self.nn[m, 1:1 + len(two)] = two
        else:
            self.nn[0] = np.inf
            self.nn[0, 0] = 0.0
        self.eids.append(e)
        self.dead.append(False)
        self.slot_of[e] = m
        self.V[m] = v
        self.m += 1
        self.active += 1
        i = self.m - 1
        if self.mode == "A":
            self.cnt[i], self.last[i] = 0, self.cands
        else:
            self.alpha[i], self.beta[i], self.last[i] = 2.0, 1.0, self.cands

    # ---- eviction
    def _evict_ts(self):
        act = [i for i in range(self.m) if not self.dead[i]]
        if not act:
            return
        sc = []
        for i in act:
            ts = float(np.random.beta(self.alpha.get(i, 1.0), self.beta.get(i, 1.0)))
            d = self.nn[i][np.isfinite(self.nn[i])]
            avg = float(np.mean(d[1:])) if len(d) > 1 else float(d[0])
            sc.append((i, ts + 0.3 * (1.0 - np.exp(-avg / (2 * self.dim)))))
        sc.sort(key=lambda x: x[1])
        self._kill(sc[0][0])

    def _evict_a(self):
        if self.active == 0:
            return
        scores = []
        for i in range(self.m):
            if self.dead[i]:
                scores.append(float("inf"))
                continue
            age = self.cands - self.last.get(i, 0)
            rec = np.exp(-age / 100.0)
            freq = self.cnt.get(i, 0) / max(self.cands, 1)
            div = 1.0 - np.exp(-float(self.nn[i, 1]) / (2 * self.dim)) if self.m > 1 else 1.0
            scores.append(0.4 * rec + 0.3 * freq + 0.3 * div)
        self._kill(int(np.argmin(scores)))

    def _kill(self, i):
        self.dead[i] = True
        self.active -= 1
        self.evictions += 1
        self._victim = self.eids[i]
        if self.evictions % 20 == 0:             # compaction: tombstones removed, statistics remapped
            keep = [j for j in range(self.m) if not self.dead[j]]
            new = {j: n for n, j in enumerate(keep)}
            for d in (self.alpha, self.beta, self.last, self.cnt):
                rem = {new[j]: val for j, val in d.items() if j in new}
                d.clear()
                d.update(rem)
            self.V[:len(keep)] = self.V[keep]
            self.eids = [self.eids[j] for j in keep]
            self.dead = [False] * len(keep)
            self.m = len(keep)
            self.slot_of = {e: n for n, e in enumerate(self.eids)}
            self._recompute_nn()

    def _recompute_nn(self):
        m = self.m
        self.nn[:m] = np.inf
        if m == 0:
            return
        D = self._pair_d2()
        kk = min(3, m)
        P = np.partition(D, kk - 1, axis=1)[:, :kk] if m > kk else D.copy()
        P = np.sort(P, axis=1)
        self.nn[:m, :kk] = P

    def choose_victim(self, cache, vb):
        if self._victim is None:
            raise RuntimeError("desync: the cache is over capacity but the port evicted nothing")
        v, self._victim = self._victim, None
        return v

    def on_evict(self, eid):
        pass


class Recorder(EvictionPolicy):
    """Delegating wrapper that logs every call the cache makes, for decision-level comparison."""

    def __init__(self, inner):
        self.inner, self.log, self.name = inner, [], inner.name

    def on_query(self, vec):
        self.log.append(("q",))
        self.inner.on_query(vec)

    def on_hit(self, entry, vec):
        self.log.append(("hit", entry.eid))
        self.inner.on_hit(entry, vec)

    def on_insert(self, entry, vec):
        self.inner.on_insert(entry, vec)
        self.log.append(("ins", entry.eid))

    def choose_victim(self, cache, vb):
        v = self.inner.choose_victim(cache, vb)
        self.log.append(("victim", v))
        return v

    def on_evict(self, eid):
        self.inner.on_evict(eid)
