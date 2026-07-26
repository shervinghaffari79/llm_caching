"""
Workload simulator: turning QQP's *static pair list* into a *dynamic query
stream*.

This is the full answer to "how was the sequence of queries (with duplicates)
generated from QQP?".

Construction
------------
1. **Equivalence classes ("topics").** Take only the positive (is_duplicate=1)
   edges of the QQP graph and compute connected components over question
   nodes. Each component is a set of questions annotated as mutually
   equivalent, i.e. a topic with several surface forms. Components larger than
   `max_topic_size` are dropped: transitive closure over noisy labels
   occasionally merges unrelated questions into one giant blob, and letting
   that through would manufacture easy hits.
2. **Hard negatives.** For every question we record the questions it is paired
   with under is_duplicate=0. These are lexically similar but semantically
   distinct -- exactly the traffic that produces *false* cache hits. The
   generator injects them deliberately; a stream of random unrelated questions
   would make any similarity model look good.
3. **Popularity.** Topics are ranked and assigned Zipf(s) probabilities, so a
   few topics dominate as in real query traffic.
4. **Stream generation.** At each step, with probability `duplicate_rate` the
   generator emits a *repeat*: it samples an already-seen topic by popularity
   and emits a different surface form of it (a paraphrase), or the same form
   with probability `exact_repeat_rate`. Otherwise it emits a *novel* event:
   with probability `hard_negative_rate` a recorded hard negative of some
   already-seen question, else a fresh topic.

Ground truth carried by each event
----------------------------------
`topic_id`      the equivalence class; two events can be served from one
                cached response iff their topic_ids match.
`is_repeat`     the topic appeared earlier in the stream. This is the *only*
                correct notion of "should hit" for a stream; note it is a
                function of position, so the same question is a miss on first
                arrival and a hit later. That is why the cache hit ratio is a
                dynamic quantity and not a relabelling of pair-level accuracy.
`is_exact`      the identical surface string appeared earlier (a trivial hit
                that even a hash cache would get).
`is_hard_negative`  the event is a near-miss distractor.
`session_id`, `turn`  conversational grouping, used by the context-aware
                baseline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .data import UnionFind, qid_of


# --------------------------------------------------------------------------
@dataclass
class StreamEvent:
    step: int
    query: str
    topic_id: int
    is_repeat: bool
    is_exact: bool
    is_hard_negative: bool
    session_id: int
    turn: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Workload:
    events: list[StreamEvent]
    meta: dict = field(default_factory=dict)

    def queries(self) -> list[str]:
        return [e.query for e in self.events]

    def unique_queries(self) -> list[str]:
        return list(dict.fromkeys(self.queries()))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([e.as_dict() for e in self.events])

    def save(self, path: Path) -> Path:
        path.write_text(json.dumps(
            {"meta": self.meta, "events": [e.as_dict() for e in self.events]},
            ensure_ascii=False, indent=1))
        return path

    @staticmethod
    def load(path: Path) -> "Workload":
        d = json.loads(Path(path).read_text(encoding="utf8"))
        return Workload(events=[StreamEvent(**e) for e in d["events"]],
                        meta=d.get("meta", {}))

    def describe(self) -> dict:
        df = self.to_frame()
        n = len(df)
        return {
            "n_events": n,
            "n_unique_queries": int(df["query"].nunique()),
            "n_topics": int(df.topic_id.nunique()),
            "repeat_rate_pct": 100.0 * float(df.is_repeat.mean()),
            "exact_repeat_rate_pct": 100.0 * float(df.is_exact.mean()),
            "paraphrase_repeat_rate_pct": 100.0 * float(
                (df.is_repeat & ~df.is_exact).mean()),
            "hard_negative_rate_pct": 100.0 * float(df.is_hard_negative.mean()),
            "n_sessions": int(df.session_id.nunique()),
            "mean_session_len": float(df.groupby("session_id").size().mean()),
            "top10_topic_share_pct": 100.0 * float(
                df.topic_id.value_counts().head(10).sum() / max(1, n)),
        }


# --------------------------------------------------------------------------
class WorkloadGenerator:
    def __init__(self, pool: pd.DataFrame, seed: int = C.SEED,
                 max_topic_size: int = 12, min_topic_size: int = 2):
        self.rng = np.random.default_rng(seed)
        self.max_topic_size = max_topic_size
        self.min_topic_size = min_topic_size
        self._build_topics(pool)

    # -- step 1 & 2 --------------------------------------------------------
    def _build_topics(self, pool: pd.DataFrame) -> None:
        pos = pool[pool.is_duplicate == 1]
        uf = UnionFind()
        text_of: dict[str, str] = {}
        for q1, q2 in zip(pos.question1, pos.question2):
            a, b = qid_of(q1), qid_of(q2)
            text_of.setdefault(a, q1)
            text_of.setdefault(b, q2)
            uf.union(a, b)

        members: dict[str, list[str]] = {}
        for qid in list(text_of):
            members.setdefault(uf.find(qid), []).append(qid)

        self.topics: list[list[str]] = []
        for root, mem in members.items():
            if self.min_topic_size <= len(mem) <= self.max_topic_size:
                self.topics.append([text_of[m] for m in mem])
        self.rng.shuffle(self.topics)

        # hard negatives: q -> [questions labelled NOT duplicate with q]
        neg = pool[pool.is_duplicate == 0]
        self.hard_neg: dict[str, list[str]] = {}
        for q1, q2 in zip(neg.question1, neg.question2):
            self.hard_neg.setdefault(q1, []).append(q2)
            self.hard_neg.setdefault(q2, []).append(q1)

        # step 3: Zipf popularity over topics
        ranks = np.arange(1, len(self.topics) + 1, dtype=float)
        w = 1.0 / np.power(ranks, C.STREAM_ZIPF_S)
        self.topic_p = w / w.sum()

    # -- step 4 ------------------------------------------------------------
    def generate(self, n: int, duplicate_rate: float = C.STREAM_DUPLICATE_RATE,
                 hard_negative_rate: float = 0.35,
                 exact_repeat_rate: float = 0.25,
                 mean_session_len: float = 4.0) -> Workload:
        if len(self.topics) < 50:
            raise RuntimeError("topic pool too small to build a stream")

        events: list[StreamEvent] = []
        seen_topics: list[int] = []
        seen_set: set[int] = set()
        seen_forms: dict[int, list[str]] = {}
        emitted_any: list[str] = []
        next_fresh = 0
        session_id, turn = 0, 0

        while len(events) < n:
            if turn >= max(1, int(self.rng.poisson(mean_session_len)) + 1):
                session_id += 1
                turn = 0

            do_repeat = (seen_topics and self.rng.random() < duplicate_rate)

            if do_repeat:
                p = self.topic_p[seen_topics]
                p = p / p.sum()
                tid = int(self.rng.choice(seen_topics, p=p))
                forms = self.topics[tid]
                used = seen_forms.setdefault(tid, [])
                unused = [f for f in forms if f not in used]
                if unused and self.rng.random() > exact_repeat_rate:
                    q, is_exact = str(self.rng.choice(unused)), False
                else:
                    q, is_exact = str(self.rng.choice(used or forms)), True
                if q not in used:
                    used.append(q)
                events.append(StreamEvent(len(events), q, tid, True, is_exact,
                                          False, session_id, turn))
                emitted_any.append(q)
            else:
                use_hn = (emitted_any and self.rng.random() < hard_negative_rate)
                q = None
                if use_hn:
                    for _ in range(8):
                        anchor = str(self.rng.choice(emitted_any[-200:]))
                        cands = self.hard_neg.get(anchor)
                        if cands:
                            q = str(self.rng.choice(cands))
                            break
                if q is not None:
                    # A distractor gets its own singleton topic: it must NOT be
                    # served from the anchor's cached response.
                    tid = -(len(events) + 1)
                    events.append(StreamEvent(len(events), q, tid, False, False,
                                              True, session_id, turn))
                else:
                    while next_fresh < len(self.topics) and next_fresh in seen_set:
                        next_fresh += 1
                    if next_fresh >= len(self.topics):
                        break
                    tid = next_fresh
                    next_fresh += 1
                    q = str(self.topics[tid][0])
                    seen_topics.append(tid)
                    seen_set.add(tid)
                    seen_forms[tid] = [q]
                    events.append(StreamEvent(len(events), q, tid, False, False,
                                              False, session_id, turn))
                emitted_any.append(q)
            turn += 1

        wl = Workload(events=events)
        wl.meta = {
            "generator": "QQP positive-edge components + Zipf popularity",
            "seed_params": {
                "duplicate_rate": duplicate_rate,
                "hard_negative_rate": hard_negative_rate,
                "exact_repeat_rate": exact_repeat_rate,
                "zipf_s": C.STREAM_ZIPF_S,
                "max_topic_size": self.max_topic_size,
                "mean_session_len": mean_session_len,
            },
            "n_topics_available": len(self.topics),
            **wl.describe(),
        }
        return wl


def build_workloads(stream_pool: pd.DataFrame, lengths=C.STREAM_LENGTHS,
                    seed: int = C.SEED, cache_dir: Path | None = None
                    ) -> dict[int, Workload]:
    """One nested family of streams: the 500-event stream is the prefix of the
    1000-event one, and so on. Prefix nesting is what makes the
    'across dataset snapshots' comparison in the paper meaningful -- the
    curves then show cache warm-up, not four unrelated samples.
    """
    cache_dir = cache_dir or C.DATA_DIR / "workloads"
    cache_dir.mkdir(exist_ok=True, parents=True)
    longest = max(lengths)
    path = cache_dir / f"stream_{longest}_seed{seed}.json"
    if path.exists():
        full = Workload.load(path)
    else:
        full = WorkloadGenerator(stream_pool, seed=seed).generate(longest)
        full.save(path)
    out: dict[int, Workload] = {}
    for L in sorted(lengths):
        w = Workload(events=full.events[:L], meta=dict(full.meta))
        w.meta.update(w.describe())
        w.meta["prefix_of"] = longest
        out[L] = w
    return out
