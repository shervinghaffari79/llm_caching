"""
ReuseGate agent for the SOLAR harness' experience memory (label-free semantic-demand admission).
Added as a NEW method; none of the authors' files is changed except three registration lines (see apply_patch.py).

Defined in docs/PREREGISTRATION_round2c.md, Stage B, before any result existed. Mirrors semcache.cache.SemanticAdmissionPolicy:
  score(e) = hits(e) + beta * demand(e)
  hits(e)    decayed count of requests for which e was among the retrieved experiences;
  demand(e)  decayed count of requests whose question lies within cosine `radius` of e's question (how often questions like
             e's are asked), initialised at admission from the remembered request history (TinyLFU-style frequency estimate);
  admission  when the memory is full a newcomer is stored only if its score exceeds the lowest resident score (ties reject);
  eviction   the lowest resident score (ties: oldest);
  aging      every `period` requests all counts and remembered weights halve.
Uses no feedback and no labels: only embeddings of the questions and the retrieval history.
"""
import numpy as np
from typing import Dict, List, Union
from pydantic import Field

from src.agent.embedder import EmbedderAgent, EmbedderAgentConfig


class ReuseGateAgentConfig(EmbedderAgentConfig):
    capacity: int = Field(default=500, description="Maximum number of memory entries")
    radius: float = Field(default=0.6, description="Cosine radius for 'a question like this one'")
    beta: float = Field(default=1.0, description="Weight of semantic demand against own hits")
    period: int = Field(default=1000, description="Requests between halvings of all counts")
    history: int = Field(default=4000, description="Remembered requests used to estimate a newcomer's demand")


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _question_of(content: str) -> str:
    if content.startswith("Q: "):
        return content[3:].split("\n", 1)[0]
    return content


class ReuseGateAgent(EmbedderAgent):
    def __init__(self, config: ReuseGateAgentConfig = ReuseGateAgentConfig()):
        super().__init__(config)
        self.rg = config
        self.total_candidates = 0
        self.total_updates = 0
        self.total_evictions = 0
        self._active_count = 0
        self.hits: Dict[int, float] = {}
        self.demand: Dict[int, float] = {}
        self.qvec: Dict[int, np.ndarray] = {}
        self._hist_v: List[np.ndarray] = []
        self._hist_w: List[float] = []
        self._requests = 0

    # ---- bookkeeping --------------------------------------------------------------------------
    def _active_indices(self) -> List[int]:
        return [i for i, m in enumerate(self.metadata) if not m.get("deleted", False)]

    def _history_demand(self, qv: np.ndarray) -> float:
        if not self._hist_v:
            return 0.0
        H = np.vstack(self._hist_v)
        return float(np.dot(H @ qv >= self.rg.radius, np.asarray(self._hist_w)))

    def _age(self):
        for i in self.hits:
            self.hits[i] *= 0.5
        for i in self.demand:
            self.demand[i] *= 0.5
        self._hist_w = [w * 0.5 for w in self._hist_w]

    def _score(self, i: int) -> float:
        return self.hits.get(i, 0.0) + self.rg.beta * self.demand.get(i, 0.0)

    # ---- request side ---------------------------------------------------------------------------
    def retrieve_memory(self, content: str, k=10) -> List[str]:
        results = super().retrieve_memory(content, k)
        qv = _unit(self._embed(content))
        self._requests += 1
        self._hist_v.append(qv)
        self._hist_w.append(1.0)
        if len(self._hist_v) > self.rg.history:
            self._hist_v.pop(0)
            self._hist_w.pop(0)
        # demand: every resident entry whose question is near this request
        for i, qe in self.qvec.items():
            if not self.metadata[i].get("deleted", False) and float(qe @ qv) >= self.rg.radius:
                self.demand[i] = self.demand.get(i, 0.0) + 1.0
        # hits: entries actually retrieved for this request (same search as the base class)
        if results and self.index.ntotal > 0:
            D, I = self.index.search(np.array([self._embed(content)]), min(k * 2, len(self.metadata)))
            got = 0
            for idx in I[0]:
                if 0 <= idx < len(self.metadata) and not self.metadata[idx].get("deleted", False):
                    self.hits[idx] = self.hits.get(idx, 0.0) + 1.0
                    got += 1
                    if got >= k:
                        break
        if self._requests % self.rg.period == 0:
            self._age()
        return results

    # ---- storage side ---------------------------------------------------------------------------
    def _compact_index(self):
        import faiss as _faiss
        keep, vecs, meta = {}, [], []
        for old, m in enumerate(self.metadata):
            if not m.get("deleted", False):
                keep[old] = len(meta)
                try:
                    vecs.append(self.index.reconstruct(old))
                except Exception:
                    vecs.append(self._embed(m["content"]))
                meta.append(m)
        self.index = _faiss.IndexFlatL2(self.config.embedding_dim)
        if vecs:
            self.index.add(np.array(vecs, dtype=np.float32))
        self.metadata = meta
        for name in ("hits", "demand", "qvec"):
            old = getattr(self, name)
            setattr(self, name, {keep[i]: v for i, v in old.items() if i in keep})
        self._active_count = len(meta)

    def add_memory_reusegate(self, content: str, doc_id=None) -> bool:
        self.total_candidates += 1
        qv = _unit(self._embed(_question_of(content)))
        d_new = self._history_demand(qv)
        if self._active_count >= self.rg.capacity:
            act = self._active_indices()
            scores = [self._score(i) for i in act]
            low = min(scores)
            if self.rg.beta * d_new <= low:           # TinyLFU rule: a tie rejects the newcomer
                return False
            victim = act[min(range(len(act)), key=lambda j: (scores[j], act[j]))]
            self.metadata[victim]["deleted"] = True
            self._active_count -= 1
            self.total_evictions += 1
            for d in (self.hits, self.demand, self.qvec):
                d.pop(victim, None)
            if self.total_evictions % 20 == 0:
                self._compact_index()
        if doc_id is None:
            doc_id = f"reusegate_doc_{len(self.metadata)}"
        self.metadata.append({"doc_id": doc_id, "content": content})
        self.index.add(np.array([self._embed(content)]))
        idx = len(self.metadata) - 1
        self._active_count += 1
        self.hits[idx], self.demand[idx], self.qvec[idx] = 0.0, d_new, qv
        self.total_updates += 1
        return True

    def add_conversation_to_memory(self, messages: List[Dict[str, str]], conversation_idx: Union[int, str] = 0):
        if isinstance(conversation_idx, int):
            conversation_idx = str(conversation_idx)
        for msg_idx, msg in enumerate(messages):
            self.add_memory_reusegate(f"Speaker {msg['role']} says: {msg['content']}", f"conv_{conversation_idx}_{msg_idx}")

    def get_reusegate_metrics(self) -> Dict:
        return {"total_candidates": self.total_candidates, "total_updates": self.total_updates,
                "total_evictions": self.total_evictions, "active_memories": self._active_count,
                "capacity": self.rg.capacity, "update_rate": self.total_updates / max(self.total_candidates, 1)}
