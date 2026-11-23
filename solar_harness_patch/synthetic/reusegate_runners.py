"""ReuseGate runners for the SOLAR synthetic workloads (appended to a COPY of synthetic_workloads/runners.py; defined in
docs/PREREGISTRATION_round2c.md, Stage D). No change to any existing class.

reusegate    score = hits + beta * demand; demand = decayed count of earlier query embeddings within cosine `radius` of the item; admission by the
             TinyLFU rule (a newcomer enters a full cache only if beta * its demand exceeds the lowest resident score; ties reject); eviction of the lowest score
             (ties: oldest); halving of all counts every `period` queries; `history` remembered queries.
reusegate_e  same scores and eviction, always admits (admission ablation).
"""


class ReuseGateRunner(BaseRunner):
    name = "reusegate"
    admit_rule = True

    def __init__(self, capacity, retrieve_k=3, seed=42, radius=0.5, beta=1.0, period=1000, history=4000):
        super().__init__(capacity, retrieve_k, seed)
        self.radius, self.beta, self.period, self.history = radius, beta, period, history
        self._hq = []            # remembered query embeddings
        self._hw = []            # their weights
        self._hits = {}
        self._demand = {}
        self._nq = 0

    def _hist_demand(self, emb):
        if not self._hq:
            return 0.0
        H = np.stack(self._hq, axis=0)
        return float(np.dot(H @ emb >= self.radius, np.asarray(self._hw)))

    def on_retrieval(self, query, retrieved_ids, step):
        super().on_retrieval(query, retrieved_ids, step)
        q = query.embedding / (np.linalg.norm(query.embedding) or 1.0)
        self._nq += 1
        self._hq.append(q)
        self._hw.append(1.0)
        if len(self._hq) > self.history:
            self._hq.pop(0)
            self._hw.pop(0)
        for cid, entry in self.cache.items():
            if float(entry.item.embedding @ q) >= self.radius:
                self._demand[cid] = self._demand.get(cid, 0.0) + 1.0
        for rid in retrieved_ids:
            if rid in self.cache:
                self._hits[rid] = self._hits.get(rid, 0.0) + 1.0
        if self._nq % self.period == 0:
            for d in (self._hits, self._demand):
                for k in d:
                    d[k] *= 0.5
            self._hw = [w * 0.5 for w in self._hw]

    def _score(self, cid):
        return self._hits.get(cid, 0.0) + self.beta * self._demand.get(cid, 0.0)

    def evict_one(self, incoming, step):
        if not self.cache:
            return
        victim = min(self.cache.items(), key=lambda kv: (self._score(kv[0]), kv[1].inserted_at))[0]
        self.cache.pop(victim)
        self._hits.pop(victim, None)
        self._demand.pop(victim, None)
        self.num_evictions += 1

    def _admit(self, item, step):
        if item.id in self.cache:
            return
        emb = item.embedding / (np.linalg.norm(item.embedding) or 1.0)
        d_new = self._hist_demand(emb)
        if self.admit_rule and len(self.cache) >= self.capacity:
            if self.beta * d_new <= min(self._score(c) for c in self.cache):
                return
        while len(self.cache) >= self.capacity:
            self.evict_one(item, step)
        self.cache[item.id] = CacheEntry(item=item, inserted_at=step, last_accessed=step, access_count=0)
        self._hits[item.id], self._demand[item.id] = 0.0, d_new
        self.num_admissions += 1


class ReuseGateEvictOnlyRunner(ReuseGateRunner):
    name = "reusegate_e"
    admit_rule = False


_RUNNER_REGISTRY.update({"reusegate": ReuseGateRunner, "reusegate_e": ReuseGateEvictOnlyRunner})
