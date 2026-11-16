"""Report the overlap between QQP evaluation streams (seeds 42-46) and tuning/reference streams
(seeds 1042-1049), with and without SEMCACHE_DISJOINT_QQP=1. Run once per setting."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semcache.data import UnionFind, build_splits, qid_of
from semcache.experiments._common import get_stream

ev = {s: {e.query for e in get_stream(s, 2000).events} for s in (42, 43, 44, 45, 46)}
ev_all = set().union(*ev.values()); ev_ids = {qid_of(q) for q in ev_all}
pos = build_splits().stream_pool; pos = pos[pos.is_duplicate == 1]
uf = UnionFind()
for a, b in zip(pos.question1, pos.question2): uf.union(qid_of(a), qid_of(b))
ev_roots = {uf.find(q) for q in ev_ids if q in uf.parent}
out = {"disjoint_mode": os.environ.get("SEMCACHE_DISJOINT_QQP") == "1", "eval_distinct_questions": len(ev_all)}
for name, seeds in (("tune", (1042, 1043, 1044)), ("ref", (1045, 1046, 1047, 1048, 1049))):
    qs = {e.query for s in seeds for e in get_stream(s, 2000).events}
    ids = {qid_of(q) for q in qs}
    out[name] = {"distinct_questions": len(qs), "shared_questions": len(qs & ev_all),
                 "shared_normalised_ids": len(ids & ev_ids),
                 "shared_components": len({uf.find(q) for q in ids if q in uf.parent} & ev_roots)}
print(json.dumps(out, indent=1))
