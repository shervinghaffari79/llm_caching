"""Fill the answer pool for every stream the revision experiments use, so
replay runs never fall back to simulated answers. Uses the served LLM
(gpt-5.4-nano) and OPENAI_API_KEY from .env."""
from semcache.experiments._common import ensure_pool, get_stream
from semcache.experiments.exp15_workload_sensitivity import settings
from semcache.experiments.exp18_eviction import TUNE_SEED

queries = []
for s in (42, 43, 44, 45, 46, TUNE_SEED):
    queries += get_stream(s, 2000).queries()
for _, _, p in settings():
    for s in (42, 43, 44):
        queries += get_stream(s, 2000, **p).queries()
print("streams built; unique queries:", len(set(queries)), flush=True)
print(ensure_pool(queries, live=True), flush=True)
