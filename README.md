# Semantic caching study — reproduction and consistency harness

Code accompanying *"Stacking Ensembles and Decay-Aware Eviction for Semantic
Caching in LLM-based Systems: When Cheap Features Beat Expensive Encoders"*,
written to answer a specific set of reviewer concerns with measurements rather
than assertions.

The organising principle: **no runtime number is reported without naming the
configuration that produced it**, and **response time, overhead, hit ratio and
token savings for a given configuration all come from one run of one process**,
so they are arithmetically forced to agree.

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env      # add your OPENAI_API_KEY
python run_all.py
```

Useful variants:

```bash
python run_all.py --quick                 # small sizes, no API calls
python run_all.py --llm-mode sim          # skip the API entirely
python run_all.py --only exp3 exp8        # just the ablation and the title check
python run_all.py --verify-live 100       # validate replay against the live API
```

A GPU is strongly recommended: the cross-encoder (`stsb-roberta-large`) is the
dominant cost and is ~40× slower on CPU. Everything runs on CPU, just slowly.

Artefacts land in `results/`:

| Path | Contents |
|---|---|
| `results/CONSISTENCY_REPORT.md` | the assembled findings, all numbers auto-filled |
| `results/raw/*.json` | per-experiment raw output with hardware manifests |
| `results/tables/*.tex` | drop-in LaTeX replacements for the manuscript tables |
| `results/figures/*.png` | regenerated figures |

---

## The four named configurations

Every result is tagged with one of these. They are defined once, in
`semcache/config.py`, and used by both the latency measurement and the serving
loop — there is no separate code path per experiment.

| Key | Bi-encoder | Cross-encoder | Geometric feats | Text feats | LightGBM | Decision |
|---|---|---|---|---|---|---|
| `C1_bi_only` | yes | no | no | no | no | cosine threshold |
| `C2_bi_geo_lgb` | yes | **no** | yes | yes | yes | meta prob ≥ τ |
| `C3_bi_cross` | yes | yes | no | no | no | both thresholds |
| `C4_full_stacking` | yes | yes | yes | yes | yes | meta prob ≥ τ |

`C2` is the "cheap features" configuration the title refers to.

---

## How each reviewer concern is addressed

### 1. 539.80 ms overhead vs 0.311 s average response time

`semcache/reporting.py::audit_manuscript_claims` applies the arithmetic
directly: overhead is paid on **every** request, before the hit/miss outcome is
known, so 539.80 ms is a hard floor on the average. No hit ratio reconciles it
with 0.311 s, and 0.311 s is below the cross-encoder term alone.

`exp1_latency_breakdown` re-measures every stage at **batch size 1** (the only
regime a live cache can use) and also reports the **batch-32 amortised** cost,
because quoting an offline-batched encoder cost beside an online serving
average is the mechanism that produced the impossible pair. The batching
speed-up factor for the cross-encoder is printed explicitly.

`exp2_end_to_end` then runs all four configurations plus a no-cache baseline
over the same stream in one process, and checks

```
avg_response_time == avg_overhead + miss_ratio × avg_generation_latency
```

reporting the residual per configuration. A configuration whose numbers cannot
satisfy this is flagged `INCONSISTENT` rather than published.

**Expected conclusion:** the sub-second average is only reachable by `C1`/`C2`.
The response-time and token-savings claims must be relabelled to
*bi-encoder caching* (or *cheap-feature stacking*), not *stacking ensemble
caching*. `results/CONSISTENCY_REPORT.md` states the relabelling explicitly.

### 2. Which components were active for response-time and token-savings

`exp2_end_to_end` emits a `component_activation_table` — a per-configuration
yes/no matrix over bi-encoder, cross-encoder, geometric features, text
features and LightGBM — and every row of the results table carries its own
`components_active`. Token savings are computed on the same run as the
response times, so the two cannot disagree about which system produced them.

Token savings are reported twice: over all hits, and over **correct hits
only**. A false hit returns the wrong answer; counting its tokens as "saved"
overstates the benefit.

### 3. Table 7 duplicating Table 4's values

`exp3_ablation` recomputes the ablation on the actual held-out 3,000-pair
60/40 subset, and in the same run recomputes the 30k reference, so the two
appear side by side with their difference. The subset is **component-disjoint**
from the meta-classifier set — verified at the component, question and pair
level in `exp0_data_audit` — not merely pair-disjoint.

It also adds what the original table lacked: a bootstrap 95% CI and a paired
test on the cross-encoder removal, so "adds <0.001 F1" becomes "the CI includes
zero" or does not.

### 4. Workload simulation — how the query sequence was generated

`semcache/workload.py`, documented in full in its module docstring:

1. equivalence classes ("topics") from connected components of QQP's
   **positive** edges, capped in size so noisy transitive closure cannot
   manufacture easy hits;
2. **hard negatives** injected from QQP's negative pairs — lexically similar,
   semantically distinct — because a stream of unrelated questions makes any
   model look good;
3. Zipf(1.1) popularity over topics;
4. each step emits either a repeat (paraphrase or exact) of an already-seen
   topic, or a novel query (fresh topic or hard negative).

Stream lengths 500/1000/1500/2000 are **nested prefixes of one stream**, so the
curves show cache warm-up rather than four unrelated samples.

### 5. Normalisation inside each CV fold

`semcache/metaclf.py` puts `StandardScaler` inside an sklearn `Pipeline`, and
the pipeline is constructed and fitted **inside the fold loop** on training
indices only. The leak is structurally impossible, not merely intended to be
absent. `leak_check_experiment` deliberately runs the wrong protocol too and
reports the measured difference, so the paper can quote a magnitude.

`exp0_data_audit` shows the larger issue: QQP is a graph, and under plain
`StratifiedKFold` a large share of validation questions have already been seen
in training via a different pair. Every headline number is reported under both
`StratifiedKFold` and `StratifiedGroupKFold` over question components, with the
leakage-driven inflation quantified.

### 6. Hit ratio as a dynamic, not static, metric

`exp5_dynamic_cache` runs a **stateful** cache: the index starts empty, a first
occurrence is always a miss, a repeat can only hit if its equivalent is still
resident, and every miss inserts and may evict. Three distinct quantities are
reported and are not interchangeable:

- `hit_ratio_pct` — realised hits / requests
- `oracle_hit_ratio_pct` — requests for which an equivalent entry was *resident*
  at arrival (the achievable ceiling)
- `hit_ratio_on_servable_repeats_pct` — realised / achievable, isolating
  similarity-model error from cold-start and eviction effects

Plus `precision_of_hits_pct`: the fraction of served hits that were genuinely
equivalent. A static pair score cannot express a false hit's cost.

The experiment demonstrates the metric is dynamic by varying only cache
capacity and stream length and showing the hit ratio move.

### 7. Comparison with GPTCache, SCALM, VectorQ, ContextCache

`semcache/baselines.py` implements all four decision rules on one shared
substrate (same bi-encoder, same FAISS index, same stream, batch size 1), so
the comparison isolates the algorithm rather than the engineering. These are
faithful re-implementations of the published logic, **not** the authors' code —
stated in the module docstring and in the emitted `fairness_statement`, and it
must be stated in the paper.

Each baseline gets its best case; VectorQ is fed **ground-truth** verification
feedback, which is strictly more informative than the LLM-judge signal it would
have in production, making its numbers an upper bound.

**ContextCache — yes, it can be used as a baseline**, with one caveat that the
paper must carry: QQP has no dialogue context, so sessions are synthesised by
the workload generator and ContextCache is given the true session boundaries.
That is a fair test on this traffic but not a test of ContextCache on the
multi-turn data it targets.

### 8. Does the ablation support the title?

`exp8_cheap_vs_expensive` joins measured latency to measured F1 and reports
ΔF1 per 100 ms per component, with explicit definitions:

- **cheap features** — arithmetic over bi-encoder vectors the retrieval step
  has already computed (cosine, L2/L1 distance, element-wise product and
  absolute difference, surface text statistics). Marginal cost is a vector op.
  They are **not encoder-free**: they consume bi-encoder output. The claim is
  that no *second, pairwise* encoder is needed.
- **expensive encoder** — the cross-encoder: a second forward pass with
  quadratic attention over the concatenated pair, per candidate, unable to
  reuse either side's cached embedding.

The script writes a `title_supported` verdict from the two conditions that
would have to hold (cheap wins per millisecond **and** the cross-encoder's
gain fails a significance test), and includes a written argument for why the
cross-encoder might still be retained as an *optional* stage — the question the
review asks directly.

### 9. Duplicate samples and re-verified F1

`exp0_data_audit` reports and removes: exact duplicate rows, mirrored `(q2,q1)`
duplicates, self-pairs, and label conflicts; then reports the component
structure of the question graph. All downstream F1 numbers are computed on the
cleaned, component-disjoint splits.

### 10. Eviction policy weight sensitivity

`exp7_eviction` sweeps the Semantic Decay simplex `(w1,w2,w3)` instead of
fixing the tuned optimum, and reports the spread and the fraction of weight
settings that still beat LRU — the sensitivity study the manuscript's
conclusion promises.

---

## LLM backend

`gpt-5.4-nano` with `reasoning_effort="none"` (verified: `reasoning_tokens=0`),
system prompt *"You are a helpful assistant who responds briefly."*, matching
the manuscript's Qwen3-4B setup in intent.

Three modes, in `semcache/llm.py`:

- **prefetch → replay** (default). Every distinct query is answered once by the
  real API with its latency and token usage recorded; each configuration then
  replays the stream, reproducing that query's recorded generation latency.
  This makes the LLM contribution identical across the four configurations, so
  differences in the reported averages are attributable to the cache.
- **live** — the API is called inside the serving loop. `--verify-live N`
  cross-checks that replayed and live averages agree, so replay is a
  variance-reduction technique, not a substitute for measurement.
- **sim** — no API; latency drawn from a fitted lognormal. For smoke tests.

The API key is read from `OPENAI_API_KEY` (or a gitignored `.env`) and is never
written into the repository.

---

## Layout

```
semcache/
  config.py       paths, models, the four ServingConfigs, hardware profile
  data.py         QQP loading, duplicate audit, component-disjoint splits
  features.py     feature groups, per-group cost accounting
  encoders.py     bi/cross encoder wrappers with stage timers
  metaclf.py      CV with fold-internal scaling, paired tests, bootstrap CIs
  cache.py        CacheBase, VectorBase (FAISS), LRU/LFU/SemanticDecay
  workload.py     QQP → dynamic query stream
  pipeline.py     the serving loop; one source for all runtime numbers
  baselines.py    GPTCache / SCALM / VectorQ / ContextCache re-implementations
  llm.py          gpt-5.4-nano backend, token accounting, prefetch/replay
  reporting.py    LaTeX tables, figures, the consistency checker
  experiments/    exp0 … exp9
run_all.py
```

## Runtime

On a single modern GPU, a full `run_all.py` is roughly 1.5–3 h, dominated by
cross-encoding the 30k meta set (cached to `data/emb_cache/` after the first
run) and by the prefetch phase's API calls. `--quick` finishes in minutes.

## Reproducibility

Seed 42 throughout. Every result file carries a manifest with device, torch
version, thread count, batch regime, LLM model and seed. Embeddings and
cross-encoder scores are cached to disk keyed by model and sample, so reruns
are cheap and identical.
