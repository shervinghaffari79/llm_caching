# Reproduction status

Where the re-evaluation stands, what it has established, and how to resume.

**Everything below was produced on a 4-thread CPU (no CUDA) with a 3,000-pair
cap** (`SEMCACHE_META_TRAIN_CAP=3000`) so the suite would finish. The full-size
run on a GPU will move the absolute numbers — particularly the latencies and
anything trained on the full 30,000-pair meta set. The *relationships* between
the numbers (which is what review questioned) are already established and are
not expected to reverse.

## Stage status

| Stage | What it answers | Status |
|---|---|---|
| exp0 data audit | duplication + leakage in QQP | **done** |
| exp1 latency breakdown | per-component cost, online vs offline batching | **done** |
| exp2 end-to-end | response time + token savings per configuration | **4 of 5 arms done**, see below |
| exp3 ablation | Table 7 recomputed on the real subset | **done** |
| exp4 meta-classifier | Table 4 + fold-scheme + normalisation | **done** |
| exp5 dynamic cache | hit ratio as a stream property | **done** |
| exp6 baselines | GPTCache / SCALM / VectorQ / ContextCache | **not run** |
| exp7 eviction | policies + weight sensitivity sweep | **not run** |
| exp8 cheap vs expensive | F1 per millisecond, the title claim | **not run** (reads exp1+exp3) |
| exp9 report | consistency check + assembled tables | **not run** (reads everything) |

exp2 completed all four cache configurations and was killed during the no-cache
baseline, so no JSON was written. The four served arms are preserved verbatim in
`results/raw/exp2_end_to_end.PARTIAL.log` and are reproduced below.

## What has been established

### 1. The 540 ms / 311 ms pair is arithmetically impossible

exp2, 300-event stream, `gpt-5.4-nano` replay, batch size 1, one process:

| Configuration | Overhead/req | Avg response | Hit ratio | Residual |
|---|---|---|---|---|
| C1 bi-encoder only | 67.56 ms | 1.346 s | 58.0% | +0.4 ms **consistent** |
| C2 bi + geometric/text + LightGBM | 70.17 ms | 1.389 s | 56.3% | +0.8 ms **consistent** |
| C3 bi + cross-encoder | 310.25 ms | 1.617 s | 57.0% | −4.7 ms **consistent** |
| C4 full stacking | 310.61 ms | 1.634 s | 56.3% | +0.1 ms **consistent** |

The identity `avg_response = avg_overhead + miss_ratio × avg_generation` holds to
within 5 ms in every arm. Adding the cross-encoder **raises** average response
time (1.389 s → 1.634 s); it cannot lower it. A configuration paying ≥540 ms of
similarity check cannot average 311 ms — the published pair describes two
different systems, and the response-time figure has to be the bi-encoder path.

### 2. The cross-encoder is not doing measurable work (exp3)

Recomputed on the actual held-out 3,000-pair 60/40 subset, group-disjoint folds,
scaler fitted inside each fold:

- Cross-encoder removal: **ΔF1 = +0.0016**, bootstrap 95% CI **[−0.0051, +0.0086]**,
  paired t **p = 0.72**. The interval contains zero.
- Split-gain importance: **element-wise 96.5%**, cross-encoder **1.5%**,
  text 1.3%, distance 0.4%, bi-cosine 0.3%.
- Bi-cosine alone (0.8591) **beats** bi+cross (0.8484) on this subset — worth a
  sentence in the paper.

### 3. Table 7's ±0.004 cannot come from a 3,000-pair CV

Measured fold-to-fold std on the ablation subset is **±0.013 to ±0.031**, three
to eight times the published ±0.004. A 3,000-pair 5-fold CV is simply less
stable than a 30,000-pair one. Independently: the recomputed reference gives
F1 = 0.8366 on the balanced reference set vs 0.8747 on the 3k 60/40 subset —
**+0.0381 apart**, as two genuinely different measurements should be.

### 4. Leakage is real and quantified (exp0/exp4)

- Under plain `StratifiedKFold`, **27.9%** of validation questions already appear
  in training. Under `StratifiedGroupKFold` on duplicate-graph components: **0%**.
- Worth **+0.0301 F1** of inflation (0.8667 → 0.8366).
- Row-level cleaning removed 559 exact duplicate rows, 250 mirrored pairs,
  869 self-pairs and 35 label conflicts.
- **The normalisation question specifically: ΔF1 = +0.0001.** Real but negligible
  for a tree ensemble. The fold scheme matters ~300× more. Both are now done
  correctly and both are reported.

### 5. Hit ratio is dynamic, and the code shows it (exp5)

Same model, same stream, varying only cache state:

- capacity ∞ / 400 / 150 → **62.5% / 58.4% / 47.1%** (evictions 0 / 432 / 908)
- stream length 500→2000 → **59.0% → 62.5%** (warm-up)
- hit ratio on *servable* repeats holds at **~98.4%** — the similarity model's
  own accuracy, with cold-start and eviction factored out
- precision of hits **~86%**: about one served hit in seven returns an answer
  generated for a different question. No static pair-level metric expresses this.

### 6. Latency breakdown (exp1, CPU, batch size 1)

cross-encoder **250.30 ms**, bi-encoder **58.52 ms**, LightGBM **1.57 ms**,
FAISS search **1.38 ms**, all feature groups **0.02–0.06 ms** combined.
Cross-encoder = **80.3%** of full-stacking overhead.

One honest caveat: on this CPU, batch-32 gave a **1.0× speedup** over batch-1 for
the cross-encoder — 4 threads are already saturated. The online/offline batching
distinction is real but is *not* the explanation on this hardware. Expect it to
reappear on a GPU. The code reports the measured ratio rather than asserting a
conclusion.

## Resume

```bash
export SEMCACHE_META_TRAIN_CAP=3000   # drop this entirely on a GPU
python run_all.py --only exp2 exp6 exp7 exp8 exp9
```

On a GPU, drop the cap and run the whole thing:

```bash
python run_all.py
```

The 199 prefetched `gpt-5.4-nano` answers are committed in
`data/llm_cache/pool_gpt-5.4-nano.jsonl`, so resuming exp2/exp6 will not re-pay
for those API calls. Put the key in `.env` as `OPENAI_API_KEY=...` — `.env` is
gitignored and no key appears in any committed artefact.

Note `--llm-mode replay` blocks for each recorded generation latency, so a
2,000-event stream costs real wall-clock time per configuration. exp5/exp7 use
the non-blocking path (their headline is hit ratio, not latency) and record that
choice in their manifest.

## Bugs found by running the suite

1. **Distractor ground truth.** Injected hard negatives were each given a
   fabricated unique topic id, so a distractor that was legitimately equivalent
   to something else in the cache counted as a false hit. This pushed the
   realised hit ratio *above* the oracle ceiling — incoherent. Distractors now
   carry their true topic when they have one.
2. **`Workload.save` wrote cp1252 while `load` read UTF-8** — crashed on the
   first non-ASCII QQP question. All file IO now passes `encoding` explicitly.
3. **exp5 capacities never bound.** Only misses insert, so a 2,000-event stream
   never exceeded ~800 entries and the 1500/2000 arms evicted nothing. Now
   (∞, 400, 150).
4. **Cold iterations skewed latency means** — p95 came out below the mean. First
   five iterations are now measured and discarded; cross-encoder added to warmup.
5. **`train_metas` cross-encoded all 30k unconditionally** (~2 h on CPU) with no
   way to bound it. Added `SEMCACHE_META_TRAIN_CAP`.
6. Progress bars corrupted captured logs (now tty-gated), and exp1 asserted a
   batching conclusion its own measurement didn't support (now adaptive).
