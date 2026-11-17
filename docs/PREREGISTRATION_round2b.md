# Pre-registration, round 2b (written before any of these runs)

Scope: reviewer comments R3.1/R5.2 (stacking vs a calibrated bi-encoder), R3.3 (precision on real logs),
R3.2 (ReuseGate on one log), R5.4 (baseline fidelity, feedback parity). Standing rules apply: no evaluation row
is used for tuning, every tuning step is causal, every overlap is disclosed, negative results are reported as
found. Nothing here is generated, replayed or simulated: serving uses `NullBackend`, so only label-based metrics
(hit ratio, precision, coverage) are reported.

## E1. Audit-budgeted decision rules (R3.1, R5.2, R3.3, R5.4 feedback parity)

Question: with the same small labelled audit sample from the target log, does a learned C2 beat a properly
calibrated bi-encoder (C1)?

Data: SemBenchmarkSearchQueries, rows in dataset order (the log has no timestamps).
* Audit pool: rows 0-39,999, never served. Evaluation windows: rows 60,000-79,999, 80,000-99,999,
  100,000-119,999, 120,000-139,999 (20,000 requests each, empty cache at the start of each window).
  The 20,000-row gap protects against adjacency; the audit pool precedes every evaluation window.
* Audit sample: the nearest earlier neighbour (cosine >= 0.6) of each audit-pool row, labelled by shared
  `id_set`; B pairs drawn at random, B in {250, 500, 1,000, 2,000, 5,000, 10,000}, three draws per budget.

Systems (all fitted or calibrated on the audit sample only):
* C1-cal: cosine threshold chosen from the audit pairs as the lowest threshold whose empirical precision is at
  least P* (accepted audit pairs >= 30); P* in {0.90, 0.95}.
* C2-in: LightGBM on (cosine, L1/L2, element-wise product and |difference|, six text features), trained on 80% of
  the audit pairs; probability threshold chosen on the other 20% by the same rule. C1 uses all B pairs for its
  threshold, which favours C1.
* C2-qqp: the existing QQP-trained C2, at its default threshold and at the same audit rule (transfer baseline).

Metrics per (system, budget, draw, window): realised precision of hits, hit ratio, and the gap between realised
and target precision. Matched-coverage curves for C1 (grid) and C2-in (grid) at B in {1,000, 10,000}.

Decision rules, fixed now:
* Favourable to C2 if, at B >= 1,000, either (a) at matched hit ratio C2-in is more precise than C1 by at least
  2 points with a 95% CI (windows x draws) above 0, or (b) the realised-minus-target precision error of C2-in is
  smaller in absolute value than C1-cal's with a 95% CI above 0.
* Unfavourable otherwise. Then the gap is diagnosed (see "Gap protocol") before any change.

## E2. ReuseGate on further domains (R3.2)

Same causal layout and statistics as exp37/exp43 (reference traffic strictly earlier than tuning traffic, tuning
strictly earlier than evaluation; Holm over lambda in {0, 1, 2, 5, 10}).
* LmArena (SemBenchmarkLmArena, LLM prompts, labelled by `id_set`) in three block rotations.
* TripClick (health search, first-click label), strictly chronological: reference rows 1-50,000, tuning
  50,001-100,000, evaluation 100,001-150,000 of the final 150,000 requests.
Claim holds in a domain only if ReuseGate minus W-TinyLFU is Holm-significant and positive at lambda 0 and 1.

## E3. Baseline fidelity by independent reimplementation (R5.4)

Third-party code is not executed. W-TinyLFU is reimplemented from the published description (4-bit count-min
sketch with periodic halving, doorkeeper, 1% window, segmented LRU main). The exact-count policy used so far is
renamed in the text. ReuseGate vs both on the logs of E2.

## Gap protocol

If a result is unfavourable, the gap is first diagnosed using the saved per-configuration outputs and, if needed,
new runs on tuning windows only. Exactly one change is specified in writing, with its decision rule, before it is
run on any evaluation window; evaluation windows are then used once. A change is not adopted because it helps on
data that guided it.

## Amendment 1 to E1 (17 Nov 2026, after the first cells of windows 2 and 3 had run)

What was seen: the budget-free C1 curve on evaluation windows 2 and 3 reaches at most about 83% precision (7% hit
ratio, cosine 0.96); no setting reaches 90%. Consequently the pre-registered targets P* = 0.90 and 0.95 were
infeasible for most small audit samples, and where feasible the realised precision was about 83%.

What is unchanged: the primary decision rules (a) matched-coverage precision and (b) realised-minus-target error,
the budgets, the audit pool, the windows and the systems. The pre-registered targets are still run and reported,
including the infeasible rate.

What is added (secondary, labelled as an amendment): the same pipeline with targets P* = 0.70 and 0.80, chosen
because they lie inside the range the budget-free curve covers. They are reported as exploratory, next to, never
in place of, the pre-registered targets.

## E4. Cheap features that target what the cosine misses (gap protocol, written before any E4 run)

Gap: C1 has almost no discriminating information among confusable pairs (top quintile of cosine: AUC 0.62-0.65),
and the current C2 text features are only lengths, length ratios, word Jaccard and edit ratio. Hypothesis: a few
cheap lexical features aimed at the usual causes of high-cosine false hits (rare-word mismatches, numbers,
negation, question type, named entities) raise precision at fixed recall by a substantive margin.

Feature families (each O(length), microseconds per pair):
* F1 IDF-weighted word Jaccard, maximum IDF over the symmetric difference, IDF mass of the symmetric difference
  over the IDF mass of the union. IDF comes from unlabelled reference text only (training-fold questions for QQP,
  the audit-pool rows 0-39,999 for the search log).
* F2 numbers: count in each question, set equality, count of differing numbers.
* F3 negation: negation word in question A, in B, and their XOR.
* F4 question type: wh-word / auxiliary class of the first token, equality.
* F5 capitalised tokens (entity proxy): Jaccard and count of differences.
* F6 character 3-gram Jaccard.

Development (only here may a family be chosen): QQP ablation split (3,000 pairs), group-disjoint 5-fold CV.
A family is kept if adding it to C2 raises the top-quintile-of-cosine AUC by at least 0.01 with a 95% bootstrap
CI (over question components) above 0. C2+ = C2 plus the kept families.

Confirmation, run once, no further choices: 
* Q: QQP meta set (30,000 pairs), group-disjoint 5-fold CV, C1 vs C2 vs C2+.
* S: transfer to the search log. Models trained on all 30,000 QQP pairs; test pairs are the nearest earlier
  neighbour (cosine >= 0.6, window-local) of each request in rows 60,000-79,999; IDF from rows 0-39,999. No
  search-log label is used in training or selection.
Metrics: precision at 90% and 80% recall (pooled out-of-fold / pooled test), ROC-AUC, top-quintile AUC.

Decision rule: a substantive leap is a gain of C2+ over C1 in precision at 90% recall of at least 5 points with a
95% CI above 0 on Q and at least 3 points on S. If met, a stream-level matched-coverage test follows on windows not
used here; if not, the result is reported as no leap.

## E5-E9. Failure-mode slices, band diagnostics, attack robustness, benchmark datasets, cost (written before any run)

Systems compared everywhere: C1 (cosine), C2 (QQP-trained, current features), C2-diff (C2 plus three families fixed
now, regardless of any result: number mismatch F2, negation flag F3, capitalised-token mismatch F5 of E4), and C2+
(the E4-selected families, if any survive its development rule). Models are trained on QQP only unless stated.
All claims are reported as tables with 95% bootstrap CIs; wins, ties and losses are all reported.

### E5. Slice-based evaluation
* Perturbation slices built by deterministic rules (fixed seeds) from QQP duplicate pairs (held-out folds, group-disjoint)
  and from search-log duplicate pairs: number change, unit change, entity swap, time-period change, negation
  insertion or removal. One side of the pair is perturbed, so the pair becomes non-equivalent. A label-preserving
  control slice (case and punctuation) must stay accepted.
* External minimal pairs, zero-shot from the QQP-trained models: HEROS (dcml0714/Heros, six subsets), SemAntoNeg
  (3,152 entries), PAWS (labelled paraphrase / non-paraphrase). Their labels are used for scoring only.
* Matched coverage: each system's threshold is set so it accepts the same fraction, 80% and 90%, of the true
  duplicate pairs of the unperturbed base set. Metric: false-acceptance rate on each slice. Delta = system minus C1,
  95% bootstrap CI over source pairs, Holm over slices. A slice "differs" if |Delta| >= 5 points with a CI excluding 0.

### E6. Band diagnostic and decision-level comparison
* AUC and precision inside the band cosine in 0.726 +/- 0.05 (QQP) and 0.895 +/- 0.05 (search log).
* McNemar-style paired comparison at 90% matched recall: counts of C1 errors fixed by the other system and of new
  errors it adds, exact McNemar p-value, and an error taxonomy by detector (number, negation, capitalised, question
  type, length ratio) for fixed and added errors.

### E7. Key-collision robustness (non-adaptive adversary)
* Attack model after Zhang et al., "From Similarity to Vulnerability: Key Collision Attack on LLM Semantic Caching"
  (arXiv 2601.23088): the attacker plants source prompt + adversarial suffix, optimised by a GCG-style token search
  against a surrogate encoder to minimise 1 - cos(source+suffix, victim) plus a fluency penalty, so that benign victim
  queries hit it. Our own implementation from the description; the authors' code is not run.
* Non-adaptive: the attacker knows the encoder (surrogate or target) but not the C2 decision features.
* Metric: attack hit rate (fraction of victim queries served by the planted entry) at operating points matched on
  clean benign traffic (same benign hit ratio for every system); suffix lengths 5, 10, 20; surrogate = target and a
  smaller surrogate (transfer). An adaptive attacker is out of scope and stated as a limitation.

### E8. Benchmark datasets
* vCache SemBenchmarkClassification (37,836 rows) and SemBenchmarkCombo (27,500 rows: 3,500 LmArena prompts that have
  no match plus 24,000 classification queries). Combo is not a pure no-hit set: a hit is a guaranteed false hit only
  for the 3,500 no-match prompts. Hypothesis to check on the data, not assumed: a shared instruction template inflates
  cosine among short prompts. Report template share, cosine distribution of non-equivalent pairs, and for C1/C2 the
  false-hit rate on the no-match subset at matched hit ratio on the matchable subset.

### E9. Cost-sensitive analysis
* Per-request decision overhead on CPU, standard and optimised LightGBM inference (single thread, contiguous float32,
  booster predict without the sklearn wrapper), with and without the cross-encoder.
* Break-even false-hit price and net benefit per 1,000 requests for false-hit prices 0, 1, 2, 5, 10, 20, 50, using the
  no-match subset of Combo as the zero-benefit traffic. Generation cost uses the recorded latency of real answers in
  the repository's answer pool; nothing is simulated.

### E7 clarification (written before any E7 run)
Operating points: thresholds are set so each system accepts 90% (and 80%) of held-out true-duplicate QQP pairs
(recall-matched, as in E5). A victim query is "hijacked" when the decision rule accepts the planted attack entry as a
match for it; the attack entry is treated as the nearest candidate (an upper bound on the attacker, since a real cache
may have a closer benign entry). Controls: the payload alone and the payload plus a random suffix of equal length.
Decision rule: C2 or C2-diff is reported as more robust than C1 when its attack hit rate is at least 20 points lower
with a 95% bootstrap CI (over victims) excluding 0, for a given suffix length and surrogate; otherwise no robustness
advantage is claimed. The attack has no fluency penalty (a stronger attack on the embedding, and one that is easy to
flag by perplexity screening, as the paper itself notes), and the attacker does not know the C2 features.
