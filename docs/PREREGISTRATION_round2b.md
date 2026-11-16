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
