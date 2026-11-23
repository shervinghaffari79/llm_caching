# Response to the co-author's comments on SOLAR, MeritKV and SCION (round 2c)

Prepared 23 Nov 2026. Every number below is generated from files in `results/raw/revision/`; the full tables are in
`docs/RESULTS_round2c_solar_comparison.md` and `docs/RESULTS_round2c_extra_baselines.md`, the extended Table 31 rows are in
`results/revision/table31_extended.tex`. Pre-registration: `docs/PREREGISTRATION_round2c.md`.

## 0. Summary

| comment | what was done | outcome |
|---|---|---|
| SOLAR (Sun, Cao, Lam 2026) is missing from Table 31; compare ReuseGate with SOLAR-A in the same streams, with the same features and reference-traffic budget | SOLAR-E, SOLAR-A, SOLAR implemented twice (first from the paper, then with the constants of the authors' public code); run in our streams; ReuseGate added to the authors' own harness and run next to their code on LoCoMo | ReuseGate (label-free) beats all SOLAR variants in our cache streams at low and moderate false-hit prices; ReuseGate-R beats them at every price. In SOLAR's own agent-memory setting the label-free ReuseGate is **below** full SOLAR (significant) and not distinguishable from SOLAR-A and SOLAR-E |
| Do not present ReuseGate as unique; compare it with other admission policies, not only eviction | FIFO, ARC, S3-FIFO, SIEVE, GDSF, DynamicAdaptiveClimb, SOLAR-A (an admission policy) added next to LRU, LFU, SphereLFU, LeCaR, Semantic Decay, W-TinyLFU | done on two datasets (search log, LmArena) |
| MeritKV (KV-cache reuse): state explicitly why the answer-cache setting changes features and failure modes | text drafted below (section 3); based on the MeritKV abstract and code page, not on its full text | text only, no experiment |
| SCION: compare with AUTO, GDSF, S3-FIFO, SIEVE, DynamicAdaptiveClimb | GDSF, S3-FIFO, SIEVE, DynamicAdaptiveClimb run; AUTO and SCION not reproducible (learned selector trained offline on many object-cache traces); a best-in-hindsight bound over the implementable portfolio members reported instead | done, with the AUTO/SCION limitation stated |

## 1. Is ReuseGate better than SOLAR? The honest answer has two parts

### 1.1 In our semantic-cache streams (adapted SOLAR)

SOLAR was designed for an agent's experience memory with a graded feedback score (1 - F1). A response cache has no such score, so the regret
signal had to be adapted (1 for a miss, 1 - cosine for a hit; no labels). Two ports give the same picture, so the conclusion does not depend on
which one is used.

* Search log (5 evaluation windows), score = correct-hit % - lambda x false-hit %, ReuseGate minus SOLAR-A (reference constants):
  +2.14 at lambda 0, +1.06, +0.49, +0.17 at lambda 5 and -0.32 at lambda 10 (label-free; the intervals exclude 0 at lambda 0 to 5 and at lambda 10, the last in SOLAR-A's favour). ReuseGate-R: +2.15, +1.47, +0.83, +0.53, +0.55 for lambda 0, 1, 2, 5, 10.
* LmArena (9 windows, first port only): ReuseGate minus SOLAR-A = +3.29, +2.46, +1.64, -1.09, -3.52 for lambda 0, 1, 2, 5, 10 (all Holm-adjusted
  p < 0.001); ReuseGate-R: +3.63, +6.31, +9.53, +17.41, +20.71.
* SOLAR-A, SOLAR-E and SOLAR are within about 0.5 points of W-TinyLFU, ARC and SIEVE on these streams. Against the best-in-hindsight bound over
  GDSF, S3-FIFO, SIEVE, DynamicAdaptiveClimb and W-TinyLFU, label-free ReuseGate is ahead at lambda 0 to 2 and behind at lambda 10; ReuseGate-R is ahead at every lambda.

### 1.2 In SOLAR's own setting (LoCoMo, the authors' code unmodified, K = 10, 3 seeds x 10 sessions)

| method | F1 gap to FIFO [95% CI] |
|---|---|
| SOLAR | +0.029 [+0.014, +0.043] |
| SOLAR-A | +0.020 [+0.010, +0.030] |
| SOLAR-E | +0.019 [+0.005, +0.032] |
| ReuseGate (ours, label-free) | +0.012 [-0.001, +0.026] |
| LRU / ARC / LFU | +0.001 / +0.000 / -0.005 (none significant) |

* The SOLAR paper's central finding **reproduces at K = 10** even with Qwen3-4B in place of GPT-4o-mini: its three variants beat FIFO, and the classic policies do not.
  The effect is about half the paper's size (+0.029 against +0.053 for SOLAR).
* ReuseGate minus SOLAR = -0.016 [-0.030, -0.003]: **SOLAR is better**. ReuseGate minus SOLAR-A = -0.008 [-0.021, +0.005] and minus SOLAR-E = -0.006
  [-0.020, +0.008]: not distinguishable. We therefore do **not** claim that ReuseGate beats SOLAR in SOLAR's setting. A plausible reading is that SOLAR uses
  a per-answer feedback signal that ReuseGate does not use; this is stated as a hypothesis, not tested.
* Not obtained: K = 20, 50, 100, DialSim, and the larger Qwen3.8-27B model (no A100 could be allocated: Colab returned HTTP 503 capacity errors).

**Suggested sentence for the paper.** "SOLAR (Sun et al., 2026) targets an agent's experience memory with graded feedback. We reimplemented its three
variants, tested them on our streams with the regret signal adapted to a response cache, and additionally ran the authors' public code with ReuseGate
added on LoCoMo. In our streams ReuseGate-R outperforms SOLAR and SOLAR-A at every false-hit price and label-free ReuseGate at prices up to lambda = 2; in
SOLAR's own setting SOLAR is better than label-free ReuseGate at K = 10, so the two address different problems."

### 1.3 Stress tests from SOLAR's own repository (Stage D; `docs/RESULTS_round2c_solar_synthetic.md`)

Four LLM-free synthetic workloads shipped with SOLAR (authors' code unmodified; our policies added in a copy that reproduces their results exactly, 480 of 480 runs), 3 seeds.

* **Strength:** cyclic access over more items than the cache holds. ReuseGate scores 0.45 / 0.30 / 0.18 at m = 20 / 30 / 50 (K = 10), above SOLAR-A (0.40 / 0.27 / 0.16), SOLAR and LFU; FIFO and LRU fall to about 0.
* **Weakness 1, severe:** topic drift. ReuseGate 0.19 against FIFO 0.95; SOLAR-A 0.68 and SOLAR 0.62 are also below FIFO but far better than ReuseGate. An exploratory aging period of 100 queries only lifts it to 0.32.
* **Weakness 2:** working-set sweep with a cache smaller than the working set. ReuseGate is up to 0.16 to 0.23 below FIFO at the pre-registered radius 0.5, while SOLAR-A and SOLAR beat FIFO by up to 0.20 there. A radius of 0.7 (labelled sensitivity) removes most of the deficit under uniform popularity.
* SOLAR's own weakness: its gate stays closed for large caches (0.02 to 0.07 below FIFO). Retrieval-noise stream: no policy separates (differences below 0.01, seed sd about 0.02).
* Reproduction check: all 66 transcribed cells of the SOLAR paper's synthetic Tables 8, 9, 10 and 11 match our run of the authors' code.
* Follow-up with 10 seeds and a radius tuned on separate seeds (0.8): the small-cache deficit against FIFO disappears (no working-set configuration significantly below FIFO except Zipf K=75, -0.010), but SOLAR-A is still ahead at K = 10 to 20 (by 0.05 to 0.13), and the topic-drift failure is unchanged (0.70 to 0.78 below FIFO).

## 2. What to write about the SOLAR-A comparison conditions

"Same streams, same features, same reference-traffic budget": the cache-stream runs use identical windows, capacities, thresholds grid and caps for every policy.
SOLAR-A uses no learned features, so there is no feature set to match; its gate state starts empty on each window and its few free settings were tuned on the
same tuning windows as ReuseGate (first port) or fixed to the authors' constants (second port). ReuseGate's reference traffic (50,000 requests) is used by
ReuseGate and not by SOLAR-A, which learns online; this asymmetry favours ReuseGate and should be stated.

## 3. Draft text: answer-cache admission versus KV-cache reuse (MeritKV)

Source: the MeritKV abstract (TMLR 2026; Khemani, Leri, Mittal) and its public code page. Not read in full, so every statement about MeritKV below should be
checked against the paper before submission.

"KV-cache reuse (e.g., MeritKV; Khemani et al., 2026) reuses the attention key-value state of a shared prompt prefix to cut prefill time. It is lossless
when the prefix matches, so the question an admission controller answers is economic: whether the saved prefill time exceeds the cost of lookup and
cache manipulation and the expected waste; MeritKV executes a reuse only when this net utility is positive and corrects its waste estimate by feedback.
A semantic answer cache differs in three ways that change the decision problem. (i) What is reused: a stored final answer returned in place of a model call,
for a request that is only semantically close to an earlier one. (ii) The failure mode: a false hit returns a wrong answer, a correctness error priced by a
false-hit cost lambda, not merely wasted time or memory; our objective is correct hits minus lambda times false hits. (iii) The features: the evidence for
a reuse is the semantic similarity of the request to a stored question, the local density of earlier requests (semantic demand), and group labels of past
traffic, where a KV controller relies on prefix length, recompute time and observed waste. The two controllers are complementary: a semantic cache can sit
in front of a model that itself reuses KV state, and the admission questions they ask (is this answer safe and worth keeping versus is this prefix worth
caching) are different."

## 4. What was removed or not claimed

* AUTO and SCION as methods: not reproducible without their offline training traces. Replaced by a labelled bound, not by a claim of equivalence.
* LHD and W-TinyLFU-AV (other portfolio members): not implemented.
* SOLAR on TripClick: not run (the data needs a token); on LmArena the reference-constants port was not run.
* A learned ReuseGate in SOLAR's setting: not attempted (about 160 warm-up examples per session).

## 5. Needs the co-author

The attached document held the comments but not the edited manuscript, so the review of Mehdi's tracked text, completion of his "?" tables and deletion of
unfinished parts could not be done. The extended Table 31 rows (`results/revision/table31_extended.tex`) and the paragraph in section 3 are ready to paste once the
manuscript source is available.
