# Pre-registration, round 2c: SOLAR in its own setting (written before any run described here)

Trigger: the reviewers'/co-author's request to benchmark ReuseGate against SOLAR (arXiv:2607.00394) and to compare in the
settings SOLAR reports, for a fair and reliable evaluation.

## What was found about SOLAR's code
The authors publish a reference implementation (github.com/ysunbp/SOLAR, Apache-2.0, pinned here to commit
8d01e9340eb3a9ab77761366e57a29c53dafefe7). Reading it showed that my first reimplementation (`semcache/policies_extra.py`,
used for the round 2c cache-stream tables) differs from the official one in: the regret signal (official: 1 - feedback score of the
agent's answer in the experience mode, or a novelty/density information gain in the conversation mode; mine: 1 - cosine of the
served hit), the adaptive threshold (official: tau = clip(sqrt(2 lambda / L_hat), 0.1, 5) with L_hat = EMA of 2 x std of the last
30 regrets, lambda = 5), the Beta prior (official optimistic (2, 1)), the novelty bonus in eviction (official: 0.3 x (1 - exp(-d/(2 dim)))
with d the mean squared L2 distance to the two nearest other entries), the beta decay (+0.05 every 5th retrieval), the SOLAR-A
eviction heuristic (official: 0.4 recency exp(-age/100) + 0.3 access frequency + 0.3 diversity) and the bootstrap rule (always store
while fewer than min(max(K/2, 5), 25) entries). The round 2c cache-stream tables are therefore labelled "first reimplementation" and
will be redone with a faithful port (Stage C) before they are used in a paper.

## Stage A: reproduce SOLAR's reported setting with the authors' unmodified code
* Code: the pinned commit, unmodified, run on a Colab runtime. Datasets: LoCoMo (10 sessions) and DialSim (3 shows) from
  THUIR/MemoryBench-Full (160 warm-up QA, ~39 scored test QA per session), exactly as their `run_experience_eval.py` does.
* Methods: their eight (embedder, fifo, lru, lfu, arc, solar_a, solar_e, solar). Capacities K in {10, 50} first; K in {20, 100} if time allows.
  Seeds 42, 1337, 2024. Their default retrieval depths and hyper-parameter configs, untouched.
* LLM: the paper used GPT-4o-mini. Here a self-hosted Qwen model served through vLLM under the served-model name `gpt-4o-mini`, so no
  configuration is edited. Absolute F1 will therefore differ from the paper's; what is compared is the ordering of the methods and the
  size of the gaps to FIFO, with their Tables 4 and 7 as the reference. The model is Qwen3-4B in float16 because the only GPU
  available when this was started was a 16 GB T4 (no bfloat16, too small for an 8B model in 16-bit).
* Reported: mean and standard deviation over seeds and sessions per method and K, SOLAR minus FIFO gaps (their Table 5), the
  ablation at K = 50 (their Table 6), and an explicit "reproduced / not reproduced" call per qualitative claim: (i) LRU, LFU and ARC do
  not beat FIFO at small K; (ii) SOLAR, SOLAR-A, SOLAR-E beat FIFO at K <= 50; (iii) FIFO overtakes at K = 100; (iv) admission-only
  helps more than eviction-only.

## Stage B: our admission policy in the same harness (label-free ReuseGate analogue)
Defined now, not tuned on any result: a new agent `reusegate` added as a new file (their code otherwise untouched, a patched copy kept
separate from the unmodified copy used in Stage A) that manages the experience memory with the semantic-demand rule of the
repository's `SemanticAdmissionPolicy` (score = hits + 1.0 x demand; demand = decayed count of earlier questions within cosine 0.6
of the entry's question; TinyLFU-style admission: a newcomer is stored only if its demand exceeds the lowest resident score; ties reject;
halving every 1,000 requests; history 4,000). Uses no feedback and no labels, only embeddings and the request history. Same
methods, K, seeds, sessions, metrics and decision rules as Stage A. A learned ReuseGate (LightGBM on warm-up data) is not attempted:
~160 warm-up examples per session are too few for a reliable model; this is stated as a limitation, not hidden.

## Stage C: faithful port of the official policies into the cache-stream experiments
After Stage A, the official admission/eviction logic is ported to the cache protocol of `semcache` (feedback = the correctness
of the served answer, i.e. the same information ReuseGate-R uses; and a label-free mode using the official novelty/density gain), and the
LmArena and search-log comparisons are rerun. Settings the paper fixes (lambda 5, adaptive window 30, thresholds 0.1 to 5, novelty weights
0.6 / 0.4, TS novelty bonus 0.3, optimistic prior, bootstrap) are used as given and not tuned.

## Decision rules
No claim is made that ReuseGate beats SOLAR in SOLAR's setting unless the Stage B gap to the best of SOLAR, SOLAR-A, SOLAR-E at the
same K has a 95% CI (over sessions x seeds) excluding 0. Negative and tied results are reported as found.

## Stage D (24 Nov 2026, written before any run): SOLAR's synthetic stress tests, to show where our policy is strong and where it is weak

Why: on LoCoMo (Stage A/B) the label-free ReuseGate is significantly below full SOLAR. A more useful comparison states the conditions under which each approach
wins. The SOLAR repository ships four LLM-free synthetic workloads (numpy embeddings, top-3 cosine retrieval), the same ones behind the paper's Tables 8, 9 and 11:
cycling (m topics in round robin, K = 10, m in {10, 15, 20, 30, 50}), topic drift (6 phases of 100 steps, K = 20, 30 topics, overlap 1), working-set sweep
(15 topics x 10 items, 500 queries, K in {5 ... 200}, uniform and Zipf topic choice) and retrieval noise in stream mode (50 topics x 100 items, K in {200, 500, 1000, 5000}).

* D1 (reproduction): the authors' code, unmodified, pinned commit 8d01e93, their seven methods plus embedder_unlimited, seeds 42, 1337, 2024. These need no LLM, so the paper's
  numbers should reproduce exactly (same code, same seeds); any difference is reported.
* D2 (our policies, same workloads and seeds, patched copy that only appends classes): `reusegate` = the label-free semantic-demand admission and eviction rule of
  Stage B (score = hits + 1.0 x demand; demand counts earlier query embeddings with cosine >= 0.5 to the item; TinyLFU admission: a newcomer enters only if its demand
  exceeds the lowest resident score, ties reject; halving every 1,000 queries; history 4,000) and `reusegate_e` (same scoring, always admits: an admission ablation). Parameters fixed now;
  the cosine radius 0.5 is a guess for this embedding space and a sensitivity run at 0.3 and 0.7 is reported as a labelled sensitivity, not as the result.
* Sanity check: the patched copy must reproduce D1 exactly for the authors' methods.
* Reported: hit rate and F1 per workload and parameter (mean +/- sd over 3 seeds), the working-set crossover K* between each policy and FIFO, the hit-rate recovery after each phase boundary in topic drift, and
  an explicit list of conditions where each approach wins or loses. The weaknesses we expect to find, stated in advance so they are not selected after the fact: our recency-free demand counts adapt slowly to topic drift
  (halving is every 1,000 queries and the drift runs are 600 steps), and a TinyLFU-style gate can refuse every newcomer when all residents already have high demand (round-robin cycling).
  No theoretical guarantee exists for ReuseGate (SOLAR proves a constant competitive ratio under stated assumptions and a sublinear regret bound); this is listed as a limitation.

### Stage D amendment (24 Nov 2026, written after the main D2 cycling/drift/working-set results were seen, before the sensitivity runs)

The main D2 result showed the weakness predicted above, far larger than expected: in topic drift ReuseGate's hit rate is 0.19 against 0.95 for FIFO. Three sensitivity runs of `reusegate` and `reusegate_e` only are added, all **labelled sensitivity, not the result**:
radius 0.3 (D3_r03), radius 0.7 (D3_r07) as pre-registered, and aging period 100 queries instead of 1,000 (D3_p100), which is **exploratory and chosen after seeing the failure**; it tests whether the drift failure is the aging constant or the admission design, and is not used to replace the pre-registered setting.

### Stage D2 (24 Nov 2026, written before any of these runs): three checks requested after the first Stage D results

**(1) Numeric comparison with the SOLAR paper's synthetic tables.** Tables 8 (cycling), 9 (working-set sweep, uniform), 10 (frozen-pool retrieval-noise probe) and 11 (stream-mode retrieval noise) of arXiv:2607.00394 were transcribed
(`scripts/compare_solar_paper_tables.py` holds the transcription with the table number next to each value). Our D1 values (authors' code unmodified, 3 seeds) are compared after rounding to the paper's precision; a cell "matches" if it agrees to the paper's last printed digit (+-0.01 for two-decimal tables, +-0.001 for three-decimal ones), otherwise the difference is listed.
Table 10 needs the authors' frozen-pool mode, which has not been run yet; it is run unmodified as D1-frozen. Nothing is excluded after the comparison.

**(2) ReuseGate radius chosen on separate reference traffic.** Radii {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9} are run for `reusegate` on **tuning seeds 101, 102, 103** (never used for evaluation) over four families: cycling (5 configurations), topic drift (6: plen 50/100/200/500 at overlap 1, plen 100 at overlap 0 and 2),
working set uniform (10 capacities) and working set Zipf (10 capacities). Rule (fixed now): for each radius the mean hit rate over seeds and configurations is computed inside each family; the four family means are averaged with equal weight; the radius with the highest value is the **global tuned radius** (ties: the one closest to 0.5).
Primary = this single global radius. Secondary, reported separately and labelled as optimistic: the best radius per family. `reusegate_e` uses the same radius as `reusegate`. This is in-distribution tuning (same generator, different seeds), which is the analogue of reference traffic in our cache streams and is optimistic for any real stream.
Expectation stated in advance: the working-set deficit shrinks; the topic-drift failure does not (0.19 against 0.95 for FIFO at the default radius).

**(3) More seeds.** Evaluation seeds 42, 1337, 2024 (as before) plus 7, 11, 13, 17, 19, 23, 29: 10 seeds. All ten methods at the default radius 0.5 (D4-default) and `reusegate`, `reusegate_e` at the tuned radius (D4-tuned), over cycling, topic drift (6 configurations) and working set (uniform and Zipf). The retrieval-noise stream is not repeated (3 seeds showed no separation and it is the slowest run).
Reported: mean hit rate, and for every pair (policy minus FIFO, policy minus SOLAR-A) the mean paired-by-seed gap with a 95% t-interval (10 seeds); a gap is called a difference only when the interval excludes 0. No multiple-comparison correction is applied to this descriptive table; it is stated as such.

### Stage D3 (24 Nov 2026, written before the run): the reference-constants SOLAR port on LmArena

Reason: a reviewer can ask why the port with the authors' constants (SOLARo-E, SOLARo-A, SOLARo) appears on the search log but only the first reimplementation on LmArena; the asymmetry looks selective. It was a compute limit, not a choice. The three SOLARo families are now run on all three LmArena rotations with the unchanged exp47 protocol
(`--only-families WTinyLFU,SOLARo-E,SOLARo-A,SOLARo --with-group --ref-max-only`, W-TinyLFU included because the experiment's primary tests are defined against it and it serves as a reproduction check; same layout, caps, tau grid, tuning rule, seed). ReuseGate and ReuseGate-R are recomputed in the same run and must reproduce the earlier numbers; the paired comparison ReuseGate minus SOLARo-A uses windows from this run. Nothing is filtered after the result; all three rotations are reported pooled (9 windows) and per rotation.
