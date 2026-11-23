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
