# Handoff for the cloud session (Colab GPU + Modal CPU)

Paper: "Cheap Features, Expensive Encoders: Rethinking Semantic Cache Decisions for LLM Systems" (Ghaffari, Ghatee), Expert Systems with Applications, second revision round, reviewers R3, R4, R5. Read this file first, then `results/revision/COMMENT_MAP.md` for the round-1 history.

Everything below was written on 2026-11-16 from the local session. Where a claim is not verified, it says so.

---

## 0. Standing rules (the user's own words, keep them)

1. **Leakage.** "In your writings, implementation, and analysis always stick to prevent data leakage as reviewers became sensitive about them. When it comes to writing be transparent on how did we overcome data leakage and why our analysis are reproducible rather than a synthetic evaluation. Don't limit the data on which we run examination to lowering the risk of criticism that measure generalization."
   - Use all the data, with k-fold or rotation. Tune causally (each tuning window scored by a model fitted on earlier windows only). Never tune on evaluation rows.
   - Disclose every overlap in the text. Do not hide a weakness; write it once, plainly.
2. **Order of work.** Easy items first, cheap experiments second, methodological novelty last and only after the user decides.
3. **Scope.** Add only what a reviewer asked for. No unrequested analyses in the manuscript or the letter.
4. **Manuscript voice.** First-draft voice. No revision leakage ("revised protocol", "post hoc", "as requested"). Terse, declarative, continuous academic prose. No contrastive "rather than / instead of" framing. Drop-in LaTeX with an exact anchor, then one or two sentences on how it addresses the comment. Response-letter text may say what changed; it must never contradict the manuscript.
5. **Report faithfully.** If a test fails or a result is negative, say so with the numbers. Pre-register decision rules before running, and report whatever comes out.
6. **Secrets.** `.env` is git-ignored and is not in the repo. Never print or commit a key or token. The OpenAI key in `.env` on the user's Mac is expired; a new one must be supplied by the user.
7. **Approvals.** Pushing to Overleaf, editing the response letter for submission and anything outward-facing need the user's go-ahead each time.

---

## 1. What is where

| Item | Location | Notes |
|---|---|---|
| Library | `semcache/` | `cache.py` (eviction policies), `baselines.py` (published systems), `metaclf.py`, `pipeline.py`, `workload.py` |
| Experiments | `semcache/experiments/exp*.py` | exp37-43 are the round-2 runs (see section 3) |
| Scripts | `scripts/` | `build_reusegate_tables.py`, `build_log_curves_table.py`, `upload_data_to_modal.sh`, `make_fig2_dynamic.py` |
| Modal launcher | `modal_cpu.py` | CPU only, memory 64 GB |
| Saved results | `results/raw/revision/*.json` | force-added to git (the rest of `results/raw/` is git-ignored) |
| Tables and letter | `results/revision/` | `reusegate_tables.txt`, `build_letter.py`, `letter_template.tex`, `figures/figure1/figure_1.tex` |
| Round-2 manuscript | `manuscript_v3_snapshot/` (plain copy; the live repo is the Overleaf project `6af802cb66228c8deecde85b`) | edited locally, **not pushed to Overleaf, not compiled** (the user's local TeX lacks `multirow`) |
| Data | **not in git** (2.3 GB) | see section 6 |

Key round-2 result files in `results/raw/revision/`:
`exp37_causal_reusegate_cpu_{causaltune,rot1,rot2,topicfold0,topicfold1,topicfold2,topic_causaltune}.json`,
`exp40_chronological_transfer_cpu_log_s{0,50000,100000}.json`, `exp38_lmarena_transfer_cpu_full.json`,
`exp41_se_candidates.json`, `exp41_se_gaming.json`, `exp43_fold{0,1,2}_split.json`, `exp16_collect_seed{43,45,46}.json`.

---

## 2. Comment status

| Comment | Status | Open piece |
|---|---|---|
| R4.1 temporal leakage | Complete | none |
| R4.2 QQP train/stream overlap | Complete | the chance overlap between QQP tuning/reference streams and evaluation streams is disclosed, not fixed (optional, section 4.E) |
| R4.3 figure fonts | Complete | needs a compiled PDF to confirm |
| R5.1 | Complete (closed by R4.1 evidence) | none |
| R3.2 one log / cold start / topic shift | Mostly | no timestamped second log exists; stated as a limitation |
| **R5.4 baselines are reimplementations** | **Text only; fidelity unchecked** | **section 4.A (this is the priority)** |
| R3.4 / R5.3 judge reliability | Needs runs | section 4.B and 4.C |
| R3.3 precision drop on the real log | Evidence complete | remedy needs a method decision |
| R3.1 / R5.2 stacking vs calibrated bi-encoder | Needs rethinking | do not start before the user decides |

---

## 3. What was found in round 2 (so nothing is re-derived)

- The SemBenchmarkSearchQueries log has no timestamps and arbitrary row order. "Chronological" is answered with past-only layouts, causal tuning, three block rotations (every row evaluated once) and a 3-fold topic-held-out split.
- ReuseGate (label-free learned admission) vs W-TinyLFU, pooled over rotations (n=15, Holm over five lambdas), reference 50k: +1.83 / +0.94 / +0.54 at lambda 0 / 1 / 2, +0.15 at 5 (n.s.), -0.26 at 10. ReuseGate-R: +1.88 / +1.33 / +0.96 / +0.51 / +0.66. Cold start: +0.70 / +0.30 / +0.22, no gain at lambda 5. Topic 3-fold: +0.89 / +0.73 / +0.52.
- The two-half topic split shows ReuseGate vs LFU only +0.06 (n.s.).
- Rotation 1 (rows 0-49,999) is the only block outside ReuseGate's design data.
- Full 150k log, default thresholds per 50k block: C1 63.5-63.6% hits at 41.1-41.3% precision; C2 47.4-47.6% at 52.8-53.3%; C1 is 0.6-1.9 points more precise than C2 at matched coverage. LmArena (63,796 prompts): C1/C2 curves within -0.12..+0.47 points.
- No Stack Exchange site works for admission: duplicate links are not title paraphrases (gaming: median cosine 0.483, recall 1.6%, pair precision 24%).
- Infrastructure facts: Modal workspaces `shervinghaffari79` (disabled), `shervinclaude`, `shervinmusic2022` (current). Launch detached jobs with `nohup caffeinate -i modal run --detach ... & disown`, at most about 5 at once. Modal splits arguments on spaces, so pass the empty suffix as `--base-suffix none`.

---

## 4. Tasks, in order

### A. R5.4 baseline fidelity (priority; mostly CPU, no GPU needed)

Goal: remove the objection that the baselines are the authors' own re-implementations, by checking each against the original code wherever it exists, and by stating plainly where it does not.

**The reviewer's exact wording is not stored in the repo.** My understanding from the user's paste is that it concerns the baselines being re-implemented and how ReuseGate differs from existing admission methods. Ask the user to paste the R5.4 text into `docs/` if the wording matters for the response.

Findings from the code audit (local, verified by reading the files):

- `semcache/baselines.py` header already says these are re-implementations of published decision logic, not the authors' code.
- **The "W-TinyLFU" baseline is not Caffeine's W-TinyLFU.** It is `WindowSemanticDemandPolicy` with the radius set to 0.9999 (`exp32_window_admission.py`, kind `"WTinyLFU"`): exact counts with exponential ageing, an LRU window of a tunable fraction, and a main segment where a candidate is promoted only if its evidence is strictly larger. There is no 4-bit count-min sketch, no doorkeeper, no segmented-LRU main, no hill-climbing window. It is the main comparator for ReuseGate, so this is the highest-risk item. Either validate it against the reference or rename it in the text (for example "windowed TinyLFU-style admission with exact counts") and say what is simplified.
- `SphereLFUPolicy` follows the authors' repository per its docstring, but see the two discrepancies below.
- `LeCaRPolicy` is an adaptation (history trigger by cosine radius instead of key equality).

| Baseline | Original implementation | Verified to exist (2026-11-16) | Our code | Check to run |
|---|---|---|---|---|
| LRU, LFU | textbook; also `code/alg` in `sylab/LeCaR` | yes | `cache.py` `LRUPolicy`, `LFUPolicy` | exact-key trace, compare hit counts with `sylab/LeCaR` `run.py` |
| LeCaR | `github.com/sylab/LeCaR` (`code/alg`, run `python3 run.py <size> lecar <trace>`; Vietri et al., HotStorage 2018) | yes, Python, no licence file | `cache.py` `LeCaRPolicy` | exact-key limit (see below); the original is randomised, so compare over seeds |
| SphereLFU | `github.com/dbiton/SemanticCaching`, `src/caches/cache.py` class `SphereQueryLFU`; paper "Semantic Caching: From Exact Hits to Close Enough", CIKM '26, doi 10.1145/3799682.3840580 (arXiv 2603.03301) | yes, CC BY 4.0 | `cache.py` `SphereLFUPolicy` | run both on the same vectors and capacities, compare hit counts and eviction order |
| W-TinyLFU | Caffeine simulator, `github.com/ben-manes/caffeine` (Java; `simulator/.../policy/sketch/` and `FrequencySketch.java`; Apache-2.0). Exact file names for the plain W-TinyLFU policy must be confirmed in the repo; only `HillClimberWindowTinyLfuPolicy` and `FrequencySketch.java` were seen in search results | repo yes; file names not confirmed | `exp32_window_admission.py` kind `WTinyLFU` over `WindowSemanticDemandPolicy` | replay an exact-key trace through the Caffeine simulator and through ours; report both |
| GPTCache | `github.com/zilliztech/GPTCache` (MIT) | yes | `baselines.py` `GPTCacheBaseline` (fixed threshold) | install the library and compare accept/reject decisions with its similarity evaluator on identical candidate pairs |
| vCache | `github.com/vcache-project/vCache` (licence shown as "other"; read `LICENSE` before copying code) | yes | `baselines.py` `VCacheBaseline` | run the official algorithm on a stream with precomputed embeddings and compare served/explored decisions |
| SCALM | no public repository found (two GitHub searches returned nothing, which is not proof) | no | `SCALMBaseline` | look at the paper page and author pages; if there is no code, say so in the text |
| VectorQ | author's MSc thesis (TUM); no repository found | no | `VectorQBaseline` | same as SCALM |
| ContextCache | no public repository found | no | `ContextCacheBaseline` | same as SCALM |

Discrepancies to resolve first (found by comparing our docstrings with the SphereLFU README):

1. **kappa.** The README states the default is `kappa = 2 / D_thresh^2`. Ours is `2 ln(1/eps) / D^2` with `eps_at_threshold = 0.1`, which is about 4.6 / D^2. Read `SphereQueryLFU.update_counters` in the authors' repository and make ours identical, or justify the difference.
2. **Counter halving.** The README says counters halve "once accumulated hits reach `capacity`". Ours halves "after every `capacity` requests that touched the cache". Confirm which one the code does.
3. **Distance.** Their threshold is an L2 distance on unit vectors (0.5 / 0.7 / 0.9). Ours converts from a cosine threshold, `D = sqrt(2(1 - cos))`. Check the conversion once.
4. **Embedding.** Their pipeline uses all-MiniLM-L6-v2. Ours uses all-mpnet-base-v2. The fidelity test must use one set of vectors for both implementations.

How to test (no leakage risk, because it uses no labels and no tuning):

- **Exact-key limit.** Build a trace where every distinct request maps to an orthogonal one-hot vector, so a semantic cache reduces to an exact-key cache. Run our policy at radius 0.9999 and the original on the same trace and capacities. LRU, LFU and W-TinyLFU are deterministic; hit counts should match exactly (W-TinyLFU only if the sketch is replaced by exact counts, otherwise report the gap). LeCaR is random; compare mean hit rate over at least 20 seeds against the original's.
- **Semantic limit.** Feed our vectors to the authors' SphereLFU and ours. Hit counts should agree to within noise at several capacities and thresholds.
- Pre-register the tolerance before running (suggested: exact equality for deterministic policies; within one standard deviation across seeds for randomised ones). If a baseline fails, fix our implementation or replace it with the original, then rerun every comparison that used it, and report both numbers in the response letter. Do not tune a baseline on evaluation rows.
- Read arXiv 2608.20280 ("Which Eviction Policy Should an LLM Cache Use?"). It appeared in search results as a unified comparison of semantic eviction policies. It was **not read**. If its released code or rankings overlap with ours, cite it as independent corroboration or as a reason to test again.

Manuscript consequences (write only after the checks):

- State once, in the baselines paragraph, which baselines were run from the original code (if any), which were validated against it and how, and which have no public code. Name W-TinyLFU as it really is.
- If the SphereLFU `kappa` changes, rerun every table that includes SphereLFU (exp37/42/43 baselines, exp32/36).
- The SphereLFU bibliography entry (`biton2026spherelfu`) cites the arXiv version; the repository gives the CIKM '26 proceedings DOI. Update the entry if the paper is accepted there.

### B. Independent judge validation (R3.4 / R5.3) — the only item that benefits from the Colab GPU

State: collected, unjudged pairs for seeds 43, 45, 46 are in `results/raw/revision/exp16_collect_seed{43,45,46}.json`. Seed 44 failed (Modal workspace error) and must be collected again (section C). The OpenAI key is expired.

Plan:
1. Freeze the prompt first. The verbatim judge prompt is Appendix A of the manuscript (`manuscript_v3_snapshot/`). Do not change it after seeing any open-model verdicts.
2. Run a second judge from a different model family on the same collected pairs, on a Colab GPU (an instruct model of the 7-14B class fits a T4 in 4-bit; an L4 or A100 is faster). Use `python -m semcache.experiments.exp16_answer_audit --judge-collected <file>` as the template, replacing the API call with the local model. Fix the seed and decoding (temperature 0).
3. Report agreement (Cohen kappa with bootstrap CI) between the two judges, and between each judge and the dataset label, on the same pairs.
4. A human-labelled subset (about 100 pairs, blind to both judges and to the dataset label) would be stronger. `results/revision/exp16_blind_labelling_sheet.xlsx` is the earlier sheet. This needs the user or the co-author; do not fabricate labels.
5. Do not describe a second LLM as ground truth. Say what it measures.

### C. Seed 44 collection (CPU, Modal)

`python -m semcache.experiments.exp16_answer_audit --collect-only` for seed 44, same settings as 43/45/46. It makes no API calls. Then judge it with the same pipeline as B.

### D. ReuseGate-R in the topic folds (CPU, Modal)

ReuseGate-R (`--with-group`) was evaluated in the rotations only. Run `exp43_topic_kfold.py --with-group --causal-tuning` for folds 0-2 and add the column to the topic-fold table via `scripts/build_reusegate_tables.py`. Group labels are used only inside the training side of each fold.

### E. Optional: remove the QQP stream overlap

The tuning and reference streams share 300 of 3,457 and 443 of 5,758 distinct questions with evaluation streams 42-46 by chance. It is disclosed in the leakage-controls text. Regenerating the tuning and reference streams from a pool that excludes the evaluation questions would remove it; this changes the QQP columns of `tab:learned_admission` and needs a rerun of exp36/37 on QQP. Do it only if the user asks.

### F. Needs a decision from the user first (do not start)

- R3.1 / R5.2: on QQP and LmArena, C1 and C2 lie on one precision-coverage curve; on the real log C1 is 0.6-1.9 points more precise at matched coverage. More runs will not change that. The paper must reframe the contribution (a cost frontier plus cheap calibration is one option) or find a regime where stacking truly wins. A GPU matters only if a stronger baseline such as an in-domain fine-tuned encoder is added, and only if a reviewer asked for it.
- R3.3 remedy: audit-budgeted calibration (a small labelled audit sample recalibrates thresholds in-domain). Not built.

### G. Overleaf and letter (needs the user's approval each time)

- Push the round-2 manuscript to the Overleaf project (`6af802cb66228c8deecde85b`), then compile there. Local compilation does not work.
- Update `Response_to_Reviewers.tex` (built by `results/revision/build_letter.py` from `letter_template.tex`) after the manuscript is final.

---

## 5. Compute guide

| Task | Where | Rough cost |
|---|---|---|
| A fidelity checks | CPU, anywhere | minutes to a few hours; Caffeine needs a JDK and Gradle |
| B open-model judging | Colab GPU | about 5k-20k pairs; a few hours on a T4 in 4-bit |
| C seed 44 collection | Modal CPU | one job |
| D topic-fold ReuseGate-R | Modal CPU | about 3 jobs, 5 hours each on 32 vCPU / 64 GB (earlier estimate, about $2.5 per job-hour) |

Colab: `uv tool install google-colab-cli`; the user must paste the Google authorisation code themselves. Modal: use the active workspace (check with `modal profile current`), and repopulate the `semcache-data` volume with `scripts/upload_data_to_modal.sh` if it is empty.

---

## 6. Data (not in git)

The repository has no data. The experiments need these files under `data/`:

- `data/external/sembenchmark_searchqueries.parquet` (150,000 rows; HuggingFace, vCache's SemBenchmarkSearchQueries)
- `data/external/sembenchmark_lmarena.parquet` and `data/external/lmarena_vectors.npz` (SemBenchmarkLmArena, 63,796 prompts)
- `data/emb_cache/exp24_stream_vectors.npz`, `exp30_qqp_vectors.npz`, `exp30_searchqueries_vectors.npz`
- `data/qqp_train.parquet`, `data/splits_seed42.pkl`, `data/workloads/`, `data/llm_cache/`, `data/models/` (trained meta-classifiers)

The user's Mac has all of it (2.3 GB). The `semcache-data` Modal volume in the current workspace holds the minimum set. Do not download anything onto the user's Mac. In the cloud, re-encode with all-mpnet-base-v2 on the Colab GPU if the vector files are missing. Check that every query has a stored vector before any run, because exp37-43 raise an error when one is missing.

---

## 7. Cautions

- The leakage rules apply to the fidelity tests too: a baseline's hyperparameters may be tuned only on tuning windows, with the same budget for every policy.
- LightGBM models retrained after a container preemption give slightly different AUC (third decimal). Use `deterministic=True` and save the model, and note any restart in the log.
- All reported p-values use Holm correction across the five lambdas; keep that family fixed.
- The earlier `eswa-revision-project` notes (round 1) are in the user's local memory, not in this repository. The essentials are above.
