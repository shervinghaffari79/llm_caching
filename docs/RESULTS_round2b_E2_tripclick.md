# E2, TripClick (health search, first-click label proxy), strictly chronological

Layout (from the result manifest): reference rows 0 and 1-50,000; tuning 50,001-100,000; evaluation 100,001-150,000 in five 10,000-request windows; causal tuning. Score = correct-hit ratio minus lambda x false-hit rate (points). Delta = ReuseGate minus baseline, mean over the 5 evaluation windows with a t-interval; Holm over the family of tests in the result file. With n = 5 windows the smallest possible Wilcoxon p is 0.0625, so significance rests on the t-test. Labels are proxy labels (first click), not human-judged equivalence.

Pre-registered claim: ReuseGate minus W-TinyLFU is Holm-significant and positive at lambda 0 and 1.

## ReuseGate (label-free, 'match')

| lambda | comparison | windows | delta | 95% CI | t p | Holm p |
|---|---|---|---|---|---|---|
| 0 | Learned[match]-WTinyLFU | 5 | -0.21 | [-0.51, +0.09] | 0.120 | 0.120 |
| 1 | Learned[match]-WTinyLFU | 5 | -0.12 | [-0.18, -0.07] | 0.004 | 0.016 |
| 2 | Learned[match]-WTinyLFU | 5 | -0.12 | [-0.22, -0.01] | 0.035 | 0.071 |
| 1 | Learned[match]-LRU | 5 | -0.43 | [-0.57, -0.29] | 0.001 | 0.005 |
| 1 | Learned[match]-LFU | 5 | -0.33 | [-0.55, -0.11] | 0.014 | 0.041 |

## ReuseGate-R (uses earlier group labels)

| lambda | comparison | windows | delta | 95% CI | t p | Holm p |
|---|---|---|---|---|---|---|
| 0 | Learned[group]-WTinyLFU | 5 | -0.25 | [-0.43, -0.07] | 0.017 | 0.017 |
| 1 | Learned[group]-WTinyLFU | 5 | +2.82 | [+2.28, +3.37] | 0.000 | 0.000 |
| 2 | Learned[group]-WTinyLFU | 5 | +8.37 | [+7.21, +9.52] | 0.000 | 0.000 |
| 5 | Learned[group]-WTinyLFU | 5 | +25.56 | [+21.95, +29.17] | 0.000 | 0.000 |
| 10 | Learned[group]-WTinyLFU | 5 | +54.12 | [+46.30, +61.95] | 0.000 | 0.000 |

## Verdict

* Label-free ReuseGate: not better than W-TinyLFU at lambda 0 (tie) and slightly but significantly worse at lambda 1 (-0.12 points, Holm p = 0.016). Claim NOT confirmed on TripClick (decision flag in the result file: false).
* ReuseGate-R: slightly worse at lambda 0 (-0.25 points, Holm p = 0.017), clearly better from lambda 1 upward (+2.8 at lambda 1, +8.4 at lambda 2, +25.6 at lambda 5, +54.1 at lambda 10). Positive at lambda 1 but not at lambda 0, so the pre-registered 'both' condition is not met either; the benefit appears where false hits are priced.
* The advantage of the label-free variant over LRU and LFU holds at lambda 1 (Holm p 0.005 and 0.041) but not over W-TinyLFU.
