# Round 2b, E5 and E6 results (pre-registered in docs/PREREGISTRATION_round2b.md)

All numbers come from `exp49_slices_cpu.json` and `exp50_band_mcnemar_cpu.json`. Models are trained on QQP only; thresholds are set so each system accepts the same share (90% or 80%) of true duplicate pairs. FAR = false-acceptance rate on pairs that are no longer equivalent. Delta = system minus C1 (points); 95% bootstrap CI over source pairs. A slice 'differs' if |Delta| >= 5 points with the CI excluding 0 (no multiplicity correction was applied to the CIs below; where the CI is close to 0 treat the call as tentative).

## E5-A. QQP perturbation slices, matched recall 90%

| slice | source pairs | FAR C1 | FAR C2 | Delta C2 [95% CI] | FAR C2-diff | Delta C2-diff [95% CI] | verdict C2 / C2-diff |
|---|---|---|---|---|---|---|---|
| number | 699 | 70.0 | 59.1 | -10.9 [-22.1, -0.7] | 47.8 | -22.2 [-35.0, -11.0] | better / better |
| unit | 266 | 62.0 | 62.8 | +0.8 [-1.1, +6.6] | 61.3 | -0.8 [-3.4, +7.3] | tie / tie |
| entity | 6437 | 29.7 | 26.6 | -3.1 [-5.0, -1.8] | 26.4 | -3.3 [-5.3, -1.7] | tie / tie |
| time | 399 | 62.4 | 57.6 | -4.8 [-12.0, +0.5] | 47.4 | -15.0 [-31.4, -5.4] | tie / better |
| negation | 14307 | 78.2 | 78.3 | +0.1 [-1.9, +1.8] | 78.4 | +0.2 [-1.5, +2.5] | tie / tie |
| control (label-preserving; value = share still ACCEPTED, higher is better) | 14997 | 86.9 | 86.1 | -0.8 [-1.7, +0.0] | 86.3 | -0.6 [-1.4, +0.3] | - |

## E5-A. QQP perturbation slices, matched recall 80%

| slice | source pairs | FAR C1 | FAR C2 | Delta C2 [95% CI] | FAR C2-diff | Delta C2-diff [95% CI] | verdict C2 / C2-diff |
|---|---|---|---|---|---|---|---|
| number | 699 | 53.2 | 39.9 | -13.3 [-28.7, +1.2] | 28.3 | -24.9 [-38.5, -11.5] | tie / better |
| unit | 266 | 48.5 | 49.2 | +0.8 [-3.8, +7.5] | 46.6 | -1.9 [-5.9, +7.5] | tie / tie |
| entity | 6437 | 19.5 | 15.3 | -4.2 [-7.0, -2.2] | 15.2 | -4.2 [-7.1, -1.9] | tie / tie |
| time | 399 | 52.4 | 38.3 | -14.0 [-27.1, -4.5] | 31.6 | -20.8 [-40.3, -8.7] | better / better |
| negation | 14307 | 62.4 | 60.9 | -1.5 [-2.4, +0.4] | 61.1 | -1.3 [-2.2, +1.0] | tie / tie |
| control (label-preserving; value = share still ACCEPTED, higher is better) | 14997 | 74.8 | 74.3 | -0.6 [-1.4, +0.8] | 74.4 | -0.4 [-1.2, +1.0] | - |

## E5-B. HEROS (zero-shot, base = Synonym+Typo (meaning-preserving)), FAR of pairs labelled non-equivalent

| subset | R=90% C1 | C2 | C2-diff | R=80% C1 | C2 | C2-diff |
|---|---|---|---|---|---|---|
| Antonym | 82.1 | 82.8 | 81.3 | 65.0 | 65.7 | 64.5 |
| Negation | 93.7 | 96.3 | 94.8 | 84.7 | 88.4 | 87.7 |
| Random | 78.5 | 80.8 | 78.4 | 60.8 | 62.8 | 62.4 |
| Negation (antonym) | 95.9 | 97.4 | 96.1 | 90.9 | 90.7 | 90.8 |
| Synonym | 96.6 | 96.2 | 96.2 | 90.1 | 90.0 | 90.0 |
| Typo | 83.4 | 83.8 | 83.8 | 69.9 | 70.0 | 70.0 |

## E5-C. PAWS (8000 pairs, 44% paraphrase)

| system | AUC | FAR at 90% recall | FAR at 80% recall |
|---|---|---|---|
| C1 | 0.652 | 80.5 | 61.7 |
| C2 | 0.626 | 81.7 | 63.9 |
| C2-diff | 0.628 | 81.5 | 62.6 |

SemAntoNeg: not run (source not located); not claimed.

## E6-A. Band diagnostic

QQP meta set, 30000 pairs, 5-fold group-disjoint out-of-fold scores. Accept = threshold at matched 90% recall on the whole set.

| band | pairs | duplicate share | system | AUC | accept rate | precision of accepted |
|---|---|---|---|---|---|---|
| QQP default 0.726 | 3961 | 43% | C1 | 0.610 | 36% | 53.8% |
| QQP default 0.726 | 3961 | 43% | C2 | 0.751 | 40% | 65.1% |
| QQP default 0.726 | 3961 | 43% | C2-diff | 0.753 | 42% | 63.9% |
| search-log default 0.895 | 8242 | 80% | C1 | 0.579 | 100% | 79.6% |
| search-log default 0.895 | 8242 | 80% | C2 | 0.744 | 94% | 83.3% |
| search-log default 0.895 | 8242 | 80% | C2-diff | 0.743 | 93% | 83.4% |

## E6-B. Paired decisions at 90% matched recall (C1 vs system)

Total errors: {'C1': 5599, 'C2': 4547, 'C2-diff': 4621}.

| system | C1 errors fixed | errors added | net | exact McNemar p | false hits fixed / added | misses fixed / added |
|---|---|---|---|---|---|---|
| C2 | 1762 | 710 | +1052 | 2.4e-102 | 1335 / 283 | 427 / 427 |
| C2-diff | 1764 | 786 | +978 | 1.6e-85 | 1304 / 326 | 460 / 460 |

## E6-C. Error taxonomy, C2 vs C1

| detector category | pairs | C1 errors | fixed | added | net |
|---|---|---|---|---|---|
| number | 3045 | 569 | 143 | 139 | +4 |
| negation | 2033 | 320 | 83 | 52 | +31 |
| capitalised | 9657 | 1881 | 865 | 172 | +693 |
| question_type | 6205 | 1179 | 287 | 164 | +123 |
| length_ratio | 271 | 48 | 9 | 11 | -2 |
| other | 8789 | 1602 | 375 | 172 | +203 |

## E6-C. Error taxonomy, C2-diff vs C1

| detector category | pairs | C1 errors | fixed | added | net |
|---|---|---|---|---|---|
| number | 3045 | 569 | 177 | 193 | -16 |
| negation | 2033 | 320 | 80 | 53 | +27 |
| capitalised | 9657 | 1881 | 851 | 189 | +662 |
| question_type | 6205 | 1179 | 286 | 166 | +120 |
| length_ratio | 271 | 48 | 10 | 7 | +3 |
| other | 8789 | 1602 | 360 | 178 | +182 |
