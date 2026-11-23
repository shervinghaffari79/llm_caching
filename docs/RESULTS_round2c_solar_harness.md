# SOLAR harness reproduction (authors' unmodified code, pinned commit 8d01e93) and ReuseGate in the same harness

LLM: Qwen3-4B (float16) served as `gpt-4o-mini`; the paper used GPT-4o-mini, so absolute F1 differs and the comparison is of method ordering and gaps. F1 = LoCoMo token F1 on the held-out test QA after the warm-up phase.

## LoCoMo, K = 10

| method | seeds | sessions x seeds | mean F1 +/- sd over seeds | gap to FIFO [95% CI], paired | relative |
|---|---|---|---|---|---|
| SOLAR | 3 | 27 | 0.277 +/- 0.005 | +0.031 [+0.016, +0.047] | +12.8% |
| SOLAR-A | 3 | 29 | 0.267 +/- 0.011 | +0.020 [+0.010, +0.030] | +8.2% |
| SOLAR-E | 3 | 29 | 0.267 +/- 0.008 | +0.020 [+0.006, +0.033] | +8.0% |
| ReuseGate (ours, label-free) | 3 | 30 | 0.261 +/- 0.003 | +0.012 [-0.001, +0.026] | +5.0% |
| FIFO | 3 | 30 | 0.249 +/- 0.008 | - | - |
| LRU | 3 | 30 | 0.249 +/- 0.010 | +0.001 [-0.003, +0.005] | +0.3% |
| ARC | 3 | 30 | 0.249 +/- 0.004 | +0.000 [-0.013, +0.014] | +0.1% |
| LFU | 3 | 30 | 0.244 +/- 0.004 | -0.005 [-0.015, +0.005] | -2.1% |

## LoCoMo, K = 50

| method | seeds | sessions x seeds | mean F1 +/- sd over seeds | gap to FIFO [95% CI], paired | relative |
|---|---|---|---|---|---|
| ReuseGate (ours, label-free) | 1 | 3 | 0.230 +/- n/a | -0.007 [-0.170, +0.156] | -3.0% |
| FIFO | 1 | 3 | 0.237 +/- n/a | - | - |
| LRU | 1 | 2 | 0.255 +/- n/a | - | - |
| LFU | 1 | 2 | 0.204 +/- n/a | - | - |

