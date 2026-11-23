# SOLAR harness reproduction (authors' unmodified code, pinned commit 8d01e93) and ReuseGate in the same harness

LLM: Qwen3-4B (float16) served as `gpt-4o-mini`; the paper used GPT-4o-mini, so absolute F1 differs and the comparison is of method ordering and gaps. F1 = LoCoMo token F1 on the held-out test QA after the warm-up phase.

## LoCoMo, K = 10

| method | seeds | sessions x seeds | mean F1 +/- sd over seeds | gap to FIFO [95% CI], paired | relative |
|---|---|---|---|---|---|
| SOLAR | 3 | 22 | 0.257 +/- 0.040 | +0.030 [+0.012, +0.048] | +12.4% |
| SOLAR-A | 3 | 24 | 0.253 +/- 0.036 | +0.023 [+0.012, +0.034] | +9.6% |
| SOLAR-E | 3 | 23 | 0.251 +/- 0.035 | +0.023 [+0.007, +0.038] | +9.4% |
| ReuseGate (ours, label-free) | 3 | 28 | 0.261 +/- 0.003 | +0.013 [-0.001, +0.028] | +5.4% |
| FIFO | 3 | 29 | 0.247 +/- 0.010 | - | - |
| LRU | 3 | 28 | 0.247 +/- 0.014 | +0.001 [-0.003, +0.005] | +0.4% |
| ARC | 3 | 26 | 0.249 +/- 0.004 | +0.003 [-0.013, +0.018] | +1.1% |
| LFU | 3 | 27 | 0.242 +/- 0.001 | -0.004 [-0.015, +0.006] | -1.8% |

