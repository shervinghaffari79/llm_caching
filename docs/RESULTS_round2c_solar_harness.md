# SOLAR harness reproduction (authors' unmodified code, pinned commit 8d01e93) and ReuseGate in the same harness

LLM: Qwen3-4B (float16) served as `gpt-4o-mini`; the paper used GPT-4o-mini, so absolute F1 differs and the comparison is of method ordering and gaps. F1 = LoCoMo token F1 on the held-out test QA after the warm-up phase.

## LoCoMo, K = 10

| method | seeds | sessions x seeds | mean F1 +/- sd over seeds | gap to FIFO [95% CI], paired | relative |
|---|---|---|---|---|---|
| SOLAR | 1 | 9 | 0.276 +/- n/a | +0.020 [-0.011, +0.050] | +7.6% |
| SOLAR-A | 2 | 11 | 0.233 +/- 0.059 | +0.017 [-0.006, +0.039] | +6.7% |
| SOLAR-E | 2 | 10 | 0.240 +/- 0.029 | +0.007 [-0.020, +0.034] | +2.7% |
| ReuseGate (ours, label-free) | 2 | 14 | 0.247 +/- 0.017 | +0.013 [-0.003, +0.030] | +5.6% |
| FIFO | 2 | 14 | 0.226 +/- 0.043 | - | - |
| LRU | 2 | 14 | 0.228 +/- 0.041 | +0.002 [-0.005, +0.009] | +0.8% |
| ARC | 2 | 12 | 0.212 +/- 0.047 | -0.013 [-0.030, +0.004] | -5.3% |
| LFU | 2 | 13 | 0.210 +/- 0.046 | -0.014 [-0.029, +0.001] | -5.7% |

