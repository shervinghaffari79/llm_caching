# SOLAR harness reproduction (authors' unmodified code, pinned commit 8d01e93) and ReuseGate in the same harness

LLM: Qwen3-4B (float16) served as `gpt-4o-mini`; the paper used GPT-4o-mini, so absolute F1 differs and the comparison is of method ordering and gaps. F1 = LoCoMo token F1 on the held-out test QA after the warm-up phase.

## LoCoMo, K = 10

| method | seeds | sessions x seeds | mean F1 +/- sd over seeds | gap to FIFO [95% CI], paired | relative |
|---|---|---|---|---|---|
| SOLAR | 2 | 12 | 0.235 +/- 0.064 | +0.018 [-0.006, +0.042] | +7.2% |
| SOLAR-A | 2 | 13 | 0.246 +/- 0.040 | +0.020 [+0.000, +0.039] | +8.2% |
| SOLAR-E | 2 | 13 | 0.243 +/- 0.037 | +0.016 [-0.010, +0.042] | +6.8% |
| ReuseGate (ours, label-free) | 2 | 18 | 0.264 +/- 0.007 | +0.011 [-0.007, +0.029] | +4.3% |
| FIFO | 2 | 19 | 0.252 +/- 0.007 | - | - |
| LRU | 2 | 17 | 0.250 +/- 0.011 | +0.001 [-0.005, +0.007] | +0.4% |
| ARC | 2 | 15 | 0.242 +/- 0.004 | -0.001 [-0.021, +0.019] | -0.4% |
| LFU | 2 | 17 | 0.236 +/- 0.009 | -0.013 [-0.024, -0.001] | -5.1% |

