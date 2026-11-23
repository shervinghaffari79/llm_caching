# SOLAR harness reproduction (authors' unmodified code, pinned commit 8d01e93) and ReuseGate in the same harness

LLM: Qwen3-4B (float16) served as `gpt-4o-mini`; the paper used GPT-4o-mini, so absolute F1 differs and the comparison is of method ordering and gaps. F1 = LoCoMo token F1 on the held-out test QA after the warm-up phase.

## LoCoMo, K = 10

| method | seeds | sessions x seeds | mean F1 +/- sd over seeds | gap to FIFO [95% CI], paired | relative |
|---|---|---|---|---|---|
| SOLAR | 1 | 4 | 0.271 +/- n/a | +0.037 [-0.035, +0.110] | +16.0% |
| SOLAR-A | 1 | 7 | 0.258 +/- n/a | +0.008 [-0.023, +0.039] | +3.4% |
| SOLAR-E | 1 | 7 | 0.253 +/- n/a | +0.004 [-0.038, +0.045] | +1.4% |
| ReuseGate (ours, label-free) | 1 | 10 | 0.259 +/- n/a | +0.003 [-0.013, +0.019] | +1.0% |
| FIFO | 1 | 10 | 0.256 +/- n/a | - | - |
| LRU | 1 | 9 | 0.257 +/- n/a | +0.001 [-0.010, +0.012] | +0.5% |
| ARC | 1 | 8 | 0.249 +/- n/a | -0.008 [-0.033, +0.016] | -3.3% |
| LFU | 1 | 9 | 0.241 +/- n/a | -0.015 [-0.037, +0.007] | -5.8% |

