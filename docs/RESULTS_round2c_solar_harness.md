# SOLAR harness reproduction (authors' unmodified code, pinned commit 8d01e93) and ReuseGate in the same harness

LLM: Qwen3-4B (float16) served as `gpt-4o-mini`; the paper used GPT-4o-mini, so absolute F1 differs and the comparison is of method ordering and gaps. F1 = LoCoMo token F1 on the held-out test QA after the warm-up phase.

## LoCoMo, K = 10

| method | seeds | sessions x seeds | mean F1 +/- sd over seeds | gap to FIFO [95% CI], paired | relative |
|---|---|---|---|---|---|
| SOLAR | 1 | 4 | 0.271 +/- n/a | +0.037 [-0.035, +0.110] | +16.0% |
| SOLAR-A | 1 | 7 | 0.258 +/- n/a | +0.008 [-0.023, +0.039] | +3.4% |
| SOLAR-E | 1 | 4 | 0.257 +/- n/a | +0.023 [-0.062, +0.108] | +9.7% |
| ReuseGate (ours, label-free) | 1 | 8 | 0.258 +/- n/a | +0.001 [-0.018, +0.020] | +0.3% |
| FIFO | 1 | 9 | 0.256 +/- n/a | - | - |
| LRU | 1 | 9 | 0.257 +/- n/a | +0.001 [-0.010, +0.012] | +0.5% |
| ARC | 1 | 7 | 0.242 +/- n/a | -0.008 [-0.038, +0.022] | -3.2% |
| LFU | 1 | 8 | 0.246 +/- n/a | -0.011 [-0.033, +0.012] | -4.2% |

