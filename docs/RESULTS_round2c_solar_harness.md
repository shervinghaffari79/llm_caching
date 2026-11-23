# SOLAR harness reproduction (authors' unmodified code, pinned commit 8d01e93) and ReuseGate in the same harness

LLM: Qwen3-4B (float16) served as `gpt-4o-mini`; the paper used GPT-4o-mini, so absolute F1 differs and the comparison is of method ordering and gaps. F1 = LoCoMo token F1 on the held-out test QA after the warm-up phase.

## LoCoMo, K = 10

| method | seeds | sessions x seeds | mean F1 +/- sd over seeds | gap to FIFO [95% CI], paired | relative |
|---|---|---|---|---|---|
| SOLAR | 2 | 19 | 0.278 +/- 0.002 | +0.026 [+0.008, +0.045] | +10.5% |
| SOLAR-A | 3 | 22 | 0.246 +/- 0.049 | +0.023 [+0.011, +0.035] | +9.5% |
| SOLAR-E | 3 | 21 | 0.244 +/- 0.047 | +0.021 [+0.004, +0.038] | +8.4% |
| ReuseGate (ours, label-free) | 3 | 24 | 0.250 +/- 0.021 | +0.015 [-0.002, +0.032] | +6.3% |
| FIFO | 3 | 24 | 0.228 +/- 0.042 | - | - |
| LRU | 3 | 24 | 0.229 +/- 0.046 | +0.001 [-0.004, +0.006] | +0.5% |
| ARC | 3 | 22 | 0.221 +/- 0.049 | -0.002 [-0.017, +0.013] | -0.7% |
| LFU | 3 | 22 | 0.224 +/- 0.031 | -0.006 [-0.018, +0.005] | -2.6% |

