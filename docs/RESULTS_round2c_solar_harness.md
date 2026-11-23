# SOLAR harness reproduction (authors' unmodified code, pinned commit 8d01e93) and ReuseGate in the same harness

LLM: Qwen3-4B (float16) served as `gpt-4o-mini`; the paper used GPT-4o-mini, so absolute F1 differs and the comparison is of method ordering and gaps. F1 = LoCoMo token F1 on the held-out test QA after the warm-up phase.

## LoCoMo, K = 10

| method | seeds | sessions x seeds | mean F1 +/- sd over seeds | gap to FIFO [95% CI], paired | relative |
|---|---|---|---|---|---|
| SOLAR | 2 | 15 | 0.256 +/- 0.034 | +0.020 [-0.001, +0.041] | +8.1% |
| SOLAR-A | 2 | 17 | 0.270 +/- 0.006 | +0.021 [+0.005, +0.036] | +8.3% |
| SOLAR-E | 2 | 17 | 0.269 +/- 0.001 | +0.019 [-0.000, +0.038] | +7.6% |
| ReuseGate (ours, label-free) | 3 | 21 | 0.239 +/- 0.039 | +0.010 [-0.006, +0.026] | +4.0% |
| FIFO | 3 | 21 | 0.224 +/- 0.049 | - | - |
| LRU | 2 | 20 | 0.255 +/- 0.003 | +0.003 [-0.003, +0.008] | +1.0% |
| ARC | 2 | 19 | 0.248 +/- 0.005 | -0.004 [-0.021, +0.014] | -1.4% |
| LFU | 2 | 19 | 0.240 +/- 0.003 | -0.012 [-0.023, -0.000] | -4.6% |

