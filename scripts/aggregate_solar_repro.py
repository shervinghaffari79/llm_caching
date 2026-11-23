"""Aggregate the SOLAR-harness runs (authors' unmodified code + our reusegate method). Usage:
python scripts/aggregate_solar_repro.py out.md <dir_with_out_subfolder> [...]
Reads out/K{K}_s{seed}/{dataset}/{method}/experience_eval_results.json. Reports, per K: mean +/- sd over seeds of the session-mean F1
(as in the paper's Table 4), the paired gap to FIFO over (session, seed) pairs with a t-interval, and the number of sessions x seeds."""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ORDER = ["embedder_message", "solar", "solar_a", "solar_e", "reusegate", "fifo", "lru", "arc", "lfu"]
NAMES = {"embedder_message": "Embedder (unlimited)", "solar": "SOLAR", "solar_a": "SOLAR-A", "solar_e": "SOLAR-E",
         "reusegate": "ReuseGate (ours, label-free)", "fifo": "FIFO", "lru": "LRU", "arc": "ARC", "lfu": "LFU"}
rows = defaultdict(dict)       # (K, method) -> {(dataset, seed): f1}
for root in sys.argv[2:]:
    for f in glob.glob(os.path.join(root, "out", "K*_s*", "*", "*", "experience_eval_results.json")):
        m = re.search(r"K(\d+)_s(\d+)/([^/]+)/([^/]+)/", f)
        K, seed, ds, meth = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
        rows[(K, meth)][(ds, seed)] = json.load(open(f))["avg_f1"]
L = ["# SOLAR harness reproduction (authors' unmodified code, pinned commit 8d01e93) and ReuseGate in the same harness", "",
     "LLM: Qwen3-4B (float16) served as `gpt-4o-mini`; the paper used GPT-4o-mini, so absolute F1 differs and the comparison is of method ordering and gaps. "
     "F1 = LoCoMo token F1 on the held-out test QA after the warm-up phase.", ""]
for K in sorted({k for k, _ in rows}):
    fifo = rows.get((K, "fifo"), {})
    L += [f"## LoCoMo, K = {K}", "", "| method | seeds | sessions x seeds | mean F1 +/- sd over seeds | gap to FIFO [95% CI], paired | relative |", "|---|---|---|---|---|---|"]
    for meth in ORDER:
        v = rows.get((K, meth))
        if not v:
            continue
        seeds = sorted({s for _, s in v})
        per_seed = [np.mean([x for (d, s2), x in v.items() if s2 == s]) for s in seeds]
        common = sorted(set(v) & set(fifo))
        gap = "-"
        rel = "-"
        if meth != "fifo" and len(common) >= 3:
            d = np.array([v[k] - fifo[k] for k in common])
            se = d.std(ddof=1) / np.sqrt(len(d))
            h = stats.t.ppf(0.975, len(d) - 1) * se
            gap = f"{d.mean():+.3f} [{d.mean() - h:+.3f}, {d.mean() + h:+.3f}]"
            rel = f"{100 * d.mean() / np.mean([fifo[k] for k in common]):+.1f}%"
        sd = f"{np.std(per_seed, ddof=1):.3f}" if len(per_seed) > 1 else "n/a"
        L.append(f"| {NAMES[meth]} | {len(seeds)} | {len(v)} | {np.mean(per_seed):.3f} +/- {sd} | {gap} | {rel} |")
    L.append("")
open(sys.argv[1], "w").write("\n".join(L) + "\n")
print("\n".join(L))
