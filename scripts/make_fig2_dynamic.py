"""
Figure 2 (dynamic cache behaviour of C2): mean over the five evaluation
streams at prefixes of 500-2,000 requests, with the min-max range across
streams shaded. Data: results/raw/revision/e2e_prefix_nocache_m2.json.

Run: python scripts/make_fig2_dynamic.py [OUT.png]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11})

d = json.load(open("results/raw/revision/e2e_prefix_nocache_m2.json"))
S = sorted(d)
N = [500, 1000, 1500, 2000]
arr = lambda m: np.array([[d[s]["prefix"][str(n)][m] for n in N] for s in S])

def line(ax, m, label, style, color, marker):
    a = arr(m)
    ax.plot(N, a.mean(0), style, color=color, marker=marker, label=label)
    ax.fill_between(N, a.min(0), a.max(0), color=color, alpha=0.15, linewidth=0)

fig, (l, r) = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=150)
line(l, "hit", "Overall served-hit ratio", "-", "#1f77b4", "o")
line(l, "correct_given_avail", "Correct hits / available equivalents", "--", "#ff7f0e", "s")
line(l, "hn_correct_miss", "Correct miss on near-duplicate distractors", "-", "#2ca02c", "^")
l.set_title("Cache behaviour as the stream advances")
l.set_ylim(0, 102)
line(r, "correct", "Correct-hit ratio", "-", "#1f77b4", "o")
c = arr("ceiling").mean(0)
r.plot(N, c, "--", color="black", label="Equivalent-entry availability (ceiling)")
r.set_title("Correct hits vs availability ceiling")
r.set_ylim(0, 100)
for ax in (l, r):
    ax.set_xlabel("Stream length (requests processed)")
    ax.set_ylabel("%")
    ax.grid(True, color="#e5e5e5")
    ax.legend(fontsize=11, loc="lower right" if ax is l else "upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "results/revision/figures/fig2_dynamic_hit_ratio.png"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
