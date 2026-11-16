"""
Figure 9 (latency-accuracy trade-off) and the overhead columns of the ablation
tables, from the batch-size-1 stage timings of experiment 1 on the RTX 4090
server. F1 values are those of the held-out 3,000-pair ablation subset
(Table tab:ablation_incremental); only the cost axis is re-measured.

Run: python scripts/make_fig9_latency_accuracy.py [OUT.png]
"""
import glob
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sorted(glob.glob("results/raw/revision/exp1_latency_breakdown_geforcertx4090*.json"))[0]
st = {k: v["mean_ms"] for k, v in json.load(open(src))["online_stages_ms"].items()}
base = st["bi_encode"] + st["vector_search"] + st["feat_bi_cosine"] + st["meta_infer"]
cheap = st["feat_distance"] + st["feat_elementwise"] + st["feat_text"]
rows = {  # name: overhead (ms)
    "Bi-encoder + cross-encoder": base + st["cross_encode"],
    "  with distance metrics": base + st["cross_encode"] + st["feat_distance"],
    "  with element-wise operations": base + st["cross_encode"] + st["feat_elementwise"],
    "  with text features": base + st["cross_encode"] + st["feat_text"],
    "Full model": base + st["cross_encode"] + cheap,
    "Full model without cross-encoder": base + cheap,
    "LightGBM on cosine feature only": base,
}
for k, v in rows.items():
    print(f"{k:34s} {v:7.3f} ms")
print("added: elementwise %.3f distance %.3f text %.3f cross %.3f meta %.3f" % (
    st["feat_elementwise"], st["feat_distance"], st["feat_text"], st["cross_encode"], st["meta_infer"]))

points = [  # label, overhead, F1, colour (as in the original figure)
    ("Bi only", rows["LightGBM on cosine feature only"], 0.8591, "#4c72b0"),
    ("Cheap + LGBM", rows["Full model without cross-encoder"], 0.8754, "#55a868"),
    ("Bi + cross", rows["Bi-encoder + cross-encoder"], 0.8500, "#c44e52"),
    ("Full stacking", rows["Full model"], 0.8799, "#8172b2"),
]
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
for lab, x, y, c in points:
    ax.scatter(x, y, s=70, color=c, zorder=3)
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
ax.set_xlim(5, 25)
ax.set_xticks([5, 10, 15, 20, 25])
ax.set_ylim(0.84, 0.887)
ax.set_xlabel("Per-request overhead (ms, batch size 1)")
ax.set_ylabel("F1 on held-out QQP subset")
ax.set_title("Latency-accuracy trade-off")
ax.grid(True, color="#e5e5e5", linewidth=0.8, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "results/revision/figures/fig9_latency_accuracy_pareto.png"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
