"""Stage D2: run one family of SOLAR's synthetic workloads at a given radius / seed list, or select the radius.
  python synth_tune_eval.py run <out_dir> <methods> <family> <seeds,csv> [radius]   (family: cycling|drift|ws_uniform|ws_zipf)
  python synth_tune_eval.py select <tune_root> <out.json>      (tune_root/r<radius>/<family>/... as written by 'run')
Run from the root of the patched SOLAR copy (needs synthetic_workloads on the path)."""
import json, re, sys
from pathlib import Path
import numpy as np

FAMILIES = ("cycling", "drift", "ws_uniform", "ws_zipf")
RADII = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def run(out, methods, family, seeds, radius):
    from synthetic_workloads import run_experiments as R
    from synthetic_workloads import runners
    if radius is not None:
        for name in ("reusegate", "reusegate_e"):
            cls = runners._RUNNER_REGISTRY[name]
            runners._RUNNER_REGISTRY[name] = (lambda c, r=radius: (lambda capacity, retrieve_k=3, seed=42: c(capacity, retrieve_k, seed, radius=r)))(cls)
    out = Path(out) / family
    if family == "cycling":
        R.run_cycling(out, seeds, methods)
    elif family == "drift":
        R.run_topic_drift(out, seeds, methods, phase_length_list=[50, 100, 200, 500])
        R.run_topic_drift(out / "ov0", seeds, methods, overlap=0)
        R.run_topic_drift(out / "ov2", seeds, methods, overlap=2)
    elif family == "ws_uniform":
        R.run_working_set(out, seeds, methods)
    elif family == "ws_zipf":
        R.run_working_set(out, seeds, methods, topic_distribution="zipf")


def family_mean(d, method="reusegate"):
    v = []
    for f in Path(d).rglob(f"{method}__*__seed*.json"):
        v.append(json.load(open(f))["avg_hit_rate"])
    return float(np.mean(v)) if v else float("nan"), len(v)


def select(root, outjson):
    res = {}
    for r in RADII:
        d = Path(root) / f"r{r}"
        res[r] = {fam: family_mean(d / fam) for fam in FAMILIES}
    score = {r: float(np.mean([res[r][f][0] for f in FAMILIES])) for r in RADII}
    best = max(RADII, key=lambda r: (round(score[r], 12), -abs(r - 0.5)))
    perfam = {f: max(RADII, key=lambda r: (round(res[r][f][0], 12), -abs(r - 0.5))) for f in FAMILIES}
    json.dump({"family_means": {str(r): {f: res[r][f][0] for f in FAMILIES} for r in RADII}, "counts": {str(r): {f: res[r][f][1] for f in FAMILIES} for r in RADII},
               "score": {str(r): score[r] for r in RADII}, "global_radius": best, "per_family_radius": perfam}, open(outjson, "w"), indent=1)
    print(json.dumps({"global_radius": best, "per_family_radius": perfam, "score": score}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        out, methods, family, seeds = sys.argv[2], sys.argv[3].split(","), sys.argv[4], [int(x) for x in sys.argv[5].split(",")]
        run(out, methods, family, seeds, float(sys.argv[6]) if len(sys.argv) > 6 else None)
    else:
        select(sys.argv[2], sys.argv[3])
