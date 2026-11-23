"""Run the authors' synthetic experiments (Stage D). Usage, from the repository root of a SOLAR checkout:
  python /path/run_synth.py <out_dir> <methods comma list> <experiments comma list> [radius]
The patched runners module is built by apply_synth_patch.py; with only the authors' methods this is identical to their driver."""
import sys
from pathlib import Path

from synthetic_workloads import run_experiments as R
from synthetic_workloads import runners

out = Path(sys.argv[1])
methods = sys.argv[2].split(",")
exps = sys.argv[3].split(",")
radius = float(sys.argv[4]) if len(sys.argv) > 4 else None
if radius is not None:
    for name in ("reusegate", "reusegate_e"):
        cls = runners._RUNNER_REGISTRY[name]
        runners._RUNNER_REGISTRY[name] = (lambda c, r=radius: (lambda capacity, retrieve_k=3, seed=42: c(capacity, retrieve_k, seed, radius=r)))(cls)
seeds = [42, 1337, 2024]
if "cycling" in exps:
    R.run_cycling(out, seeds, methods)
if "topic_drift" in exps:
    R.run_topic_drift(out, seeds, methods, phase_length_list=[50, 100, 200, 500])
    R.run_topic_drift(out / "ov0", seeds, methods, overlap=0)
    R.run_topic_drift(out / "ov2", seeds, methods, overlap=2)
if "working_set" in exps:
    R.run_working_set(out, seeds, methods)
    R.run_working_set(out / "zipf", seeds, methods, topic_distribution="zipf")
if "retrieval_noise" in exps:
    R.run_retrieval_noise(out, seeds, [m for m in methods if m != "embedder_unlimited"] + ["embedder_unlimited"], capacity_list=[200, 500, 1000, 5000], mode="stream")
