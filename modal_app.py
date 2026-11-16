"""
Modal entry point for the GPU runs of the revision experiments.

The container mirrors the local layout (`/root/proj/semcache`, `/root/proj/data`,
`/root/proj/results`) so `semcache.config` resolves paths unchanged. Data and
results live on two Modal Volumes:

    semcache-data      QQP parquet, cached splits, embedding caches, the answer
                       pool, workloads, trained meta-classifiers
    semcache-results   results/raw/revision/*.json written by the experiments

One-time upload of the local data directory:

    modal volume put semcache-data data/ /

Run an experiment on the A10G (arguments after the module are passed through):

    SEMCACHE_GPU=L4 modal run modal_app.py --module exp14_latency
    modal run modal_app.py --module exp11_seeds --args "--seeds 42 43 --no-fetch"

Quality-only experiments that need no GPU run from modal_cpu.py.

Fetch results:

    modal volume get semcache-results raw/revision results/raw/revision_a10g

Package versions match the local environment; the only differences are the
CUDA build of torch and faiss-cpu (the code uses IndexIDMap2.remove_ids, which
GPU indexes do not support, and the index is small enough that CPU search is
not a bottleneck).
"""
from __future__ import annotations

import os

import modal

APP_NAME = "semcache-revision"
GPU = os.environ.get("SEMCACHE_GPU", "A10G")   # e.g. SEMCACHE_GPU=L4 modal run ...
PROJ = "/root/proj"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "numpy==1.26.4", "pandas==2.3.3", "scipy==1.13.1", "scikit-learn==1.6.1",
        "sentence-transformers==2.7.0", "transformers==4.46.3",
        "lightgbm==4.6.0", "xgboost==2.1.4", "faiss-cpu==1.8.0",
        "pyarrow==21.0.0", "datasets==4.5.0", "tiktoken==0.9.0",
        "python-Levenshtein==0.27.1", "matplotlib==3.9.4", "tqdm==4.67.1",
        "openai==2.48.0", "joblib==1.5.3",
    )
    .env({"HF_HOME": "/root/hf", "TOKENIZERS_PARALLELISM": "false",
          "SEMCACHE_DEVICE": "cuda", "PYTHONPATH": PROJ})
    .run_commands(
        "python -c \"from sentence_transformers import SentenceTransformer, CrossEncoder;"
        "[SentenceTransformer(m) for m in ('sentence-transformers/all-mpnet-base-v2',"
        "'sentence-transformers/all-distilroberta-v1','sentence-transformers/all-MiniLM-L6-v2')];"
        "CrossEncoder('cross-encoder/stsb-roberta-large')\"")
    .add_local_dir("semcache", remote_path=f"{PROJ}/semcache")
)

data_vol = modal.Volume.from_name("semcache-data", create_if_missing=True)
results_vol = modal.Volume.from_name("semcache-results", create_if_missing=True)
app = modal.App(APP_NAME, image=image)


@app.function(gpu=GPU, cpu=8.0, memory=32768, timeout=6 * 3600,
              volumes={f"{PROJ}/data": data_vol, f"{PROJ}/results": results_vol})
def run_experiment(module: str, args: list[str]) -> str:
    import os
    import subprocess
    import sys
    import time

    os.chdir(PROJ)
    t0 = time.time()
    cmd = [sys.executable, "-m", f"semcache.experiments.{module}", *args]
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd)
    # Persist new caches (embeddings, meta-classifiers) and results even when
    # the experiment failed part-way, so a rerun resumes from them.
    data_vol.commit()
    results_vol.commit()
    elapsed = time.time() - t0
    msg = f"{module} exited {proc.returncode} after {elapsed / 60:.1f} min on {GPU}"
    print(msg, flush=True)
    if proc.returncode:
        raise RuntimeError(msg)
    return msg


@app.function(gpu=GPU, timeout=600)
def gpu_info() -> str:
    import subprocess
    import torch
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    return f"{out.strip()} | torch {torch.__version__} cuda {torch.version.cuda} available={torch.cuda.is_available()}"


@app.local_entrypoint()
def main(module: str = "", args: str = "", info: bool = False):
    if info or not module:
        print(gpu_info.remote())
        if not module:
            return
    print(run_experiment.remote(module, args.split()))
