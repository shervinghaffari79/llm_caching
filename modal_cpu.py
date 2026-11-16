"""
CPU-only Modal entry point for quality-only experiments that reuse recorded
embeddings (no encoder timing, so no GPU). It is a separate app from
modal_app.py so that it can run without a GPU payment method; it reuses the
same image and volumes.

    modal run --detach modal_cpu.py --module exp28_shift_lecar --args "--workers 30 --vectors data/emb_cache/exp24_stream_vectors.npz"
"""
from __future__ import annotations

import modal

PROJ = "/root/proj"

# Same image and volumes as modal_app.py (kept identical so the cached image is reused).
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
app = modal.App("semcache-revision-cpu", image=image)


@app.function(cpu=32.0, memory=65536, timeout=6 * 3600,
              volumes={f"{PROJ}/data": data_vol, f"{PROJ}/results": results_vol})
def run_experiment_cpu(module: str, args: list[str]) -> str:
    import os
    import subprocess
    import sys
    import time

    os.chdir(PROJ)
    env = {**os.environ, "SEMCACHE_DEVICE": "cpu", "OMP_NUM_THREADS": "1",
           "SEMCACHE_FAISS_THREADS": "1", "PYTHONWARNINGS": "ignore"}
    t0 = time.time()
    cmd = [sys.executable, "-u", "-m", f"semcache.experiments.{module}", *args]
    print("running:", " ".join(cmd), flush=True)
    import threading
    stop = threading.Event()

    def keep_committing():                      # persist checkpoints even if the run is cut short
        while not stop.wait(120):
            try:
                results_vol.commit()
            except Exception as exc:            # a failed commit is retried on the next tick
                print("commit failed:", exc, flush=True)

    threading.Thread(target=keep_committing, daemon=True).start()
    proc = subprocess.run(cmd, env=env)
    stop.set()
    data_vol.commit()
    results_vol.commit()
    msg = f"{module} exited {proc.returncode} after {(time.time() - t0) / 60:.1f} min on 32 CPUs"
    print(msg, flush=True)
    if proc.returncode:
        raise RuntimeError(msg)
    return msg


@app.local_entrypoint()
def main(module: str, args: str = ""):
    print(run_experiment_cpu.remote(module, args.split()))
