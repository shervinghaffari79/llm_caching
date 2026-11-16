#!/bin/bash
# SearchQueries operating-point sweeps on a rented host: 6 processes, each
# serving two points of the C2 (retrained) or C1 grid.
cd /root/proj
export SEMCACHE_DEVICE=cuda TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONWARNINGS=ignore
export SEMCACHE_FAISS_THREADS=4 OMP_NUM_THREADS=2
PY=/root/venv/bin/python
L=results/logs/batch_status.log
echo "START sweeps $(date -u +%FT%TZ)" >> $L
run() { tag=$1; shift; $PY -u -m semcache.experiments.exp25_searchqueries_retrain "$@" --tag $tag > results/logs/gpu_exp25_$tag.log 2>&1; echo "END $tag rc=$? $(date -u +%FT%TZ)" >> $L; }
run sweep_C2_a --systems C2 --meta-grid 0.02 0.05 &
run sweep_C2_b --systems C2 --meta-grid 0.1 0.15 &
run sweep_C2_c --systems C2 --meta-grid 0.2 0.3 &
run sweep_C1_a --systems none --bi-grid 0.92 0.94 &
run sweep_C1_b --systems none --bi-grid 0.95 0.96 &
run sweep_C1_c --systems none --bi-grid 0.97 0.98 &
wait
echo BATCH_DONE >> $L
