#!/bin/bash
# Overlapping operating-point sweeps on SearchQueries: retrained C2 at low
# probability thresholds, C1 at high cosine thresholds.
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false SEMCACHE_FAISS_THREADS=1
L=results/logs/local_exp25_sweeps_status.log
PY=/usr/bin/python3
echo "START fit $(date -u +%FT%TZ)" >> $L
$PY -u -m semcache.experiments.exp25_searchqueries_retrain --systems C2 --fit-only > results/logs/local_exp25_sweep_fit.log 2>&1
echo "END fit rc=$? $(date -u +%FT%TZ)" >> $L
$PY -u -m semcache.experiments.exp25_searchqueries_retrain --systems C2 --meta-grid 0.02 0.05 0.1 0.15 0.2 0.3 \
  --tag sweep_C2 > results/logs/local_exp25_sweep_C2.log 2>&1 &
$PY -u -m semcache.experiments.exp25_searchqueries_retrain --systems none --bi-grid 0.92 0.94 0.95 0.96 0.97 0.98 \
  --tag sweep_C1 > results/logs/local_exp25_sweep_C1.log 2>&1 &
wait
echo "END sweeps $(date -u +%FT%TZ)" >> $L
