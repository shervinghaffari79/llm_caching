#!/bin/bash
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false SEMCACHE_FAISS_THREADS=1
L=results/logs/local_exp25_status.log
echo "START fit $(date -u +%FT%TZ)" >> $L
/usr/bin/python3 -u -m semcache.experiments.exp25_searchqueries_retrain --fit-only > results/logs/local_exp25_fit.log 2>&1
echo "END fit rc=$? $(date -u +%FT%TZ)" >> $L
/usr/bin/python3 -u -m semcache.experiments.exp25_searchqueries_retrain --systems C1 C2 C4 C5 --tag n50000 > results/logs/local_exp25.log 2>&1
echo "END exp25 rc=$? $(date -u +%FT%TZ)" >> $L
