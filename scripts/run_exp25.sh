#!/bin/bash
# exp25 on a GPU host: fit the in-domain models once, then serve the test
# stream with each adapted configuration in parallel, then merge.
cd /root/proj
export SEMCACHE_DEVICE=cuda TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONWARNINGS=ignore
PY=/root/venv/bin/python
L=results/logs/batch_status.log
echo "START exp25_fit $(date -u +%FT%TZ)" >> $L
SEMCACHE_FAISS_THREADS=${FIT_THREADS:-12} $PY -u -m semcache.experiments.exp25_searchqueries_retrain --fit-only > results/logs/gpu_exp25_fit.log 2>&1
echo "END exp25_fit rc=$? $(date -u +%FT%TZ)" >> $L
for part in "C1 C5" "C2" "C4"; do
  t=$(echo $part | tr ' ' '-')
  SEMCACHE_FAISS_THREADS=${PART_THREADS:-3} OMP_NUM_THREADS=2 $PY -u -m semcache.experiments.exp25_searchqueries_retrain \
    --systems $part --tag part_$t > results/logs/gpu_exp25_$t.log 2>&1 &
done
wait
echo "END exp25_parts $(date -u +%FT%TZ)" >> $L
$PY -m semcache.experiments.exp25_searchqueries_retrain --merge $(ls results/raw/revision/exp25_searchqueries_retrain_*_part_*.json) --tag n50000 >> results/logs/gpu_exp25_merge.log 2>&1
echo "END exp25_merge rc=$? $(date -u +%FT%TZ)" >> $L
echo BATCH_DONE >> $L
