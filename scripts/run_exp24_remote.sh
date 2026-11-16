#!/bin/bash
# exp24 on the rented host beside exp25, then a combined completion marker.
cd /root/proj
export SEMCACHE_DEVICE=cpu TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
A=results/logs/all_status.log
echo "START exp24 $(date -u +%FT%TZ)" >> $A
/root/venv/bin/python -u -m semcache.experiments.exp24_sd_admit --workers ${WORKERS:-5} \
  --vectors data/emb_cache/exp24_stream_vectors.npz \
  --exp18 results/raw/revision/exp18_eviction_m2.json \
  --exp23 results/raw/revision/exp23_eviction_headroom_m2.json > results/logs/gpu_exp24.log 2>&1
echo "END exp24 rc=$? $(date -u +%FT%TZ)" >> $A
until grep -q BATCH_DONE results/logs/batch_status.log 2>/dev/null; do sleep 30; done
echo "exp25 done $(date -u +%FT%TZ)" >> $A
echo BATCH_DONE >> $A
