#!/bin/bash
# exp26 (RiskPriced) on the rented host, then a combined completion marker
# once the SearchQueries sweeps on the same host are also done.
cd /root/proj
export SEMCACHE_DEVICE=cpu TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
A=results/logs/all_status.log
echo "START exp26 $(date -u +%FT%TZ)" >> $A
/root/venv/bin/python -u -m semcache.experiments.exp26_risk_priced --workers ${WORKERS:-8} \
  --vectors data/emb_cache/exp24_stream_vectors.npz > results/logs/gpu_exp26.log 2>&1
echo "END exp26 rc=$? $(date -u +%FT%TZ)" >> $A
until grep -q BATCH_DONE results/logs/batch_status.log 2>/dev/null; do sleep 30; done
echo "sweeps done $(date -u +%FT%TZ)" >> $A
echo BATCH_DONE >> $A
