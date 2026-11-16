#!/bin/bash
# Multi-seed operating-curve sweeps (C1-C4, extended threshold grids), three
# seeds at a time on a 4-thread host.
cd /root/proj
export SEMCACHE_DEVICE=cuda TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
PY=/root/venv/bin/python
L=results/logs/batch_status.log
run() { s=$1; t0=$(date +%s); echo "START seed$s $(date -u +%FT%TZ)" >> $L
  $PY -u -m semcache.experiments.exp12_threshold_risk --seed $s --configs-only --no-fetch --tag ext \
    > results/logs/gpu_exp12_seed$s.log 2>&1
  echo "END seed$s rc=$? $(( $(date +%s)-t0 ))s" >> $L; }
run 42 & run 43 & run 44 &
wait
run 45 & run 46 &
wait
echo BATCH_DONE >> $L
