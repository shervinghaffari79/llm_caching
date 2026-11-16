#!/bin/bash
# Revision GPU batch (run on the rented server from /root/proj, inside screen).
# Phase 1: decision-path timing; Phase 2: end-to-end with real generation waits,
# alone on the host; Phase 3: five seeds with real waits, one process per seed.
cd /root/proj
export SEMCACHE_DEVICE=cuda TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONWARNINGS=ignore
PY=/root/venv/bin/python
L=results/logs/batch_status.log
run() { name=$1; log=$2; shift 2; t0=$(date +%s); echo "START $log $(date -u +%FT%TZ)" >> $L
  $PY -u -m semcache.experiments.$name "$@" > results/logs/gpu_$log.log 2>&1; rc=$?
  echo "END $log rc=$rc $(( $(date +%s)-t0 ))s" >> $L; }
run exp1_latency_breakdown exp1
run exp14_latency exp14
run exp13_topk_gate exp13 --no-fetch
run exp17_encoder_ann exp17 --no-fetch
run exp2_end_to_end exp2_real --no-prefetch --tag real
echo "START exp11_real_parallel $(date -u +%FT%TZ)" >> $L
for s in 42 43 44 45 46; do
  OMP_NUM_THREADS=4 $PY -u -m semcache.experiments.exp11_seeds --seeds $s --real-time \
    --no-fetch --tag real_s$s > results/logs/gpu_exp11_real_s$s.log 2>&1 &
done
wait
echo "END exp11_real_parallel $(date -u +%FT%TZ)" >> $L
$PY -m semcache.experiments.exp11_seeds --merge \
  $(ls results/raw/revision/exp11_seeds_*_real_s4[2-6].json) --tag real \
  >> results/logs/gpu_exp11_merge.log 2>&1
echo "END exp11_merge rc=$?" >> $L
echo BATCH_DONE >> $L
