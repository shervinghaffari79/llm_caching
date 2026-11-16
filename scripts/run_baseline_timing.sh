#!/bin/bash
# GPU timing for the published baselines (real waits, seed-42 stream), one
# process per system, plus an exp14 reference run to compare hosts.
cd /root/proj
export SEMCACHE_DEVICE=cuda TOKENIZERS_PARALLELISM=false PYTHONPATH=. PYTHONWARNINGS=ignore
PY=/root/venv/bin/python
L=results/logs/batch_status.log
echo "START exp14_reference $(date -u +%FT%TZ)" >> $L
$PY -u -m semcache.experiments.exp14_latency --tag host_check > results/logs/gpu_exp14_host_check.log 2>&1
echo "END exp14_reference rc=$? $(date -u +%FT%TZ)" >> $L
echo "START baselines $(date -u +%FT%TZ)" >> $L
for s in gptcache scalm vectorq contextcache; do
  OMP_NUM_THREADS=4 $PY -u -m semcache.experiments.exp21_baseline_timing --system $s --no-fetch \
    > results/logs/gpu_exp21_$s.log 2>&1 &
done
OMP_NUM_THREADS=4 $PY -u -m semcache.experiments.exp21_baseline_timing --system vcache --delta 0.02 --no-fetch \
  > results/logs/gpu_exp21_vcache.log 2>&1 &
wait
echo "END baselines $(date -u +%FT%TZ)" >> $L
echo BATCH_DONE >> $L
