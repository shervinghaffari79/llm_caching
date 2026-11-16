#!/bin/bash
# Waits for the long-stream answer fetch, then runs the long QQP stream and
# the full answer-substitutability audit (judge prompt v2).
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
L=results/logs/local_chain_status.log
while pgrep -f "ensure_pool" > /dev/null; do sleep 30; done
echo "START exp20 $(date -u +%FT%TZ)" >> $L
/usr/bin/python3 -u -m semcache.experiments.exp20_long_stream --n 10000 > results/logs/local_exp20_long_stream.log 2>&1
echo "END exp20 rc=$? $(date -u +%FT%TZ)" >> $L
echo "START exp16 $(date -u +%FT%TZ)" >> $L
/usr/bin/python3 -u -m semcache.experiments.exp16_answer_audit --per-stratum 150 --no-fetch > results/logs/local_exp16_audit.log 2>&1
echo "END exp16 rc=$? $(date -u +%FT%TZ)" >> $L
echo CHAIN_DONE >> $L
