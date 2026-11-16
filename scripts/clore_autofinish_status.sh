#!/bin/bash
# Runs on the local machine, independently of any interactive session:
# copies results back from the rented server every few minutes, and when the
# batch finishes (or the time limit is reached) does a final copy and cancels
# the order so billing stops.
#
#   nohup caffeinate -i scripts/clore_autofinish.sh ORDER_ID [MAX_HOURS] &
#
# Uses the ssh host alias clore-semcache (~/.ssh/config) and CLORE_API_KEY in .env.
ORDER=$1
MAX_HOURS=${2:-22}
HOST_ALIAS=${3:-clore-semcache}
STATUS_NAME=${4:-batch_status.log}
cd "$(dirname "$0")/.."
LOG=results/logs/autofinish_${HOST_ALIAS}.log
mkdir -p results/logs results/raw/revision results/logs/gpu_clore
start=$(date +%s)
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
sync_back() {
  rsync -az ${HOST_ALIAS}:/root/proj/results/raw/revision/ results/raw/revision/ &&
  rsync -az ${HOST_ALIAS}:/root/proj/results/logs/ results/logs/gpu_clore/
}
say "watching order $ORDER (limit ${MAX_HOURS}h)"
while true; do
  if sync_back; then
    status=$(ssh ${HOST_ALIAS} "cat /root/proj/results/logs/${STATUS_NAME}" 2>/dev/null)
    say "synced; last: $(echo "$status" | tail -1)"
    if echo "$status" | grep -q BATCH_DONE; then
      sync_back && say "final sync done"
      break
    fi
  else
    say "sync failed (server unreachable?)"
  fi
  if [ $(( $(date +%s) - start )) -ge $(( MAX_HOURS * 3600 )) ]; then
    sync_back; say "time limit reached"
    break
  fi
  sleep 300
done
for i in 1 2 3; do
  out=$(/usr/bin/python3 scripts/clore.py cancel "$ORDER" 2>&1)
  say "cancel: $out"
  echo "$out" | grep -qE '"code": 0|order_not_active' && break
  sleep 10
done
say "done"
