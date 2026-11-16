#!/bin/bash
# Quality-only runs for the 2026 baselines (decisions do not depend on hardware).
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
PY=/usr/bin/python3
L=results/logs/baselines_local_status.log
run() { name=$1; shift; t0=$(date +%s); echo "START $name $(date -u +%FT%TZ)" >> $L
  $PY -u -m semcache.experiments.$name "$@" > results/logs/local_$name.log 2>&1
  echo "END $name rc=$? $(( $(date +%s)-t0 ))s" >> $L; }
run exp19_baselines_seeds --no-fetch --c-results results/raw/revision/exp11_seeds_geforcertx4090.json
run exp12_threshold_risk --no-fetch
run exp18_eviction --no-fetch
echo LOCAL_DONE >> $L
