#!/bin/bash
# Upload the minimal data set the revision experiments need to the active Modal workspace's
# semcache-data volume, from the local project copy (no download involved).
cd "$(dirname "$0")/.." || exit 1
V=semcache-data
put() { modal volume put -f "$V" "$1" "$2" > "results/logs/up_$(echo "$2" | tr '/' '_').log" 2>&1; }
put data/models models &
put data/workloads workloads &
put data/llm_cache llm_cache &
put data/splits_seed42.pkl splits_seed42.pkl &
put data/qqp_train.parquet qqp_train.parquet &
for f in exp24_stream_vectors exp30_qqp_vectors exp30_searchqueries_vectors; do put data/emb_cache/$f.npz emb_cache/$f.npz & done
for f in sembenchmark_searchqueries.parquet sembenchmark_lmarena.parquet lmarena_vectors.npz; do put data/external/$f external/$f & done
wait
echo ALL_UPLOADED
