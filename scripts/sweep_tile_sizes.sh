#!/usr/bin/env bash
# Rebuilds and runs the forward benchmark across a KEY_TILE_SIZE sweep to find the best config
set -euo pipefail

cd "$(dirname "$0")/.."

# QUERY_BLOCK_SIZE * head_dim(64) must stay <= 1024
KEY_TILE_SIZES=(4 8 16 32 64)
QUERY_BLOCK_SIZE=16

for kts in "${KEY_TILE_SIZES[@]}"; do
  echo "=== QUERY_BLOCK_SIZE=$QUERY_BLOCK_SIZE KEY_TILE_SIZE=$kts ==="
  NVCC_DEFINES="-DQUERY_BLOCK_SIZE=$QUERY_BLOCK_SIZE -DKEY_TILE_SIZE=$kts" bash scripts/build_benchmark.sh
  bash scripts/run_benchmark.sh
  echo
done
