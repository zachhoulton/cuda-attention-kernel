#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x build/bench_flash_attention ]]; then
  echo "Binary not found. Build it first with scripts/build_benchmark.sh" >&2
  exit 1
fi

./build/bench_flash_attention
