#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p build

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found. Load the CUDA toolchain first, e.g.:" >&2
  echo "  export PATH=/usr/local/cuda/bin:$PATH" >&2
  exit 1
fi

nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || true
nvcc --version | tail -n 1

nvcc \
  -std=c++17 \
  -O3 \
  -arch=sm_80 \
  -I. \
  csrc/flash_attention.cu \
  -o build/flash_attention

echo "Built CUDA binary: build/flash_attention"
