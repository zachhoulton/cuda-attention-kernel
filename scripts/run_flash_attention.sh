#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x build/flash_attention ]]; then
  echo "Binary not found. Build it first with scripts/build_flash_attention.sh" >&2
  exit 1
fi

./build/flash_attention
