#!/usr/bin/env bash
set -euo pipefail
# Usage:
#   ./build_index.sh <CORPUS_DIR> <VOCAB_PATH> <INDEX_DIR> [--time]
# Example:
#   ./build_index.sh Data/Doc Data/vocab.txt out_index --time

CORPUS_DIR="${1:?}"
VOCAB_PATH="${2:?}"
INDEX_DIR="${3:?}"
TIME_FLAG="${4:-}"

mkdir -p "$INDEX_DIR"

RUN_CMD=(python3 build_index.py "$CORPUS_DIR" "$VOCAB_PATH" "$INDEX_DIR")

if [[ "${TIME_FLAG}" == "--time" ]]; then
  time "${RUN_CMD[@]}"
else
  "${RUN_CMD[@]}"
fi
