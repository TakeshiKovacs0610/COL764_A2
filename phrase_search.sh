#!/usr/bin/env bash
set -euo pipefail
# Usage:
#   ./phrase_search.sh <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> [--time]
# Writes TREC-style output to <OUTPUT_DIR>/phrase_search_docids.txt

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> [--time]"
  exit 1
fi

INDEX_DIR="$1"
QUERY_FILE="$2"
OUT_DIR="$3"
TIME_FLAG="${4:-}"

mkdir -p "$OUT_DIR"

RUN_CMD=(python3 "$(dirname "$0")/phrase_search.py" "$INDEX_DIR" "$QUERY_FILE" "$OUT_DIR")

if [[ "${TIME_FLAG}" == "--time" ]]; then
  time "${RUN_CMD[@]}"
else
  "${RUN_CMD[@]}"
fi

echo "Wrote: $OUT_DIR/phrase_search_docids.txt"

# bash Task2/phrase_search.sh temp1/out_index Data/CORD19/queries.json temp1/output --time