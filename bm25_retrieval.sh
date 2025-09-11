#!/usr/bin/env bash
set -euo pipefail
# Usage:
#   ./bm25_retrieval.sh <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <k> [--time]
# Writes TREC-style output to <OUTPUT_DIR>/bm25_docids.txt

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo "Usage: $0 <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <k> [--time]"
  exit 1
fi

INDEX_DIR="$1"
QUERY_FILE="$2"
OUT_DIR="$3"
TOPK="$4"
TIME_FLAG="${5:-}"

mkdir -p "$OUT_DIR"

RUN_CMD=(python3 "$(dirname "$0")/bm25_retrieval.py" "$INDEX_DIR" "$QUERY_FILE" "$OUT_DIR" "$TOPK")

if [[ "${TIME_FLAG}" == "--time" ]]; then
  time "${RUN_CMD[@]}"
else
  "${RUN_CMD[@]}"
fi

echo "Wrote: $OUT_DIR/bm25_docids.txt"

# bash Task4/bm25_retrieval.sh temp/out_index Data/CORD19/queries.json temp/output 100 --time