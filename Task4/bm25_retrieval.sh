#!/usr/bin/env bash
set -euo pipefail
# Usage:
#   ./bm25_retrieval.sh <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <PATH_OF_STOPWORDS_FILE> <k> [--time]
# Writes TREC-style output to <OUTPUT_DIR>/bm25_docids.txt

if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
  echo "Usage: $0 <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <PATH_OF_STOPWORDS_FILE> <k> [--time]"
  exit 1
fi

INDEX_DIR="$1"
QUERY_FILE="$2"
OUT_DIR="$3"
STOPWORDS="$4"   # accepted but ignored by bm25_retrieval.py
TOPK="$5"
TIME_FLAG="${6:-}"

mkdir -p "$OUT_DIR"

RUN_CMD=(python3 "$(dirname "$0")/bm25_retrieval.py" "$INDEX_DIR" "$QUERY_FILE" "$OUT_DIR" "$STOPWORDS" "$TOPK")

if [[ "${TIME_FLAG}" == "--time" ]]; then
  time "${RUN_CMD[@]}"
else
  "${RUN_CMD[@]}"
fi

echo "Wrote: $OUT_DIR/bm25_docids.txt"

# bash Task4/bm25_retrieval.sh temp/out_index Data/CORD19/queries.json temp/output Data/stopwords.txt 100 --time