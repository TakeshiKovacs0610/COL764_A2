#!/usr/bin/env bash
set -euo pipefail
# Usage:
#   ./phrase_search.sh <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <PATH_OF_STOPWORDS_FILE> [--time]
# Writes TREC-style output to <OUTPUT_DIR>/phrase_search_docids.txt

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo "Usage: $0 <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <PATH_OF_STOPWORDS_FILE> [--time]"
  exit 1
fi

INDEX_DIR="$1"
QUERY_FILE="$2"
OUT_DIR="$3"
STOPWORDS="$4"  # accepted but ignored per updated tokenizer
TIME_FLAG="${5:-}"

mkdir -p "$OUT_DIR"

RUN_CMD=(python3 "$(dirname "$0")/phrase_search.py" "$INDEX_DIR" "$QUERY_FILE" "$OUT_DIR" "$STOPWORDS")

if [[ "${TIME_FLAG}" == "--time" ]]; then
  time "${RUN_CMD[@]}"
else
  "${RUN_CMD[@]}"
fi

echo "Wrote: $OUT_DIR/phrase_search_docids.txt"

# bash Task2/phrase_search.sh temp1/out_index Data/CORD19/queries.json temp1/output Data/stopwords.txt --time