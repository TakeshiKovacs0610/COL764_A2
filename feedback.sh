#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./feedback.sh <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <k> [--time]
# Writes TREC-style output to <OUTPUT_DIR>/feedback_docids.txt
# Also reports precision, recall, F1 based on qrels file (handled inside feedback_retrieval.py)

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo "Usage: $0 <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <k> [--time]"
  exit 1
fi

INDEX_DIR="$1"
QUERY_FILE="$2"
OUT_DIR="$3"
K="$4"
TIME_FLAG="${5:-}"

mkdir -p "$OUT_DIR"

RUN_CMD=(python3 "$(dirname "$0")/feedback_retrieval.py" "$INDEX_DIR" "$QUERY_FILE" "$OUT_DIR" "$K")

if [[ "${TIME_FLAG}" == "--time" ]]; then
  time "${RUN_CMD[@]}"
else
  "${RUN_CMD[@]}"
fi

echo "Wrote: $OUT_DIR/feedback_docids.txt"


# bash feedback.sh temp/out_index Data/CORD19/queries.json temp/results 100 --time