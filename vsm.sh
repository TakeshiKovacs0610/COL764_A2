#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./vsm.sh <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <k> [--time]
# Writes TREC-style output to <OUTPUT_DIR>/vsm_docids.txt
# Uses query field: title only

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

RUN_CMD=(python3 "$(dirname "$0")/vsm.py" "$INDEX_DIR" "$QUERY_FILE" "$OUT_DIR" "$K")

if [[ "${TIME_FLAG}" == "--time" ]]; then
  time "${RUN_CMD[@]}"
else
  "${RUN_CMD[@]}"
fi

echo "Wrote: $OUT_DIR/vsm_docids.txt"

# bash Task3/vsm.sh temp/out_index Data/CORD19/queries.json temp/results 100 --time