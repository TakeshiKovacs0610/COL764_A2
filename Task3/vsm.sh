#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./vsm.sh <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <STOPWORDS_FILE> <k>
#
# Example:
#   ./vsm.sh temp/out_index queries.json temp/results stopwords.txt 100

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <STOPWORDS_FILE> <k>"
    exit 1
fi

INDEX_DIR=$1
QUERY_FILE=$2
OUTPUT_DIR=$3
STOPWORDS_FILE=$4
K=$5

mkdir -p "$OUTPUT_DIR"
OUTFILE="$OUTPUT_DIR/docids.txt"

# Run VSM retrieval
python3 "$(dirname "$0")/vsm.py" "$INDEX_DIR" "$QUERY_FILE" "$OUTPUT_DIR" "$STOPWORDS_FILE" "$K" "$OUTFILE"
