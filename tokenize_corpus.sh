#!/bin/bash
# Usage: ./tokenize_corpus.sh <CORPUS_DIR> <VOCAB_DIR> [--time]
# Calls tokenize_corpus.py with spaCy-only tokenization.

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <CORPUS_DIR> <VOCAB_DIR> [--time]"
    exit 1
fi

CORPUS_PATH=$1
VOCAB_DIR=$2
TIME_ENABLED="${3:-}"

if [[ "$TIME_ENABLED" == "--time" ]]; then
    time python3 "$(dirname "$0")/tokenize_corpus.py" "$CORPUS_PATH" "$VOCAB_DIR"
else
    python3 "$(dirname "$0")/tokenize_corpus.py" "$CORPUS_PATH" "$VOCAB_DIR"
fi

# Example:
#   bash Task0/tokenize_corpus.sh Data/Doc temp --time 
