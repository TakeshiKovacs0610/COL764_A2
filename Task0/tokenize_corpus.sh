#!/bin/bash
# Usage: ./tokenize_corpus.sh <CORPUS_DIR> <PATH_OF_STOPWORD_FILE> <VOCAB_DIR> [--time]
# Calls tokenize_corpus.py with spaCy-only tokenization (stopwords arg accepted but ignored).

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <CORPUS_DIR> <PATH_OF_STOPWORD_FILE> <VOCAB_DIR> [--time]"
    exit 1
fi

CORPUS_PATH=$1
STOPWORDS_FILE=$2
VOCAB_DIR=$3
TIME_ENABLED="${4:-}"

if [[ "$TIME_ENABLED" == "--time" ]]; then
    time python3 "$(dirname "$0")/tokenize_corpus.py" "$CORPUS_PATH" "$STOPWORDS_FILE" "$VOCAB_DIR"
else
    python3 "$(dirname "$0")/tokenize_corpus.py" "$CORPUS_PATH" "$STOPWORDS_FILE" "$VOCAB_DIR"
fi

# Example:
#   bash Task0/tokenize_corpus.sh Data/Doc Data/stopwords.txt temp --time 
