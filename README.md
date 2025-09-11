# COL764 – Assignment 2

This repository implements boolean phrase search, VSM, BM25, and feedback retrieval over the provided CORD-19 subset using spaCy's default English tokenizer.

## Environment

- Python 3.12
- Dependencies: `spacy` (tokenizer only)
- No internet required at build time.

## Project Layout

- `tokenize_corpus.py` – Tokenizes corpus and generates vocabulary.
- `build_index.py` – Builds positional inverted index and sidecar stats.
- `phrase_search.py` – Performs boolean phrase search.
- `vsm.py` – Implements VSM with tf-idf and cosine similarity.
- `bm25_retrieval.py` – Implements Okapi BM25 retrieval.
- `feedback_retrieval.py` – Implements pseudo-relevance feedback.
- Shell scripts for each task with optional `--time` flag for timing.

## Run Steps

Replace placeholders with actual paths.

```bash
bash tokenize_corpus.sh <CORPUS_DIR> <VOCAB_DIR> [--time]
bash build_index.sh <CORPUS_DIR> <VOCAB_PATH> <INDEX_DIR> [--time]
bash phrase_search.sh <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> [--time]
bash vsm.sh <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <k> [--time]
bash vsm.sh <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <k> [--time]
bash bm25_retrieval.sh <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <k> [--time]
bash bm25_retrieval.sh <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <k> [--time]
bash feedback.sh <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <k> [--time]
bash feedback.sh <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <k> [--time]
```

## Outputs

Each retrieval program writes a TREC run file (one line per result):

```
qid docid rank score
```

Expected files in `<OUTPUT_DIR>`:
- `phrase_search_docids.txt`
- `vsm_docids.txt`
- `bm25_docids.txt`
- `feedback_docids.txt`
