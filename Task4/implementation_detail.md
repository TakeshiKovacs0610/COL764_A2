# BM25 Retrieval — Implementation Details

## Overview

This implements Okapi BM25 ranking over the Task 1 positional index. Two new components were added:

- `Task4/bm25_retrieval.py`: BM25 query scoring with two APIs (`bm25_query`, `bm25`).
- `Task4/bm25_retrieval.sh`: Thin shell wrapper to invoke the Python entrypoint.

Minimal changes were made to `Task1/build_index.py` to export document-length statistics in a new `bm25.json` file (next to `index.json`). The original index format remains unchanged.

## Index and Stats

- `index.json` is unchanged: for each term, we store `df` and a `postings` map of `docid -> { tf, pos }`.
- A new file `bm25.json` is produced by `build_index.py` with the structure:

  ```json
  {
    "N": 123456,
    "avgdl": 284.7,
    "doc_len": { "docA": 350, "docB": 120 },
    "hyperparams": { "k1": 1.5, "b": 0.75, "k3": 0, "idf_clamp_zero": true }
  }
  ```

Design choices:
- `doc_len[docid]` is the count of tokens after tokenization across the concatenated fields `title, doi, date, abstract` (same field set and order as Task 1). This includes all tokens produced by the tokenizer, not just vocabulary hits, aligning BM25 length normalization with the tokenization pipeline.
- `avgdl` is the mean of `doc_len` values over all indexed documents. If there are no documents, it safely defaults to `0.0`.
- Hyperparameters are saved to document the final choices compiled into the system.

## Tokenization

- Uses spaCy’s rule-based English tokenizer (`spacy.blank("en")`), same as Task 1 and Task 2. No stemming, lemmatization, or external stopword removal.
- The stopword file path is accepted by the CLI and shell script but deliberately ignored, per instructions.
- No lowercasing is applied because the Task 1 index also keeps tokens as-is.

## BM25 Scoring

For a query, we compute:

- `idf(t) = max(0, ln((N - df_t + 0.5) / (df_t + 0.5)))` (natural log, clamped at 0).
- Per-document term contribution:
  `norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len_d / avgdl))` with safeguards if `avgdl == 0`.
- Query term frequency weighting is disabled (`k3 = 0`), so each unique query term contributes once.
- Final score is the sum over unique query terms appearing in the index.
- Results are sorted by score descending, then by `docid` ascending for tie-breaking.

Hyperparameters:
- Defaults are `k1 = 1.5`, `b = 0.75`, `k3 = 0`. These are set in `bm25.json` and used in scoring.

## Query I/O and Output Format

- `bm25()` reads queries from flexible JSON/JSONL formats, consistent with Task 2 (`query_id|qid|id` and `query|text|title`).
- Output is written in TREC-style lines to `<OUTPUT_DIR>/bm25_docids.txt` with four fields:
  `qid docid rank score`
- We write up to `k` lines per query (fewer if there are not enough matching documents).

## Shell Script

- Invocation: `bm25_retrieval.sh <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <PATH_OF_STOPWORDS_FILE> <k>`
- Ensures output directory exists and calls the Python script.

## Code Changes Summary

- `Task1/build_index.py`
  - Added `self.doc_len` to `InvertedIndex` to track per-document token counts.
  - Recorded token counts (`pos`) per document during index construction.
  - Added `save_bm25_stats()` to emit `bm25.json` with `N`, `avgdl`, `doc_len`, and hyperparameters.
  - Called `save_bm25_stats()` from `__main__` after writing `index.json`.

- `Task4/bm25_retrieval.py`
  - Implemented loading of both `index.json` and `bm25.json`.
  - Implemented BM25 scoring (`bm25_query`) and multi-query driver (`bm25`).
  - Added CLI compatible with the shell wrapper.

- `Task4/bm25_retrieval.sh`
  - Created; passes five args (including unused stopwords) and writes `bm25_docids.txt`.

## Edge Cases and Safeguards

- If `avgdl` is zero, we avoid division-by-zero by clamping the denominator’s length component.
- Negative IDF values are clamped to zero to avoid penalizing ubiquitous terms too heavily.
- Empty queries or missing terms yield no results for that query.

## Out of Scope

- Precision/Recall/F1 computation is not performed here. You can use `Scoring/evaluate_ir.py` externally with the produced run file and your qrels to compute metrics.

