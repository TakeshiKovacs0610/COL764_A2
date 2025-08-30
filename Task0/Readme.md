
# Tokenize Corpus (Assignment 2 – Task 0)

This program generates the **vocabulary file (`vocab.txt`)** required for later tasks in COL764 Assignment 2.
It provides two tokenization modes:

* **`a1_style` (default)**:

  * Tokenization using spaCy
  * Post-processing: ASCII filtering, lowercasing, digit removal, and stopword removal (using the provided `stopwords.txt`)
  * Matches the behavior we implemented in Assignment 1

* **`spacy_raw`**:

  * Pure spaCy tokenization (no ASCII filtering, no lowercasing, no digit removal, no external stopwords)
  * Tokens are returned exactly as spaCy generates them

---

## Usage

### Python

```bash
python3 tokenize_corpus.py <CORPUS_DIR_OR_FILE> <STOPWORDS_FILE> <VOCAB_DIR> [--mode {a1_style,spacy_raw}]
```

Examples:

```bash
# Default behavior (A1-style post-processing)
python3 tokenize_corpus.py Data/Doc Data/stopwords.txt temp

# Pure spaCy tokenization
python3 tokenize_corpus.py Data/Doc Data/stopwords.txt temp --mode spacy_raw
```

### Shell Script

A wrapper is provided to match assignment requirements:

```bash
bash tokenize_corpus.sh <CORPUS_DIR_OR_FILE> <STOPWORDS_FILE> <VOCAB_DIR> [--time]
```

Example:

```bash
bash tokenize_corpus.sh Data/Doc Data/stopwords.txt temp --time
```

---

## Output

* The program writes a single file:

  ```
  <VOCAB_DIR>/vocab.txt
  ```
* Contains one token per line, sorted lexicographically.

---

## Notes

* `spacy.blank("en")` is used to avoid dependency on internet downloads.
* For very large files (e.g. CORD-19), we increase `nlp.max_length` to avoid errors.
* Deterministic sorting of files ensures reproducibility across runs.
* No stemming or lemmatization is performed (explicitly disallowed in assignment spec).


