

# BM25 Retrieval — Coder-Agent Instructions

## 0) Core definition

* **BM25 (Okapi)**: rank by

  $$
  \text{score}(d,q)=\sum_{t\in q}\max\!\bigg(0, \ln\frac{N-df_t+0.5}{df_t+0.5}\bigg) \cdot \frac{tf_{t,d}(k_1+1)}{tf_{t,d}+k_1\big(1-b+b\cdot |d|/\text{avgdl}\big)}
  $$
* Clamp IDF at 0 (never negative).
* Hyperparameters: use `k1=1.5`, `b=0.75`, `k3=0` unless you later hard-code tuned values.

## 1) Files & entrypoints

* Modify **`build_index.py`** so that it creates **two outputs**:

  1. `index.json` (unchanged; same as before, including positions).
  2. `bm25.json` (new sibling file with document statistics).
* Add new program **`bm25_retrieval.py`**.
* Add shell script **`bm25_retrieval.sh`**.
  Invocation must be:

  ```
  bm25_retrieval.sh <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <PATH_OF_STOPWORDS_FILE> <k>
  ```
* Output file: `<OUTPUT_DIR>/bm25_docids.txt`.

## 2) Structure of `bm25.json`

```json
{
  "N": 123456,
  "avgdl": 284.7,
  "doc_len": {
    "docA": 350,
    "docB": 120,
    "docC": 240
  },
  "hyperparams": { "k1": 1.5, "b": 0.75, "k3": 0, "idf_clamp_zero": true }
}
```

* `N` = total #docs.
* `avgdl` = average document length.
* `doc_len` = per-doc token count (post tokenization).
* `hyperparams` = fixed choices you will later tune.

## 3) Index handling

* Keep **`index.json`** format the same:

  * Each term maps to `df` and postings with `tf` (and positions, unused for BM25).
* `bm25.json` supplements it with doc lengths and global stats.

## 4) Query processing

* **Ignore the stopword file.**
  It must be accepted as an argument but not used.
* **Tokenization**: use **spaCy blank model**. Concatenate full query text, pass into the blank spaCy tokenizer, take resulting tokens directly.
* Do **not** perform stemming, lemmatization, or external stopword removal.
* Lowercasing only if the index used lowercased tokens.

## 5) Scoring loop

For each query:

1. Tokenize as above → `query_terms`.
2. Initialize empty `scores: dict[doc_id → float]`.
3. For each **unique term** present in both query and index:

   * Load `df`, compute `idf = max(0, log((N - df + 0.5)/(df + 0.5)))`.
   * For each doc in postings:

     * `tf = postings[doc]["tf"]`
     * `len_d = doc_len[doc]`
     * `norm = (tf * (k1+1)) / (tf + k1 * (1 - b + b * len_d/avgdl))`
     * Add `idf * norm` to `scores[doc]`.
4. Select **top-k** docs by score.
5. Break ties by `docid` (string order).

## 6) Functions to implement

### `bm25_query(query: str, index: object, k: int) -> list`

* Input: raw query string, combined index data (`index.json` + `bm25.json` loaded into memory), `k`.
* Output: list of `(docid, score)` sorted by score desc, tie-broken by docid, max length `k`.

### `bm25(queryFile: str, index_dir: str, stopword_file: str, k: int, outFile: str) -> None`

* Load `index.json` + `bm25.json` from `index_dir`.
* Read queries from `queryFile` (same format as earlier tasks).
* For each query:

  * Run `bm25_query`.
  * Write exactly **k lines** to `outFile`, format:

    ```
    qid docid rank score
    ```

    where `rank` is 1..k.

## 7) Shell script behavior

* Passes 5 args (including the unused stopword file).
* Ensures `<OUTPUT_DIR>/bm25_docids.txt` exists.
* Calls `bm25_retrieval.py` with those args.

## 8) Scope limitations

* **Do not** compute precision/recall/F1 yet.
* The only required output is the correctly formatted `bm25_docids.txt`.

