#!/usr/bin/env python3
import os
import sys
import json
import math
from collections import defaultdict
from typing import Optional

# Fast tokenizer (no spaCy model needed)
try:
    import spacy
except ImportError as e:
    raise SystemExit(
        "spaCy is required. Install with:\n  pip install spacy"
    ) from e


class InvertedIndex:
    """
    postings: dict[int -> dict[int -> {"tf": int, "positions": list[int]}]]
        term_id -> { doc_internal_id: {"tf": TF, "positions": [...]} }

    token2id/id2token and doc2id/id2doc are build-time helpers and are not
    serialized to index.json for Task-1.
    """
    def __init__(self):
        self.postings = defaultdict(dict)   # term_id -> {doc_i: {"tf": int, "positions": list[int]}}
        self.token2id = {}                  # token -> term_id
        self.id2token = []                  # term_id -> token
        self.doc2id = {}                    # ext_doc_id -> internal int id
        self.id2doc = []                    # internal int id -> ext_doc_id
        # For BM25 stats
        self.doc_len = {}                   # ext_doc_id -> int (token count post-tokenization)

    def load_vocab(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                token = line.strip()
                if token:
                    tid = len(self.id2token)
                    self.id2token.append(token)
                    self.token2id[token] = tid

    def assign_doc_id(self, ext_doc_id: str) -> int:
        if ext_doc_id in self.doc2id:
            return self.doc2id[ext_doc_id]
        did = len(self.id2doc)
        self.id2doc.append(ext_doc_id)
        self.doc2id[ext_doc_id] = did
        return did


def _init_tokenizer():
    """
    Fast tokenizer using spaCy's blank English tokenizer.
    Much faster than loading full models like en_core_web_sm.
    """
    nlp = spacy.blank("en")  # rule-based English tokenizer; no internet needed
    nlp.max_length = 300_000_000  # allow very large inputs safely
    return nlp

def _tokenize_spacy_raw(nlp, text: str):
    """
    Tokenize with spaCy as-is (no ASCII filtering, no lowercasing, no digit removal).
    Skips whitespace tokens.
    """
    if not isinstance(text, str):
        text = str(text)
    doc = nlp(text)
    for tok in doc:
        if tok.is_space:
            continue
        yield tok.text


# --------------- REQUIRED FUNCTIONS ----------------

def build_index(corpus_dir: str, vocab_path: str, doc_limit: Optional[int] = None) -> InvertedIndex:
    """
    Build a positional inverted index with explicit TF per (term, doc).
    Tokenization uses spaCy's rule-based tokenizer (no stopword removal, no normalization).
    Only tokens present in vocab are indexed.
    Positions are document-wide (single counter across selected fields).

    Args:
        corpus_dir: directory containing input corpus files
        vocab_path: vocabulary file
        doc_limit: maximum number of documents to process (None = no limit)
    """
    inv = InvertedIndex()
    inv.load_vocab(vocab_path)

    nlp = _init_tokenizer()
    seen_docs = set()
    field_order = ["title", "doi", "date", "abstract"]  # same order as A1

    print("DOC LIMIT IS:", doc_limit if doc_limit is not None else "NO LIMIT")
    doc_count = 0

    for fname in os.listdir(corpus_dir):
        path = os.path.join(corpus_dir, fname)
        if not os.path.isfile(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if doc_limit is not None and doc_count >= doc_limit:
                    print(f"Reached doc limit {doc_limit}, stopping.")
                    return inv  

                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ext_doc_id = obj.get("doc_id")
                if not ext_doc_id:
                    continue
                if ext_doc_id in seen_docs:
                    continue
                seen_docs.add(ext_doc_id)

                did = inv.assign_doc_id(ext_doc_id)
                doc_count += 1
                # print(f"Processed doc {doc_count}: {ext_doc_id}")

                per_doc_positions = defaultdict(list)
                pos = 0

                # Concatenate all relevant fields into one string and tokenize (spaCy raw)
                text_parts = [str(obj[field]) for field in field_order if obj.get(field)]
                text = " ".join(text_parts)
                for token in _tokenize_spacy_raw(nlp, text):
                    tid = inv.token2id.get(token)
                    if tid is not None:
                        per_doc_positions[tid].append(pos)
                    pos += 1

                # Record document length for BM25 normalization (post-tokenization count)
                inv.doc_len[ext_doc_id] = pos

                for tid, positions in per_doc_positions.items():
                    positions_sorted = sorted(positions)
                    inv.postings[tid][did] = {
                        "tf": len(positions_sorted),
                        "positions": positions_sorted
                    }

    return inv


def save_index(inv: InvertedIndex, index_dir: str) -> None:
    """
    Save to index.json with deterministic ordering:
      - terms sorted by token alphabetically
      - docs sorted by external doc_id alphabetically
    {
      "term": {
        "df": int,
        "postings": {
          "ext_doc_id": {"tf": <int>, "pos": [ ... ]},
          ...
        }
      },
      ...
    }
    """
    os.makedirs(index_dir, exist_ok=True)
    result = {}

    term_ids_sorted = sorted(inv.postings.keys(), key=lambda tid: inv.id2token[tid])

    for tid in term_ids_sorted:
        term = inv.id2token[tid]
        postings = inv.postings[tid]

        doc_ids_sorted = sorted(postings.keys(), key=lambda did: inv.id2doc[did])

        term_obj = {}
        for did in doc_ids_sorted:
            ext_id = inv.id2doc[did]
            entry = postings[did]
            term_obj[ext_id] = {"tf": entry["tf"], "pos": entry["positions"]}
            
        result[term] = {"df": len(term_obj), "postings": term_obj}

    out_path = os.path.join(index_dir, "index.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, separators=(",", ":"))  # compact
        # json.dump(result, f, indent=2)  # pretty


def load_index(index_dir: str) -> dict:
    """
    Load index.json produced by save_index.
    Returns dict[str -> {"df": int, "postings": dict[str -> {"tf": int, "pos": list[int]}]}].
    """
    path = os.path.join(index_dir, "index.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_bm25_stats(inv: InvertedIndex, index_dir: str) -> None:
    """
    Save BM25-global statistics alongside index.json as bm25.json.
    Structure:
    {
      "N": <int>,
      "avgdl": <float>,
      "doc_len": { "ext_doc_id": <int>, ... },
      "hyperparams": { "k1": 1.5, "b": 0.75, "k3": 0, "idf_clamp_zero": true }
    }
    """
    os.makedirs(index_dir, exist_ok=True)
    N = len(inv.id2doc)
    # avgdl over recorded doc lengths (fallback to 0.0 for empty)
    lengths = list(inv.doc_len.values())
    avgdl = (sum(lengths) / len(lengths)) if lengths else 0.0
    payload = {
        "N": N,
        "avgdl": avgdl,
        "doc_len": inv.doc_len,
    }
    out_path = os.path.join(index_dir, "bm25.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))


def save_vsm_index(inv: InvertedIndex, index_dir: str):
    """Compute and save compact VSM index directly from InvertedIndex object."""
    N = len(inv.id2doc)
    idf = {}  # calculated for each term
    doc_norms = defaultdict(float)
    postings_out = defaultdict(dict)  # containing the list of documents and the tf for each term

    # loop through each term
    for tid, postings in inv.postings.items():
        term = inv.id2token[tid]
        df = len(postings)
        idf[term] = math.log(N / df)

        for did, info in postings.items():
            ext_doc_id = inv.id2doc[did]
            tf = info["tf"]

            # store for postings
            postings_out[term][ext_doc_id] = tf

            # accumulate squared weights for doc norms
            w_td = (1 + math.log(tf)) * idf[term]
            doc_norms[ext_doc_id] += w_td ** 2

    for doc in doc_norms:
        doc_norms[doc] = math.sqrt(doc_norms[doc])

    vsm_data = {
        "N": N,
        "idf": idf,
        "doc_norms": dict(doc_norms),
        "postings": postings_out
    }

    os.makedirs(index_dir, exist_ok=True)
    with open(os.path.join(index_dir, "vsm.json"), "w", encoding="utf-8") as f:
        json.dump(vsm_data, f, separators=(",", ":"))


if __name__ == "__main__":
    # Usage: python3 build_index.py <CORPUS_DIR> <VOCAB.txt> <INDEX_DIR>
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <CORPUS_DIR> <VOCAB.txt> <INDEX_DIR>")
        sys.exit(1)

    corpus_dir, vocab_file, index_dir = sys.argv[1:4]

    inv = build_index(corpus_dir, vocab_file)
    save_index(inv, index_dir)
    save_bm25_stats(inv, index_dir)
    save_vsm_index(inv, index_dir)
    print(f"Wrote index to: {os.path.join(index_dir, 'index.json')}")
    print(f"Wrote BM25 stats to: {os.path.join(index_dir, 'bm25.json')}")
    print(f"Wrote VSM index to: {os.path.join(index_dir, 'vsm.json')}")
