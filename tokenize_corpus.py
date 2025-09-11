#!/usr/bin/env python3
"""
tokenize_corpus.py — Assignment 2 Task 0 (spaCy tokenizer)

Updated per Task 2 instructions:
- Do NOT use any stopwords list. Ignore Data/stopwords.txt entirely.
- Use spaCy's rule-based English tokenizer (spacy.blank("en")) only.
- No extra normalization (no forced lowercase, no digit stripping). Whatever
  spaCy emits as tokens becomes the vocabulary token.

Usage: python3 tokenize_corpus.py <CORPUS_DIR_OR_FILE> <VOCAB_DIR>
"""

import os
import sys
import json
from typing import Iterable, Set
from pathlib import Path

import spacy  # Allowed in A2 for tokenization only

import argparse


# Fields to concatenate for tokenization
SELECT_FIELDS = ["title", "doi", "date", "abstract"]

def _iter_paths(corpus_path: str):
    """
    Yield files to read, in lexicographic order for deterministic behavior.
    (Your A1 code iterated directory entries; we sort them here to avoid any nondeterminism.)
    """
    if os.path.isdir(corpus_path):
        for name in sorted(os.listdir(corpus_path)):
            p = os.path.join(corpus_path, name)
            if os.path.isfile(p):
                yield p
    elif os.path.isfile(corpus_path):
        yield corpus_path

def _pick_text(doc: dict) -> str:
    """
    Concatenate chosen fields into one string.
    - If SELECT_FIELDS is None: use all fields except 'doc_id'
    - Else: only fields listed (if present), still skipping 'doc_id'
    """
    if SELECT_FIELDS is None:
        keys = [k for k in doc.keys() if k != "doc_id"]
    else:
        keys = [k for k in SELECT_FIELDS if k != "doc_id" and k in doc]

    parts = []
    for k in keys:
        v = doc.get(k)
        if isinstance(v, str):
            parts.append(v)
        elif v is None:
            continue
        else:
            # fall back to str() for non-strings
            parts.append(str(v))
    return " ".join(parts)

def _tokenize_spacy_raw(nlp, text: str):
    """Pure spaCy tokenization. Skips only whitespace tokens."""
    doc = nlp(text)
    for tok in doc:
        if tok.is_space:
            continue
        yield tok.text


def _read_jsonlines(path: str):
    """Yield JSON objects from a JSON-lines/NDJSON file, skipping malformed lines."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def build_vocab(corpus_path: str, vocab_dir: str) -> None:
    os.makedirs(vocab_dir, exist_ok=True)
    vocab: Set[str] = set()
    seen_doc_ids: Set[str] = set()

    # spaCy tokenizer setup (parser/NER not needed)
    try:
        nlp = spacy.blank("en")  # rule-based English tokenizer; no internet needed
    except Exception:
        print("Had to switch to ultra-fallback tokenizer")
        nlp = spacy.blank("xx")  # fallback, but "en" is expected
    nlp.max_length = 300_000_000

    for fp in _iter_paths(corpus_path):
        # Expect NDJSON (same as A1); if it is a single large JSON, users can pipe/convert beforehand.
        for doc in _read_jsonlines(fp):
            doc_id = doc.get("doc_id")
            if doc_id:
                if doc_id in seen_doc_ids:
                    continue  # deduplicate by doc_id
                seen_doc_ids.add(doc_id)

            text = _pick_text(doc)
            if not text:
                continue

            for t in _tokenize_spacy_raw(nlp, text):
                vocab.add(t)

    out_path = os.path.join(vocab_dir, "vocab.txt")
    with open(out_path, "w", encoding="utf-8") as out:
        for tok in sorted(vocab):
            out.write(tok + "\n")

    print(f"Wrote vocabulary: {out_path} ({len(vocab)} tokens)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="tokenize_corpus.py")
    parser.add_argument("corpus", help="CORPUS_DIR_OR_FILE")
    parser.add_argument("vocab_dir", help="VOCAB_DIR (output)")
    args = parser.parse_args()

    corpus = os.path.abspath(args.corpus)
    vocab_out = os.path.abspath(args.vocab_dir)
    build_vocab(corpus, vocab_out)
