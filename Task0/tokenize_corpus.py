# TODO: Clarify whether digits should be stripped *before* running spaCy tokenizer,
# or whether to run spaCy first and then filter tokens post-hoc. Current design
# strips digits after tokenization, but the spec only says "remove digits" (A2 §1.3):contentReference[oaicite:0]{index=0}.

# TODO: spaCy has a built-in stopword list, but the assignment requires using the
# provided stopwords.txt file. Double-check if we must ignore spaCy’s stock stopwords
# entirely and stick only to the given file (A2 §3.1, Task 0):contentReference[oaicite:1]{index=1}.

# TODO: Directory iteration currently sorts files deterministically for consistent vocab.
# This may not be strictly required by the assignment spec (A2 only specifies using all
# docs in the collection), so we could drop sorting if not needed:contentReference[oaicite:2]{index=2}.




#!/usr/bin/env python3
# tokenize_corpus.py — Assignment 2 Task 0 (spaCy tokenizer)
# Usage: python3 tokenize_corpus.py <CORPUS_DIR_OR_FILE> <STOPWORDS_FILE> <VOCAB_DIR>
#
# Behavior (kept consistent with your A1 conventions):
#  - Tokenize with spaCy (English, rule-based).
#  - ASCII-only (strip non-ASCII), lowercase.
#  - Drop tokens containing any ASCII digit 0–9.
#  - Remove stopwords from the provided file (exact-match after lowercase).
#  - Keep punctuation/symbols as tokens if they survive the above filters.
#  - De-duplicate by doc_id if present; accumulate a global vocab set; write sorted vocab.txt.

import os
import sys
import json
from typing import Iterable, Set, Optional
from pathlib import Path

import spacy  # Allowed in A2 for tokenization only

import argparse

DEFAULT_MODE = "a1_style"  # "a1_style" -> your current pipeline; "spacy_raw" -> spaCy-only tokens


# Choose fields like your A1 script (easy to tweak if needed)
SELECT_FIELDS = ["title", "doi", "date", "abstract"]

# Translate table to delete ASCII digits 0-9
_DIGIT_DELETE = str.maketrans("", "", "0123456789")

def _load_stopwords(path: str) -> Set[str]:
    sw = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.strip()
            if tok:
                sw.add(tok.lower())
    return sw

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

def _tokenize_spacy(nlp, text: str, stopwords: Set[str]):
    """
    Assignment tokenizer:
    1. Keep only ASCII by removing any non-ASCII bytes
    2. Lowercase
    3. Remove digits 0–9 completely from text (keep punctuation/symbols)
    4. Tokenize with spaCy
    5. Filter stopwords
    """
    # Step 1: ASCII-only filtering
    text = text.encode("ascii", "ignore").decode("ascii")
    
    # Step 2 & 3: Lowercase and remove digits completely
    text = text.lower().translate(_DIGIT_DELETE)
    
    # Step 4: spaCy tokenize the cleaned text
    doc = nlp(text)
    
    # Step 5: Filter tokens and stopwords
    for tok in doc:
        if tok.is_space:
            continue
        t = tok.text
        # Apply stopword filter
        if t in stopwords:
            continue
        # Note: we intentionally keep punctuation/symbols (as in A1 rule-set)
        if t:
            yield t


def _tokenize_spacy_raw(nlp, text: str):
    """
    Pure spaCy tokenization:
    - No ASCII stripping, no lowercasing, no digit filtering, no external stopwords.
    - We only skip whitespace tokens.
    - NOTE: spaCy does NOT drop stopwords by default; to drop them, we'd need to filter
      using `nlp.Defaults.stop_words`, which we intentionally do NOT do in spacy_raw mode.
    """
    doc = nlp(text)
    for tok in doc:
        if tok.is_space:
            continue
        yield tok.text  # as-is


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

def build_vocab(corpus_path: str, stopwords_file: str, vocab_dir: str, mode: str = DEFAULT_MODE) -> None:
    os.makedirs(vocab_dir, exist_ok=True)
    stopwords = _load_stopwords(stopwords_file)
    vocab: Set[str] = set()
    seen_doc_ids: Set[str] = set()

    # Use spaCy WITHOUT downloading a model (offline-safe):
    # spaCy's rule-based English tokenizer via blank('en').
    # Use spaCy WITHOUT parser/NER, just tokenizer
        # spaCy tokenizer setup (parser/NER not needed)
    try:
        nlp = spacy.blank("en")          # rule-based English tokenizer; no internet needed
    except Exception:
        print("Had to switch to ultra-fallback tokenizer")
        nlp = spacy.blank("xx")          # ultra-fallback, but "en" is expected
    nlp.max_length = 300_000_000         # allow very large inputs safely for tokenization

    # Only load external stopwords in a1_style mode
    stopwords = _load_stopwords(stopwords_file) if mode == "a1_style" else set()



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

            
            if mode == "spacy_raw":
                for t in _tokenize_spacy_raw(nlp, text):
                    vocab.add(t)
            else:
                for t in _tokenize_spacy(nlp, text, stopwords):
                    vocab.add(t)

    out_path = os.path.join(vocab_dir, "vocab.txt")
    with open(out_path, "w", encoding="utf-8") as out:
        for tok in sorted(vocab):
            out.write(tok + "\n")

    print(f"Wrote vocabulary: {out_path} ({len(vocab)} tokens)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="tokenize_corpus.py")
    parser.add_argument("corpus", help="CORPUS_DIR_OR_FILE")
    parser.add_argument("stopwords", help="PATH_OF_STOPWORDS_FILE")
    parser.add_argument("vocab_dir", help="VOCAB_DIR (output)")
    parser.add_argument("--mode", choices=["a1_style", "spacy_raw"], default=DEFAULT_MODE,
                        help="a1_style = ASCII/lower/digit/stopword filters; spacy_raw = spaCy tokens as-is")
    args = parser.parse_args()

    corpus = os.path.abspath(args.corpus)
    stop = os.path.abspath(args.stopwords)
    vocab_out = os.path.abspath(args.vocab_dir)
    build_vocab(corpus, stop, vocab_out, mode=args.mode)

