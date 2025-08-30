#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
phrase_retrieval.py — Assignment 2 / Task 2: Boolean + Phrase Retrieval

CLI:
  python phrase_retrieval.py <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR>

Assumptions (aligned with the project's coding guide):
- The inverted index is *positional* and saved (uncompressed) by Task 1.
- Tokenization order for *both* indexing and querying:
    1) ASCII filter    -> text.encode("ascii", "ignore").decode("ascii")
    2) Lowercase       -> text.lower()
    3) Digit removal   -> translate(_DIGIT_DELETE)
    4) spaCy tokenize  -> spacy.blank("en") (rule-based & fast)
- No stopword removal at retrieval time (preserves phrase integrity).
- Operators: AND, OR, NOT; parentheses supported; implicit ANDs inferred.

Index layout expected in <INDEX_DIR> (choose one of the supported formats):
  (A) Single JSON file named 'index.json' with structure:
      {
        "token": {
          "doc_id_1": {"tf": int, "positions": [int, ...]},
          "doc_id_2": {"tf": int, "positions": [int, ...]},
          ...
        },
        ...
      }
  (B) Legacy compressed layout (Assignment 1 style): 'index.z' + 'docid_map.z'
      — if present, we will autodetect and load it for compatibility.

Query file format: JSONL; one object per line, e.g.
  {"query_id": "1", "title": "\"information retrieval\" AND (neural OR \"deep learning\") NOT survey"}

Output:
  <OUTPUT_DIR>/docids.txt  in TREC-like format:
    <qid>\t<docid>\t<rank>\t1

"""

from __future__ import annotations
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Iterable

# -------------------------- spaCy setup --------------------------
try:
    import spacy
except ImportError as e:
    raise SystemExit("spaCy is required. Install with:\n  pip install spacy") from e

# Lightweight, rule-based tokenizer (fast; no model download)
NLP = spacy.blank("en")
NLP.max_length = 300_000_000  # allow very large query strings if needed

_DIGIT_DELETE = str.maketrans("", "", "0123456789")


# -------------------------- Utilities --------------------------
def _ascii_lower_no_digits(text: str) -> str:
    """Apply the project-specified preprocessing to text."""
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    return text.translate(_DIGIT_DELETE)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _read_jsonl_any(path: str) -> List[dict]:
    """
    Read JSONL that may be UTF-16 (Windows) or UTF-8.
    Each line is a JSON object.
    """
    last_err: Optional[Exception] = None
    for enc in ("utf-16", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                lines = [ln.strip().rstrip(",") for ln in f if ln.strip()]
            out: List[dict] = []
            for ln in lines:
                try:
                    obj = json.loads(ln)
                    if isinstance(obj, dict):
                        out.append(obj)
                except json.JSONDecodeError:
                    # tolerate malformed lines
                    continue
            return out
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read queries file: {path} ({last_err})")


# -------------------------- Index loading --------------------------
def _vle_decode(buf: bytes, pos: int) -> Tuple[int, int]:
    """Legacy variable-length decoder (Assignment 1 compatibility)."""
    x = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        x |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return x, pos
        shift += 7


def _load_index_json(index_dir: str) -> Dict[str, Dict[str, List[int]]]:
    """
    Load positional inverted index from index.json (Assignment 2 default format).
    We normalize postings so values are *positions only* (List[int]).
    """
    p = os.path.join(index_dir, "index.json")
    if not os.path.isfile(p):
        raise FileNotFoundError("index.json not found in index_dir")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize: token -> docid -> positions[]
    norm: Dict[str, Dict[str, List[int]]] = {}
    for tok, posting in data.items():
        inner = {}
        if isinstance(posting, dict):
            for docid, val in posting.items():
                if isinstance(val, dict) and "positions" in val:
                    inner[docid] = list(map(int, val["positions"]))
                elif isinstance(val, list) and (not val or isinstance(val[0], int)):
                    inner[docid] = list(map(int, val))
                else:
                    # Unknown shape; be permissive
                    try:
                        inner[docid] = list(map(int, val.get("pos", [])))
                    except Exception:
                        inner[docid] = []
        norm[tok] = inner
    return norm


def _load_index_legacy_compressed(index_dir: str) -> Dict[str, Dict[str, List[int]]]:
    """
    Legacy loader: reads 'docid_map.z' and 'index.z' (Assignment 1 style).
    Returns token -> docid -> positions[].
    """
    import zlib

    with open(os.path.join(index_dir, "docid_map.z"), "rb") as f:
        compressed_docmap = f.read()
    doc_id_buffer = zlib.decompress(compressed_docmap)

    pos = 0
    n, pos = _vle_decode(doc_id_buffer, pos)  # number of docs
    docid_to_int = {}
    for _ in range(n):
        length, pos = _vle_decode(doc_id_buffer, pos)
        docid = doc_id_buffer[pos:pos+length].decode("utf-8")
        pos += length
        doc_int, pos = _vle_decode(doc_id_buffer, pos)
        docid_to_int[docid] = doc_int
    int_to_docid = {val: key for key, val in docid_to_int.items()}

    with open(os.path.join(index_dir, "index.z"), "rb") as f:
        compressed_index = f.read()
    inv_index_buffer = zlib.decompress(compressed_index)

    inv_index: Dict[str, Dict[str, List[int]]] = {}
    pos = 0
    while pos < len(inv_index_buffer):
        term_len, pos = _vle_decode(inv_index_buffer, pos)
        term = inv_index_buffer[pos:pos+term_len].decode("utf-8")
        pos += term_len

        df, pos = _vle_decode(inv_index_buffer, pos)
        postings: Dict[str, List[int]] = {}
        doc_ids = []
        for _ in range(df):
            doc_gap, pos = _vle_decode(inv_index_buffer, pos)
            doc_ids.append(doc_gap if not doc_ids else doc_ids[-1] + doc_gap)

            tf, pos = _vle_decode(inv_index_buffer, pos)
            positions: List[int] = []
            for _ in range(tf):
                gap, pos = _vle_decode(inv_index_buffer, pos)
                positions.append(gap if not positions else positions[-1] + gap)

            postings[int_to_docid[doc_ids[-1]]] = positions
        inv_index[term] = postings
    return inv_index


def load_index(index_dir: str) -> Dict[str, Dict[str, List[int]]]:
    """
    Autodetect and load a positional inverted index from index_dir.

    Preferred: index.json  (Assignment 2)
    Fallback:  legacy compressed  (Assignment 1)
    """
    json_path = os.path.join(index_dir, "index.json")
    if os.path.isfile(json_path):
        return _load_index_json(index_dir)
    # Legacy?
    i_z = os.path.join(index_dir, "index.z")
    d_z = os.path.join(index_dir, "docid_map.z")
    if os.path.isfile(i_z) and os.path.isfile(d_z):
        return _load_index_legacy_compressed(index_dir)
    raise FileNotFoundError(
        f"No supported index files in {index_dir}. Expected 'index.json' or legacy 'index.z' + 'docid_map.z'.")


# -------------------------- Query parsing --------------------------
OPERATORS: Set[str] = {"AND", "OR", "NOT"}

@dataclass(frozen=True)
class Atom:
    """
    Represents a query atom:
      - kind = "TOKEN": single token, .tokens = ["foo"]
      - kind = "PHRASE": multi-token phrase, .tokens = ["new", "york"]
    """
    kind: str
    tokens: Tuple[str, ...]  # immutable for hashing/caching


def _spacy_tokens(text: str) -> List[str]:
    """spaCy tokenization after the required preprocessing."""
    clean = _ascii_lower_no_digits(text)
    doc = NLP(clean)
    return [t.text for t in doc if not t.is_space]


def lex_query(raw_query: str) -> List[object]:
    """
    Lex the raw query into a stream of:
      - Atom(kind="TOKEN", tokens=(tok,))
      - Atom(kind="PHRASE", tokens=(tok1, tok2, ...))
      - Operator strings: "AND" | "OR" | "NOT"
      - Parentheses: "(" | ")"

    Rules:
      - Double quotes define phrases: "new york"
      - Outside quotes, we tokenize with spaCy; multi-token runs are emitted
        as multiple TOKEN atoms. (Implicit ANDs will be inserted separately.)
      - Operators are case-insensitive in input; we emit uppercase.
    """
    s = _ascii_lower_no_digits(raw_query)

    out: List[object] = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(" or ch == ")":
            out.append(ch)
            i += 1
            continue
        if ch == '"':
            # phrase
            j = i + 1
            while j < n and s[j] != '"':
                j += 1
            phrase_text = s[i+1:j]  # may be empty
            toks = _spacy_tokens(phrase_text)
            if toks:
                out.append(Atom("PHRASE", tuple(toks)))
            i = j + 1 if j < n else n
            continue

        # parse a word-ish chunk until whitespace or paren or quote
        j = i
        while j < n and (not s[j].isspace()) and s[j] not in '()"':
            j += 1
        chunk = s[i:j]
        i = j

        # Is it an operator?
        if chunk in ("and", "or", "not"):
            out.append(chunk.upper())
            continue

        # Otherwise, tokenize this chunk with spaCy and emit TOKEN atoms
        toks = _spacy_tokens(chunk)
        for t in toks:
            out.append(Atom("TOKEN", (t,)))
    return out


def _is_atom(x: object) -> bool:
    return isinstance(x, Atom)


def insert_implicit_ands(tokens: List[object]) -> List[object]:
    """
    Insert AND between:
      - atom )    (  atom NOT
      - ) (
    """
    out: List[object] = []
    prev: Optional[object] = None
    for cur in tokens:
        if prev is not None:
            need_and = (
                (_is_atom(prev) or prev == ")")
                and (_is_atom(cur) or cur in {"(", "NOT"})
            )
            if need_and:
                out.append("AND")
        out.append(cur)
        prev = cur
    return out


def to_postfix(tokens: List[object]) -> List[object]:
    """Shunting-yard: operators to postfix; atoms flow through."""
    prec = {"NOT": 3, "AND": 2, "OR": 1}
    right_assoc = {"NOT"}
    out: List[object] = []
    st: List[str] = []
    for tok in tokens:
        if _is_atom(tok):
            out.append(tok)
        elif tok in OPERATORS:
            while st:
                top = st[-1]
                if top in OPERATORS:
                    if ((tok not in right_assoc and prec[tok] <= prec[top]) or
                        (tok in right_assoc and prec[tok] <  prec[top])):
                        out.append(st.pop()); continue
                break
            st.append(tok)  # type: ignore
        elif tok == "(":
            st.append(tok)  # type: ignore
        elif tok == ")":
            while st and st[-1] != "(":
                out.append(st.pop())
            if not st:
                raise ValueError("mismatched ')'")
            st.pop()  # pop "("
        else:
            raise ValueError(f"Unexpected token: {tok}")
    while st:
        top = st.pop()
        if top in {"(", ")"}:
            raise ValueError("mismatched parens")
        out.append(top)
    return out


# -------------------------- Evaluation --------------------------
class BooleanPhraseEvaluator:
    def __init__(self, index: Dict[str, Dict[str, List[int]]]):
        self.idx = index
        all_docs: Set[str] = set()
        for posting in index.values():
            all_docs.update(posting.keys())
        self._universe = all_docs
        # Simple caches to avoid recomputation on complex queries
        self._token_cache: Dict[str, Set[str]] = {}
        self._phrase_cache: Dict[Tuple[str, ...], Set[str]] = {}

    # ---- Basic accessors ----
    def docs_for_token(self, tok: str) -> Set[str]:
        s = self._token_cache.get(tok)
        if s is not None:
            return s
        s = set(self.idx.get(tok, {}).keys())
        self._token_cache[tok] = s
        return s

    # ---- Phrase matching ----
    @staticmethod
    def _phrase_occurs_in_doc(pos_lists: List[List[int]]) -> bool:
        """
        Given k position lists (already filtered to the same doc),
        return True iff there exists p in pos_lists[0] such that
        for each i>0, there is an occurrence at p+i in pos_lists[i].

        Two-pointer sweep that only moves forward across lists.
        """
        k = len(pos_lists)
        if k == 1:
            return len(pos_lists[0]) > 0

        # Index pointers for lists 1..k-1 (we scan base positions from list0)
        ptrs = [0] * k
        base = pos_lists[0]
        for p in base:
            ok = True
            prev = p
            # For each subsequent list, advance pointer until >= prev+1
            for i in range(1, k):
                lst = pos_lists[i]
                j = ptrs[i]
                need = prev + 1
                # advance monotonically
                L = len(lst)
                while j < L and lst[j] < need:
                    j += 1
                ptrs[i] = j
                if j >= L or lst[j] != need:
                    ok = False
                    break
                prev = lst[j]
            if ok:
                return True
        return False

    def docs_for_phrase(self, phrase: Tuple[str, ...]) -> Set[str]:
        cached = self._phrase_cache.get(phrase)
        if cached is not None:
            return cached

        # Gather postings for each token in phrase
        postings_per_token: List[Dict[str, List[int]]] = []
        for tok in phrase:
            postings = self.idx.get(tok)
            if not postings:
                self._phrase_cache[phrase] = set()
                return set()
            postings_per_token.append(postings)

        # Intersect candidate doc sets, starting from the rarest token
        # (helps prune early).
        # Build (token_index, doc_set) pairs to sort by df.
        token_docsets = [(i, set(p.keys())) for i, p in enumerate(postings_per_token)]
        token_docsets.sort(key=lambda x: len(x[1]))  # increasing df
        candidate_docs = token_docsets[0][1].copy()
        for _, s in token_docsets[1:]:
            candidate_docs.intersection_update(s)
            if not candidate_docs:
                self._phrase_cache[phrase] = set()
                return set()

        # For each candidate doc, check adjacency using the stored positions
        hits: Set[str] = set()
        for d in candidate_docs:
            pos_lists = [postings_per_token[i][d] for i in range(len(postings_per_token))]
            if self._phrase_occurs_in_doc(pos_lists):
                hits.add(d)

        self._phrase_cache[phrase] = hits
        return hits

    # ---- Boolean eval on postfix ----
    def eval_postfix(self, postfix: List[object]) -> Set[str]:
        st: List[Set[str]] = []
        for tok in postfix:
            if isinstance(tok, Atom):
                if tok.kind == "TOKEN":
                    st.append(self.docs_for_token(tok.tokens[0]))
                elif tok.kind == "PHRASE":
                    st.append(self.docs_for_phrase(tok.tokens))
                else:
                    raise ValueError(f"Unknown atom kind {tok.kind}")
            elif tok == "NOT":
                if not st:
                    raise ValueError("NOT missing operand")
                s = st.pop()
                st.append(self._universe - s)
            elif tok in {"AND", "OR"}:
                if len(st) < 2:
                    raise ValueError(f"{tok} missing operands")
                b = st.pop(); a = st.pop()
                st.append(a & b if tok == "AND" else a | b)
            else:
                raise ValueError(f"Unexpected token in postfix: {tok}")
        if len(st) != 1:
            raise ValueError("Invalid expression")
        return st[0]


# -------------------------- Public API --------------------------
def boolean_phrase_retrieval(index_dir: str, query_file: str, output_dir: str) -> None:
    """
    Load index, read queries, run boolean+phrase retrieval, and write TREC output.
    """
    idx = load_index(index_dir)
    ev = BooleanPhraseEvaluator(idx)

    _ensure_dir(output_dir)
    out_path = os.path.join(output_dir, "docids.txt")
    with open(out_path, "w", encoding="utf-8") as out:
        for q in _read_jsonl_any(query_file):
            qid = str(q.get("query_id", "")).strip()
            title = str(q.get("title", "")).strip()
            if not qid or not title:
                continue
            # Parse
            lexed = lex_query(title)
            lexed = insert_implicit_ands(lexed)
            postfix = to_postfix(lexed)
            # Evaluate
            docs = ev.eval_postfix(postfix)
            # Deterministic ordering
            for rank, docid in enumerate(sorted(docs), start=1):
                out.write(f"{qid}\t{docid}\t{rank}\t1\n")


# -------------------------- CLI --------------------------
def _usage() -> None:
    sys.stderr.write(
        "Usage:\n"
        "  python phrase_retrieval.py <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR>\n"
    )

def main(argv: List[str]) -> int:
    if len(argv) != 4:
        _usage()
        return 2
    index_dir, query_file, output_dir = argv[1:4]
    boolean_phrase_retrieval(index_dir, query_file, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
