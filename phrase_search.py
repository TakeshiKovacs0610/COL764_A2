#!/usr/bin/env python3
"""
Phrase Search — Task 2

Conforms to Assignment.md:
- Two functions with specific signatures:
  * phrase_search_query(query: str, index: object) -> list
  * phrase_search(queryFile: str, index_dir: str, stopword_file: str, outFile: str) -> None
- Multi-query output must be in TREC-style lines with four fields:
  qid docid rank score

Behavior:
- Uses spaCy's rule-based English tokenizer (spacy.blank("en")) only. No stopword removal
  and no normalization beyond spaCy tokenization. This must match Task 0 and Task 1.
- Exact phrase search using positional postings: tokens appear at consecutive positions.
"""

import os
import sys
import json
from typing import Dict, List, Iterable, Tuple, Optional

try:
    import spacy
except ImportError as e:
    raise SystemExit(
        "spaCy is required. Install with: pip install spacy"
    ) from e


def load_index(index_dir: str) -> Dict[str, dict]:
    path = os.path.join(index_dir, "index.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def init_tokenizer():
    nlp = spacy.blank("en")  # rule-based English tokenizer; no internet needed
    nlp.max_length = 300_000_000
    return nlp


def tokenize_query(nlp, text: str) -> List[str]:
    doc = nlp(text if isinstance(text, str) else str(text))
    return [t.text for t in doc if not t.is_space]


def phrase_match_in_doc(positions_lists: List[List[int]]) -> bool:
    """
    positions_lists[i] is the sorted list of positions of the i-th query token
    in the same document. Returns True if there exists a starting position p in
    positions_lists[0] such that for every i >= 1, (p + i) is in positions_lists[i].
    """
    if not positions_lists:
        return False
    if len(positions_lists) == 1:
        return len(positions_lists[0]) > 0

    # For efficient membership checks, convert subsequent lists to sets.
    sets = [set(lst) for lst in positions_lists]
    first = positions_lists[0]
    for p in first:
        ok = True
        for i in range(1, len(positions_lists)):
            if (p + i) not in sets[i]:
                ok = False
                break
        if ok:
            return True
    return False


def _phrase_search_candidates(index: Dict[str, dict], query_tokens: List[str]) -> List[str]:
    if not query_tokens:
        return []
    for tok in query_tokens:
        if tok not in index:
            return []
    posting_keys = [set(index[tok]["postings"].keys()) for tok in query_tokens]
    if not posting_keys:
        return []
    candidate_docs = set.intersection(*posting_keys)
    if not candidate_docs:
        return []
    matches = []
    for doc_id in candidate_docs:
        positions_lists = []
        for tok in query_tokens:
            entry = index[tok]["postings"][doc_id]
            positions = entry.get("pos") if isinstance(entry, dict) else None
            if positions is None:
                positions = entry.get("positions", [])
            positions_lists.append(list(positions))
        if phrase_match_in_doc(positions_lists):
            matches.append(doc_id)
    matches.sort()
    return matches


# Public API per Assignment.md
def phrase_search_query(query: str, index: dict) -> list:
    """Given the query string, return all matching document IDs (exact phrase)."""
    nlp = init_tokenizer()
    toks = tokenize_query(nlp, query)
    return _phrase_search_candidates(index, toks)


def phrase_search(queryFile: str, index_dir: str, stopword_file: str, outFile: str) -> None:
    """Given a file containing queries, write results in TREC format to outFile.

    TREC format: one line per hit -> "qid docid rank score"
    For boolean retrieval, score is a constant (1). Rank is 1..N in lexicographic docid order.
    """
    index = load_index(index_dir)
    nlp = init_tokenizer()
    queries = list(_read_queries_json(queryFile))
    with open(outFile, "w", encoding="utf-8") as out:
        for qid, text in queries:
            toks = tokenize_query(nlp, text)
            matches = _phrase_search_candidates(index, toks)
            rank = 1
            for doc_id in matches:
                out.write(f"{qid} {doc_id} {rank} 1\n")
                rank += 1


def _read_queries_json(path: str) -> Iterable[Tuple[str, str]]:
    """Attempt to read a JSON or JSONL file of queries.
    Accepts lines with objects having keys like: (query_id|qid|id) and (query|text|title).
    Yields (qid, text).
    """
    # Try JSON-lines first
    try:
        with open(path, "r", encoding="utf-16") as f:
            content = f.read().strip()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    if not content:
        return []

    # Try JSON-lines
    lines = content.splitlines()
    if len(lines) > 1:
        out = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = obj.get("query_id") or obj.get("qid") or obj.get("id") or ""
            q = obj.get("query") or obj.get("text") or obj.get("title") or ""
            if q:
                out.append((str(qid) if qid != "" else q[:30], q))
        return out

    # Else try parsing as a single JSON object/array
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []
    out = []
    if isinstance(data, list):
        for obj in data:
            if not isinstance(obj, dict):
                continue
            qid = obj.get("query_id") or obj.get("qid") or obj.get("id") or ""
            q = obj.get("query") or obj.get("text") or obj.get("title") or ""
            if q:
                out.append((str(qid) if qid != "" else q[:30], q))
    elif isinstance(data, dict):
        # either mapping id->text or an object with 'queries'
        if "queries" in data and isinstance(data["queries"], list):
            for obj in data["queries"]:
                if not isinstance(obj, dict):
                    continue
                qid = obj.get("query_id") or obj.get("qid") or obj.get("id") or ""
                q = obj.get("query") or obj.get("text") or obj.get("title") or ""
                if q:
                    out.append((str(qid) if qid != "" else q[:30], q))
        else:
            for k, v in data.items():
                out.append((str(k), str(v)))
    return out


def main(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="phrase_search.py")
    # Shell compatibility: phrase_search.sh <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <PATH_OF_STOPWORDS_FILE>
    ap.add_argument("index_dir", help="Directory containing index.json from Task 1")
    ap.add_argument("query_file", help="Path to query JSON/JSONL file")
    ap.add_argument("output_dir", help="Directory where phrasesearchdocids.txt is written")
    ap.add_argument("stopwords", help="Stopwords file path (ignored)")
    args = ap.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "phrase_search_docids.txt")
    phrase_search(args.query_file, args.index_dir, args.stopwords, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
