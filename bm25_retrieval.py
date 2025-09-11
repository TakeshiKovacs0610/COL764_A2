#!/usr/bin/env python3
"""
BM25 Retrieval — Task 4

Implements two required functions:
  - bm25_query(query: str, index: object, k: int) -> list
  - bm25(queryFile: str, index_dir: str, k: int, outFile: str) -> None

Index handling follows Task1's index.json format, plus bm25.json with doc-level stats:
  bm25.json = { "N": int, "avgdl": float, "doc_len": {docid: int}, "hyperparams": {...} }

Tokenization uses spaCy's rule-based English tokenizer (spacy.blank("en")).
Uses only the "title" field from queries.

CLI usage mirrors other tasks via a shell wrapper:
  python3 bm25_retrieval.py <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR> <k>
and writes TREC-style lines to <OUTPUT_DIR>/bm25_docids.txt
"""

import os
import sys
import json
import math
from typing import Dict, List, Tuple, Iterable, Optional

try:
    import spacy
except ImportError as e:
    raise SystemExit(
        "spaCy is required. Install with: pip install spacy"
    ) from e


# BM25 hyperparameters by k
HYPERPARAMS_BY_K = {
    20: {"k1": 0.8, "b": 0.08},
    200: {"k1": 1.618, "b": 0.498},
}

K3 = 0
IDF_CLAMP_ZERO = True

def get_hyperparams_for_k(k, hyperparams_by_k):
    keys = sorted(hyperparams_by_k.keys())
    closest = min(keys, key=lambda x: abs(x - k))
    return hyperparams_by_k[closest]


# ---------------- I/O helpers ----------------

def _load_index(index_dir: str) -> Dict[str, dict]:
    with open(os.path.join(index_dir, "index.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _load_bm25(index_dir: str) -> Dict:
    with open(os.path.join(index_dir, "bm25.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _init_tokenizer():
    nlp = spacy.blank("en")
    nlp.max_length = 300_000_000
    return nlp


def _tokenize(nlp, text: str) -> List[str]:
    doc = nlp(text if isinstance(text, str) else str(text))
    return [t.text for t in doc if not t.is_space]


def _read_queries_json(path: str) -> Iterable[Tuple[str, str]]:
    """Read queries in flexible JSON/JSONL formats.
    Accepts entries with keys: (query_id|qid|id) and various text fields.
    Yields (qid, text).
    Uses only the "title" field.
    """
    fields = ["title"]
    
    try:
        with open(path, "r", encoding="utf-16") as f:
            content = f.read().strip()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    if not content:
        return []
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
            
            # Concatenate chosen fields
            text_parts = []
            for field in fields:
                if obj.get(field):
                    text_parts.append(str(obj[field]))
            q = " ".join(text_parts)
            
            if q:
                out.append((str(qid) if qid != "" else q[:30], q))
        return out
    # else single JSON object/array
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
            
            # Concatenate chosen fields
            text_parts = []
            for field in fields:
                if obj.get(field):
                    text_parts.append(str(obj[field]))
            q = " ".join(text_parts)
            
            if q:
                out.append((str(qid) if qid != "" else q[:30], q))
    elif isinstance(data, dict):
        if "queries" in data and isinstance(data["queries"], list):
            for obj in data["queries"]:
                if not isinstance(obj, dict):
                    continue
                qid = obj.get("query_id") or obj.get("qid") or obj.get("id") or ""
                
                # Concatenate chosen fields
                text_parts = []
                for field in fields:
                    if obj.get(field):
                        text_parts.append(str(obj[field]))
                q = " ".join(text_parts)
                
                if q:
                    out.append((str(qid) if qid != "" else q[:30], q))
        else:
            for k, v in data.items():
                out.append((str(k), str(v)))
    return out


# ---------------- Core BM25 ----------------

def _bm25_idf(N: int, df: int) -> float:
    # Natural log; clamp negative values to 0
    val = math.log((N - df + 0.5) / (df + 0.5)) if df > 0 else math.log((N + 0.5) / 0.5)
    return val if val > 0 else 0.0


def bm25_query(query: str, index: object, k: int) -> list:
    """Given the raw query string and a combined index object, return top-k (docid, score).

    index object must be a dict with keys:
      - "lexicon": index.json dictionary
      - "bm25":    bm25.json dictionary
    """
    if not isinstance(index, dict) or "lexicon" not in index or "bm25" not in index:
        raise ValueError("index must be a dict with keys 'lexicon' and 'bm25'")

    lex = index["lexicon"]
    bm = index["bm25"]
    N = int(bm.get("N", 0))
    avgdl = float(bm.get("avgdl", 0.0))
    doc_len = bm.get("doc_len", {})
    params = get_hyperparams_for_k(k, HYPERPARAMS_BY_K)
    k1 = params["k1"]
    b = params["b"]

    nlp = _init_tokenizer()
    q_tokens = _tokenize(nlp, query)
    if not q_tokens or N <= 0:
        return []

    # Unique terms; k3=0 (ignore query tf weighting)
    seen = set()
    scores: Dict[str, float] = {}
    for t in q_tokens:
        if t in seen:
            continue
        seen.add(t)
        term_entry = lex.get(t)
        if not term_entry:
            continue
        df = int(term_entry.get("df", 0))
        if df <= 0:
            continue
        idf = _bm25_idf(N, df)
        if idf <= 0.0:
            continue
        postings = term_entry.get("postings", {})
        for doc_id, entry in postings.items():
            tf = int(entry.get("tf", 0))
            if tf <= 0:
                continue
            len_d = int(doc_len.get(doc_id, 0))
            denom_extra = (1 - b)
            if avgdl > 0:
                denom_extra += b * (len_d / avgdl)
            # guard against denominator being zero
            denom = tf + k1 * max(denom_extra, 1e-9)
            norm = (tf * (k1 + 1.0)) / denom
            scores[doc_id] = scores.get(doc_id, 0.0) + idf * norm

    if not scores:
        return []

    # sort by score desc then docid asc; take top-k
    items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return items[:k]


def bm25(queryFile: str, index_dir: str, k: int, outFile: str) -> None:
    """Run BM25 for all queries from queryFile and write TREC-style results to outFile.

    Each output line: "qid docid rank score"
    Rank starts at 1. Writes up to k lines per query (fewer if not enough matches).
    Uses only the "title" field from queries.
    """
        
    lex = _load_index(index_dir)
    bm = _load_bm25(index_dir)
    combo = {"lexicon": lex, "bm25": bm}
    queries = list(_read_queries_json(queryFile))
    with open(outFile, "w", encoding="utf-8") as out:
        for qid, text in queries:
            results = bm25_query(text, combo, k)
            for rank, (doc_id, score) in enumerate(results, start=1):
                out.write(f"{qid} {doc_id} {rank} {score}\n")


def main(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="bm25_retrieval.py")
    ap.add_argument("index_dir", help="Directory containing index.json and bm25.json")
    ap.add_argument("query_file", help="Path to query JSON/JSONL file")
    ap.add_argument("output_dir", help="Directory where bm25_docids.txt is written")
    ap.add_argument("k", type=int, help="Top-k documents per query")
    args = ap.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "bm25_retrieval_docids.txt")
    bm25(args.query_file, args.index_dir, args.k, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
