#!/usr/bin/env python3
import os
import sys
import json
import math
from collections import defaultdict
from typing import List, Optional
import spacy

def _init_tokenizer():
    nlp = spacy.blank("en")
    nlp.max_length = 300_000_000
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

# ---------------- QUERY FUNCTION ----------------

def vsm_query(query: str, vsm_index: object, k: int) -> list:
    """
    Given a query string, return the top-k documents ranked by cosine similarity
    using log-normalized TF-IDF weighting.
    """
    nlp = _init_tokenizer()
    idf = vsm_index["idf"] # type: ignore
    doc_norms = vsm_index["doc_norms"] # type: ignore
    postings = vsm_index["postings"] # type: ignore

    # 1. Build query term frequency
    q_tf = defaultdict(int)
    for token in _tokenize_spacy_raw(nlp, query):
        if token in idf:  # only terms in vocab
            q_tf[token] += 1

    if not q_tf:
        return []

    # 2. Compute query weights (log-normalized)
    q_weights = {}
    for term, tf in q_tf.items():
        q_weights[term] = (1 + math.log(tf)) * idf[term]

    # 3. Compute scores for docs
    scores = defaultdict(float)
    for term, w_tq in q_weights.items():
        if term not in postings:
            continue
        for doc, tf in postings[term].items():
            w_td = (1 + math.log(tf)) * idf[term]
            scores[doc] += w_tq * w_td

    # 4. Normalize by doc norms and query norm
    q_norm = math.sqrt(sum(w**2 for w in q_weights.values()))
    for doc in list(scores.keys()):
        if doc in doc_norms and doc_norms[doc] > 0 and q_norm > 0:
            scores[doc] /= (doc_norms[doc] * q_norm)

    # 5. Sort and return top-k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]

# ---------------- MULTI-QUERY DRIVER ----------------

def _read_queries_json(path: str):
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


def vsm(queryFile: str, index_dir: str, k: int, outFile: str) -> None:
    """
    Given a JSONL file containing queries, run VSM retrieval for each query
    and write top-k results to outFile in required format:
    qid docid rank score
    Uses only the "title" field from queries.
    """

    # load vsm index
    with open(os.path.join(index_dir, "vsm.json"), "r", encoding="utf-8") as f:
        vsm_index = json.load(f)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(outFile), exist_ok=True)

    nlp = _init_tokenizer()
    queries = list(_read_queries_json(queryFile))
    
    with open(outFile, "w", encoding="utf-8") as out:
        for qid, text in queries:
            tokens = [tok for tok in _tokenize_spacy_raw(nlp, text)]
            clean_query = " ".join(tokens)

            ranked = vsm_query(clean_query, vsm_index, k)

            # Write: qid docid rank score (space-separated like BM25)
            for rank, (doc, score) in enumerate(ranked, start=1):
                out.write(f"{qid} {doc} {rank} {score:.4f}\n")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(f"Usage: {sys.argv[0]} <INDEX_DIR> <QUERY_FILE.jsonl> <OUTPUT_DIR> <k>")
        print("Example: python vsm.py index_dir queries.jsonl output_dir 10")
        sys.exit(1)

    index_dir, queryFile, out_dir, k = sys.argv[1:5]
    k = int(k)
    outFile = os.path.join(out_dir, "vsm_docids.txt")
    vsm(queryFile, index_dir, k, outFile)
    print(f"Results written to {outFile}")