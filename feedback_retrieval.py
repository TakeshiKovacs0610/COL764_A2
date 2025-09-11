#!/usr/bin/env python3
import os
import sys
import json
import math
from collections import defaultdict, Counter
from typing import List, Optional, Dict, Tuple, Any
import spacy

# -----------------------------
# Tokenizer
# -----------------------------

def _init_tokenizer():
    nlp = spacy.blank("en")
    nlp.max_length = 300_000_000
    return nlp

def _tokenize_spacy_raw(nlp, text: str):
    """Tokenize without filters (no lowercasing, no stopword removal)."""
    if not isinstance(text, str):
        text = str(text)
    doc = nlp(text)
    for tok in doc:
        if tok.is_space:
            continue
        yield tok.text

# -----------------------------
# Helper functions
# -----------------------------

def _build_query_vector(tokens, idf):
    """Build query vector using log-TF and IDF."""
    tf = Counter(tokens)
    vec = {}
    for term, freq in tf.items():
        if term in idf:
            vec[term] = (1 + math.log(freq)) * idf[term]
    return vec

# -----------------------------
# PRF Query Function
# -----------------------------

def prf_query(query: str, index: object, k: int) -> List[Tuple[str, float]]:
    """
    Given a query string, return top-k documents with scores AFTER one round of PRF.
    """
    idf = index["idf"] # type: ignore
    postings = index["postings"] # type: ignore
    doc_norms = index["doc_norms"] # type: ignore
    documents = index["documents"] # type: ignore

    # Hyperparameters (can be tuned)
    R = 55      # top R docs used for feedback
    alpha = 1.0 # weight of original query
    beta = 0.8  # weight of feedback terms
    top_m = 45  # keep top-m feedback terms

    # 1. Build initial query vector
    tokens = query.split()
    q0 = _build_query_vector(tokens, idf)

    if not q0:
        return []

    # 2. First retrieval
    scores = defaultdict(float)
    for term, w in q0.items():
        if term in postings:
            for doc_id, tf in postings[term].items():   # postings[term] is dict
                scores[doc_id] += w * ((1 + math.log(tf)) * idf[term])
    
    q_norm = math.sqrt(sum(w**2 for w in q0.values()))
    for doc in list(scores.keys()):
        if doc in doc_norms and doc_norms[doc] > 0 and q_norm > 0:
            scores[doc] /= (doc_norms[doc] * q_norm)


    ranked = sorted(scores.items(), key=lambda x: -x[1])

    if not ranked:
        return []

    # 3. Select top-R feedback docs
    feedback_docs = [doc_id for doc_id, _ in ranked[:R]]

    # 4. Compute centroid of feedback docs
    centroid = defaultdict(float)
    for d in feedback_docs:
        for term, w in documents[d].items():
            centroid[term] += w
    for term in centroid:
        centroid[term] /= R

    # 5. Select top-m feedback terms
    top_terms = sorted(centroid.items(), key=lambda x: -x[1])[:top_m]
    centroid_filtered = {t: w for t, w in top_terms}

    # 6. Rocchio update (combine original + feedback)
    qm = defaultdict(float)
    for term, w in q0.items():
        qm[term] += alpha * w
    for term, w in centroid_filtered.items():
        qm[term] += beta * w

    # 7. Second retrieval with expanded query
    scores2 = defaultdict(float)
    for term, w in qm.items():
        if term in postings:
            for doc_id, tf in postings[term].items():
                scores2[doc_id] += w * ((1 + math.log(tf)) * idf[term])

                
    q_norm2 = math.sqrt(sum(w**2 for w in qm.values()))
    for d in list(scores2.keys()):
        if d in doc_norms and doc_norms[d] > 0 and q_norm2 > 0:
            scores2[d] /= (doc_norms[d] * q_norm2)
    ranked2 = sorted(scores2.items(), key=lambda x: -x[1])[:k]

    return ranked2

# -----------------------------
# Query file reader (same style as VSM)
# -----------------------------

def _read_queries_json(path: str, fields: Optional[List[str]] = None):
    """Read queries from JSON/JSONL file, yield (qid, text)."""
    if fields is None:
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
    out = []

    # JSONL (multiple lines)
    if len(lines) > 1:
        for line in lines:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = obj.get("query_id") or obj.get("qid") or obj.get("id") or ""
            text_parts = [str(obj[f]) for f in fields if obj.get(f)]
            q = " ".join(text_parts)
            if q:
                out.append((str(qid) if qid else q[:30], q))
        return out

    # Single JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []

    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                qid = obj.get("query_id") or obj.get("qid") or obj.get("id") or ""
                text_parts = [str(obj[f]) for f in fields if obj.get(f)]
                q = " ".join(text_parts)
                if q:
                    out.append((str(qid) if qid else q[:30], q))
    elif isinstance(data, dict):
        if "queries" in data and isinstance(data["queries"], list):
            for obj in data["queries"]:
                if isinstance(obj, dict):
                    qid = obj.get("query_id") or obj.get("qid") or obj.get("id") or ""
                    text_parts = [str(obj[f]) for f in fields if obj.get(f)]
                    q = " ".join(text_parts)
                    if q:
                        out.append((str(qid) if qid else q[:30], q))
        else:
            for k, v in data.items():
                out.append((str(k), str(v)))
    return out

# -----------------------------
# PRF Batch Function
# -----------------------------

def prf(queryFile: str, index_dir: str, k: int, outFile: str) -> None:
    """
    Process query file and write PRF results in TREC-eval format:
    qid docid rank score
    """

    fields = ["title"]

    # Load index (JSON version from save_vsm_index)
    with open(os.path.join(index_dir, "vsm.json"), "r", encoding="utf-8") as f:
        index = json.load(f)

    # Recompute document vectors for PRF centroid
    documents = defaultdict(dict)
    for term, doc_tf_map in index["postings"].items():
        for doc_id, tf in doc_tf_map.items():
            w_td = (1 + math.log(tf)) * index["idf"][term]
            documents[doc_id][term] = w_td
    index["documents"] = documents

    nlp = _init_tokenizer()
    queries = _read_queries_json(queryFile, fields)

    os.makedirs(os.path.dirname(outFile), exist_ok=True)

    with open(outFile, "w", encoding="utf-8") as outf:
        for qid, text in queries:
            tokens = [tok for tok in _tokenize_spacy_raw(nlp, text)]
            clean_query = " ".join(tokens)
            results = prf_query(clean_query, index, k)
            for rank, (doc_id, score) in enumerate(results, start=1):
                outf.write(f"{qid} {doc_id} {rank} {score:.4f}\n")

# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(f"Usage: {sys.argv[0]} <INDEX_DIR> <QUERY_FILE.jsonl> <OUTPUT_DIR> <k>")
        sys.exit(1)

    index_dir, queryFile, out_dir, k = sys.argv[1:5]
    k = int(k)

    outFile = os.path.join(out_dir, "feedback_docids.txt")
    prf(queryFile, index_dir, k, outFile)

    print(f"PRF results written to {outFile}")
