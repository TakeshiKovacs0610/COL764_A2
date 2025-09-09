#!/usr/bin/env python3
import os
import sys
import json
import math
from collections import defaultdict
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

def vsm_query(query: str, vsm_index: dict, k: int) -> list:
    """
    Given a query string, return the top-k documents ranked by cosine similarity
    using log-normalized TF-IDF weighting.
    """
    nlp = _init_tokenizer()
    idf = vsm_index["idf"]
    doc_norms = vsm_index["doc_norms"]
    postings = vsm_index["postings"]

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

    # 4. Normalize by doc norms
    for doc in list(scores.keys()):
        if doc in doc_norms and doc_norms[doc] > 0:
            scores[doc] /= doc_norms[doc]

    # 5. Sort and return top-k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]

# ---------------- MULTI-QUERY DRIVER ----------------

def vsm(queryFile: str, index_dir: str, stopword_file: str, k: int, outFile: str) -> None:
    """
    Given a JSONL file containing queries, run VSM retrieval for each query
    and write top-k results to outFile in required format:
    qid docid rank score
    """
    
    fields = ["title", "description", "narrative"] 

    # load vsm index
    with open(os.path.join(index_dir, "vsm.json"), "r", encoding="utf-8") as f:
        vsm_index = json.load(f)

    # load stopwords
    stopwords = set()
    if stopword_file and os.path.exists(stopword_file):
        with open(stopword_file, "r", encoding="utf-8") as f:
            for line in f:
                stopwords.add(line.strip())

    nlp = _init_tokenizer()
    results = []

    with open(queryFile, "r", encoding="utf-16") as f:
        for line in f:
            try:
                q_obj = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            qid = q_obj.get("query_id")
            if not qid:
                continue

            # Concatenate chosen fields
            text_parts = []
            for field in fields:
                if q_obj.get(field):
                    text_parts.append(str(q_obj[field]))
            raw_query = " ".join(text_parts)

            # dont Filter stopwords
            tokens = [tok for tok in _tokenize_spacy_raw(nlp, raw_query)]
            # tokens = [tok for tok in _tokenize_spacy_raw(nlp, raw_query) if tok not in stopwords]
            clean_query = " ".join(tokens)

            ranked = vsm_query(clean_query, vsm_index, k)

            # Write: qid docid rank score
            for rank, (doc, score) in enumerate(ranked, start=1):
                results.append(f"{qid}\t{doc}\t{rank}\t{score:.4f}")

    # outFile = os.path.join(out_dir, "vsm_docids.txt")
    with open(outFile, "w", encoding="utf-8") as f:
        f.write("\n".join(results))


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print(f"Usage: {sys.argv[0]} <QUERY_FILE.jsonl> <INDEX_DIR> <STOPWORDS.txt> <k> <OUT_FILE> <FIELDS>")
        print("Example: python vsm.py queries.jsonl index_dir stopwords.txt 10 results.txt title description narrative")
        sys.exit(1)

    index_dir, queryFile, out_dir,  stopword_file, k = sys.argv[1:6]
    fields = sys.argv[6:]  # list of fields to use
    k = int(k)
    outFile = os.path.join(out_dir, "vsm_docids.txt")
    vsm(queryFile, index_dir, stopword_file, k, outFile)
    print(f"Results written to {outFile}")