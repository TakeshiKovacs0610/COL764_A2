#!/usr/bin/env python3
import os
import sys
import json
from collections import defaultdict

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


# ---------- Fast tokenizer setup ----------
# Translate table to delete ASCII digits 0-9
_DIGIT_DELETE = str.maketrans("", "", "0123456789")

def _init_tokenizer():
    """
    Fast tokenizer using spaCy's blank English tokenizer.
    Much faster than loading full models like en_core_web_sm.
    """
    nlp = spacy.blank("en")  # rule-based English tokenizer; no internet needed
    nlp.max_length = 300_000_000  # allow very large inputs safely
    return nlp


def _fast_tokenize(nlp, text: str):
    """
    Assignment tokenizer:
    1. Keep only ASCII by removing any non-ASCII bytes
    2. Lowercase
    3. Remove digits 0–9 completely from text (keep punctuation/symbols)
    4. Tokenize with spaCy
    5. Skip whitespace tokens
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Step 1: ASCII-only filtering
    text = text.encode("ascii", "ignore").decode("ascii")
    
    # Step 2 & 3: Lowercase and remove digits completely
    text = text.lower().translate(_DIGIT_DELETE)
    
    # Step 4: spaCy tokenize the cleaned text
    doc = nlp(text)
    
    # Step 5: Yield non-whitespace tokens
    for tok in doc:
        if tok.is_space:
            continue
        t = tok.text
        if t:
            yield t


# --------------- REQUIRED FUNCTIONS ----------------
def build_index(corpus_dir: str, vocab_path: str) -> InvertedIndex:
    """
    Build a positional inverted index with explicit TF per (term, doc).
    Uses spaCy tokenizer (no extra normalization).
    Only tokens present in vocab are indexed.
    Positions are document-wide (single counter across selected fields).
    """
    inv = InvertedIndex()
    inv.load_vocab(vocab_path)

    nlp = _init_tokenizer()

    seen_docs = set()
    field_order = ["title", "doi", "date", "abstract"]  # same order as A1

    for fname in os.listdir(corpus_dir):
        path = os.path.join(corpus_dir, fname)
        if not os.path.isfile(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
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

                per_doc_positions = defaultdict(list)
                pos = 0

                for field in field_order:
                    if field not in obj:
                        continue
                    for token in _fast_tokenize(nlp, obj[field]):
                        tid = inv.token2id.get(token)
                        if tid is not None:
                            per_doc_positions[tid].append(pos)
                        pos += 1

                for tid, positions in per_doc_positions.items():
                    inv.postings[tid][did] = {
                        "tf": len(positions),
                        "positions": positions
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
            #term_obj[ext_id] = {"tf": entry["tf"], "pos": sorted(entry["positions"])}
            term_obj[ext_id] = {"tf": entry["tf"], "positions": entry["positions"]}
            
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


if __name__ == "__main__":
    # Usage: python3 build_index.py <CORPUS_DIR> <VOCAB.txt> <INDEX_DIR>
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <CORPUS_DIR> <VOCAB.txt> <INDEX_DIR>")
        sys.exit(1)

    corpus_dir, vocab_file, index_dir = sys.argv[1:4]
    inv = build_index(corpus_dir, vocab_file)
    save_index(inv, index_dir)
    print(f"Wrote index to: {os.path.join(index_dir, 'index.json')}")
