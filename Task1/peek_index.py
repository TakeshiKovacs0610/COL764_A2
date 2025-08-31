#!/usr/bin/env python3
import sys
import json
from itertools import islice

# Usage: python3 peek_index.py <path/to/index.json> [N_terms]
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <index.json> [N_terms]")
        sys.exit(1)

    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) >= 3 else 5

    with open(path, "r", encoding="utf-8") as f:
        idx = json.load(f)

    print(f"Total terms: {len(idx)}")
    print("--- sample terms ---")
    for term, postings in islice(idx.items(), n):
        print(f"\nTERM: {term}")
        print(f"  df = {len(postings)}")
        for i, (docid, entry) in enumerate(postings.items()):
            if i >= 3:
                break
            print(entry)