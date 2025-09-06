#!/usr/bin/env python3
"""
lookup_index.py: Utility to inspect terms and postings in a large index.json file.

Usage:
    python3 lookup_index.py <index.json> [--term <term>] [--n N]

- If --term is provided, prints the entry for that term (if present).
- Otherwise, prints the structure for the first N terms (default: 5).
- If the index file does not exist, can optionally call build_index.py to create it (not implemented here for safety).
"""
import sys
import json
from itertools import islice

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <index.json> [--term <term>] [--n N]")
        sys.exit(1)

    path = sys.argv[1]
    term = None
    n = 5
    args = sys.argv[2:]
    if '--term' in args:
        idx = args.index('--term')
        if idx + 1 < len(args):
            term = args[idx + 1]
    if '--n' in args:
        idx = args.index('--n')
        if idx + 1 < len(args):
            try:
                n = int(args[idx + 1])
            except Exception:
                pass

    with open(path, "r", encoding="utf-8") as f:
        idx_data = json.load(f)

    if term:
        if term in idx_data:
            entry = idx_data[term]
            print(f"TERM: {term}")
            print(f"  df = {entry['df']}")
            print(f"  postings:")
            for docid, docinfo in entry['postings'].items():
                print(f"    doc_id: {docid}, tf: {docinfo['tf']}, positions: {docinfo['positions']}")
        else:
            print(f"Term '{term}' not found in index.")
    else:
        print(f"Total terms: {len(idx_data)}")
        print("--- sample terms ---")
        for t, entry in islice(idx_data.items(), n):
            print(f"\nTERM: {t}")
            print(f"  df = {entry['df']}")
            print(f"  postings:")
            for i, (docid, docinfo) in enumerate(entry['postings'].items()):
                if i >= 3:
                    break
                print(f"    doc_id: {docid}, tf: {docinfo['tf']}, positions: {docinfo['positions']}")
