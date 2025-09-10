#!/usr/bin/env python3
"""
Script to compare rankings between two result files.
Each file should have lines in the format: query_id doc_id rank score
Compares the set of documents and their ordering for each query.
"""

import sys
from collections import defaultdict

def parse_file(filepath):
    """Parse the file into a dict: query_id -> list of (doc_id, rank)"""
    data = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue  # Skip malformed lines
            query_id, doc_id, rank, score = parts
            data[query_id].append((doc_id, int(rank)))
    # Sort by rank for each query
    for query in data:
        data[query].sort(key=lambda x: x[1])
    return data

def compare_rankings(file1, file2):
    """Compare rankings between two files."""
    data1 = parse_file(file1)
    data2 = parse_file(file2)
    
    all_queries = set(data1.keys()) | set(data2.keys())
    
    different_sets = []
    different_orderings = []
    
    for query in sorted(all_queries):
        list1 = [doc for doc, rank in data1.get(query, [])]
        list2 = [doc for doc, rank in data2.get(query, [])]
        
        set1 = set(list1)
        set2 = set(list2)
        
        if query not in data1:
            different_sets.append(f"Query {query}: Missing in {file1}")
        elif query not in data2:
            different_sets.append(f"Query {query}: Missing in {file2}")
        elif len(list1) != 10 or len(list2) != 10:
            different_sets.append(f"Query {query}: Different number of documents ({len(list1)} in {file1}, {len(list2)} in {file2})")
        elif set1 != set2:
            common = len(set1 & set2)
            diff = len(set1 - set2) + len(set2 - set1)
            different_sets.append(f"Query {query}: Different set of documents ({common} common, {diff} different)")
        elif list1 != list2:
            # Find differing positions
            min_len = min(len(list1), len(list2))
            diff_positions = []
            for i in range(min_len):
                if list1[i] != list2[i]:
                    diff_positions.append(f"Rank {i+1}: {file1} has {list1[i]}, {file2} has {list2[i]}")
            if len(list1) != len(list2):
                diff_positions.append(f"Different lengths: {file1} has {len(list1)}, {file2} has {len(list2)}")
            different_orderings.append(f"Query {query}: Same documents, different ordering\n" + "\n".join(diff_positions))
    
    print(f"Queries with different document sets ({len(different_sets)}):")
    if different_sets:
        for diff in different_sets:
            print(diff)
    else:
        print("None")
    
    print(f"\nQueries with same documents but different orderings ({len(different_orderings)}):")
    if different_orderings:
        for diff in different_orderings:
            print(diff)
    else:
        print("None")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_rankings.py <file1> <file2>")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    compare_rankings(file1, file2)
