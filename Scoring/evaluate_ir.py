#!/usr/bin/env python3
# evaluate_ir.py
# Usage:
#   python evaluate_ir.py --qrels qrels.json --run bm25_docids.txt [--k 100]
#   (qrels.json is JSONL; run is TREC format: qid docid rank score)

import argparse
import json
from collections import defaultdict

def load_qrels_jsonl(path):
    """Load qrels from JSONL; return dict[qid] -> set of relevant doc_ids (relevance > 0)."""
    rel = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            obj = json.loads(line)
            qid = str(obj["query_id"]).strip()
            doc = str(obj["doc_id"]).strip()
            rel_val = int(obj.get("relevance", 0))
            if rel_val > 0:
                rel[qid].add(doc)
    return rel

def load_run_trec(path):
    """
    Load run in TREC format:
      qid docid rank score
    Return dict[qid] -> list of (docid, rank, score), sorted by rank asc.
    """
    run = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                # allow optional extra columns but require at least 4
                raise ValueError(f"Bad run line (need at least 4 fields): {line}")
            qid, docid, rank_str, score_str = parts[:4]
            try:
                rank = int(rank_str)
            except ValueError:
                # Some runs use an extra column before rank. Try to parse last two as rank/score
                # but keep strictness for this assignment.
                raise
            try:
                score = float(score_str)
            except ValueError:
                score = 0.0
            run[qid].append((docid, rank, score))
    # sort by rank
    for q in run:
        run[q].sort(key=lambda t: t[1])
    return run

def precision_recall_f1(num_rel_ret, num_ret, num_rel):
    p = (num_rel_ret / num_ret) if num_ret > 0 else 0.0
    r = (num_rel_ret / num_rel) if num_rel > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1

def evaluate(qrels, run, k=None):
    """
    qrels: dict[qid] -> set(relevant docids)
    run:   dict[qid] -> list of (docid, rank, score)
    k:     optional cutoff (use only top-k retrieved per query)
    Returns:
      per_query: dict[qid] -> (P, R, F1, num_ret, num_rel_ret, num_rel)
      macro_avg: (P_macro, R_macro, F1_macro)
      micro_avg: (P_micro, R_micro, F1_micro)
    """
    per_query = {}
    all_qids = set(qrels.keys()) | set(run.keys())

    # Micro counts
    micro_tp = micro_ret = micro_rel = 0

    for qid in sorted(all_qids, key=lambda x: (len(x), x)):
        relset = qrels.get(qid, set())
        retrieved = run.get(qid, [])
        if k is not None:
            retrieved = retrieved[:k]
        ret_docs = [d for (d, _, _) in retrieved]

        num_rel = len(relset)
        num_ret = len(ret_docs)
        num_rel_ret = sum(1 for d in ret_docs if d in relset)

        P, R, F1 = precision_recall_f1(num_rel_ret, num_ret, num_rel)
        per_query[qid] = (P, R, F1, num_ret, num_rel_ret, num_rel)

        micro_tp += num_rel_ret
        micro_ret += num_ret
        micro_rel += num_rel

    # Macro averages (average of per-query metrics)
    if per_query:
        P_macro = sum(v[0] for v in per_query.values()) / len(per_query)
        R_macro = sum(v[1] for v in per_query.values()) / len(per_query)
        F1_macro = sum(v[2] for v in per_query.values()) / len(per_query)
    else:
        P_macro = R_macro = F1_macro = 0.0

    # Micro averages (aggregate over all queries)
    P_micro, R_micro, F1_micro = precision_recall_f1(micro_tp, micro_ret, micro_rel)

    return per_query, (P_macro, R_macro, F1_macro), (P_micro, R_micro, F1_micro)

def print_report(per_query, macro_avg, micro_avg):
    print("qid\tP\tR\tF1\tret\trel_ret\trel_total")
    for qid, (P, R, F1, num_ret, num_rel_ret, num_rel) in per_query.items():
        print(f"{qid}\t{P:.4f}\t{R:.4f}\t{F1:.4f}\t{num_ret}\t{num_rel_ret}\t{num_rel}")
    Pm, Rm, Fm = macro_avg
    Pmi, Rmi, F1mi = micro_avg
    print("\n== Macro Averages ==")
    print(f"P_macro={Pm:.4f}  R_macro={Rm:.4f}  F1_macro={Fm:.4f}")
    print("== Micro Averages ==")
    print(f"P_micro={Pmi:.4f}  R_micro={Rmi:.4f}  F1_micro={F1mi:.4f}")

def write_trec_run(path, items):
    """
    Helper: write a TREC-format run file.
    items: iterable of (qid, docid, rank, score)
    """
    with open(path, "w", encoding="utf-8") as f:
        for qid, docid, rank, score in items:
            f.write(f"{qid} {docid} {rank} {score}\n")

def main():
    ap = argparse.ArgumentParser(description="Evaluate IR run against qrels (Precision/Recall/F1).")
    ap.add_argument("--qrels", required=True, help="Path to qrels JSONL (qrels.json).")
    ap.add_argument("--run", required=True, help="Path to run file in TREC format.")
    ap.add_argument("--k", type=int, default=None, help="Optional cutoff (top-k).")
    args = ap.parse_args()

    qrels = load_qrels_jsonl(args.qrels)
    run = load_run_trec(args.run)
    per_query, macro_avg, micro_avg = evaluate(qrels, run, k=args.k)
    print_report(per_query, macro_avg, micro_avg)

if __name__ == "__main__":
    main()
