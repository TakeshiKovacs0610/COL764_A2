#!/usr/bin/env python3
# bm25_tuner.py — Adaptive, gradient-free hyperparameter search for BM25
# Configure constants at the top; run:  python bm25_tuner.py
#
# It edits <INDEX_DIR>/bm25.json["hyperparams"] for (k1,b), calls your bm25()
# to produce a TREC run, and evaluates macro/micro P/R/F1 using evaluate_ir.py.
#
# No external libraries required.

import os, sys, json, time, csv, random, math, shutil, datetime, traceback, importlib.util
from typing import Dict, Any, Tuple

# =========================
# ======= CONFIGURE =======
# =========================

# Paths (edit these to your local project layout)
INDEX_DIR = "temp/out_index"                     # directory containing index.json and bm25.json
QUERIES_JSON = "Data/CORD19/queries.json"           # your queries file (JSON or JSONL per bm25_retrieval reader)
QRELS_JSONL = "Scoring/qrels.json"              # qrels in JSONL format
STOPWORDS_PATH = "Data/stopwords.txt"        # accepted but ignored by bm25_retrieval
OUTPUT_ROOT = "sweeps"                  # dir to store run outputs and CSV

# Modules (edit if your files are elsewhere)
PATH_BM25_FILE = "Task4/bm25_retrieval.py"
PATH_EVAL_FILE = "Scoring/evaluate_ir.py"

# Search space bounds
K_MIN, K_MAX = 20, 20                  # integer k (top-k)
K1_LOW, K1_HIGH = 0.8, 1.4              # typical BM25 k1 range
B_LOW, B_HIGH = 0.00, 0.40              # typical BM25 b range

# Optimization settings
N_TRIALS = 120                          # total trials
N_WARMUP = 30                           # random exploration trials before local refinement
INIT_SIGMA = {"k": 1.0, "k1": 0.10, "b": 0.04}  # initial local step sizes
SIGMA_DECAY_ON_IMPROVE = 0.85           # shrink step size when improvement found
SIGMA_GROW_ON_STALL = 1.10              # slightly increase on stalls to escape local minima
PATIENCE = 10                           # after these many non-improving trials, random restart
SEED = 42                               # reproducibility

# Objective to maximize: one of {"macro_f1", "micro_f1", "macro_p", "micro_p", "macro_r", "micro_r"}
METRIC = "macro_f1"

# CSV filename (will be created under OUTPUT_ROOT)
CSV_NAME = "bm25_tuner_results_k20.2.csv"

# =========================
# ====== END CONFIG =======
# =========================

random.seed(SEED)

os.makedirs(OUTPUT_ROOT, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_ROOT, CSV_NAME)
RUNS_DIR = os.path.join(OUTPUT_ROOT, "runsk20")
os.makedirs(RUNS_DIR, exist_ok=True)

def _import_from_path(module_name: str, path: str):
    """Import a module given an explicit filesystem path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:
        raise ImportError(f"Cannot create spec for {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

# Try to import bm25 and evaluate modules
bm25_mod = _import_from_path("bm25_retrieval", PATH_BM25_FILE)
eval_mod = _import_from_path("evaluate_ir", PATH_EVAL_FILE)

# Validate required callables
assert hasattr(bm25_mod, "bm25"), "bm25_retrieval.py must expose bm25(queryFile, index_dir, stopword_file, k, outFile)"
assert hasattr(eval_mod, "load_qrels_jsonl"), "evaluate_ir.py must expose load_qrels_jsonl"
assert hasattr(eval_mod, "load_run_trec"), "evaluate_ir.py must expose load_run_trec"
assert hasattr(eval_mod, "evaluate"), "evaluate_ir.py must expose evaluate"

# Read qrels once
QRELS = eval_mod.load_qrels_jsonl(QRELS_JSONL)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def trunc_norm(mean, sigma, lo, hi):
    # Sample from Normal(mean, sigma) and clamp to bounds
    x = random.gauss(mean, sigma)
    return clamp(x, lo, hi)

def propose_candidate(best: Dict[str, Any], sigma: Dict[str, float]) -> Dict[str, Any]:
    """Local proposal around current best settings."""
    k = int(round(trunc_norm(best["k"], sigma["k"], K_MIN, K_MAX)))
    k1 = trunc_norm(best["k1"], sigma["k1"], K1_LOW, K1_HIGH)
    b = trunc_norm(best["b"], sigma["b"], B_LOW, B_HIGH)
    return {"k": k, "k1": k1, "b": b}

def random_candidate() -> Dict[str, Any]:
    k = random.randint(K_MIN, K_MAX)
    k1 = random.uniform(K1_LOW, K1_HIGH)
    b = random.uniform(B_LOW, B_HIGH)
    return {"k": k, "k1": k1, "b": b}

def set_hyperparams_in_bm25(index_dir: str, k1: float, b: float) -> Dict[str, Any]:
    """Edit <index_dir>/bm25.json to set hyperparams and return previous JSON (for restoration)."""
    bm25_path = os.path.join(index_dir, "bm25.json")
    with open(bm25_path, "r", encoding="utf-8") as f:
        bm = json.load(f)
    prev = json.dumps(bm)  # keep string backup to restore exactly

    hp = bm.get("hyperparams", {})
    hp["k1"] = float(k1)
    hp["b"] = float(b)
    bm["hyperparams"] = hp
    with open(bm25_path, "w", encoding="utf-8") as f:
        json.dump(bm, f, ensure_ascii=False, indent=2)
    return {"path": bm25_path, "prev": prev}

def restore_bm25(snapshot: Dict[str, Any]):
    if not snapshot:
        return
    with open(snapshot["path"], "w", encoding="utf-8") as f:
        f.write(snapshot["prev"])

def evaluate_once(params: Dict[str, Any], trial_id: int) -> Tuple[Dict[str, float], str]:
    """
    Run bm25 with given params and evaluate. Returns (metrics, run_path).
    metrics keys: macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1
    """
    k, k1, b = params["k"], params["k1"], params["b"]

    # Set BM25 hyperparameters in bm25.json (snapshot for safety)
    snap = set_hyperparams_in_bm25(INDEX_DIR, k1, b)
    try:
        run_path = os.path.join(RUNS_DIR, f"run_trial{trial_id}_k{k}_k1{round(k1,4)}_b{round(b,4)}.txt")
        bm25_mod.bm25(QUERIES_JSON, INDEX_DIR, STOPWORDS_PATH, int(k), run_path)

        # Evaluate
        run = eval_mod.load_run_trec(run_path)
        per_query, macro_avg, micro_avg = eval_mod.evaluate(QRELS, run, k=None)  # run already contains top-k
        metrics = {
            "macro_p": macro_avg[0], "macro_r": macro_avg[1], "macro_f1": macro_avg[2],
            "micro_p": micro_avg[0], "micro_r": micro_avg[1], "micro_f1": micro_avg[2],
        }
        return metrics, run_path
    finally:
        # Restore bm25.json to previous content (avoid accumulating floating diffs)
        restore_bm25(snap)

def obj_value(metrics: Dict[str, float]) -> float:
    return float(metrics[METRIC])

def read_existing_csv(csv_path: str):
    cache = {}
    if not os.path.exists(csv_path):
        return cache
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["k"]), float(row["k1"]), float(row["b"]))
            cache[key] = {k: (float(v) if k not in ("trial", "run_path", "ts") else v)
                          for k, v in row.items()}
    return cache

def append_to_csv(csv_path: str, header: list, row: Dict[str, Any]):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def main():
    print(f"[bm25_tuner] optimizing {METRIC} over k∈[{K_MIN},{K_MAX}], k1∈[{K1_LOW},{K1_HIGH}], b∈[{B_LOW},{B_HIGH}]")
    cache = read_existing_csv(CSV_PATH)

    # CSV header
    header = ["trial", "ts", "k", "k1", "b",
              "macro_p", "macro_r", "macro_f1",
              "micro_p", "micro_r", "micro_f1",
              "run_path"]

    # If cache existed, locate current best
    best = None
    best_val = -1.0
    if cache:
        for (k, k1, b), row in cache.items():
            v = float(row.get(METRIC, -1.0))
            if v > best_val:
                best_val = v
                best = {"k": k, "k1": k1, "b": b}
        print(f"[resume] Loaded {len(cache)} past trials. Current best {METRIC}={best_val:.4f} @ {best}")
    else:
        print("[fresh] No previous CSV found; starting new sweep.")

    # Initialize sigmas for local proposals
    sigma = INIT_SIGMA.copy()
    non_improve = 0

    trial_idx = 0
    # We ensure N_TRIALS fresh evaluations (skip duplicates if they occur)
    while trial_idx < N_TRIALS:
        trial = trial_idx + 1
        # Choose candidate
        if trial <= N_WARMUP or best is None or (non_improve >= PATIENCE and random.random() < 0.6):
            cand = random_candidate()
            if non_improve >= PATIENCE:
                # small "restart" near a random point by resetting sigmas
                sigma = INIT_SIGMA.copy()
                non_improve = 0
        else:
            cand = propose_candidate(best, sigma)

        key = (cand["k"], round(cand["k1"], 6), round(cand["b"], 6))
        if key in cache:
            # Already computed—skip without advancing trial index
            print(f"[skip] duplicate candidate {key}; resampling")
            continue

        print(f"[trial {trial}/{N_TRIALS}] k={cand['k']} k1={cand['k1']:.4f} b={cand['b']:.4f} ...")
        try:
            metrics, run_path = evaluate_once(cand, trial_id=trial)
        except Exception as e:
            print("[error] trial failed:", e)
            traceback.print_exc()
            # count as a trial to avoid infinite loops on persistent errors
            trial_idx += 1
            continue

        # Log row
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        row = {
            "trial": trial, "ts": ts,
            "k": cand["k"], "k1": round(cand["k1"], 6), "b": round(cand["b"], 6),
            "macro_p": metrics["macro_p"], "macro_r": metrics["macro_r"], "macro_f1": metrics["macro_f1"],
            "micro_p": metrics["micro_p"], "micro_r": metrics["micro_r"], "micro_f1": metrics["micro_f1"],
            "run_path": run_path,
        }
        append_to_csv(CSV_PATH, header, row)
        cache[(row["k"], row["k1"], row["b"])] = row

        val = obj_value(metrics)
        improved = val > best_val
        if improved or best is None:
            best = {"k": cand["k"], "k1": cand["k1"], "b": cand["b"]}
            best_val = val
            non_improve = 0
            # shrink sigmas to zoom in
            sigma = {p: max(1e-3, s * SIGMA_DECAY_ON_IMPROVE) for p, s in sigma.items()}
            print(f"  ↳ improved {METRIC} = {best_val:.4f}; new best {best}; shrink σ → {sigma}")
        else:
            non_improve += 1
            # mild expansion encourages exploration
            sigma = {p: min(999.0, s * SIGMA_GROW_ON_STALL) for p, s in sigma.items()}
            print(f"  ↳ no improvement (streak={non_improve}); expand σ → {sigma}")
        trial_idx += 1

    # Print best summary
    print("\n=== BEST SETTINGS ===")
    print(f"{METRIC} = {best_val:.4f} at k={best['k']}, k1={best['k1']:.4f}, b={best['b']:.4f}")
    print(f"CSV saved at: {CSV_PATH}")
    print(f"All runs in: {RUNS_DIR}")

if __name__ == "__main__":
    main()
