#!/usr/bin/env python3
import subprocess
import os

# Hyperparameter ranges to test
R_values = [52, 55, 58]
alpha_values = [1.0]
beta_values = [0.75, 0.8, 0.85]
top_m_values = [42, 45, 48]

best_f1 = 0.0
best_config = None

workspace_dir = '/Users/priyanshuagrawal/Desktop/COL764_A2'

for R in R_values:
    for alpha in alpha_values:
        for beta in beta_values:
            for top_m in top_m_values:
                print(f"Testing R={R}, alpha={alpha}, beta={beta}, top_m={top_m}")

                # Run feedback retrieval
                cmd_feedback = f"python3 feedback_retrieval.py temp/out_index Data/CORD19/queries.json temp/results 100 {R} {alpha} {beta} {top_m}"
                subprocess.run(cmd_feedback, shell=True, cwd=workspace_dir)

                # Evaluate the results
                cmd_eval = f"python3 temp/results/evaluate_ir.py --qrels Data/CORD19/qrels.json --run temp/results/feedback_docids.txt --k 100"
                result = subprocess.run(cmd_eval, shell=True, cwd=workspace_dir, capture_output=True, text=True)

                f1_str = result.stdout.strip()
                try:
                    f1 = float(f1_str)
                    print(f"F1: {f1:.4f}")
                    if f1 > best_f1:
                        best_f1 = f1
                        best_config = (R, alpha, beta, top_m)
                except ValueError:
                    print(f"Error parsing F1: '{f1_str}'")

if best_config is not None:
    print(f"\nBest configuration: R={best_config[0]}, alpha={best_config[1]}, beta={best_config[2]}, top_m={best_config[3]}")
    print(f"Best F1 score: {best_f1:.4f}")
else:
    print("No valid configurations found.")
